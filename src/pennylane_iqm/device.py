"""IQMDevice: PennyLane Device executing circuits on IQM hardware via iqm-client."""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections.abc import Hashable, Sequence
from typing import cast, overload

import numpy as np
import pennylane as qml
from iqm.iqm_client import IQMClient
from iqm.iqm_client.iqm_client import CircuitJob
from iqm.iqm_client.models import CircuitCompilationOptions
from iqm.iqm_client.transpile import transpile_insert_moves
from iqm.pulse.circuit_operations import Circuit
from iqm.iqm_server_client.models import JobStatus
from iqm.station_control.interface.models import CircuitMeasurementResultsBatch, DynamicQuantumArchitecture
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import decompose, validate_device_wires, validate_measurements
from pennylane.measurements import SampleMeasurement
from pennylane.tape import QuantumScript
from pennylane.typing import Result

from .gates import stopping_condition
from .result import iqm_result_to_samples
from .translate import WireMap, build_wire_map, tape_to_iqm_circuit


@dataclasses.dataclass(frozen=True)
class _PreparedCircuit:
	index: int
	tape: QuantumScript
	circuit: Circuit
	wire_map: WireMap
	measured_wires: list[Hashable]
	shots: int


@qml.transform
def _transpile(tape: QuantumScript, coupling_map: list[tuple[Hashable, Hashable]]) -> tuple:
	"""Like qml.transforms.transpile but supports tensor-product and Hamiltonian measurements.

	PennyLane's built-in transpile raises NotImplementedError for Prod/LinearCombination
	observables even though routing only depends on gate connectivity, not measurements.

	Fix: route a proxy tape whose measurements are a single qml.probs over all wires
	(which transpile accepts), derive the wire permutation from how those proxy wires
	changed, then apply the same permutation to the original measurements.
	"""
	complex_obs = (qml.ops.Prod, qml.ops.LinearCombination)
	needs_workaround = any(isinstance(getattr(m, "obs", None), complex_obs) for m in tape.measurements)

	# `qml.transforms.transpile` is a Transform; .tape_transform is the
	# underlying function, typed `Callable | None`. It is never None for
	# the upstream transpile transform, but narrow explicitly.
	transpile_tape = qml.transforms.transpile.tape_transform
	if transpile_tape is None:
		raise RuntimeError("qml.transforms.transpile.tape_transform is unexpectedly None")

	if not needs_workaround:
		return transpile_tape(tape, coupling_map=coupling_map)

	orig_wires = list(tape.wires)
	proxy_tape = qml.tape.QuantumScript(tape.operations, [qml.probs(wires=orig_wires)], shots=tape.shots)
	routed_batch, fn = transpile_tape(proxy_tape, coupling_map=coupling_map)
	[routed] = routed_batch

	# Derive the wire permutation that routing applied from the proxy measurement
	new_wires = list(routed.measurements[0].wires)
	wire_map = {o: n for o, n in zip(orig_wires, new_wires)}
	remapped_mps = [m.map_wires(wire_map) for m in tape.measurements]

	final_tape = qml.tape.QuantumScript(routed.operations, remapped_mps, shots=tape.shots)
	return [final_tape], fn


@simulator_tracking
@single_tape_support
class IQMDevice(Device):
	"""PennyLane device executing circuits directly on IQM hardware via iqm-client.

	Args:
	    server_url: URL of the IQM Server.
	    wires: Wire count or wire labels. Defaults to None, which auto-detects
	           the qubit count from the server's DynamicQuantumArchitecture.
	           Requires use_connectivity=True (the default).
	    shots: Default shots per execution. Can also be set per-QNode via
	           ``@qml.qnode(dev, shots=1024)``. Must be >= 1 if provided.
	    token: IQM API token. When None, IQMClient reads IQM_TOKEN from the
	           environment automatically — do not set both.
	    poll_interval: Seconds between job-status polls (default 2).
	    timeout: Max wait seconds for a job (default 300).
	    use_connectivity: Fetch DQA and enforce routing (default True).
	"""

	name = "iqm.direct"

	def __init__(
		self,
		server_url: str,
		wires: int | list | None = None,
		shots: int | None = None,
		token: str | None = None,
		poll_interval: float = 2.0,
		timeout: float = 300.0,
		use_connectivity: bool = True,
		options: CircuitCompilationOptions | None = None,
		qubit_mapping: dict[str, str] | None = None,
	):
		if shots is not None and shots < 1:
			raise ValueError("IQMDevice requires shots >= 1 (hardware execution).")

		self._server_url = server_url
		# Only forward an explicit token to IQMClient.
		# When token=None, IQMClient reads IQM_TOKEN from the env on its own —
		# passing it again here would cause a ClientConfigurationError.
		self._token = token
		self._poll_interval = poll_interval
		self._timeout = timeout
		self._use_connectivity = use_connectivity
		self._default_shots = shots
		self._options = options
		self._qubit_mapping = qubit_mapping

		self._client: IQMClient | None = None
		self._dqa: DynamicQuantumArchitecture | None = None

		if wires is None:
			if not use_connectivity:
				raise ValueError("wires must be specified explicitly when use_connectivity=False.")
			# Eagerly fetch the DQA so we know how many qubits this device has.
			try:
				self._dqa = self.client.get_dynamic_quantum_architecture()
				wires = len(self._dqa.qubits)
			except (RuntimeError, OSError, ValueError) as exc:
				raise ValueError(f"Could not auto-detect wires from server: {exc}") from exc

		# Do NOT pass shots to super().__init__ — PennyLane deprecated that.
		# Shots are managed per-QNode or via _default_shots.
		super().__init__(wires=wires)

	@property
	def client(self) -> IQMClient:
		"""IQMClient instance, created once and reused."""
		if self._client is None:
			kwargs: dict[str, str] = {}
			if self._token is not None:
				kwargs["token"] = self._token
			self._client = IQMClient(self._server_url, **kwargs)
		return self._client

	@property
	def architecture(self) -> DynamicQuantumArchitecture | None:
		"""Cached DynamicQuantumArchitecture, fetched on first access.

		Returns None when use_connectivity=False or when the server is
		unreachable (a warning is emitted in that case).
		"""
		if not self._use_connectivity:
			return None
		if self._dqa is None:
			try:
				self._dqa = self.client.get_dynamic_quantum_architecture()
			except (RuntimeError, OSError, ValueError) as exc:
				warnings.warn(
					f"[IQMDevice] Could not fetch DQA: {exc}. Connectivity enforcement is disabled.", stacklevel=2
				)
		return self._dqa

	@property
	def is_star(self) -> bool:
		"""True when the connected device has computational resonators (IQM Star architecture)."""
		dqa = self.architecture
		return dqa is not None and bool(dqa.computational_resonators)

	def _pl_coupling_map(self) -> list[tuple] | None:
		"""Derive a PennyLane coupling map from the DQA's CZ loci.

		Returns None when the architecture is unknown, causing the transpile
		step to be skipped. Qubit-resonator CZ loci are excluded here because
		transpile_insert_moves handles those separately for Star devices.
		"""
		dqa = self.architecture
		if dqa is None or "cz" not in dqa.gates:
			return None

		wire_labels = list(self.wires)
		iqm_to_pl = {f"QB{idx + 1}": w for idx, w in enumerate(wire_labels)}

		coupling: list[tuple] = []
		for locus in dqa.gates["cz"].loci:
			if len(locus) == 2 and all(c.startswith("QB") for c in locus):
				a, b = locus
				if a in iqm_to_pl and b in iqm_to_pl:
					coupling.append((iqm_to_pl[a], iqm_to_pl[b]))

		return coupling or None

	def setup_execution_config(
		self, config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
	) -> ExecutionConfig:
		"""Declare parameter-shift as the preferred gradient method.

		IQMDevice does not compute device-side Jacobians; the PennyLane
		param_shift transform generates shifted circuits and calls execute().
		"""
		if config is None:
			config = ExecutionConfig()

		gradient_method = config.gradient_method
		if gradient_method in (None, "best"):
			gradient_method = "parameter-shift"

		return dataclasses.replace(
			config, gradient_method=gradient_method, use_device_gradient=False, grad_on_execution=False
		)

	def supports_derivatives(
		self, execution_config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
	) -> bool:
		"""Return False to delegate gradient computation to PennyLane transforms."""
		return False

	def preprocess_transforms(self, execution_config: ExecutionConfig | None = None) -> qml.CompilePipeline:
		"""Build the compilation pipeline for IQM hardware.

		Order:
		1. validate_device_wires  - reject undeclared wires
		2. validate_measurements  - reject unsupported measurement types
		3. split_non_commuting    - one tape per commuting observable group
		4. diagonalize_measurements - prepend Z-basis rotation gates
		5. decompose              - reduce all gates to SUPPORTED_OPS
		6. transpile (optional)   - SWAP routing to hardware topology
		7. decompose (if routed)  - reduce routing SWAPs to SUPPORTED_OPS
		8. broadcast_expand       - split parameter batches into scalar tapes
		"""
		program = qml.CompilePipeline()

		if self.wires is not None:
			program.add_transform(validate_device_wires, wires=self.wires, name=self.name)

		program.add_transform(validate_measurements, name=self.name)
		program.add_transform(qml.transforms.split_non_commuting)
		program.add_transform(qml.transforms.diagonalize_measurements)
		program.add_transform(decompose, stopping_condition=stopping_condition, name=self.name)

		coupling_map = self._pl_coupling_map()
		if coupling_map is not None:
			program.add_transform(_transpile, coupling_map=coupling_map)
			program.add_transform(decompose, stopping_condition=stopping_condition, name=self.name)

		program.add_transform(qml.transforms.broadcast_expand)

		return program

	@overload
	def execute(self, circuits: QuantumScript, execution_config: ExecutionConfig | None = ...) -> Result: ...

	@overload
	def execute(
		self, circuits: Sequence[QuantumScript], execution_config: ExecutionConfig | None = ...
	) -> Sequence[Result]: ...

	def execute(
		self, circuits: QuantumScript | Sequence[QuantumScript], execution_config: ExecutionConfig | None = None
	) -> Result | Sequence[Result]:
		"""Execute one or more QuantumScripts on IQM hardware.

		``single_tape_support`` wraps single-tape calls into a batch before this
		method runs, so ``circuits`` is always a sequence at runtime. Circuits
		with the same shot count are submitted together in one IQM job.
		"""
		batch = [circuits] if isinstance(circuits, QuantumScript) else circuits
		prepared = [self._prepare_circuit(index, tape) for index, tape in enumerate(batch)]

		shot_groups: dict[int, list[_PreparedCircuit]] = {}
		for item in prepared:
			shot_groups.setdefault(item.shots, []).append(item)

		results: dict[int, Result] = {}
		for shots, items in shot_groups.items():
			job = self.client.submit_circuits(
				[item.circuit for item in items], shots=shots, qubit_mapping=self._qubit_mapping, options=self._options
			)
			batch_result = self._wait_for_job(job)
			if len(batch_result) != len(items):
				raise RuntimeError(
					f"IQM job {job.job_id} returned {len(batch_result)} circuit results "
					f"for {len(items)} submitted circuits."
				)

			for item, measurements in zip(items, batch_result, strict=True):
				samples = iqm_result_to_samples(measurements, item.wire_map, item.measured_wires)
				results[item.index] = self._postprocess(item.tape, samples, item.measured_wires)

		return tuple(results[index] for index in range(len(prepared)))

	def _prepare_circuit(self, index: int, tape: QuantumScript) -> _PreparedCircuit:
		"""Translate one QuantumScript and retain the metadata needed to decode its result."""
		wire_map = build_wire_map(tape, self.wires)
		measured_wires = self._collect_measured_wires(tape)

		circuit = tape_to_iqm_circuit(tape, wire_map)

		dqa = self.architecture
		if dqa is not None and dqa.computational_resonators:
			circuit = transpile_insert_moves(circuit, dqa)

		shots = tape.shots.total_shots if tape.shots else self._default_shots
		if shots is None:
			raise ValueError(
				"shots must be specified on the QNode (e.g. @qml.qnode(dev, shots=1024)) "
				"or as a device default (IQMDevice(..., shots=1024))."
			)

		return _PreparedCircuit(index, tape, circuit, wire_map, measured_wires, shots)

	def _collect_measured_wires(self, tape: QuantumScript) -> list[Hashable]:
		"""Return ordered, deduplicated list of wires that need measure instructions."""
		measured: list[Hashable] = []
		for mp in tape.measurements:
			src = mp.wires if mp.wires else tape.wires
			for w in src:
				if w not in measured:
					measured.append(w)
		return measured

	def _wait_for_job(self, job: CircuitJob) -> CircuitMeasurementResultsBatch:
		"""Block-poll until the job reaches a terminal state; return result."""
		deadline = time.time() + self._timeout
		while time.time() < deadline:
			status = job.update()
			if status == JobStatus.COMPLETED:
				result = job.result()
				if result is None:
					raise RuntimeError(f"IQM job {job.job_id} completed but returned no result.")
				return result
			if status in (JobStatus.FAILED, JobStatus.CANCELLED):
				raise RuntimeError(
					f"IQM job {job.job_id} ended with status '{status}'. Server errors: {job.data.errors}"
				)
			time.sleep(self._poll_interval)

		raise TimeoutError(f"IQM job {job.job_id} did not complete within {self._timeout}s.")

	def _postprocess(self, tape: QuantumScript, samples: np.ndarray, measured_wires: list) -> Result:
		"""Convert (shots, n_wires) bit samples to PennyLane result types.

		After diagonalize_measurements all observables are in the Z-basis,
		so every measurement is a SampleMeasurement (expval/var/probs/sample/counts).
		"""
		results: list[Result] = []
		for mp in tape.measurements:
			if not isinstance(mp, SampleMeasurement):
				raise TypeError(f"IQMDevice expects SampleMeasurement after diagonalization, got {type(mp).__name__}")
			if mp.wires:
				col_idx = [measured_wires.index(w) for w in mp.wires]
				mp_samples = samples[:, col_idx]
				mp_wires = qml.wires.Wires(list(mp.wires))
			else:
				mp_samples = samples
				mp_wires = qml.wires.Wires(measured_wires)

			processed = mp.process_samples(mp_samples, mp_wires)
			if processed is None:
				raise RuntimeError(f"{type(mp).__name__}.process_samples returned None")
			results.append(processed)

		# `Result` is a constrained TypeVar (dict, tuple, TensorLike); a
		# `tuple[Result, ...]` is a valid `tuple` instantiation, but pyrefly
		# cannot bind the constraint at the union site.
		return cast(Result, results[0] if len(results) == 1 else tuple(results))
