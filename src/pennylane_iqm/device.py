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
from iqm.station_control.client.qon import ObservationFinder
from iqm.station_control.interface.models import CircuitMeasurementResultsBatch, DynamicQuantumArchitecture
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import decompose, validate_device_wires, validate_measurements
from pennylane.measurements import SampleMeasurement
from pennylane.tape import QuantumScript
from pennylane.typing import Result

from .gates import stopping_condition
from .layout import _DEFAULT_MAX_CANDIDATES, _map_tape, _select_initial_layout, _transpile
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
def _optimized_transpile(
	tape: QuantumScript, dqa: DynamicQuantumArchitecture, metrics: ObservationFinder | None, max_candidates: int
) -> tuple:
	"""Select physical qubits using topology and optional quality metrics."""
	selected_layout, coupling_map = _select_initial_layout(tape, dqa, metrics, max_candidates)
	mapped_tape = _map_tape(tape, selected_layout)

	if not any(len(operation.wires) == 2 for operation in mapped_tape.operations):
		return [mapped_tape], lambda results: results[0]

	transpile_tape = _transpile.tape_transform
	if transpile_tape is None:
		raise RuntimeError("_transpile.tape_transform is unexpectedly None")
	return transpile_tape(mapped_tape, coupling_map=coupling_map)


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
	    optimize_layout: Select a circuit-aware initial layout from the DQA
		    topology. Native zero-SWAP embeddings take priority; routed layouts
		    minimize emitted CZ gates. Requires use_connectivity=True and no
		    qubit_mapping. Defaults to True.
	    use_metrics: Select a circuit-aware initial layout using calibration
		    quality metrics. Implies optimize_layout=True.
	    max_candidates: Maximum complete layouts scored in each optimized
		    search phase. Defaults to 10000.
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
		use_metrics: bool = False,
		max_candidates: int = _DEFAULT_MAX_CANDIDATES,
		optimize_layout: bool = True,
	):
		if shots is not None and shots < 1:
			raise ValueError("IQMDevice requires shots >= 1 (hardware execution).")
		if (optimize_layout or use_metrics) and not use_connectivity:
			raise ValueError("Layout optimization requires use_connectivity=True.")
		if (optimize_layout or use_metrics) and qubit_mapping is not None:
			raise ValueError("Layout optimization cannot be combined with qubit_mapping.")
		if max_candidates < 1:
			raise ValueError("max_candidates must be at least 1.")

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
		self._optimize_layout = optimize_layout or use_metrics
		self._use_metrics = use_metrics
		self._max_candidates = max_candidates

		self._client: IQMClient | None = None
		self._dqa: DynamicQuantumArchitecture | None = None
		self._metrics: ObservationFinder | None = None

		if wires is None:
			if not use_connectivity:
				raise ValueError("wires must be specified explicitly when use_connectivity=False.")
			# Eagerly fetch the DQA so we know how many qubits this device has.
			try:
				self._dqa = self.client.get_dynamic_quantum_architecture()
				wires = len(self._dqa.qubits)
			except (RuntimeError, OSError, ValueError) as exc:
				raise RuntimeError(f"Could not auto-detect wires from server: {exc}") from exc

		# Do NOT pass shots to super().__init__ — due to Pennylane deprecation
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

	@property
	def metrics(self) -> ObservationFinder | None:
		"""Return calibration quality metrics used for initial layout selection.

		Returns:
			Quality metrics for the DQA calibration set, or None when metric-based
			layout selection is disabled.

		Raises:
			RuntimeError: If metrics are enabled but no DQA is available.
		"""
		if not self._use_metrics:
			return None
		_, metrics = self._layout_context()
		return metrics

	def _layout_context(self) -> tuple[DynamicQuantumArchitecture, ObservationFinder | None]:
		"""Return the DQA and optional metrics required by layout optimization."""
		dqa = self.architecture
		if dqa is None:
			raise RuntimeError("Cannot optimize layout without a dynamic quantum architecture.")
		if self._use_metrics and self._metrics is None:
			self._metrics = self.client.get_calibration_quality_metrics(dqa.calibration_set_id)
		return dqa, self._metrics

	def _wire_map(self, tape: QuantumScript | None = None) -> WireMap:
		"""Map device wires to circuit qubit names."""
		dqa = self.architecture
		if self._optimize_layout and tape is not None:
			if dqa is None:
				raise RuntimeError("Cannot translate an optimized layout without a dynamic quantum architecture.")
			unknown = set(tape.wires) - set(dqa.qubits)
			if unknown:
				raise RuntimeError(f"Layout preprocessing did not map wires to physical IQM qubits: {unknown}.")
			return {wire: str(wire) for wire in tape.wires}
		iqm_qubits = dqa.qubits if dqa is not None and self._qubit_mapping is None else None
		return build_wire_map(tape, self.wires, iqm_qubits)

	def _physical_wire_map(self) -> WireMap:
		"""Map device wires to the physical qubits used for routing."""
		wire_map = self._wire_map()
		if self._qubit_mapping is None:
			return wire_map

		missing = set(wire_map.values()) - self._qubit_mapping.keys()
		if missing:
			raise ValueError(f"qubit_mapping does not contain the circuit qubits {sorted(missing)}.")

		physical_wire_map = {wire: self._qubit_mapping[logical] for wire, logical in wire_map.items()}
		if len(set(physical_wire_map.values())) != len(physical_wire_map):
			raise ValueError("qubit_mapping must map circuit qubits to distinct physical qubits.")

		dqa = self.architecture
		if dqa is not None:
			unknown = set(physical_wire_map.values()) - set(dqa.qubits)
			if unknown:
				raise ValueError(f"qubit_mapping contains physical qubits absent from the DQA: {sorted(unknown)}.")
		return physical_wire_map

	def _pl_coupling_map(self) -> list[tuple] | None:
		"""Derive a PennyLane coupling map from the DQA's CZ loci.

		Returns None when the architecture is unknown, causing the transpile
		step to be skipped. Qubit-resonator CZ loci are excluded here because
		transpile_insert_moves handles those separately for Star devices.
		"""
		dqa = self.architecture
		if dqa is None or "cz" not in dqa.gates:
			return None

		iqm_to_pl = {iqm: wire for wire, iqm in self._physical_wire_map().items()}

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

	def to_iqm_circuits(self, tape: QuantumScript, circuit_name: str = "pennylane_circuit") -> tuple[Circuit, ...]:
		"""Convert a PennyLane tape into IQM circuits without executing it.

		The device preprocessing pipeline decomposes, diagonalizes, broadcasts,
		and routes the tape exactly as it does before hardware execution. A single
		PennyLane tape can therefore produce more than one IQM circuit.

		Args:
			tape: PennyLane tape to preprocess and translate.
			circuit_name: Base name for the IQM circuits. When preprocessing
				produces multiple circuits, their zero-based index is appended.

		Returns:
			IQM circuits ready for IQM Client submission or Pulla compilation.
		"""
		program, _ = self.preprocess()
		preprocessed_tapes, _ = program(tape)
		multiple_circuits = len(preprocessed_tapes) > 1

		circuits = []
		for index, preprocessed_tape in enumerate(preprocessed_tapes):
			name = f"{circuit_name}_{index}" if multiple_circuits else circuit_name
			circuit, _, _ = self._translate_circuit(preprocessed_tape, name)
			circuits.append(circuit)
		return tuple(circuits)

	def preprocess_transforms(self, execution_config: ExecutionConfig | None = None) -> qml.CompilePipeline:
		"""Build the compilation pipeline for IQM hardware.

		Order:
		1. validate_device_wires  - reject undeclared wires
		2. validate_measurements  - reject unsupported measurement types
		3. split_non_commuting    - one tape per commuting observable group
		4. diagonalize_measurements - prepend Z-basis rotation gates
		5. decompose              - reduce all gates to SUPPORTED_OPS
		6. broadcast_expand       - split parameter batches before metric scoring
		7. transpile (optional)   - layout selection and SWAP routing
		8. decompose (if routed)  - reduce routing SWAPs to SUPPORTED_OPS
		"""
		program = qml.CompilePipeline()

		if self.wires is not None:
			program.add_transform(validate_device_wires, wires=self.wires, name=self.name)

		program.add_transform(validate_measurements, name=self.name)
		program.add_transform(qml.transforms.split_non_commuting)
		program.add_transform(qml.transforms.diagonalize_measurements)
		program.add_transform(decompose, stopping_condition=stopping_condition, name=self.name)

		if self._optimize_layout:
			dqa, metrics = self._layout_context()
			program.add_transform(qml.transforms.broadcast_expand)
			program.add_transform(_optimized_transpile, dqa=dqa, metrics=metrics, max_candidates=self._max_candidates)
			program.add_transform(decompose, stopping_condition=stopping_condition, name=self.name)
		else:
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
			if self._optimize_layout and self._dqa is not None:
				job = self.client.submit_circuits(
					[item.circuit for item in items],
					shots=shots,
					qubit_mapping=None,
					calibration_set_id=self._dqa.calibration_set_id,
					options=self._options,
				)
			else:
				job = self.client.submit_circuits(
					[item.circuit for item in items],
					shots=shots,
					qubit_mapping=self._qubit_mapping,
					options=self._options,
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
		circuit, wire_map, measured_wires = self._translate_circuit(tape)

		shots = tape.shots.total_shots if tape.shots else self._default_shots
		if shots is None:
			raise ValueError(
				"shots must be specified on the QNode (e.g. @qml.qnode(dev, shots=1024)) "
				"or as a device default (IQMDevice(..., shots=1024))."
			)

		return _PreparedCircuit(index, tape, circuit, wire_map, measured_wires, shots)

	def _translate_circuit(
		self, tape: QuantumScript, circuit_name: str = "pennylane_circuit"
	) -> tuple[Circuit, WireMap, list[Hashable]]:
		"""Translate a preprocessed tape and retain its measurement metadata."""
		wire_map = self._wire_map(tape)
		measured_wires = self._collect_measured_wires(tape)

		circuit = tape_to_iqm_circuit(tape, wire_map, circuit_name)

		dqa = self.architecture
		if dqa is not None and dqa.computational_resonators:
			circuit = transpile_insert_moves(circuit, dqa)

		return circuit, wire_map, measured_wires

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
