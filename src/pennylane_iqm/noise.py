"""Local noisy simulation of IQM QPUs from calibration data.

Calibration values are read through iqm-client's ``ObservationFinder``, so no
Qiskit or qiskit-iqm dependency is required.

Physical assumptions:

* Qubits relax towards the ground state (excited-state population 0).
* Gate infidelity is modelled as depolarizing noise with probability
  ``1 - fidelity``, spread uniformly over the non-identity Paulis, composed
  with thermal relaxation over the gate duration.
* One-qubit gates are charged as the hardware compiles them: each run of
  one-qubit gates between two-qubit gates is one PRX, and Z rotations are
  virtual and noiseless. Idle qubits do not decohere.
* Readout error is a classical confusion map applied immediately before
  measurement.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
import pennylane as qml
from iqm.station_control.client.qon import ObservationFinder
from iqm.station_control.interface.models import DynamicQuantumArchitecture
from pennylane.devices import Device
from pennylane.tape import QuantumScript


_PAULIS: tuple[np.ndarray, ...] = (
	np.eye(2),
	np.array([[0.0, 1.0], [1.0, 0.0]]),
	np.array([[0.0, -1.0j], [1.0j, 0.0]]),
	np.diag([1.0, -1.0]).astype(complex),
)


def _two_qubit_depolarizing_kraus(p: float) -> list[np.ndarray]:
	"""Kraus operators of a two-qubit depolarizing channel with error probability ``p``."""
	two_qubit_paulis = [np.kron(a, b) for a in _PAULIS for b in _PAULIS]
	return [np.sqrt(1.0 - p) * two_qubit_paulis[0]] + [np.sqrt(p / 15.0) * pauli for pauli in two_qubit_paulis[1:]]


def _readout_kraus(error_0_to_1: float, error_1_to_0: float) -> list[np.ndarray]:
	"""Kraus operators of the classical readout confusion map."""
	return [
		np.diag([np.sqrt(1.0 - error_0_to_1), np.sqrt(1.0 - error_1_to_0)]).astype(complex),
		np.sqrt(error_0_to_1) * np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex),
		np.sqrt(error_1_to_0) * np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
	]


@dataclasses.dataclass(frozen=True)
class IQMCalibration:
	"""Calibration data of an IQM QPU, keyed by physical qubit name.

	Times are in seconds. Loci absent from the calibration set are simply absent
	from these mappings, and receive no noise in :func:`mock_device`.

	Attributes:
		t1: Energy relaxation time per qubit (s).
		t2: Dephasing time per qubit (s).
		readout_errors: ``(error_0_to_1, error_1_to_0)`` per qubit.
		prx_error: PRX depolarizing probability per qubit.
		prx_duration: PRX gate duration per qubit (s).
		cz_error: CZ depolarizing probability per qubit pair.
		cz_duration: CZ gate duration per qubit pair (s).
	"""

	t1: dict[str, float]
	t2: dict[str, float]
	readout_errors: dict[str, tuple[float, float]]
	prx_error: dict[str, float]
	prx_duration: dict[str, float]
	cz_error: dict[tuple[str, str], float]
	cz_duration: dict[tuple[str, str], float]

	@property
	def qubits(self) -> list[str]:
		"""Qubits with coherence data, in calibration order."""
		return list(self.t1)

	@classmethod
	def from_metrics(cls, dqa: DynamicQuantumArchitecture, metrics: ObservationFinder) -> IQMCalibration:
		"""Read calibration data for the loci of a dynamic quantum architecture.

		The gate implementation queried for each locus is the one the calibration
		set declares as default, so no implementation name is hardcoded.

		Args:
			dqa: Dynamic quantum architecture of the calibration set.
			metrics: Quality metrics of the same calibration set.

		Returns:
			Calibration data for every locus the metrics cover.
		"""
		qubits = set(dqa.qubits)
		t1, t2 = metrics.get_coherence_times(dqa.qubits)

		prx_error: dict[str, float] = {}
		prx_duration: dict[str, float] = {}
		if (prx := dqa.gates.get("prx")) is not None:
			for locus in prx.loci:
				if len(locus) != 1 or locus[0] not in qubits:
					continue
				implementation = prx.get_default_implementation(locus)
				fidelity = metrics.get_gate_fidelity("prx", implementation, locus)
				duration = metrics.get_gate_duration("prx", implementation, locus)
				if fidelity is not None and duration is not None:
					prx_error[locus[0]] = max(0.0, 1.0 - fidelity)
					prx_duration[locus[0]] = duration

		cz_error: dict[tuple[str, str], float] = {}
		cz_duration: dict[tuple[str, str], float] = {}
		if (cz := dqa.gates.get("cz")) is not None:
			for locus in cz.loci:
				if len(locus) != 2 or not set(locus) <= qubits:
					continue
				implementation = cz.get_default_implementation(locus)
				fidelity = metrics.get_gate_fidelity("cz", implementation, locus)
				duration = metrics.get_gate_duration("cz", implementation, locus)
				if fidelity is not None and duration is not None:
					cz_error[(locus[0], locus[1])] = max(0.0, 1.0 - fidelity)
					cz_duration[(locus[0], locus[1])] = duration

		readout_errors: dict[str, tuple[float, float]] = {}
		if (measure := dqa.gates.get("measure")) is not None:
			for locus in measure.loci:
				if len(locus) != 1 or locus[0] not in qubits:
					continue
				implementation = measure.get_default_implementation(locus)
				errors = metrics.get_measure_errors("measure", implementation, locus)
				if errors is not None:
					readout_errors[locus[0]] = errors

		return cls(
			t1=t1,
			t2=t2,
			readout_errors=readout_errors,
			prx_error=prx_error,
			prx_duration=prx_duration,
			cz_error=cz_error,
			cz_duration=cz_duration,
		)

	@classmethod
	def uniform(
		cls,
		qubits: Sequence[str],
		connectivity: Iterable[tuple[str, str]],
		*,
		t1: float = 40e-6,
		t2: float = 30e-6,
		readout_errors: tuple[float, float] = (0.02, 0.03),
		prx_fidelity: float = 0.9995,
		prx_duration: float = 40e-9,
		cz_fidelity: float = 0.99,
		cz_duration: float = 100e-9,
	) -> IQMCalibration:
		"""Build sample calibration data with the same value on every locus.

		Useful for offline work when no calibration set is available. The
		defaults are representative of a superconducting IQM QPU but are not
		measured values of any specific device.

		Args:
			qubits: Physical qubit names, e.g. ``["QB1", "QB2"]``.
			connectivity: Qubit pairs that support CZ.
			t1: Energy relaxation time (s).
			t2: Dephasing time (s).
			readout_errors: ``(error_0_to_1, error_1_to_0)``.
			prx_fidelity: PRX gate fidelity.
			prx_duration: PRX gate duration (s).
			cz_fidelity: CZ gate fidelity.
			cz_duration: CZ gate duration (s).

		Returns:
			Calibration data covering every given qubit and pair.
		"""
		pairs = [(control, target) for control, target in connectivity]
		return cls(
			t1={qubit: t1 for qubit in qubits},
			t2={qubit: t2 for qubit in qubits},
			readout_errors={qubit: readout_errors for qubit in qubits},
			prx_error={qubit: max(0.0, 1.0 - prx_fidelity) for qubit in qubits},
			prx_duration={qubit: prx_duration for qubit in qubits},
			cz_error={pair: max(0.0, 1.0 - cz_fidelity) for pair in pairs},
			cz_duration={pair: cz_duration for pair in pairs},
		)


# Z rotations are virtual on IQM hardware: frame updates with no pulse, error or duration.
_VIRTUAL_Z_OPS = frozenset({"RZ", "PhaseShift", "PauliZ", "S", "T"})


@qml.transform
def _add_iqm_noise(tape: QuantumScript, calibration: IQMCalibration) -> tuple:
	"""Insert calibration-derived noise the way the circuit would run on hardware.

	The IQM translator merges each run of one-qubit gates between two-qubit gates
	into a single physical PRX (see ``_optimize_single_qubit_gates``), so each run
	gets PRX noise once, just before the gate that ends it. A run of only virtual
	Z rotations compiles to no pulse and gets none. A PRX whose merged angle
	happens to be zero is still charged, as the angle is unknown at trace time.
	CNOT is compiled as ``H CZ H`` on the target. Other multi-qubit operators
	receive no gate noise. Readout noise is applied to every calibrated qubit on
	the tape after the last gate.
	"""
	coherent = calibration.t1.keys() & calibration.t2.keys()
	operations: list[qml.operation.Operator] = []
	pending: set[str] = set()

	def relax(qubit: str, duration: float) -> None:
		operations.append(
			qml.ThermalRelaxationError(0.0, calibration.t1[qubit], calibration.t2[qubit], duration, wires=qubit)
		)

	def flush(wires: Iterable[str]) -> None:
		for qubit in wires:
			if qubit in pending:
				pending.discard(qubit)
				if qubit in calibration.prx_error and qubit in coherent:
					relax(qubit, calibration.prx_duration[qubit])
					operations.append(qml.DepolarizingChannel(calibration.prx_error[qubit], wires=qubit))

	for op in tape.operations:
		if isinstance(op, qml.operation.Channel):
			operations.append(op)
		elif len(op.wires) == 1:
			if op.name not in _VIRTUAL_Z_OPS:
				pending.update(op.wires.tolist())
			operations.append(op)
		elif op.name in ("CZ", "CNOT"):
			control, target = op.wires
			if op.name == "CNOT":
				pending.add(target)
			flush(op.wires)
			operations.append(op)
			pair = (control, target) if (control, target) in calibration.cz_error else (target, control)
			if pair in calibration.cz_error and {control, target} <= coherent:
				operations.append(
					qml.QubitChannel(_two_qubit_depolarizing_kraus(calibration.cz_error[pair]), wires=op.wires)
				)
				for qubit in op.wires:
					relax(qubit, calibration.cz_duration[pair])
			if op.name == "CNOT":
				pending.add(target)
		else:
			flush(op.wires)
			operations.append(op)

	flush(tape.wires)
	for qubit in tape.wires:
		if qubit in calibration.readout_errors:
			operations.append(qml.QubitChannel(_readout_kraus(*calibration.readout_errors[qubit]), wires=qubit))

	return [tape.copy(operations=operations)], lambda results: results[0]


def mock_device(calibration: IQMCalibration, wires: Sequence[str]) -> Device:
	"""Create a local noisy simulator standing in for an IQM QPU.

	The returned device is ``default.mixed`` with the calibration-derived noise
	model applied, so its memory cost grows as ``4 ** len(wires)``. Use the
	handful of physical qubits the circuit actually runs on. Shots are set
	per-QNode, as with :class:`~pennylane_iqm.IQMDevice`.

	Args:
		calibration: Calibration data of the QPU being modelled.
		wires: Physical IQM qubit names to simulate, e.g. ``["QB1", "QB2"]``.

	Returns:
		A PennyLane device that samples from the noisy QPU model.

	Raises:
		ValueError: If a requested qubit has no calibration data.
	"""
	unknown = set(wires) - calibration.t1.keys()
	if unknown:
		raise ValueError(f"No calibration data for qubits {sorted(unknown)}.")

	device = qml.device("default.mixed", wires=list(wires))
	# Transforms are untyped dispatchers; applied to a Device they return a
	# Device subclass, which their signature does not express.
	return cast(Device, _add_iqm_noise(device, calibration=calibration))
