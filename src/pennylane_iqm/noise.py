"""Local noisy simulation of IQM QPUs from calibration data.

Calibration values are read through iqm-client's ``ObservationFinder``, so no
Qiskit or qiskit-iqm dependency is required.

Physical assumptions:

* Qubits relax towards the ground state (excited-state population 0).
* Gate infidelity is modelled as depolarizing noise with probability
  ``1 - fidelity``, spread uniformly over the non-identity Paulis, composed
  with thermal relaxation over the gate duration.
* Readout error is a classical confusion map applied immediately before
  measurement.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import cast

import numpy as np
import pennylane as qml
from iqm.station_control.client.qon import ObservationFinder
from iqm.station_control.interface.models import DynamicQuantumArchitecture
from pennylane.devices import Device

from .gates import SUPPORTED_SINGLE_QUBIT_OPS

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
	from these mappings, and receive no noise in the generated noise model.

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


def _single_qubit_noise(t1: float, t2: float, duration: float, error: float) -> Callable:
	def apply(op: qml.operation.Operator, **kwargs) -> None:
		qml.ThermalRelaxationError(0.0, t1, t2, duration, wires=op.wires)
		qml.DepolarizingChannel(error, wires=op.wires)

	return apply


def _two_qubit_noise(coherence: dict[str, tuple[float, float]], duration: float, error: float) -> Callable:
	kraus = _two_qubit_depolarizing_kraus(error)

	def apply(op: qml.operation.Operator, **kwargs) -> None:
		qml.QubitChannel(kraus, wires=op.wires)
		for wire in op.wires:
			t1, t2 = coherence[wire]
			qml.ThermalRelaxationError(0.0, t1, t2, duration, wires=wire)

	return apply


def _readout_noise(qubit: str, error_0_to_1: float, error_1_to_0: float) -> Callable:
	kraus = _readout_kraus(error_0_to_1, error_1_to_0)

	def apply(op: qml.measurements.MeasurementProcess, **kwargs) -> None:
		qml.QubitChannel(kraus, wires=qubit)

	return apply


def iqm_noise_model(calibration: IQMCalibration) -> qml.NoiseModel:
	"""Build a PennyLane noise model from IQM calibration data.

	Circuit wires must be labelled with physical IQM qubit names, which is how
	:class:`~pennylane_iqm.IQMDevice` labels them once layout optimization has run.

	Args:
		calibration: Calibration data of the QPU being modelled.

	Returns:
		Noise model applying gate and readout noise to the calibrated loci.
	"""
	single_qubit_ops = sorted(SUPPORTED_SINGLE_QUBIT_OPS)
	model_map: dict = {}

	for qubit, error in calibration.prx_error.items():
		if qubit not in calibration.t1 or qubit not in calibration.t2:
			continue
		condition = qml.noise.op_in(single_qubit_ops) & qml.noise.wires_eq(qubit)
		model_map[condition] = _single_qubit_noise(
			calibration.t1[qubit], calibration.t2[qubit], calibration.prx_duration[qubit], error
		)

	for (control, target), error in calibration.cz_error.items():
		if not {control, target} <= calibration.t1.keys() & calibration.t2.keys():
			continue
		coherence = {qubit: (calibration.t1[qubit], calibration.t2[qubit]) for qubit in (control, target)}
		noise = _two_qubit_noise(coherence, calibration.cz_duration[(control, target)], error)
		# CZ is symmetric, but PennyLane matches the operator's wire order.
		model_map[qml.noise.op_eq(qml.CZ) & qml.noise.wires_eq([control, target])] = noise
		model_map[qml.noise.op_eq(qml.CZ) & qml.noise.wires_eq([target, control])] = noise

	meas_map = {
		qml.noise.wires_in([qubit]): _readout_noise(qubit, error_0_to_1, error_1_to_0)
		for qubit, (error_0_to_1, error_1_to_0) in calibration.readout_errors.items()
	}

	return qml.NoiseModel(model_map, meas_map=meas_map)


def mock_device(calibration: IQMCalibration, wires: Sequence[str], shots: int | None = None) -> Device:
	"""Create a local noisy simulator standing in for an IQM QPU.

	The returned device is ``default.mixed`` with the calibration-derived noise
	model applied, so its memory cost grows as ``4 ** len(wires)``. Use the
	handful of physical qubits the circuit actually runs on.

	Args:
		calibration: Calibration data of the QPU being modelled.
		wires: Physical IQM qubit names to simulate, e.g. ``["QB1", "QB2"]``.
		shots: Shots per execution. Defaults to None (analytic).

	Returns:
		A PennyLane device that samples from the noisy QPU model.

	Raises:
		ValueError: If a requested qubit has no calibration data.
	"""
	unknown = set(wires) - calibration.t1.keys()
	if unknown:
		raise ValueError(f"No calibration data for qubits {sorted(unknown)}.")

	device = qml.device("default.mixed", wires=list(wires), shots=shots)
	# qml.add_noise is an untyped dispatching transform; applied to a Device it
	# returns a Device subclass, which its signature does not express.
	return cast(Device, qml.add_noise(device, iqm_noise_model(calibration)))
