"""Translation layer: PennyLane operators -> IQM CircuitOperations."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from math import pi

import numpy as np
import pennylane as qml
from iqm.pulse.builder import CircuitOperation
from iqm.pulse.circuit_operations import Circuit
from pennylane.tape import QuantumScript

WireMap = dict[Hashable, str]
_TOLERANCE = 1e-10


def _prx(qubit: str, angle: float, phase: float) -> CircuitOperation:
	return CircuitOperation("prx", locus=(qubit,), args={"angle": float(angle), "phase": float(phase)})


def _rz_as_prx(qubit: str, theta: float) -> list[CircuitOperation]:
	"""RZ(theta) decomposed as three PRX gates: RX(-pi/2) RY(theta) RX(pi/2)."""
	return [_prx(qubit, pi / 2, pi), _prx(qubit, theta, pi / 2), _prx(qubit, pi / 2, 0.0)]


def _hadamard_as_prx(qubit: str) -> list[CircuitOperation]:
	"""H = RY(pi/2) RX(pi)."""
	return [_prx(qubit, pi / 2, pi / 2), _prx(qubit, pi, 0.0)]


def _cnot_as_cz(ctrl: str, tgt: str) -> list[CircuitOperation]:
	"""CNOT(ctrl, tgt) = H_tgt CZ H_tgt."""
	h = _hadamard_as_prx(tgt)
	return [*h, CircuitOperation("cz", locus=(ctrl, tgt), args={}), *h]


def _prx_matrix(operation: CircuitOperation) -> np.ndarray:
	"""Return the unitary matrix of a PRX operation."""
	angle = float(operation.args["angle"])
	phase = float(operation.args["phase"])
	cosine = np.cos(angle / 2)
	sine = np.sin(angle / 2)
	return np.array([[cosine, -1j * np.exp(-1j * phase) * sine], [-1j * np.exp(1j * phase) * sine, cosine]])


def _optimized_prx(unitary: np.ndarray, qubit: str, rz_angle: float) -> tuple[CircuitOperation | None, float]:
	"""Convert a one-qubit unitary to one PRX and an accumulated virtual RZ."""
	first_rz, ry, final_rz = qml.ops.one_qubit_decomposition(unitary, wire=0, rotations="ZYZ")
	lambda_angle = float(np.asarray(first_rz.parameters[0]).item())
	angle = float(np.asarray(ry.parameters[0]).item())
	phi_angle = float(np.asarray(final_rz.parameters[0]).item())

	phase = pi / 2 - lambda_angle - rz_angle
	new_rz_angle = rz_angle + phi_angle + lambda_angle
	if np.isclose(angle, 0.0, atol=_TOLERANCE, rtol=0):
		return None, new_rz_angle
	return _prx(qubit, angle, phase), new_rz_angle


def _optimize_single_qubit_gates(
	instructions: Sequence[CircuitOperation], drop_final_rz: bool = True
) -> tuple[CircuitOperation, ...]:
	"""Merge neighboring PRX gates and commute virtual RZ rotations through CZ gates.

	This follows qiskit-iqm's single-qubit optimization transpilation pass:
	each one-qubit block is expressed as one physical PRX and an RZ frame update.
	RZ commutes with CZ, so the frame only changes the phase of the next PRX.

	Args:
		instructions: Native PRX and CZ operations to optimize.
		drop_final_rz: Whether to drop terminal RZ frame updates. This preserves
			computational-basis measurement probabilities but not the unitary.

	Returns:
		Optimized native operations.

	Raises:
		ValueError: If an instruction is neither PRX nor CZ.
	"""
	optimized: list[CircuitOperation] = []
	pending: dict[str, np.ndarray] = {}
	rz_angles: dict[str, float] = {}

	def flush(qubit: str) -> None:
		unitary = pending.pop(qubit, None)
		if unitary is None:
			return
		operation, rz_angle = _optimized_prx(unitary, qubit, rz_angles.get(qubit, 0.0))
		rz_angles[qubit] = rz_angle
		if operation is not None:
			optimized.append(operation)

	for instruction in instructions:
		if instruction.name == "prx":
			(qubit,) = instruction.locus
			unitary = _prx_matrix(instruction)
			previous = pending.get(qubit)
			pending[qubit] = unitary if previous is None else unitary @ previous
		elif instruction.name == "cz":
			for qubit in instruction.locus:
				flush(qubit)
			optimized.append(instruction)
		else:
			raise ValueError(f"Unexpected operation '{instruction.name}' in single-qubit optimization.")

	for qubit in tuple(pending):
		flush(qubit)

	if not drop_final_rz:
		for qubit, rz_angle in rz_angles.items():
			if not np.isclose(rz_angle, 0.0, atol=_TOLERANCE, rtol=0):
				optimized.extend((_prx(qubit, -pi, 0.0), _prx(qubit, pi, rz_angle / 2)))

	return tuple(optimized)


def op_to_iqm(op: qml.operation.Operator, wire_map: WireMap) -> list[CircuitOperation]:
	"""Translate one PennyLane operator into a list of IQM CircuitOperations.

	Args:
		op: PennyLane operator to translate. Must be in SUPPORTED_OPS.
		wire_map: Mapping from PennyLane wire labels to IQM qubit names.

	Returns:
		Native IQM operations implementing the PennyLane gate.

	Raises:
		ValueError: If the operator's name is not in SUPPORTED_OPS.
	"""
	name = op.name
	# PennyLane parameters are TensorLike (numpy scalar / array / interface tensor).
	# np.asarray normalises them to ndarray, then .item() extracts a Python scalar
	# that float() accepts without static-type ambiguity.
	params = [float(np.asarray(p).item()) for p in op.parameters]
	wires = [wire_map[w] for w in op.wires]

	match name:
		case "RX":
			return [_prx(wires[0], params[0], 0.0)]
		case "RY":
			return [_prx(wires[0], params[0], pi / 2)]
		case "RZ" | "PhaseShift":
			return _rz_as_prx(wires[0], params[0])
		case "PauliX":
			return [_prx(wires[0], pi, 0.0)]
		case "PauliY":
			return [_prx(wires[0], pi, pi / 2)]
		case "PauliZ":
			return _rz_as_prx(wires[0], pi)
		case "Hadamard":
			return _hadamard_as_prx(wires[0])
		case "S":
			return _rz_as_prx(wires[0], pi / 2)
		case "T":
			return _rz_as_prx(wires[0], pi / 4)
		case "SX":
			return [_prx(wires[0], pi / 2, 0.0)]
		case "CZ":
			return [CircuitOperation("cz", locus=(wires[0], wires[1]), args={})]
		case "CNOT":
			return _cnot_as_cz(wires[0], wires[1])
		case _:
			raise ValueError(f"Unsupported gate '{name}' reached translator.")


def build_wire_map(tape: QuantumScript, device_wires: qml.wires.Wires | None) -> WireMap:
	"""Map PennyLane wire labels to IQM qubit names (QB1, QB2, ...).

	The i-th wire maps to ``QB{i+1}``, consistent with _pl_coupling_map.

	Args:
		tape: Quantum tape providing fallback wire ordering when device_wires
			is None.
		device_wires: Authoritative wire ordering. When provided, the mapping
			follows this order; otherwise it falls back to the tape's wires.

	Returns:
		Mapping from PennyLane wire labels to IQM qubit names.
	"""
	ordered = list(device_wires) if device_wires is not None else list(tape.wires)
	return {w: f"QB{i + 1}" for i, w in enumerate(ordered)}


def tape_to_iqm_circuit(tape: QuantumScript, wire_map: WireMap, circuit_name: str = "pennylane_circuit") -> Circuit:
	"""Convert a preprocessed QuantumScript into an IQM Circuit.

	Gate operations are translated first; measure instructions are appended
	last because IQM requires all measurements at the end.

	Args:
		tape: Tape whose operations have already been decomposed into
			SUPPORTED_OPS by the preprocessing pipeline.
		wire_map: Mapping from PennyLane wire labels to IQM qubit names.
		circuit_name: Name to attach to the resulting IQM circuit.

	Returns:
		An IQM Circuit ready for submission via IQMClient.submit_circuits.
	"""
	instructions: list[CircuitOperation] = []

	for op in tape.operations:
		instructions.extend(op_to_iqm(op, wire_map))
	instructions = list(_optimize_single_qubit_gates(instructions))

	measured_wires: list[Hashable] = []
	for mp in tape.measurements:
		src = mp.wires if mp.wires else tape.wires
		for w in src:
			if w not in measured_wires:
				measured_wires.append(w)

	for w in measured_wires:
		iqm_name = wire_map[w]
		instructions.append(CircuitOperation("measure", locus=(iqm_name,), args={"key": f"meas_{iqm_name}"}))

	return Circuit(name=circuit_name, instructions=tuple(instructions))
