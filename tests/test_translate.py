"""Tests for the PennyLane -> IQM translation layer."""

from __future__ import annotations

from math import pi

import numpy as np
import pennylane as qml
import pytest
from pennylane.tape import QuantumScript

from iqm.pulse.builder import CircuitOperation

from pennylane_iqm.translate import (
	_hadamard_as_prx,
	_optimize_single_qubit_gates,
	_prx,
	_rz_as_prx,
	build_wire_map,
	op_to_iqm,
	tape_to_iqm_circuit,
)


WIRE_MAP_2Q = {0: "QB1", 1: "QB2"}


def is_prx(op: CircuitOperation, qubit: str, angle: float, phase: float, tol: float = 1e-9) -> bool:
	return (
		op.name == "prx"
		and op.locus == (qubit,)
		and abs(op.args["angle"] - angle) < tol
		and abs(op.args["phase"] - phase) < tol
	)


def is_cz(op: CircuitOperation, q0: str, q1: str) -> bool:
	return op.name == "cz" and op.locus == (q0, q1) and op.args == {}


def is_measure(op: CircuitOperation, qubit: str) -> bool:
	return op.name == "measure" and op.locus == (qubit,) and op.args.get("key") == f"meas_{qubit}"


class TestPrxHelper:
	def test_structure(self):
		op = _prx("QB1", pi / 2, 0.0)
		assert op.name == "prx"
		assert op.locus == ("QB1",)
		assert op.args == {"angle": pi / 2, "phase": 0.0}


class TestRzAsPrx:
	def test_decomposition_structure(self):
		theta = 0.7
		ops = _rz_as_prx("QB1", theta)
		assert len(ops) == 3
		assert is_prx(ops[0], "QB1", pi / 2, pi)
		assert is_prx(ops[1], "QB1", theta, pi / 2)
		assert is_prx(ops[2], "QB1", pi / 2, 0.0)


class TestHadamardAsPrx:
	def test_structure(self):
		ops = _hadamard_as_prx("QB1")
		assert len(ops) == 2
		assert is_prx(ops[0], "QB1", pi / 2, pi / 2)
		assert is_prx(ops[1], "QB1", pi, 0.0)


class TestOpToIqmSingleQubit:
	"""Each PennyLane op decomposes into the expected sequence of IQM PRX ops.

	For ops that map directly to one PRX, ``expected_args`` is (angle, phase) of
	the single PRX. For RZ-style ops that decompose into three PRX gates we
	check the middle PRX's angle (which carries the rotation parameter).
	"""

	@pytest.mark.parametrize(
		"op_factory,n_ops,check",
		[
			(lambda: qml.RX(0.5, wires=0), 1, ("single_prx", 0.5, 0.0)),
			(lambda: qml.RY(1.2, wires=0), 1, ("single_prx", 1.2, pi / 2)),
			(lambda: qml.PauliX(wires=0), 1, ("single_prx", pi, 0.0)),
			(lambda: qml.PauliY(wires=0), 1, ("single_prx", pi, pi / 2)),
			(lambda: qml.Hadamard(wires=0), 2, None),
			(lambda: qml.SX(wires=0), 1, ("single_prx", pi / 2, 0.0)),
			(lambda: qml.RZ(0.7, wires=0), 3, ("rz_angle", 0.7)),
			(lambda: qml.PhaseShift(0.7, wires=0), 3, ("rz_angle", 0.7)),
			(lambda: qml.PauliZ(wires=0), 3, ("rz_angle", pi)),
			(lambda: qml.S(wires=0), 3, ("rz_angle", pi / 2)),
			(lambda: qml.T(wires=0), 3, ("rz_angle", pi / 4)),
		],
	)
	def test_single_qubit_decomposition(self, op_factory, n_ops, check):
		result = op_to_iqm(op_factory(), WIRE_MAP_2Q)
		assert len(result) == n_ops
		assert all(o.locus == ("QB1",) for o in result)
		if check is None:
			return
		kind, *params = check
		if kind == "single_prx":
			angle, phase = params
			assert is_prx(result[0], "QB1", angle, phase)
		elif kind == "rz_angle":
			(angle,) = params
			assert result[1].args["angle"] == pytest.approx(angle)

	def test_phase_shift_matches_rz(self):
		# PhaseShift(phi) == RZ(phi) up to global phase, so decomposition is identical.
		ps = op_to_iqm(qml.PhaseShift(0.7, wires=0), WIRE_MAP_2Q)
		rz = op_to_iqm(qml.RZ(0.7, wires=0), WIRE_MAP_2Q)
		assert [o.args for o in ps] == [o.args for o in rz]

	def test_wire_mapping_applied(self):
		result = op_to_iqm(qml.PauliX(wires=5), {5: "QB3", 6: "QB4"})
		assert result[0].locus == ("QB3",)

	def test_unsupported_op_raises(self):
		with pytest.raises(ValueError, match="Unsupported gate 'SWAP'"):
			op_to_iqm(qml.SWAP(wires=[0, 1]), WIRE_MAP_2Q)


class TestOpToIqmTwoQubit:
	def test_cz(self):
		result = op_to_iqm(qml.CZ(wires=[0, 1]), WIRE_MAP_2Q)
		assert len(result) == 1
		assert is_cz(result[0], "QB1", "QB2")

	def test_cnot_decomposes_to_h_cz_h(self):
		result = op_to_iqm(qml.CNOT(wires=[0, 1]), WIRE_MAP_2Q)
		# H_tgt CZ H_tgt -> 2 + 1 + 2 = 5 ops, CZ in the middle, no PRX on ctrl.
		assert len(result) == 5
		assert result[2].name == "cz"
		assert result[2].locus == ("QB1", "QB2")
		assert [o for o in result if o.name == "prx" and o.locus == ("QB1",)] == []

	def test_cz_wire_order_preserved(self):
		result = op_to_iqm(qml.CZ(wires=[1, 0]), WIRE_MAP_2Q)
		assert result[0].locus == ("QB2", "QB1")


class TestBuildWireMap:
	def test_integer_wires(self):
		tape = QuantumScript([qml.PauliX(wires=0), qml.PauliX(wires=1)], [qml.sample(wires=[0, 1])])
		wmap = build_wire_map(tape, qml.wires.Wires([0, 1]), ["QB1", "QB2"])
		assert wmap == {0: "QB1", 1: "QB2"}

	def test_string_wires(self):
		tape = QuantumScript([qml.PauliX(wires="a")], [qml.sample(wires=["a"])])
		wmap = build_wire_map(tape, qml.wires.Wires(["a", "b", "c"]), ["QB1", "QB2", "QB3"])
		assert wmap == {"a": "QB1", "b": "QB2", "c": "QB3"}

	def test_no_device_wires_falls_back_to_tape(self):
		tape = QuantumScript([qml.PauliX(wires=3), qml.PauliX(wires=5)], [qml.sample(wires=[3, 5])])
		with pytest.warns(UserWarning, match="not validated against a physical IQM backend"):
			assert build_wire_map(tape, None) == {3: "QB1", 5: "QB2"}

	def test_device_wires_override_tape_order(self):
		# Device-wires ordering determines the mapping even when the tape
		# applies gates in a different order.
		tape = QuantumScript([qml.PauliX(wires=1), qml.PauliX(wires=0)], [qml.sample(wires=[0, 1])])
		wmap = build_wire_map(tape, qml.wires.Wires([0, 1]), ["QB1", "QB2"])
		assert wmap == {0: "QB1", 1: "QB2"}

	def test_physical_qubit_names_are_used(self):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0, 1])])
		wmap = build_wire_map(tape, qml.wires.Wires([0, 1]), ["QB1", "QB3"])
		assert wmap == {0: "QB1", 1: "QB3"}

	@pytest.mark.parametrize("iqm_qubits", [None, []])
	def test_generated_qubit_names_warn(self, iqm_qubits):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])])
		with pytest.warns(UserWarning, match=r"generated QB\{i \+ 1\} names"):
			assert build_wire_map(tape, qml.wires.Wires([0]), iqm_qubits) == {0: "QB1"}

	def test_too_few_physical_qubits_raises(self):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0, 1])])
		with pytest.raises(ValueError, match="Cannot map 2 PennyLane wires to the 1 provided IQM qubit names"):
			build_wire_map(tape, qml.wires.Wires([0, 1]), ["QB1"])


class TestTapeToIqmCircuit:
	WIRE_MAP = {0: "QB1", 1: "QB2"}

	def test_pauli_x_then_measure(self):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])])
		ops = list(tape_to_iqm_circuit(tape, self.WIRE_MAP).instructions)
		assert [o.name for o in ops] == ["prx", "measure"]
		assert is_measure(ops[-1], "QB1")

	def test_measurements_come_last(self):
		# IQM requires all measurement instructions at the end of the circuit.
		tape = QuantumScript([qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])])
		ops = list(tape_to_iqm_circuit(tape, self.WIRE_MAP).instructions)
		measure_ops = [o for o in ops if o.name == "measure"]
		assert len(measure_ops) == 2
		assert ops[-2].name == "measure" and ops[-1].name == "measure"

	def test_no_duplicate_measure_per_wire(self):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0]), qml.probs(wires=[0])])
		circuit = tape_to_iqm_circuit(tape, self.WIRE_MAP)
		assert len([o for o in circuit.instructions if o.name == "measure"]) == 1

	def test_measure_key_format(self):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])])
		meas = [o for o in tape_to_iqm_circuit(tape, self.WIRE_MAP).instructions if o.name == "measure"][0]
		assert meas.args["key"] == "meas_QB1"

	def test_circuit_name(self):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])])
		assert tape_to_iqm_circuit(tape, self.WIRE_MAP).name == "pennylane_circuit"
		assert tape_to_iqm_circuit(tape, self.WIRE_MAP, circuit_name="x").name == "x"

	def test_empty_operations(self):
		tape = QuantumScript([], [qml.sample(wires=[0])])
		ops = list(tape_to_iqm_circuit(tape, self.WIRE_MAP).instructions)
		assert all(o.name == "measure" for o in ops)

	def test_cnot_uses_one_prx_on_each_side_of_cz(self):
		tape = QuantumScript([qml.RY(0.37, wires=0), qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])])

		ops = list(tape_to_iqm_circuit(tape, self.WIRE_MAP).instructions)
		cz_index = next(index for index, op in enumerate(ops) if op.name == "cz")
		target_prx_before = [op for op in ops[:cz_index] if op.name == "prx" and op.locus == ("QB2",)]
		target_prx_after = [op for op in ops[cz_index + 1 :] if op.name == "prx" and op.locus == ("QB2",)]

		assert len(target_prx_before) == 1
		assert len(target_prx_after) == 1
		assert target_prx_before[0].args["angle"] == pytest.approx(pi / 2)
		assert target_prx_after[0].args["angle"] == pytest.approx(pi / 2)
		assert sum(op.name == "prx" for op in ops) == 3

	def test_adjacent_single_qubit_gates_fuse_to_one_prx(self):
		tape = QuantumScript(
			[qml.RX(0.2, wires=0), qml.RY(0.3, wires=0), qml.RZ(0.4, wires=0)], [qml.sample(wires=[0])]
		)

		ops = tape_to_iqm_circuit(tape, self.WIRE_MAP).instructions

		assert [op.name for op in ops] == ["prx", "measure"]

	def test_terminal_rz_is_virtual(self):
		tape = QuantumScript([qml.RZ(0.4, wires=0)], [qml.sample(wires=[0])])

		ops = tape_to_iqm_circuit(tape, self.WIRE_MAP).instructions

		assert [op.name for op in ops] == ["measure"]


class TestOptimizeSingleQubitGates:
	@staticmethod
	def mixed_instructions() -> tuple[CircuitOperation, ...]:
		operations = [
			qml.Hadamard(wires=0),
			qml.RZ(0.23, wires=0),
			qml.RX(-0.41, wires=1),
			qml.CNOT(wires=[0, 1]),
			qml.RY(0.67, wires=0),
			qml.S(wires=1),
			qml.CZ(wires=[0, 1]),
			qml.T(wires=0),
			qml.Hadamard(wires=1),
		]
		return tuple(instruction for operation in operations for instruction in op_to_iqm(operation, WIRE_MAP_2Q))

	@staticmethod
	def circuit_matrix(instructions: list[CircuitOperation] | tuple[CircuitOperation, ...]) -> np.ndarray:
		operations = []
		wire_map = {"QB1": 0, "QB2": 1}
		for instruction in instructions:
			if instruction.name == "prx":
				angle = instruction.args["angle"]
				phase = instruction.args["phase"]
				cosine = np.cos(angle / 2)
				sine = np.sin(angle / 2)
				matrix = np.array(
					[[cosine, -1j * np.exp(-1j * phase) * sine], [-1j * np.exp(1j * phase) * sine, cosine]]
				)
				operations.append(qml.QubitUnitary(matrix, wires=wire_map[instruction.locus[0]]))
			elif instruction.name == "cz":
				operations.append(qml.CZ(wires=[wire_map[qubit] for qubit in instruction.locus]))
		tape = QuantumScript(operations)
		return qml.matrix(tape, wire_order=[0, 1])

	def test_preserves_unitary_when_final_rz_is_kept(self):
		instructions = self.mixed_instructions()

		original = self.circuit_matrix(instructions)
		optimized = self.circuit_matrix(_optimize_single_qubit_gates(instructions, drop_final_rz=False))
		relative = original @ optimized.conj().T
		global_phase = relative[0, 0]

		assert abs(global_phase) == pytest.approx(1.0)
		assert relative == pytest.approx(global_phase * np.eye(4), abs=1e-9)

	def test_dropped_final_rz_preserves_measurement_probabilities(self):
		instructions = self.mixed_instructions()

		original = self.circuit_matrix(instructions)
		optimized = self.circuit_matrix(_optimize_single_qubit_gates(instructions))
		rng = np.random.default_rng(1234)
		states = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
		states /= np.linalg.norm(states, axis=1, keepdims=True)

		for state in states:
			assert np.abs(optimized @ state) ** 2 == pytest.approx(np.abs(original @ state) ** 2)

	def test_rejects_non_native_instruction(self):
		measure = CircuitOperation("measure", locus=("QB1",), args={"key": "m"})

		with pytest.raises(ValueError, match="Unexpected operation 'measure'"):
			_optimize_single_qubit_gates([measure])
