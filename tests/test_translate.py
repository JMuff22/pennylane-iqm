"""Tests for the PennyLane -> IQM translation layer."""

from __future__ import annotations

from math import pi

import pennylane as qml
import pytest
from pennylane.tape import QuantumScript

from iqm.pulse.builder import CircuitOperation

from pennylane_iqm.translate import _hadamard_as_prx, _prx, _rz_as_prx, build_wire_map, op_to_iqm, tape_to_iqm_circuit


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
		wmap = build_wire_map(tape, qml.wires.Wires([0, 1]))
		assert wmap == {0: "QB1", 1: "QB2"}

	def test_string_wires(self):
		tape = QuantumScript([qml.PauliX(wires="a")], [qml.sample(wires=["a"])])
		wmap = build_wire_map(tape, qml.wires.Wires(["a", "b", "c"]))
		assert wmap == {"a": "QB1", "b": "QB2", "c": "QB3"}

	def test_no_device_wires_falls_back_to_tape(self):
		tape = QuantumScript([qml.PauliX(wires=3), qml.PauliX(wires=5)], [qml.sample(wires=[3, 5])])
		assert build_wire_map(tape, None) == {3: "QB1", 5: "QB2"}

	def test_device_wires_override_tape_order(self):
		# Device-wires ordering is what determines QB numbering, even when the
		# tape applies gates in a different order.
		tape = QuantumScript([qml.PauliX(wires=1), qml.PauliX(wires=0)], [qml.sample(wires=[0, 1])])
		wmap = build_wire_map(tape, qml.wires.Wires([0, 1]))
		assert wmap == {0: "QB1", 1: "QB2"}


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
