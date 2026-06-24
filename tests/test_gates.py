"""Tests for gate set constants and stopping condition."""

import pennylane as qml
import pytest

from pennylane_iqm.gates import SUPPORTED_OPS, SUPPORTED_SINGLE_QUBIT_OPS, SUPPORTED_TWO_QUBIT_OPS, stopping_condition


def test_supported_ops_is_union_of_subsets():
	assert SUPPORTED_SINGLE_QUBIT_OPS <= SUPPORTED_OPS
	assert SUPPORTED_TWO_QUBIT_OPS <= SUPPORTED_OPS
	assert SUPPORTED_OPS == SUPPORTED_SINGLE_QUBIT_OPS | SUPPORTED_TWO_QUBIT_OPS


class TestStoppingCondition:
	@pytest.mark.parametrize(
		"name,wires,params,expected",
		[
			("RX", [0], [0.5], True),
			("RY", [0], [0.5], True),
			("RZ", [0], [0.5], True),
			("PauliX", [0], [], True),
			("PauliY", [0], [], True),
			("PauliZ", [0], [], True),
			("Hadamard", [0], [], True),
			("PhaseShift", [0], [0.3], True),
			("S", [0], [], True),
			("T", [0], [], True),
			("SX", [0], [], True),
			("CZ", [0, 1], [], True),
			("CNOT", [0, 1], [], True),
			("SWAP", [0, 1], [], False),
			("Toffoli", [0, 1, 2], [], False),
		],
	)
	def test_stopping_condition(self, name, wires, params, expected):
		op = getattr(qml, name)(*params, wires=wires)
		assert stopping_condition(op) is expected
