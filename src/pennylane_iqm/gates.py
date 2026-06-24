"""Supported gate set and decomposition stopping condition."""

from __future__ import annotations

import pennylane as qml

SUPPORTED_SINGLE_QUBIT_OPS: frozenset[str] = frozenset(
	{"RX", "RY", "RZ", "PauliX", "PauliY", "PauliZ", "Hadamard", "PhaseShift", "S", "T", "SX"}
)
SUPPORTED_TWO_QUBIT_OPS: frozenset[str] = frozenset({"CZ", "CNOT"})
SUPPORTED_OPS: frozenset[str] = SUPPORTED_SINGLE_QUBIT_OPS | SUPPORTED_TWO_QUBIT_OPS


def stopping_condition(op: qml.operation.Operator) -> bool:
	"""Return True when the operator is already in the IQM translator's vocabulary."""
	return op.name in SUPPORTED_OPS
