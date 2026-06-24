"""IQM job result -> numpy sample array conversion."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np


def iqm_result_to_samples(
	measurements: dict[str, list[list[int]]], wire_map: dict[Hashable, str], measured_wires: list[Hashable]
) -> np.ndarray:
	"""Return a (shots, n_wires) int array from an IQM circuit measurements dict.

	The IQM result dict maps measurement key -> list of shots, where each shot
	is a list of bits (one element per qubit in the measurement locus). Because
	we measure each qubit individually, every shot list has exactly one bit.

	Args:
		measurements: One entry of ``CircuitMeasurementResultsBatch`` -- the
			result for a single submitted circuit.
		wire_map: Mapping from PennyLane wire labels to IQM qubit names.
		measured_wires: Wires whose measurements should appear in the output,
			in column order.

	Returns:
		Array of shape ``(shots, len(measured_wires))`` with integer 0/1 values.

	Raises:
		KeyError: If a measurement key for a requested wire is absent from
			``measurements``.
	"""
	columns = []
	for w in measured_wires:
		key = f"meas_{wire_map[w]}"
		if key not in measurements:
			raise KeyError(f"Key '{key}' missing from IQM result. Available: {list(measurements.keys())}")
		columns.append(np.array([row[0] for row in measurements[key]], dtype=int))

	return np.column_stack(columns) if columns else np.empty((0, 0), dtype=int)
