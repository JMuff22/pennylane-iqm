"""Tests for IQM result -> numpy sample array conversion."""

from __future__ import annotations

import numpy as np
import pytest

from pennylane_iqm.result import iqm_result_to_samples


class TestIqmResultToSamples:
	def test_single_qubit_single_shot(self):
		measurements = {"meas_QB1": [[1]]}
		wire_map = {0: "QB1"}
		samples = iqm_result_to_samples(measurements, wire_map, [0])
		assert samples.shape == (1, 1)
		assert samples[0, 0] == 1

	def test_two_qubit_four_shots(self):
		measurements = {"meas_QB1": [[0], [1], [0], [1]], "meas_QB2": [[1], [0], [1], [0]]}
		wire_map = {0: "QB1", 1: "QB2"}
		samples = iqm_result_to_samples(measurements, wire_map, [0, 1])
		assert samples.shape == (4, 2)
		np.testing.assert_array_equal(samples[:, 0], [0, 1, 0, 1])
		np.testing.assert_array_equal(samples[:, 1], [1, 0, 1, 0])

	def test_column_order_follows_measured_wires(self):
		measurements = {"meas_QB1": [[0], [0]], "meas_QB2": [[1], [1]]}
		wire_map = {0: "QB1", 1: "QB2"}
		# QB2 first in measured_wires
		samples = iqm_result_to_samples(measurements, wire_map, [1, 0])
		np.testing.assert_array_equal(samples[:, 0], [1, 1])
		np.testing.assert_array_equal(samples[:, 1], [0, 0])

	def test_missing_key_raises_key_error(self):
		measurements = {"meas_QB1": [[0]]}
		wire_map = {0: "QB1", 1: "QB2"}
		with pytest.raises(KeyError, match="meas_QB2"):
			iqm_result_to_samples(measurements, wire_map, [0, 1])

	def test_integer_dtype(self):
		measurements = {"meas_QB1": [[0], [1]]}
		wire_map = {0: "QB1"}
		samples = iqm_result_to_samples(measurements, wire_map, [0])
		assert samples.dtype == int

	def test_empty_measured_wires_returns_empty_array(self):
		samples = iqm_result_to_samples({}, {}, [])
		assert samples.shape == (0, 0)

	def test_all_zeros_circuit(self):
		measurements = {"meas_QB1": [[0], [0], [0]]}
		wire_map = {0: "QB1"}
		samples = iqm_result_to_samples(measurements, wire_map, [0])
		np.testing.assert_array_equal(samples, [[0], [0], [0]])

	def test_three_qubits(self):
		n_shots = 8
		measurements = {
			"meas_QB1": [[i % 2] for i in range(n_shots)],
			"meas_QB2": [[(i // 2) % 2] for i in range(n_shots)],
			"meas_QB3": [[(i // 4) % 2] for i in range(n_shots)],
		}
		wire_map = {0: "QB1", 1: "QB2", 2: "QB3"}
		samples = iqm_result_to_samples(measurements, wire_map, [0, 1, 2])
		assert samples.shape == (n_shots, 3)
