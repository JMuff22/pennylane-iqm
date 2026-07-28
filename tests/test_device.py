"""Tests for IQMDevice."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pennylane as qml
import pytest
from iqm.iqm_server_client.models import JobStatus
from pennylane.devices import ExecutionConfig
from pennylane.tape import QuantumScript

from pennylane_iqm.device import IQMDevice

from .conftest import make_completed_job, make_mock_job


def make_quality_metrics(cz_qb2: float | None = 0.8, cz_qb3: float | None = 0.99) -> MagicMock:
	"""Return deterministic calibration metrics for the 3-qubit test DQA."""
	fidelities = {("cz", ("QB1", "QB2")): cz_qb2, ("cz", ("QB1", "QB3")): cz_qb3}
	metrics = MagicMock()
	metrics.get_gate_fidelity.side_effect = lambda gate, _implementation, locus: fidelities.get((gate, locus), 0.99)
	return metrics


def make_device(
	wires: int | list = 3,
	shots: int | None = 100,
	use_connectivity: bool = False,
	server_url: str = "https://test.iqm.fi",
	**kwargs,
) -> IQMDevice:
	if not use_connectivity:
		kwargs.setdefault("optimize_layout", False)
	return IQMDevice(server_url, wires=wires, shots=shots, use_connectivity=use_connectivity, **kwargs)


class TestIQMDeviceInit:
	def test_integer_wires_expanded(self):
		dev = make_device(wires=3, shots=512)
		assert len(dev.wires) == 3
		assert dev._default_shots == 512

	def test_list_wires(self):
		dev = make_device(wires=["a", "b"])
		assert "a" in dev.wires
		assert "b" in dev.wires

	def test_shots_zero_raises(self):
		with pytest.raises(ValueError, match="shots >= 1"):
			IQMDevice("https://x", wires=2, shots=0)

	def test_shots_none_allowed(self):
		# shots=None is valid: shots must then be provided per-QNode.
		dev = make_device(wires=2, shots=None)
		assert dev._default_shots is None

	def test_token_from_explicit_kwarg(self):
		assert make_device(token="my-token")._token == "my-token"

	def test_token_not_read_from_env(self, monkeypatch):
		# IQM_TOKEN belongs to IQMClient, not to IQMDevice.
		monkeypatch.setenv("IQM_TOKEN", "env-token")
		assert make_device()._token is None

	def test_wires_none_requires_connectivity(self):
		with pytest.raises(ValueError, match="use_connectivity=False"):
			IQMDevice("https://x", wires=None, use_connectivity=False, optimize_layout=False)

	@pytest.mark.parametrize("layout_options", [{"optimize_layout": True}, {"use_metrics": True}])
	def test_layout_optimization_requires_connectivity(self, layout_options):
		with pytest.raises(ValueError, match="Layout optimization requires use_connectivity=True"):
			IQMDevice("https://x", wires=2, use_connectivity=False, **layout_options)

	@pytest.mark.parametrize("layout_options", [{"optimize_layout": True}, {"use_metrics": True}])
	def test_layout_optimization_cannot_be_combined_with_qubit_mapping(self, layout_options):
		with pytest.raises(ValueError, match="cannot be combined with qubit_mapping"):
			IQMDevice(
				"https://x",
				wires=2,
				use_connectivity=True,
				qubit_mapping={"QB1": "QB2", "QB2": "QB3"},
				**layout_options,
			)

	def test_candidate_limit_must_be_positive(self):
		with pytest.raises(ValueError, match="max_candidates must be at least 1"):
			IQMDevice("https://x", wires=2, use_metrics=True, max_candidates=0)

	def test_wires_auto_detected_and_cached_from_dqa(self, crystal_3q_dqa):
		with patch("pennylane_iqm.device.IQMClient") as MockClient:
			mock_client = MagicMock()
			mock_client.get_dynamic_quantum_architecture.return_value = crystal_3q_dqa
			MockClient.return_value = mock_client
			dev = IQMDevice("https://x", wires=None, use_connectivity=True)
		assert len(dev.wires) == 3
		assert dev._dqa is crystal_3q_dqa

	def test_wires_auto_detect_failure_raises_runtime_error(self):
		with patch("pennylane_iqm.device.IQMClient") as MockClient:
			mock_client = MagicMock()
			mock_client.get_dynamic_quantum_architecture.side_effect = RuntimeError("unreachable")
			MockClient.return_value = mock_client
			with pytest.raises(RuntimeError, match="Could not auto-detect"):
				IQMDevice("https://x", wires=None, use_connectivity=True)

	def test_no_deprecation_warning_on_construction(self, recwarn):
		# Regression: passing shots= must not surface PennyLane's device-shots
		# deprecation warning, because shots are stored on the device itself.
		make_device(wires=2, shots=100)
		pl_warnings = [
			w
			for w in recwarn.list
			if "PennyLaneDeprecationWarning" in str(type(w.category)) and "shots" in str(w.message).lower()
		]
		assert pl_warnings == []


class TestIQMDeviceClient:
	def test_client_created_without_token_kwarg(self):
		dev = make_device(token=None)
		with patch("pennylane_iqm.device.IQMClient") as MockClient:
			MockClient.return_value = MagicMock()
			_ = dev.client
			MockClient.assert_called_once_with("https://test.iqm.fi")

	def test_client_passes_explicit_token(self):
		dev = make_device(token="tok123")
		with patch("pennylane_iqm.device.IQMClient") as MockClient:
			MockClient.return_value = MagicMock()
			_ = dev.client
			MockClient.assert_called_once_with("https://test.iqm.fi", token="tok123")

	def test_client_does_not_pass_env_token(self, monkeypatch):
		# Critical: IQMClient raises ClientConfigurationError if token is passed
		# AND IQM_TOKEN is set. The device must let IQMClient read the env itself.
		monkeypatch.setenv("IQM_TOKEN", "env-token")
		dev = make_device(token=None)
		with patch("pennylane_iqm.device.IQMClient") as MockClient:
			MockClient.return_value = MagicMock()
			_ = dev.client
			MockClient.assert_called_once_with("https://test.iqm.fi")


class TestArchitecture:
	def test_use_connectivity_false_returns_none(self):
		dev = make_device(use_connectivity=False)
		assert dev.architecture is None
		assert dev.is_star is False

	def test_architecture_fetched_and_cached(self, crystal_3q_dqa, mock_client):
		dev = make_device(use_connectivity=True)
		dev._client = mock_client
		assert dev.architecture is crystal_3q_dqa
		_ = dev.architecture
		mock_client.get_dynamic_quantum_architecture.assert_called_once()

	def test_architecture_failure_emits_warning(self):
		dev = make_device(use_connectivity=True)
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.side_effect = RuntimeError("unreachable")
		dev._client = mock_client
		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")
			arch = dev.architecture
		assert arch is None
		assert any("Could not fetch DQA" in str(warning.message) for warning in w)

	def test_is_star_false_for_crystal(self, mock_client):
		dev = make_device(use_connectivity=True)
		dev._client = mock_client
		assert dev.is_star is False

	def test_is_star_true_for_star_arch(self, star_dqa):
		dev = make_device(use_connectivity=True)
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = star_dqa
		dev._client = mock_client
		assert dev.is_star is True


class TestCouplingMap:
	def test_no_arch_returns_none(self):
		assert make_device(use_connectivity=False)._pl_coupling_map() is None

	def test_crystal_coupling_map(self, mock_client):
		dev = make_device(wires=3, use_connectivity=True)
		dev._client = mock_client
		coupling = dev._pl_coupling_map()
		assert coupling is not None
		assert (0, 1) in coupling
		assert (0, 2) in coupling

	def test_star_arch_excludes_resonator_loci(self, star_dqa):
		# Star CZ loci are QB-CR pairs; transpile_insert_moves handles those
		# separately, so _pl_coupling_map yields no qubit-qubit pairs.
		dev = make_device(wires=3, use_connectivity=True)
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = star_dqa
		dev._client = mock_client
		assert dev._pl_coupling_map() is None

	def test_no_cz_gate_returns_none(self, crystal_3q_dqa, mock_client):
		dev = make_device(wires=3, use_connectivity=True)
		dev._client = mock_client
		crystal_3q_dqa.gates.pop("cz")
		assert dev._pl_coupling_map() is None


class TestExecutionConfig:
	@pytest.mark.parametrize("incoming", [None, "best"])
	def test_default_or_best_gradient_method_becomes_param_shift(self, incoming):
		dev = make_device()
		config = ExecutionConfig() if incoming is None else ExecutionConfig(gradient_method=incoming)
		# setup_execution_config replaces None/"best" with parameter-shift and
		# pins both gradient flags to False (PennyLane handles the shifts).
		out = dev.setup_execution_config(config)
		assert out.gradient_method == "parameter-shift"
		assert out.use_device_gradient is False
		assert out.grad_on_execution is False

	def test_explicit_gradient_method_preserved(self):
		out = make_device().setup_execution_config(ExecutionConfig(gradient_method="finite-diff"))
		assert out.gradient_method == "finite-diff"

	def test_supports_derivatives_false(self):
		assert make_device().supports_derivatives() is False


class TestPreprocessTransforms:
	def test_pipeline_without_connectivity(self):
		# Without a coupling map, _transpile is not added to the pipeline.
		dev = make_device(use_connectivity=False)
		names = [t.tape_transform.__name__ for t in dev.preprocess_transforms()]
		assert "validate_device_wires" in names
		assert "validate_measurements" in names
		assert "decompose" in names
		assert "_transpile" not in names

	def test_optimized_transpile_is_enabled_by_default(self, mock_client):
		dev = make_device(wires=3, use_connectivity=True)
		dev._client = mock_client
		names = [t.tape_transform.__name__ for t in dev.preprocess_transforms()]
		assert "_optimized_transpile" in names

	def test_routing_swaps_are_decomposed_before_execution(self):
		dev = make_device(wires=3)
		tape = QuantumScript([qml.CNOT(wires=[0, 2])], [qml.sample(wires=[0, 2])], shots=100)

		with patch.object(dev, "_pl_coupling_map", return_value=[(0, 1), (1, 2)]):
			[tape], _ = dev.preprocess_transforms()((tape,))

		assert [op.name for op in tape.operations] == ["CNOT", "CNOT", "CNOT", "CNOT"]

	def test_broadcasted_parameters_are_expanded_before_execution(self):
		dev = make_device(wires=3)
		tape = QuantumScript(
			[qml.RX(np.array([0.1, 0.2]), wires=0), qml.CNOT(wires=[0, 2])], [qml.expval(qml.PauliZ(0))], shots=100
		)

		with patch.object(dev, "_pl_coupling_map", return_value=[(0, 1), (1, 2)]):
			tapes, _ = dev.preprocess_transforms()((tape,))

		assert len(tapes) == 2
		assert all(tape.batch_size is None for tape in tapes)
		assert all(op.name != "SWAP" for tape in tapes for op in tape.operations)


class TestLayoutOptimization:
	def test_topology_layout_avoids_unnecessary_swap_without_metrics(self, crystal_4q_dqa):
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = crystal_4q_dqa
		fixed_device = make_device(wires=3, shots=None, use_connectivity=True, optimize_layout=False)
		fixed_device._client = mock_client
		topology_device = make_device(wires=3, shots=None, use_connectivity=True)
		topology_device._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 2])], [qml.sample(wires=[0, 1, 2])], shots=100)

		(fixed_circuit,) = fixed_device.to_iqm_circuits(tape)
		(topology_circuit,) = topology_device.to_iqm_circuits(tape)

		fixed_cz = [instruction for instruction in fixed_circuit.instructions if instruction.name == "cz"]
		topology_cz = [instruction for instruction in topology_circuit.instructions if instruction.name == "cz"]
		assert len(fixed_cz) == 4
		assert len(topology_cz) == 1
		mock_client.get_calibration_quality_metrics.assert_not_called()

	def test_topology_and_quality_native_layouts_emit_the_same_gate_counts(self, crystal_4q_dqa):
		fidelities = {("cz", ("QB1", "QB2")): 0.5, ("cz", ("QB2", "QB3")): 0.98, ("cz", ("QB3", "QB4")): 0.99}
		metrics = MagicMock()
		metrics.get_gate_fidelity.side_effect = lambda gate, _implementation, locus: fidelities.get((gate, locus), 0.99)
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = crystal_4q_dqa
		mock_client.get_calibration_quality_metrics.return_value = metrics
		topology_device = make_device(wires=3, shots=None, use_connectivity=True, optimize_layout=True)
		topology_device._client = mock_client
		quality_device = make_device(wires=3, shots=None, use_connectivity=True, use_metrics=True)
		quality_device._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1]), qml.CNOT(wires=[1, 2])], [qml.sample(wires=[0, 1, 2])], shots=100)

		(topology_circuit,) = topology_device.to_iqm_circuits(tape)
		(quality_circuit,) = quality_device.to_iqm_circuits(tape)

		topology_instructions = [instruction.name for instruction in topology_circuit.instructions]
		quality_instructions = [instruction.name for instruction in quality_circuit.instructions]
		assert topology_instructions == quality_instructions
		assert {
			instruction.locus[0] for instruction in topology_circuit.instructions if instruction.name == "measure"
		} == {"QB1", "QB2", "QB3"}
		assert {
			instruction.locus[0] for instruction in quality_circuit.instructions if instruction.name == "measure"
		} == {"QB2", "QB3", "QB4"}

	def test_metrics_imply_layout_optimization(self, mock_client):
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics()
		dev = make_device(wires=2, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=100)

		(circuit,) = dev.to_iqm_circuits(tape)

		assert len([instruction for instruction in circuit.instructions if instruction.name == "cz"]) == 1

	def test_topology_submission_pins_dqa_calibration_set(self, mock_client, crystal_3q_dqa):
		dev = make_device(wires=2, shots=4, use_connectivity=True, optimize_layout=True)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=4)
		program, _ = dev.preprocess()
		(preprocessed_tape,), _ = program(tape)
		physical_qubits = list(preprocessed_tape.measurements[0].wires)
		mock_client.submit_circuits.return_value = make_completed_job(n_shots=4, qubits=physical_qubits)

		dev.execute((preprocessed_tape,))

		_, kwargs = mock_client.submit_circuits.call_args
		assert kwargs["calibration_set_id"] == crystal_3q_dqa.calibration_set_id
		assert kwargs["qubit_mapping"] is None
		mock_client.get_calibration_quality_metrics.assert_not_called()


class TestQualityAwareLayout:
	def test_selects_higher_fidelity_cz_locus(self, mock_client, crystal_3q_dqa):
		metrics = make_quality_metrics()
		mock_client.get_calibration_quality_metrics.return_value = metrics
		dev = make_device(wires=2, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=100)

		(circuit,) = dev.to_iqm_circuits(tape)

		used_qubits = {
			component
			for instruction in circuit.instructions
			for component in instruction.locus
			if component.startswith("QB")
		}
		assert used_qubits == {"QB1", "QB3"}
		mock_client.get_calibration_quality_metrics.assert_called_once_with(crystal_3q_dqa.calibration_set_id)

	def test_missing_fidelity_excludes_locus(self, mock_client):
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics(cz_qb2=None, cz_qb3=0.1)
		dev = make_device(wires=2, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=100)

		(circuit,) = dev.to_iqm_circuits(tape)

		assert all("QB2" not in instruction.locus for instruction in circuit.instructions)

	def test_scores_prx_and_measure_fidelities(self, mock_client):
		fidelities = {
			("prx", ("QB1",)): 0.99,
			("measure", ("QB1",)): 0.5,
			("prx", ("QB2",)): 0.9,
			("measure", ("QB2",)): 0.99,
			("prx", ("QB3",)): 0.7,
			("measure", ("QB3",)): 0.99,
		}
		metrics = MagicMock()
		metrics.get_gate_fidelity.side_effect = lambda gate, _implementation, locus: fidelities.get((gate, locus), 0.99)
		mock_client.get_calibration_quality_metrics.return_value = metrics
		dev = make_device(wires=1, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.RX(0.2, wires=0)], [qml.sample(wires=[0])], shots=100)

		(circuit,) = dev.to_iqm_circuits(tape)

		assert {instruction.locus for instruction in circuit.instructions} == {("QB2",)}

	def test_candidate_feasibility_uses_emitted_native_gates(self, mock_client):
		fidelities = {
			("prx", ("QB1",)): None,
			("measure", ("QB1",)): 0.99,
			("prx", ("QB2",)): 0.99,
			("measure", ("QB2",)): 0.8,
			("prx", ("QB3",)): 0.99,
			("measure", ("QB3",)): 0.7,
		}
		metrics = MagicMock()
		metrics.get_gate_fidelity.side_effect = lambda gate, _implementation, locus: fidelities.get((gate, locus), 0.99)
		mock_client.get_calibration_quality_metrics.return_value = metrics
		dev = make_device(wires=1, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.RZ(0.2, wires=0)], [qml.sample(wires=[0])], shots=100)

		(circuit,) = dev.to_iqm_circuits(tape)

		assert [(instruction.name, instruction.locus) for instruction in circuit.instructions] == [
			("measure", ("QB1",))
		]

	def test_finds_minimum_cost_native_embedding(self, crystal_4q_dqa):
		fidelities = {("cz", ("QB1", "QB2")): 0.5, ("cz", ("QB2", "QB3")): 0.98, ("cz", ("QB3", "QB4")): 0.99}
		metrics = MagicMock()
		metrics.get_gate_fidelity.side_effect = lambda gate, _implementation, locus: fidelities.get((gate, locus), 0.99)
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = crystal_4q_dqa
		mock_client.get_calibration_quality_metrics.return_value = metrics
		dev = make_device(wires=3, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1]), qml.CNOT(wires=[1, 2])], [qml.sample(wires=[0, 1, 2])], shots=100)

		(circuit,) = dev.to_iqm_circuits(tape)

		measurement_loci = [
			instruction.locus[0] for instruction in circuit.instructions if instruction.name == "measure"
		]
		assert measurement_loci == ["QB2", "QB3", "QB4"]

	def test_warns_when_native_search_is_bounded(self, crystal_4q_dqa):
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = crystal_4q_dqa
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics()
		dev = make_device(wires=2, shots=None, use_connectivity=True, use_metrics=True, max_candidates=5)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=100)

		with pytest.warns(UserWarning, match="best candidate found, not a proven global optimum"):
			dev.to_iqm_circuits(tape)

	def test_warns_when_routed_search_is_bounded(self, mock_client):
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics()
		dev = make_device(wires=3, shots=None, use_connectivity=True, use_metrics=True, max_candidates=1)
		dev._client = mock_client
		tape = QuantumScript(
			[qml.CNOT(wires=[0, 1]), qml.CNOT(wires=[1, 2]), qml.CNOT(wires=[0, 2])],
			[qml.sample(wires=[0, 1, 2])],
			shots=100,
		)

		with pytest.warns(UserWarning, match="best candidate found, not a proven global optimum"):
			dev.to_iqm_circuits(tape)

	def test_equal_cost_layouts_follow_dqa_order(self, crystal_4q_dqa):
		metrics = MagicMock()
		metrics.get_gate_fidelity.return_value = 0.99
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = crystal_4q_dqa
		mock_client.get_calibration_quality_metrics.return_value = metrics
		dev = make_device(wires=3, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1]), qml.CNOT(wires=[1, 2])], [qml.sample(wires=[0, 1, 2])], shots=100)

		(circuit,) = dev.to_iqm_circuits(tape)

		measurement_loci = [
			instruction.locus[0] for instruction in circuit.instructions if instruction.name == "measure"
		]
		assert measurement_loci == ["QB1", "QB2", "QB3"]

	def test_broadcasts_are_expanded_before_layout_scoring(self, mock_client):
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics()
		dev = make_device(wires=2, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript(
			[qml.RX(np.array([0.1, 0.2]), wires=0), qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=100
		)

		circuits = dev.to_iqm_circuits(tape)

		assert len(circuits) == 2

	def test_routes_and_scores_when_no_native_embedding_exists(self, mock_client, crystal_3q_dqa):
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics()
		dev = make_device(wires=3, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript(
			[qml.CNOT(wires=[0, 1]), qml.CNOT(wires=[1, 2]), qml.CNOT(wires=[0, 2])],
			[qml.sample(wires=[0, 1, 2])],
			shots=100,
		)

		(circuit,) = dev.to_iqm_circuits(tape)

		valid_cz_loci = {frozenset(locus) for locus in crystal_3q_dqa.gates["cz"].loci}
		cz_loci = [instruction.locus for instruction in circuit.instructions if instruction.name == "cz"]
		assert len(cz_loci) > 3
		assert all(frozenset(locus) in valid_cz_loci for locus in cz_loci)

	def test_routed_layout_avoids_low_fidelity_coupler(self, crystal_4q_dqa):
		fidelities = {("cz", ("QB1", "QB2")): 0.2, ("cz", ("QB2", "QB3")): 0.99, ("cz", ("QB3", "QB4")): 0.99}
		metrics = MagicMock()
		metrics.get_gate_fidelity.side_effect = lambda gate, _implementation, locus: fidelities.get((gate, locus), 0.99)
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = crystal_4q_dqa
		mock_client.get_calibration_quality_metrics.return_value = metrics
		dev = make_device(wires=3, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript(
			[qml.CNOT(wires=[0, 1]), qml.CNOT(wires=[1, 2]), qml.CNOT(wires=[0, 2])],
			[qml.sample(wires=[0, 1, 2])],
			shots=100,
		)

		(circuit,) = dev.to_iqm_circuits(tape)

		cz_loci = {frozenset(instruction.locus) for instruction in circuit.instructions if instruction.name == "cz"}
		assert cz_loci == {frozenset(("QB2", "QB3")), frozenset(("QB3", "QB4"))}

	def test_routed_layout_preserves_output_probabilities(self, mock_client):
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics()
		dev = make_device(wires=3, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript(
			[
				qml.RY(0.3, wires=0),
				qml.RX(-0.2, wires=1),
				qml.CNOT(wires=[0, 1]),
				qml.CNOT(wires=[1, 2]),
				qml.CNOT(wires=[0, 2]),
			],
			[qml.sample(wires=[0, 1, 2])],
			shots=100,
		)
		program, _ = dev.preprocess()

		(routed_tape,), _ = program(tape)

		original_matrix = qml.matrix(QuantumScript(tape.operations), wire_order=tape.wires)
		routed_wires = routed_tape.measurements[0].wires
		routed_matrix = qml.matrix(QuantumScript(routed_tape.operations), wire_order=routed_wires)
		assert np.abs(routed_matrix[:, 0]) ** 2 == pytest.approx(np.abs(original_matrix[:, 0]) ** 2)

	def test_submission_pins_metrics_calibration_set(self, mock_client, crystal_3q_dqa):
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics()
		dev = make_device(wires=2, shots=4, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=4)
		program, _ = dev.preprocess()
		(preprocessed_tape,), _ = program(tape)
		physical_qubits = list(preprocessed_tape.measurements[0].wires)
		mock_client.submit_circuits.return_value = make_completed_job(n_shots=4, qubits=physical_qubits)

		dev.execute((preprocessed_tape,))

		_, kwargs = mock_client.submit_circuits.call_args
		assert kwargs["calibration_set_id"] == crystal_3q_dqa.calibration_set_id
		assert kwargs["qubit_mapping"] is None

	@pytest.mark.parametrize("unsupported_capability", ["move", "computational_resonators"])
	def test_quality_aware_layout_rejects_star_capabilities(self, star_dqa, unsupported_capability):
		if unsupported_capability == "move":
			dqa = star_dqa.model_copy(update={"computational_resonators": []})
		else:
			gates = {name: gate for name, gate in star_dqa.gates.items() if name != "move"}
			dqa = star_dqa.model_copy(update={"gates": gates})
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = dqa
		mock_client.get_calibration_quality_metrics.return_value = make_quality_metrics()
		dev = make_device(wires=2, shots=None, use_connectivity=True, use_metrics=True)
		dev._client = mock_client
		tape = QuantumScript([qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=100)

		with pytest.raises(NotImplementedError, match="does not support MOVE gates or computational resonators"):
			dev.to_iqm_circuits(tape)


class TestToIQMCircuits:
	def test_preprocesses_and_translates_without_execution(self):
		dev = make_device(wires=2, shots=None)
		tape = QuantumScript(
			[qml.Rot(0.1, 0.2, 0.3, wires=0), qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])], shots=100
		)

		(circuit,) = dev.to_iqm_circuits(tape, circuit_name="bell")

		assert circuit.name == "bell"
		assert {instruction.name for instruction in circuit.instructions} <= {"prx", "cz", "measure"}
		assert [instruction.name for instruction in circuit.instructions[-2:]] == ["measure", "measure"]

	def test_names_circuits_created_by_broadcast_expansion(self):
		dev = make_device(wires=1)
		tape = QuantumScript([qml.RX(np.array([0.1, 0.2]), wires=0)], [qml.sample(wires=[0])], shots=100)

		circuits = dev.to_iqm_circuits(tape, circuit_name="sweep")

		assert [circuit.name for circuit in circuits] == ["sweep_0", "sweep_1"]


class TestTranspileWorkaround:
	"""_transpile must handle tensor-product observables that qml.transforms.transpile rejects."""

	COUPLING = [(0, 1), (1, 2)]

	def test_simple_measurement_delegates_to_pl_transpile(self):
		from pennylane_iqm.device import _transpile

		tape = QuantumScript([qml.CNOT(wires=[0, 1])], [qml.sample(wires=[0, 1])])
		[routed], _ = _transpile(tape, coupling_map=self.COUPLING)
		assert routed is not None

	def test_tensor_product_expval_does_not_raise(self):
		from pennylane_iqm.device import _transpile

		tape = QuantumScript(
			[qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])], [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))]
		)
		[routed], _ = _transpile(tape, coupling_map=self.COUPLING)
		assert any(isinstance(m.obs, qml.ops.Prod) for m in routed.measurements)

	def test_hamiltonian_expval_does_not_raise(self):
		from pennylane_iqm.device import _transpile

		H = -0.5 * qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliX(0) @ qml.PauliX(1)
		tape = QuantumScript([qml.RY(0.5, wires=0), qml.CZ(wires=[0, 1])], [qml.expval(H)])
		[routed], _ = _transpile(tape, coupling_map=self.COUPLING)
		assert routed is not None

	def test_mixed_measurements_routed_correctly(self):
		from pennylane_iqm.device import _transpile

		tape = QuantumScript(
			[qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])],
			[qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.probs(wires=[0, 1])],
		)
		[routed], _ = _transpile(tape, coupling_map=self.COUPLING)
		assert len(routed.measurements) == 2


class TestCollectMeasuredWires:
	def test_explicit_wires(self):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0, 1])])
		assert make_device(wires=2)._collect_measured_wires(tape) == [0, 1]

	def test_deduplicated(self):
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0]), qml.probs(wires=[0])])
		assert make_device(wires=2)._collect_measured_wires(tape) == [0]

	def test_empty_wires_falls_back_to_tape_wires(self):
		# qml.counts() with no wires arg means "all tape wires".
		tape = QuantumScript([qml.PauliX(wires=0), qml.PauliX(wires=1)], [qml.counts()])
		assert set(make_device(wires=2)._collect_measured_wires(tape)) == {0, 1}


class TestWaitForJob:
	def test_completes_immediately(self):
		from iqm.iqm_server_client.models import JobStatus

		dev = make_device(poll_interval=0)
		measurements = {"meas_QB1": [[0]]}
		job = make_mock_job([JobStatus.COMPLETED], measurements)
		assert dev._wait_for_job(job) == [measurements]

	def test_waits_for_processing_then_completes(self):
		from iqm.iqm_server_client.models import JobStatus

		dev = make_device(poll_interval=0)
		measurements = {"meas_QB1": [[0]]}
		job = make_mock_job([JobStatus.WAITING, JobStatus.PROCESSING, JobStatus.COMPLETED], measurements)
		with patch("pennylane_iqm.device.time.sleep"):
			assert dev._wait_for_job(job) == [measurements]

	@pytest.mark.parametrize("terminal", ["FAILED", "CANCELLED"])
	def test_terminal_failure_raises_runtime_error(self, terminal):
		from iqm.iqm_server_client.models import JobStatus

		dev = make_device(poll_interval=0)
		job = make_mock_job([JobStatus[terminal]])
		job.data = MagicMock()
		job.data.errors = []
		with pytest.raises(RuntimeError, match=terminal.lower()):
			dev._wait_for_job(job)

	def test_timeout_raises_timeout_error(self):
		from iqm.iqm_server_client.models import JobStatus

		dev = make_device(poll_interval=0, timeout=0.0)
		job = make_mock_job([JobStatus.PROCESSING])
		with pytest.raises(TimeoutError):
			dev._wait_for_job(job)


class TestPostprocess:
	def test_single_expval(self):
		dev = make_device(wires=1, shots=1000)
		samples = np.zeros((1000, 1), dtype=int)
		tape = QuantumScript([], [qml.expval(qml.PauliZ(0))], shots=1000)
		assert dev._postprocess(tape, samples, [0]) == pytest.approx(1.0, abs=0.01)

	def test_single_probs(self):
		dev = make_device(wires=1, shots=100)
		samples = np.zeros((100, 1), dtype=int)
		tape = QuantumScript([], [qml.probs(wires=[0])], shots=100)
		np.testing.assert_allclose(dev._postprocess(tape, samples, [0]), [1.0, 0.0], atol=1e-9)

	def test_multiple_measurements_returns_tuple(self):
		dev = make_device(wires=2, shots=10)
		samples = np.zeros((10, 2), dtype=int)
		tape = QuantumScript([], [qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])], shots=10)
		result = dev._postprocess(tape, samples, [0, 1])
		assert isinstance(result, tuple)
		assert len(result) == 2

	def test_non_sample_measurement_raises(self):
		# After diagonalize_measurements every measurement is a SampleMeasurement;
		# anything else (e.g. qml.state()) indicates a pipeline bug, so fail loudly.
		dev = make_device(wires=1, shots=10)
		samples = np.zeros((10, 1), dtype=int)
		tape = QuantumScript([], [qml.state()], shots=10)
		with pytest.raises(TypeError, match="SampleMeasurement"):
			dev._postprocess(tape, samples, [0])

	def test_process_samples_none_raises(self):
		# Guard against a SampleMeasurement subclass that returns None.
		dev = make_device(wires=1, shots=10)
		samples = np.zeros((10, 1), dtype=int)
		bad_mp = qml.expval(qml.PauliZ(0))
		with patch.object(type(bad_mp), "process_samples", return_value=None):
			tape = QuantumScript([], [bad_mp], shots=10)
			with pytest.raises(RuntimeError, match="returned None"):
				dev._postprocess(tape, samples, [0])


class TestShotsResolution:
	def test_no_shots_anywhere_raises(self):
		dev = make_device(wires=1, shots=None)
		dev._client = MagicMock()
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=None)
		with pytest.raises(ValueError, match="shots must be specified"):
			dev.execute((tape,))

	@pytest.mark.parametrize("device_shots,tape_shots,expected", [(100, 32, 32), (64, None, 64)])
	def test_resolution(self, device_shots, tape_shots, expected):
		mock_client = MagicMock()
		mock_client.submit_circuits.return_value = make_completed_job(n_shots=expected, qubits=["QB1"])

		dev = make_device(wires=1, shots=device_shots)
		dev._client = mock_client
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=tape_shots)
		dev.execute((tape,))

		_, kwargs = mock_client.submit_circuits.call_args
		assert kwargs.get("shots") == expected


class TestExecute:
	@staticmethod
	def _device_with_completed_job(mock_client, n_shots, qubits):
		mock_client.submit_circuits.return_value = make_completed_job(n_shots=n_shots, qubits=qubits)
		dev = make_device(wires=len(qubits), shots=n_shots, use_connectivity=False)
		dev._client = mock_client
		return dev

	def test_single_tape_executes_and_submits_once(self):
		mock_client = MagicMock()
		dev = self._device_with_completed_job(mock_client, n_shots=10, qubits=["QB1", "QB2"])
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0, 1])], shots=10)
		(result,) = dev.execute((tape,))
		assert result.shape == (10, 2)
		mock_client.submit_circuits.assert_called_once()

	def test_batch_execute_returns_one_result_per_tape(self):
		mock_client = MagicMock()
		measurements = [{"meas_QB1": [[0], [0], [0], [0]]}, {"meas_QB1": [[1], [1], [1], [1]]}]
		job = make_mock_job([JobStatus.COMPLETED])
		job.result.return_value = measurements
		mock_client.submit_circuits.return_value = job
		dev = make_device(wires=1, shots=4, use_connectivity=False)
		dev._client = mock_client
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=4)
		first, second = dev.execute((tape, tape))

		np.testing.assert_array_equal(first, np.zeros((4, 1), dtype=int))
		np.testing.assert_array_equal(second, np.ones((4, 1), dtype=int))
		mock_client.submit_circuits.assert_called_once()
		submitted_circuits = mock_client.submit_circuits.call_args.args[0]
		assert len(submitted_circuits) == 2

	def test_different_shot_counts_are_submitted_separately(self):
		mock_client = MagicMock()
		mock_client.submit_circuits.side_effect = [
			make_completed_job(n_shots=4, qubits=["QB1"]),
			make_completed_job(n_shots=8, qubits=["QB1"]),
		]
		dev = make_device(wires=1, shots=None, use_connectivity=False)
		dev._client = mock_client
		tapes = (
			QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=4),
			QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=8),
		)

		results = dev.execute(tapes)

		assert [result.shape for result in results] == [(4, 1), (8, 1)]
		assert [call.kwargs["shots"] for call in mock_client.submit_circuits.call_args_list] == [4, 8]

	def test_star_architecture_invokes_transpile_insert_moves(self, star_dqa):
		# Star DQA -> Device should call iqm-client's transpile_insert_moves
		# to insert MOVE gates around qubit-resonator CZ loci.
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = star_dqa
		mock_client.submit_circuits.return_value = make_completed_job(n_shots=4, qubits=["QB1"])

		dev = make_device(wires=3, shots=4, use_connectivity=True, optimize_layout=False)
		dev._client = mock_client

		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=4)
		with patch("pennylane_iqm.device.transpile_insert_moves") as mock_insert:
			mock_insert.side_effect = lambda circuit, _arch: circuit
			dev.execute((tape,))
			mock_insert.assert_called_once()
