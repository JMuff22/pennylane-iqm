"""Tests for IQMDevice."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pennylane as qml
import pytest
from pennylane.devices import ExecutionConfig
from pennylane.tape import QuantumScript

from pennylane_iqm.device import IQMDevice

from .conftest import make_completed_job, make_mock_job


def make_device(
	wires: int | list = 3,
	shots: int | None = 100,
	use_connectivity: bool = False,
	server_url: str = "https://test.iqm.fi",
	**kwargs,
) -> IQMDevice:
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
			IQMDevice("https://x", wires=None, use_connectivity=False)

	def test_wires_auto_detected_and_cached_from_dqa(self, crystal_3q_dqa):
		with patch("pennylane_iqm.device.IQMClient") as MockClient:
			mock_client = MagicMock()
			mock_client.get_dynamic_quantum_architecture.return_value = crystal_3q_dqa
			MockClient.return_value = mock_client
			dev = IQMDevice("https://x", wires=None, use_connectivity=True)
		assert len(dev.wires) == 3
		assert dev._dqa is crystal_3q_dqa

	def test_wires_auto_detect_failure_raises_value_error(self):
		with patch("pennylane_iqm.device.IQMClient") as MockClient:
			mock_client = MagicMock()
			mock_client.get_dynamic_quantum_architecture.side_effect = RuntimeError("unreachable")
			MockClient.return_value = mock_client
			with pytest.raises(ValueError, match="Could not auto-detect"):
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

	def test_transpile_added_when_coupling_map_available(self, mock_client):
		dev = make_device(wires=3, use_connectivity=True)
		dev._client = mock_client
		names = [t.tape_transform.__name__ for t in dev.preprocess_transforms()]
		assert "_transpile" in names


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
			dev._execute_single(tape)

	@pytest.mark.parametrize("device_shots,tape_shots,expected", [(100, 32, 32), (64, None, 64)])
	def test_resolution(self, device_shots, tape_shots, expected):
		mock_client = MagicMock()
		mock_client.submit_circuits.return_value = make_completed_job(n_shots=expected, qubits=["QB1"])

		dev = make_device(wires=1, shots=device_shots)
		dev._client = mock_client
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=tape_shots)
		dev._execute_single(tape)

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
		dev = self._device_with_completed_job(mock_client, n_shots=4, qubits=["QB1"])
		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=4)
		assert len(dev.execute((tape, tape))) == 2

	def test_star_architecture_invokes_transpile_insert_moves(self, star_dqa):
		# Star DQA -> Device should call iqm-client's transpile_insert_moves
		# to insert MOVE gates around qubit-resonator CZ loci.
		mock_client = MagicMock()
		mock_client.get_dynamic_quantum_architecture.return_value = star_dqa
		mock_client.submit_circuits.return_value = make_completed_job(n_shots=4, qubits=["QB1"])

		dev = make_device(wires=3, shots=4, use_connectivity=True)
		dev._client = mock_client

		tape = QuantumScript([qml.PauliX(wires=0)], [qml.sample(wires=[0])], shots=4)
		with patch("pennylane_iqm.device.transpile_insert_moves") as mock_insert:
			mock_insert.side_effect = lambda circuit, _arch: circuit
			dev.execute((tape,))
			mock_insert.assert_called_once()
