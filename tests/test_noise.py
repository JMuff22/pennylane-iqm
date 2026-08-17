"""Tests for building PennyLane noise models from IQM calibration data."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from iqm.station_control.client.qon import ObservationFinder
from iqm.station_control.interface.models.observation import ObservationBase

from pennylane_iqm.noise import IQMCalibration, mock_device

T1 = {"QB1": 40e-6, "QB2": 45e-6, "QB3": 50e-6}
T2 = {"QB1": 30e-6, "QB2": 35e-6, "QB3": 40e-6}
READOUT = {"QB1": (0.02, 0.03), "QB2": (0.04, 0.05), "QB3": (0.06, 0.07)}
PRX_FIDELITY = 0.999
PRX_DURATION = 40e-9
CZ_FIDELITY = 0.98
CZ_DURATION = 100e-9


def _observation(dut_field: str, value: float, unit: str = "") -> ObservationBase:
	return ObservationBase(dut_field=dut_field, value=value, unit=unit)


def _observations(qubits: list[str], cz_loci: list[tuple[str, str]]) -> list[ObservationBase]:
	observations = []
	for qubit in qubits:
		observations.append(_observation(f"characterization.model.{qubit}.t1_time", T1[qubit], "s"))
		observations.append(_observation(f"characterization.model.{qubit}.t2_time", T2[qubit], "s"))
		observations.append(_observation(f"metrics.ssro.measure.constant.{qubit}.error_0_to_1", READOUT[qubit][0]))
		observations.append(_observation(f"metrics.ssro.measure.constant.{qubit}.error_1_to_0", READOUT[qubit][1]))
		observations.append(_observation(f"metrics.rb.prx.drag_gaussian.{qubit}.fidelity", PRX_FIDELITY))
		observations.append(_observation(f"gates.prx.drag_gaussian.{qubit}.duration", PRX_DURATION, "s"))
	for control, target in cz_loci:
		locus = f"{control}__{target}"
		observations.append(_observation(f"metrics.irb.cz.tgss.{locus}.fidelity", CZ_FIDELITY))
		observations.append(_observation(f"gates.cz.tgss.{locus}.duration", CZ_DURATION, "s"))
	return observations


@pytest.fixture
def metrics() -> ObservationFinder:
	"""Quality metrics covering every locus of the 3-qubit crystal architecture."""
	return ObservationFinder(_observations(["QB1", "QB2", "QB3"], [("QB1", "QB2"), ("QB1", "QB3")]))


@pytest.fixture
def calibration(crystal_3q_dqa, metrics) -> IQMCalibration:
	return IQMCalibration.from_metrics(crystal_3q_dqa, metrics)


def test_from_metrics_reads_every_calibrated_locus(calibration):
	assert calibration.qubits == ["QB1", "QB2", "QB3"]
	assert calibration.t1["QB2"] == 45e-6
	assert calibration.t2["QB2"] == 35e-6
	assert calibration.readout_errors["QB1"] == (0.02, 0.03)
	assert calibration.prx_error["QB1"] == pytest.approx(1.0 - PRX_FIDELITY)
	assert calibration.prx_duration["QB1"] == PRX_DURATION
	assert calibration.cz_error[("QB1", "QB3")] == pytest.approx(1.0 - CZ_FIDELITY)
	assert calibration.cz_duration[("QB1", "QB3")] == CZ_DURATION


def test_from_metrics_skips_loci_without_metrics(crystal_3q_dqa):
	observations = [
		obs for obs in _observations(["QB1", "QB2", "QB3"], [("QB1", "QB2")]) if "QB3.fidelity" not in obs.dut_field
	]
	calibration = IQMCalibration.from_metrics(crystal_3q_dqa, ObservationFinder(observations))

	assert "QB3" not in calibration.prx_error
	assert ("QB1", "QB3") not in calibration.cz_error
	assert "QB3" in calibration.t1


@pytest.mark.parametrize("qubit", ["QB1", "QB2", "QB3"])
def test_readout_error_reaches_measured_probabilities(calibration, qubit):
	dev = mock_device(calibration, wires=[qubit])

	@qml.qnode(dev)
	def idle():
		return qml.probs(wires=qubit)

	error_0_to_1 = READOUT[qubit][0]
	assert idle() == pytest.approx([1.0 - error_0_to_1, error_0_to_1])


def test_readout_error_reaches_multi_wire_measurements(calibration):
	"""A per-qubit measurement condition would only fire on single-wire measurements."""
	dev = mock_device(calibration, wires=["QB1", "QB2"])

	@qml.qnode(dev)
	def idle():
		return qml.probs(wires=["QB1", "QB2"])

	no_flip = (1.0 - READOUT["QB1"][0]) * (1.0 - READOUT["QB2"][0])
	assert idle()[0] == pytest.approx(no_flip)


def test_cz_noise_is_applied_in_either_wire_order(calibration):
	dev = mock_device(calibration, wires=["QB1", "QB2"])

	def bell(control, target):
		@qml.qnode(dev)
		def circuit():
			qml.Hadamard("QB1")
			qml.Hadamard("QB2")
			qml.CZ([control, target])
			qml.Hadamard("QB2")
			return qml.probs(wires=["QB1", "QB2"])

		return circuit()

	forward = bell("QB1", "QB2")
	assert forward == pytest.approx(bell("QB2", "QB1"))
	# A Bell state read out through a noisy channel keeps most weight on |00> and |11>.
	assert forward[0] + forward[3] == pytest.approx(0.90776, abs=1e-5)


def test_mock_device_rejects_uncalibrated_wires(calibration):
	with pytest.raises(ValueError, match=r"No calibration data for qubits \['QB9'\]"):
		mock_device(calibration, wires=["QB1", "QB9"])


def test_uniform_covers_every_given_locus():
	calibration = IQMCalibration.uniform(["QB1", "QB2"], [("QB1", "QB2")], t1=1e-5, cz_fidelity=0.95)

	assert calibration.t1 == {"QB1": 1e-5, "QB2": 1e-5}
	assert calibration.cz_error == {("QB1", "QB2"): pytest.approx(0.05)}


def test_readout_kraus_operators_are_trace_preserving():
	from pennylane_iqm.noise import _readout_kraus

	kraus = _readout_kraus(0.02, 0.03)
	assert sum(k.conj().T @ k for k in kraus) == pytest.approx(np.eye(2))
