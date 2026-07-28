"""Shared test fixtures for pennylane-iqm tests."""

from __future__ import annotations

import uuid
from math import pi
from unittest.mock import MagicMock

import pytest
from iqm.pulse.builder import CircuitOperation
from iqm.pulse.circuit_operations import Circuit
from iqm.station_control.interface.models import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo

SAMPLE_CALSET_ID = uuid.UUID("9ddb9586-8f27-49a9-90ed-41086b47f6bd")


@pytest.fixture
def crystal_3q_dqa() -> DynamicQuantumArchitecture:
	"""3-qubit crystal architecture with CZ edges QB1-QB2 and QB1-QB3."""
	return DynamicQuantumArchitecture(
		calibration_set_id=SAMPLE_CALSET_ID,
		qubits=["QB1", "QB2", "QB3"],
		computational_resonators=[],
		gates={
			"prx": GateInfo(
				implementations={"drag_gaussian": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",)))},
				default_implementation="drag_gaussian",
				override_default_implementation={},
			),
			"cz": GateInfo(
				implementations={"tgss": GateImplementationInfo(loci=(("QB1", "QB2"), ("QB1", "QB3")))},
				default_implementation="tgss",
				override_default_implementation={},
			),
			"measure": GateInfo(
				implementations={"constant": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",)))},
				default_implementation="constant",
				override_default_implementation={},
			),
		},
	)


@pytest.fixture
def star_dqa() -> DynamicQuantumArchitecture:
	"""Star architecture with computational resonators and MOVE gates."""
	return DynamicQuantumArchitecture(
		calibration_set_id=SAMPLE_CALSET_ID,
		qubits=["QB1", "QB2", "QB3"],
		computational_resonators=["CR1"],
		gates={
			"prx": GateInfo(
				implementations={"drag_gaussian": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",)))},
				default_implementation="drag_gaussian",
				override_default_implementation={},
			),
			"cz": GateInfo(
				implementations={"tgss": GateImplementationInfo(loci=(("QB1", "CR1"), ("QB2", "CR1")))},
				default_implementation="tgss",
				override_default_implementation={},
			),
			"move": GateInfo(
				implementations={"tgss_crf": GateImplementationInfo(loci=(("QB3", "CR1"),))},
				default_implementation="tgss_crf",
				override_default_implementation={},
			),
			"measure": GateInfo(
				implementations={"constant": GateImplementationInfo(loci=(("QB1",), ("QB2",), ("QB3",)))},
				default_implementation="constant",
				override_default_implementation={},
			),
		},
	)


@pytest.fixture
def crystal_4q_dqa() -> DynamicQuantumArchitecture:
	"""4-qubit crystal chain QB1-QB2-QB3-QB4."""
	qubits = ["QB1", "QB2", "QB3", "QB4"]
	return DynamicQuantumArchitecture(
		calibration_set_id=SAMPLE_CALSET_ID,
		qubits=qubits,
		computational_resonators=[],
		gates={
			"prx": GateInfo(
				implementations={"drag_gaussian": GateImplementationInfo(loci=tuple((qubit,) for qubit in qubits))},
				default_implementation="drag_gaussian",
				override_default_implementation={},
			),
			"cz": GateInfo(
				implementations={"tgss": GateImplementationInfo(loci=(("QB1", "QB2"), ("QB2", "QB3"), ("QB3", "QB4")))},
				default_implementation="tgss",
				override_default_implementation={},
			),
			"measure": GateInfo(
				implementations={"constant": GateImplementationInfo(loci=tuple((qubit,) for qubit in qubits))},
				default_implementation="constant",
				override_default_implementation={},
			),
		},
	)


def make_mock_job(status_sequence, measurements: dict | None = None):
	"""Return a mock job that cycles through the given status values on .update()."""

	job = MagicMock()
	job.job_id = uuid.uuid4()

	statuses = list(status_sequence)
	call_count = {"n": 0}

	def _update():
		idx = min(call_count["n"], len(statuses) - 1)
		call_count["n"] += 1
		return statuses[idx]

	job.update.side_effect = _update

	if measurements is not None:
		# job.result() returns CircuitMeasurementResultsBatch = list[dict[...]]
		# one dict per submitted circuit; we always submit exactly one circuit.
		job.result.return_value = [measurements]

	return job


def make_completed_job(n_shots: int = 4, qubits: list[str] | None = None):
	"""Return a mock job that immediately returns COMPLETED with binary measurements."""
	from iqm.iqm_server_client.models import JobStatus

	if qubits is None:
		qubits = ["QB1", "QB2"]

	measurements = {f"meas_{q}": [[int(i % 2)] for i in range(n_shots)] for q in qubits}
	return make_mock_job([JobStatus.COMPLETED], measurements)


@pytest.fixture
def mock_client(crystal_3q_dqa):
	"""IQMClient mock with a 3-qubit crystal architecture."""
	client = MagicMock()
	client.get_dynamic_quantum_architecture.return_value = crystal_3q_dqa
	return client


@pytest.fixture
def simple_circuit() -> Circuit:
	"""A simple 2-qubit IQM circuit for verifying translation."""
	return Circuit(
		name="test",
		instructions=(
			CircuitOperation("prx", locus=("QB1",), args={"angle": pi / 2, "phase": 0.0}),
			CircuitOperation("cz", locus=("QB1", "QB2"), args={}),
			CircuitOperation("measure", locus=("QB1",), args={"key": "meas_QB1"}),
			CircuitOperation("measure", locus=("QB2",), args={"key": "meas_QB2"}),
		),
	)
