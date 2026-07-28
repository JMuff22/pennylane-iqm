"""Initial layout selection for IQM quantum computers.

Native zero-SWAP layouts take priority. Without calibration metrics, routed
layouts minimize the number of emitted CZ gates. With metrics, the score is the
negative logarithm of the product of emitted gate fidelities, which assumes
independent gate errors.
"""

from __future__ import annotations

import warnings
from collections import Counter, deque
from collections.abc import Hashable
from math import inf, log

import pennylane as qml
import rustworkx as rx
from iqm.station_control.client.qon import ObservationFinder
from iqm.station_control.interface.models import DynamicQuantumArchitecture
from pennylane.devices.preprocess import decompose
from pennylane.tape import QuantumScript

from .gates import stopping_condition
from .translate import tape_to_iqm_circuit

WireLayout = dict[Hashable, str]
CouplingMap = dict[str, list[str]]
EdgeCosts = dict[frozenset[str], float]
GateCosts = dict[tuple[str, tuple[str, ...]], float]
InstructionTemplate = tuple[tuple[str, tuple[Hashable, ...]], ...]
_DEFAULT_MAX_CANDIDATES = 10_000


@qml.transform
def _transpile(
	tape: QuantumScript, coupling_map: list[tuple[Hashable, Hashable]] | dict[Hashable, list[Hashable]]
) -> tuple:
	"""Like qml.transforms.transpile but support composite observables.

	PennyLane routing depends only on operation connectivity. For unsupported
	composite measurements, a probability measurement supplies the wire
	permutation that is then applied to the original measurements.
	"""
	complex_observables = (qml.ops.Prod, qml.ops.LinearCombination)
	needs_workaround = any(
		isinstance(getattr(measurement, "obs", None), complex_observables) for measurement in tape.measurements
	)

	transpile_tape = qml.transforms.transpile.tape_transform
	if transpile_tape is None:
		raise RuntimeError("qml.transforms.transpile.tape_transform is unexpectedly None")
	if not needs_workaround:
		return transpile_tape(tape, coupling_map=coupling_map)

	original_wires = list(tape.wires)
	proxy_tape = QuantumScript(tape.operations, [qml.probs(wires=original_wires)], shots=tape.shots)
	routed_batch, postprocessing = transpile_tape(proxy_tape, coupling_map=coupling_map)
	[routed] = routed_batch

	new_wires = list(routed.measurements[0].wires)
	wire_map = dict(zip(original_wires, new_wires, strict=True))
	measurements = [measurement.map_wires(wire_map) for measurement in tape.measurements]
	final_tape = QuantumScript(routed.operations, measurements, shots=tape.shots)
	return [final_tape], postprocessing


def _fidelity_cost(fidelity: float | None) -> float:
	"""Convert fidelity into an additive cost.

	A missing fidelity identifies an uncalibrated locus and therefore makes
	that locus unavailable for quality-aware layout selection.
	"""
	if fidelity is None:
		return inf
	if not 0 <= fidelity <= 1:
		raise ValueError(f"gate fidelity must be between 0 and 1, got {fidelity}.")
	return inf if fidelity == 0 else -log(fidelity)


def _gate_costs(dqa: DynamicQuantumArchitecture, metrics: ObservationFinder | None) -> GateCosts:
	"""Build topology-only costs or read each calibrated gate cost once."""
	costs: GateCosts = {}
	qubits = set(dqa.qubits)
	for gate_name in ("prx", "cz", "measure"):
		gate = dqa.gates.get(gate_name)
		if gate is None:
			continue
		for locus in gate.loci:
			if set(locus) <= qubits:
				if metrics is None:
					cost = 1.0 if gate_name == "cz" else 0.0
				else:
					implementation = gate.get_default_implementation(locus)
					fidelity = metrics.get_gate_fidelity(gate_name, implementation, locus)
					cost = _fidelity_cost(fidelity)
				costs[(gate_name, locus)] = cost
				if gate_name == "cz" and len(locus) == 2:
					costs.setdefault(("cz", tuple(reversed(locus))), cost)
	return costs


def _qubit_graph(dqa: DynamicQuantumArchitecture, gate_costs: GateCosts) -> tuple[CouplingMap, EdgeCosts]:
	"""Build the calibrated native qubit-qubit connectivity graph."""
	coupling = {qubit: [] for qubit in dqa.qubits}
	costs: EdgeCosts = {}
	qubits = set(dqa.qubits)

	if (cz_gate := dqa.gates.get("cz")) is not None:
		for locus in cz_gate.loci:
			if len(locus) == 2 and set(locus) <= qubits:
				first, second = locus
				cost = gate_costs.get(("cz", locus), inf)
				key = frozenset(locus)
				if cost >= costs.get(key, inf):
					continue
				costs[key] = cost
				if second not in coupling[first]:
					coupling[first].append(second)
				if first not in coupling[second]:
					coupling[second].append(first)

	return coupling, costs


def _routed_two_qubit_cost(first: str, second: str, coupling: CouplingMap, edge_costs: EdgeCosts) -> float:
	"""Estimate CZ and SWAP costs along PennyLane's initial shortest-hop path.

	The estimate only orders the bounded routed-layout search. Complete layouts
	are scored from the actual routed and translated circuit.
	"""
	if first == second:
		return inf

	parents: dict[str, str | None] = {first: None}
	queue = deque([first])
	while queue and second not in parents:
		current = queue.popleft()
		for neighbor in coupling[current]:
			if neighbor not in parents:
				parents[neighbor] = current
				queue.append(neighbor)

	if second not in parents:
		return inf

	path = [second]
	while (parent := parents[path[-1]]) is not None:
		path.append(parent)
	path.reverse()
	path_costs = [edge_costs[frozenset((left, right))] for left, right in zip(path, path[1:], strict=False)]
	if len(path_costs) == 1:
		return path_costs[0]
	return path_costs[0] + 3 * sum(path_costs[1:])


def _instruction_template(tape: QuantumScript) -> InstructionTemplate:
	"""Translate a tape once while retaining its logical wire labels."""
	wire_map = {wire: f"__logical_{index}" for index, wire in enumerate(tape.wires)}
	logical_wires = {name: wire for wire, name in wire_map.items()}
	circuit = tape_to_iqm_circuit(tape, wire_map, "layout_template")
	return tuple(
		(instruction.name, tuple(logical_wires[component] for component in instruction.locus))
		for instruction in circuit.instructions
	)


def _instruction_counts(
	instructions: InstructionTemplate,
) -> tuple[Counter[Hashable], Counter[Hashable], Counter[frozenset[Hashable]]]:
	"""Count emitted PRX, measurement, and CZ instructions by logical wire."""
	single_qubit: Counter[Hashable] = Counter()
	measurements: Counter[Hashable] = Counter()
	interactions: Counter[frozenset[Hashable]] = Counter()

	for gate_name, locus in instructions:
		if gate_name == "prx":
			single_qubit[locus[0]] += 1
		elif gate_name == "measure":
			measurements[locus[0]] += 1
		elif gate_name == "cz":
			interactions[frozenset(locus)] += 1

	return single_qubit, measurements, interactions


def _native_graphs(
	wires: list[Hashable],
	interactions: Counter[frozenset[Hashable]],
	dqa: DynamicQuantumArchitecture,
	coupling: CouplingMap,
) -> tuple[rx.PyGraph, rx.PyGraph]:
	"""Build physical and logical graphs for native VF2 layout search."""
	physical_graph = rx.PyGraph(multigraph=False)
	physical_graph.add_nodes_from(dqa.qubits)
	physical_index = {qubit: index for index, qubit in enumerate(dqa.qubits)}
	for first_index, first in enumerate(dqa.qubits):
		for second in coupling[first]:
			second_index = physical_index[second]
			if first_index < second_index:
				physical_graph.add_edge(first_index, second_index, None)

	logical_graph = rx.PyGraph(multigraph=False)
	logical_graph.add_nodes_from(wires)
	logical_index = {wire: index for index, wire in enumerate(wires)}
	for interaction in interactions:
		first, second = sorted(interaction, key=logical_index.__getitem__)
		logical_graph.add_edge(logical_index[first], logical_index[second], None)

	return physical_graph, logical_graph


def _map_tape(tape: QuantumScript, layout: WireLayout) -> QuantumScript:
	"""Map every logical tape wire to its selected physical IQM qubit."""
	wire_map: dict[Hashable, Hashable] = dict(layout)
	return tape.copy(
		operations=[operation.map_wires(wire_map) for operation in tape.operations],
		measurements=[measurement.map_wires(wire_map) for measurement in tape.measurements],
		trainable_params=tape.trainable_params,
	)


def _route_tape(tape: QuantumScript, coupling: CouplingMap) -> QuantumScript:
	"""Route and decompose a mapped tape exactly as the device pipeline does."""
	if not any(len(operation.wires) == 2 for operation in tape.operations):
		return tape

	transpile_tape = _transpile.tape_transform
	if transpile_tape is None:
		raise RuntimeError("_transpile.tape_transform is unexpectedly None")
	routed_batch, _ = transpile_tape(tape, coupling_map=coupling)
	[routed] = routed_batch

	decompose_tape = decompose.tape_transform
	if decompose_tape is None:
		raise RuntimeError("decompose.tape_transform is unexpectedly None")
	decomposed_batch, _ = decompose_tape(routed, stopping_condition=stopping_condition, name="iqm.direct")
	[decomposed] = decomposed_batch
	return decomposed


def _template_cost(instructions: InstructionTemplate, layout: WireLayout, gate_costs: GateCosts) -> float:
	"""Score a logical instruction template on a physical layout."""
	total = 0.0
	for gate_name, logical_locus in instructions:
		physical_locus = tuple(layout[wire] for wire in logical_locus)
		cost = gate_costs.get((gate_name, physical_locus), inf)
		if cost == inf:
			return inf
		total += cost
	return total


def _routed_layout_cost(tape: QuantumScript, layout: WireLayout, coupling: CouplingMap, gate_costs: GateCosts) -> float:
	"""Route a complete layout and score its actual emitted instructions."""
	mapped_tape = _map_tape(tape, layout)
	routed_tape = _route_tape(mapped_tape, coupling)
	wire_map = {wire: str(wire) for wire in routed_tape.wires}
	circuit = tape_to_iqm_circuit(routed_tape, wire_map, "layout_candidate")

	total = 0.0
	for instruction in circuit.instructions:
		cost = gate_costs.get((instruction.name, instruction.locus), inf)
		if cost == inf:
			return inf
		total += cost
	return total


def _search_native_layouts(
	tape: QuantumScript,
	dqa: DynamicQuantumArchitecture,
	instructions: InstructionTemplate,
	coupling: CouplingMap,
	gate_costs: GateCosts,
	max_candidates: int,
) -> tuple[WireLayout | None, bool]:
	"""Search native zero-SWAP embeddings with Rustworkx VF2."""
	wires = list(tape.wires)
	_, _, interactions = _instruction_counts(instructions)
	physical_index = {qubit: index for index, qubit in enumerate(dqa.qubits)}
	physical_graph, logical_graph = _native_graphs(wires, interactions, dqa, coupling)
	mappings = rx.vf2_mapping(physical_graph, logical_graph, subgraph=True, id_order=False, induced=False)

	best_layout: WireLayout | None = None
	best_key: tuple[float, tuple[int, ...]] | None = None
	for candidate_index, mapping in enumerate(mappings):
		if candidate_index >= max_candidates:
			return best_layout, True

		layout = {wires[logical_node]: dqa.qubits[physical_node] for physical_node, logical_node in mapping.items()}
		cost = _template_cost(instructions, layout, gate_costs)
		if cost == inf:
			continue
		tie_break = tuple(physical_index[layout[wire]] for wire in wires)
		key = (cost, tie_break)
		if best_key is None or key < best_key:
			best_key = key
			best_layout = layout

	return best_layout, False


def _search_routed_layouts(
	tape: QuantumScript,
	dqa: DynamicQuantumArchitecture,
	instructions: InstructionTemplate,
	coupling: CouplingMap,
	edge_costs: EdgeCosts,
	gate_costs: GateCosts,
	max_candidates: int,
) -> tuple[WireLayout | None, bool]:
	"""Search routed injective layouts and return the lowest-cost candidate."""
	wires = list(tape.wires)
	single_counts, measurement_counts, interactions = _instruction_counts(instructions)
	wire_index = {wire: index for index, wire in enumerate(wires)}
	physical_index = {qubit: index for index, qubit in enumerate(dqa.qubits)}
	degree = {wire: sum(count for pair, count in interactions.items() if wire in pair) for wire in wires}
	ordered_wires = sorted(
		wires, key=lambda wire: (-degree[wire], -single_counts[wire] - measurement_counts[wire], wire_index[wire])
	)

	local_costs: dict[tuple[Hashable, str], float] = {}
	for wire in wires:
		for qubit in dqa.qubits:
			cost = 0.0
			if single_counts[wire]:
				cost += single_counts[wire] * gate_costs.get(("prx", (qubit,)), inf)
			if measurement_counts[wire]:
				cost += measurement_counts[wire] * gate_costs.get(("measure", (qubit,)), inf)
			local_costs[(wire, qubit)] = cost

	routed_costs = {
		frozenset((first, second)): _routed_two_qubit_cost(first, second, coupling, edge_costs)
		for index, first in enumerate(dqa.qubits)
		for second in dqa.qubits[index + 1 :]
	}

	best_layout: WireLayout | None = None
	best_key: tuple[float, tuple[int, ...]] | None = None
	layout: WireLayout = {}
	used_qubits: set[str] = set()
	candidates_scored = 0
	limit_reached = False

	def visit(depth: int) -> None:
		nonlocal best_key, best_layout, candidates_scored, limit_reached
		if depth == len(ordered_wires):
			if candidates_scored >= max_candidates:
				limit_reached = True
				return
			candidates_scored += 1
			cost = _routed_layout_cost(tape, layout, coupling, gate_costs)
			if cost == inf:
				return
			tie_break = tuple(physical_index[layout[wire]] for wire in wires)
			key = (cost, tie_break)
			if best_key is None or key < best_key:
				best_key = key
				best_layout = dict(layout)
			return

		wire = ordered_wires[depth]
		options: list[tuple[float, int, str]] = []
		for qubit in dqa.qubits:
			if qubit in used_qubits:
				continue

			increment = local_costs[(wire, qubit)]
			compatible = True
			for assigned_wire, assigned_qubit in layout.items():
				count = interactions[frozenset((wire, assigned_wire))]
				if not count:
					continue
				pair = frozenset((qubit, assigned_qubit))
				pair_cost = routed_costs[pair]
				if pair_cost == inf:
					compatible = False
					break
				increment += count * pair_cost
			if compatible:
				options.append((increment, physical_index[qubit], qubit))

		for _, _, qubit in sorted(options):
			layout[wire] = qubit
			used_qubits.add(qubit)
			visit(depth + 1)
			used_qubits.remove(qubit)
			del layout[wire]
			if limit_reached:
				return

	visit(0)
	return best_layout, limit_reached


def _warn_search_limit(max_candidates: int, phase: str, candidate_found: bool) -> None:
	result = (
		"the selected layout is the best candidate found, not a proven global optimum"
		if candidate_found
		else "the search stopped before finding a candidate"
	)
	warnings.warn(
		f"{phase.capitalize()} layout search reached max_candidates={max_candidates}; {result}.",
		UserWarning,
		stacklevel=3,
	)


def _select_initial_layout(
	tape: QuantumScript,
	dqa: DynamicQuantumArchitecture,
	metrics: ObservationFinder | None,
	max_candidates: int = _DEFAULT_MAX_CANDIDATES,
) -> tuple[WireLayout, CouplingMap]:
	"""Select a topology- or quality-optimized physical layout.

	Native zero-SWAP embeddings are searched first. If none exists, injective
	initial layouts are routed and scored using their actual translated IQM
	instructions. Without metrics, each emitted CZ has unit cost and other gates
	have zero cost. With metrics, each emitted gate uses its calibrated fidelity
	cost. Up to ``max_candidates`` complete layouts are scored in each phase.
	Search order and tie-breaking are deterministic.
	"""
	if dqa.computational_resonators or "move" in dqa.gates:
		raise NotImplementedError(
			"Initial layout optimization does not support MOVE gates or computational resonators."
		)

	wires = list(tape.wires)
	if len(wires) > len(dqa.qubits):
		raise ValueError(f"Cannot lay out {len(wires)} circuit wires on {len(dqa.qubits)} IQM qubits.")
	if not wires:
		return {}, {qubit: [] for qubit in dqa.qubits}

	gate_costs = _gate_costs(dqa, metrics)
	coupling, edge_costs = _qubit_graph(dqa, gate_costs)
	instructions = _instruction_template(tape)
	native_layout, native_limit_reached = _search_native_layouts(
		tape, dqa, instructions, coupling, gate_costs, max_candidates
	)
	if native_layout is not None:
		if native_limit_reached:
			_warn_search_limit(max_candidates, "native", candidate_found=True)
		return native_layout, coupling
	if native_limit_reached:
		_warn_search_limit(max_candidates, "native", candidate_found=False)
		raise RuntimeError("Native layout search stopped before finding a candidate; increase max_candidates.")

	routed_layout, routed_limit_reached = _search_routed_layouts(
		tape, dqa, instructions, coupling, edge_costs, gate_costs, max_candidates
	)
	if routed_layout is not None:
		if routed_limit_reached:
			_warn_search_limit(max_candidates, "routed", candidate_found=True)
		return routed_layout, coupling
	if routed_limit_reached:
		_warn_search_limit(max_candidates, "routed", candidate_found=False)
		raise RuntimeError("Routed layout search stopped before finding a candidate; increase max_candidates.")
	raise ValueError("No IQM qubit layout supports this circuit.")
