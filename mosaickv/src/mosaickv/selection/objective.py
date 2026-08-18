"""Sparse facility-location objective and tiny-graph property audits."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from mosaickv.config import SelectionConfig
from mosaickv.graph import SparseEvidenceGraph
from mosaickv.selection.types import (
    BlockUtilityTable,
    ObjectiveBreakdown,
    ObjectivePropertyReport,
)


class BudgetedObjective:
    """Exact signed objective over a fixed sparse graph.

    ``facility_location_coverage`` and ``modality_coverage`` are ordinary
    nonnegative coverage rewards, and the implementation uses the prompt's
    minus signs literally.  Consequently the objective is submodular when
    ``lambda_g`` and ``lambda_m`` are nonpositive; positive values describe a
    coverage penalty and do not satisfy the lazy-greedy proof assumptions.
    """

    def __init__(
        self,
        graph: SparseEvidenceGraph,
        utilities: BlockUtilityTable,
        config: SelectionConfig,
    ) -> None:
        if len(graph.nodes) != len(utilities.blocks):
            raise ValueError("utility table and evidence graph node counts differ")
        if not config.enabled:
            raise ValueError("cannot construct selection objective when selection.enabled is false")
        self.graph = graph
        self.utilities = utilities
        self.config = config
        self._facility: tuple[dict[int, float], ...] = self._build_facility()
        self._modalities = frozenset(node.block.modality for node in graph.nodes)
        offsets = [0]
        targets: list[int] = []
        weights: list[float] = []
        for row in self._facility:
            for target, weight in sorted(row.items()):
                targets.append(target)
                weights.append(weight)
            offsets.append(len(targets))
        self._facility_offsets = np.asarray(offsets, dtype=np.int64)
        self._facility_targets = np.asarray(targets, dtype=np.int64)
        self._facility_weights = np.asarray(weights, dtype=np.float64)
        self._local_utilities = np.asarray(
            tuple(block.local_utility for block in utilities.blocks),
            dtype=np.float64,
        )
        ordered_modalities = tuple(sorted(self._modalities, key=str))
        modality_ids = {modality: index for index, modality in enumerate(ordered_modalities)}
        self._node_modality_ids = np.asarray(
            tuple(modality_ids[node.block.modality] for node in graph.nodes),
            dtype=np.int64,
        )

    def _build_facility(self) -> tuple[dict[int, float], ...]:
        facilities: list[dict[int, float]] = [{node.node_id: 1.0} for node in self.graph.nodes]
        for source, target, weight in zip(
            self.graph.row_indices,
            self.graph.column_indices,
            self.graph.weights,
            strict=True,
        ):
            facilities[source][target] = max(
                weight,
                facilities[source].get(target, 0.0),
            )
        return tuple(facilities)

    @property
    def is_submodular(self) -> bool:
        """Whether coefficient signs preserve the coverage submodularity proof."""

        return self.config.lambda_g <= 0 and self.config.lambda_m <= 0

    def facility_similarities(self, node_id: int) -> dict[int, float]:
        """Return a copy of one representative's sparse target similarities."""

        self._validate_node(node_id)
        return dict(self._facility[node_id])

    def _validate_node(self, node_id: int) -> None:
        if node_id < 0 or node_id >= len(self.graph.nodes):
            raise IndexError(f"objective node does not exist: {node_id}")

    def _selected(self, selected_node_ids: Iterable[int]) -> tuple[int, ...]:
        result = tuple(sorted(int(node_id) for node_id in selected_node_ids))
        if len(set(result)) != len(result):
            raise ValueError("objective selected nodes must be unique")
        for node_id in result:
            self._validate_node(node_id)
        return result

    def evaluate(self, selected_node_ids: Iterable[int]) -> ObjectiveBreakdown:
        """Evaluate the exact set objective without dense graph materialization."""

        selected = self._selected(selected_node_ids)
        local_sum = sum(self.utilities.for_node(node_id).local_utility for node_id in selected)
        covered = [0.0] * len(self.graph.nodes)
        for node_id in selected:
            for target, similarity in self._facility[node_id].items():
                covered[target] = max(covered[target], similarity)
        facility_coverage = sum(covered) / len(covered)
        selected_modalities = {self.graph.nodes[node_id].block.modality for node_id in selected}
        modality_coverage = len(selected_modalities) / len(self._modalities)
        facility_deficit = 1.0 - facility_coverage
        modality_deficit = 1.0 - modality_coverage
        total = (
            local_sum
            - self.config.lambda_g * facility_coverage
            - self.config.lambda_m * modality_coverage
        )
        result = ObjectiveBreakdown(
            selected_node_ids=selected,
            local_utility_sum=local_sum,
            facility_location_coverage=facility_coverage,
            facility_location_deficit=facility_deficit,
            modality_coverage=modality_coverage,
            modality_deficit=modality_deficit,
            total=total,
        )
        expected = (
            result.local_utility_sum
            - self.config.lambda_g * result.facility_location_coverage
            - self.config.lambda_m * result.modality_coverage
        )
        if not math.isclose(result.total, expected, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("objective breakdown violates its configured formula")
        return result

    def marginal_gain(self, selected_node_ids: Iterable[int], candidate: int) -> float:
        """Return ``F(S union {i}) - F(S)`` exactly."""

        selected = self._selected(selected_node_ids)
        self._validate_node(candidate)
        if candidate in selected:
            raise ValueError("candidate is already selected")
        current = self.evaluate(selected).total
        following = self.evaluate((*selected, candidate)).total
        return following - current

    def incremental_state(self) -> IncrementalObjectiveState:
        """Return an exact mutable coverage state for scalable lazy greedy."""

        return IncrementalObjectiveState(self)


class IncrementalObjectiveState:
    """Maintain exact facility/modality coverage under append-only selection.

    The mathematical objective is unchanged.  This state avoids rebuilding
    ``F(S)`` and ``F(S union {i})`` from every previously selected block for each
    heap refresh; a marginal calculation touches only the candidate's sparse
    neighbors.
    """

    def __init__(self, objective: BudgetedObjective) -> None:
        self._objective = objective
        self._covered = np.zeros(len(objective.graph.nodes), dtype=np.float64)
        self._selected: set[int] = set()
        self._selected_modalities: set[object] = set()

    def marginal_gain(self, candidate: int) -> float:
        """Return the exact gain for adding ``candidate`` to the current state."""

        objective = self._objective
        objective._validate_node(candidate)
        if candidate in self._selected:
            raise ValueError("candidate is already selected")
        facility_delta = sum(
            max(self._covered[target], similarity) - self._covered[target]
            for target, similarity in objective._facility[candidate].items()
        ) / len(self._covered)
        modality = objective.graph.nodes[candidate].block.modality
        modality_delta = (
            0.0 if modality in self._selected_modalities else 1.0 / len(objective._modalities)
        )
        return float(
            objective.utilities.for_node(candidate).local_utility
            - objective.config.lambda_g * facility_delta
            - objective.config.lambda_m * modality_delta
        )

    def marginal_gains(self, candidates: Iterable[int]) -> tuple[float, ...]:
        """Return exact gains for many candidates from one sparse array pass."""

        objective = self._objective
        identifiers = np.asarray(tuple(int(value) for value in candidates), dtype=np.int64)
        if identifiers.size == 0:
            return ()
        if bool(np.any(identifiers < 0)) or bool(
            np.any(identifiers >= len(objective.graph.nodes))
        ):
            raise IndexError("objective candidate does not exist")
        if any(int(candidate) in self._selected for candidate in identifiers):
            raise ValueError("candidate is already selected")
        edge_delta = np.maximum(
            self._covered[objective._facility_targets],
            objective._facility_weights,
        ) - self._covered[objective._facility_targets]
        # Every row contains at least its unit-weight self edge.
        facility_delta = np.add.reduceat(
            edge_delta,
            objective._facility_offsets[:-1],
        ) / len(self._covered)
        modality_missing = np.asarray(
            tuple(
                objective.graph.nodes[int(candidate)].block.modality
                not in self._selected_modalities
                for candidate in identifiers
            ),
            dtype=np.float64,
        ) / len(objective._modalities)
        gains = (
            objective._local_utilities[identifiers]
            - objective.config.lambda_g * facility_delta[identifiers]
            - objective.config.lambda_m * modality_missing
        )
        return tuple(float(value) for value in gains)

    def select(self, candidate: int) -> float:
        """Add one node, update sparse coverage, and return its exact gain."""

        gain = self.marginal_gain(candidate)
        objective = self._objective
        for target, similarity in objective._facility[candidate].items():
            self._covered[target] = max(self._covered[target], similarity)
        self._selected_modalities.add(objective.graph.nodes[candidate].block.modality)
        self._selected.add(candidate)
        return gain

    def reserve_many(self, candidates: Iterable[int]) -> tuple[float, ...]:
        """Atomically add hard-reserved nodes and return snapshot marginals.

        Hard reservations are constraints, not greedy choices.  Their order
        must therefore not change which nodes survive.  This operation records
        every node's exact marginal gain against the same pre-reservation
        snapshot, then applies the set-union coverage update in one vectorized
        transaction.  The resulting objective state is exactly the same as
        selecting the nodes sequentially because facility and modality
        coverage depend only on the selected set.
        """

        objective = self._objective
        identifiers = tuple(int(value) for value in candidates)
        if not identifiers:
            return ()
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("reserved objective candidates must be unique")
        gains = self.marginal_gains(identifiers)
        edge_segments = tuple(
            slice(
                int(objective._facility_offsets[candidate]),
                int(objective._facility_offsets[candidate + 1]),
            )
            for candidate in identifiers
        )
        targets = np.concatenate(
            tuple(objective._facility_targets[segment] for segment in edge_segments)
        )
        weights = np.concatenate(
            tuple(objective._facility_weights[segment] for segment in edge_segments)
        )
        np.maximum.at(self._covered, targets, weights)
        self._selected.update(identifiers)
        self._selected_modalities.update(
            objective.graph.nodes[candidate].block.modality for candidate in identifiers
        )
        return gains


def audit_objective_properties(
    objective: BudgetedObjective,
    *,
    maximum_nodes: int = 10,
    tolerance: float = 1e-12,
) -> ObjectivePropertyReport:
    """Exhaustively audit monotonicity and diminishing returns on a tiny graph."""

    node_count = len(objective.graph.nodes)
    if node_count > maximum_nodes:
        raise ValueError(
            f"property audit is limited to {maximum_nodes} nodes; received {node_count}"
        )
    subsets: dict[int, tuple[int, ...]] = {
        mask: tuple(node for node in range(node_count) if mask & (1 << node))
        for mask in range(1 << node_count)
    }
    values = {mask: objective.evaluate(nodes).total for mask, nodes in subsets.items()}
    monotonicity_witness: tuple[tuple[int, ...], int] | None = None
    diminishing_witness: tuple[tuple[int, ...], tuple[int, ...], int] | None = None
    for smaller_mask, smaller in subsets.items():
        for candidate in range(node_count):
            candidate_bit = 1 << candidate
            if smaller_mask & candidate_bit:
                continue
            smaller_gain = values[smaller_mask | candidate_bit] - values[smaller_mask]
            if monotonicity_witness is None and smaller_gain < -tolerance:
                monotonicity_witness = (smaller, candidate)
            if diminishing_witness is not None:
                continue
            remaining = ((1 << node_count) - 1) ^ (smaller_mask | candidate_bit)
            extension = remaining
            while True:
                larger_mask = smaller_mask | extension
                larger_gain = values[larger_mask | candidate_bit] - values[larger_mask]
                if smaller_gain + tolerance < larger_gain:
                    diminishing_witness = (smaller, subsets[larger_mask], candidate)
                    break
                if extension == 0:
                    break
                extension = (extension - 1) & remaining
    return ObjectivePropertyReport(
        monotone=monotonicity_witness is None,
        diminishing_returns=diminishing_witness is None,
        monotonicity_witness=monotonicity_witness,
        diminishing_returns_witness=diminishing_witness,
        evaluated_subsets=len(subsets),
    )


__all__ = ["BudgetedObjective", "IncrementalObjectiveState", "audit_objective_properties"]
