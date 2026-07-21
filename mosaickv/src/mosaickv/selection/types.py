"""Auditable utility, objective, budget, and selection result schemas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from mosaickv.cache_state import ExactTier, FullKVState, KVBlockDescriptor
from mosaickv.config import UtilityConfig
from mosaickv.types import BudgetUnit


@dataclass(frozen=True, slots=True)
class SelectionBudget:
    """One explicit hard budget and its accounting unit."""

    value: int
    unit: BudgetUnit = BudgetUnit.BLOCKS

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("selection budget value must be >= 1")

    def cost(self, block: KVBlockDescriptor) -> int:
        """Return the exact cost of one source block under this budget."""

        if self.unit is BudgetUnit.BLOCKS:
            return 1
        if self.unit is BudgetUnit.RETAINED_SLOTS:
            return block.position_count
        if self.unit is BudgetUnit.BYTES:
            return block.byte_size
        raise ValueError(f"unsupported selection budget unit: {self.unit}")


@dataclass(frozen=True, slots=True)
class BlockUtility:
    """All raw signals and the signed local utility for one graph block."""

    node_id: int
    forecast_attention_probability: float
    value_novelty: float
    expected_attention_output_contribution: float
    graph_centrality: float
    singleton_coverage_gain: float
    modality_rarity: float
    redundancy_penalty: float
    mandatory_priority: float
    local_utility: float

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("utility node_id must be nonnegative")
        values = (
            self.forecast_attention_probability,
            self.value_novelty,
            self.expected_attention_output_contribution,
            self.graph_centrality,
            self.singleton_coverage_gain,
            self.modality_rarity,
            self.redundancy_penalty,
            self.mandatory_priority,
            self.local_utility,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("block utility signals must be finite")
        bounded = (
            self.forecast_attention_probability,
            self.value_novelty,
            self.graph_centrality,
            self.singleton_coverage_gain,
            self.modality_rarity,
            self.redundancy_penalty,
            self.mandatory_priority,
        )
        if any(not 0 <= value <= 1 for value in bounded):
            raise ValueError("normalized block utility signals must lie in [0, 1]")
        if self.expected_attention_output_contribution < 0:
            raise ValueError("attention-output contribution must be nonnegative")
        if not math.isclose(
            self.value_novelty + self.redundancy_penalty,
            1.0,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError("value novelty and redundancy penalty must be complements")

    @property
    def value_contribution(self) -> float:
        """Formula alias for expected attention-output contribution."""

        return self.expected_attention_output_contribution

    @property
    def uniqueness(self) -> float:
        """Formula alias for value novelty relative to graph neighbors."""

        return self.value_novelty


@dataclass(frozen=True, slots=True)
class BlockUtilityTable:
    """Complete, node-aligned local-utility table and its weights."""

    blocks: tuple[BlockUtility, ...]
    weights: UtilityConfig
    attention_provenance: str

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("utility table cannot be empty")
        if tuple(block.node_id for block in self.blocks) != tuple(range(len(self.blocks))):
            raise ValueError("utility blocks must be contiguous and ordered by node_id")
        if not self.attention_provenance.strip():
            raise ValueError("attention_provenance must be non-empty")
        for block in self.blocks:
            expected = (
                self.weights.lambda_q * block.forecast_attention_probability
                - self.weights.lambda_v * block.value_contribution
                - self.weights.lambda_o * block.uniqueness
            )
            if not math.isclose(block.local_utility, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("local utility does not match the configured equation")

    def for_node(self, node_id: int) -> BlockUtility:
        try:
            return self.blocks[node_id]
        except IndexError as error:
            raise IndexError(f"utility node does not exist: {node_id}") from error


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    """Exact value of the requested signed set objective."""

    selected_node_ids: tuple[int, ...]
    local_utility_sum: float
    facility_location_coverage: float
    facility_location_deficit: float
    modality_coverage: float
    modality_deficit: float
    total: float

    def __post_init__(self) -> None:
        values = (
            self.local_utility_sum,
            self.facility_location_coverage,
            self.facility_location_deficit,
            self.modality_coverage,
            self.modality_deficit,
            self.total,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("objective values must be finite")
        if self.selected_node_ids != tuple(sorted(set(self.selected_node_ids))):
            raise ValueError("objective selected_node_ids must be sorted and unique")
        bounded = (
            self.facility_location_coverage,
            self.facility_location_deficit,
            self.modality_coverage,
            self.modality_deficit,
        )
        if any(not 0 <= value <= 1 for value in bounded):
            raise ValueError("coverage values must lie in [0, 1]")
        if not math.isclose(
            self.facility_location_coverage + self.facility_location_deficit,
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("facility coverage and deficit must be complements")
        if not math.isclose(
            self.modality_coverage + self.modality_deficit,
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("modality coverage and deficit must be complements")


class SelectionReason(StrEnum):
    """Machine-readable reason recorded for every block."""

    MANDATORY_EXACT = "mandatory_exact"
    RETENTION_ONE_EXACT = "retention_one_exact"
    LAZY_GREEDY = "lazy_greedy"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NONPOSITIVE_MARGINAL_GAIN = "nonpositive_marginal_gain"
    NOT_SELECTED_LOWER_GAIN = "not_selected_lower_gain"


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Final disposition and relevant marginal gain for one node."""

    node_id: int
    selected: bool
    selection_rank: int | None
    marginal_gain: float
    marginal_gain_per_cost: float
    cost: int
    reason: SelectionReason

    def __post_init__(self) -> None:
        if self.node_id < 0 or self.cost < 1:
            raise ValueError("selection decision node and cost must be positive-domain values")
        if not math.isfinite(self.marginal_gain) or not math.isfinite(self.marginal_gain_per_cost):
            raise ValueError("selection marginal gains must be finite")
        if self.selected != (self.selection_rank is not None):
            raise ValueError("selected decisions must have exactly one selection rank")
        if self.reason in {
            SelectionReason.MANDATORY_EXACT,
            SelectionReason.RETENTION_ONE_EXACT,
            SelectionReason.LAZY_GREEDY,
        }:
            if not self.selected:
                raise ValueError("selected reason cannot be assigned to an unselected block")
        elif self.selected:
            raise ValueError("unselected reason cannot be assigned to a selected block")


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Mandatory-first lazy-greedy output with exact accounting."""

    selected_blocks: tuple[KVBlockDescriptor, ...]
    selection_order: tuple[int, ...]
    decisions: tuple[SelectionDecision, ...]
    budget: SelectionBudget
    budget_spent: int
    active_bytes: int
    objective: ObjectiveBreakdown
    heap_pops: int
    marginal_recomputations: int

    def __post_init__(self) -> None:
        selected_ids = tuple(block_id for block_id in self.objective.selected_node_ids)
        if tuple(sorted(self.selection_order)) != selected_ids:
            raise ValueError("selection order and objective node set differ")
        if len(self.selected_blocks) != len(selected_ids):
            raise ValueError("selected blocks do not align with selected node IDs")
        if tuple(decision.node_id for decision in self.decisions) != tuple(
            range(len(self.decisions))
        ):
            raise ValueError("selection decisions must cover every ordered graph node")
        decided_selected = tuple(
            decision.node_id for decision in self.decisions if decision.selected
        )
        if decided_selected != selected_ids:
            raise ValueError("selected decisions do not match objective node IDs")
        ranks = sorted(
            decision.selection_rank
            for decision in self.decisions
            if decision.selection_rank is not None
        )
        if ranks != list(range(len(selected_ids))):
            raise ValueError("selection ranks must be contiguous")
        if self.budget_spent < 0 or self.budget_spent > self.budget.value:
            raise ValueError("selection exceeds its configured budget")
        if self.active_bytes != sum(block.byte_size for block in self.selected_blocks):
            raise ValueError("selected active-byte accounting does not match block storage")
        if self.budget.unit is BudgetUnit.BYTES and self.active_bytes != self.budget_spent:
            raise ValueError("byte-budget cost must exactly equal selected active bytes")
        for node_id, block in zip(selected_ids, self.selected_blocks, strict=True):
            marked_mandatory = self.decisions[node_id].reason is SelectionReason.MANDATORY_EXACT
            if block.mandatory != marked_mandatory:
                raise ValueError("mandatory blocks and mandatory selection reasons differ")
        if self.heap_pops < 0 or self.marginal_recomputations < 0:
            raise ValueError("selector operation counts cannot be negative")

    @property
    def selected_node_ids(self) -> tuple[int, ...]:
        return self.objective.selected_node_ids

    @property
    def remaining_budget(self) -> int:
        return self.budget.value - self.budget_spent

    def to_exact_tier(self, full_state: FullKVState) -> ExactTier:
        """Gather the selected block payloads from their unchanged source cache."""

        return full_state.gather_exact_blocks(self.selected_blocks)


@dataclass(frozen=True, slots=True)
class ExhaustiveSelectionResult:
    """Exact optimum for a deliberately tiny candidate universe."""

    selected_node_ids: tuple[int, ...]
    objective: ObjectiveBreakdown
    budget_spent: int
    evaluated_subsets: int

    def __post_init__(self) -> None:
        if self.selected_node_ids != tuple(sorted(set(self.selected_node_ids))):
            raise ValueError("exhaustive selected node IDs must be sorted and unique")
        if self.objective.selected_node_ids != self.selected_node_ids:
            raise ValueError("exhaustive node IDs and objective differ")
        if self.budget_spent < 0 or self.evaluated_subsets < 1:
            raise ValueError("exhaustive accounting values are invalid")


@dataclass(frozen=True, slots=True)
class GreedyOptimalityComparison:
    """Greedy-versus-exhaustive objective gap on a tiny graph."""

    greedy_objective: float
    optimal_objective: float
    absolute_gap: float
    relative_gap: float
    greedy_node_ids: tuple[int, ...]
    optimal_node_ids: tuple[int, ...]
    evaluated_subsets: int

    def __post_init__(self) -> None:
        values = (
            self.greedy_objective,
            self.optimal_objective,
            self.absolute_gap,
            self.relative_gap,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("optimality comparison values must be finite")
        if self.absolute_gap < 0 or self.relative_gap < 0:
            raise ValueError("optimality gaps cannot be negative")
        if self.optimal_objective + 1e-12 < self.greedy_objective:
            raise ValueError("exhaustive objective cannot be below the greedy objective")
        expected_gap = max(0.0, self.optimal_objective - self.greedy_objective)
        if not math.isclose(self.absolute_gap, expected_gap, rel_tol=0, abs_tol=1e-12):
            raise ValueError("absolute optimality gap does not match objective values")


@dataclass(frozen=True, slots=True)
class ObjectivePropertyReport:
    """Exhaustive monotonicity and diminishing-returns audit for a tiny graph."""

    monotone: bool
    diminishing_returns: bool
    monotonicity_witness: tuple[tuple[int, ...], int] | None
    diminishing_returns_witness: tuple[tuple[int, ...], tuple[int, ...], int] | None
    evaluated_subsets: int

    def __post_init__(self) -> None:
        if self.monotone != (self.monotonicity_witness is None):
            raise ValueError("monotonicity status and witness disagree")
        if self.diminishing_returns != (self.diminishing_returns_witness is None):
            raise ValueError("diminishing-returns status and witness disagree")
        if self.evaluated_subsets < 1:
            raise ValueError("property audit must evaluate at least one subset")


__all__ = [
    "BlockUtility",
    "BlockUtilityTable",
    "ExhaustiveSelectionResult",
    "GreedyOptimalityComparison",
    "ObjectiveBreakdown",
    "ObjectivePropertyReport",
    "SelectionBudget",
    "SelectionDecision",
    "SelectionReason",
    "SelectionResult",
]
