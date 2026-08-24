"""Deterministic, exact-tier simple baseline selectors."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, cast

import numpy as np

from mosaickv.baselines.types import (
    BaselineCompressionPlan,
    BaselineSelectionDecision,
    BaselineSelectionReason,
    BaselineSelectionResult,
    BaselineStratum,
)
from mosaickv.cache_state import FullKVState, KVBlockDescriptor, Modality, MosaicKVState
from mosaickv.config import CacheConfig
from mosaickv.graph import pool_block_descriptors
from mosaickv.selection import SelectionBudget
from mosaickv.types import BudgetUnit, MosaicKVMethod


class InfeasibleBaselineBudgetError(ValueError):
    """Raised when mandatory exact blocks cannot fit the common budget."""


@dataclass(frozen=True, slots=True)
class HeadLocalCapacityRecord:
    """Auditable physical-capacity use for one independently routed KV head."""

    layer: int
    kv_head: int
    position_limit: int
    selected_positions: int
    shortfall_positions: int
    selected_bytes: int
    selected_visual_positions: int = 0
    target_visual_positions: int = 0
    adaptive_swap_count: int = 0
    baseline_text_prompt_mass: float = 0.0
    selected_text_prompt_mass: float = 0.0
    normalized_visual_evidence_gain: float = 0.0
    normalized_text_prompt_mass_loss: float = 0.0
    normalized_forecast_visual_gain: float = 0.0
    normalized_forecast_text_loss: float = 0.0
    adaptive_swap_candidate_count: int = 0
    adaptive_swap_risk_rejected_count: int = 0
    adaptive_swap_pairs: tuple[tuple[int, int], ...] = ()
    future_rerank_count: int = 0
    baseline_selected_prompt_mass: float = 0.0
    selected_prompt_mass_after_future_rerank: float = 0.0
    normalized_future_query_gain: float = 0.0
    normalized_future_prompt_mass_loss: float = 0.0
    future_rerank_candidate_count: int = 0
    future_rerank_rejected_count: int = 0
    future_rerank_risk_rejected_count: int = 0
    future_rerank_pairs: tuple[tuple[int, int], ...] = ()
    prompt_forecast_total_variation: float = 0.0
    forecast_divergence_eligible: bool = False
    query_coverage_rerank_count: int = 0
    baseline_query_coverage: float = 0.0
    selected_query_coverage: float = 0.0
    normalized_query_coverage_gain: float = 0.0
    query_coverage_baseline_prompt_mass: float = 0.0
    query_coverage_selected_prompt_mass: float = 0.0
    query_coverage_normalized_prompt_loss: float = 0.0
    query_coverage_candidate_count: int = 0
    query_coverage_rejected_count: int = 0
    query_coverage_pairs: tuple[tuple[int, int], ...] = ()
    query_coverage_initial_reserved_count: int = 0
    query_coverage_initial_changed_count: int = 0
    query_coverage_initial_baseline: float = 0.0
    query_coverage_initial_selected: float = 0.0
    query_coverage_initial_normalized_gain: float = 0.0
    query_coverage_initial_baseline_prompt_mass: float = 0.0
    query_coverage_initial_selected_prompt_mass: float = 0.0
    query_coverage_initial_normalized_prompt_loss: float = 0.0
    query_coverage_initial_candidate_count: int = 0
    query_coverage_initial_reserved_bytes: int = 0
    query_coverage_initial_allocated_bytes: int = 0
    query_coverage_initial_cross_head_risk: float = 0.0
    query_coverage_initial_proposed_normalized_gain: float = 0.0
    query_coverage_initial_proposed_normalized_prompt_loss: float = 0.0
    query_coverage_initial_proposed_output_impact_risk: float = 0.0
    query_coverage_initial_output_impact_donor_count: int = 0
    query_coverage_initial_output_impact_protected_count: int = 0
    query_coverage_initial_output_impact_protected_bytes: int = 0
    query_coverage_initial_output_impact_protected_score: float = 0.0
    query_coverage_initial_layout_group_candidate_count: int = 0
    query_coverage_initial_layout_group_protected_group_count: int = 0
    query_coverage_initial_layout_group_protected_block_count: int = 0
    query_coverage_initial_layout_group_protected_bytes: int = 0
    query_coverage_initial_layout_group_protected_score: float = 0.0
    query_coverage_initial_layout_group_target_bytes: int = 0
    query_coverage_initial_layout_group_accepted_candidate_count: int = 0
    query_coverage_initial_layout_group_risk_rejected_count: int = 0
    query_coverage_initial_layout_group_pareto_candidate_count: int = 0
    query_coverage_initial_layout_group_knapsack_selected_count: int = 0
    query_coverage_initial_layout_group_joint_donor_candidate_count: int = 0
    query_coverage_initial_layout_group_normalized_query_gain: float = 0.0
    query_coverage_initial_layout_group_normalized_prompt_risk: float = 0.0
    query_coverage_initial_layout_group_output_impact_risk: float = 0.0
    query_coverage_initial_layout_group_node_ids: tuple[tuple[int, ...], ...] = ()
    query_coverage_initial_stability_rejected: bool = False
    query_coverage_initial_global_allocation: bool = False
    query_coverage_initial_added_node_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kv_head < 0:
            raise ValueError("layer and KV head must be nonnegative")
        if self.position_limit < 0 or self.selected_positions < 0:
            raise ValueError("position capacities must be nonnegative")
        if self.selected_positions > self.position_limit:
            raise ValueError("selected positions exceed the route capacity")
        if self.shortfall_positions != self.position_limit - self.selected_positions:
            raise ValueError("route capacity shortfall is inconsistent")
        if self.selected_bytes < 0:
            raise ValueError("selected bytes must be nonnegative")
        if not 0 <= self.selected_visual_positions <= self.selected_positions:
            raise ValueError("selected visual positions must lie within selected positions")
        if not 0 <= self.target_visual_positions <= self.position_limit:
            raise ValueError("target visual positions must lie within the route capacity")
        if self.adaptive_swap_count < 0:
            raise ValueError("adaptive swap count must be nonnegative")
        if self.adaptive_swap_candidate_count < self.adaptive_swap_count:
            raise ValueError("adaptive swap candidates cannot be fewer than accepted swaps")
        if not 0 <= self.adaptive_swap_risk_rejected_count <= self.adaptive_swap_candidate_count:
            raise ValueError("adaptive swap risk rejections must lie within candidate count")
        if len(self.adaptive_swap_pairs) != self.adaptive_swap_count:
            raise ValueError("adaptive swap pairs must align with the accepted swap count")
        if any(text < 0 or visual < 0 for text, visual in self.adaptive_swap_pairs):
            raise ValueError("adaptive swap node IDs must be nonnegative")
        if self.future_rerank_count < 0:
            raise ValueError("future rerank count must be nonnegative")
        if self.future_rerank_candidate_count < self.future_rerank_count:
            raise ValueError("future rerank candidates cannot be fewer than accepted replacements")
        if not 0 <= self.future_rerank_rejected_count <= self.future_rerank_candidate_count:
            raise ValueError("future rerank rejections must lie within candidate count")
        if not 0 <= self.future_rerank_risk_rejected_count <= self.future_rerank_rejected_count:
            raise ValueError("future rerank risk rejections must lie within rejected count")
        if len(self.future_rerank_pairs) != self.future_rerank_count:
            raise ValueError("future rerank pairs must align with the accepted replacement count")
        if any(donor < 0 or recipient < 0 for donor, recipient in self.future_rerank_pairs):
            raise ValueError("future rerank node IDs must be nonnegative")
        if self.query_coverage_rerank_count < 0:
            raise ValueError("query coverage rerank count must be nonnegative")
        if self.query_coverage_candidate_count < self.query_coverage_rerank_count:
            raise ValueError("query coverage candidates cannot be fewer than accepted replacements")
        if not 0 <= self.query_coverage_rejected_count <= self.query_coverage_candidate_count:
            raise ValueError("query coverage rejections must lie within candidate count")
        if len(self.query_coverage_pairs) != self.query_coverage_rerank_count:
            raise ValueError("query coverage pairs must align with the accepted replacement count")
        if any(donor < 0 or recipient < 0 for donor, recipient in self.query_coverage_pairs):
            raise ValueError("query coverage rerank node IDs must be nonnegative")
        if self.query_coverage_initial_reserved_count < 0:
            raise ValueError("initial query coverage reserved count must be nonnegative")
        if self.query_coverage_initial_changed_count < 0:
            raise ValueError("initial query coverage changed count must be nonnegative")
        if self.query_coverage_initial_candidate_count < 0:
            raise ValueError("initial query coverage candidate count must be nonnegative")
        if self.query_coverage_initial_reserved_bytes < 0:
            raise ValueError("initial query coverage reserved bytes must be nonnegative")
        if self.query_coverage_initial_allocated_bytes < 0:
            raise ValueError("initial query coverage allocated bytes must be nonnegative")
        if self.query_coverage_initial_output_impact_donor_count < 0:
            raise ValueError("initial query coverage output-impact donors must be nonnegative")
        if self.query_coverage_initial_output_impact_protected_count < 0:
            raise ValueError("initial query coverage protected count must be nonnegative")
        if self.query_coverage_initial_output_impact_protected_bytes < 0:
            raise ValueError("initial query coverage protected bytes must be nonnegative")
        for name in (
            "query_coverage_initial_layout_group_candidate_count",
            "query_coverage_initial_layout_group_protected_group_count",
            "query_coverage_initial_layout_group_protected_block_count",
            "query_coverage_initial_layout_group_protected_bytes",
            "query_coverage_initial_layout_group_target_bytes",
            "query_coverage_initial_layout_group_accepted_candidate_count",
            "query_coverage_initial_layout_group_risk_rejected_count",
            "query_coverage_initial_layout_group_pareto_candidate_count",
            "query_coverage_initial_layout_group_knapsack_selected_count",
            "query_coverage_initial_layout_group_joint_donor_candidate_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            len(self.query_coverage_initial_layout_group_node_ids)
            != self.query_coverage_initial_layout_group_protected_group_count
        ):
            raise ValueError("layout-group memberships must align with protected group count")
        flattened_layout_nodes = tuple(
            node_id
            for group in self.query_coverage_initial_layout_group_node_ids
            for node_id in group
        )
        if (
            len(flattened_layout_nodes)
            != self.query_coverage_initial_layout_group_protected_block_count
        ):
            raise ValueError("layout-group memberships must align with protected block count")
        if len(flattened_layout_nodes) != len(set(flattened_layout_nodes)):
            raise ValueError("protected layout groups cannot overlap")
        if any(node_id < 0 for node_id in flattened_layout_nodes):
            raise ValueError("protected layout-group node IDs must be nonnegative")
        if (
            len(self.query_coverage_initial_added_node_ids)
            != self.query_coverage_initial_changed_count
        ):
            raise ValueError("initial query coverage added IDs must align with changed count")
        if any(node_id < 0 for node_id in self.query_coverage_initial_added_node_ids):
            raise ValueError("initial query coverage node IDs must be nonnegative")
        for name in (
            "baseline_text_prompt_mass",
            "selected_text_prompt_mass",
            "normalized_visual_evidence_gain",
            "normalized_text_prompt_mass_loss",
            "normalized_forecast_visual_gain",
            "normalized_forecast_text_loss",
            "baseline_selected_prompt_mass",
            "selected_prompt_mass_after_future_rerank",
            "normalized_future_query_gain",
            "normalized_future_prompt_mass_loss",
            "prompt_forecast_total_variation",
            "baseline_query_coverage",
            "selected_query_coverage",
            "normalized_query_coverage_gain",
            "query_coverage_baseline_prompt_mass",
            "query_coverage_selected_prompt_mass",
            "query_coverage_normalized_prompt_loss",
            "query_coverage_initial_baseline",
            "query_coverage_initial_selected",
            "query_coverage_initial_normalized_gain",
            "query_coverage_initial_baseline_prompt_mass",
            "query_coverage_initial_selected_prompt_mass",
            "query_coverage_initial_normalized_prompt_loss",
            "query_coverage_initial_cross_head_risk",
            "query_coverage_initial_proposed_normalized_gain",
            "query_coverage_initial_proposed_normalized_prompt_loss",
            "query_coverage_initial_proposed_output_impact_risk",
            "query_coverage_initial_output_impact_protected_score",
            "query_coverage_initial_layout_group_protected_score",
            "query_coverage_initial_layout_group_normalized_query_gain",
            "query_coverage_initial_layout_group_normalized_prompt_risk",
            "query_coverage_initial_layout_group_output_impact_risk",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.selected_text_prompt_mass > self.baseline_text_prompt_mass + 1e-12:
            raise ValueError("selected text prompt mass cannot exceed its baseline")


@dataclass(frozen=True, slots=True)
class HeadLocalPromptSelectionAlternative:
    """One same-byte external exact-cache point in a bounded Pareto shortlist."""

    candidate_id: str
    selected_node_ids: tuple[int, ...]
    capacity_records: tuple[HeadLocalCapacityRecord, ...]


@dataclass(frozen=True, slots=True)
class HeadLocalPromptSelection:
    """Selected source blocks plus exact per-route capacity accounting."""

    selected_node_ids: tuple[int, ...]
    capacity_records: tuple[HeadLocalCapacityRecord, ...]
    logit_stability_shortlist: tuple[HeadLocalPromptSelectionAlternative, ...] = ()

    @property
    def total_shortfall_positions(self) -> int:
        return sum(record.shortfall_positions for record in self.capacity_records)


@dataclass(frozen=True, slots=True)
class _AdaptiveSwapResult:
    selected_optional_node_ids: tuple[int, ...]
    swap_count: int
    baseline_text_prompt_mass: float
    selected_text_prompt_mass: float
    normalized_visual_evidence_gain: float
    normalized_text_prompt_mass_loss: float
    normalized_forecast_visual_gain: float
    normalized_forecast_text_loss: float
    candidate_count: int
    risk_rejected_count: int
    accepted_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class _FutureRerankResult:
    selected_optional_node_ids: tuple[int, ...]
    replacement_count: int
    baseline_selected_prompt_mass: float
    selected_prompt_mass: float
    normalized_future_query_gain: float
    normalized_prompt_mass_loss: float
    candidate_count: int
    rejected_count: int
    risk_rejected_count: int
    accepted_pairs: tuple[tuple[int, int], ...]
    prompt_forecast_total_variation: float
    forecast_divergence_eligible: bool


@dataclass(frozen=True, slots=True)
class _QueryCoverageRerankResult:
    """Same-byte replacements that protect distinct prompt-query evidence."""

    selected_optional_node_ids: tuple[int, ...]
    replacement_count: int
    baseline_query_coverage: float
    selected_query_coverage: float
    normalized_query_coverage_gain: float
    baseline_selected_prompt_mass: float
    selected_prompt_mass: float
    normalized_prompt_mass_loss: float
    candidate_count: int
    rejected_count: int
    accepted_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class _QueryCoverageInitialSelectionResult:
    """Coverage-reserved initial selection at one fixed head capacity."""

    selected_optional_node_ids: tuple[int, ...]
    reserved_count: int
    changed_count: int
    baseline_query_coverage: float
    selected_query_coverage: float
    normalized_query_coverage_gain: float
    baseline_selected_prompt_mass: float
    selected_prompt_mass: float
    normalized_prompt_mass_loss: float
    candidate_count: int
    reserved_node_ids: tuple[int, ...]
    added_node_ids: tuple[int, ...]
    cross_head_risk: float = 0.0
    allocated_bytes: int = 0
    proposed_normalized_gain: float = 0.0
    proposed_normalized_prompt_loss: float = 0.0
    proposed_output_impact_risk: float = 0.0
    output_impact_donor_count: int = 0
    output_impact_protected_count: int = 0
    output_impact_protected_bytes: int = 0
    output_impact_protected_score: float = 0.0
    layout_group_candidate_count: int = 0
    layout_group_protected_groups: tuple[tuple[int, ...], ...] = ()
    layout_group_protected_bytes: int = 0
    layout_group_protected_score: float = 0.0
    layout_group_target_bytes: int = 0
    layout_group_accepted_candidate_count: int = 0
    layout_group_risk_rejected_count: int = 0
    layout_group_pareto_candidate_count: int = 0
    layout_group_knapsack_selected_count: int = 0
    layout_group_joint_donor_candidate_count: int = 0
    layout_group_normalized_query_gain: float = 0.0
    layout_group_normalized_prompt_risk: float = 0.0
    layout_group_output_impact_risk: float = 0.0
    stability_rejected: bool = False


@dataclass(frozen=True, slots=True)
class _AtomicLayoutGroupCandidate:
    """One route-local atomic group and its exact-capacity counterfactual."""

    identity: tuple[int, int]
    group: tuple[int, ...]
    selected_optional_node_ids: tuple[int, ...]
    normalized_query_gain: float
    normalized_prompt_risk: float
    output_impact_risk: float
    group_score: float
    group_bytes: int
    baseline_coverage: float
    selected_coverage: float
    baseline_prompt_mass: float
    selected_prompt_mass: float
    donor_node_ids: tuple[int, ...]
    recipient_node_ids: tuple[int, ...]


def _stratum(block: KVBlockDescriptor) -> BaselineStratum:
    return (block.layer, block.kv_head, block.modality)


def _source_cost(full_state: FullKVState, unit: BudgetUnit) -> int:
    budget = SelectionBudget(1, unit)
    return sum(budget.cost(block) for block in full_state.blocks)


def resolve_baseline_budget(
    full_state: FullKVState,
    config: CacheConfig,
) -> tuple[int, SelectionBudget]:
    """Resolve the same retention-ratio/upper-bound rule used by MosaicKV."""

    source = _source_cost(full_state, config.budget_unit)
    requested = math.ceil(source * config.retention_ratio)
    target = min(requested, config.budget_value)
    budget = SelectionBudget(target, config.budget_unit)
    mandatory = sum(budget.cost(block) for block in full_state.blocks if block.mandatory)
    if config.retention_ratio == 1.0 and target != source:
        raise InfeasibleBaselineBudgetError(
            "retention 1.0 requires cache.budget_value to cover the complete FullKV cache"
        )
    if mandatory > target:
        raise InfeasibleBaselineBudgetError(
            "mandatory exact blocks exceed the baseline budget: "
            f"required={mandatory}, budget={target}, unit={config.budget_unit.value}"
        )
    return source, budget


def _finalize(
    full_state: FullKVState,
    method: MosaicKVMethod,
    budget: SelectionBudget,
    *,
    order: Sequence[int],
    scores: Mapping[int, float],
    selected_reason: BaselineSelectionReason,
    seed: int,
    score_provenance: str,
    position_limits: Mapping[tuple[int, int], int] | None = None,
) -> BaselineSelectionResult:
    node_count = len(full_state.blocks)
    if set(scores) != set(range(node_count)):
        raise ValueError("baseline scores must cover every source block exactly once")
    if tuple(sorted(order)) != tuple(range(node_count)):
        raise ValueError("baseline candidate order must contain every source node exactly once")
    if any(not math.isfinite(float(score)) for score in scores.values()):
        raise ValueError("baseline scores must be finite")
    costs = tuple(budget.cost(block) for block in full_state.blocks)
    mandatory = tuple(node_id for node_id, block in enumerate(full_state.blocks) if block.mandatory)
    spent = sum(costs[node_id] for node_id in mandatory)
    if spent > budget.value:
        raise InfeasibleBaselineBudgetError("mandatory exact blocks exceed baseline budget")
    selection_order = list(mandatory)
    selected = set(mandatory)
    selected_positions: dict[tuple[int, int], int] = defaultdict(int)
    for node_id in mandatory:
        block = full_state.blocks[node_id]
        identity = (block.layer, block.kv_head)
        selected_positions[identity] += block.position_count
    if position_limits is not None:
        for identity, count in selected_positions.items():
            if count > position_limits.get(identity, 0):
                raise InfeasibleBaselineBudgetError(
                    "mandatory exact blocks exceed the dense cache capacity: "
                    f"layer={identity[0]}, kv_head={identity[1]}, "
                    f"required={count}, limit={position_limits.get(identity, 0)}"
                )
    reason_by_id = {node_id: BaselineSelectionReason.MANDATORY_EXACT for node_id in mandatory}
    capacity_rejected: set[int] = set()
    candidate_reason = (
        BaselineSelectionReason.RETENTION_ONE_EXACT
        if budget.value == _source_cost(full_state, budget.unit)
        else selected_reason
    )
    for node_id in order:
        if node_id in selected:
            continue
        block = full_state.blocks[node_id]
        identity = (block.layer, block.kv_head)
        limit = None if position_limits is None else position_limits.get(identity)
        if limit is not None and selected_positions[identity] + block.position_count > limit:
            capacity_rejected.add(node_id)
            continue
        cost = costs[node_id]
        if cost <= budget.value - spent:
            selected.add(node_id)
            selection_order.append(node_id)
            spent += cost
            selected_positions[identity] += block.position_count
            reason_by_id[node_id] = candidate_reason
    rank_by_id = {node_id: rank for rank, node_id in enumerate(selection_order)}
    decisions = tuple(
        BaselineSelectionDecision(
            node_id=node_id,
            selected=node_id in selected,
            selection_rank=rank_by_id.get(node_id),
            score=float(scores[node_id]),
            cost=costs[node_id],
            reason=reason_by_id.get(
                node_id,
                (
                    BaselineSelectionReason.DENSE_CACHE_CAPACITY
                    if node_id in capacity_rejected
                    else BaselineSelectionReason.BUDGET_EXHAUSTED
                ),
            ),
            stratum=_stratum(block),
        )
        for node_id, block in enumerate(full_state.blocks)
    )
    selected_ids = tuple(sorted(selected))
    selected_blocks = tuple(full_state.blocks[node_id] for node_id in selected_ids)
    return BaselineSelectionResult(
        method=method,
        source_blocks=full_state.blocks,
        selected_blocks=selected_blocks,
        selection_order=tuple(selection_order),
        decisions=decisions,
        budget=budget,
        budget_spent=spent,
        active_bytes=sum(block.byte_size for block in selected_blocks),
        seed=seed,
        score_provenance=score_provenance,
    )


def _uniform_candidate_order(
    node_ids: Sequence[int],
    mandatory: frozenset[int],
) -> tuple[int, ...]:
    """Return a deterministic farthest-point traversal over logical order."""

    ordered = tuple(sorted(node_ids))
    index_by_id = {node_id: index for index, node_id in enumerate(ordered)}
    anchors = {float(index_by_id[node_id]) for node_id in mandatory if node_id in index_by_id}
    anchors.update({-0.5, len(ordered) - 0.5})
    remaining = set(ordered) - mandatory
    result: list[int] = []
    while remaining:
        node_id = min(
            remaining,
            key=lambda candidate: (
                -min(abs(index_by_id[candidate] - anchor) for anchor in anchors),
                candidate,
            ),
        )
        result.append(node_id)
        anchors.add(float(index_by_id[node_id]))
        remaining.remove(node_id)
    return tuple(result)


def _select_uniform(
    full_state: FullKVState,
    budget: SelectionBudget,
    *,
    seed: int,
    position_limits: Mapping[tuple[int, int], int] | None = None,
) -> BaselineSelectionResult:
    groups: dict[BaselineStratum, list[int]] = defaultdict(list)
    for node_id, block in enumerate(full_state.blocks):
        groups[_stratum(block)].append(node_id)
    costs = tuple(budget.cost(block) for block in full_state.blocks)
    mandatory = frozenset(
        node_id for node_id, block in enumerate(full_state.blocks) if block.mandatory
    )
    spent = sum(costs[node_id] for node_id in mandatory)
    if spent > budget.value:
        raise InfeasibleBaselineBudgetError("mandatory exact blocks exceed baseline budget")
    source_cost = {
        group: sum(costs[node_id] for node_id in node_ids) for group, node_ids in groups.items()
    }
    selected_cost = {
        group: sum(costs[node_id] for node_id in node_ids if node_id in mandatory)
        for group, node_ids in groups.items()
    }
    queues = {
        group: list(_uniform_candidate_order(node_ids, mandatory))
        for group, node_ids in groups.items()
    }
    selection_order = list(sorted(mandatory))
    selected = set(mandatory)
    selected_positions: dict[tuple[int, int], int] = defaultdict(int)
    for node_id in mandatory:
        block = full_state.blocks[node_id]
        selected_positions[(block.layer, block.kv_head)] += block.position_count
    if position_limits is not None and any(
        count > position_limits.get(identity, 0) for identity, count in selected_positions.items()
    ):
        raise InfeasibleBaselineBudgetError(
            "mandatory exact blocks exceed the dense cache capacity"
        )
    score_at_selection = {node_id: 1.0 for node_id in mandatory}
    while spent < budget.value:
        remaining = budget.value - spent
        viable = [
            group
            for group, queue in queues.items()
            if any(
                costs[node_id] <= remaining
                and (
                    position_limits is None
                    or selected_positions[
                        (
                            full_state.blocks[node_id].layer,
                            full_state.blocks[node_id].kv_head,
                        )
                    ]
                    + full_state.blocks[node_id].position_count
                    <= position_limits.get(
                        (
                            full_state.blocks[node_id].layer,
                            full_state.blocks[node_id].kv_head,
                        ),
                        0,
                    )
                )
                for node_id in queue
            )
        ]
        if not viable:
            break
        group = min(
            viable,
            key=lambda item: (
                Fraction(selected_cost[item], source_cost[item]),
                item[0],
                item[1],
                item[2].value,
            ),
        )
        queue = queues[group]
        candidate_index = next(
            index
            for index, node_id in enumerate(queue)
            if costs[node_id] <= remaining
            and (
                position_limits is None
                or selected_positions[
                    (
                        full_state.blocks[node_id].layer,
                        full_state.blocks[node_id].kv_head,
                    )
                ]
                + full_state.blocks[node_id].position_count
                <= position_limits.get(
                    (
                        full_state.blocks[node_id].layer,
                        full_state.blocks[node_id].kv_head,
                    ),
                    0,
                )
            )
        )
        node_id = queue.pop(candidate_index)
        fraction_before = selected_cost[group] / source_cost[group]
        selected.add(node_id)
        selection_order.append(node_id)
        spent += costs[node_id]
        selected_cost[group] += costs[node_id]
        block = full_state.blocks[node_id]
        selected_positions[(block.layer, block.kv_head)] += block.position_count
        score_at_selection[node_id] = 1.0 - fraction_before
    scores = {
        node_id: score_at_selection.get(
            node_id,
            1.0 - selected_cost[_stratum(block)] / source_cost[_stratum(block)],
        )
        for node_id, block in enumerate(full_state.blocks)
    }
    replay_order = (
        *selection_order,
        *(node_id for node_id in range(len(full_state.blocks)) if node_id not in selected),
    )
    return _finalize(
        full_state,
        MosaicKVMethod.UNIFORM_KV,
        budget,
        order=replay_order,
        scores=scores,
        selected_reason=BaselineSelectionReason.UNIFORM_STRATUM,
        seed=seed,
        score_provenance="fair_retained_cost_fraction_by_layer_kv_head_modality_v1",
        position_limits=position_limits,
    )


def value_novelty_scores(
    full_state: FullKVState,
    *,
    chunk_size: int = 256,
) -> dict[int, float]:
    """Compute nearest-value cosine novelty within each layer/KV head."""

    if chunk_size < 1:
        raise ValueError("value novelty chunk_size must be positive")
    descriptors = pool_block_descriptors(full_state)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for descriptor in descriptors:
        groups[(descriptor.block.layer, descriptor.block.kv_head)].append(descriptor.node_id)
    result: dict[int, float] = {}
    for node_ids in groups.values():
        vectors = np.stack(
            [
                np.asarray(descriptors[node_id].pooled_value, dtype=np.float32)
                for node_id in node_ids
            ]
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        if len(node_ids) == 1:
            result[node_ids[0]] = 1.0
            continue
        for start in range(0, len(node_ids), chunk_size):
            stop = min(start + chunk_size, len(node_ids))
            similarities = normalized[start:stop] @ normalized.T
            rows = np.arange(stop - start)
            similarities[rows, np.arange(start, stop)] = -np.inf
            maximum = np.maximum(0.0, np.max(similarities, axis=1))
            novelty = 1.0 - np.minimum(1.0, maximum)
            for local_index, value in enumerate(novelty):
                result[node_ids[start + local_index]] = float(value)
    if set(result) != set(range(len(full_state.blocks))):
        raise RuntimeError("value novelty scores do not cover every source block")
    return result


def select_exact_baseline(
    full_state: FullKVState,
    method: MosaicKVMethod,
    budget: SelectionBudget,
    *,
    seed: int,
    prompt_attention_by_node: Mapping[int, float] | None = None,
    value_chunk_size: int = 256,
    position_limits: Mapping[tuple[int, int], int] | None = None,
) -> BaselineSelectionResult:
    """Select one exact-only simple baseline under a shared hard budget."""

    if seed < 0:
        raise ValueError("baseline seed must be nonnegative")
    node_ids = list(range(len(full_state.blocks)))
    if method is MosaicKVMethod.RANDOM_KV:
        random.Random(seed).shuffle(node_ids)
        scores = {node_id: float(len(node_ids) - rank) for rank, node_id in enumerate(node_ids)}
        return _finalize(
            full_state,
            method,
            budget,
            order=node_ids,
            scores=scores,
            selected_reason=BaselineSelectionReason.RANDOM_SEEDED,
            seed=seed,
            score_provenance=f"python_mt19937_shuffle_v1:seed={seed}",
            position_limits=position_limits,
        )
    if method is MosaicKVMethod.UNIFORM_KV:
        return _select_uniform(
            full_state,
            budget,
            seed=seed,
            position_limits=position_limits,
        )
    if method is MosaicKVMethod.PROMPT_ATTENTION_TOPK:
        if prompt_attention_by_node is None:
            raise ValueError("prompt_attention_topk requires per-node prompt attention mass")
        scores = {int(node_id): float(value) for node_id, value in prompt_attention_by_node.items()}
        if set(scores) != set(node_ids):
            raise ValueError(
                "prompt_attention_topk scores must cover every source block exactly once"
            )
        order = sorted(node_ids, key=lambda node_id: (-scores[node_id], node_id))
        return _finalize(
            full_state,
            method,
            budget,
            order=order,
            scores=scores,
            selected_reason=BaselineSelectionReason.PROMPT_ATTENTION_TOPK,
            seed=seed,
            score_provenance="eager_prompt_window_attention_mass_v1",
            position_limits=position_limits,
        )
    if method is MosaicKVMethod.VALUE_TOPK:
        scores = value_novelty_scores(full_state, chunk_size=value_chunk_size)
        order = sorted(node_ids, key=lambda node_id: (-scores[node_id], node_id))
        return _finalize(
            full_state,
            method,
            budget,
            order=order,
            scores=scores,
            selected_reason=BaselineSelectionReason.VALUE_NOVELTY_TOPK,
            seed=seed,
            score_provenance="nearest_value_cosine_novelty_same_layer_kv_head_v1",
            position_limits=position_limits,
        )
    raise ValueError(f"method is not an exact compressed baseline: {method.value}")


def select_prompt_attention_topk_ids(
    full_state: FullKVState,
    budget: SelectionBudget,
    *,
    prompt_attention_by_node: Mapping[int, float],
    position_limits: Mapping[tuple[int, int], int] | None = None,
) -> tuple[int, ...]:
    """Return only Prompt Top-K selected IDs under the canonical policy.

    Stable-head MosaicKV routing needs the baseline's selected set, not its
    40,000-element audit table.  This narrow path deliberately mirrors
    :func:`_finalize`: mandatory blocks are installed first, optional blocks
    use descending prompt mass with node-ID tie-breaking, and both the common
    hard budget and per-head dense capacities are enforced.  The ordinary
    baseline path remains unchanged and continues to emit every decision.
    """

    node_count = len(full_state.blocks)
    node_ids = tuple(range(node_count))
    scores = {int(node_id): float(value) for node_id, value in prompt_attention_by_node.items()}
    if set(scores) != set(node_ids):
        raise ValueError("prompt_attention_topk scores must cover every source block exactly once")
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("prompt_attention_topk scores must be finite")
    costs = tuple(budget.cost(block) for block in full_state.blocks)
    mandatory = tuple(node_id for node_id, block in enumerate(full_state.blocks) if block.mandatory)
    spent = sum(costs[node_id] for node_id in mandatory)
    if spent > budget.value:
        raise InfeasibleBaselineBudgetError("mandatory exact blocks exceed baseline budget")
    selected = set(mandatory)
    selected_positions: dict[tuple[int, int], int] = defaultdict(int)
    for node_id in mandatory:
        block = full_state.blocks[node_id]
        identity = (block.layer, block.kv_head)
        selected_positions[identity] += block.position_count
    if position_limits is not None:
        for identity, count in selected_positions.items():
            if count > position_limits.get(identity, 0):
                raise InfeasibleBaselineBudgetError(
                    "mandatory exact blocks exceed the dense cache capacity: "
                    f"layer={identity[0]}, kv_head={identity[1]}, "
                    f"required={count}, limit={position_limits.get(identity, 0)}"
                )
    for node_id in sorted(node_ids, key=lambda candidate: (-scores[candidate], candidate)):
        if node_id in selected:
            continue
        block = full_state.blocks[node_id]
        identity = (block.layer, block.kv_head)
        limit = None if position_limits is None else position_limits.get(identity)
        if limit is not None and selected_positions[identity] + block.position_count > limit:
            continue
        cost = costs[node_id]
        if cost <= budget.value - spent:
            selected.add(node_id)
            spent += cost
            selected_positions[identity] += block.position_count
    return tuple(sorted(selected))


def _adaptive_visual_swap_selection(
    full_state: FullKVState,
    *,
    route_node_ids: Sequence[int],
    mandatory_node_ids: Sequence[int],
    selected_optional_node_ids: Sequence[int],
    score_array: np.ndarray[Any, np.dtype[np.float64]],
    forecast_score_array: np.ndarray[Any, np.dtype[np.float64]] | None,
    minimum_text_mass_fraction: float,
    visual_gain_ratio: float,
    maximum_visual_swaps: int,
    swap_risk_gate: bool,
    maximum_normalized_text_loss: float,
    forecast_gain_ratio: float,
) -> _AdaptiveSwapResult:
    """Exchange equal-size blocks only when prompt and forecast evidence justify risk."""

    selected_optional = set(int(node_id) for node_id in selected_optional_node_ids)
    mandatory = frozenset(int(node_id) for node_id in mandatory_node_ids)
    selected_all = mandatory | selected_optional
    text_nodes = tuple(
        node_id
        for node_id in route_node_ids
        if full_state.blocks[node_id].modality is Modality.TEXT
    )
    visual_nodes = tuple(
        node_id
        for node_id in route_node_ids
        if full_state.blocks[node_id].modality in {Modality.IMAGE, Modality.VIDEO}
    )
    total_text_mass = math.fsum(float(score_array[node_id]) for node_id in text_nodes)
    total_visual_mass = math.fsum(float(score_array[node_id]) for node_id in visual_nodes)
    baseline_text_mass = math.fsum(
        float(score_array[node_id]) for node_id in selected_all if node_id in text_nodes
    )
    minimum_text_mass = baseline_text_mass * minimum_text_mass_fraction
    if (
        maximum_visual_swaps == 0
        or total_text_mass <= 0
        or total_visual_mass <= 0
        or baseline_text_mass <= 0
    ):
        return _AdaptiveSwapResult(
            tuple(sorted(selected_optional)),
            0,
            baseline_text_mass,
            baseline_text_mass,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            (),
        )
    if swap_risk_gate and forecast_score_array is None:
        raise ValueError("swap risk gating requires forecast attention scores")
    total_forecast_text_mass = (
        0.0
        if forecast_score_array is None
        else math.fsum(float(forecast_score_array[node_id]) for node_id in text_nodes)
    )
    total_forecast_visual_mass = (
        0.0
        if forecast_score_array is None
        else math.fsum(float(forecast_score_array[node_id]) for node_id in visual_nodes)
    )

    selected_text_by_size: dict[int, list[int]] = defaultdict(list)
    unselected_visual_by_size: dict[int, list[int]] = defaultdict(list)
    for node_id in selected_optional:
        block = full_state.blocks[node_id]
        if block.modality is Modality.TEXT:
            selected_text_by_size[block.position_count].append(node_id)
    for node_id in visual_nodes:
        if node_id not in selected_all:
            block = full_state.blocks[node_id]
            unselected_visual_by_size[block.position_count].append(node_id)

    swaps: list[tuple[float, float, float, float, float, float, int, int]] = []
    for size in sorted(set(selected_text_by_size).intersection(unselected_visual_by_size)):
        text_donors = sorted(
            selected_text_by_size[size],
            key=lambda node_id: (float(score_array[node_id]), -node_id),
        )
        visual_recipients = sorted(
            unselected_visual_by_size[size],
            key=lambda node_id: (-float(score_array[node_id]), node_id),
        )
        for text_node_id, visual_node_id in zip(
            text_donors,
            visual_recipients,
            strict=False,
        ):
            normalized_loss = float(score_array[text_node_id]) / total_text_mass
            normalized_gain = float(score_array[visual_node_id]) / total_visual_mass
            forecast_loss = (
                0.0
                if forecast_score_array is None or total_forecast_text_mass <= 0
                else float(forecast_score_array[text_node_id]) / total_forecast_text_mass
            )
            forecast_gain = (
                0.0
                if forecast_score_array is None or total_forecast_visual_mass <= 0
                else float(forecast_score_array[visual_node_id]) / total_forecast_visual_mass
            )
            swaps.append(
                (
                    normalized_gain - visual_gain_ratio * normalized_loss,
                    normalized_gain,
                    normalized_loss,
                    forecast_gain - forecast_gain_ratio * forecast_loss,
                    forecast_gain,
                    forecast_loss,
                    visual_node_id,
                    text_node_id,
                )
            )

    selected_text_mass = baseline_text_mass
    normalized_visual_gain = 0.0
    normalized_text_loss = 0.0
    normalized_forecast_gain = 0.0
    normalized_forecast_loss = 0.0
    swap_count = 0
    candidate_count = 0
    risk_rejected_count = 0
    accepted_pairs: list[tuple[int, int]] = []
    for (
        surplus,
        gain,
        loss,
        forecast_surplus,
        forecast_gain,
        forecast_loss,
        visual_node_id,
        text_node_id,
    ) in sorted(
        swaps,
        key=lambda item: (
            -item[0],
            -item[3] if swap_risk_gate else 0.0,
            -item[1],
            item[2],
            item[6],
            item[7],
        ),
    ):
        if surplus < 0:
            break
        candidate_count += 1
        if swap_risk_gate and (
            loss > maximum_normalized_text_loss
            or total_forecast_text_mass <= 0
            or total_forecast_visual_mass <= 0
            or forecast_surplus < 0
        ):
            risk_rejected_count += 1
            continue
        if swap_count >= maximum_visual_swaps:
            continue
        raw_text_loss = float(score_array[text_node_id])
        if selected_text_mass - raw_text_loss + 1e-12 < minimum_text_mass:
            continue
        if text_node_id not in selected_optional or visual_node_id in selected_optional:
            continue
        selected_optional.remove(text_node_id)
        selected_optional.add(visual_node_id)
        selected_text_mass -= raw_text_loss
        normalized_visual_gain += gain
        normalized_text_loss += loss
        normalized_forecast_gain += forecast_gain
        normalized_forecast_loss += forecast_loss
        accepted_pairs.append((text_node_id, visual_node_id))
        swap_count += 1

    return _AdaptiveSwapResult(
        tuple(sorted(selected_optional)),
        swap_count,
        baseline_text_mass,
        selected_text_mass,
        normalized_visual_gain,
        normalized_text_loss,
        normalized_forecast_gain,
        normalized_forecast_loss,
        candidate_count,
        risk_rejected_count,
        tuple(accepted_pairs),
    )


def _future_query_rerank_selection(
    full_state: FullKVState,
    *,
    route_node_ids: Sequence[int],
    mandatory_node_ids: Sequence[int],
    selected_optional_node_ids: Sequence[int],
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
    forecast_score_array: np.ndarray[Any, np.dtype[np.float64]],
    minimum_prompt_mass_fraction: float,
    future_gain_ratio: float,
    maximum_replacements: int,
    minimum_prompt_forecast_tv: float,
    risk_gate: bool,
    maximum_normalized_prompt_loss: float,
    minimum_normalized_future_gain: float,
) -> _FutureRerankResult:
    """Rerank stable exact blocks without changing their byte footprint.

    Prompt Top-K is the immutable anchor.  Only equal-position-count optional
    blocks may exchange places, so dense capacity and active KV bytes cannot
    change.  A replacement must increase forecasted future-query attention
    and preserve the configured fraction of the anchor's selected prompt
    mass.  The gain/loss test is normalized within the head to make the
    threshold comparable across layers and heads.
    """

    selected_optional = set(int(node_id) for node_id in selected_optional_node_ids)
    mandatory = frozenset(int(node_id) for node_id in mandatory_node_ids)
    selected_all = mandatory | selected_optional
    route_nodes = tuple(int(node_id) for node_id in route_node_ids)
    baseline_prompt_mass = math.fsum(float(prompt_score_array[node_id]) for node_id in selected_all)
    total_prompt_mass = math.fsum(float(prompt_score_array[node_id]) for node_id in route_nodes)
    total_forecast_mass = math.fsum(float(forecast_score_array[node_id]) for node_id in route_nodes)
    epsilon = float(np.finfo(np.float64).tiny)
    prompt_distribution = np.asarray(
        [float(prompt_score_array[node_id]) for node_id in route_nodes],
        dtype=np.float64,
    ) / max(total_prompt_mass, epsilon)
    forecast_distribution = np.asarray(
        [float(forecast_score_array[node_id]) for node_id in route_nodes],
        dtype=np.float64,
    ) / max(total_forecast_mass, epsilon)
    prompt_forecast_tv = 0.5 * float(np.abs(prompt_distribution - forecast_distribution).sum())
    divergence_eligible = bool(
        total_prompt_mass > 0
        and total_forecast_mass > 0
        and prompt_forecast_tv + 1e-12 >= minimum_prompt_forecast_tv
    )
    if (
        maximum_replacements == 0
        or not selected_optional
        or total_forecast_mass <= 0
        or baseline_prompt_mass <= 0
        or not divergence_eligible
    ):
        return _FutureRerankResult(
            tuple(sorted(selected_optional)),
            0,
            baseline_prompt_mass,
            baseline_prompt_mass,
            0.0,
            0.0,
            0,
            0,
            0,
            (),
            prompt_forecast_tv,
            divergence_eligible,
        )

    selected_by_size: dict[int, list[int]] = defaultdict(list)
    unselected_by_size: dict[int, list[int]] = defaultdict(list)
    for node_id in selected_optional:
        selected_by_size[full_state.blocks[node_id].position_count].append(node_id)
    for node_id in route_nodes:
        if node_id not in selected_all:
            unselected_by_size[full_state.blocks[node_id].position_count].append(node_id)

    candidates: list[tuple[float, float, float, int, int]] = []
    for size in sorted(set(selected_by_size).intersection(unselected_by_size)):
        donors = sorted(
            selected_by_size[size],
            key=lambda node_id: (
                float(forecast_score_array[node_id]),
                -float(prompt_score_array[node_id]),
                -node_id,
            ),
        )
        recipients = sorted(
            unselected_by_size[size],
            key=lambda node_id: (
                -float(forecast_score_array[node_id]),
                -float(prompt_score_array[node_id]),
                node_id,
            ),
        )
        for donor, recipient in zip(donors, recipients, strict=False):
            future_gain = (
                float(forecast_score_array[recipient]) - float(forecast_score_array[donor])
            ) / total_forecast_mass
            prompt_loss = max(
                0.0,
                float(prompt_score_array[donor]) - float(prompt_score_array[recipient]),
            ) / max(total_prompt_mass, float(np.finfo(np.float64).tiny))
            candidates.append(
                (
                    future_gain - future_gain_ratio * prompt_loss,
                    future_gain,
                    prompt_loss,
                    donor,
                    recipient,
                )
            )

    selected_prompt_mass = baseline_prompt_mass
    minimum_prompt_mass = baseline_prompt_mass * minimum_prompt_mass_fraction
    normalized_future_gain = 0.0
    normalized_prompt_loss = 0.0
    candidate_count = 0
    rejected_count = 0
    risk_rejected_count = 0
    accepted_pairs: list[tuple[int, int]] = []
    for surplus, future_gain, prompt_loss, donor, recipient in sorted(
        candidates,
        key=lambda item: (-item[0], -item[1], item[2], item[3], item[4]),
    ):
        if len(accepted_pairs) >= maximum_replacements:
            break
        candidate_count += 1
        if risk_gate and (
            prompt_loss > maximum_normalized_prompt_loss + 1e-12
            or future_gain + 1e-12 < minimum_normalized_future_gain
        ):
            rejected_count += 1
            risk_rejected_count += 1
            continue
        next_prompt_mass = (
            selected_prompt_mass
            - float(prompt_score_array[donor])
            + float(prompt_score_array[recipient])
        )
        if (
            surplus <= 0
            or future_gain <= 0
            or next_prompt_mass + 1e-12 < minimum_prompt_mass
            or donor not in selected_optional
            or recipient in selected_optional
        ):
            rejected_count += 1
            continue
        selected_optional.remove(donor)
        selected_optional.add(recipient)
        selected_prompt_mass = next_prompt_mass
        normalized_future_gain += future_gain
        normalized_prompt_loss += prompt_loss
        accepted_pairs.append((donor, recipient))

    return _FutureRerankResult(
        tuple(sorted(selected_optional)),
        len(accepted_pairs),
        baseline_prompt_mass,
        selected_prompt_mass,
        normalized_future_gain,
        normalized_prompt_loss,
        candidate_count,
        rejected_count,
        risk_rejected_count,
        tuple(accepted_pairs),
        prompt_forecast_tv,
        divergence_eligible,
    )


def _prompt_query_coverage_rerank_selection(
    full_state: FullKVState,
    *,
    route_node_ids: Sequence[int],
    mandatory_node_ids: Sequence[int],
    selected_optional_node_ids: Sequence[int],
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
    query_attention_by_node: Mapping[int, np.ndarray[Any, np.dtype[np.float64]]],
    minimum_prompt_mass_fraction: float,
    maximum_replacements: int,
    minimum_normalized_gain: float,
) -> _QueryCoverageRerankResult:
    """Protect evidence that is important to one prompt query but diluted by averaging.

    Stable Prompt Top-K is the anchor.  The facility-style coverage score is
    the sum, over prompt query positions, of the largest selected block mass.
    Replacements are optional-to-optional, equal-size, and same-modality, so
    the physical cache footprint and modality allocation are unchanged.  The
    scalar Prompt Top-K mass floor prevents query coverage from buying a large
    loss in the ordinary aggregate attention signal.
    """

    selected_optional = set(int(node_id) for node_id in selected_optional_node_ids)
    mandatory = frozenset(int(node_id) for node_id in mandatory_node_ids)
    route_nodes = tuple(sorted(int(node_id) for node_id in route_node_ids))
    selected_all = mandatory | selected_optional
    baseline_prompt_mass = math.fsum(
        float(prompt_score_array[node_id]) for node_id in selected_optional
    )
    total_prompt_mass = math.fsum(float(prompt_score_array[node_id]) for node_id in route_nodes)
    epsilon = float(np.finfo(np.float64).tiny)

    def coverage_vector(
        node_ids: Sequence[int] | set[int] | frozenset[int],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        ordered = tuple(sorted(node_ids))
        if not ordered:
            first = query_attention_by_node[route_nodes[0]]
            return np.zeros_like(first, dtype=np.float64)
        return cast(
            "np.ndarray[Any, np.dtype[np.float64]]",
            np.max(
                np.stack([query_attention_by_node[node_id] for node_id in ordered], axis=0),
                axis=0,
            ),
        )

    attainable_coverage = float(coverage_vector(route_nodes).sum())
    current_coverage_vector = coverage_vector(selected_all)
    baseline_coverage = float(current_coverage_vector.sum())
    if (
        maximum_replacements == 0
        or not selected_optional
        or baseline_prompt_mass <= 0
        or attainable_coverage <= 0
    ):
        return _QueryCoverageRerankResult(
            tuple(sorted(selected_optional)),
            0,
            baseline_coverage,
            baseline_coverage,
            0.0,
            baseline_prompt_mass,
            baseline_prompt_mass,
            0.0,
            0,
            0,
            (),
        )

    minimum_prompt_mass = baseline_prompt_mass * minimum_prompt_mass_fraction
    selected_prompt_mass = baseline_prompt_mass
    accepted_pairs: list[tuple[int, int]] = []
    candidate_count = 0
    rejected_count = 0

    for _replacement_index in range(maximum_replacements):
        selected_all = mandatory | selected_optional
        unselected = tuple(node_id for node_id in route_nodes if node_id not in selected_all)
        best: tuple[float, float, int, int, np.ndarray[Any, np.dtype[np.float64]]] | None = None
        current_coverage = float(current_coverage_vector.sum())
        for donor in sorted(selected_optional):
            donor_block = full_state.blocks[donor]
            without_donor = coverage_vector(selected_all - {donor})
            recipients = tuple(
                recipient
                for recipient in unselected
                if (
                    full_state.blocks[recipient].position_count == donor_block.position_count
                    and full_state.blocks[recipient].modality is donor_block.modality
                )
            )
            if not recipients:
                continue
            candidate_count += len(recipients)
            recipient_matrix = np.stack(
                [query_attention_by_node[recipient] for recipient in recipients],
                axis=0,
            )
            next_coverage_matrix = np.maximum(without_donor[None, :], recipient_matrix)
            normalized_gains = (next_coverage_matrix.sum(axis=1) - current_coverage) / max(
                attainable_coverage, epsilon
            )
            next_prompt_masses = (
                selected_prompt_mass
                - float(prompt_score_array[donor])
                + prompt_score_array[np.asarray(recipients, dtype=np.intp)]
            )
            feasible = (
                (next_prompt_masses + 1e-12 >= minimum_prompt_mass)
                & (normalized_gains + 1e-12 >= minimum_normalized_gain)
                & (normalized_gains > 0)
            )
            rejected_count += int((~feasible).sum())
            feasible_indices = np.flatnonzero(feasible)
            if not feasible_indices.size:
                continue
            # Larger coverage and prompt mass win. Lower node IDs make exact
            # ties deterministic and independent of mapping order.
            local_index = max(
                (int(index) for index in feasible_indices),
                key=lambda index: (
                    float(normalized_gains[index]),
                    float(next_prompt_masses[index]),
                    -recipients[index],
                ),
            )
            recipient = recipients[local_index]
            proposal = (
                float(normalized_gains[local_index]),
                float(next_prompt_masses[local_index]),
                -donor,
                -recipient,
                next_coverage_matrix[local_index],
            )
            if best is None or proposal[:4] > best[:4]:
                best = proposal
        if best is None:
            break
        _gain, next_prompt_mass, neg_donor, neg_recipient, next_coverage_vector = best
        donor = -neg_donor
        recipient = -neg_recipient
        selected_optional.remove(donor)
        selected_optional.add(recipient)
        selected_prompt_mass = next_prompt_mass
        current_coverage_vector = next_coverage_vector
        accepted_pairs.append((donor, recipient))

    selected_coverage = float(current_coverage_vector.sum())
    return _QueryCoverageRerankResult(
        tuple(sorted(selected_optional)),
        len(accepted_pairs),
        baseline_coverage,
        selected_coverage,
        max(0.0, selected_coverage - baseline_coverage) / max(attainable_coverage, epsilon),
        baseline_prompt_mass,
        selected_prompt_mass,
        max(0.0, baseline_prompt_mass - selected_prompt_mass) / max(total_prompt_mass, epsilon),
        candidate_count,
        rejected_count,
        tuple(accepted_pairs),
    )


def _can_attain_exact_positions(
    full_state: FullKVState,
    node_ids: Sequence[int],
    target_positions: int,
) -> bool:
    """Return whether some subset occupies exactly ``target_positions``."""

    if target_positions < 0:
        return False
    reachable = 1
    mask = (1 << (target_positions + 1)) - 1
    for node_id in node_ids:
        size = full_state.blocks[node_id].position_count
        reachable = (reachable | (reachable << size)) & mask
    return bool((reachable >> target_positions) & 1)


def _best_prompt_subset_for_exact_positions(
    full_state: FullKVState,
    node_ids: Sequence[int],
    *,
    target_positions: int,
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[int, ...]:
    """Maximize scalar prompt mass at one exact route-local capacity."""

    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for node_id in sorted(int(value) for value in node_ids):
        size = full_state.blocks[node_id].position_count
        score = float(prompt_score_array[node_id])
        updated = dict(states)
        for occupied, (prior_score, prior_nodes) in states.items():
            next_occupied = occupied + size
            if next_occupied > target_positions:
                continue
            next_score = prior_score + score
            next_nodes = (*prior_nodes, node_id)
            incumbent = updated.get(next_occupied)
            if (
                incumbent is None
                or next_score > incumbent[0]
                or (next_score == incumbent[0] and next_nodes < incumbent[1])
            ):
                updated[next_occupied] = (next_score, next_nodes)
        states = updated
    if target_positions not in states:
        raise InfeasibleBaselineBudgetError(
            "route-local blocks cannot attain the required exact position capacity"
        )
    return states[target_positions][1]


@dataclass(frozen=True, slots=True)
class _PromptSubsetIntervalExclusionIndex:
    """Exact prompt-mass fill tables for excluding one contiguous node interval."""

    ordered_node_ids: tuple[int, ...]
    node_index: Mapping[int, int]
    prefix: tuple[Mapping[int, tuple[float, tuple[int, ...]]], ...]
    suffix: tuple[Mapping[int, tuple[float, tuple[int, ...]]], ...]
    maximum_positions: int


def _extend_prompt_subset_states(
    states: Mapping[int, tuple[float, tuple[int, ...]]],
    *,
    node_id: int,
    node_size: int,
    node_score: float,
    maximum_positions: int,
    prepend: bool,
) -> dict[int, tuple[float, tuple[int, ...]]]:
    updated = dict(states)
    for occupied, (prior_score, prior_nodes) in states.items():
        next_occupied = occupied + node_size
        if next_occupied > maximum_positions:
            continue
        next_score = prior_score + node_score
        next_nodes = (node_id, *prior_nodes) if prepend else (*prior_nodes, node_id)
        incumbent = updated.get(next_occupied)
        if (
            incumbent is None
            or next_score > incumbent[0]
            or (next_score == incumbent[0] and next_nodes < incumbent[1])
        ):
            updated[next_occupied] = (next_score, next_nodes)
    return updated


def _build_prompt_subset_interval_exclusion_index(
    full_state: FullKVState,
    node_ids: Sequence[int],
    *,
    maximum_positions: int,
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
) -> _PromptSubsetIntervalExclusionIndex:
    ordered = tuple(sorted(int(node_id) for node_id in node_ids))
    prefix: list[Mapping[int, tuple[float, tuple[int, ...]]]] = [{0: (0.0, ())}]
    for node_id in ordered:
        prefix.append(
            _extend_prompt_subset_states(
                prefix[-1],
                node_id=node_id,
                node_size=full_state.blocks[node_id].position_count,
                node_score=float(prompt_score_array[node_id]),
                maximum_positions=maximum_positions,
                prepend=False,
            )
        )
    suffix: list[Mapping[int, tuple[float, tuple[int, ...]]]] = [
        {0: (0.0, ())} for _ in range(len(ordered) + 1)
    ]
    for index in range(len(ordered) - 1, -1, -1):
        node_id = ordered[index]
        suffix[index] = _extend_prompt_subset_states(
            suffix[index + 1],
            node_id=node_id,
            node_size=full_state.blocks[node_id].position_count,
            node_score=float(prompt_score_array[node_id]),
            maximum_positions=maximum_positions,
            prepend=True,
        )
    return _PromptSubsetIntervalExclusionIndex(
        ordered,
        {node_id: index for index, node_id in enumerate(ordered)},
        tuple(prefix),
        tuple(suffix),
        maximum_positions,
    )


def _best_prompt_subset_excluding_group(
    full_state: FullKVState,
    index: _PromptSubsetIntervalExclusionIndex,
    group: Sequence[int],
    *,
    target_positions: int,
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[int, ...] | None:
    """Return the exact best fill outside one group, reusing route-local tables."""

    if not 0 <= target_positions <= index.maximum_positions:
        return None
    group_indices = tuple(sorted(index.node_index[int(node_id)] for node_id in group))
    if not group_indices:
        return None
    if group_indices != tuple(range(group_indices[0], group_indices[-1] + 1)):
        remaining = tuple(node_id for node_id in index.ordered_node_ids if node_id not in group)
        if not _can_attain_exact_positions(full_state, remaining, target_positions):
            return None
        return _best_prompt_subset_for_exact_positions(
            full_state,
            remaining,
            target_positions=target_positions,
            prompt_score_array=prompt_score_array,
        )
    left = index.prefix[group_indices[0]]
    right = index.suffix[group_indices[-1] + 1]
    best: tuple[float, tuple[int, ...]] | None = None
    for left_positions, (left_score, left_nodes) in left.items():
        right_state = right.get(target_positions - left_positions)
        if right_state is None:
            continue
        right_score, right_nodes = right_state
        candidate = (left_score + right_score, (*left_nodes, *right_nodes))
        if (
            best is None
            or candidate[0] > best[0]
            or (candidate[0] == best[0] and candidate[1] < best[1])
        ):
            best = candidate
    return None if best is None else best[1]


@dataclass(frozen=True, slots=True)
class _JointGroupDonorSelection:
    """One exact-capacity group insertion with jointly optimized donors."""

    selected_optional_node_ids: tuple[int, ...]
    donor_node_ids: tuple[int, ...]
    recipient_node_ids: tuple[int, ...]
    output_impact_risk: float


def _joint_group_donor_selection(
    full_state: FullKVState,
    *,
    mandatory_node_ids: frozenset[int],
    scalar_node_ids: frozenset[int],
    group: tuple[int, ...],
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
    query_attention_by_node: Mapping[int, np.ndarray[Any, np.dtype[np.float64]]],
    caps: np.ndarray[Any, np.dtype[np.float64]],
    attainable_coverage: float,
    optional_prompt_mass: float,
    value_by_node: Mapping[int, np.ndarray[Any, np.dtype[np.float64]]],
    prompt_weight: float,
    output_impact_weight: float,
) -> _JointGroupDonorSelection | None:
    """Insert a complete group while selecting donors by query, prompt, and output risk."""

    recipients = tuple(sorted(set(group) - scalar_node_ids))
    if not recipients:
        return None
    target_removed_positions = sum(
        full_state.blocks[node_id].position_count for node_id in recipients
    )
    donor_pool = tuple(sorted(scalar_node_ids - set(group)))
    if not _can_attain_exact_positions(full_state, donor_pool, target_removed_positions):
        return None

    epsilon = float(np.finfo(np.float64).tiny)
    base_with_group = mandatory_node_ids | scalar_node_ids | frozenset(group)
    total_attention = np.stack(
        [query_attention_by_node[node_id] for node_id in sorted(base_with_group)],
        axis=0,
    ).sum(axis=0)
    covered_with_group = np.minimum(caps, total_attention)
    covered_sum = float(covered_with_group.sum())
    donor_metrics: dict[int, tuple[float, float, float, float]] = {}
    for donor in donor_pool:
        query_loss = max(
            0.0,
            covered_sum
            - float(
                np.minimum(
                    caps,
                    total_attention - query_attention_by_node[donor],
                ).sum()
            ),
        ) / max(attainable_coverage, epsilon)
        prompt_loss = max(0.0, float(prompt_score_array[donor])) / max(
            optional_prompt_mass,
            epsilon,
        )
        donor_value = value_by_node[donor]
        donor_norm = float(np.linalg.norm(donor_value))
        value_deviation = min(
            float(np.linalg.norm(donor_value - value_by_node[recipient]))
            / max(
                donor_norm + float(np.linalg.norm(value_by_node[recipient])),
                epsilon,
            )
            for recipient in recipients
        )
        output_risk = prompt_loss * value_deviation
        cost = query_loss + prompt_weight * prompt_loss + output_impact_weight * output_risk
        donor_metrics[donor] = (cost, query_loss, prompt_loss, output_risk)

    states: dict[
        int,
        tuple[float, float, float, float, tuple[int, ...]],
    ] = {0: (0.0, 0.0, 0.0, 0.0, ())}
    for donor in donor_pool:
        size = full_state.blocks[donor].position_count
        cost, query_loss, prompt_loss, output_risk = donor_metrics[donor]
        updated = dict(states)
        for occupied, prior in states.items():
            next_occupied = occupied + size
            if next_occupied > target_removed_positions:
                continue
            candidate = (
                prior[0] + cost,
                prior[1] + output_risk,
                prior[2] + prompt_loss,
                prior[3] + query_loss,
                (*prior[4], donor),
            )
            incumbent = updated.get(next_occupied)
            if incumbent is None or candidate < incumbent:
                updated[next_occupied] = candidate
        states = updated
    best = states.get(target_removed_positions)
    if best is None:  # pragma: no cover - guarded by exact feasibility
        return None
    donors = best[4]
    selected = tuple(sorted((scalar_node_ids | set(group)) - set(donors)))
    return _JointGroupDonorSelection(
        selected,
        donors,
        recipients,
        best[1],
    )


def _prompt_query_coverage_initial_selection(
    full_state: FullKVState,
    *,
    route_node_ids: Sequence[int],
    mandatory_node_ids: Sequence[int],
    scalar_prompt_selected_optional_node_ids: Sequence[int],
    optional_target_positions: int,
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
    query_attention_by_node: Mapping[int, np.ndarray[Any, np.dtype[np.float64]]],
    reservation_fraction: float,
    prompt_weight: float,
    minimum_prompt_mass_fraction: float,
    coverage_objective: str,
    cumulative_cap_fraction: float,
    reservation_target_positions: int | None = None,
    pre_reserved_optional_node_ids: Sequence[int] = (),
) -> _QueryCoverageInitialSelectionResult:
    """Build a feasible selection with query diversity reserved from the start.

    The scalar Prompt Top-K solution defines only the capacity target and an
    auditable comparison point.  Query-diverse blocks are chosen before the
    remaining capacity is filled by prompt mass. ``max`` preserves the V88
    facility objective. ``capped_cumulative`` uses
    ``sum_q min(tau_q, sum_i attention[q, i])`` with each ``tau_q`` fixed from
    the complete route before selection. Each greedy reservation is accepted
    only when the unselected block sizes can still attain the exact target, so
    the coverage objective cannot create padding or alter bytes.
    """

    route_nodes = tuple(sorted(int(node_id) for node_id in route_node_ids))
    mandatory = frozenset(int(node_id) for node_id in mandatory_node_ids)
    optional_nodes = tuple(node_id for node_id in route_nodes if node_id not in mandatory)
    scalar_selected = tuple(
        sorted(int(node_id) for node_id in scalar_prompt_selected_optional_node_ids)
    )
    pre_reserved = tuple(sorted(int(node_id) for node_id in pre_reserved_optional_node_ids))
    if len(pre_reserved) != len(set(pre_reserved)):
        raise ValueError("pre-reserved node IDs must be unique")
    if not set(pre_reserved) <= set(optional_nodes):
        raise ValueError("pre-reserved nodes must be optional route nodes")
    epsilon = float(np.finfo(np.float64).tiny)
    if coverage_objective not in {"max", "capped_cumulative"}:
        raise ValueError("coverage_objective must be 'max' or 'capped_cumulative'")
    if not math.isfinite(cumulative_cap_fraction) or not 0 < cumulative_cap_fraction <= 1:
        raise ValueError("cumulative_cap_fraction must be finite and in (0, 1]")
    if coverage_objective == "max" and cumulative_cap_fraction != 1.0:
        raise ValueError("max query coverage requires cumulative_cap_fraction=1.0")
    route_attention = np.stack(
        [query_attention_by_node[node_id] for node_id in route_nodes],
        axis=0,
    )
    cumulative_caps = (
        cumulative_cap_fraction * route_attention.sum(axis=0)
        if coverage_objective == "capped_cumulative"
        else None
    )

    def coverage_vector(
        node_ids: Sequence[int] | set[int] | frozenset[int],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        ordered = tuple(sorted(node_ids))
        if not ordered:
            return np.zeros_like(query_attention_by_node[route_nodes[0]], dtype=np.float64)
        values = np.stack([query_attention_by_node[node_id] for node_id in ordered], axis=0)
        if coverage_objective == "capped_cumulative":
            if cumulative_caps is None:  # pragma: no cover - guarded by objective
                raise RuntimeError("cumulative query coverage caps are unexpectedly absent")
            return cast(
                "np.ndarray[Any, np.dtype[np.float64]]",
                np.minimum(cumulative_caps, values.sum(axis=0)),
            )
        return cast(
            "np.ndarray[Any, np.dtype[np.float64]]",
            np.max(values, axis=0),
        )

    def add_node_coverage(
        current: np.ndarray[Any, np.dtype[np.float64]],
        node_id: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        if coverage_objective == "capped_cumulative":
            if cumulative_caps is None:  # pragma: no cover - guarded by objective
                raise RuntimeError("cumulative query coverage caps are unexpectedly absent")
            return cast(
                "np.ndarray[Any, np.dtype[np.float64]]",
                np.minimum(cumulative_caps, current + query_attention_by_node[node_id]),
            )
        return cast(
            "np.ndarray[Any, np.dtype[np.float64]]",
            np.maximum(current, query_attention_by_node[node_id]),
        )

    baseline_selected = mandatory | frozenset(scalar_selected)
    baseline_coverage = float(coverage_vector(baseline_selected).sum())
    attainable_coverage = float(coverage_vector(route_nodes).sum())
    baseline_prompt_mass = math.fsum(
        float(prompt_score_array[node_id]) for node_id in scalar_selected
    )
    total_prompt_mass = math.fsum(float(prompt_score_array[node_id]) for node_id in optional_nodes)
    if optional_target_positions == 0 or attainable_coverage <= 0:
        return _QueryCoverageInitialSelectionResult(
            scalar_selected,
            0,
            0,
            baseline_coverage,
            baseline_coverage,
            0.0,
            baseline_prompt_mass,
            baseline_prompt_mass,
            0.0,
            0,
            (),
            (),
        )

    reservation_target = (
        min(
            optional_target_positions,
            max(1, math.ceil(optional_target_positions * reservation_fraction)),
        )
        if reservation_target_positions is None
        else int(reservation_target_positions)
    )
    if not 0 <= reservation_target <= optional_target_positions:
        raise ValueError("query coverage reservation target must fit the optional capacity")
    pre_reserved_positions = sum(
        full_state.blocks[node_id].position_count for node_id in pre_reserved
    )
    if pre_reserved_positions > reservation_target:
        raise ValueError("pre-reserved positions exceed the reservation target")
    if reservation_target == 0:
        return _QueryCoverageInitialSelectionResult(
            scalar_selected,
            0,
            0,
            baseline_coverage,
            baseline_coverage,
            0.0,
            baseline_prompt_mass,
            baseline_prompt_mass,
            0.0,
            0,
            (),
            (),
        )
    reserved = list(pre_reserved)
    remaining = set(optional_nodes) - set(pre_reserved)
    reserved_positions = pre_reserved_positions
    current_coverage_vector = coverage_vector(mandatory | frozenset(pre_reserved))
    candidate_count = 0
    while reserved_positions < reservation_target:
        best: tuple[float, float, float, int] | None = None
        for node_id in sorted(remaining):
            size = full_state.blocks[node_id].position_count
            next_positions = reserved_positions + size
            if next_positions > optional_target_positions:
                continue
            after = tuple(sorted(remaining - {node_id}))
            if not _can_attain_exact_positions(
                full_state,
                after,
                optional_target_positions - next_positions,
            ):
                continue
            candidate_count += 1
            next_coverage = add_node_coverage(current_coverage_vector, node_id)
            marginal = float(next_coverage.sum() - current_coverage_vector.sum()) / max(
                attainable_coverage,
                epsilon,
            )
            prompt_share = float(prompt_score_array[node_id]) / max(total_prompt_mass, epsilon)
            priority = marginal + prompt_weight * prompt_share
            proposal = (priority, marginal, prompt_share, -node_id)
            if best is None or proposal > best:
                best = proposal
        if best is None:
            break
        node_id = -best[3]
        reserved.append(node_id)
        remaining.remove(node_id)
        reserved_positions += full_state.blocks[node_id].position_count
        current_coverage_vector = add_node_coverage(current_coverage_vector, node_id)

    fill = _best_prompt_subset_for_exact_positions(
        full_state,
        tuple(sorted(remaining)),
        target_positions=optional_target_positions - reserved_positions,
        prompt_score_array=prompt_score_array,
    )
    selected = tuple(sorted((*reserved, *fill)))
    selected_prompt_mass = math.fsum(float(prompt_score_array[node_id]) for node_id in selected)
    selected_coverage = float(coverage_vector(mandatory | frozenset(selected)).sum())
    prompt_floor = baseline_prompt_mass * minimum_prompt_mass_fraction
    if (
        selected_coverage <= baseline_coverage + 1e-12
        or selected_prompt_mass + 1e-12 < prompt_floor
    ):
        return _QueryCoverageInitialSelectionResult(
            scalar_selected,
            0,
            0,
            baseline_coverage,
            baseline_coverage,
            0.0,
            baseline_prompt_mass,
            baseline_prompt_mass,
            0.0,
            candidate_count,
            (),
            (),
        )
    added = tuple(sorted(set(selected) - set(scalar_selected)))
    return _QueryCoverageInitialSelectionResult(
        selected,
        len(reserved),
        len(added),
        baseline_coverage,
        selected_coverage,
        (selected_coverage - baseline_coverage) / max(attainable_coverage, epsilon),
        baseline_prompt_mass,
        selected_prompt_mass,
        max(0.0, baseline_prompt_mass - selected_prompt_mass) / max(total_prompt_mass, epsilon),
        candidate_count,
        tuple(sorted(reserved)),
        added,
    )


def _coherent_layout_groups(
    full_state: FullKVState,
    node_ids: Sequence[int],
    *,
    maximum_blocks: int,
    row_bins: int,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic, nonoverlapping spatial reading-order groups.

    Only authoritative normalized block regions are used. Blocks are grouped
    within one layer, KV head, modality, media/page/frame source, and coarse
    row band, then split into contiguous reading-order chunks. Missing spatial
    metadata fails closed: such blocks are not described as OCR/layout
    evidence. The grouping therefore preserves coherent raster evidence
    without inventing OCR boxes or using benchmark labels.
    """

    if maximum_blocks < 2:
        raise ValueError("coherent layout groups require maximum_blocks >= 2")
    if row_bins < 1:
        raise ValueError("coherent layout groups require at least one row bin")
    buckets: dict[
        tuple[int, int, str, int | None, int | None, int | None, int],
        list[int],
    ] = defaultdict(list)
    for node_id in sorted(int(value) for value in node_ids):
        block = full_state.blocks[node_id]
        region = block.region
        if block.modality not in {Modality.IMAGE, Modality.VIDEO} or region is None:
            continue
        if not all(0.0 <= coordinate <= 1.0 for coordinate in region):
            continue
        center_y = (region[1] + region[3]) / 2.0
        row_band = min(row_bins - 1, math.floor(center_y * row_bins))
        buckets[
            (
                block.layer,
                block.kv_head,
                block.modality.value,
                block.image_index,
                block.page_index,
                block.frame_index,
                row_band,
            )
        ].append(node_id)

    groups: list[tuple[int, ...]] = []
    for identity in sorted(buckets, key=repr):
        ordered = sorted(
            buckets[identity],
            key=lambda node_id: (
                full_state.blocks[node_id].original_logical_positions[0],
                node_id,
            ),
        )
        run: list[int] = []

        def flush(current_run: Sequence[int]) -> None:
            for start in range(0, len(current_run), maximum_blocks):
                group = tuple(current_run[start : start + maximum_blocks])
                if len(group) >= 2:
                    groups.append(group)

        for node_id in ordered:
            block = full_state.blocks[node_id]
            if run:
                prior = full_state.blocks[run[-1]]
                if block.original_logical_positions[0] != prior.original_logical_positions[-1] + 1:
                    flush(run)
                    run = []
            run.append(node_id)
        flush(run)
    return tuple(sorted(groups))


def coherent_layout_groups(
    full_state: FullKVState,
    node_ids: Sequence[int],
    *,
    maximum_blocks: int,
    row_bins: int,
) -> tuple[tuple[int, ...], ...]:
    """Public, label-blind coherent-group constructor for runtime policies."""

    return _coherent_layout_groups(
        full_state,
        node_ids,
        maximum_blocks=maximum_blocks,
        row_bins=row_bins,
    )


def _capped_attention_coverage(
    node_ids: Sequence[int] | set[int] | frozenset[int],
    *,
    query_attention_by_node: Mapping[int, np.ndarray[Any, np.dtype[np.float64]]],
    caps: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    ordered = tuple(sorted(node_ids))
    if not ordered:
        return 0.0
    mass = np.stack(
        [query_attention_by_node[node_id] for node_id in ordered],
        axis=0,
    ).sum(axis=0)
    return float(np.minimum(caps, mass).sum())


def _atomic_layout_group_candidates(
    full_state: FullKVState,
    *,
    routes: Sequence[tuple[int, int]],
    node_ids_by_route: Mapping[tuple[int, int], Sequence[int]],
    mandatory_by_route: Mapping[tuple[int, int], set[int]],
    scalar_by_route: Mapping[tuple[int, int], set[int]],
    optional_target_by_route: Mapping[tuple[int, int], int],
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
    query_attention_by_node: Mapping[int, np.ndarray[Any, np.dtype[np.float64]]],
    cumulative_cap_fraction: float,
    prompt_weight: float,
    maximum_blocks: int,
    row_bins: int,
    selection_mode: str,
) -> tuple[
    dict[tuple[int, int], tuple[_AtomicLayoutGroupCandidate, ...]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:
    """Build route-local atomic counterfactuals without labels.

    V95 audits the four highest standalone query/prompt-density groups per
    route. V96 evaluates every feasible group and retains only the label-blind
    frontier that is not dominated in query gain, combined donor risk, and
    byte cost.
    """

    if selection_mode not in {
        "topk_greedy",
        "pareto_knapsack",
        "pareto_joint_donor_knapsack",
    }:
        raise ValueError("atomic layout-group selection mode is unsupported")
    epsilon = float(np.finfo(np.float64).tiny)
    candidates: dict[tuple[int, int], tuple[_AtomicLayoutGroupCandidate, ...]] = {}
    candidate_counts: dict[tuple[int, int], int] = {}
    joint_candidate_counts: dict[tuple[int, int], int] = {identity: 0 for identity in routes}
    joint_value_by_node: dict[int, np.ndarray[Any, np.dtype[np.float64]]] = {}
    if selection_mode == "pareto_joint_donor_knapsack":
        all_route_nodes = tuple(
            sorted({node_id for identity in routes for node_id in node_ids_by_route[identity]})
        )
        descriptors = pool_block_descriptors(
            full_state,
            host_transfer_mode="deferred_layer_vectorized",
            eligible_node_ids=all_route_nodes,
            node_table_ids=all_route_nodes,
        )
        joint_value_by_node = {
            descriptor.source_id: np.asarray(descriptor.pooled_value, dtype=np.float64)
            for descriptor in descriptors
        }
    provisional: list[_AtomicLayoutGroupCandidate] = []
    for identity in routes:
        route_nodes = tuple(sorted(int(node_id) for node_id in node_ids_by_route[identity]))
        mandatory = frozenset(mandatory_by_route[identity])
        scalar = frozenset(scalar_by_route[identity])
        optional_nodes = tuple(node_id for node_id in route_nodes if node_id not in mandatory)
        groups = _coherent_layout_groups(
            full_state,
            optional_nodes,
            maximum_blocks=maximum_blocks,
            row_bins=row_bins,
        )
        candidate_counts[identity] = len(groups)
        if not groups:
            candidates[identity] = ()
            continue
        route_attention = np.stack(
            [query_attention_by_node[node_id] for node_id in route_nodes],
            axis=0,
        )
        caps = cumulative_cap_fraction * route_attention.sum(axis=0)

        attainable = float(caps.sum())
        baseline_coverage = _capped_attention_coverage(
            mandatory | scalar,
            query_attention_by_node=query_attention_by_node,
            caps=caps,
        )
        baseline_prompt_mass = math.fsum(
            max(0.0, float(prompt_score_array[node_id])) for node_id in scalar
        )
        optional_prompt_mass = math.fsum(
            max(0.0, float(prompt_score_array[node_id])) for node_id in optional_nodes
        )
        fill_index = _build_prompt_subset_interval_exclusion_index(
            full_state,
            optional_nodes,
            maximum_positions=optional_target_by_route[identity],
            prompt_score_array=prompt_score_array,
        )
        scored_groups: list[tuple[float, float, float, tuple[int, ...]]] = []
        for group in groups:
            if set(group) <= scalar:
                continue
            group_attention = np.stack(
                [query_attention_by_node[node_id] for node_id in group],
                axis=0,
            ).sum(axis=0)
            query_share = float(np.minimum(caps, group_attention).sum()) / max(
                attainable,
                epsilon,
            )
            prompt_share = math.fsum(
                max(0.0, float(prompt_score_array[node_id])) for node_id in group
            ) / max(optional_prompt_mass, epsilon)
            group_score = query_share + prompt_weight * prompt_share
            positions = sum(full_state.blocks[node_id].position_count for node_id in group)
            scored_groups.append(
                (group_score / max(positions, 1), group_score, prompt_share, group)
            )
        route_candidates: list[_AtomicLayoutGroupCandidate] = []
        ranked_groups = sorted(
            scored_groups,
            key=lambda item: (-item[0], -item[1], -item[2], item[3]),
        )
        if selection_mode == "topk_greedy":
            ranked_groups = ranked_groups[:4]
        for _density, group_score, _prompt_share, group in ranked_groups:
            group_positions = sum(full_state.blocks[node_id].position_count for node_id in group)
            remaining_target = optional_target_by_route[identity] - group_positions
            if remaining_target < 0:
                continue
            joint: _JointGroupDonorSelection | None = None
            if selection_mode == "pareto_joint_donor_knapsack":
                joint = _joint_group_donor_selection(
                    full_state,
                    mandatory_node_ids=mandatory,
                    scalar_node_ids=scalar,
                    group=group,
                    prompt_score_array=prompt_score_array,
                    query_attention_by_node=query_attention_by_node,
                    caps=caps,
                    attainable_coverage=attainable,
                    optional_prompt_mass=optional_prompt_mass,
                    value_by_node=joint_value_by_node,
                    prompt_weight=prompt_weight,
                    output_impact_weight=1.0,
                )
                if joint is None:
                    continue
                selected = joint.selected_optional_node_ids
                joint_candidate_counts[identity] += 1
            else:
                fill = _best_prompt_subset_excluding_group(
                    full_state,
                    fill_index,
                    group,
                    target_positions=remaining_target,
                    prompt_score_array=prompt_score_array,
                )
                if fill is None:
                    continue
                selected = tuple(sorted((*group, *fill)))
            selected_coverage = _capped_attention_coverage(
                mandatory | frozenset(selected),
                query_attention_by_node=query_attention_by_node,
                caps=caps,
            )
            selected_prompt_mass = math.fsum(
                max(0.0, float(prompt_score_array[node_id])) for node_id in selected
            )
            route_candidates.append(
                _AtomicLayoutGroupCandidate(
                    identity=identity,
                    group=group,
                    selected_optional_node_ids=selected,
                    normalized_query_gain=max(0.0, selected_coverage - baseline_coverage)
                    / max(attainable, epsilon),
                    normalized_prompt_risk=max(0.0, baseline_prompt_mass - selected_prompt_mass)
                    / max(optional_prompt_mass, epsilon),
                    output_impact_risk=(0.0 if joint is None else joint.output_impact_risk),
                    group_score=group_score,
                    group_bytes=sum(full_state.blocks[node_id].byte_size for node_id in group),
                    baseline_coverage=baseline_coverage,
                    selected_coverage=selected_coverage,
                    baseline_prompt_mass=baseline_prompt_mass,
                    selected_prompt_mass=selected_prompt_mass,
                    donor_node_ids=(
                        tuple(sorted(scalar - set(selected)))
                        if joint is None
                        else joint.donor_node_ids
                    ),
                    recipient_node_ids=(
                        tuple(sorted(set(selected) - scalar))
                        if joint is None
                        else joint.recipient_node_ids
                    ),
                )
            )
        candidates[identity] = tuple(route_candidates)
        provisional.extend(route_candidates)

    changed = tuple(
        sorted(
            {
                node_id
                for candidate in provisional
                for node_id in (*candidate.donor_node_ids, *candidate.recipient_node_ids)
            }
        )
    )
    if not changed:
        return (
            candidates,
            candidate_counts,
            {identity: 0 for identity in routes},
            joint_candidate_counts,
        )
    if joint_value_by_node:
        value_by_node = joint_value_by_node
    else:
        descriptors = pool_block_descriptors(
            full_state,
            host_transfer_mode="deferred_layer_vectorized",
            eligible_node_ids=changed,
            node_table_ids=changed,
        )
        value_by_node = {
            descriptor.source_id: np.asarray(descriptor.pooled_value, dtype=np.float64)
            for descriptor in descriptors
        }
    audited: dict[tuple[int, int], tuple[_AtomicLayoutGroupCandidate, ...]] = {}
    for identity in routes:
        optional_prompt_mass = math.fsum(
            max(0.0, float(prompt_score_array[node_id]))
            for node_id in node_ids_by_route[identity]
            if node_id not in mandatory_by_route[identity]
        )
        route_audited: list[_AtomicLayoutGroupCandidate] = []
        for candidate in candidates[identity]:
            if candidate.donor_node_ids and not candidate.recipient_node_ids:
                output_risk = 1.0
            else:
                output_risk = 0.0
                for donor in candidate.donor_node_ids:
                    donor_value = value_by_node[donor]
                    donor_norm = float(np.linalg.norm(donor_value))
                    closest_deviation = min(
                        float(np.linalg.norm(donor_value - value_by_node[recipient]))
                        / max(
                            donor_norm + float(np.linalg.norm(value_by_node[recipient])),
                            epsilon,
                        )
                        for recipient in candidate.recipient_node_ids
                    )
                    prompt_share = max(0.0, float(prompt_score_array[donor])) / max(
                        optional_prompt_mass,
                        epsilon,
                    )
                    output_risk += prompt_share * closest_deviation
            route_audited.append(replace(candidate, output_impact_risk=output_risk))
        audited[identity] = (
            _gain_risk_pareto_frontier(route_audited)
            if selection_mode in {"pareto_knapsack", "pareto_joint_donor_knapsack"}
            else tuple(route_audited)
        )
    pareto_counts = {
        identity: (
            len(audited[identity])
            if selection_mode in {"pareto_knapsack", "pareto_joint_donor_knapsack"}
            else 0
        )
        for identity in routes
    }
    return audited, candidate_counts, pareto_counts, joint_candidate_counts


def _gain_risk_pareto_frontier(
    candidates: Sequence[_AtomicLayoutGroupCandidate],
) -> tuple[_AtomicLayoutGroupCandidate, ...]:
    """Return candidates not dominated in gain, combined risk, and bytes."""

    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.group))
    frontier: list[_AtomicLayoutGroupCandidate] = []
    for candidate in ordered:
        risk = max(candidate.normalized_prompt_risk, candidate.output_impact_risk)
        dominated = False
        for other in ordered:
            if other is candidate:
                continue
            other_risk = max(other.normalized_prompt_risk, other.output_impact_risk)
            weakly_better = bool(
                other.normalized_query_gain + 1e-12 >= candidate.normalized_query_gain
                and other_risk <= risk + 1e-12
                and other.group_bytes <= candidate.group_bytes
            )
            strictly_better = bool(
                other.normalized_query_gain > candidate.normalized_query_gain + 1e-12
                or other_risk + 1e-12 < risk
                or other.group_bytes < candidate.group_bytes
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)


def _atomic_group_multiple_choice_knapsack(
    accepted_by_route: Mapping[tuple[int, int], Sequence[_AtomicLayoutGroupCandidate]],
    *,
    byte_budget: int,
    gain_to_risk_ratio: float,
    defer_surrogate_risk_to_logit: bool = False,
) -> tuple[_AtomicLayoutGroupCandidate, ...]:
    """Choose at most one complete group per route under one byte budget."""

    if byte_budget < 0:
        raise ValueError("atomic group knapsack byte budget must be nonnegative")
    states: dict[
        int,
        tuple[float, tuple[_AtomicLayoutGroupCandidate, ...]],
    ] = {0: (0.0, ())}

    def signature(
        choices: Sequence[_AtomicLayoutGroupCandidate],
    ) -> tuple[tuple[tuple[int, int], tuple[int, ...]], ...]:
        return tuple((choice.identity, choice.group) for choice in choices)

    for identity in sorted(accepted_by_route):
        options = tuple(sorted(accepted_by_route[identity], key=lambda item: item.group))
        updated = dict(states)
        for occupied, (utility, choices) in states.items():
            for candidate in options:
                next_occupied = occupied + candidate.group_bytes
                if next_occupied > byte_budget:
                    continue
                risk = max(
                    candidate.normalized_prompt_risk,
                    candidate.output_impact_risk,
                )
                next_utility = utility + (
                    candidate.normalized_query_gain
                    if defer_surrogate_risk_to_logit
                    else candidate.normalized_query_gain - gain_to_risk_ratio * risk
                )
                next_choices = (*choices, candidate)
                incumbent = updated.get(next_occupied)
                if (
                    incumbent is None
                    or next_utility > incumbent[0] + 1e-12
                    or (
                        abs(next_utility - incumbent[0]) <= 1e-12
                        and signature(next_choices) < signature(incumbent[1])
                    )
                ):
                    updated[next_occupied] = (next_utility, next_choices)
        states = updated
    _occupied, (_utility, selected) = min(
        states.items(),
        key=lambda item: (-item[1][0], -item[0], signature(item[1][1])),
    )
    return selected


def _cross_head_capped_query_initial_selection(
    full_state: FullKVState,
    *,
    head_routes: Sequence[tuple[int, int]],
    scalar_selection: HeadLocalPromptSelection,
    prompt_score_array: np.ndarray[Any, np.dtype[np.float64]],
    query_attention_by_node: Mapping[int, np.ndarray[Any, np.dtype[np.float64]]],
    reservation_fraction: float,
    prompt_weight: float,
    minimum_prompt_mass_fraction: float,
    cumulative_cap_fraction: float,
    stability_gate: bool,
    stability_risk_mode: str,
    minimum_normalized_gain: float,
    gain_to_prompt_loss_ratio: float,
    output_impact_reservation: bool,
    output_impact_reservation_fraction: float,
    layout_group_reservation: bool,
    layout_group_reservation_fraction: float,
    layout_group_maximum_blocks: int,
    layout_group_row_bins: int,
    layout_group_atomic_selection: bool,
    layout_group_atomic_selection_mode: str,
    layout_group_minimum_normalized_gain: float,
    layout_group_gain_to_risk_ratio: float,
    defer_surrogate_stability_to_logit: bool,
) -> dict[tuple[int, int], _QueryCoverageInitialSelectionResult]:
    """Allocate one query-conditioned exact-byte reservation across KV heads.

    Dense Hugging Face tensors retain their shared physical sequence extent.
    This allocator therefore redistributes the byte budget *inside* that
    extent: capped query mass globally decides which exact blocks are reserved
    before each route's remaining exact slots are filled by scalar Prompt
    Top-K.  A high-risk route may consume more of the global query-conditioned
    byte reservation than a low-risk route, but no route can exceed its
    backend-safe physical position limit.
    """

    routes = tuple(sorted((int(layer), int(head)) for layer, head in head_routes))
    route_set = frozenset(routes)
    record_by_route = {
        (record.layer, record.kv_head): record for record in scalar_selection.capacity_records
    }
    if set(record_by_route) != set(routes):
        raise ValueError("scalar cross-head reference does not cover every requested route")
    node_ids_by_route: dict[tuple[int, int], list[int]] = defaultdict(list)
    mandatory_by_route: dict[tuple[int, int], set[int]] = defaultdict(set)
    scalar_by_route: dict[tuple[int, int], set[int]] = defaultdict(set)
    scalar_ids = frozenset(int(node_id) for node_id in scalar_selection.selected_node_ids)
    for node_id, block in enumerate(full_state.blocks):
        identity = (block.layer, block.kv_head)
        if identity not in route_set:
            continue
        node_ids_by_route[identity].append(node_id)
        if block.mandatory:
            mandatory_by_route[identity].add(node_id)
        elif node_id in scalar_ids:
            scalar_by_route[identity].add(node_id)

    epsilon = float(np.finfo(np.float64).tiny)
    optional_target_by_route: dict[tuple[int, int], int] = {}
    risk_by_route: dict[tuple[int, int], float] = {}
    bytes_per_position_by_route: dict[tuple[int, int], int] = {}
    allocated_positions_by_route = {identity: 0 for identity in routes}
    baseline_optional_bytes = 0
    for identity in routes:
        nodes = tuple(sorted(node_ids_by_route[identity]))
        mandatory = mandatory_by_route[identity]
        scalar = scalar_by_route[identity]
        if not nodes:
            raise ValueError(f"cross-head query allocation route {identity} contains no blocks")
        route_attention = np.stack([query_attention_by_node[node_id] for node_id in nodes], axis=0)
        caps = cumulative_cap_fraction * route_attention.sum(axis=0)
        optional = set(nodes) - mandatory
        optional_target = sum(full_state.blocks[node_id].position_count for node_id in scalar)
        expected = record_by_route[identity].selected_positions - sum(
            full_state.blocks[node_id].position_count for node_id in mandatory
        )
        if optional_target != expected:
            raise ValueError("scalar cross-head reference has inconsistent optional capacity")
        optional_target_by_route[identity] = optional_target
        selected_optional_bytes = sum(full_state.blocks[node_id].byte_size for node_id in scalar)
        baseline_optional_bytes += selected_optional_bytes
        ratios = {
            full_state.blocks[node_id].byte_size // full_state.blocks[node_id].position_count
            for node_id in optional
        }
        if optional and any(
            full_state.blocks[node_id].byte_size % full_state.blocks[node_id].position_count
            for node_id in optional
        ):
            raise ValueError("cross-head allocation requires integral bytes per position")
        if optional and len(ratios) != 1:
            raise ValueError("cross-head allocation requires one byte cost per position per route")
        bytes_per_position_by_route[identity] = next(iter(ratios)) if ratios else 0
        scalar_with_mandatory = mandatory | scalar
        scalar_attention = (
            np.stack(
                [query_attention_by_node[node_id] for node_id in sorted(scalar_with_mandatory)],
                axis=0,
            ).sum(axis=0)
            if scalar_with_mandatory
            else np.zeros_like(caps)
        )
        attainable = float(caps.sum())
        baseline_coverage = float(np.minimum(caps, scalar_attention).sum())
        risk_by_route[identity] = max(0.0, attainable - baseline_coverage) / max(
            attainable,
            epsilon,
        )

    # Allocate only block-aligned query-conditioned reservation quanta.  The
    # score is the route's remaining capped-mass deficit divided by one plus
    # its reserved fraction, which gives high-risk routes more capacity while
    # preventing one route from monopolizing the complete reservation.
    target_reserved_bytes = math.floor(baseline_optional_bytes * reservation_fraction)
    protected_by_route: dict[tuple[int, int], set[int]] = {identity: set() for identity in routes}
    protected_score_by_route = {identity: 0.0 for identity in routes}
    layout_group_score_by_route = {identity: 0.0 for identity in routes}
    layout_group_candidates_by_route = {identity: 0 for identity in routes}
    layout_group_pareto_candidates_by_route = {identity: 0 for identity in routes}
    layout_group_joint_donor_candidates_by_route = {identity: 0 for identity in routes}
    protected_layout_groups_by_route: dict[tuple[int, int], list[tuple[int, ...]]] = {
        identity: [] for identity in routes
    }
    atomic_layout_selection_by_route: dict[tuple[int, int], _AtomicLayoutGroupCandidate] = {}
    layout_group_risk_rejections_by_route = {identity: 0 for identity in routes}
    layout_group_target_bytes = 0
    allocated_bytes = 0
    if output_impact_reservation:
        scalar_optional_ids = tuple(
            sorted(node_id for identity in routes for node_id in scalar_by_route[identity])
        )
        if scalar_optional_ids:
            descriptors = pool_block_descriptors(
                full_state,
                host_transfer_mode="deferred_layer_vectorized",
                eligible_node_ids=scalar_optional_ids,
                node_table_ids=scalar_optional_ids,
            )
            value_by_node = {
                descriptor.source_id: np.asarray(descriptor.pooled_value, dtype=np.float64)
                for descriptor in descriptors
            }
            scored_candidates: list[tuple[float, float, int, tuple[int, int]]] = []
            for identity in routes:
                scalar_nodes = tuple(sorted(scalar_by_route[identity]))
                optional_prompt_mass = math.fsum(
                    max(0.0, float(prompt_score_array[node_id]))
                    for node_id in node_ids_by_route[identity]
                    if node_id not in mandatory_by_route[identity]
                )
                for node_id in scalar_nodes:
                    prompt_share = max(0.0, float(prompt_score_array[node_id])) / max(
                        optional_prompt_mass,
                        epsilon,
                    )
                    node_value = value_by_node[node_id]
                    node_norm = float(np.linalg.norm(node_value))
                    alternatives = tuple(other for other in scalar_nodes if other != node_id)
                    value_deviation = (
                        min(
                            float(np.linalg.norm(node_value - value_by_node[other]))
                            / max(
                                node_norm + float(np.linalg.norm(value_by_node[other])),
                                epsilon,
                            )
                            for other in alternatives
                        )
                        if alternatives
                        else 1.0
                    )
                    output_impact = prompt_share * value_deviation
                    if output_impact > 1e-12:
                        scored_candidates.append((output_impact, prompt_share, node_id, identity))
            protected_target_bytes = math.floor(
                target_reserved_bytes * output_impact_reservation_fraction
            )
            for output_impact, _prompt_share, node_id, identity in sorted(
                scored_candidates,
                key=lambda item: (-item[0], -item[1], item[2]),
            ):
                block = full_state.blocks[node_id]
                if allocated_bytes + block.byte_size > protected_target_bytes:
                    continue
                protected_by_route[identity].add(node_id)
                protected_score_by_route[identity] += output_impact
                allocated_positions_by_route[identity] += block.position_count
                allocated_bytes += block.byte_size
    elif layout_group_reservation and layout_group_atomic_selection:
        protected_target_bytes = math.floor(
            target_reserved_bytes * layout_group_reservation_fraction
        )
        layout_group_target_bytes = protected_target_bytes
        (
            atomic_candidates,
            layout_group_candidates_by_route,
            layout_group_pareto_candidates_by_route,
            layout_group_joint_donor_candidates_by_route,
        ) = _atomic_layout_group_candidates(
            full_state,
            routes=routes,
            node_ids_by_route=node_ids_by_route,
            mandatory_by_route=mandatory_by_route,
            scalar_by_route=scalar_by_route,
            optional_target_by_route=optional_target_by_route,
            prompt_score_array=prompt_score_array,
            query_attention_by_node=query_attention_by_node,
            cumulative_cap_fraction=cumulative_cap_fraction,
            prompt_weight=prompt_weight,
            maximum_blocks=layout_group_maximum_blocks,
            row_bins=layout_group_row_bins,
            selection_mode=layout_group_atomic_selection_mode,
        )
        accepted_by_route: dict[tuple[int, int], tuple[_AtomicLayoutGroupCandidate, ...]] = {}
        for identity in routes:
            route_accepted: list[_AtomicLayoutGroupCandidate] = []
            for candidate in atomic_candidates[identity]:
                combined_risk = max(
                    candidate.normalized_prompt_risk,
                    candidate.output_impact_risk,
                )
                prompt_floor = candidate.baseline_prompt_mass * minimum_prompt_mass_fraction
                accepted = bool(
                    candidate.normalized_query_gain + 1e-12 >= layout_group_minimum_normalized_gain
                    and candidate.selected_prompt_mass + 1e-12 >= prompt_floor
                    and (
                        defer_surrogate_stability_to_logit
                        or candidate.normalized_query_gain + 1e-12
                        >= layout_group_gain_to_risk_ratio * combined_risk
                    )
                )
                if accepted:
                    route_accepted.append(candidate)
                else:
                    layout_group_risk_rejections_by_route[identity] += 1
            accepted_by_route[identity] = tuple(route_accepted)
        if layout_group_atomic_selection_mode in {
            "pareto_knapsack",
            "pareto_joint_donor_knapsack",
        }:
            selected_atomic_candidates = _atomic_group_multiple_choice_knapsack(
                accepted_by_route,
                byte_budget=protected_target_bytes,
                gain_to_risk_ratio=layout_group_gain_to_risk_ratio,
                defer_surrogate_risk_to_logit=defer_surrogate_stability_to_logit,
            )
        else:
            best_by_route = [
                max(
                    route_accepted,
                    key=lambda candidate: (
                        candidate.normalized_query_gain
                        / max(
                            max(
                                candidate.normalized_prompt_risk,
                                candidate.output_impact_risk,
                            ),
                            epsilon,
                        ),
                        candidate.normalized_query_gain,
                        candidate.group_score,
                        tuple(-node_id for node_id in candidate.group),
                    ),
                )
                for route_accepted in accepted_by_route.values()
                if route_accepted
            ]
            selected_atomic_candidates = tuple(
                sorted(
                    best_by_route,
                    key=lambda item: (
                        -item.normalized_query_gain
                        / max(
                            max(item.normalized_prompt_risk, item.output_impact_risk),
                            epsilon,
                        ),
                        -item.normalized_query_gain,
                        -item.group_score,
                        item.identity,
                        item.group,
                    ),
                )
            )
        for candidate in selected_atomic_candidates:
            if allocated_bytes + candidate.group_bytes > protected_target_bytes:
                continue
            identity = candidate.identity
            protected_by_route[identity].update(candidate.group)
            protected_layout_groups_by_route[identity].append(candidate.group)
            layout_group_score_by_route[identity] += candidate.group_score
            allocated_positions_by_route[identity] = sum(
                full_state.blocks[node_id].position_count for node_id in candidate.group
            )
            allocated_bytes += candidate.group_bytes
            atomic_layout_selection_by_route[identity] = candidate
    elif layout_group_reservation:
        protected_target_bytes = math.floor(
            target_reserved_bytes * layout_group_reservation_fraction
        )
        layout_group_target_bytes = protected_target_bytes
        scored_groups: list[tuple[float, float, float, tuple[int, int], tuple[int, ...]]] = []
        for identity in routes:
            optional_nodes = tuple(
                node_id
                for node_id in node_ids_by_route[identity]
                if node_id not in mandatory_by_route[identity]
            )
            groups = _coherent_layout_groups(
                full_state,
                optional_nodes,
                maximum_blocks=layout_group_maximum_blocks,
                row_bins=layout_group_row_bins,
            )
            layout_group_candidates_by_route[identity] = len(groups)
            if not groups:
                continue
            route_attention = np.stack(
                [query_attention_by_node[node_id] for node_id in node_ids_by_route[identity]],
                axis=0,
            )
            caps = cumulative_cap_fraction * route_attention.sum(axis=0)
            attainable = float(caps.sum())
            optional_prompt_mass = math.fsum(
                max(0.0, float(prompt_score_array[node_id])) for node_id in optional_nodes
            )
            for group in groups:
                group_attention = np.stack(
                    [query_attention_by_node[node_id] for node_id in group],
                    axis=0,
                ).sum(axis=0)
                query_share = float(np.minimum(caps, group_attention).sum()) / max(
                    attainable,
                    epsilon,
                )
                prompt_share = math.fsum(
                    max(0.0, float(prompt_score_array[node_id])) for node_id in group
                ) / max(optional_prompt_mass, epsilon)
                group_score = query_share + prompt_weight * prompt_share
                group_positions = sum(
                    full_state.blocks[node_id].position_count for node_id in group
                )
                score_density = group_score / max(group_positions, 1)
                scored_groups.append((score_density, group_score, prompt_share, identity, group))
        for _density, group_score, _prompt_share, identity, group in sorted(
            scored_groups,
            key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]),
        ):
            if any(node_id in protected_by_route[identity] for node_id in group):
                continue
            group_bytes = sum(full_state.blocks[node_id].byte_size for node_id in group)
            group_positions = sum(full_state.blocks[node_id].position_count for node_id in group)
            next_positions = allocated_positions_by_route[identity] + group_positions
            if allocated_bytes + group_bytes > protected_target_bytes:
                continue
            if next_positions > optional_target_by_route[identity]:
                continue
            remaining = tuple(
                node_id
                for node_id in node_ids_by_route[identity]
                if node_id not in mandatory_by_route[identity]
                and node_id not in protected_by_route[identity]
                and node_id not in group
            )
            if not _can_attain_exact_positions(
                full_state,
                remaining,
                optional_target_by_route[identity] - next_positions,
            ):
                continue
            protected_by_route[identity].update(group)
            protected_layout_groups_by_route[identity].append(group)
            layout_group_score_by_route[identity] += group_score
            allocated_positions_by_route[identity] = next_positions
            allocated_bytes += group_bytes
    while True:
        best: tuple[float, float, int, int] | None = None
        best_cost = 0
        for identity in routes:
            if identity in atomic_layout_selection_by_route:
                continue
            optional_target = optional_target_by_route[identity]
            allocated_positions = allocated_positions_by_route[identity]
            remaining_positions = optional_target - allocated_positions
            quantum_positions = min(full_state.block_size, remaining_positions)
            if quantum_positions <= 0 or risk_by_route[identity] <= 1e-12:
                continue
            quantum_bytes = bytes_per_position_by_route[identity] * quantum_positions
            if allocated_bytes + quantum_bytes > target_reserved_bytes:
                continue
            allocated_fraction = allocated_positions / max(optional_target, 1)
            priority = risk_by_route[identity] / (1.0 + allocated_fraction)
            proposal = (priority, risk_by_route[identity], -identity[0], -identity[1])
            if best is None or proposal > best:
                best = proposal
                best_cost = quantum_bytes
        if best is None:
            break
        identity = (-best[2], -best[3])
        remaining_positions = (
            optional_target_by_route[identity] - allocated_positions_by_route[identity]
        )
        allocated_positions_by_route[identity] += min(full_state.block_size, remaining_positions)
        allocated_bytes += best_cost

    proposals: dict[tuple[int, int], _QueryCoverageInitialSelectionResult] = {}
    for identity in routes:
        atomic = atomic_layout_selection_by_route.get(identity)
        if atomic is not None:
            scalar = scalar_by_route[identity]
            added = tuple(sorted(set(atomic.selected_optional_node_ids) - scalar))
            proposals[identity] = _QueryCoverageInitialSelectionResult(
                selected_optional_node_ids=atomic.selected_optional_node_ids,
                reserved_count=len(atomic.group),
                changed_count=len(added),
                baseline_query_coverage=atomic.baseline_coverage,
                selected_query_coverage=atomic.selected_coverage,
                normalized_query_coverage_gain=atomic.normalized_query_gain,
                baseline_selected_prompt_mass=atomic.baseline_prompt_mass,
                selected_prompt_mass=atomic.selected_prompt_mass,
                normalized_prompt_mass_loss=atomic.normalized_prompt_risk,
                candidate_count=layout_group_candidates_by_route[identity],
                reserved_node_ids=atomic.group,
                added_node_ids=added,
            )
        else:
            proposals[identity] = _prompt_query_coverage_initial_selection(
                full_state,
                route_node_ids=node_ids_by_route[identity],
                mandatory_node_ids=tuple(sorted(mandatory_by_route[identity])),
                scalar_prompt_selected_optional_node_ids=tuple(sorted(scalar_by_route[identity])),
                optional_target_positions=optional_target_by_route[identity],
                prompt_score_array=prompt_score_array,
                query_attention_by_node=query_attention_by_node,
                reservation_fraction=reservation_fraction,
                prompt_weight=prompt_weight,
                minimum_prompt_mass_fraction=minimum_prompt_mass_fraction,
                coverage_objective="capped_cumulative",
                cumulative_cap_fraction=cumulative_cap_fraction,
                reservation_target_positions=allocated_positions_by_route[identity],
                pre_reserved_optional_node_ids=tuple(sorted(protected_by_route[identity])),
            )

    output_impact_by_route: dict[tuple[int, int], tuple[float, int]] = {
        identity: (0.0, 0) for identity in routes
    }
    if stability_gate and stability_risk_mode == "output_impact":
        changed_node_ids: set[int] = set()
        for identity, local in proposals.items():
            if identity in atomic_layout_selection_by_route:
                continue
            scalar = scalar_by_route[identity]
            proposed = set(local.selected_optional_node_ids)
            changed_node_ids.update(scalar - proposed)
            changed_node_ids.update(proposed - scalar)
        if changed_node_ids:
            ordered_changed = tuple(sorted(changed_node_ids))
            descriptors = pool_block_descriptors(
                full_state,
                host_transfer_mode="deferred_layer_vectorized",
                eligible_node_ids=ordered_changed,
                node_table_ids=ordered_changed,
            )
            value_by_node = {
                descriptor.source_id: np.asarray(descriptor.pooled_value, dtype=np.float64)
                for descriptor in descriptors
            }
            for identity, local in proposals.items():
                if identity in atomic_layout_selection_by_route:
                    continue
                scalar = scalar_by_route[identity]
                proposed = set(local.selected_optional_node_ids)
                donors = tuple(sorted(scalar - proposed))
                recipients = tuple(sorted(proposed - scalar))
                if not donors:
                    continue
                optional_prompt_mass = math.fsum(
                    max(0.0, float(prompt_score_array[node_id]))
                    for node_id in node_ids_by_route[identity]
                    if node_id not in mandatory_by_route[identity]
                )
                if not recipients:
                    output_impact_by_route[identity] = (1.0, len(donors))
                    continue
                if optional_prompt_mass <= epsilon:
                    output_impact_by_route[identity] = (0.0, len(donors))
                    continue
                output_impact = 0.0
                for donor in donors:
                    donor_value = value_by_node[donor]
                    donor_norm = float(np.linalg.norm(donor_value))
                    closest_deviation = min(
                        float(np.linalg.norm(donor_value - value_by_node[recipient]))
                        / max(
                            donor_norm + float(np.linalg.norm(value_by_node[recipient])),
                            epsilon,
                        )
                        for recipient in recipients
                    )
                    donor_attention_share = max(
                        0.0,
                        float(prompt_score_array[donor]),
                    ) / max(optional_prompt_mass, epsilon)
                    output_impact += donor_attention_share * closest_deviation
                output_impact_by_route[identity] = (output_impact, len(donors))

    results: dict[tuple[int, int], _QueryCoverageInitialSelectionResult] = {}
    for identity in routes:
        local = proposals[identity]
        proposed_gain = local.normalized_query_coverage_gain
        proposed_prompt_loss = local.normalized_prompt_mass_loss
        proposed_output_impact, output_impact_donor_count = output_impact_by_route[identity]
        donor_risk = (
            proposed_output_impact
            if stability_risk_mode == "output_impact"
            else proposed_prompt_loss
        )
        stability_rejected = bool(
            stability_gate
            and identity not in atomic_layout_selection_by_route
            and local.changed_count
            and (
                proposed_gain + 1e-12 < minimum_normalized_gain
                or proposed_gain + 1e-12 < gain_to_prompt_loss_ratio * donor_risk
            )
        )
        if stability_rejected:
            local = _QueryCoverageInitialSelectionResult(
                tuple(sorted(scalar_by_route[identity])),
                0,
                0,
                local.baseline_query_coverage,
                local.baseline_query_coverage,
                0.0,
                local.baseline_selected_prompt_mass,
                local.baseline_selected_prompt_mass,
                0.0,
                local.candidate_count,
                (),
                (),
                proposed_normalized_gain=proposed_gain,
                proposed_normalized_prompt_loss=proposed_prompt_loss,
                proposed_output_impact_risk=proposed_output_impact,
                output_impact_donor_count=output_impact_donor_count,
                stability_rejected=True,
            )
        active_layout_groups = tuple(
            group
            for group in sorted(protected_layout_groups_by_route[identity])
            if set(group) <= set(local.selected_optional_node_ids)
        )
        atomic_layout = atomic_layout_selection_by_route.get(identity)
        results[identity] = replace(
            local,
            cross_head_risk=risk_by_route[identity],
            allocated_bytes=(
                allocated_positions_by_route[identity] * bytes_per_position_by_route[identity]
            ),
            proposed_normalized_gain=proposed_gain,
            proposed_normalized_prompt_loss=proposed_prompt_loss,
            proposed_output_impact_risk=proposed_output_impact,
            output_impact_donor_count=output_impact_donor_count,
            output_impact_protected_count=(
                len(protected_by_route[identity]) if output_impact_reservation else 0
            ),
            output_impact_protected_bytes=(
                sum(
                    full_state.blocks[node_id].byte_size for node_id in protected_by_route[identity]
                )
                if output_impact_reservation
                else 0
            ),
            output_impact_protected_score=(
                protected_score_by_route[identity] if output_impact_reservation else 0.0
            ),
            layout_group_candidate_count=layout_group_candidates_by_route[identity],
            layout_group_protected_groups=active_layout_groups,
            layout_group_protected_bytes=sum(
                full_state.blocks[node_id].byte_size
                for group in active_layout_groups
                for node_id in group
            ),
            layout_group_protected_score=(
                layout_group_score_by_route[identity] if active_layout_groups else 0.0
            ),
            layout_group_target_bytes=layout_group_target_bytes,
            layout_group_accepted_candidate_count=int(atomic_layout is not None),
            layout_group_risk_rejected_count=(layout_group_risk_rejections_by_route[identity]),
            layout_group_pareto_candidate_count=(layout_group_pareto_candidates_by_route[identity]),
            layout_group_knapsack_selected_count=int(
                layout_group_atomic_selection_mode
                in {"pareto_knapsack", "pareto_joint_donor_knapsack"}
                and atomic_layout is not None
            ),
            layout_group_joint_donor_candidate_count=(
                layout_group_joint_donor_candidates_by_route[identity]
            ),
            layout_group_normalized_query_gain=(
                atomic_layout.normalized_query_gain if atomic_layout is not None else 0.0
            ),
            layout_group_normalized_prompt_risk=(
                atomic_layout.normalized_prompt_risk if atomic_layout is not None else 0.0
            ),
            layout_group_output_impact_risk=(
                atomic_layout.output_impact_risk if atomic_layout is not None else 0.0
            ),
            stability_rejected=stability_rejected,
        )
    return results


def select_prompt_attention_topk_ids_by_head(
    full_state: FullKVState,
    budget: SelectionBudget,
    *,
    prompt_attention_by_node: Mapping[int, float],
    forecast_attention_by_node: Mapping[int, float] | None = None,
    head_routes: Sequence[tuple[int, int]],
    position_limits: Mapping[tuple[int, int], int],
    minimum_visual_fraction: float = 0.0,
    adaptive_visual_swaps: bool = False,
    minimum_text_mass_fraction: float = 1.0,
    visual_gain_ratio: float = 1.0,
    maximum_visual_swaps: int = 0,
    swap_risk_gate: bool = False,
    maximum_normalized_text_loss: float = 1.0,
    forecast_gain_ratio: float = 1.0,
    future_query_reranking: bool = False,
    future_minimum_prompt_mass_fraction: float = 1.0,
    future_gain_ratio: float = 1.0,
    maximum_future_replacements: int = 0,
    future_divergence_fail_closed: bool = False,
    future_minimum_prompt_forecast_tv: float = 0.0,
    future_risk_gate: bool = False,
    future_maximum_normalized_prompt_loss: float = 1.0,
    future_minimum_normalized_gain: float = 0.0,
    prompt_query_attention_by_node: Mapping[int, Any] | None = None,
    query_coverage_reranking: bool = False,
    query_coverage_minimum_prompt_mass_fraction: float = 1.0,
    query_coverage_maximum_replacements: int = 0,
    query_coverage_minimum_normalized_gain: float = 0.0,
    query_coverage_initial_selection: bool = False,
    query_coverage_initial_fraction: float = 0.0,
    query_coverage_initial_prompt_weight: float = 1.0,
    query_coverage_initial_minimum_prompt_mass_fraction: float = 1.0,
    query_coverage_initial_objective: str = "max",
    query_coverage_initial_cap_fraction: float = 1.0,
    query_coverage_cross_head_allocation: bool = False,
    query_coverage_cross_head_stability_gate: bool = False,
    query_coverage_cross_head_stability_risk_mode: str = "prompt_mass",
    query_coverage_cross_head_minimum_normalized_gain: float = 0.0,
    query_coverage_cross_head_gain_to_prompt_loss_ratio: float = 1.0,
    query_coverage_output_impact_reservation: bool = False,
    query_coverage_output_impact_fraction: float = 0.0,
    query_coverage_layout_group_reservation: bool = False,
    query_coverage_layout_group_fraction: float = 0.0,
    query_coverage_layout_group_maximum_blocks: int = 0,
    query_coverage_layout_group_row_bins: int = 4,
    query_coverage_layout_group_atomic_selection: bool = False,
    query_coverage_layout_group_atomic_selection_mode: str = "topk_greedy",
    query_coverage_layout_group_minimum_utilization_fraction: float = 0.0,
    query_coverage_layout_group_minimum_normalized_gain: float = 0.0,
    query_coverage_layout_group_gain_to_risk_ratio: float = 1.0,
    short_horizon_logit_stability: bool = False,
    progressive_logit_safe_eviction: bool = False,
    hierarchical_logit_safe_eviction: bool = False,
) -> HeadLocalPromptSelection:
    """Select exact Prompt Top-K blocks independently for fixed head routes.

    Dense Hugging Face caches impose a physical position capacity on every
    layer/KV-head identity.  Once stable heads are routed independently, the
    global greedy scan is unnecessary: each identity can consume only its own
    capacity.  Mandatory blocks are installed first.  Optional blocks then
    maximize attainable occupied positions, prompt-attention score, and
    deterministic node-ID order, in that order.  This avoids paying for dense
    padding while useful exact source evidence remains attainable, without a
    40k-node global sort or audit construction.
    """

    if not math.isfinite(minimum_visual_fraction) or not 0 <= minimum_visual_fraction <= 1:
        raise ValueError("minimum_visual_fraction must be finite and in [0, 1]")
    if not 0 <= minimum_text_mass_fraction <= 1:
        raise ValueError("minimum_text_mass_fraction must be in [0, 1]")
    if not math.isfinite(visual_gain_ratio) or visual_gain_ratio <= 0:
        raise ValueError("visual_gain_ratio must be finite and positive")
    if maximum_visual_swaps < 0:
        raise ValueError("maximum_visual_swaps must be nonnegative")
    if not math.isfinite(maximum_normalized_text_loss) or not 0 < maximum_normalized_text_loss <= 1:
        raise ValueError("maximum_normalized_text_loss must be finite and in (0, 1]")
    if not math.isfinite(forecast_gain_ratio) or forecast_gain_ratio <= 0:
        raise ValueError("forecast_gain_ratio must be finite and positive")
    if not 0 <= future_minimum_prompt_mass_fraction <= 1:
        raise ValueError("future_minimum_prompt_mass_fraction must be in [0, 1]")
    if not math.isfinite(future_gain_ratio) or future_gain_ratio <= 0:
        raise ValueError("future_gain_ratio must be finite and positive")
    if maximum_future_replacements < 0:
        raise ValueError("maximum_future_replacements must be nonnegative")
    if (
        not math.isfinite(future_minimum_prompt_forecast_tv)
        or not 0 <= future_minimum_prompt_forecast_tv <= 1
    ):
        raise ValueError("future_minimum_prompt_forecast_tv must be in [0, 1]")
    if (
        not math.isfinite(future_maximum_normalized_prompt_loss)
        or not 0 <= future_maximum_normalized_prompt_loss <= 1
    ):
        raise ValueError("future_maximum_normalized_prompt_loss must be in [0, 1]")
    if (
        not math.isfinite(future_minimum_normalized_gain)
        or not 0 <= future_minimum_normalized_gain <= 1
    ):
        raise ValueError("future_minimum_normalized_gain must be in [0, 1]")
    if adaptive_visual_swaps and future_query_reranking:
        raise ValueError("adaptive visual swaps and future-query reranking are mutually exclusive")
    if not 0 <= query_coverage_minimum_prompt_mass_fraction <= 1:
        raise ValueError("query_coverage_minimum_prompt_mass_fraction must be in [0, 1]")
    if query_coverage_maximum_replacements < 0:
        raise ValueError("query_coverage_maximum_replacements must be nonnegative")
    if (
        not math.isfinite(query_coverage_minimum_normalized_gain)
        or not 0 <= query_coverage_minimum_normalized_gain <= 1
    ):
        raise ValueError("query_coverage_minimum_normalized_gain must be in [0, 1]")
    if (
        not math.isfinite(query_coverage_initial_fraction)
        or not 0 <= query_coverage_initial_fraction <= 1
    ):
        raise ValueError("query_coverage_initial_fraction must be finite and in [0, 1]")
    if (
        not math.isfinite(query_coverage_initial_prompt_weight)
        or query_coverage_initial_prompt_weight < 0
    ):
        raise ValueError("query_coverage_initial_prompt_weight must be finite and nonnegative")
    if not 0 <= query_coverage_initial_minimum_prompt_mass_fraction <= 1:
        raise ValueError("query_coverage_initial_minimum_prompt_mass_fraction must be in [0, 1]")
    if query_coverage_initial_objective not in {"max", "capped_cumulative"}:
        raise ValueError("query_coverage_initial_objective must be 'max' or 'capped_cumulative'")
    if (
        not math.isfinite(query_coverage_initial_cap_fraction)
        or not 0 < query_coverage_initial_cap_fraction <= 1
    ):
        raise ValueError("query_coverage_initial_cap_fraction must be finite and in (0, 1]")
    if (
        not math.isfinite(query_coverage_cross_head_minimum_normalized_gain)
        or not 0 <= query_coverage_cross_head_minimum_normalized_gain <= 1
    ):
        raise ValueError(
            "query_coverage_cross_head_minimum_normalized_gain must be finite and in [0, 1]"
        )
    if query_coverage_cross_head_stability_risk_mode not in {
        "prompt_mass",
        "output_impact",
    }:
        raise ValueError(
            "query_coverage_cross_head_stability_risk_mode must be 'prompt_mass' or 'output_impact'"
        )
    if (
        not math.isfinite(query_coverage_cross_head_gain_to_prompt_loss_ratio)
        or query_coverage_cross_head_gain_to_prompt_loss_ratio <= 0
    ):
        raise ValueError(
            "query_coverage_cross_head_gain_to_prompt_loss_ratio must be finite and positive"
        )
    if (
        not math.isfinite(query_coverage_output_impact_fraction)
        or not 0 <= query_coverage_output_impact_fraction <= 1
    ):
        raise ValueError("query_coverage_output_impact_fraction must be finite and in [0, 1]")
    if (
        not math.isfinite(query_coverage_layout_group_fraction)
        or not 0 <= query_coverage_layout_group_fraction <= 1
    ):
        raise ValueError("query_coverage_layout_group_fraction must be finite and in [0, 1]")
    if query_coverage_layout_group_maximum_blocks < 0:
        raise ValueError("query_coverage_layout_group_maximum_blocks must be nonnegative")
    if not 1 <= query_coverage_layout_group_row_bins <= 32:
        raise ValueError("query_coverage_layout_group_row_bins must be between 1 and 32")
    if (
        not math.isfinite(query_coverage_layout_group_minimum_utilization_fraction)
        or not 0 <= query_coverage_layout_group_minimum_utilization_fraction <= 1
    ):
        raise ValueError(
            "query_coverage_layout_group_minimum_utilization_fraction must be in [0, 1]"
        )
    if (
        not math.isfinite(query_coverage_layout_group_minimum_normalized_gain)
        or not 0 <= query_coverage_layout_group_minimum_normalized_gain <= 1
    ):
        raise ValueError("query_coverage_layout_group_minimum_normalized_gain must be in [0, 1]")
    if (
        not math.isfinite(query_coverage_layout_group_gain_to_risk_ratio)
        or query_coverage_layout_group_gain_to_risk_ratio <= 0
    ):
        raise ValueError("query_coverage_layout_group_gain_to_risk_ratio must be positive")
    if query_coverage_layout_group_atomic_selection_mode not in {
        "topk_greedy",
        "pareto_knapsack",
        "pareto_joint_donor_knapsack",
    }:
        raise ValueError(
            "query_coverage_layout_group_atomic_selection_mode must be "
            "'topk_greedy', 'pareto_knapsack', or 'pareto_joint_donor_knapsack'"
        )
    if (
        sum(
            (
                adaptive_visual_swaps,
                future_query_reranking,
                query_coverage_reranking,
                query_coverage_initial_selection,
            )
        )
        > 1
    ):
        raise ValueError(
            "stable-head visual swaps, future reranking, query reranking, and initial "
            "query coverage selection are mutually exclusive"
        )
    if adaptive_visual_swaps:
        if minimum_visual_fraction:
            raise ValueError("adaptive visual swaps cannot use a fixed visual fraction")
        if maximum_visual_swaps < 1:
            raise ValueError("adaptive visual swaps require at least one permitted swap")
        if swap_risk_gate and (maximum_normalized_text_loss >= 1.0 and forecast_gain_ratio <= 1.0):
            raise ValueError("swap risk gating must tighten text loss, forecast gain, or both")
    elif (
        minimum_text_mass_fraction != 1.0
        or visual_gain_ratio != 1.0
        or maximum_visual_swaps != 0
        or swap_risk_gate
        or maximum_normalized_text_loss != 1.0
        or forecast_gain_ratio != 1.0
    ):
        raise ValueError("adaptive swap controls require adaptive_visual_swaps")
    if future_query_reranking:
        if minimum_visual_fraction:
            raise ValueError("future-query reranking cannot use a fixed visual fraction")
        if maximum_future_replacements < 1:
            raise ValueError("future-query reranking requires at least one replacement")
    elif (
        future_minimum_prompt_mass_fraction != 1.0
        or future_gain_ratio != 1.0
        or maximum_future_replacements != 0
    ):
        raise ValueError("future-query controls require future_query_reranking")
    if future_risk_gate:
        if not future_query_reranking:
            raise ValueError("future risk gating requires future-query reranking")
        if future_maximum_normalized_prompt_loss >= 1.0 and future_minimum_normalized_gain <= 0.0:
            raise ValueError("future risk gating must tighten prompt loss or future gain")
    elif future_maximum_normalized_prompt_loss != 1.0 or future_minimum_normalized_gain != 0.0:
        raise ValueError("future per-swap risk controls require future risk gating")
    if future_divergence_fail_closed:
        if not future_query_reranking:
            raise ValueError("future divergence fail-closed requires future_query_reranking")
        if future_minimum_prompt_forecast_tv <= 0:
            raise ValueError("future divergence fail-closed requires positive minimum TV")
    elif future_minimum_prompt_forecast_tv != 0.0:
        raise ValueError("minimum future prompt/forecast TV requires fail-closed mode")
    if query_coverage_reranking:
        if prompt_query_attention_by_node is None:
            raise ValueError("prompt-query coverage reranking requires per-query prompt attention")
        if query_coverage_maximum_replacements < 1:
            raise ValueError("prompt-query coverage reranking requires at least one replacement")
        if query_coverage_minimum_normalized_gain <= 0:
            raise ValueError(
                "prompt-query coverage reranking requires positive minimum normalized gain"
            )
    elif (
        query_coverage_minimum_prompt_mass_fraction != 1.0
        or query_coverage_maximum_replacements != 0
        or query_coverage_minimum_normalized_gain != 0.0
    ):
        raise ValueError("prompt-query coverage controls require query_coverage_reranking")
    if query_coverage_initial_selection:
        if prompt_query_attention_by_node is None:
            raise ValueError("initial prompt-query coverage requires per-query prompt attention")
        if query_coverage_initial_fraction <= 0:
            raise ValueError("initial prompt-query coverage requires a positive fraction")
        if query_coverage_initial_objective == "max" and query_coverage_initial_cap_fraction != 1.0:
            raise ValueError("max initial query coverage requires cap fraction 1.0")
        if query_coverage_cross_head_allocation:
            if query_coverage_initial_objective != "capped_cumulative":
                raise ValueError("cross-head query allocation requires capped cumulative coverage")
            if minimum_visual_fraction != 0.0:
                raise ValueError("cross-head query allocation cannot use a fixed visual quota")
        if query_coverage_cross_head_stability_gate:
            if not query_coverage_cross_head_allocation:
                raise ValueError("cross-head stability gating requires cross-head allocation")
            if query_coverage_cross_head_minimum_normalized_gain <= 0:
                raise ValueError("cross-head stability gating requires a positive minimum gain")
            if query_coverage_cross_head_gain_to_prompt_loss_ratio <= 1:
                raise ValueError("cross-head stability gating requires a gain/loss ratio above 1")
        elif (
            query_coverage_cross_head_stability_risk_mode != "prompt_mass"
            or query_coverage_cross_head_minimum_normalized_gain != 0.0
            or query_coverage_cross_head_gain_to_prompt_loss_ratio != 1.0
        ):
            raise ValueError("cross-head stability controls require stability gating")
        if query_coverage_output_impact_reservation:
            if not query_coverage_cross_head_allocation:
                raise ValueError("output-impact reservation requires cross-head allocation")
            if query_coverage_initial_objective != "capped_cumulative":
                raise ValueError("output-impact reservation requires capped cumulative coverage")
            if query_coverage_output_impact_fraction <= 0:
                raise ValueError("output-impact reservation requires a positive fraction")
            if query_coverage_cross_head_stability_gate:
                raise ValueError(
                    "output-impact reservation cannot use post-selection stability gating"
                )
        elif query_coverage_output_impact_fraction != 0.0:
            raise ValueError(
                "output-impact reservation fraction requires output-impact reservation"
            )
        if query_coverage_layout_group_reservation:
            if not query_coverage_cross_head_allocation:
                raise ValueError("layout-group reservation requires cross-head allocation")
            if query_coverage_initial_objective != "capped_cumulative":
                raise ValueError("layout-group reservation requires capped cumulative coverage")
            if query_coverage_layout_group_fraction <= 0:
                raise ValueError("layout-group reservation requires a positive fraction")
            if query_coverage_layout_group_maximum_blocks < 2:
                raise ValueError("layout-group reservation requires at least two blocks per group")
            if query_coverage_output_impact_reservation:
                raise ValueError(
                    "layout-group and output-impact initial reservations are mutually exclusive"
                )
            if query_coverage_cross_head_stability_gate and not (
                query_coverage_layout_group_atomic_selection
                and query_coverage_layout_group_atomic_selection_mode
                in {"pareto_knapsack", "pareto_joint_donor_knapsack"}
                and query_coverage_cross_head_stability_risk_mode == "output_impact"
            ):
                raise ValueError(
                    "layout-group reservation permits stability gating only for "
                    "Pareto-knapsack atomic groups with output-impact risk"
                )
            if query_coverage_layout_group_atomic_selection:
                if query_coverage_layout_group_minimum_utilization_fraction <= 0:
                    raise ValueError(
                        "atomic layout-group selection requires positive minimum utilization"
                    )
                if query_coverage_layout_group_minimum_normalized_gain <= 0:
                    raise ValueError("atomic layout-group selection requires positive gain")
                if query_coverage_layout_group_gain_to_risk_ratio <= 1:
                    raise ValueError(
                        "atomic layout-group selection requires a gain/risk ratio above 1"
                    )
                if (
                    query_coverage_layout_group_atomic_selection_mode
                    in {"pareto_knapsack", "pareto_joint_donor_knapsack"}
                    and not query_coverage_cross_head_stability_gate
                    and not short_horizon_logit_stability
                    and not progressive_logit_safe_eviction
                    and not hierarchical_logit_safe_eviction
                ):
                    raise ValueError(
                        "Pareto-knapsack atomic groups require non-group stability gating"
                    )
            elif (
                query_coverage_layout_group_atomic_selection_mode != "topk_greedy"
                or query_coverage_layout_group_minimum_utilization_fraction != 0.0
                or query_coverage_layout_group_minimum_normalized_gain != 0.0
                or query_coverage_layout_group_gain_to_risk_ratio != 1.0
            ):
                raise ValueError(
                    "atomic layout-group controls require atomic layout-group selection"
                )
        elif (
            query_coverage_layout_group_fraction != 0.0
            or query_coverage_layout_group_maximum_blocks != 0
            or query_coverage_layout_group_atomic_selection
            or query_coverage_layout_group_atomic_selection_mode != "topk_greedy"
            or query_coverage_layout_group_minimum_utilization_fraction != 0.0
            or query_coverage_layout_group_minimum_normalized_gain != 0.0
            or query_coverage_layout_group_gain_to_risk_ratio != 1.0
        ):
            raise ValueError("layout-group controls require layout-group reservation")
    elif (
        query_coverage_initial_fraction != 0.0
        or query_coverage_initial_prompt_weight != 1.0
        or query_coverage_initial_minimum_prompt_mass_fraction != 1.0
        or query_coverage_initial_objective != "max"
        or query_coverage_initial_cap_fraction != 1.0
        or query_coverage_cross_head_allocation
        or query_coverage_cross_head_stability_gate
        or query_coverage_cross_head_stability_risk_mode != "prompt_mass"
        or query_coverage_cross_head_minimum_normalized_gain != 0.0
        or query_coverage_cross_head_gain_to_prompt_loss_ratio != 1.0
        or query_coverage_output_impact_reservation
        or query_coverage_output_impact_fraction != 0.0
        or query_coverage_layout_group_reservation
        or query_coverage_layout_group_fraction != 0.0
        or query_coverage_layout_group_maximum_blocks != 0
        or query_coverage_layout_group_atomic_selection
        or query_coverage_layout_group_atomic_selection_mode != "topk_greedy"
        or query_coverage_layout_group_minimum_utilization_fraction != 0.0
        or query_coverage_layout_group_minimum_normalized_gain != 0.0
        or query_coverage_layout_group_gain_to_risk_ratio != 1.0
    ):
        raise ValueError(
            "initial prompt-query coverage controls require query_coverage_initial_selection"
        )
    routes = frozenset((int(layer), int(head)) for layer, head in head_routes)
    if not routes:
        return HeadLocalPromptSelection((), ())
    if len(routes) != len(head_routes):
        raise ValueError("stable Prompt Top-K head routes must be unique")
    missing_limits = routes - set(position_limits)
    if missing_limits:
        raise ValueError(
            f"stable Prompt Top-K routes lack position limits: {sorted(missing_limits)}"
        )
    node_count = len(full_state.blocks)
    if set(prompt_attention_by_node) != set(range(node_count)):
        raise ValueError("prompt_attention_topk scores must cover every source block exactly once")
    score_array = np.fromiter(
        (float(prompt_attention_by_node[node_id]) for node_id in range(node_count)),
        dtype=np.float64,
        count=node_count,
    )
    if not bool(np.all(np.isfinite(score_array))):
        raise ValueError("prompt_attention_topk scores must be finite")
    forecast_score_array: np.ndarray[Any, np.dtype[np.float64]] | None = None
    if forecast_attention_by_node is not None:
        if set(forecast_attention_by_node) != set(range(node_count)):
            raise ValueError("forecast attention scores must cover every source block exactly once")
        forecast_score_array = np.fromiter(
            (float(forecast_attention_by_node[node_id]) for node_id in range(node_count)),
            dtype=np.float64,
            count=node_count,
        )
        if not bool(np.all(np.isfinite(forecast_score_array))) or bool(
            np.any(forecast_score_array < 0)
        ):
            raise ValueError("forecast attention scores must be finite and nonnegative")
    if swap_risk_gate and forecast_score_array is None:
        raise ValueError("swap risk gating requires forecast attention scores")
    if future_query_reranking and forecast_score_array is None:
        raise ValueError("future-query reranking requires forecast attention scores")
    query_attention_arrays: dict[int, np.ndarray[Any, np.dtype[np.float64]]] = {}
    if query_coverage_reranking or query_coverage_initial_selection:
        if prompt_query_attention_by_node is None:  # pragma: no cover - validated above
            raise ValueError("prompt-query coverage attention is unexpectedly absent")
        if set(prompt_query_attention_by_node) != set(range(node_count)):
            raise ValueError("prompt-query attention must cover every source block exactly once")
        query_count: int | None = None
        for node_id in range(node_count):
            values = np.asarray(prompt_query_attention_by_node[node_id], dtype=np.float64)
            if values.ndim != 1 or not values.size:
                raise ValueError("prompt-query attention vectors must be nonempty and 1-D")
            if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0)):
                raise ValueError("prompt-query attention vectors must be finite and nonnegative")
            if query_count is None:
                query_count = int(values.size)
            elif int(values.size) != query_count:
                raise ValueError("prompt-query attention vectors must have one common length")
            query_attention_arrays[node_id] = values
    node_ids_by_route: dict[tuple[int, int], list[int]] = defaultdict(list)
    mandatory_by_route: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node_id, block in enumerate(full_state.blocks):
        identity = (block.layer, block.kv_head)
        if identity not in routes:
            continue
        node_ids_by_route[identity].append(node_id)
        if block.mandatory:
            mandatory_by_route[identity].append(node_id)

    cross_head_initial_by_route: dict[tuple[int, int], _QueryCoverageInitialSelectionResult] = {}
    scalar_reference: HeadLocalPromptSelection | None = None
    if query_coverage_cross_head_allocation:
        scalar_reference = select_prompt_attention_topk_ids_by_head(
            full_state,
            budget,
            prompt_attention_by_node=prompt_attention_by_node,
            head_routes=tuple(sorted(routes)),
            position_limits=position_limits,
        )
        cross_head_initial_by_route = _cross_head_capped_query_initial_selection(
            full_state,
            head_routes=tuple(sorted(routes)),
            scalar_selection=scalar_reference,
            prompt_score_array=score_array,
            query_attention_by_node=query_attention_arrays,
            reservation_fraction=query_coverage_initial_fraction,
            prompt_weight=query_coverage_initial_prompt_weight,
            minimum_prompt_mass_fraction=(query_coverage_initial_minimum_prompt_mass_fraction),
            cumulative_cap_fraction=query_coverage_initial_cap_fraction,
            stability_gate=query_coverage_cross_head_stability_gate,
            stability_risk_mode=query_coverage_cross_head_stability_risk_mode,
            minimum_normalized_gain=(query_coverage_cross_head_minimum_normalized_gain),
            gain_to_prompt_loss_ratio=(query_coverage_cross_head_gain_to_prompt_loss_ratio),
            output_impact_reservation=query_coverage_output_impact_reservation,
            output_impact_reservation_fraction=query_coverage_output_impact_fraction,
            layout_group_reservation=query_coverage_layout_group_reservation,
            layout_group_reservation_fraction=query_coverage_layout_group_fraction,
            layout_group_maximum_blocks=query_coverage_layout_group_maximum_blocks,
            layout_group_row_bins=query_coverage_layout_group_row_bins,
            layout_group_atomic_selection=query_coverage_layout_group_atomic_selection,
            layout_group_atomic_selection_mode=(query_coverage_layout_group_atomic_selection_mode),
            layout_group_minimum_normalized_gain=(
                query_coverage_layout_group_minimum_normalized_gain
            ),
            layout_group_gain_to_risk_ratio=(query_coverage_layout_group_gain_to_risk_ratio),
            defer_surrogate_stability_to_logit=(
                short_horizon_logit_stability
                or progressive_logit_safe_eviction
                or hierarchical_logit_safe_eviction
            ),
        )

    selected: list[int] = []
    capacity_records: list[HeadLocalCapacityRecord] = []
    total_cost = 0
    for identity in sorted(routes):
        mandatory = tuple(mandatory_by_route.get(identity, ()))
        used_positions = sum(full_state.blocks[node_id].position_count for node_id in mandatory)
        limit = int(position_limits[identity])
        if used_positions > limit:
            raise InfeasibleBaselineBudgetError(
                "mandatory exact blocks exceed the dense cache capacity: "
                f"layer={identity[0]}, kv_head={identity[1]}, "
                f"required={used_positions}, limit={limit}"
            )
        selected.extend(mandatory)
        total_cost += sum(budget.cost(full_state.blocks[node_id]) for node_id in mandatory)
        mandatory_set = frozenset(mandatory)
        mandatory_visual_positions = sum(
            full_state.blocks[node_id].position_count
            for node_id in mandatory
            if full_state.blocks[node_id].modality in {Modality.IMAGE, Modality.VIDEO}
        )
        remaining_capacity = limit - used_positions
        candidates_by_size_and_modality: dict[tuple[int, bool], list[int]] = defaultdict(list)
        for node_id in node_ids_by_route.get(identity, ()):
            if node_id not in mandatory_set:
                block = full_state.blocks[node_id]
                candidates_by_size_and_modality[
                    (
                        block.position_count,
                        block.modality in {Modality.IMAGE, Modality.VIDEO},
                    )
                ].append(node_id)

        # Equal-size blocks are interchangeable for capacity.  Keeping only
        # each score-sorted prefix reduces the exact knapsack from one state
        # transition per source block to one per distinct block size/count.
        # Dense storage pays for the full physical route capacity even when a
        # score-optimal subset leaves holes.  Choose the largest attainable
        # footprint first, then maximize prompt-attention score within that
        # footprint; lower node IDs break exact ties.  This converts padding
        # into exact evidence without changing the physical byte budget.
        states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0.0, ())}
        for size, is_visual in sorted(candidates_by_size_and_modality):
            ranked = tuple(
                sorted(
                    candidates_by_size_and_modality[(size, is_visual)],
                    key=lambda node_id: (-score_array[node_id], node_id),
                )
            )
            maximum_count = min(len(ranked), remaining_capacity // size)
            prefix_scores = np.concatenate(
                (
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(score_array[np.asarray(ranked[:maximum_count], dtype=np.intp)]),
                )
            )
            previous = states
            updated: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
            for (occupied, visual_positions), (prior_score, prior_nodes) in previous.items():
                available_count = min(maximum_count, (remaining_capacity - occupied) // size)
                for count in range(available_count + 1):
                    next_occupied = occupied + count * size
                    next_visual_positions = visual_positions + (count * size if is_visual else 0)
                    next_score = prior_score + float(prefix_scores[count])
                    next_nodes = tuple(sorted((*prior_nodes, *ranked[:count])))
                    state_key = (next_occupied, next_visual_positions)
                    incumbent = updated.get(state_key)
                    if (
                        incumbent is None
                        or next_score > incumbent[0]
                        or (next_score == incumbent[0] and next_nodes < incumbent[1])
                    ):
                        updated[state_key] = (next_score, next_nodes)
            states = updated
        best_occupied = max(occupied for occupied, _visual_positions in states)
        capacity_states = [
            (visual_positions, score, nodes)
            for (occupied, visual_positions), (score, nodes) in states.items()
            if occupied == best_occupied
        ]
        target_visual_positions = math.ceil(limit * minimum_visual_fraction)
        required_optional_visual = max(
            0,
            target_visual_positions - mandatory_visual_positions,
        )
        quota_states = [state for state in capacity_states if state[0] >= required_optional_visual]
        choice_pool = quota_states or capacity_states
        optional_visual_positions, _best_score, chosen = max(
            choice_pool,
            key=lambda item: (
                item[1] if quota_states else item[0],
                item[0] if quota_states else item[1],
                tuple(-node for node in item[2]),
            ),
        )
        if query_coverage_cross_head_allocation:
            initial_query_coverage = cross_head_initial_by_route[identity]
        elif query_coverage_initial_selection:
            initial_query_coverage = _prompt_query_coverage_initial_selection(
                full_state,
                route_node_ids=node_ids_by_route.get(identity, ()),
                mandatory_node_ids=mandatory,
                scalar_prompt_selected_optional_node_ids=chosen,
                optional_target_positions=best_occupied,
                prompt_score_array=score_array,
                query_attention_by_node=query_attention_arrays,
                reservation_fraction=query_coverage_initial_fraction,
                prompt_weight=query_coverage_initial_prompt_weight,
                minimum_prompt_mass_fraction=(query_coverage_initial_minimum_prompt_mass_fraction),
                coverage_objective=query_coverage_initial_objective,
                cumulative_cap_fraction=query_coverage_initial_cap_fraction,
            )
        else:
            initial_query_coverage = _QueryCoverageInitialSelectionResult(
                tuple(chosen),
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                (),
                (),
            )
        chosen = initial_query_coverage.selected_optional_node_ids
        query_coverage = (
            _prompt_query_coverage_rerank_selection(
                full_state,
                route_node_ids=node_ids_by_route.get(identity, ()),
                mandatory_node_ids=mandatory,
                selected_optional_node_ids=chosen,
                prompt_score_array=score_array,
                query_attention_by_node=query_attention_arrays,
                minimum_prompt_mass_fraction=(query_coverage_minimum_prompt_mass_fraction),
                maximum_replacements=query_coverage_maximum_replacements,
                minimum_normalized_gain=query_coverage_minimum_normalized_gain,
            )
            if query_coverage_reranking
            else _QueryCoverageRerankResult(
                tuple(chosen),
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                (),
            )
        )
        chosen = query_coverage.selected_optional_node_ids
        adaptive = (
            _adaptive_visual_swap_selection(
                full_state,
                route_node_ids=node_ids_by_route.get(identity, ()),
                mandatory_node_ids=mandatory,
                selected_optional_node_ids=chosen,
                score_array=score_array,
                forecast_score_array=forecast_score_array,
                minimum_text_mass_fraction=minimum_text_mass_fraction,
                visual_gain_ratio=visual_gain_ratio,
                maximum_visual_swaps=maximum_visual_swaps,
                swap_risk_gate=swap_risk_gate,
                maximum_normalized_text_loss=maximum_normalized_text_loss,
                forecast_gain_ratio=forecast_gain_ratio,
            )
            if adaptive_visual_swaps
            else _AdaptiveSwapResult(
                tuple(chosen),
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                (),
            )
        )
        chosen = adaptive.selected_optional_node_ids
        future_rerank = (
            _future_query_rerank_selection(
                full_state,
                route_node_ids=node_ids_by_route.get(identity, ()),
                mandatory_node_ids=mandatory,
                selected_optional_node_ids=chosen,
                prompt_score_array=score_array,
                forecast_score_array=forecast_score_array,
                minimum_prompt_mass_fraction=future_minimum_prompt_mass_fraction,
                future_gain_ratio=future_gain_ratio,
                maximum_replacements=maximum_future_replacements,
                minimum_prompt_forecast_tv=future_minimum_prompt_forecast_tv,
                risk_gate=future_risk_gate,
                maximum_normalized_prompt_loss=(future_maximum_normalized_prompt_loss),
                minimum_normalized_future_gain=future_minimum_normalized_gain,
            )
            if future_query_reranking and forecast_score_array is not None
            else _FutureRerankResult(
                tuple(chosen),
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0,
                (),
                0.0,
                False,
            )
        )
        chosen = future_rerank.selected_optional_node_ids
        optional_visual_positions = sum(
            full_state.blocks[node_id].position_count
            for node_id in chosen
            if full_state.blocks[node_id].modality in {Modality.IMAGE, Modality.VIDEO}
        )
        selected.extend(chosen)
        used_positions += best_occupied
        selected_bytes = sum(
            full_state.blocks[node_id].byte_size for node_id in (*mandatory, *chosen)
        )
        total_cost += sum(budget.cost(full_state.blocks[node_id]) for node_id in chosen)
        capacity_records.append(
            HeadLocalCapacityRecord(
                layer=identity[0],
                kv_head=identity[1],
                position_limit=limit,
                selected_positions=used_positions,
                shortfall_positions=limit - used_positions,
                selected_bytes=selected_bytes,
                selected_visual_positions=(mandatory_visual_positions + optional_visual_positions),
                target_visual_positions=target_visual_positions,
                adaptive_swap_count=adaptive.swap_count,
                baseline_text_prompt_mass=adaptive.baseline_text_prompt_mass,
                selected_text_prompt_mass=adaptive.selected_text_prompt_mass,
                normalized_visual_evidence_gain=(adaptive.normalized_visual_evidence_gain),
                normalized_text_prompt_mass_loss=(adaptive.normalized_text_prompt_mass_loss),
                normalized_forecast_visual_gain=(adaptive.normalized_forecast_visual_gain),
                normalized_forecast_text_loss=adaptive.normalized_forecast_text_loss,
                adaptive_swap_candidate_count=adaptive.candidate_count,
                adaptive_swap_risk_rejected_count=adaptive.risk_rejected_count,
                adaptive_swap_pairs=adaptive.accepted_pairs,
                future_rerank_count=future_rerank.replacement_count,
                baseline_selected_prompt_mass=(future_rerank.baseline_selected_prompt_mass),
                selected_prompt_mass_after_future_rerank=(future_rerank.selected_prompt_mass),
                normalized_future_query_gain=(future_rerank.normalized_future_query_gain),
                normalized_future_prompt_mass_loss=(future_rerank.normalized_prompt_mass_loss),
                future_rerank_candidate_count=future_rerank.candidate_count,
                future_rerank_rejected_count=future_rerank.rejected_count,
                future_rerank_risk_rejected_count=(future_rerank.risk_rejected_count),
                future_rerank_pairs=future_rerank.accepted_pairs,
                prompt_forecast_total_variation=(future_rerank.prompt_forecast_total_variation),
                forecast_divergence_eligible=future_rerank.forecast_divergence_eligible,
                query_coverage_rerank_count=query_coverage.replacement_count,
                baseline_query_coverage=query_coverage.baseline_query_coverage,
                selected_query_coverage=query_coverage.selected_query_coverage,
                normalized_query_coverage_gain=(query_coverage.normalized_query_coverage_gain),
                query_coverage_baseline_prompt_mass=(query_coverage.baseline_selected_prompt_mass),
                query_coverage_selected_prompt_mass=(query_coverage.selected_prompt_mass),
                query_coverage_normalized_prompt_loss=(query_coverage.normalized_prompt_mass_loss),
                query_coverage_candidate_count=query_coverage.candidate_count,
                query_coverage_rejected_count=query_coverage.rejected_count,
                query_coverage_pairs=query_coverage.accepted_pairs,
                query_coverage_initial_reserved_count=(initial_query_coverage.reserved_count),
                query_coverage_initial_changed_count=(initial_query_coverage.changed_count),
                query_coverage_initial_baseline=(initial_query_coverage.baseline_query_coverage),
                query_coverage_initial_selected=(initial_query_coverage.selected_query_coverage),
                query_coverage_initial_normalized_gain=(
                    initial_query_coverage.normalized_query_coverage_gain
                ),
                query_coverage_initial_baseline_prompt_mass=(
                    initial_query_coverage.baseline_selected_prompt_mass
                ),
                query_coverage_initial_selected_prompt_mass=(
                    initial_query_coverage.selected_prompt_mass
                ),
                query_coverage_initial_normalized_prompt_loss=(
                    initial_query_coverage.normalized_prompt_mass_loss
                ),
                query_coverage_initial_candidate_count=(initial_query_coverage.candidate_count),
                query_coverage_initial_reserved_bytes=sum(
                    full_state.blocks[node_id].byte_size
                    for node_id in initial_query_coverage.reserved_node_ids
                ),
                query_coverage_initial_allocated_bytes=(initial_query_coverage.allocated_bytes),
                query_coverage_initial_cross_head_risk=(initial_query_coverage.cross_head_risk),
                query_coverage_initial_proposed_normalized_gain=(
                    initial_query_coverage.proposed_normalized_gain
                ),
                query_coverage_initial_proposed_normalized_prompt_loss=(
                    initial_query_coverage.proposed_normalized_prompt_loss
                ),
                query_coverage_initial_proposed_output_impact_risk=(
                    initial_query_coverage.proposed_output_impact_risk
                ),
                query_coverage_initial_output_impact_donor_count=(
                    initial_query_coverage.output_impact_donor_count
                ),
                query_coverage_initial_output_impact_protected_count=(
                    initial_query_coverage.output_impact_protected_count
                ),
                query_coverage_initial_output_impact_protected_bytes=(
                    initial_query_coverage.output_impact_protected_bytes
                ),
                query_coverage_initial_output_impact_protected_score=(
                    initial_query_coverage.output_impact_protected_score
                ),
                query_coverage_initial_layout_group_candidate_count=(
                    initial_query_coverage.layout_group_candidate_count
                ),
                query_coverage_initial_layout_group_protected_group_count=len(
                    initial_query_coverage.layout_group_protected_groups
                ),
                query_coverage_initial_layout_group_protected_block_count=sum(
                    len(group) for group in initial_query_coverage.layout_group_protected_groups
                ),
                query_coverage_initial_layout_group_protected_bytes=(
                    initial_query_coverage.layout_group_protected_bytes
                ),
                query_coverage_initial_layout_group_protected_score=(
                    initial_query_coverage.layout_group_protected_score
                ),
                query_coverage_initial_layout_group_target_bytes=(
                    initial_query_coverage.layout_group_target_bytes
                ),
                query_coverage_initial_layout_group_accepted_candidate_count=(
                    initial_query_coverage.layout_group_accepted_candidate_count
                ),
                query_coverage_initial_layout_group_risk_rejected_count=(
                    initial_query_coverage.layout_group_risk_rejected_count
                ),
                query_coverage_initial_layout_group_pareto_candidate_count=(
                    initial_query_coverage.layout_group_pareto_candidate_count
                ),
                query_coverage_initial_layout_group_knapsack_selected_count=(
                    initial_query_coverage.layout_group_knapsack_selected_count
                ),
                query_coverage_initial_layout_group_joint_donor_candidate_count=(
                    initial_query_coverage.layout_group_joint_donor_candidate_count
                ),
                query_coverage_initial_layout_group_normalized_query_gain=(
                    initial_query_coverage.layout_group_normalized_query_gain
                ),
                query_coverage_initial_layout_group_normalized_prompt_risk=(
                    initial_query_coverage.layout_group_normalized_prompt_risk
                ),
                query_coverage_initial_layout_group_output_impact_risk=(
                    initial_query_coverage.layout_group_output_impact_risk
                ),
                query_coverage_initial_layout_group_node_ids=(
                    initial_query_coverage.layout_group_protected_groups
                ),
                query_coverage_initial_stability_rejected=(
                    initial_query_coverage.stability_rejected
                ),
                query_coverage_initial_global_allocation=(query_coverage_cross_head_allocation),
                query_coverage_initial_added_node_ids=(initial_query_coverage.added_node_ids),
            )
        )
    if future_divergence_fail_closed and not any(
        record.forecast_divergence_eligible for record in capacity_records
    ):
        raise ValueError(
            "future-query reranking forecast did not diverge from prompt evidence on any route"
        )
    if total_cost > budget.value:
        raise InfeasibleBaselineBudgetError(
            "stable Prompt Top-K exact blocks exceed the shared active budget"
        )
    primary = HeadLocalPromptSelection(
        selected_node_ids=tuple(sorted(selected)),
        capacity_records=tuple(capacity_records),
    )
    if not short_horizon_logit_stability:
        return primary
    if scalar_reference is None:
        raise ValueError("short-horizon logit stability requires a scalar cross-head anchor")
    alternatives = [
        HeadLocalPromptSelectionAlternative(
            "pareto_joint_group_donor",
            primary.selected_node_ids,
            primary.capacity_records,
        )
    ]
    alternatives.append(
        HeadLocalPromptSelectionAlternative(
            "scalar_prompt_topk_anchor",
            scalar_reference.selected_node_ids,
            scalar_reference.capacity_records,
        )
    )
    return HeadLocalPromptSelection(
        selected_node_ids=primary.selected_node_ids,
        capacity_records=primary.capacity_records,
        logit_stability_shortlist=tuple(alternatives),
    )


def build_exact_baseline_plan(
    full_state: FullKVState,
    method: MosaicKVMethod,
    cache_config: CacheConfig,
    *,
    seed: int,
    prompt_attention_by_node: Mapping[int, float] | None = None,
    value_chunk_size: int = 256,
    position_limits: Mapping[tuple[int, int], int] | None = None,
) -> BaselineCompressionPlan:
    """Build an exact-only plan without prototypes, residuals, or repair."""

    source_budget, budget = resolve_baseline_budget(full_state, cache_config)
    started = time.perf_counter()
    selection = select_exact_baseline(
        full_state,
        method,
        budget,
        seed=seed,
        prompt_attention_by_node=prompt_attention_by_node,
        value_chunk_size=value_chunk_size,
        position_limits=position_limits,
    )
    selection_seconds = time.perf_counter() - started
    tier_started = time.perf_counter()
    exact = full_state.gather_exact_blocks(selection.selected_blocks)
    state = MosaicKVState.create(full_state, exact=exact)
    if cache_config.retention_ratio == 1.0:
        state.reconstruct_full_state(full_state)
    tier_seconds = time.perf_counter() - tier_started
    return BaselineCompressionPlan(
        method=method,
        full_state=full_state,
        state=state,
        selection=selection,
        source_budget_value=source_budget,
        active_budget_value=budget.value,
        selection_seconds=selection_seconds,
        tier_seconds=tier_seconds,
    )


__all__ = [
    "HeadLocalCapacityRecord",
    "HeadLocalPromptSelection",
    "InfeasibleBaselineBudgetError",
    "build_exact_baseline_plan",
    "resolve_baseline_budget",
    "select_exact_baseline",
    "select_prompt_attention_topk_ids",
    "select_prompt_attention_topk_ids_by_head",
    "value_novelty_scores",
]
