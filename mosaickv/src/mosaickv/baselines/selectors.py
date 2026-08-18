"""Deterministic, exact-tier simple baseline selectors."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.selected_text_prompt_mass > self.baseline_text_prompt_mass + 1e-12:
            raise ValueError("selected text prompt mass cannot exceed its baseline")


@dataclass(frozen=True, slots=True)
class HeadLocalPromptSelection:
    """Selected source blocks plus exact per-route capacity accounting."""

    selected_node_ids: tuple[int, ...]
    capacity_records: tuple[HeadLocalCapacityRecord, ...]

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
    added_node_ids: tuple[int, ...]


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
) -> _QueryCoverageInitialSelectionResult:
    """Build a feasible selection with query diversity reserved from the start.

    The scalar Prompt Top-K solution defines only the capacity target and an
    auditable comparison point.  Query-diverse blocks are chosen before the
    remaining capacity is filled by prompt mass.  Each greedy reservation is
    accepted only when the unselected block sizes can still attain the exact
    target, so the coverage objective cannot create padding or alter bytes.
    """

    route_nodes = tuple(sorted(int(node_id) for node_id in route_node_ids))
    mandatory = frozenset(int(node_id) for node_id in mandatory_node_ids)
    optional_nodes = tuple(node_id for node_id in route_nodes if node_id not in mandatory)
    scalar_selected = tuple(
        sorted(int(node_id) for node_id in scalar_prompt_selected_optional_node_ids)
    )
    epsilon = float(np.finfo(np.float64).tiny)

    def coverage_vector(
        node_ids: Sequence[int] | set[int] | frozenset[int],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        ordered = tuple(sorted(node_ids))
        if not ordered:
            return np.zeros_like(query_attention_by_node[route_nodes[0]], dtype=np.float64)
        return cast(
            "np.ndarray[Any, np.dtype[np.float64]]",
            np.max(
                np.stack([query_attention_by_node[node_id] for node_id in ordered], axis=0),
                axis=0,
            ),
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
        )

    reservation_target = min(
        optional_target_positions,
        max(1, math.ceil(optional_target_positions * reservation_fraction)),
    )
    reserved: list[int] = []
    remaining = set(optional_nodes)
    reserved_positions = 0
    current_coverage_vector = coverage_vector(mandatory)
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
            next_coverage = np.maximum(
                current_coverage_vector,
                query_attention_by_node[node_id],
            )
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
        current_coverage_vector = np.maximum(
            current_coverage_vector,
            query_attention_by_node[node_id],
        )

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
        added,
    )


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
    elif (
        query_coverage_initial_fraction != 0.0
        or query_coverage_initial_prompt_weight != 1.0
        or query_coverage_initial_minimum_prompt_mass_fraction != 1.0
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
        initial_query_coverage = (
            _prompt_query_coverage_initial_selection(
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
            )
            if query_coverage_initial_selection
            else _QueryCoverageInitialSelectionResult(
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
            )
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
    return HeadLocalPromptSelection(
        selected_node_ids=tuple(sorted(selected)),
        capacity_records=tuple(capacity_records),
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
