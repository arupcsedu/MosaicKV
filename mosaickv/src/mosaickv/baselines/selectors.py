"""Deterministic, exact-tier simple baseline selectors."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np

from mosaickv.baselines.types import (
    BaselineCompressionPlan,
    BaselineSelectionDecision,
    BaselineSelectionReason,
    BaselineSelectionResult,
    BaselineStratum,
)
from mosaickv.cache_state import FullKVState, KVBlockDescriptor, MosaicKVState
from mosaickv.config import CacheConfig
from mosaickv.graph import pool_block_descriptors
from mosaickv.selection import SelectionBudget
from mosaickv.types import BudgetUnit, MosaicKVMethod


class InfeasibleBaselineBudgetError(ValueError):
    """Raised when mandatory exact blocks cannot fit the common budget."""


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
    reason_by_id = {node_id: BaselineSelectionReason.MANDATORY_EXACT for node_id in mandatory}
    candidate_reason = (
        BaselineSelectionReason.RETENTION_ONE_EXACT
        if budget.value == _source_cost(full_state, budget.unit)
        else selected_reason
    )
    for node_id in order:
        if node_id in selected:
            continue
        cost = costs[node_id]
        if cost <= budget.value - spent:
            selected.add(node_id)
            selection_order.append(node_id)
            spent += cost
            reason_by_id[node_id] = candidate_reason
    rank_by_id = {node_id: rank for rank, node_id in enumerate(selection_order)}
    decisions = tuple(
        BaselineSelectionDecision(
            node_id=node_id,
            selected=node_id in selected,
            selection_rank=rank_by_id.get(node_id),
            score=float(scores[node_id]),
            cost=costs[node_id],
            reason=reason_by_id.get(node_id, BaselineSelectionReason.BUDGET_EXHAUSTED),
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
    score_at_selection = {node_id: 1.0 for node_id in mandatory}
    while spent < budget.value:
        remaining = budget.value - spent
        viable = [
            group
            for group, queue in queues.items()
            if any(costs[node_id] <= remaining for node_id in queue)
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
            index for index, node_id in enumerate(queue) if costs[node_id] <= remaining
        )
        node_id = queue.pop(candidate_index)
        fraction_before = selected_cost[group] / source_cost[group]
        selected.add(node_id)
        selection_order.append(node_id)
        spent += costs[node_id]
        selected_cost[group] += costs[node_id]
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
        )
    if method is MosaicKVMethod.UNIFORM_KV:
        return _select_uniform(full_state, budget, seed=seed)
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
        )
    raise ValueError(f"method is not an exact compressed baseline: {method.value}")


def build_exact_baseline_plan(
    full_state: FullKVState,
    method: MosaicKVMethod,
    cache_config: CacheConfig,
    *,
    seed: int,
    prompt_attention_by_node: Mapping[int, float] | None = None,
    value_chunk_size: int = 256,
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
    "InfeasibleBaselineBudgetError",
    "build_exact_baseline_plan",
    "resolve_baseline_budget",
    "select_exact_baseline",
    "value_novelty_scores",
]
