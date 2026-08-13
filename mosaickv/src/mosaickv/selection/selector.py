"""Mandatory-first deterministic lazy-greedy budget selection."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

from mosaickv.selection.objective import BudgetedObjective
from mosaickv.selection.types import (
    SelectionBudget,
    SelectionDecision,
    SelectionReason,
    SelectionResult,
)
from mosaickv.types import BudgetUnit


class InfeasibleSelectionBudgetError(ValueError):
    """Raised when mandatory exact blocks alone exceed the hard budget."""


def select_all_exact(
    objective: BudgetedObjective,
    budget: SelectionBudget,
) -> SelectionResult:
    """Select every block exactly in deterministic mandatory-first order.

    Retention ratio 1.0 has no optimization decision: every source block must
    survive. Running the versioned lazy-greedy heap to reach that unique set
    needlessly invalidates all cached entries after each choice. This linear
    path still records the exact sequential marginal gain and reason for every
    block, and it fails closed if the declared budget cannot hold the source.
    """

    graph = objective.graph
    costs = tuple(budget.cost(node.block) for node in graph.nodes)
    required = sum(costs)
    if required > budget.value:
        raise InfeasibleSelectionBudgetError(
            "all-exact source blocks exceed the configured selection budget: "
            f"required={required}, budget={budget.value}, unit={budget.unit.value}"
        )
    mandatory = tuple(node.node_id for node in graph.nodes if node.block.mandatory)
    optional = tuple(node.node_id for node in graph.nodes if not node.block.mandatory)
    order = mandatory + optional
    incremental = objective.incremental_state()
    decisions_by_id: dict[int, SelectionDecision] = {}
    for rank, node_id in enumerate(order):
        gain = incremental.select(node_id)
        cost = costs[node_id]
        decisions_by_id[node_id] = SelectionDecision(
            node_id=node_id,
            selected=True,
            selection_rank=rank,
            marginal_gain=gain,
            marginal_gain_per_cost=gain / cost,
            cost=cost,
            reason=(
                SelectionReason.MANDATORY_EXACT
                if graph.nodes[node_id].block.mandatory
                else SelectionReason.RETENTION_ONE_EXACT
            ),
        )
    selected_ids = tuple(range(len(graph.nodes)))
    selected_blocks = tuple(node.block for node in graph.nodes)
    return SelectionResult(
        selected_blocks=selected_blocks,
        selection_order=order,
        decisions=tuple(decisions_by_id[node_id] for node_id in selected_ids),
        budget=budget,
        budget_spent=required,
        active_bytes=sum(block.byte_size for block in selected_blocks),
        objective=objective.evaluate(selected_ids),
        heap_pops=0,
        marginal_recomputations=0,
    )


def _entry(
    node_id: int,
    marginal_gain: float,
    cost: int,
    state_version: int,
) -> tuple[float, float, int, int, int, float]:
    density = marginal_gain / cost
    # Heap order: greatest density, greatest raw gain, lower cost, lower node
    # ID.  The cached state version and raw value follow the deterministic key.
    return (-density, -marginal_gain, cost, node_id, state_version, marginal_gain)


def lazy_greedy_select(
    objective: BudgetedObjective,
    budget: SelectionBudget,
    *,
    position_limits: Mapping[tuple[int, int], int] | None = None,
    fill_position_limits: bool = False,
    position_fill_quantum: int | None = None,
    required_node_ids: Iterable[int] = (),
    prompt_anchor_node_ids: Iterable[int] = (),
    evidence_node_ids: Iterable[int] = (),
) -> SelectionResult:
    """Select mandatory blocks, then maximize cached marginal-gain density.

    ``position_limits`` is the fail-closed constraint used by dense HF cache
    materialization.  A dense cache pays for the maximum retained length in
    every layer/KV head; limiting each identity prevents a logical selection
    from silently exceeding the declared physical byte budget.
    """

    if not objective.is_submodular:
        raise ValueError("lazy greedy requires a submodular objective coefficient configuration")
    if position_fill_quantum is not None:
        if not fill_position_limits or position_limits is None:
            raise ValueError("position_fill_quantum requires dense-cache position filling")
        if position_fill_quantum < 1:
            raise ValueError("position_fill_quantum must be positive")
    graph = objective.graph
    required = tuple(sorted(set(int(node_id) for node_id in required_node_ids)))
    prompt_anchors = tuple(sorted(set(int(node_id) for node_id in prompt_anchor_node_ids)))
    evidence_nodes = tuple(sorted(set(int(node_id) for node_id in evidence_node_ids)))
    all_required = (*required, *prompt_anchors, *evidence_nodes)
    if any(node_id < 0 or node_id >= len(graph.nodes) for node_id in all_required):
        raise ValueError("required selector IDs reference a node outside the evidence graph")
    costs = tuple(budget.cost(node.block) for node in graph.nodes)
    mandatory = tuple(node.node_id for node in graph.nodes if node.block.mandatory)
    mandatory_cost = sum(costs[node_id] for node_id in mandatory)
    if mandatory_cost > budget.value:
        raise InfeasibleSelectionBudgetError(
            "mandatory exact blocks exceed the configured selection budget: "
            f"required={mandatory_cost}, budget={budget.value}, unit={budget.unit.value}"
        )

    selected: set[int] = set()
    selected_positions: defaultdict[tuple[int, int], int] = defaultdict(int)
    constrained: set[int] = set()
    incremental = objective.incremental_state()
    selection_order: list[int] = []
    decisions: dict[int, SelectionDecision] = {}
    spent = 0

    def is_position_representable(node_id: int) -> bool:
        block = graph.nodes[node_id].block
        return (
            block.mandatory
            or position_fill_quantum is None
            or block.position_count % position_fill_quantum == 0
        )

    for node_id in mandatory:
        block = graph.nodes[node_id].block
        identity = (block.layer, block.kv_head)
        limit = None if position_limits is None else position_limits.get(identity)
        if limit is not None and selected_positions[identity] + block.position_count > limit:
            raise InfeasibleSelectionBudgetError(
                "mandatory exact blocks exceed the dense cache capacity: "
                f"layer={block.layer}, kv_head={block.kv_head}, limit={limit}"
            )
        gain = incremental.select(node_id)
        cost = costs[node_id]
        rank = len(selection_order)
        selected.add(node_id)
        selection_order.append(node_id)
        spent += cost
        selected_positions[identity] += block.position_count
        decisions[node_id] = SelectionDecision(
            node_id=node_id,
            selected=True,
            selection_rank=rank,
            marginal_gain=gain,
            marginal_gain_per_cost=gain / cost,
            cost=cost,
            reason=SelectionReason.MANDATORY_EXACT,
        )

    def select_required_nodes(
        node_ids: Iterable[int],
        reason: SelectionReason,
        label: str,
        *,
        allow_partial: bool = False,
    ) -> None:
        nonlocal spent
        for node_id in node_ids:
            if node_id in selected:
                continue
            block = graph.nodes[node_id].block
            identity = (block.layer, block.kv_head)
            limit = None if position_limits is None else position_limits.get(identity)
            cost = costs[node_id]
            if not allow_partial and not is_position_representable(node_id):
                raise InfeasibleSelectionBudgetError(
                    f"required {label} {node_id} is not representable in the dense cache"
                )
            if cost > budget.value - spent:
                raise InfeasibleSelectionBudgetError(
                    f"required {label}s exceed the configured selection budget"
                )
            if limit is not None and selected_positions[identity] + block.position_count > limit:
                raise InfeasibleSelectionBudgetError(
                    f"required {label}s exceed the dense cache capacity: "
                    f"layer={block.layer}, kv_head={block.kv_head}, limit={limit}"
                )
            gain = incremental.select(node_id)
            rank = len(selection_order)
            selected.add(node_id)
            selection_order.append(node_id)
            spent += cost
            selected_positions[identity] += block.position_count
            decisions[node_id] = SelectionDecision(
                node_id=node_id,
                selected=True,
                selection_rank=rank,
                marginal_gain=gain,
                marginal_gain_per_cost=gain / cost,
                cost=cost,
                reason=reason,
            )

    # Prompt anchors are a quality guard, evidence nodes preserve fine local
    # structure, and prototype anchors are representation constraints. All are
    # auditable hard reservations inside the same active budget.
    select_required_nodes(prompt_anchors, SelectionReason.PROMPT_ANCHOR, "prompt anchor")
    select_required_nodes(
        evidence_nodes,
        SelectionReason.EVIDENCE_PRESERVATION,
        "evidence-preservation block",
    )
    # A modality boundary may leave a short source block. Prototype planning
    # prefers aligned anchors, but an unavoidable partial anchor remains a
    # valid exact payload. A deterministic residue-completion pass below adds
    # complementary boundary blocks before the ordinary fixed-block greedy
    # sweep, preserving an exactly filled dense head layout.
    select_required_nodes(
        required,
        SelectionReason.PROTOTYPE_ANCHOR,
        "prototype anchor",
        allow_partial=True,
    )

    if position_fill_quantum is not None:
        if position_limits is None:  # pragma: no cover - validated above
            raise ValueError("dense residue completion requires position limits")
        nodes_by_identity: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for node in graph.nodes:
            nodes_by_identity[(node.block.layer, node.block.kv_head)].append(node.node_id)
        for identity in sorted(position_limits):
            limit = position_limits[identity]
            residue = (limit - selected_positions[identity]) % position_fill_quantum
            if residue == 0:
                continue
            remaining_positions = limit - selected_positions[identity]
            remaining_budget = budget.value - spent
            # residue -> (positions, budget cost, negative static gain, IDs).
            # The smallest completing boundary footprint leaves the most room
            # for fixed blocks; utility and node IDs make the choice stable.
            best: dict[int, tuple[int, int, float, tuple[int, ...]]] = {
                0: (0, 0, 0.0, ())
            }
            for node_id in sorted(nodes_by_identity.get(identity, ())):
                if node_id in selected:
                    continue
                block = graph.nodes[node_id].block
                local_residue = block.position_count % position_fill_quantum
                if local_residue == 0:
                    continue
                cost = costs[node_id]
                static_gain = incremental.marginal_gain(node_id)
                updates = dict(best)
                for current_residue, state in best.items():
                    positions, current_cost, negative_gain, identifiers = state
                    candidate_positions = positions + block.position_count
                    candidate_cost = current_cost + cost
                    if (
                        candidate_positions > remaining_positions
                        or candidate_cost > remaining_budget
                    ):
                        continue
                    candidate = (
                        candidate_positions,
                        candidate_cost,
                        negative_gain - static_gain,
                        (*identifiers, node_id),
                    )
                    candidate_residue = (
                        current_residue + local_residue
                    ) % position_fill_quantum
                    previous = updates.get(candidate_residue)
                    if previous is None or candidate < previous:
                        updates[candidate_residue] = candidate
                best = updates
            completion = best.get(residue)
            if completion is None or not completion[3]:
                raise InfeasibleSelectionBudgetError(
                    "required dense-cache fragments cannot be completed on the fixed-block "
                    f"lattice: identity={identity}, selected_positions="
                    f"{selected_positions[identity]}, limit={limit}, "
                    f"quantum={position_fill_quantum}"
                )
            for node_id in completion[3]:
                block = graph.nodes[node_id].block
                cost = costs[node_id]
                gain = incremental.select(node_id)
                rank = len(selection_order)
                selected.add(node_id)
                selection_order.append(node_id)
                spent += cost
                selected_positions[identity] += block.position_count
                decisions[node_id] = SelectionDecision(
                    node_id=node_id,
                    selected=True,
                    selection_rank=rank,
                    marginal_gain=gain,
                    marginal_gain_per_cost=gain / cost,
                    cost=cost,
                    reason=SelectionReason.DENSE_CACHE_FILL,
                )

    if position_limits is not None and objective.config.minimum_stratum_fraction > 0:
        groups: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for node in graph.nodes:
            groups[(node.block.layer, node.block.kv_head)].append(node.node_id)
        for identity in sorted(position_limits):
            limit = position_limits[identity]
            reserve = math.floor(limit * objective.config.minimum_stratum_fraction)
            while selected_positions[identity] < reserve:
                candidates: list[tuple[float, float, int, int, float]] = []
                for node_id in groups.get(identity, ()):
                    if node_id in selected:
                        continue
                    if not is_position_representable(node_id):
                        constrained.add(node_id)
                        continue
                    block = graph.nodes[node_id].block
                    cost = costs[node_id]
                    if selected_positions[identity] + block.position_count > limit:
                        continue
                    if cost > budget.value - spent:
                        continue
                    gain = incremental.marginal_gain(node_id)
                    candidates.append((-gain / cost, -gain, cost, node_id, gain))
                if not candidates:
                    raise InfeasibleSelectionBudgetError(
                        "configured per-stratum reserve cannot be represented within the "
                        f"hard budget: identity={identity}, selected_positions="
                        f"{selected_positions[identity]}, reserve={reserve}, limit={limit}, "
                        f"remaining_budget={budget.value - spent}"
                    )
                _density, _negative_gain, cost, node_id, gain = min(candidates)
                block = graph.nodes[node_id].block
                gain = incremental.select(node_id)
                rank = len(selection_order)
                selected.add(node_id)
                selection_order.append(node_id)
                spent += cost
                selected_positions[identity] += block.position_count
                decisions[node_id] = SelectionDecision(
                    node_id=node_id,
                    selected=True,
                    selection_rank=rank,
                    marginal_gain=gain,
                    marginal_gain_per_cost=gain / cost,
                    cost=cost,
                    reason=SelectionReason.STRATUM_RESERVE,
                )

    heap: list[tuple[float, float, int, int, int, float]] = []
    state_version = len(selection_order)
    for node in graph.nodes:
        node_id = node.node_id
        if node_id in selected:
            continue
        if not is_position_representable(node_id):
            constrained.add(node_id)
            continue
        gain = incremental.marginal_gain(node_id)
        heapq.heappush(heap, _entry(node_id, gain, costs[node_id], state_version))

    heap_pops = 0
    marginal_recomputations = 0
    while heap and spent < budget.value:
        _negative_density, _negative_gain, cost, node_id, cached_version, gain = heapq.heappop(heap)
        heap_pops += 1
        block = graph.nodes[node_id].block
        identity = (block.layer, block.kv_head)
        limit = None if position_limits is None else position_limits.get(identity)
        if limit is not None and selected_positions[identity] + block.position_count > limit:
            constrained.add(node_id)
            continue
        if cost > budget.value - spent:
            continue
        if cached_version != state_version:
            gain = incremental.marginal_gain(node_id)
            marginal_recomputations += 1
            heapq.heappush(heap, _entry(node_id, gain, cost, state_version))
            continue
        if objective.config.stop_on_nonpositive_gain and gain <= 0:
            break
        gain = incremental.select(node_id)
        rank = len(selection_order)
        selected.add(node_id)
        selection_order.append(node_id)
        spent += cost
        selected_positions[identity] += block.position_count
        decisions[node_id] = SelectionDecision(
            node_id=node_id,
            selected=True,
            selection_rank=rank,
            marginal_gain=gain,
            marginal_gain_per_cost=gain / cost,
            cost=cost,
            reason=SelectionReason.LAZY_GREEDY,
        )
        state_version += 1

    if fill_position_limits:
        if position_limits is None:
            raise ValueError("dense-cache fill requires position_limits")
        fill_groups: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for node in graph.nodes:
            fill_groups[(node.block.layer, node.block.kv_head)].append(node.node_id)
        for identity in sorted(position_limits):
            limit = position_limits[identity]
            while selected_positions[identity] < limit:
                fill_candidates: list[tuple[float, float, int, int, float]] = []
                for node_id in fill_groups.get(identity, ()):
                    if node_id in selected:
                        continue
                    if not is_position_representable(node_id):
                        constrained.add(node_id)
                        continue
                    block = graph.nodes[node_id].block
                    cost = costs[node_id]
                    if selected_positions[identity] + block.position_count > limit:
                        continue
                    if cost > budget.value - spent:
                        continue
                    gain = incremental.marginal_gain(node_id)
                    fill_candidates.append((-gain / cost, -gain, cost, node_id, gain))
                if not fill_candidates:
                    break
                _density, _negative_gain, cost, node_id, _gain = min(fill_candidates)
                block = graph.nodes[node_id].block
                gain = incremental.select(node_id)
                rank = len(selection_order)
                selected.add(node_id)
                selection_order.append(node_id)
                spent += cost
                selected_positions[identity] += block.position_count
                decisions[node_id] = SelectionDecision(
                    node_id=node_id,
                    selected=True,
                    selection_rank=rank,
                    marginal_gain=gain,
                    marginal_gain_per_cost=gain / cost,
                    cost=cost,
                    reason=SelectionReason.DENSE_CACHE_FILL,
                )
        incomplete = {
            identity: (selected_positions[identity], limit)
            for identity, limit in position_limits.items()
            if selected_positions[identity] != limit
        }
        if incomplete:
            raise InfeasibleSelectionBudgetError(
                f"dense cache capacity could not be filled exactly: {incomplete}"
            )

    remaining = budget.value - spent
    for node in graph.nodes:
        node_id = node.node_id
        if node_id in decisions:
            continue
        gain = incremental.marginal_gain(node_id)
        cost = costs[node_id]
        block = graph.nodes[node_id].block
        identity = (block.layer, block.kv_head)
        limit = None if position_limits is None else position_limits.get(identity)
        if node_id in constrained or (
            limit is not None and selected_positions[identity] + block.position_count > limit
        ):
            reason = SelectionReason.DENSE_CACHE_CAPACITY
        elif cost > remaining:
            reason = SelectionReason.BUDGET_EXHAUSTED
        elif objective.config.stop_on_nonpositive_gain and gain <= 0:
            reason = SelectionReason.NONPOSITIVE_MARGINAL_GAIN
        else:
            reason = SelectionReason.NOT_SELECTED_LOWER_GAIN
        decisions[node_id] = SelectionDecision(
            node_id=node_id,
            selected=False,
            selection_rank=None,
            marginal_gain=gain,
            marginal_gain_per_cost=gain / cost,
            cost=cost,
            reason=reason,
        )

    selected_ids = tuple(sorted(selected))
    selected_blocks = tuple(graph.nodes[node_id].block for node_id in selected_ids)
    active_bytes = sum(block.byte_size for block in selected_blocks)
    if budget.unit is BudgetUnit.BYTES and active_bytes > budget.value:
        raise RuntimeError("byte-budget selector exceeded its configured hard limit")
    return SelectionResult(
        selected_blocks=selected_blocks,
        selection_order=tuple(selection_order),
        decisions=tuple(decisions[node_id] for node_id in range(len(graph.nodes))),
        budget=budget,
        budget_spent=spent,
        active_bytes=active_bytes,
        objective=objective.evaluate(selected_ids),
        heap_pops=heap_pops,
        marginal_recomputations=marginal_recomputations,
    )


__all__ = ["InfeasibleSelectionBudgetError", "lazy_greedy_select", "select_all_exact"]
