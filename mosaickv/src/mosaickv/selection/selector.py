"""Mandatory-first deterministic lazy-greedy budget selection."""

from __future__ import annotations

import heapq

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
) -> SelectionResult:
    """Select mandatory blocks, then maximize cached marginal-gain density."""

    if not objective.is_submodular:
        raise ValueError("lazy greedy requires a submodular objective coefficient configuration")
    graph = objective.graph
    costs = tuple(budget.cost(node.block) for node in graph.nodes)
    mandatory = tuple(node.node_id for node in graph.nodes if node.block.mandatory)
    mandatory_cost = sum(costs[node_id] for node_id in mandatory)
    if mandatory_cost > budget.value:
        raise InfeasibleSelectionBudgetError(
            "mandatory exact blocks exceed the configured selection budget: "
            f"required={mandatory_cost}, budget={budget.value}, unit={budget.unit.value}"
        )

    selected: set[int] = set()
    incremental = objective.incremental_state()
    selection_order: list[int] = []
    decisions: dict[int, SelectionDecision] = {}
    spent = 0
    for node_id in mandatory:
        gain = incremental.select(node_id)
        cost = costs[node_id]
        rank = len(selection_order)
        selected.add(node_id)
        selection_order.append(node_id)
        spent += cost
        decisions[node_id] = SelectionDecision(
            node_id=node_id,
            selected=True,
            selection_rank=rank,
            marginal_gain=gain,
            marginal_gain_per_cost=gain / cost,
            cost=cost,
            reason=SelectionReason.MANDATORY_EXACT,
        )

    heap: list[tuple[float, float, int, int, int, float]] = []
    state_version = len(selection_order)
    for node in graph.nodes:
        node_id = node.node_id
        if node_id in selected:
            continue
        gain = incremental.marginal_gain(node_id)
        heapq.heappush(heap, _entry(node_id, gain, costs[node_id], state_version))

    heap_pops = 0
    marginal_recomputations = 0
    while heap and spent < budget.value:
        _negative_density, _negative_gain, cost, node_id, cached_version, gain = heapq.heappop(heap)
        heap_pops += 1
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

    remaining = budget.value - spent
    for node in graph.nodes:
        node_id = node.node_id
        if node_id in decisions:
            continue
        gain = incremental.marginal_gain(node_id)
        cost = costs[node_id]
        if cost > remaining:
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
