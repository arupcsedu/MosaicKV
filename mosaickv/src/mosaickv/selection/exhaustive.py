"""Brute-force tiny-graph oracle for greedy optimality-gap measurement."""

from __future__ import annotations

import math
from itertools import combinations

from mosaickv.selection.objective import BudgetedObjective
from mosaickv.selection.selector import InfeasibleSelectionBudgetError
from mosaickv.selection.types import (
    ExhaustiveSelectionResult,
    GreedyOptimalityComparison,
    SelectionBudget,
    SelectionResult,
)


def exhaustive_select(
    objective: BudgetedObjective,
    budget: SelectionBudget,
) -> ExhaustiveSelectionResult:
    """Return the exact feasible optimum for a guarded tiny graph."""

    graph = objective.graph
    node_count = len(graph.nodes)
    maximum = objective.config.exhaustive_max_nodes
    if node_count > maximum:
        raise ValueError(
            f"exhaustive selection is limited to {maximum} nodes; received {node_count}"
        )
    costs = tuple(budget.cost(node.block) for node in graph.nodes)
    mandatory = tuple(node.node_id for node in graph.nodes if node.block.mandatory)
    mandatory_cost = sum(costs[node_id] for node_id in mandatory)
    if mandatory_cost > budget.value:
        raise InfeasibleSelectionBudgetError(
            "mandatory exact blocks exceed the configured exhaustive-selection budget"
        )
    optional = tuple(node_id for node_id in range(node_count) if node_id not in mandatory)
    best_nodes = tuple(sorted(mandatory))
    best_cost = mandatory_cost
    best_objective = objective.evaluate(best_nodes)
    evaluated = 1
    for count in range(1, len(optional) + 1):
        for subset in combinations(optional, count):
            cost = mandatory_cost + sum(costs[node_id] for node_id in subset)
            if cost > budget.value:
                continue
            selected = tuple(sorted((*mandatory, *subset)))
            candidate = objective.evaluate(selected)
            evaluated += 1
            better = candidate.total > best_objective.total + 1e-12
            tied = math.isclose(candidate.total, best_objective.total, rel_tol=0, abs_tol=1e-12)
            deterministic_tie = tied and (cost, selected) < (best_cost, best_nodes)
            if better or deterministic_tie:
                best_nodes = selected
                best_cost = cost
                best_objective = candidate
    return ExhaustiveSelectionResult(
        selected_node_ids=best_nodes,
        objective=best_objective,
        budget_spent=best_cost,
        evaluated_subsets=evaluated,
    )


def compare_greedy_to_exhaustive(
    greedy: SelectionResult,
    objective: BudgetedObjective,
) -> GreedyOptimalityComparison:
    """Measure the observed objective gap against the tiny-graph optimum."""

    optimal = exhaustive_select(objective, greedy.budget)
    absolute_gap = max(0.0, optimal.objective.total - greedy.objective.total)
    relative_gap = absolute_gap / max(abs(optimal.objective.total), 1e-12)
    return GreedyOptimalityComparison(
        greedy_objective=greedy.objective.total,
        optimal_objective=optimal.objective.total,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        greedy_node_ids=greedy.selected_node_ids,
        optimal_node_ids=optimal.selected_node_ids,
        evaluated_subsets=optimal.evaluated_subsets,
    )


__all__ = ["compare_greedy_to_exhaustive", "exhaustive_select"]
