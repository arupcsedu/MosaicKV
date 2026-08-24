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
        decisions_by_id[node_id] = SelectionDecision._from_validated(
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


def select_exact_subset(
    objective: BudgetedObjective,
    budget: SelectionBudget,
    selected_node_ids: Iterable[int],
) -> SelectionResult:
    """Record one explicit exact subset without rerunning the optimizer.

    This path is used after a separate, auditable controller has already made
    its eviction decisions. It preserves the graph-local node namespace and
    reconstructs complete selection telemetry for cache construction.
    """

    graph = objective.graph
    selected_ids = tuple(sorted(set(int(node_id) for node_id in selected_node_ids)))
    if any(node_id < 0 or node_id >= len(graph.nodes) for node_id in selected_ids):
        raise ValueError("explicit exact subset contains a node outside the graph")
    selected_set = frozenset(selected_ids)
    missing_mandatory = tuple(
        node.node_id
        for node in graph.nodes
        if node.block.mandatory and node.node_id not in selected_set
    )
    if missing_mandatory:
        raise InfeasibleSelectionBudgetError(
            "explicit exact subset omits mandatory nodes: "
            f"{missing_mandatory}"
        )
    costs = tuple(budget.cost(node.block) for node in graph.nodes)
    spent = sum(costs[node_id] for node_id in selected_ids)
    if spent > budget.value:
        raise InfeasibleSelectionBudgetError(
            "explicit exact subset exceeds the configured selection budget: "
            f"required={spent}, budget={budget.value}, unit={budget.unit.value}"
        )

    mandatory = tuple(
        node_id for node_id in selected_ids if graph.nodes[node_id].block.mandatory
    )
    optional = tuple(
        node_id for node_id in selected_ids if not graph.nodes[node_id].block.mandatory
    )
    order = mandatory + optional
    incremental = objective.incremental_state()
    decisions: dict[int, SelectionDecision] = {}
    for rank, node_id in enumerate(order):
        gain = incremental.select(node_id)
        cost = costs[node_id]
        decisions[node_id] = SelectionDecision._from_validated(
            node_id=node_id,
            selected=True,
            selection_rank=rank,
            marginal_gain=gain,
            marginal_gain_per_cost=gain / cost,
            cost=cost,
            reason=(
                SelectionReason.MANDATORY_EXACT
                if graph.nodes[node_id].block.mandatory
                else SelectionReason.DENSE_CACHE_FILL
            ),
        )
    unselected_ids = tuple(
        node.node_id for node in graph.nodes if node.node_id not in selected_set
    )
    final_gains = dict(
        zip(
            unselected_ids,
            incremental.marginal_gains(unselected_ids),
            strict=True,
        )
    )
    for node_id in unselected_ids:
        cost = costs[node_id]
        gain = final_gains[node_id]
        decisions[node_id] = SelectionDecision._from_validated(
            node_id=node_id,
            selected=False,
            selection_rank=None,
            marginal_gain=gain,
            marginal_gain_per_cost=gain / cost,
            cost=cost,
            reason=SelectionReason.NOT_SELECTED_LOWER_GAIN,
        )

    selected_blocks = tuple(graph.nodes[node_id].block for node_id in selected_ids)
    return SelectionResult(
        selected_blocks=selected_blocks,
        selection_order=order,
        decisions=tuple(decisions[node_id] for node_id in range(len(graph.nodes))),
        budget=budget,
        budget_spent=spent,
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
    optimization_node_ids: Iterable[int] | None = None,
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
    # Dense position limits are hard upper bounds, but a fixed-block cache may
    # not be able to represent every integer below that bound. Keep a private
    # mutable copy so an impossible equality can be lowered deterministically
    # without mutating the caller's preregistered physical caps.
    position_limits = (
        None
        if position_limits is None
        else {identity: int(limit) for identity, limit in position_limits.items()}
    )
    graph = objective.graph
    optimization_nodes = (
        frozenset(range(len(graph.nodes)))
        if optimization_node_ids is None
        else frozenset(int(node_id) for node_id in optimization_node_ids)
    )
    unknown_optimization_nodes = optimization_nodes - set(range(len(graph.nodes)))
    if unknown_optimization_nodes:
        raise ValueError(
            "selector optimization IDs reference nodes outside the evidence graph: "
            f"{sorted(unknown_optimization_nodes)}"
        )
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
        decisions[node_id] = SelectionDecision._from_validated(
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
        atomic: bool = False,
    ) -> None:
        nonlocal spent
        pending: list[tuple[int, int]] = []
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
            if atomic:
                pending.append((node_id, cost))
                selected.add(node_id)
                selection_order.append(node_id)
                spent += cost
                selected_positions[identity] += block.position_count
                continue
            gain = incremental.select(node_id)
            rank = len(selection_order)
            selected.add(node_id)
            selection_order.append(node_id)
            spent += cost
            selected_positions[identity] += block.position_count
            decisions[node_id] = SelectionDecision._from_validated(
                node_id=node_id,
                selected=True,
                selection_rank=rank,
                marginal_gain=gain,
                marginal_gain_per_cost=gain / cost,
                cost=cost,
                reason=reason,
            )
        if not pending:
            return
        # Prompt anchors form one simultaneous quality constraint.  Apply
        # their set-union coverage update atomically instead of replaying tens
        # of thousands of sparse Python updates.  Deterministic ranks retain
        # the sorted input order; recorded gains share the documented
        # pre-reservation objective snapshot.
        gains = incremental.reserve_many(node_id for node_id, _cost in pending)
        first_rank = len(selection_order) - len(pending)
        for offset, ((node_id, cost), gain) in enumerate(
            zip(pending, gains, strict=True)
        ):
            decisions[node_id] = SelectionDecision._from_validated(
                node_id=node_id,
                selected=True,
                selection_rank=first_rank + offset,
                marginal_gain=gain,
                marginal_gain_per_cost=gain / cost,
                cost=cost,
                reason=reason,
            )

    # Prompt anchors are a quality guard, evidence nodes preserve fine local
    # structure, and prototype anchors are representation constraints. All are
    # auditable hard reservations inside the same active budget.
    # A modality boundary can produce a short block that is nevertheless the
    # highest-attention Prompt Top-K choice. Preserve it exactly and let the
    # deterministic residue-completion pass below fill the remaining dense
    # head capacity, just as it already does for partial prototype anchors.
    select_required_nodes(
        prompt_anchors,
        SelectionReason.PROMPT_ANCHOR,
        "prompt anchor",
        allow_partial=True,
        atomic=True,
    )
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
                if node_id not in optimization_nodes:
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
                # No eligible boundary fragment can complete the advertised
                # limit. Select the largest lower capacity reachable from the
                # already required positions plus eligible aligned blocks.
                # The dynamic program stores the minimum budget cost for each
                # attainable additional position count; node IDs are included
                # only to make equal-cost choices deterministic.
                attainable: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
                for node_id in sorted(nodes_by_identity.get(identity, ())):
                    if node_id in selected or node_id not in optimization_nodes:
                        continue
                    block = graph.nodes[node_id].block
                    if block.position_count % position_fill_quantum != 0:
                        continue
                    cost = costs[node_id]
                    aligned_updates = dict(attainable)
                    for positions, (current_cost, identifiers) in attainable.items():
                        candidate_positions = positions + block.position_count
                        candidate_cost = current_cost + cost
                        if (
                            candidate_positions > remaining_positions
                            or candidate_cost > remaining_budget
                        ):
                            continue
                        aligned_candidate = (
                            candidate_cost,
                            (*identifiers, node_id),
                        )
                        aligned_previous = aligned_updates.get(candidate_positions)
                        if aligned_previous is None or aligned_candidate < aligned_previous:
                            aligned_updates[candidate_positions] = aligned_candidate
                    attainable = aligned_updates
                attainable_positions = max(attainable)
                lowered_limit = selected_positions[identity] + attainable_positions
                if lowered_limit > limit:  # pragma: no cover - DP bound invariant
                    raise RuntimeError("dense-cache feasible capacity exceeds its hard limit")
                position_limits[identity] = lowered_limit
                continue
            for node_id in completion[3]:
                block = graph.nodes[node_id].block
                cost = costs[node_id]
                gain = incremental.select(node_id)
                rank = len(selection_order)
                selected.add(node_id)
                selection_order.append(node_id)
                spent += cost
                selected_positions[identity] += block.position_count
                decisions[node_id] = SelectionDecision._from_validated(
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
                    if node_id not in optimization_nodes:
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
                decisions[node_id] = SelectionDecision._from_validated(
                    node_id=node_id,
                    selected=True,
                    selection_rank=rank,
                    marginal_gain=gain,
                    marginal_gain_per_cost=gain / cost,
                    cost=cost,
                    reason=SelectionReason.STRATUM_RESERVE,
                )

    heap_pops = 0
    marginal_recomputations = 0
    if objective.config.algorithm == "lazy_greedy":
        heap: list[tuple[float, float, int, int, int, float]] = []
        state_version = len(selection_order)
        for node in graph.nodes:
            node_id = node.node_id
            if node_id in selected:
                continue
            if node_id not in optimization_nodes:
                continue
            if not is_position_representable(node_id):
                constrained.add(node_id)
                continue
            gain = incremental.marginal_gain(node_id)
            heapq.heappush(heap, _entry(node_id, gain, costs[node_id], state_version))

        while heap and spent < budget.value:
            (
                _negative_density,
                _negative_gain,
                cost,
                node_id,
                cached_version,
                gain,
            ) = heapq.heappop(heap)
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
            decisions[node_id] = SelectionDecision._from_validated(
                node_id=node_id,
                selected=True,
                selection_rank=rank,
                marginal_gain=gain,
                marginal_gain_per_cost=gain / cost,
                cost=cost,
                reason=SelectionReason.LAZY_GREEDY,
            )
            state_version += 1
    else:
        # Re-rank a bounded number of batches. Each pass computes all sparse
        # marginal gains against one coverage snapshot, selects a deterministic
        # budget slice, and then refreshes coverage before the next pass. This
        # removes the per-selected-node heap invalidation storm while retaining
        # one or more explicit facility-coverage correction rounds.
        rounds = objective.config.coverage_correction_rounds + 1
        for round_index in range(rounds):
            if spent >= budget.value:
                break
            ranked: list[tuple[float, float, int, int]] = []
            candidate_ids: list[int] = []
            for node in graph.nodes:
                node_id = node.node_id
                if node_id in selected:
                    continue
                if node_id not in optimization_nodes:
                    continue
                if not is_position_representable(node_id):
                    constrained.add(node_id)
                    continue
                block = node.block
                identity = (block.layer, block.kv_head)
                limit = None if position_limits is None else position_limits.get(identity)
                if limit is not None and (
                    selected_positions[identity] + block.position_count > limit
                ):
                    constrained.add(node_id)
                    continue
                cost = costs[node_id]
                if cost > budget.value - spent:
                    continue
                candidate_ids.append(node_id)
            candidate_gains = incremental.marginal_gains(candidate_ids)
            for node_id, gain in zip(candidate_ids, candidate_gains, strict=True):
                cost = costs[node_id]
                ranked.append((-gain / cost, -gain, cost, node_id))
            if not ranked:
                break
            ranked.sort()
            remaining_rounds = rounds - round_index
            remaining_budget = budget.value - spent
            round_allowance = (
                remaining_budget
                if remaining_rounds == 1
                else max(1, math.ceil(remaining_budget / remaining_rounds))
            )
            round_spent = 0
            made_progress = False
            for _density, _negative_gain, cost, node_id in ranked:
                if spent >= budget.value:
                    break
                if round_spent and round_spent + cost > round_allowance:
                    continue
                block = graph.nodes[node_id].block
                identity = (block.layer, block.kv_head)
                limit = None if position_limits is None else position_limits.get(identity)
                if limit is not None and (
                    selected_positions[identity] + block.position_count > limit
                ):
                    constrained.add(node_id)
                    continue
                if cost > budget.value - spent:
                    continue
                gain = incremental.marginal_gain(node_id)
                marginal_recomputations += 1
                if objective.config.stop_on_nonpositive_gain and gain <= 0:
                    continue
                gain = incremental.select(node_id)
                rank = len(selection_order)
                selected.add(node_id)
                selection_order.append(node_id)
                spent += cost
                round_spent += cost
                selected_positions[identity] += block.position_count
                decisions[node_id] = SelectionDecision._from_validated(
                    node_id=node_id,
                    selected=True,
                    selection_rank=rank,
                    marginal_gain=gain,
                    marginal_gain_per_cost=gain / cost,
                    cost=cost,
                    reason=SelectionReason.VECTORIZED_COVERAGE,
                )
                made_progress = True
                if round_spent >= round_allowance:
                    break
            if not made_progress:
                break

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
                    if node_id not in optimization_nodes:
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
                decisions[node_id] = SelectionDecision._from_validated(
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
    undecided_ids = tuple(
        node.node_id for node in graph.nodes if node.node_id not in decisions
    )
    final_gains = dict(
        zip(
            undecided_ids,
            incremental.marginal_gains(undecided_ids),
            strict=True,
        )
    )
    for node in graph.nodes:
        node_id = node.node_id
        if node_id in decisions:
            continue
        gain = final_gains[node_id]
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
        decisions[node_id] = SelectionDecision._from_validated(
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


__all__ = [
    "InfeasibleSelectionBudgetError",
    "lazy_greedy_select",
    "select_all_exact",
    "select_exact_subset",
]
