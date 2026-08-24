"""CPU-only primitives for bounded hierarchical logit-safe cache search.

This module does not execute a model or inspect benchmark labels. It builds
deterministic exact-cache candidates that a backend may later evaluate with an
isolated logit probe. Keeping proposal construction separate from model
execution makes byte accounting, probe limits, and rollback independently
testable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

RouteId = tuple[int, int]


class HierarchicalSelectionError(ValueError):
    """Raised when a hierarchical exact-cache plan violates an invariant."""


class ProbeBudgetExceeded(HierarchicalSelectionError):
    """Raised before a probe would exceed a preregistered hard limit."""


class CapacityLevelKind(StrEnum):
    """Semantic position of one nested route-capacity choice."""

    FULL = "full"
    COARSE_MIDPOINT = "coarse_midpoint"
    TARGET_ADJACENT = "target_adjacent"


class HierarchicalProposalKind(StrEnum):
    """Role of one cache candidate in the bounded decision process."""

    SCREENING = "screening"
    JOINT = "joint"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class RouteBlockCandidate:
    """One immutable exact KV block used by the CPU candidate planner."""

    node_id: int
    layer: int
    kv_head: int
    byte_size: int
    evidence_utility: float
    eviction_risk: float
    mandatory: bool = False
    atomic_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.node_id < 0 or self.layer < 0 or self.kv_head < 0:
            raise HierarchicalSelectionError("block and route IDs must be nonnegative")
        if self.byte_size < 1:
            raise HierarchicalSelectionError("block byte_size must be positive")
        if not math.isfinite(self.evidence_utility):
            raise HierarchicalSelectionError("block evidence utility must be finite")
        if not math.isfinite(self.eviction_risk) or self.eviction_risk < 0:
            raise HierarchicalSelectionError("block eviction risk must be finite and nonnegative")
        if self.atomic_group_id is not None and not self.atomic_group_id:
            raise HierarchicalSelectionError("atomic_group_id cannot be empty")

    @property
    def route(self) -> RouteId:
        return (self.layer, self.kv_head)

    @property
    def protected(self) -> bool:
        """Whether V100's protected core forbids this block's eviction."""

        return self.mandatory or self.atomic_group_id is not None


@dataclass(frozen=True, slots=True)
class RouteCapacityLevel:
    """One exact, nested capacity choice for a layer/KV-head route."""

    route: RouteId
    level_index: int
    kind: CapacityLevelKind
    selected_node_ids: tuple[int, ...]
    active_bytes: int
    retained_utility: float
    evicted_risk: float

    def __post_init__(self) -> None:
        if min(self.route) < 0 or self.level_index < 0:
            raise HierarchicalSelectionError("route and capacity level must be nonnegative")
        if self.selected_node_ids != tuple(sorted(set(self.selected_node_ids))):
            raise HierarchicalSelectionError("capacity-level node IDs must be sorted and unique")
        if self.active_bytes < 1:
            raise HierarchicalSelectionError("capacity-level active bytes must be positive")
        if not math.isfinite(self.retained_utility):
            raise HierarchicalSelectionError("retained utility must be finite")
        if not math.isfinite(self.evicted_risk) or self.evicted_risk < 0:
            raise HierarchicalSelectionError("evicted risk must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class RouteCapacityLadder:
    """A FullKV-to-target sequence of nested exact block sets."""

    route: RouteId
    levels: tuple[RouteCapacityLevel, ...]
    protected_node_ids: tuple[int, ...]
    target_active_bytes: int
    target_reached: bool

    def __post_init__(self) -> None:
        if not self.levels:
            raise HierarchicalSelectionError("a route ladder requires at least one level")
        if self.target_active_bytes < 1:
            raise HierarchicalSelectionError("route target bytes must be positive")
        if self.levels[0].kind is not CapacityLevelKind.FULL:
            raise HierarchicalSelectionError("the first route level must be FullKV")
        if tuple(level.level_index for level in self.levels) != tuple(range(len(self.levels))):
            raise HierarchicalSelectionError("route level indices must be contiguous")
        if any(level.route != self.route for level in self.levels):
            raise HierarchicalSelectionError("route ladder mixes layer/head routes")
        protected = frozenset(self.protected_node_ids)
        if self.protected_node_ids != tuple(sorted(protected)):
            raise HierarchicalSelectionError("protected node IDs must be sorted and unique")
        previous = frozenset(self.levels[0].selected_node_ids)
        previous_bytes = self.levels[0].active_bytes
        for level in self.levels:
            selected = frozenset(level.selected_node_ids)
            if not protected <= selected:
                raise HierarchicalSelectionError("every route level must retain the protected core")
            if not selected <= previous or level.active_bytes > previous_bytes:
                raise HierarchicalSelectionError(
                    "route levels must be nested with monotonically decreasing bytes"
                )
            previous = selected
            previous_bytes = level.active_bytes
        observed_reached = self.levels[-1].active_bytes <= self.target_active_bytes
        if observed_reached != self.target_reached:
            raise HierarchicalSelectionError("route target-reached flag is inconsistent")

    @property
    def full_level(self) -> RouteCapacityLevel:
        return self.levels[0]

    @property
    def target_level(self) -> RouteCapacityLevel:
        return self.levels[-1]

    @property
    def proposed_removed_bytes(self) -> int:
        return self.full_level.active_bytes - self.target_level.active_bytes

    def level(self, index: int) -> RouteCapacityLevel:
        if index < 0 or index >= len(self.levels):
            raise HierarchicalSelectionError("route capacity level index is out of range")
        return self.levels[index]


def _level(
    route: RouteId,
    level_index: int,
    kind: CapacityLevelKind,
    selected: frozenset[int],
    candidates: Mapping[int, RouteBlockCandidate],
) -> RouteCapacityLevel:
    all_ids = frozenset(candidates)
    removed = all_ids - selected
    return RouteCapacityLevel(
        route=route,
        level_index=level_index,
        kind=kind,
        selected_node_ids=tuple(sorted(selected)),
        active_bytes=sum(candidates[node_id].byte_size for node_id in selected),
        retained_utility=sum(candidates[node_id].evidence_utility for node_id in selected),
        evicted_risk=sum(candidates[node_id].eviction_risk for node_id in removed),
    )


def build_route_capacity_ladders(
    blocks: Sequence[RouteBlockCandidate],
    target_bytes_by_route: Mapping[RouteId, int],
) -> tuple[RouteCapacityLadder, ...]:
    """Build deterministic Full/midpoint/target exact sets for every route.

    Blocks belonging to a declared atomic group are part of the protected core
    in this first implementation. Optional blocks are evicted in ascending
    label-blind risk, then ascending utility and node-ID order. Reusing one
    eviction order guarantees set nesting across all capacity levels.
    """

    if not blocks:
        raise HierarchicalSelectionError("route ladder construction requires blocks")
    if len({block.node_id for block in blocks}) != len(blocks):
        raise HierarchicalSelectionError("source block node IDs must be globally unique")
    by_route: dict[RouteId, list[RouteBlockCandidate]] = defaultdict(list)
    for block in blocks:
        by_route[block.route].append(block)
    if set(target_bytes_by_route) != set(by_route):
        raise HierarchicalSelectionError(
            "target byte routes must exactly match the source block routes"
        )

    ladders: list[RouteCapacityLadder] = []
    for route in sorted(by_route):
        route_blocks = sorted(by_route[route], key=lambda block: block.node_id)
        candidates = {block.node_id: block for block in route_blocks}
        target = int(target_bytes_by_route[route])
        full_bytes = sum(block.byte_size for block in route_blocks)
        if target < 1 or target > full_bytes:
            raise HierarchicalSelectionError(
                "each route target must be positive and no greater than FullKV bytes"
            )
        protected = frozenset(block.node_id for block in route_blocks if block.protected)
        if not protected:
            raise HierarchicalSelectionError(
                "each route requires at least one mandatory or atomic protected block"
            )
        removable = tuple(
            sorted(
                (block for block in route_blocks if not block.protected),
                key=lambda block: (
                    block.eviction_risk,
                    block.evidence_utility,
                    block.node_id,
                ),
            )
        )
        midpoint = target + math.ceil((full_bytes - target) / 2)
        selected = set(candidates)
        active = full_bytes
        levels = [_level(route, 0, CapacityLevelKind.FULL, frozenset(selected), candidates)]
        removal_index = 0
        for kind, cap in (
            (CapacityLevelKind.COARSE_MIDPOINT, midpoint),
            (CapacityLevelKind.TARGET_ADJACENT, target),
        ):
            while active > cap and removal_index < len(removable):
                block = removable[removal_index]
                removal_index += 1
                selected.remove(block.node_id)
                active -= block.byte_size
            selected_frozen = frozenset(selected)
            if selected_frozen == frozenset(levels[-1].selected_node_ids):
                continue
            levels.append(_level(route, len(levels), kind, selected_frozen, candidates))
        ladders.append(
            RouteCapacityLadder(
                route=route,
                levels=tuple(levels),
                protected_node_ids=tuple(sorted(protected)),
                target_active_bytes=target,
                target_reached=levels[-1].active_bytes <= target,
            )
        )
    return tuple(ladders)


@dataclass(frozen=True, slots=True)
class RoutePartition:
    """A deterministic byte-balanced group of routes for one probe."""

    partition_id: str
    depth: int
    routes: tuple[RouteId, ...]
    proposed_removed_bytes: int

    def __post_init__(self) -> None:
        if not self.partition_id or self.depth < 0:
            raise HierarchicalSelectionError("partition ID and depth are invalid")
        if not self.routes or self.routes != tuple(sorted(set(self.routes))):
            raise HierarchicalSelectionError("partition routes must be sorted and unique")
        if self.proposed_removed_bytes < 1:
            raise HierarchicalSelectionError("partition must propose positive byte savings")


def partition_route_ladders(
    ladders: Sequence[RouteCapacityLadder],
    partition_count: int,
    *,
    depth: int = 0,
    id_prefix: str = "p",
) -> tuple[RoutePartition, ...]:
    """Greedily balance target-level byte savings across deterministic bins."""

    if partition_count < 1:
        raise HierarchicalSelectionError("partition_count must be positive")
    if depth < 0 or not id_prefix:
        raise HierarchicalSelectionError("partition depth and prefix are invalid")
    by_route = {ladder.route: ladder for ladder in ladders}
    if len(by_route) != len(ladders):
        raise HierarchicalSelectionError("route ladders must have unique routes")
    ordered = sorted(
        (ladder for ladder in ladders if ladder.proposed_removed_bytes > 0),
        key=lambda ladder: (-ladder.proposed_removed_bytes, ladder.route),
    )
    if not ordered:
        return ()
    count = min(partition_count, len(ordered))
    bins: list[list[RouteId]] = [[] for _ in range(count)]
    totals = [0] * count
    for ladder in ordered:
        index = min(range(count), key=lambda item: (totals[item], item))
        bins[index].append(ladder.route)
        totals[index] += ladder.proposed_removed_bytes
    return tuple(
        RoutePartition(
            partition_id=f"{id_prefix}{index}",
            depth=depth,
            routes=tuple(sorted(routes)),
            proposed_removed_bytes=totals[index],
        )
        for index, routes in enumerate(bins)
        if routes
    )


def split_route_partition(
    partition: RoutePartition,
    ladders_by_route: Mapping[RouteId, RouteCapacityLadder],
) -> tuple[RoutePartition, ...]:
    """Split one unsafe partition into at most two byte-balanced children."""

    if len(partition.routes) <= 1:
        return ()
    try:
        ladders = tuple(ladders_by_route[route] for route in partition.routes)
    except KeyError as error:
        raise HierarchicalSelectionError("partition references an unknown route") from error
    return partition_route_ladders(
        ladders,
        2,
        depth=partition.depth + 1,
        id_prefix=f"{partition.partition_id}.",
    )


@dataclass(frozen=True, slots=True)
class ScreeningUpdate:
    """One pure transition after a complete hierarchy probe round."""

    accepted_partitions: tuple[RoutePartition, ...]
    next_partitions: tuple[RoutePartition, ...]
    unresolved_exact_routes: tuple[RouteId, ...]


def update_partition_screening(
    partitions: Sequence[RoutePartition],
    passed_by_partition_id: Mapping[str, bool],
    ladders_by_route: Mapping[RouteId, RouteCapacityLadder],
    *,
    maximum_refinement_depth: int,
) -> ScreeningUpdate:
    """Accept safe partitions and deterministically refine unsafe partitions."""

    if maximum_refinement_depth < 0:
        raise HierarchicalSelectionError("maximum refinement depth cannot be negative")
    expected = {partition.partition_id for partition in partitions}
    if set(passed_by_partition_id) != expected:
        raise HierarchicalSelectionError("screening outcomes must cover the complete round")
    accepted: list[RoutePartition] = []
    pending: list[RoutePartition] = []
    unresolved: set[RouteId] = set()
    for partition in sorted(partitions, key=lambda item: item.partition_id):
        if passed_by_partition_id[partition.partition_id]:
            accepted.append(partition)
            continue
        children = (
            split_route_partition(partition, ladders_by_route)
            if partition.depth < maximum_refinement_depth
            else ()
        )
        if children:
            pending.extend(children)
        else:
            unresolved.update(partition.routes)
    return ScreeningUpdate(
        accepted_partitions=tuple(accepted),
        next_partitions=tuple(sorted(pending, key=lambda item: item.partition_id)),
        unresolved_exact_routes=tuple(sorted(unresolved)),
    )


@dataclass(frozen=True, slots=True)
class LogitProbeResult:
    """Backend-neutral measurements for one isolated candidate cache."""

    greedy_token_match: bool
    maximum_forward_kl: float
    maximum_absolute_logit_difference: float
    evaluated_steps: int

    def __post_init__(self) -> None:
        if self.evaluated_steps < 1:
            raise HierarchicalSelectionError("a logit probe must evaluate at least one step")
        if (
            not math.isfinite(self.maximum_forward_kl)
            or self.maximum_forward_kl < 0
            or not math.isfinite(self.maximum_absolute_logit_difference)
            or self.maximum_absolute_logit_difference < 0
        ):
            raise HierarchicalSelectionError("logit probe measurements must be nonnegative")

    def passes(
        self,
        *,
        maximum_forward_kl: float,
        maximum_absolute_logit_difference: float,
        required_steps: int,
    ) -> bool:
        return bool(
            self.greedy_token_match
            and self.evaluated_steps == required_steps
            and self.maximum_forward_kl <= maximum_forward_kl
            and self.maximum_absolute_logit_difference <= maximum_absolute_logit_difference
        )


@dataclass(frozen=True, slots=True)
class ProbeRoundRecord:
    """One reserved group of candidate evaluations."""

    round_index: int
    proposal_ids: tuple[str, ...]


class ProbeBudgetTracker:
    """Fail-before-execution accounting for candidate and round budgets."""

    def __init__(self, maximum_candidate_evaluations: int, maximum_sequential_rounds: int):
        if maximum_candidate_evaluations < 1 or maximum_sequential_rounds < 1:
            raise HierarchicalSelectionError("probe budgets must be positive")
        self.maximum_candidate_evaluations = maximum_candidate_evaluations
        self.maximum_sequential_rounds = maximum_sequential_rounds
        self._records: list[ProbeRoundRecord] = []
        self._proposal_ids: set[str] = set()

    @property
    def evaluations_used(self) -> int:
        return sum(len(record.proposal_ids) for record in self._records)

    @property
    def rounds_used(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[ProbeRoundRecord, ...]:
        return tuple(self._records)

    def reserve_round(self, proposal_ids: Iterable[str]) -> ProbeRoundRecord:
        ids = tuple(proposal_ids)
        if not ids or len(set(ids)) != len(ids) or any(not item for item in ids):
            raise HierarchicalSelectionError("a probe round requires unique, nonempty proposal IDs")
        if self._proposal_ids.intersection(ids):
            raise HierarchicalSelectionError("a candidate cache cannot be probed twice")
        if self.rounds_used + 1 > self.maximum_sequential_rounds:
            raise ProbeBudgetExceeded("sequential probe-round budget would be exceeded")
        if self.evaluations_used + len(ids) > self.maximum_candidate_evaluations:
            raise ProbeBudgetExceeded("candidate-cache evaluation budget would be exceeded")
        record = ProbeRoundRecord(self.rounds_used, ids)
        self._records.append(record)
        self._proposal_ids.update(ids)
        return record


@dataclass(frozen=True, slots=True)
class HierarchicalCapacityProposal:
    """Immutable one-level-per-route candidate passed to a backend probe."""

    proposal_id: str
    kind: HierarchicalProposalKind
    level_indices: tuple[tuple[RouteId, int], ...]
    affected_routes: tuple[RouteId, ...]

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise HierarchicalSelectionError("hierarchical proposal ID cannot be empty")
        routes = tuple(route for route, _index in self.level_indices)
        if routes != tuple(sorted(set(routes))):
            raise HierarchicalSelectionError(
                "hierarchical proposal levels must cover sorted unique routes"
            )
        if any(index < 0 for _route, index in self.level_indices):
            raise HierarchicalSelectionError("hierarchical proposal levels cannot be negative")
        if self.affected_routes != tuple(sorted(set(self.affected_routes))):
            raise HierarchicalSelectionError(
                "hierarchical proposal affected routes must be sorted and unique"
            )
        if not set(self.affected_routes) <= set(routes):
            raise HierarchicalSelectionError("proposal affected routes must have capacity levels")

    def level_index_by_route(self) -> dict[RouteId, int]:
        return dict(self.level_indices)


@dataclass(frozen=True, slots=True)
class HierarchicalProbeEvaluation:
    """One proposal and its backend-neutral short-horizon result."""

    proposal: HierarchicalCapacityProposal
    result: LogitProbeResult
    passed: bool


@dataclass(frozen=True, slots=True)
class RouteCapacityChoice:
    """One selected level in the global multiple-choice solution."""

    route: RouteId
    level_index: int
    active_bytes: int
    retained_utility: float
    evicted_risk: float


@dataclass(frozen=True, slots=True)
class KnapsackSelection:
    """One exact capacity level per route with actual-byte accounting."""

    choices: tuple[RouteCapacityChoice, ...]
    fixed_active_bytes: int
    active_bytes: int
    target_active_bytes: int
    retained_utility: float
    target_reached: bool
    target_underfill_bytes: int
    target_shortfall_bytes: int

    def __post_init__(self) -> None:
        routes = tuple(choice.route for choice in self.choices)
        if routes != tuple(sorted(set(routes))):
            raise HierarchicalSelectionError("knapsack choices must cover unique sorted routes")
        if self.fixed_active_bytes < 0 or self.active_bytes < 1 or self.target_active_bytes < 1:
            raise HierarchicalSelectionError("knapsack byte counts are invalid")
        if self.active_bytes != self.fixed_active_bytes + sum(
            choice.active_bytes for choice in self.choices
        ):
            raise HierarchicalSelectionError("knapsack active-byte accounting is inconsistent")
        if self.target_reached != (self.active_bytes <= self.target_active_bytes):
            raise HierarchicalSelectionError("knapsack target-reached flag is inconsistent")
        if self.target_underfill_bytes != max(0, self.target_active_bytes - self.active_bytes):
            raise HierarchicalSelectionError("knapsack underfill accounting is inconsistent")
        if self.target_shortfall_bytes != max(0, self.active_bytes - self.target_active_bytes):
            raise HierarchicalSelectionError("knapsack shortfall accounting is inconsistent")

    def level_index_by_route(self) -> dict[RouteId, int]:
        return {choice.route: choice.level_index for choice in self.choices}


@dataclass(frozen=True, slots=True)
class _KnapsackState:
    utility: float
    level_indices: tuple[int, ...]


def select_verified_capacity_levels(
    ladders: Sequence[RouteCapacityLadder],
    verified_level_indices_by_route: Mapping[RouteId, Iterable[int]],
    target_active_bytes: int,
    *,
    fixed_active_bytes: int = 0,
    require_full_level: bool = True,
) -> KnapsackSelection:
    """Choose one verified level per route using exact sparse dynamic programming.

    The primary objective is the largest active cache at or below the target,
    avoiding unnecessary over-compression. Retained evidence utility is the
    secondary objective and lexicographic level indices break exact ties.
    """

    if target_active_bytes < 1 or fixed_active_bytes < 0:
        raise HierarchicalSelectionError("global target and fixed bytes are invalid")
    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    if len({ladder.route for ladder in ordered}) != len(ordered):
        raise HierarchicalSelectionError("knapsack route ladders must be unique")
    if set(verified_level_indices_by_route) != {ladder.route for ladder in ordered}:
        raise HierarchicalSelectionError("verified levels must cover every route")

    states: dict[int, _KnapsackState] = {0: _KnapsackState(0.0, ())}
    options_by_route: dict[RouteId, tuple[RouteCapacityLevel, ...]] = {}
    for ladder in ordered:
        indices = tuple(sorted(set(verified_level_indices_by_route[ladder.route])))
        if not indices:
            raise HierarchicalSelectionError("every route requires a verified capacity level")
        if require_full_level and 0 not in indices:
            raise HierarchicalSelectionError("FullKV level zero must always be verified")
        options = tuple(ladder.level(index) for index in indices)
        options_by_route[ladder.route] = options
        next_states: dict[int, _KnapsackState] = {}
        for previous_bytes, state in states.items():
            for level in options:
                active = previous_bytes + level.active_bytes
                candidate = _KnapsackState(
                    utility=state.utility + level.retained_utility,
                    level_indices=(*state.level_indices, level.level_index),
                )
                existing = next_states.get(active)
                if (
                    existing is None
                    or candidate.utility > existing.utility + 1e-12
                    or (
                        math.isclose(candidate.utility, existing.utility, abs_tol=1e-12)
                        and candidate.level_indices < existing.level_indices
                    )
                ):
                    next_states[active] = candidate
        states = next_states

    feasible = [
        (active, state)
        for route_bytes, state in states.items()
        if (active := route_bytes + fixed_active_bytes) <= target_active_bytes
    ]
    if feasible:
        active, selected_state = min(
            feasible,
            key=lambda item: (-item[0], -item[1].utility, item[1].level_indices),
        )
    else:
        active, selected_state = min(
            ((route_bytes + fixed_active_bytes, state) for route_bytes, state in states.items()),
            key=lambda item: (item[0], -item[1].utility, item[1].level_indices),
        )
    choices = tuple(
        RouteCapacityChoice(
            route=ladder.route,
            level_index=level_index,
            active_bytes=ladder.level(level_index).active_bytes,
            retained_utility=ladder.level(level_index).retained_utility,
            evicted_risk=ladder.level(level_index).evicted_risk,
        )
        for ladder, level_index in zip(ordered, selected_state.level_indices, strict=True)
    )
    return KnapsackSelection(
        choices=choices,
        fixed_active_bytes=fixed_active_bytes,
        active_bytes=active,
        target_active_bytes=target_active_bytes,
        retained_utility=selected_state.utility,
        target_reached=active <= target_active_bytes,
        target_underfill_bytes=max(0, target_active_bytes - active),
        target_shortfall_bytes=max(0, active - target_active_bytes),
    )


@dataclass(frozen=True, slots=True)
class ConflictRollbackPlan:
    """One bounded rollback plus verified donor exchange after a joint failure."""

    selection: KnapsackSelection
    rollback_applied: bool
    rolled_back_routes: tuple[RouteId, ...]
    donor_routes: tuple[RouteId, ...]
    bytes_added_by_rollback: int
    bytes_removed_by_donors: int
    reason: str


def plan_conflict_rollback(
    current: KnapsackSelection,
    ladders: Sequence[RouteCapacityLadder],
    verified_level_indices_by_route: Mapping[RouteId, Iterable[int]],
    *,
    previously_rolled_back_routes: Iterable[RouteId] = (),
) -> ConflictRollbackPlan:
    """Rollback the highest-risk compressed route and exchange verified donors.

    Each candidate route is tried in deterministic descending output-impact
    risk. The rolled-back route moves to its nearest safer verified level;
    unverified intermediate levels are never inferred safe. Other routes may
    stay fixed or move only to a more aggressive verified level. The first
    route admitting a target-feasible exchange is returned.
    """

    by_route = {ladder.route: ladder for ladder in ladders}
    if set(by_route) != set(current.level_index_by_route()):
        raise HierarchicalSelectionError("rollback ladders do not match current routes")
    verified = {
        route: tuple(sorted(set(indices)))
        for route, indices in verified_level_indices_by_route.items()
    }
    if set(verified) != set(by_route):
        raise HierarchicalSelectionError("rollback verified levels do not cover every route")
    current_indices = current.level_index_by_route()
    excluded = frozenset(previously_rolled_back_routes)
    rollback_candidates = sorted(
        (
            choice
            for choice in current.choices
            if choice.level_index > 0 and choice.route not in excluded
        ),
        key=lambda choice: (-choice.evicted_risk, choice.route),
    )
    for rollback_choice in rollback_candidates:
        route = rollback_choice.route
        safer_verified = tuple(
            index for index in verified[route] if index < rollback_choice.level_index
        )
        if not safer_verified:
            continue
        rollback_index = max(safer_verified)
        allowed: dict[RouteId, tuple[int, ...]] = {}
        for candidate_route in sorted(by_route):
            current_index = current_indices[candidate_route]
            if candidate_route == route:
                allowed[candidate_route] = (rollback_index,)
            else:
                allowed[candidate_route] = tuple(
                    index for index in verified[candidate_route] if index >= current_index
                )
                if current_index not in allowed[candidate_route]:
                    raise HierarchicalSelectionError(
                        "the current route level must remain a verified rollback option"
                    )
        candidate = select_verified_capacity_levels(
            tuple(by_route[route_id] for route_id in sorted(by_route)),
            allowed,
            current.target_active_bytes,
            fixed_active_bytes=current.fixed_active_bytes,
            require_full_level=False,
        )
        if not candidate.target_reached:
            continue
        new_indices = candidate.level_index_by_route()
        donors = tuple(
            sorted(
                candidate_route
                for candidate_route, index in new_indices.items()
                if candidate_route != route and index > current_indices[candidate_route]
            )
        )
        rollback_old = by_route[route].level(rollback_choice.level_index).active_bytes
        rollback_new = by_route[route].level(rollback_index).active_bytes
        return ConflictRollbackPlan(
            selection=candidate,
            rollback_applied=True,
            rolled_back_routes=(route,),
            donor_routes=donors,
            bytes_added_by_rollback=rollback_new - rollback_old,
            bytes_removed_by_donors=sum(
                by_route[donor].level(current_indices[donor]).active_bytes
                - by_route[donor].level(new_indices[donor]).active_bytes
                for donor in donors
            ),
            reason="highest_risk_route_rolled_back_with_verified_donor_exchange",
        )
    return ConflictRollbackPlan(
        selection=current,
        rollback_applied=False,
        rolled_back_routes=(),
        donor_routes=(),
        bytes_added_by_rollback=0,
        bytes_removed_by_donors=0,
        reason="no_target_feasible_verified_rollback_exchange",
    )


@dataclass(frozen=True, slots=True)
class BoundedHierarchicalProbeDecision:
    """Complete fail-closed outcome of the CPU hierarchy controller."""

    selection: KnapsackSelection
    evaluations: tuple[HierarchicalProbeEvaluation, ...]
    accepted_target_routes: tuple[RouteId, ...]
    unresolved_exact_routes: tuple[RouteId, ...]
    rollback_attempts: tuple[ConflictRollbackPlan, ...]
    target_reached: bool
    selected_candidate_passed: bool
    final_joint_probe_passed: bool
    used_fullkv_fallback: bool
    evaluations_used: int
    rounds_used: int

    def __post_init__(self) -> None:
        if self.evaluations_used != len(self.evaluations):
            raise HierarchicalSelectionError("hierarchical evaluation count is inconsistent")
        if self.evaluations_used < 1 or self.rounds_used < 1:
            raise HierarchicalSelectionError("hierarchical decision requires a joint probe")
        if self.target_reached != self.selection.target_reached:
            raise HierarchicalSelectionError("hierarchical target status is inconsistent")
        if self.used_fullkv_fallback and self.target_reached:
            raise HierarchicalSelectionError("FullKV fallback cannot claim the compressed target")
        if self.final_joint_probe_passed and not self.selected_candidate_passed:
            raise HierarchicalSelectionError("a passing joint probe must select a passing cache")


def _proposal(
    proposal_id: str,
    kind: HierarchicalProposalKind,
    ordered_ladders: Sequence[RouteCapacityLadder],
    level_indices: Mapping[RouteId, int],
    affected_routes: Iterable[RouteId],
) -> HierarchicalCapacityProposal:
    if set(level_indices) != {ladder.route for ladder in ordered_ladders}:
        raise HierarchicalSelectionError("proposal levels must cover every route")
    for ladder in ordered_ladders:
        ladder.level(level_indices[ladder.route])
    return HierarchicalCapacityProposal(
        proposal_id=proposal_id,
        kind=kind,
        level_indices=tuple(
            (ladder.route, level_indices[ladder.route]) for ladder in ordered_ladders
        ),
        affected_routes=tuple(sorted(set(affected_routes))),
    )


def run_bounded_hierarchical_probe_loop(
    ladders: Sequence[RouteCapacityLadder],
    *,
    target_active_bytes: int,
    initial_partition_count: int,
    maximum_refinement_depth: int,
    maximum_rollback_rounds: int,
    maximum_forward_kl: float,
    maximum_absolute_logit_difference: float,
    required_steps: int,
    probe_budget: ProbeBudgetTracker,
    probe: Callable[[HierarchicalCapacityProposal], LogitProbeResult],
) -> BoundedHierarchicalProbeDecision:
    """Screen, assemble, jointly verify, and fail closed within hard budgets.

    Screening probes each route partition at its most aggressive ladder level
    while every other route remains FullKV. Refinement is truncated before it
    can consume the one evaluation and one round reserved for joint checking.
    Only explicitly passing levels enter the knapsack or rollback donor pool.
    """

    if target_active_bytes < 1 or required_steps < 1:
        raise HierarchicalSelectionError("hierarchical target and probe horizon must be positive")
    if maximum_rollback_rounds < 0:
        raise HierarchicalSelectionError("maximum rollback rounds cannot be negative")
    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    if not ordered or len({ladder.route for ladder in ordered}) != len(ordered):
        raise HierarchicalSelectionError("hierarchical probing requires unique route ladders")
    by_route = {ladder.route: ladder for ladder in ordered}
    verified: dict[RouteId, set[int]] = {ladder.route: {0} for ladder in ordered}
    accepted_routes: set[RouteId] = set()
    unresolved_routes: set[RouteId] = set()
    evaluations: list[HierarchicalProbeEvaluation] = []

    def evaluate(proposal: HierarchicalCapacityProposal) -> bool:
        result = probe(proposal)
        passed = result.passes(
            maximum_forward_kl=maximum_forward_kl,
            maximum_absolute_logit_difference=maximum_absolute_logit_difference,
            required_steps=required_steps,
        )
        evaluations.append(HierarchicalProbeEvaluation(proposal, result, passed))
        return passed

    pending = partition_route_ladders(ordered, initial_partition_count)
    while pending:
        if probe_budget.rounds_used >= probe_budget.maximum_sequential_rounds - 1:
            unresolved_routes.update(route for item in pending for route in item.routes)
            break
        available = probe_budget.maximum_candidate_evaluations - probe_budget.evaluations_used - 1
        if available <= 0:
            unresolved_routes.update(route for item in pending for route in item.routes)
            break
        current = pending[:available]
        skipped = pending[available:]
        unresolved_routes.update(route for item in skipped for route in item.routes)
        probe_budget.reserve_round(item.partition_id for item in current)
        outcomes: dict[str, bool] = {}
        for partition in current:
            levels = {ladder.route: 0 for ladder in ordered}
            for route in partition.routes:
                levels[route] = by_route[route].target_level.level_index
            proposal = _proposal(
                partition.partition_id,
                HierarchicalProposalKind.SCREENING,
                ordered,
                levels,
                partition.routes,
            )
            passed = evaluate(proposal)
            outcomes[partition.partition_id] = passed
            if passed:
                for route in partition.routes:
                    verified[route].add(by_route[route].target_level.level_index)
                    accepted_routes.add(route)
        update = update_partition_screening(
            current,
            outcomes,
            by_route,
            maximum_refinement_depth=maximum_refinement_depth,
        )
        unresolved_routes.update(update.unresolved_exact_routes)
        pending = update.next_partitions

    assembled = select_verified_capacity_levels(
        ordered,
        verified,
        target_active_bytes,
    )
    joint_proposal = _proposal(
        "joint-0",
        HierarchicalProposalKind.JOINT,
        ordered,
        assembled.level_index_by_route(),
        (choice.route for choice in assembled.choices if choice.level_index > 0),
    )
    probe_budget.reserve_round((joint_proposal.proposal_id,))
    joint_passed = evaluate(joint_proposal)
    rollback_attempts: list[ConflictRollbackPlan] = []
    selected = assembled
    selected_passed = joint_passed
    rolled_back_routes: set[RouteId] = set()
    if not joint_passed:
        for rollback_index in range(maximum_rollback_rounds):
            if (
                probe_budget.rounds_used >= probe_budget.maximum_sequential_rounds
                or probe_budget.evaluations_used >= probe_budget.maximum_candidate_evaluations
            ):
                break
            rollback = plan_conflict_rollback(
                assembled,
                ordered,
                verified,
                previously_rolled_back_routes=rolled_back_routes,
            )
            rollback_attempts.append(rollback)
            if not rollback.rollback_applied:
                break
            rolled_back_routes.update(rollback.rolled_back_routes)
            rollback_proposal = _proposal(
                f"rollback-{rollback_index}",
                HierarchicalProposalKind.ROLLBACK,
                ordered,
                rollback.selection.level_index_by_route(),
                (*rollback.rolled_back_routes, *rollback.donor_routes),
            )
            probe_budget.reserve_round((rollback_proposal.proposal_id,))
            if evaluate(rollback_proposal):
                selected = rollback.selection
                selected_passed = True
                break

    used_fullkv_fallback = not selected_passed
    if used_fullkv_fallback:
        selected = select_verified_capacity_levels(
            ordered,
            {ladder.route: (0,) for ladder in ordered},
            target_active_bytes,
        )
    return BoundedHierarchicalProbeDecision(
        selection=selected,
        evaluations=tuple(evaluations),
        accepted_target_routes=tuple(sorted(accepted_routes)),
        unresolved_exact_routes=tuple(sorted(unresolved_routes)),
        rollback_attempts=tuple(rollback_attempts),
        target_reached=selected.target_reached,
        selected_candidate_passed=selected_passed,
        final_joint_probe_passed=joint_passed,
        used_fullkv_fallback=used_fullkv_fallback,
        evaluations_used=probe_budget.evaluations_used,
        rounds_used=probe_budget.rounds_used,
    )


__all__ = [
    "BoundedHierarchicalProbeDecision",
    "CapacityLevelKind",
    "ConflictRollbackPlan",
    "HierarchicalCapacityProposal",
    "HierarchicalProbeEvaluation",
    "HierarchicalProposalKind",
    "HierarchicalSelectionError",
    "KnapsackSelection",
    "LogitProbeResult",
    "ProbeBudgetExceeded",
    "ProbeBudgetTracker",
    "ProbeRoundRecord",
    "RouteBlockCandidate",
    "RouteCapacityChoice",
    "RouteCapacityLadder",
    "RouteCapacityLevel",
    "RouteId",
    "RoutePartition",
    "ScreeningUpdate",
    "build_route_capacity_ladders",
    "partition_route_ladders",
    "plan_conflict_rollback",
    "run_bounded_hierarchical_probe_loop",
    "select_verified_capacity_levels",
    "split_route_partition",
    "update_partition_screening",
]
