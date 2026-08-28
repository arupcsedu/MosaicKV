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
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import combinations, islice

RouteId = tuple[int, int]


class HierarchicalSelectionError(ValueError):
    """Raised when a hierarchical exact-cache plan violates an invariant."""


class ProbeBudgetExceeded(HierarchicalSelectionError):
    """Raised before a probe would exceed a preregistered hard limit."""


class CapacityLevelKind(StrEnum):
    """Semantic position of one nested route-capacity choice."""

    FULL = "full"
    FEASIBLE_BASELINE = "feasible_baseline"
    COARSE_MIDPOINT = "coarse_midpoint"
    TARGET_ADJACENT = "target_adjacent"
    REALLOCATION_ALTERNATIVE = "reallocation_alternative"


class HierarchicalProposalKind(StrEnum):
    """Role of one cache candidate in the bounded decision process."""

    SCREENING = "screening"
    JOINT = "joint"
    ROLLBACK = "rollback"
    REALLOCATION = "reallocation"


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
        if self.levels[0].kind not in {
            CapacityLevelKind.FULL,
            CapacityLevelKind.FEASIBLE_BASELINE,
        }:
            raise HierarchicalSelectionError(
                "the first route level must be FullKV or a feasible baseline"
            )
        if tuple(level.level_index for level in self.levels) != tuple(range(len(self.levels))):
            raise HierarchicalSelectionError("route level indices must be contiguous")
        if any(level.route != self.route for level in self.levels):
            raise HierarchicalSelectionError("route ladder mixes layer/head routes")
        protected = frozenset(self.protected_node_ids)
        if self.protected_node_ids != tuple(sorted(protected)):
            raise HierarchicalSelectionError("protected node IDs must be sorted and unique")
        previous = frozenset(self.levels[0].selected_node_ids)
        previous_bytes = self.levels[0].active_bytes
        target_level: RouteCapacityLevel | None = None
        ordinary_bytes: set[int] = set()
        for level in self.levels:
            selected = frozenset(level.selected_node_ids)
            if not protected <= selected:
                raise HierarchicalSelectionError("every route level must retain the protected core")
            if level.kind is CapacityLevelKind.REALLOCATION_ALTERNATIVE:
                if not ordinary_bytes:
                    raise HierarchicalSelectionError(
                        "reallocation alternatives require an ordinary anchor level"
                    )
                if level.active_bytes not in ordinary_bytes:
                    raise HierarchicalSelectionError(
                        "reallocation alternatives must match an ordinary route capacity"
                    )
                continue
            if not selected <= previous or level.active_bytes > previous_bytes:
                raise HierarchicalSelectionError(
                    "route levels must be nested with monotonically decreasing bytes"
                )
            if target_level is None and level.active_bytes <= self.target_active_bytes:
                target_level = level
            previous = selected
            previous_bytes = level.active_bytes
            ordinary_bytes.add(level.active_bytes)
        observed_reached = target_level is not None
        if observed_reached != self.target_reached:
            raise HierarchicalSelectionError("route target-reached flag is inconsistent")

    @property
    def full_level(self) -> RouteCapacityLevel:
        return self.levels[0]

    @property
    def target_level(self) -> RouteCapacityLevel:
        return next(
            (level for level in self.levels if level.active_bytes <= self.target_active_bytes),
            self.levels[-1],
        )

    @property
    def reallocation_levels(self) -> tuple[RouteCapacityLevel, ...]:
        """Return same-capacity alternatives reserved for failed joint probes."""

        return tuple(
            level
            for level in self.levels
            if level.kind is CapacityLevelKind.REALLOCATION_ALTERNATIVE
        )

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
        target_level = next(
            (level for level in levels if level.active_bytes <= target),
            None,
        )
        if target_level is not None:
            target_selected = frozenset(target_level.selected_node_ids)
            selected_optional = sorted(
                (
                    candidates[node_id]
                    for node_id in target_selected
                    if node_id not in protected
                ),
                key=lambda block: (
                    block.eviction_risk,
                    block.evidence_utility,
                    block.node_id,
                ),
            )
            removed_optional = sorted(
                (
                    block
                    for block in route_blocks
                    if block.node_id not in target_selected and not block.protected
                ),
                key=lambda block: (
                    -block.eviction_risk,
                    -block.evidence_utility,
                    block.node_id,
                ),
            )
            used_selected: set[int] = set()
            # Two deterministic boundary swaps provide bounded alternatives
            # without changing route length or dense HF storage.  Each swap
            # restores the highest-risk omitted block of a given byte size
            # and displaces the lowest-risk retained optional block with the
            # same byte size.  The alternatives are never inferred safe; a
            # complete joint candidate must pass the online probe.
            for removed in removed_optional:
                displaced = next(
                    (
                        block
                        for block in selected_optional
                        if block.node_id not in used_selected
                        and block.byte_size == removed.byte_size
                    ),
                    None,
                )
                if displaced is None:
                    continue
                alternative = frozenset(
                    (target_selected - {displaced.node_id}) | {removed.node_id}
                )
                levels.append(
                    _level(
                        route,
                        len(levels),
                        CapacityLevelKind.REALLOCATION_ALTERNATIVE,
                        alternative,
                        candidates,
                    )
                )
                used_selected.add(displaced.node_id)
                if len(used_selected) >= 2:
                    break
        ladders.append(
            RouteCapacityLadder(
                route=route,
                levels=tuple(levels),
                protected_node_ids=tuple(sorted(protected)),
                target_active_bytes=target,
                target_reached=target_level is not None,
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
class RouteCapacityViolation:
    """One proposal route whose selected positions exceed its physical limit."""

    route: RouteId
    selected_positions: int
    position_limit: int

    def __post_init__(self) -> None:
        if min(self.route) < 0:
            raise HierarchicalSelectionError("capacity-violation route must be nonnegative")
        if self.selected_positions < 0 or self.position_limit < 1:
            raise HierarchicalSelectionError("route position counts must be valid")
        if self.selected_positions <= self.position_limit:
            raise HierarchicalSelectionError("a route-capacity violation must exceed its limit")


def proposal_route_capacity_violations(
    proposal: HierarchicalCapacityProposal,
    ladders: Sequence[RouteCapacityLadder],
    position_count_by_node: Mapping[int, int],
    position_limits: Mapping[RouteId, int],
    *,
    affected_routes_only: bool = False,
) -> tuple[RouteCapacityViolation, ...]:
    """Check selected positions before a proposal reaches a model probe.

    Screening candidates intentionally keep unrelated routes at FullKV, so a
    screening call checks only its affected routes. Joint, rollback, and
    reallocation candidates check every route because a passing candidate can
    become the final packed cache.
    """

    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    by_route = {ladder.route: ladder for ladder in ordered}
    indices = proposal.level_index_by_route()
    if set(indices) != set(by_route) or set(position_limits) != set(by_route):
        raise HierarchicalSelectionError(
            "proposal, ladder, and position-limit routes must match exactly"
        )
    checked_routes = (
        frozenset(proposal.affected_routes) if affected_routes_only else frozenset(by_route)
    )
    violations: list[RouteCapacityViolation] = []
    for route in sorted(checked_routes):
        level = by_route[route].level(indices[route])
        try:
            counts = tuple(
                int(position_count_by_node[node_id]) for node_id in level.selected_node_ids
            )
        except KeyError as error:
            raise HierarchicalSelectionError(
                f"missing position count for selected node {error.args[0]}"
            ) from error
        if any(count < 1 for count in counts):
            raise HierarchicalSelectionError("selected node position counts must be positive")
        selected_positions = sum(counts)
        limit = int(position_limits[route])
        if limit < 1:
            raise HierarchicalSelectionError("route position limits must be positive")
        if selected_positions > limit:
            violations.append(RouteCapacityViolation(route, selected_positions, limit))
    return tuple(violations)


def restrict_route_capacity_ladders_to_position_limits(
    ladders: Sequence[RouteCapacityLadder],
    position_count_by_node: Mapping[int, int],
    position_limits: Mapping[RouteId, int],
) -> tuple[RouteCapacityLadder, ...]:
    """Build final-candidate ladders containing only representable levels.

    FullKV remains the reference state outside this returned view.  For every
    route, the first returned level is the largest existing level whose exact
    selected positions fit the route's physical cache extent.  All safer or
    reallocation levels that exceed that extent are removed before knapsack
    assembly or restore/donor proposal generation.
    """

    ordered = tuple(sorted(ladders, key=lambda item: item.route))
    by_route = {ladder.route: ladder for ladder in ordered}
    if not ordered or len(by_route) != len(ordered):
        raise HierarchicalSelectionError("position filtering requires unique route ladders")
    if set(position_limits) != set(by_route):
        raise HierarchicalSelectionError(
            "position limits must exactly cover the route ladders"
        )
    result: list[RouteCapacityLadder] = []
    for ladder in ordered:
        limit = int(position_limits[ladder.route])
        if limit < 1:
            raise HierarchicalSelectionError("route position limits must be positive")

        def fits(level: RouteCapacityLevel, position_limit: int = limit) -> bool:
            try:
                counts = tuple(
                    int(position_count_by_node[node_id])
                    for node_id in level.selected_node_ids
                )
            except KeyError as error:
                raise HierarchicalSelectionError(
                    f"missing position count for selected node {error.args[0]}"
                ) from error
            if any(count < 1 for count in counts):
                raise HierarchicalSelectionError(
                    "selected node position counts must be positive"
                )
            return sum(counts) <= position_limit

        feasible_ordinary = tuple(
            level
            for level in ladder.levels
            if level.kind is not CapacityLevelKind.REALLOCATION_ALTERNATIVE
            and fits(level)
        )
        if not feasible_ordinary:
            raise HierarchicalSelectionError(
                f"route {ladder.route} has no physically feasible protected level"
            )
        baseline = feasible_ordinary[0]
        kept: list[RouteCapacityLevel] = list(feasible_ordinary)
        if any(level.active_bytes <= ladder.target_active_bytes for level in feasible_ordinary):
            kept.extend(
                level
                for level in ladder.reallocation_levels
                if fits(level)
            )
        rebuilt: list[RouteCapacityLevel] = []
        for index, level in enumerate(kept):
            kind = level.kind
            if index == 0 and kind is not CapacityLevelKind.FULL:
                kind = CapacityLevelKind.FEASIBLE_BASELINE
            rebuilt.append(replace(level, level_index=index, kind=kind))
        target_reached = any(
            level.kind is not CapacityLevelKind.REALLOCATION_ALTERNATIVE
            and level.active_bytes <= ladder.target_active_bytes
            for level in rebuilt
        )
        result.append(
            RouteCapacityLadder(
                route=ladder.route,
                levels=tuple(rebuilt),
                protected_node_ids=ladder.protected_node_ids,
                target_active_bytes=ladder.target_active_bytes,
                target_reached=target_reached,
            )
        )
        if result[-1].full_level.active_bytes != baseline.active_bytes:
            raise HierarchicalSelectionError(
                "position filtering did not preserve the largest feasible baseline"
            )
    return tuple(result)


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


def _selection_from_level_indices(
    ladders: Sequence[RouteCapacityLadder],
    level_indices: Mapping[RouteId, int],
    target_active_bytes: int,
    *,
    fixed_active_bytes: int = 0,
) -> KnapsackSelection:
    """Materialize one deterministic selection from explicit route levels."""

    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    if set(level_indices) != {ladder.route for ladder in ordered}:
        raise HierarchicalSelectionError("explicit levels must cover every route")
    choices = tuple(
        RouteCapacityChoice(
            route=ladder.route,
            level_index=level_indices[ladder.route],
            active_bytes=ladder.level(level_indices[ladder.route]).active_bytes,
            retained_utility=ladder.level(level_indices[ladder.route]).retained_utility,
            evicted_risk=ladder.level(level_indices[ladder.route]).evicted_risk,
        )
        for ladder in ordered
    )
    active_bytes = fixed_active_bytes + sum(choice.active_bytes for choice in choices)
    return KnapsackSelection(
        choices=choices,
        fixed_active_bytes=fixed_active_bytes,
        active_bytes=active_bytes,
        target_active_bytes=target_active_bytes,
        retained_utility=sum(choice.retained_utility for choice in choices),
        target_reached=active_bytes <= target_active_bytes,
        target_underfill_bytes=max(0, target_active_bytes - active_bytes),
        target_shortfall_bytes=max(0, active_bytes - target_active_bytes),
    )


def plan_joint_reallocation_candidates(
    current: KnapsackSelection,
    ladders: Sequence[RouteCapacityLadder],
    *,
    rollback_width: int,
    maximum_candidates: int,
    excluded_level_signatures: Iterable[tuple[tuple[RouteId, int], ...]] = (),
) -> tuple[ConflictRollbackPlan, ...]:
    """Build bounded fixed-capacity alternatives after a failed joint probe.

    Each alternative changes the eviction boundary in ``rollback_width``
    high-risk compressed routes while retaining exactly the same bytes in
    every route.  This is compatible with the dense HF cache layout: no head
    exceeds the common physical sequence capacity and no padded storage is
    hidden from byte accounting.  Alternatives are *not* inferred safe; the
    complete reallocated cache must pass an isolated joint logit probe.
    """

    if rollback_width < 1 or maximum_candidates < 1:
        raise HierarchicalSelectionError("reallocation width and candidate cap must be positive")
    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    by_route = {ladder.route: ladder for ladder in ordered}
    current_indices = current.level_index_by_route()
    if set(by_route) != set(current_indices):
        raise HierarchicalSelectionError("reallocation ladders do not match current routes")
    excluded = frozenset(excluded_level_signatures)
    rollback_routes = tuple(
        choice.route
        for choice in sorted(
            (
                choice
                for choice in current.choices
                if any(
                    level.active_bytes == choice.active_bytes
                    for level in by_route[choice.route].reallocation_levels
                )
            ),
            key=lambda choice: (-choice.evicted_risk, choice.route),
        )
    )
    if len(rollback_routes) < rollback_width:
        return ()

    # Inspect only a small deterministic multiple of the output cap.  This
    # keeps proposal construction bounded even for thousands of layer/head
    # routes while allowing infeasible or duplicate alternatives to be
    # skipped.
    combination_cap = max(maximum_candidates, maximum_candidates * 8)
    route_sets = islice(combinations(rollback_routes, rollback_width), combination_cap)
    plans: list[ConflictRollbackPlan] = []
    seen: set[tuple[tuple[RouteId, int], ...]] = set(excluded)
    for restored_routes_tuple in route_sets:
        restored_routes = frozenset(restored_routes_tuple)
        indices = dict(current_indices)
        for route in restored_routes_tuple:
            current_bytes = next(
                choice.active_bytes for choice in current.choices if choice.route == route
            )
            alternatives = tuple(
                level
                for level in by_route[route].reallocation_levels
                if level.active_bytes == current_bytes
            )
            alternative_index = min(rollback_width - 1, len(alternatives) - 1)
            indices[route] = alternatives[alternative_index].level_index
        signature = tuple((route, indices[route]) for route in sorted(indices))
        if signature in seen:
            continue
        seen.add(signature)
        selection = _selection_from_level_indices(
            ordered,
            indices,
            current.target_active_bytes,
            fixed_active_bytes=current.fixed_active_bytes,
        )
        if selection.active_bytes != current.active_bytes:
            raise HierarchicalSelectionError(
                "fixed-capacity reallocation changed global active bytes"
            )
        plans.append(
            ConflictRollbackPlan(
                selection=selection,
                rollback_applied=True,
                rolled_back_routes=tuple(sorted(restored_routes)),
                donor_routes=(),
                bytes_added_by_rollback=0,
                bytes_removed_by_donors=0,
                reason=(
                    "joint_failure_reallocated_eviction_boundaries_at_fixed_route_capacity"
                ),
            )
        )
        if len(plans) >= maximum_candidates:
            break
    return tuple(plans)


def _logit_stability_rank(
    evaluation: HierarchicalProbeEvaluation,
    *,
    maximum_forward_kl: float,
    maximum_absolute_logit_difference: float,
    required_steps: int,
) -> tuple[bool, bool, float, float, float, str]:
    """Rank failed candidates using label-blind short-horizon stability only."""

    result = evaluation.result
    normalized_kl = result.maximum_forward_kl / maximum_forward_kl
    normalized_logit = (
        result.maximum_absolute_logit_difference / maximum_absolute_logit_difference
    )
    return (
        result.evaluated_steps != required_steps,
        not result.greedy_token_match,
        max(normalized_kl, normalized_logit),
        normalized_logit,
        normalized_kl,
        evaluation.proposal.proposal_id,
    )


def plan_beam_reallocation_candidates(
    current: KnapsackSelection,
    ladders: Sequence[RouteCapacityLadder],
    single_route_evaluations: Sequence[HierarchicalProbeEvaluation],
    *,
    beam_width: int,
    maximum_combination_width: int,
    maximum_candidates: int,
    maximum_forward_kl: float,
    maximum_absolute_logit_difference: float,
    required_steps: int,
    excluded_level_signatures: Iterable[tuple[tuple[RouteId, int], ...]] = (),
) -> tuple[ConflictRollbackPlan, ...]:
    """Combine the most stable label-blind single swaps into a bounded beam.

    No benchmark answer, reference, task score, or failure identity is an
    input. The ordering depends only on the isolated short-horizon token,
    KL, and logit-difference diagnostics already required by the safety gate.
    Every composed candidate preserves the current byte count in each route.
    """

    if beam_width < 2:
        raise HierarchicalSelectionError("reallocation beam width must be at least two")
    if not 2 <= maximum_combination_width <= min(3, beam_width):
        raise HierarchicalSelectionError(
            "reallocation combination width must be between two and min(three, beam width)"
        )
    if maximum_candidates < 1:
        raise HierarchicalSelectionError("reallocation beam candidate cap must be positive")
    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    current_indices = current.level_index_by_route()
    if {ladder.route for ladder in ordered} != set(current_indices):
        raise HierarchicalSelectionError("reallocation beam ladders do not match current routes")
    eligible: list[HierarchicalProbeEvaluation] = []
    seen_routes: set[RouteId] = set()
    for evaluation in sorted(
        single_route_evaluations,
        key=lambda item: _logit_stability_rank(
            item,
            maximum_forward_kl=maximum_forward_kl,
            maximum_absolute_logit_difference=maximum_absolute_logit_difference,
            required_steps=required_steps,
        ),
    ):
        proposal = evaluation.proposal
        if proposal.kind is not HierarchicalProposalKind.REALLOCATION:
            raise HierarchicalSelectionError("beam inputs must be reallocation probes")
        if len(proposal.affected_routes) != 1:
            raise HierarchicalSelectionError("beam inputs must change exactly one route")
        route = proposal.affected_routes[0]
        if route in seen_routes:
            continue
        indices = proposal.level_index_by_route()
        if set(indices) != set(current_indices):
            raise HierarchicalSelectionError("beam input does not cover every route")
        if any(
            indices[item] != current_indices[item]
            for item in current_indices
            if item != route
        ):
            raise HierarchicalSelectionError("single-route beam input changes another route")
        seen_routes.add(route)
        eligible.append(evaluation)
        if len(eligible) >= beam_width:
            break
    if len(eligible) < 2:
        return ()

    excluded = frozenset(excluded_level_signatures)
    seen = set(excluded)
    plans: list[ConflictRollbackPlan] = []
    for width in range(2, min(maximum_combination_width, len(eligible)) + 1):
        for members in combinations(eligible, width):
            indices = dict(current_indices)
            routes: list[RouteId] = []
            for member in members:
                route = member.proposal.affected_routes[0]
                indices[route] = member.proposal.level_index_by_route()[route]
                routes.append(route)
            signature = tuple((route, indices[route]) for route in sorted(indices))
            if signature in seen:
                continue
            seen.add(signature)
            selection = _selection_from_level_indices(
                ordered,
                indices,
                current.target_active_bytes,
                fixed_active_bytes=current.fixed_active_bytes,
            )
            if selection.active_bytes != current.active_bytes:
                raise HierarchicalSelectionError(
                    "beam reallocation changed global active bytes"
                )
            plans.append(
                ConflictRollbackPlan(
                    selection=selection,
                    rollback_applied=True,
                    rolled_back_routes=tuple(sorted(routes)),
                    donor_routes=(),
                    bytes_added_by_rollback=0,
                    bytes_removed_by_donors=0,
                    reason="joint_failure_combined_label_blind_single_swap_beam",
                )
            )
            if len(plans) >= maximum_candidates:
                return tuple(plans)
    return tuple(plans)


def plan_logit_feedback_restore_donor_candidates(
    current: KnapsackSelection,
    ladders: Sequence[RouteCapacityLadder],
    screening_evaluations: Sequence[HierarchicalProbeEvaluation],
    *,
    maximum_candidates: int,
    maximum_forward_kl: float,
    maximum_absolute_logit_difference: float,
    required_steps: int,
    excluded_level_signatures: Iterable[tuple[tuple[RouteId, int], ...]] = (),
) -> tuple[ConflictRollbackPlan, ...]:
    """Exchange a sensitive recipient with an equal-byte low-impact donor.

    Recipient and donor ranks use only isolated FullKV-relative token, KL,
    and logit-difference measurements. Benchmark labels, references, scores,
    generated answers, and sample failure identities are not inputs. Every
    returned candidate preserves the current global logical byte count; the
    backend must additionally reject any candidate that its physical cache
    layout cannot represent within per-route limits.
    """

    if maximum_candidates < 1:
        raise HierarchicalSelectionError("restore/donor candidate cap must be positive")
    if maximum_forward_kl <= 0 or maximum_absolute_logit_difference <= 0:
        raise HierarchicalSelectionError("restore/donor stability limits must be positive")
    if required_steps < 1:
        raise HierarchicalSelectionError("restore/donor probe horizon must be positive")
    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    by_route = {ladder.route: ladder for ladder in ordered}
    current_indices = current.level_index_by_route()
    if set(by_route) != set(current_indices):
        raise HierarchicalSelectionError("restore/donor ladders do not match current routes")

    impact = {route: 0.0 for route in by_route}
    evidence_count = {route: 0 for route in by_route}
    for evaluation in screening_evaluations:
        if evaluation.proposal.kind is not HierarchicalProposalKind.SCREENING:
            raise HierarchicalSelectionError("restore/donor feedback must come from screening")
        result = evaluation.result
        gate_ratio = max(
            result.maximum_forward_kl / maximum_forward_kl,
            result.maximum_absolute_logit_difference / maximum_absolute_logit_difference,
            0.0 if result.greedy_token_match else 2.0,
            0.0 if result.evaluated_steps == required_steps else 2.0,
        )
        for route in evaluation.proposal.affected_routes:
            if route not in impact:
                raise HierarchicalSelectionError("screening feedback contains an unknown route")
            impact[route] = max(impact[route], gate_ratio)
            evidence_count[route] += 1

    recipients = sorted(
        (
            choice
            for choice in current.choices
            if any(
                level.active_bytes > choice.active_bytes
                and level.kind is not CapacityLevelKind.REALLOCATION_ALTERNATIVE
                for level in by_route[choice.route].levels
            )
        ),
        key=lambda choice: (
            evidence_count[choice.route] == 0,
            -impact[choice.route],
            -choice.evicted_risk,
            choice.route,
        ),
    )
    excluded = frozenset(excluded_level_signatures)
    seen = set(excluded)
    ranked: list[tuple[tuple[object, ...], ConflictRollbackPlan]] = []
    for recipient in recipients:
        safer_levels = sorted(
            (
                level
                for level in by_route[recipient.route].levels
                if level.active_bytes > recipient.active_bytes
                and level.kind is not CapacityLevelKind.REALLOCATION_ALTERNATIVE
            ),
            key=lambda level: (level.active_bytes, -level.level_index),
        )
        for safer in safer_levels:
            added = safer.active_bytes - recipient.active_bytes
            donors = sorted(
                (choice for choice in current.choices if choice.route != recipient.route),
                key=lambda choice: (
                    evidence_count[choice.route] == 0,
                    impact[choice.route],
                    choice.evicted_risk,
                    choice.route,
                ),
            )
            for donor in donors:
                donor_levels = sorted(
                    (
                        level
                        for level in by_route[donor.route].levels
                        if donor.active_bytes - level.active_bytes == added
                        and level.kind is not CapacityLevelKind.REALLOCATION_ALTERNATIVE
                    ),
                    key=lambda level: (-level.active_bytes, level.level_index),
                )
                for donor_level in donor_levels:
                    indices = dict(current_indices)
                    indices[recipient.route] = safer.level_index
                    indices[donor.route] = donor_level.level_index
                    signature = tuple(sorted(indices.items()))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    selection = _selection_from_level_indices(
                        ordered,
                        indices,
                        current.target_active_bytes,
                        fixed_active_bytes=current.fixed_active_bytes,
                    )
                    if selection.active_bytes != current.active_bytes:
                        raise HierarchicalSelectionError(
                            "restore/donor exchange changed global active bytes"
                        )
                    plan = ConflictRollbackPlan(
                        selection=selection,
                        rollback_applied=True,
                        rolled_back_routes=(recipient.route,),
                        donor_routes=(donor.route,),
                        bytes_added_by_rollback=added,
                        bytes_removed_by_donors=added,
                        reason=(
                            "fullkv_logit_sensitive_route_restored_with_equal_byte_"
                            "low_impact_donor"
                        ),
                    )
                    rank: tuple[object, ...] = (
                        evidence_count[recipient.route] == 0,
                        -impact[recipient.route],
                        evidence_count[donor.route] == 0,
                        impact[donor.route],
                        -added,
                        recipient.route,
                        donor.route,
                        safer.level_index,
                        donor_level.level_index,
                    )
                    ranked.append((rank, plan))
    return tuple(
        plan
        for _rank, plan in sorted(ranked, key=lambda item: item[0])[:maximum_candidates]
    )


def _normalized_logit_instability(
    result: LogitProbeResult,
    *,
    maximum_forward_kl: float,
    maximum_absolute_logit_difference: float,
    required_steps: int,
) -> float:
    """One label-blind scalar used only to rank bounded search proposals."""

    return max(
        result.maximum_forward_kl / maximum_forward_kl,
        result.maximum_absolute_logit_difference / maximum_absolute_logit_difference,
        0.0 if result.greedy_token_match else 2.0,
        0.0 if result.evaluated_steps == required_steps else 2.0,
    )


def plan_relative_logit_impact_reallocation_candidates(
    current: KnapsackSelection,
    ladders: Sequence[RouteCapacityLadder],
    screening_evaluations: Sequence[HierarchicalProbeEvaluation],
    baseline_evaluation: HierarchicalProbeEvaluation,
    *,
    maximum_candidates: int,
    maximum_forward_kl: float,
    maximum_absolute_logit_difference: float,
    required_steps: int,
    excluded_level_signatures: Iterable[tuple[tuple[RouteId, int], ...]] = (),
) -> tuple[ConflictRollbackPlan, ...]:
    """Swap equal-byte route evidence using relative FullKV-logit impact.

    The compressed baseline and every screening candidate are evaluated
    against the same FullKV logits.  Their normalized-instability difference
    is used only to rank routes; it never relaxes the absolute acceptance
    gate.  Each proposal restores omitted evidence and displaces an equal-byte
    donor block through a same-capacity route alternative, so global and
    per-route storage remain unchanged.
    """

    if maximum_candidates < 1:
        raise HierarchicalSelectionError("relative-impact candidate cap must be positive")
    if baseline_evaluation.proposal.kind is not HierarchicalProposalKind.JOINT:
        raise HierarchicalSelectionError("relative-impact baseline must be a joint probe")
    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    by_route = {ladder.route: ladder for ladder in ordered}
    current_indices = current.level_index_by_route()
    if set(by_route) != set(current_indices):
        raise HierarchicalSelectionError("relative-impact ladders do not match current routes")
    baseline_instability = _normalized_logit_instability(
        baseline_evaluation.result,
        maximum_forward_kl=maximum_forward_kl,
        maximum_absolute_logit_difference=maximum_absolute_logit_difference,
        required_steps=required_steps,
    )
    route_evidence: dict[RouteId, tuple[int, float, float, str]] = {}
    for evaluation in screening_evaluations:
        if evaluation.proposal.kind is not HierarchicalProposalKind.SCREENING:
            raise HierarchicalSelectionError("relative-impact evidence must be screening probes")
        absolute = _normalized_logit_instability(
            evaluation.result,
            maximum_forward_kl=maximum_forward_kl,
            maximum_absolute_logit_difference=maximum_absolute_logit_difference,
            required_steps=required_steps,
        )
        relative = absolute - baseline_instability
        group_size = len(evaluation.proposal.affected_routes)
        for route in evaluation.proposal.affected_routes:
            if route not in by_route:
                raise HierarchicalSelectionError(
                    "relative-impact evidence contains an unknown route"
                )
            candidate = (
                group_size,
                relative,
                absolute,
                evaluation.proposal.proposal_id,
            )
            previous = route_evidence.get(route)
            if (
                previous is None
                or group_size < previous[0]
                or (
                    group_size == previous[0]
                    and (-relative, -absolute, candidate[3])
                    < (-previous[1], -previous[2], previous[3])
                )
            ):
                route_evidence[route] = candidate

    current_choice = {choice.route: choice for choice in current.choices}
    ranked_routes: list[tuple[tuple[object, ...], RouteId, RouteCapacityLevel]] = []
    for route, choice in current_choice.items():
        alternatives = tuple(
            level
            for level in by_route[route].reallocation_levels
            if level.active_bytes == choice.active_bytes
        )
        if not alternatives:
            continue
        group_size, relative, absolute, proposal_id = route_evidence.get(
            route,
            (10**9, 0.0, baseline_instability, "no_screening_evidence"),
        )
        for alternative in alternatives:
            ranked_routes.append(
                (
                    (
                        group_size,
                        -relative,
                        -absolute,
                        -alternative.retained_utility,
                        alternative.evicted_risk,
                        route,
                        alternative.level_index,
                        proposal_id,
                    ),
                    route,
                    alternative,
                )
            )
    excluded = frozenset(excluded_level_signatures)
    seen = set(excluded)
    plans: list[ConflictRollbackPlan] = []
    for _rank, route, alternative in sorted(ranked_routes, key=lambda item: item[0]):
        indices = dict(current_indices)
        indices[route] = alternative.level_index
        signature = tuple(sorted(indices.items()))
        if signature in seen:
            continue
        seen.add(signature)
        selection = _selection_from_level_indices(
            ordered,
            indices,
            current.target_active_bytes,
            fixed_active_bytes=current.fixed_active_bytes,
        )
        if selection.active_bytes != current.active_bytes:
            raise HierarchicalSelectionError(
                "relative-impact restore/donor changed global active bytes"
            )
        plans.append(
            ConflictRollbackPlan(
                selection=selection,
                rollback_applied=True,
                rolled_back_routes=(route,),
                donor_routes=(route,),
                bytes_added_by_rollback=0,
                bytes_removed_by_donors=0,
                reason="relative_fullkv_logit_impact_equal_byte_intra_route_restore_donor",
            )
        )
        if len(plans) >= maximum_candidates:
            break
    return tuple(plans)


def select_adaptive_refinement_frontier(
    pending: Sequence[RoutePartition],
    screening_evaluations: Sequence[HierarchicalProbeEvaluation],
    *,
    maximum_partitions: int,
    maximum_forward_kl: float,
    maximum_absolute_logit_difference: float,
    required_steps: int,
) -> tuple[RoutePartition, ...]:
    """Keep both high- and low-impact branches for bounded deep refinement."""

    if maximum_partitions < 1:
        raise HierarchicalSelectionError("adaptive refinement cap must be positive")
    if len(pending) <= maximum_partitions:
        return tuple(pending)
    evaluation_by_id = {
        evaluation.proposal.proposal_id: evaluation
        for evaluation in screening_evaluations
        if evaluation.proposal.kind is HierarchicalProposalKind.SCREENING
    }
    children_by_parent: dict[str, list[RoutePartition]] = defaultdict(list)
    for partition in pending:
        parent_id, separator, _suffix = partition.partition_id.rpartition(".")
        if not separator or parent_id not in evaluation_by_id:
            raise HierarchicalSelectionError(
                "adaptive refinement partition lacks evaluated parent evidence"
            )
        children_by_parent[parent_id].append(partition)
    ranked_parents = sorted(
        children_by_parent,
        key=lambda parent_id: (
            _normalized_logit_instability(
                evaluation_by_id[parent_id].result,
                maximum_forward_kl=maximum_forward_kl,
                maximum_absolute_logit_difference=maximum_absolute_logit_difference,
                required_steps=required_steps,
            ),
            parent_id,
        ),
    )
    alternating: list[str] = []
    lower = 0
    upper = len(ranked_parents) - 1
    while lower <= upper:
        alternating.append(ranked_parents[upper])
        upper -= 1
        if lower <= upper:
            alternating.append(ranked_parents[lower])
            lower += 1
    selected: list[RoutePartition] = []
    for parent_id in alternating:
        children = sorted(
            children_by_parent[parent_id], key=lambda item: item.partition_id
        )
        if len(selected) + len(children) > maximum_partitions:
            continue
        selected.extend(children)
        if len(selected) == maximum_partitions:
            break
    if not selected:
        raise HierarchicalSelectionError("adaptive refinement could not select a frontier")
    return tuple(sorted(selected, key=lambda item: item.partition_id))


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
    reserved_reallocation_evaluations: int = 0
    reallocation_beam_width: int = 0
    reallocation_maximum_combination_width: int = 0
    logit_feedback_restore_donor: bool = False
    infeasible_proposal_ids: tuple[str, ...] = ()
    adaptive_no_pass_refinement: bool = False
    reclaimed_reallocation_evaluations: int = 0
    retained_reallocation_evaluations: int = 0
    smallest_screened_partition_route_count: int = 0
    relative_impact_reallocation_candidates: int = 0

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
        if self.reserved_reallocation_evaluations < 0:
            raise HierarchicalSelectionError("reserved reallocation evaluations cannot be negative")
        if self.reserved_reallocation_evaluations == 0 and (
            self.reallocation_beam_width != 0
            or self.reallocation_maximum_combination_width != 0
            or self.logit_feedback_restore_donor
        ):
            raise HierarchicalSelectionError("reallocation beam metadata requires a reservation")
        if min(
            self.reclaimed_reallocation_evaluations,
            self.retained_reallocation_evaluations,
            self.smallest_screened_partition_route_count,
            self.relative_impact_reallocation_candidates,
        ) < 0:
            raise HierarchicalSelectionError("adaptive hierarchy telemetry cannot be negative")
        if self.reclaimed_reallocation_evaluations > self.reserved_reallocation_evaluations:
            raise HierarchicalSelectionError("reclaimed probes exceed their reservation")
        if not self.adaptive_no_pass_refinement and (
            self.reclaimed_reallocation_evaluations != 0
            or self.relative_impact_reallocation_candidates != 0
        ):
            raise HierarchicalSelectionError(
                "adaptive hierarchy telemetry requires adaptive refinement"
            )


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
    fixed_route_capacity_reallocation: bool = False,
    reserved_reallocation_evaluations: int = 0,
    reallocation_beam_width: int = 0,
    reallocation_maximum_combination_width: int = 0,
    logit_feedback_restore_donor: bool = False,
    adaptive_no_pass_refinement: bool = False,
    selection_feasible: Callable[[KnapsackSelection], bool] | None = None,
    fullkv_fallback_selection: KnapsackSelection | None = None,
) -> BoundedHierarchicalProbeDecision:
    """Screen, assemble, jointly verify, and fail closed within hard budgets.

    Screening probes each route partition at its most aggressive ladder level
    while every other route remains at its largest physically feasible
    baseline. Refinement is truncated before it
    can consume the one evaluation and one round reserved for joint checking.
    Only explicitly passing levels enter the knapsack or rollback donor pool.
    """

    if target_active_bytes < 1 or required_steps < 1:
        raise HierarchicalSelectionError("hierarchical target and probe horizon must be positive")
    if maximum_rollback_rounds < 0:
        raise HierarchicalSelectionError("maximum rollback rounds cannot be negative")
    if reserved_reallocation_evaluations < 0:
        raise HierarchicalSelectionError("reserved reallocation evaluations cannot be negative")
    reserved_beam_enabled = reserved_reallocation_evaluations > 0
    if reserved_beam_enabled:
        if not fixed_route_capacity_reallocation:
            raise HierarchicalSelectionError(
                "reserved reallocation requires fixed route capacities"
            )
        if maximum_rollback_rounds < 2:
            raise HierarchicalSelectionError(
                "reserved reallocation beam requires two post-joint rounds"
            )
        if reallocation_beam_width < 2:
            raise HierarchicalSelectionError(
                "reserved reallocation beam width must be at least two"
            )
        if not 2 <= reallocation_maximum_combination_width <= min(
            3, reallocation_beam_width
        ):
            raise HierarchicalSelectionError("reserved reallocation combination width is invalid")
        minimum_reservation = reallocation_beam_width + 1
        if reserved_reallocation_evaluations < minimum_reservation:
            raise HierarchicalSelectionError(
                "reserved reallocation evaluations must cover the beam and one combination"
            )
        if (
            initial_partition_count + 1 + reserved_reallocation_evaluations
            > probe_budget.maximum_candidate_evaluations
        ):
            raise HierarchicalSelectionError(
                "probe budget cannot cover initial screening, joint check, and reservation"
            )
        if probe_budget.maximum_sequential_rounds < 4:
            raise HierarchicalSelectionError(
                "reserved reallocation needs screening, joint, single, and combination rounds"
            )
    elif reallocation_beam_width != 0 or reallocation_maximum_combination_width != 0:
        raise HierarchicalSelectionError("reallocation beam controls require a reservation")
    if logit_feedback_restore_donor and not reserved_beam_enabled:
        raise HierarchicalSelectionError(
            "logit-feedback restore/donor requires reserved post-joint evaluations"
        )
    if adaptive_no_pass_refinement and (
        not logit_feedback_restore_donor
        or maximum_refinement_depth < 3
        or probe_budget.maximum_sequential_rounds < 6
    ):
        raise HierarchicalSelectionError(
            "adaptive no-pass refinement requires logit feedback, depth >= 3, and six rounds"
        )
    ordered = tuple(sorted(ladders, key=lambda ladder: ladder.route))
    if not ordered or len({ladder.route for ladder in ordered}) != len(ordered):
        raise HierarchicalSelectionError("hierarchical probing requires unique route ladders")
    by_route = {ladder.route: ladder for ladder in ordered}
    if fullkv_fallback_selection is not None and (
        set(fullkv_fallback_selection.level_index_by_route()) != set(by_route)
        or fullkv_fallback_selection.target_active_bytes != target_active_bytes
        or fullkv_fallback_selection.target_reached
    ):
        raise HierarchicalSelectionError(
            "the external FullKV fallback selection is inconsistent"
        )
    verified: dict[RouteId, set[int]] = {ladder.route: {0} for ladder in ordered}
    accepted_routes: set[RouteId] = set()
    unresolved_routes: set[RouteId] = set()
    evaluations: list[HierarchicalProbeEvaluation] = []
    infeasible_proposal_ids: list[str] = []

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
    post_joint_round_reservation = 2 if reserved_beam_enabled else 0
    effective_reallocation_reservation = reserved_reallocation_evaluations
    reclaimed_reallocation_evaluations = 0
    smallest_screened_partition_route_count = 0
    while pending:
        if probe_budget.rounds_used >= (
            probe_budget.maximum_sequential_rounds - 1 - post_joint_round_reservation
        ):
            unresolved_routes.update(route for item in pending for route in item.routes)
            break
        available = (
            probe_budget.maximum_candidate_evaluations
            - probe_budget.evaluations_used
            - 1
            - effective_reallocation_reservation
        )
        if available <= 0:
            unresolved_routes.update(route for item in pending for route in item.routes)
            break
        if reclaimed_reallocation_evaluations:
            current = select_adaptive_refinement_frontier(
                pending,
                evaluations,
                maximum_partitions=min(4, available),
                maximum_forward_kl=maximum_forward_kl,
                maximum_absolute_logit_difference=maximum_absolute_logit_difference,
                required_steps=required_steps,
            )
            selected_partition_ids = {item.partition_id for item in current}
            skipped = tuple(
                item for item in pending if item.partition_id not in selected_partition_ids
            )
        else:
            current = pending[:available]
            skipped = pending[available:]
        unresolved_routes.update(route for item in skipped for route in item.routes)
        probe_budget.reserve_round(item.partition_id for item in current)
        outcomes: dict[str, bool] = {}
        for partition in current:
            smallest_screened_partition_route_count = (
                len(partition.routes)
                if smallest_screened_partition_route_count == 0
                else min(smallest_screened_partition_route_count, len(partition.routes))
            )
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
        if (
            adaptive_no_pass_refinement
            and not accepted_routes
            and reclaimed_reallocation_evaluations == 0
        ):
            retained = max(1, reallocation_beam_width)
            reclaimed_reallocation_evaluations = max(
                0, reserved_reallocation_evaluations - retained
            )
            effective_reallocation_reservation = retained
            post_joint_round_reservation = 1

    assembled = select_verified_capacity_levels(
        ordered,
        verified,
        target_active_bytes,
        require_full_level=False,
    )
    joint_proposal = _proposal(
        "joint-0",
        HierarchicalProposalKind.JOINT,
        ordered,
        assembled.level_index_by_route(),
        (choice.route for choice in assembled.choices if choice.level_index > 0),
    )
    joint_feasible = selection_feasible is None or selection_feasible(assembled)
    if joint_feasible:
        probe_budget.reserve_round((joint_proposal.proposal_id,))
        joint_passed = evaluate(joint_proposal)
    else:
        infeasible_proposal_ids.append(joint_proposal.proposal_id)
        joint_passed = False
    rollback_attempts: list[ConflictRollbackPlan] = []
    selected = assembled
    selected_passed = joint_passed
    rolled_back_routes: set[RouteId] = set()
    attempted_signatures: set[tuple[tuple[RouteId, int], ...]] = {
        joint_proposal.level_indices
    }
    if not joint_passed and reserved_beam_enabled and logit_feedback_restore_donor:
        remaining_evaluations = (
            probe_budget.maximum_candidate_evaluations - probe_budget.evaluations_used
        )
        screening_evaluations = tuple(
            evaluation
            for evaluation in evaluations
            if evaluation.proposal.kind is HierarchicalProposalKind.SCREENING
        )
        if reclaimed_reallocation_evaluations:
            baseline_evaluation = evaluations[-1]
            restore_candidates = plan_relative_logit_impact_reallocation_candidates(
                assembled,
                ordered,
                screening_evaluations,
                baseline_evaluation,
                maximum_candidates=max(1, remaining_evaluations),
                maximum_forward_kl=maximum_forward_kl,
                maximum_absolute_logit_difference=maximum_absolute_logit_difference,
                required_steps=required_steps,
                excluded_level_signatures=attempted_signatures,
            )
        else:
            restore_candidates = plan_logit_feedback_restore_donor_candidates(
                assembled,
                ordered,
                screening_evaluations,
                maximum_candidates=max(1, remaining_evaluations),
                maximum_forward_kl=maximum_forward_kl,
                maximum_absolute_logit_difference=maximum_absolute_logit_difference,
                required_steps=required_steps,
                excluded_level_signatures=attempted_signatures,
            )
        rollback_attempts.extend(restore_candidates)
        feasible_pairs: list[tuple[ConflictRollbackPlan, HierarchicalCapacityProposal]] = []
        for index, candidate in enumerate(restore_candidates):
            proposal = _proposal(
                (
                    f"relative-restore-donor-{index}"
                    if reclaimed_reallocation_evaluations
                    else f"restore-donor-{index}"
                ),
                HierarchicalProposalKind.REALLOCATION,
                ordered,
                candidate.selection.level_index_by_route(),
                (*candidate.rolled_back_routes, *candidate.donor_routes),
            )
            if selection_feasible is not None and not selection_feasible(candidate.selection):
                infeasible_proposal_ids.append(proposal.proposal_id)
                continue
            feasible_pairs.append((candidate, proposal))
        feasible_pairs = feasible_pairs[:remaining_evaluations]
        if feasible_pairs:
            probe_budget.reserve_round(
                proposal.proposal_id for _candidate, proposal in feasible_pairs
            )
            passing: list[tuple[ConflictRollbackPlan, HierarchicalProbeEvaluation]] = []
            for candidate, proposal in feasible_pairs:
                attempted_signatures.add(proposal.level_indices)
                passed = evaluate(proposal)
                evaluation = evaluations[-1]
                if passed:
                    passing.append((candidate, evaluation))
            if passing:
                selected = min(
                    passing,
                    key=lambda item: _logit_stability_rank(
                        item[1],
                        maximum_forward_kl=maximum_forward_kl,
                        maximum_absolute_logit_difference=maximum_absolute_logit_difference,
                        required_steps=required_steps,
                    ),
                )[0].selection
                selected_passed = True
    if (
        not joint_passed
        and reserved_beam_enabled
        and not logit_feedback_restore_donor
    ):
        remaining_evaluations = (
            probe_budget.maximum_candidate_evaluations - probe_budget.evaluations_used
        )
        maximum_beam_combinations = sum(
            math.comb(reallocation_beam_width, width)
            for width in range(2, reallocation_maximum_combination_width + 1)
        )
        combination_slots = min(
            maximum_beam_combinations,
            max(1, remaining_evaluations - reallocation_beam_width),
        )
        single_slots = remaining_evaluations - combination_slots
        single_candidates = plan_joint_reallocation_candidates(
            assembled,
            ordered,
            rollback_width=1,
            maximum_candidates=single_slots,
            excluded_level_signatures=attempted_signatures,
        )
        rollback_attempts.extend(single_candidates)
        single_evaluations: list[HierarchicalProbeEvaluation] = []
        if single_candidates:
            single_proposals = tuple(
                _proposal(
                    f"reallocation-single-{index}",
                    HierarchicalProposalKind.REALLOCATION,
                    ordered,
                    candidate.selection.level_index_by_route(),
                    candidate.rolled_back_routes,
                )
                for index, candidate in enumerate(single_candidates)
            )
            probe_budget.reserve_round(item.proposal_id for item in single_proposals)
            passing_single: list[
                tuple[ConflictRollbackPlan, HierarchicalProbeEvaluation]
            ] = []
            for candidate, proposal in zip(single_candidates, single_proposals, strict=True):
                attempted_signatures.add(proposal.level_indices)
                passed = evaluate(proposal)
                evaluation = evaluations[-1]
                single_evaluations.append(evaluation)
                if passed:
                    passing_single.append((candidate, evaluation))
            if passing_single:
                selected = min(
                    passing_single,
                    key=lambda item: _logit_stability_rank(
                        item[1],
                        maximum_forward_kl=maximum_forward_kl,
                        maximum_absolute_logit_difference=(
                            maximum_absolute_logit_difference
                        ),
                        required_steps=required_steps,
                    ),
                )[0].selection
                selected_passed = True
        if not selected_passed and single_evaluations:
            remaining_evaluations = (
                probe_budget.maximum_candidate_evaluations - probe_budget.evaluations_used
            )
            beam_candidates = plan_beam_reallocation_candidates(
                assembled,
                ordered,
                single_evaluations,
                beam_width=reallocation_beam_width,
                maximum_combination_width=reallocation_maximum_combination_width,
                maximum_candidates=remaining_evaluations,
                maximum_forward_kl=maximum_forward_kl,
                maximum_absolute_logit_difference=maximum_absolute_logit_difference,
                required_steps=required_steps,
                excluded_level_signatures=attempted_signatures,
            )
            rollback_attempts.extend(beam_candidates)
            if beam_candidates:
                beam_proposals = tuple(
                    _proposal(
                        f"reallocation-beam-{len(candidate.rolled_back_routes)}-{index}",
                        HierarchicalProposalKind.REALLOCATION,
                        ordered,
                        candidate.selection.level_index_by_route(),
                        candidate.rolled_back_routes,
                    )
                    for index, candidate in enumerate(beam_candidates)
                )
                probe_budget.reserve_round(item.proposal_id for item in beam_proposals)
                passing_beam: list[
                    tuple[ConflictRollbackPlan, HierarchicalProbeEvaluation]
                ] = []
                for candidate, proposal in zip(
                    beam_candidates, beam_proposals, strict=True
                ):
                    attempted_signatures.add(proposal.level_indices)
                    passed = evaluate(proposal)
                    evaluation = evaluations[-1]
                    if passed:
                        passing_beam.append((candidate, evaluation))
                if passing_beam:
                    selected = min(
                        passing_beam,
                        key=lambda item: _logit_stability_rank(
                            item[1],
                            maximum_forward_kl=maximum_forward_kl,
                            maximum_absolute_logit_difference=(
                                maximum_absolute_logit_difference
                            ),
                            required_steps=required_steps,
                        ),
                    )[0].selection
                    selected_passed = True
        if not single_candidates:
            rollback_attempts.append(
                ConflictRollbackPlan(
                    selection=assembled,
                    rollback_applied=False,
                    rolled_back_routes=(),
                    donor_routes=(),
                    bytes_added_by_rollback=0,
                    bytes_removed_by_donors=0,
                    reason="no_fixed_capacity_single_reallocation_candidate",
                )
            )
    if not joint_passed and not reserved_beam_enabled:
        for rollback_index in range(maximum_rollback_rounds):
            if (
                probe_budget.rounds_used >= probe_budget.maximum_sequential_rounds
                or probe_budget.evaluations_used >= probe_budget.maximum_candidate_evaluations
            ):
                break
            remaining_evaluations = (
                probe_budget.maximum_candidate_evaluations - probe_budget.evaluations_used
            )
            if remaining_evaluations <= 0:
                break
            # Retain the original verified-only rollback when it is feasible.
            # It is cheapest and has the strongest per-route evidence.
            candidates: list[ConflictRollbackPlan] = []
            if not fixed_route_capacity_reallocation:
                verified_rollback = plan_conflict_rollback(
                    assembled,
                    ordered,
                    verified,
                    previously_rolled_back_routes=rolled_back_routes,
                )
                if verified_rollback.rollback_applied:
                    signature = tuple(
                        sorted(verified_rollback.selection.level_index_by_route().items())
                    )
                    if signature not in attempted_signatures:
                        candidates.append(verified_rollback)
            remaining_slots = remaining_evaluations - len(candidates)
            if remaining_slots > 0 and not candidates:
                candidates.extend(
                    plan_joint_reallocation_candidates(
                        assembled,
                        ordered,
                        rollback_width=rollback_index + 1,
                        maximum_candidates=remaining_slots,
                        excluded_level_signatures=attempted_signatures
                        | {
                            tuple(sorted(item.selection.level_index_by_route().items()))
                            for item in candidates
                        },
                    )
                )
            if not candidates:
                rollback_attempts.append(
                    ConflictRollbackPlan(
                        selection=assembled,
                        rollback_applied=False,
                        rolled_back_routes=(),
                        donor_routes=(),
                        bytes_added_by_rollback=0,
                        bytes_removed_by_donors=0,
                        reason="no_target_feasible_joint_reallocation_candidate",
                    )
                )
                break
            rollback_attempts.extend(candidates)
            proposals = tuple(
                _proposal(
                    (
                        f"rollback-{rollback_index}"
                        if candidate.reason
                        == "highest_risk_route_rolled_back_with_verified_donor_exchange"
                        else f"reallocation-{rollback_index}-{candidate_index}"
                    ),
                    (
                        HierarchicalProposalKind.ROLLBACK
                        if candidate.reason
                        == "highest_risk_route_rolled_back_with_verified_donor_exchange"
                        else HierarchicalProposalKind.REALLOCATION
                    ),
                    ordered,
                    candidate.selection.level_index_by_route(),
                    (*candidate.rolled_back_routes, *candidate.donor_routes),
                )
                for candidate_index, candidate in enumerate(candidates)
            )
            probe_budget.reserve_round(proposal.proposal_id for proposal in proposals)
            passed_candidates: list[ConflictRollbackPlan] = []
            for candidate, proposal in zip(candidates, proposals, strict=True):
                attempted_signatures.add(proposal.level_indices)
                rolled_back_routes.update(candidate.rolled_back_routes)
                if evaluate(proposal):
                    passed_candidates.append(candidate)
            if passed_candidates:
                selected = min(
                    passed_candidates,
                    key=lambda item: (
                        -item.selection.active_bytes,
                        -item.selection.retained_utility,
                        tuple(sorted(item.selection.level_index_by_route().items())),
                    ),
                ).selection
                selected_passed = True
                break

    used_fullkv_fallback = not selected_passed
    if used_fullkv_fallback:
        selected = (
            fullkv_fallback_selection
            if fullkv_fallback_selection is not None
            else select_verified_capacity_levels(
                ordered,
                {ladder.route: (0,) for ladder in ordered},
                target_active_bytes,
            )
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
        reserved_reallocation_evaluations=reserved_reallocation_evaluations,
        reallocation_beam_width=reallocation_beam_width,
        reallocation_maximum_combination_width=(
            reallocation_maximum_combination_width
        ),
        logit_feedback_restore_donor=logit_feedback_restore_donor,
        infeasible_proposal_ids=tuple(infeasible_proposal_ids),
        adaptive_no_pass_refinement=adaptive_no_pass_refinement,
        reclaimed_reallocation_evaluations=reclaimed_reallocation_evaluations,
        retained_reallocation_evaluations=effective_reallocation_reservation,
        smallest_screened_partition_route_count=(
            smallest_screened_partition_route_count
        ),
        relative_impact_reallocation_candidates=(
            len(restore_candidates)
            if reclaimed_reallocation_evaluations and not joint_passed
            else 0
        ),
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
    "RouteCapacityViolation",
    "RouteId",
    "RoutePartition",
    "ScreeningUpdate",
    "build_route_capacity_ladders",
    "partition_route_ladders",
    "plan_beam_reallocation_candidates",
    "plan_conflict_rollback",
    "plan_joint_reallocation_candidates",
    "plan_logit_feedback_restore_donor_candidates",
    "plan_relative_logit_impact_reallocation_candidates",
    "proposal_route_capacity_violations",
    "restrict_route_capacity_ladders_to_position_limits",
    "run_bounded_hierarchical_probe_loop",
    "select_adaptive_refinement_frontier",
    "select_verified_capacity_levels",
    "split_route_partition",
    "update_partition_screening",
]
