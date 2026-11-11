"""Actual-cost byte intersections for the bounded-reader diagnostic.

This is not a new compression/quality result. A raw-head item replaces a real
projection reader and retires that head's actual numerical-patch owners. There
is no fill allocation, budget rounding, or change to the registered 49 MiB
failure. At least one explicitly reserved head per layer remains source-read.

The bounded residue search records the FIRST witness for each residue. It is
deliberately not an exhaustive bounded-subset solver: a later cheaper witness
can be useful. A missing witness is therefore ``unsupported``, not a proof that
no encoding could work. Returned witnesses are independently checked exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd

import numpy as np

DENSE_BASELINE_QUANTUM = 32 * 32 * 128 * 2 * 2


@dataclass(frozen=True)
class FunctionalRawIncrement:
    """Measured head-local raw K/V bytes minus its exclusively retired patches."""

    layer: int
    head: int
    raw_bytes: int
    retired_patch_bytes: int

    @property
    def increment_bytes(self) -> int:
        return self.raw_bytes - self.retired_patch_bytes


@dataclass(frozen=True)
class ExactBudgetWitness:
    supported: bool
    reason: str
    source_base_bytes: int
    matched_bytes: int | None
    baseline_positions: int | None
    selected_raw_heads: tuple[tuple[int, int], ...]
    selected_increment_bytes: tuple[int, ...]
    reserved_source_heads: tuple[tuple[int, int], ...]
    baseline_quantum: int
    baseline_min_positions: int
    maximum_bytes: int
    reached_residues: int
    search_complete: bool
    largest_feasible_allocation_certified: bool = False
    dummy_fill_bytes: int = 0
    old_49_mib_gate_changed: bool = False
    constructed_cache_verified: bool = False
    initial_final_peak_matching_certified: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _integer(value: int, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def solve_exact_budget(
    source_base_bytes: int,
    increments: tuple[FunctionalRawIncrement, ...],
    *,
    source_positions: int,
    reserved_source_heads: tuple[tuple[int, int], ...],
    baseline_quantum: int = DENSE_BASELINE_QUANTUM,
    baseline_min_positions: int = 1,
    maximum_bytes: int | None = None,
    kv_heads_per_layer: int = 32,
    maximum_residue_states: int = DENSE_BASELINE_QUANTUM,
) -> ExactBudgetWitness:
    """Find a deterministic, exact constructed-source/raw-baseline byte witness.

    Costs must come from actual private raw K/V owners and patches that really
    disappear after replacement. Patch owners must not overlap across items.
    The caller must remeasure the constructed packet and match this witness;
    this arithmetic alone does not certify a cache, numerical parity, or peak.

    Signed increments are supported by initially selecting every negative-cost
    item, then searching optional absolute-cost toggles. Zero-cost items are
    omitted deterministically. Snapshot expansion prevents reusing an item or
    mutating the predecessor chain through an in-place DP update.
    """
    for value, name, minimum in (
        (source_base_bytes, "source_base_bytes", 1),
        (source_positions, "source_positions", 1),
        (baseline_quantum, "baseline_quantum", 1),
        (baseline_min_positions, "baseline_min_positions", 1),
        (kv_heads_per_layer, "kv_heads_per_layer", 2),
        (maximum_residue_states, "maximum_residue_states", 1),
    ):
        _integer(value, name, minimum)
    if baseline_min_positions > source_positions:
        raise ValueError("mandatory baseline positions exceed the original source")
    if baseline_quantum > maximum_residue_states:
        raise ValueError("baseline quantum exceeds the explicit residue-state bound")
    full_raw_bytes = source_positions * baseline_quantum
    if maximum_bytes is None:
        maximum_bytes = full_raw_bytes
    _integer(maximum_bytes, "maximum_bytes", 1)
    if maximum_bytes > full_raw_bytes:
        raise ValueError("maximum_bytes cannot exceed the original raw FullKV storage")
    if max(source_base_bytes, maximum_bytes) >= np.iinfo(np.int64).max:
        raise ValueError("actual byte costs exceed the bounded int64 solver")
    if type(increments) is not tuple or type(reserved_source_heads) is not tuple:
        raise ValueError("immutable increment and reserved-head tuples required")
    reserves: dict[int, int] = {}
    for layer, head in reserved_source_heads:
        _integer(layer, "reserved layer")
        _integer(head, "reserved head")
        if layer in reserves or head >= kv_heads_per_layer:
            raise ValueError("one distinct valid source-reader head per layer required")
        reserves[layer] = head
    if any(type(item) is not FunctionalRawIncrement for item in increments):
        raise ValueError("explicit functional raw-head cost items required")
    items = tuple(sorted(increments, key=lambda item: (item.layer, item.head)))
    seen: set[tuple[int, int]] = set()
    for item in items:
        if type(item) is not FunctionalRawIncrement:
            raise ValueError("explicit functional raw-head cost items required")
        for value, name, minimum in (
            (item.layer, "item layer", 0),
            (item.head, "item head", 0),
            (item.raw_bytes, "functional raw_bytes", 1),
            (item.retired_patch_bytes, "retired_patch_bytes", 0),
        ):
            _integer(value, name, minimum)
        key = item.layer, item.head
        if key in seen or item.head >= kv_heads_per_layer:
            raise ValueError("unique valid layer/head items required")
        if item.layer not in reserves or reserves[item.layer] == item.head:
            raise ValueError("every item must preserve its layer's reserved source-reader head")
        seen.add(key)
    if sum(item.retired_patch_bytes for item in items) > source_base_bytes:
        raise ValueError("retired private patch costs cannot exceed owned source backing")
    signed = tuple(item.increment_bytes for item in items)
    initial_negative = {i for i, cost in enumerate(signed) if cost < 0}
    normalized_base = source_base_bytes + sum(signed[i] for i in initial_negative)
    if normalized_base < 1:
        raise ValueError("functional source ownership cannot be retired completely")
    minimum_bytes = baseline_min_positions * baseline_quantum
    canonical_reserves = tuple(sorted(reserves.items()))

    def result(
        indices: set[int] | None, reason: str, reached: int, complete: bool
    ) -> ExactBudgetWitness:
        if indices is None:
            return ExactBudgetWitness(
                False,
                reason,
                source_base_bytes,
                None,
                None,
                (),
                (),
                canonical_reserves,
                baseline_quantum,
                baseline_min_positions,
                maximum_bytes,
                reached,
                complete,
            )
        chosen = tuple(sorted(indices))
        actual = source_base_bytes + sum(signed[i] for i in chosen)
        if (
            len(set(chosen)) != len(chosen)
            or actual % baseline_quantum
            or not minimum_bytes <= actual <= maximum_bytes
        ):
            raise RuntimeError("invalid exact-byte predecessor witness")
        return ExactBudgetWitness(
            True,
            reason,
            source_base_bytes,
            actual,
            actual // baseline_quantum,
            tuple((items[i].layer, items[i].head) for i in chosen),
            tuple(signed[i] for i in chosen),
            canonical_reserves,
            baseline_quantum,
            baseline_min_positions,
            maximum_bytes,
            reached,
            complete,
        )

    absolute = tuple(abs(cost) for cost in signed)
    if normalized_base + sum(absolute) >= np.iinfo(np.int64).max:
        raise ValueError("actual subset-byte costs exceed the bounded int64 solver")
    if normalized_base > maximum_bytes:
        return result(None, "minimum_functional_storage_exceeds_cap", 0, True)
    if normalized_base + sum(absolute) < minimum_bytes:
        return result(None, "maximum_functional_storage_below_mandatory_baseline", 0, True)
    lattice = reduce(gcd, absolute, baseline_quantum)
    if normalized_base % lattice:
        return result(None, "incompatible_actual_cost_lattice", 0, True)
    if minimum_bytes <= normalized_base <= maximum_bytes and not normalized_base % baseline_quantum:
        return result(initial_negative, "exact_actual_cost_witness", 1, False)

    costs = np.full(baseline_quantum, -1, dtype=np.int64)
    predecessor = np.full(baseline_quantum, -1, dtype=np.int32)
    predecessor_item = np.full(baseline_quantum, -1, dtype=np.int32)
    origin = normalized_base % baseline_quantum
    costs[origin] = 0
    for index, cost in enumerate(absolute):
        if cost == 0 or cost > maximum_bytes - normalized_base:
            continue
        # Snapshot indices AND costs before publishing any destination.
        before = np.flatnonzero(costs >= 0)
        previous_costs = costs[before].copy()
        after_costs = previous_costs + cost
        allowed = after_costs <= maximum_bytes - normalized_base
        before, after_costs = before[allowed], after_costs[allowed]
        after = (before + cost) % baseline_quantum
        target = np.flatnonzero((after == 0) & (normalized_base + after_costs >= minimum_bytes))
        if target.size:
            at = int(target[0])
            cursor = int(before[at])
            toggles = [index]
            while cursor != origin:
                chosen_item = int(predecessor_item[cursor])
                if chosen_item < 0 or chosen_item in toggles:
                    raise RuntimeError("cyclic or missing bounded-byte predecessor")
                toggles.append(chosen_item)
                cursor = int(predecessor[cursor])
            selected = initial_negative.symmetric_difference(toggles)
            return result(selected, "exact_actual_cost_witness", int((costs >= 0).sum()), False)
        unseen = costs[after] < 0
        destinations = after[unseen]
        costs[destinations] = after_costs[unseen]
        predecessor[destinations] = before[unseen]
        predecessor_item[destinations] = index
    return result(None, "no_witness_within_first_reach_search", int((costs >= 0).sum()), False)
