"""CPU-only accounting arithmetic; never a constructed-cache or GPU claim."""

import itertools
import random

import pytest

from mosaickv.evaluation.bounded_reader_budget import (
    DENSE_BASELINE_QUANTUM,
    FunctionalRawIncrement,
    solve_exact_budget,
)


def item(head, cost, *, layer=0):
    raw = max(cost, 1)
    return FunctionalRawIncrement(layer, head, raw, raw - cost)


def solve(base, costs=(), **kwargs):
    kwargs.setdefault("source_positions", 30)
    kwargs.setdefault("baseline_quantum", 10)
    kwargs.setdefault("reserved_source_heads", ((0, 31),))
    return solve_exact_budget(base, tuple(item(i, cost) for i, cost in enumerate(costs)), **kwargs)


def test_exact_witness_uses_actual_functional_increments():
    result = solve(13, (4, 3))
    assert result.supported
    assert result.matched_bytes == 20
    assert result.baseline_positions == 2
    assert result.selected_raw_heads == ((0, 0), (0, 1))
    assert result.selected_increment_bytes == (4, 3)
    assert result.dummy_fill_bytes == 0
    assert not result.old_49_mib_gate_changed
    assert not result.constructed_cache_verified
    assert not result.initial_final_peak_matching_certified
    assert not result.largest_feasible_allocation_certified


def test_one_item_is_not_reused_to_reach_a_multiple():
    result = solve(2, (4,))
    assert not result.supported  # Reusing the sole +4 twice would incorrectly reach 10.
    assert result.matched_bytes is None


def test_snapshot_predecessors_survive_other_destinations():
    result = solve(1, (2, 7))
    assert result.supported
    assert result.selected_raw_heads == ((0, 0), (0, 1))
    assert result.matched_bytes == 10


def test_gcd_proves_an_incompatible_actual_cost_lattice():
    result = solve(13, (2, 4))
    assert not result.supported
    assert result.search_complete
    assert result.reason == "incompatible_actual_cost_lattice"


def test_actual_cost_bound_is_not_rounded_up():
    result = solve(21, (9,), maximum_bytes=29)
    assert not result.supported
    assert result.maximum_bytes == 29
    assert result.reason == "no_witness_within_first_reach_search"
    assert not result.search_complete


def test_complete_negative_cost_items_are_functional_not_fill():
    result = solve(23, (-3,))
    assert result.supported
    assert result.selected_increment_bytes == (-3,)
    assert result.matched_bytes == 20


def test_signed_toggle_can_omit_a_negative_item_when_exactness_requires_it():
    result = solve(21, (-2, 9))
    assert result.supported
    assert result.matched_bytes == 30
    assert result.selected_raw_heads == ((0, 1),)
    assert result.selected_increment_bytes == (9,)


def test_zero_cost_items_are_not_added_as_padding():
    result = solve(20, (0,))
    assert result.supported
    assert not result.selected_raw_heads
    assert result.matched_bytes == 20


def test_mandatory_minimum_does_not_accept_an_undersized_initial_multiple():
    result = solve(10, (10,), baseline_min_positions=2)
    assert result.supported
    assert result.matched_bytes == 20


def test_source_head_cannot_be_silently_retired():
    with pytest.raises(ValueError, match="reserved source-reader"):
        solve_exact_budget(
            100,
            (FunctionalRawIncrement(0, 31, 12, 0),),
            source_positions=20,
            reserved_source_heads=((0, 31),),
            baseline_quantum=10,
        )


@pytest.mark.parametrize(
    ("costs", "reserves", "message"),
    [
        ((FunctionalRawIncrement(0, 0, 0, 0),), ((0, 31),), "functional raw_bytes"),
        ((FunctionalRawIncrement(0, 0, 10, 101),), ((0, 31),), "cannot exceed"),
        ((FunctionalRawIncrement(0, 0, 10, 0),) * 2, ((0, 31),), "unique valid"),
        ((FunctionalRawIncrement(1, 0, 10, 0),), ((0, 31),), "reserved source-reader"),
    ],
)
def test_nonfunctional_or_unowned_costs_fail_closed(costs, reserves, message):
    with pytest.raises(ValueError, match=message):
        solve_exact_budget(
            100,
            costs,
            source_positions=20,
            reserved_source_heads=reserves,
            baseline_quantum=10,
        )


def test_item_order_is_canonical_and_deterministic():
    costs = (item(1, 3), item(0, 4))
    kwargs = dict(source_positions=20, reserved_source_heads=((0, 31),), baseline_quantum=10)
    forward = solve_exact_budget(13, costs, **kwargs)
    assert forward == solve_exact_budget(13, costs[::-1], **kwargs)


def test_raw_fullkv_and_residue_memory_bounds_are_explicit():
    with pytest.raises(ValueError, match="original raw FullKV"):
        solve(20, maximum_bytes=301)
    with pytest.raises(ValueError, match="residue-state"):
        solve(20, baseline_quantum=101, maximum_residue_states=100)
    with pytest.raises(ValueError, match="mandatory baseline"):
        solve(20, baseline_min_positions=31)


def test_all_returned_random_witnesses_match_exact_subset_costs():
    generator = random.Random(2311)
    for _ in range(100):
        base = generator.randrange(60, 90)
        changes = tuple(generator.randrange(-3, 15) for _ in range(4))
        result = solve(base, changes, maximum_bytes=100)
        totals = {
            base + sum(changes[i] for i in subset)
            for count in range(len(changes) + 1)
            for subset in itertools.combinations(range(len(changes)), count)
        }
        if result.supported:
            assert result.matched_bytes in totals
            assert result.matched_bytes % 10 == 0
            assert 10 <= result.matched_bytes <= 100
            assert len(result.selected_raw_heads) == len(set(result.selected_raw_heads))
        elif result.search_complete:
            assert not any(10 <= total <= 100 and total % 10 == 0 for total in totals)


def test_production_quantum_is_the_original_dense_bf16_geometry():
    assert DENSE_BASELINE_QUANTUM == 524288


@pytest.mark.parametrize("positions", (600, 604))
def test_existing_all_prompt_lossless_source_is_unsupported_at_49_mib(positions):
    # Source BF16 + per-layer captured cos/sin and int64 positions; no patches.
    actual_source = 32 * positions * (4096 * 2 + 128 * 2 * 2 + 8)
    result = solve_exact_budget(
        actual_source,
        (),
        source_positions=positions,
        reserved_source_heads=(),
        maximum_bytes=49 * 1024 * 1024,
    )
    assert not result.supported
    assert result.reason == "minimum_functional_storage_exceeds_cap"
    assert result.search_complete
