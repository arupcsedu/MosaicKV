from __future__ import annotations

import random
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel, HuggingFaceRuntimeError
from mosaickv.config import ConfigurationError, SelectionConfig
from mosaickv.selection.hierarchical import (
    CapacityLevelKind,
    HierarchicalSelectionError,
    LogitProbeResult,
    ProbeBudgetExceeded,
    ProbeBudgetTracker,
    RouteBlockCandidate,
    RouteCapacityLadder,
    RouteCapacityLevel,
    build_route_capacity_ladders,
    partition_route_ladders,
    plan_conflict_rollback,
    run_bounded_hierarchical_probe_loop,
    select_verified_capacity_levels,
    update_partition_screening,
)


def _hierarchical_config() -> SelectionConfig:
    return SelectionConfig(
        stable_head_prompt_routing=True,
        stable_head_query_coverage_initial_selection=True,
        stable_head_query_coverage_initial_fraction=0.5,
        stable_head_query_coverage_initial_objective="capped_cumulative",
        stable_head_query_coverage_initial_cap_fraction=0.5,
        stable_head_query_coverage_cross_head_allocation=True,
        stable_head_query_coverage_layout_group_reservation=True,
        stable_head_query_coverage_layout_group_fraction=0.25,
        stable_head_query_coverage_layout_group_maximum_blocks=4,
        stable_head_query_coverage_layout_group_atomic_selection=True,
        stable_head_query_coverage_layout_group_atomic_selection_mode=(
            "pareto_joint_donor_knapsack"
        ),
        stable_head_query_coverage_layout_group_minimum_utilization_fraction=0.75,
        stable_head_query_coverage_layout_group_minimum_normalized_gain=0.000001,
        stable_head_query_coverage_layout_group_gain_to_risk_ratio=1.01,
        stable_head_hierarchical_logit_safe_eviction=True,
        stable_head_hierarchical_partition_count=8,
        stable_head_hierarchical_maximum_refinement_depth=2,
        stable_head_hierarchical_maximum_candidate_evaluations=24,
        stable_head_hierarchical_maximum_sequential_rounds=6,
        stable_head_hierarchical_maximum_rollback_rounds=2,
        stable_head_hierarchical_horizon=2,
        stable_head_hierarchical_maximum_kl=0.02,
        stable_head_hierarchical_maximum_logit_difference=1.0,
    )


def _blocks(route_count: int = 4, blocks_per_route: int = 6) -> list[RouteBlockCandidate]:
    blocks: list[RouteBlockCandidate] = []
    node_id = 0
    for head in range(route_count):
        for index in range(blocks_per_route):
            blocks.append(
                RouteBlockCandidate(
                    node_id=node_id,
                    layer=0,
                    kv_head=head,
                    byte_size=8 + index,
                    evidence_utility=float(index + head),
                    eviction_risk=float(index + 1) / 10.0,
                    mandatory=index == blocks_per_route - 1,
                    atomic_group_id=f"g-{head}" if index in {2, 3} else None,
                )
            )
            node_id += 1
    return blocks


def _ladder(
    route: tuple[int, int],
    base_node: int,
    utilities: tuple[float, float, float],
    risks: tuple[float, float, float],
) -> RouteCapacityLadder:
    levels = tuple(
        RouteCapacityLevel(
            route=route,
            level_index=index,
            kind=(
                CapacityLevelKind.FULL
                if index == 0
                else CapacityLevelKind.COARSE_MIDPOINT
                if index == 1
                else CapacityLevelKind.TARGET_ADJACENT
            ),
            selected_node_ids=tuple(range(base_node, base_node + 3 - index)),
            active_bytes=30 - 10 * index,
            retained_utility=utilities[index],
            evicted_risk=risks[index],
        )
        for index in range(3)
    )
    return RouteCapacityLadder(
        route=route,
        levels=levels,
        protected_node_ids=(base_node,),
        target_active_bytes=10,
        target_reached=True,
    )


def test_hierarchical_configuration_is_strict_and_mutually_exclusive() -> None:
    config = _hierarchical_config()

    assert config.stable_head_hierarchical_maximum_candidate_evaluations == 24
    with pytest.raises(ConfigurationError, match="controls require the hierarchical gate"):
        replace(config, stable_head_hierarchical_logit_safe_eviction=False)
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        replace(
            config,
            stable_head_progressive_logit_safe_eviction=True,
            stable_head_progressive_eviction_initial_batch_blocks=16,
            stable_head_progressive_eviction_maximum_probes=32,
            stable_head_progressive_eviction_horizon=2,
            stable_head_progressive_eviction_maximum_kl=0.02,
            stable_head_progressive_eviction_maximum_logit_difference=1.0,
        )
    with pytest.raises(ConfigurationError, match="every initial partition"):
        replace(config, stable_head_hierarchical_maximum_candidate_evaluations=8)
    with pytest.raises(ConfigurationError, match="leave screening and joint-check rounds"):
        replace(
            config,
            stable_head_hierarchical_maximum_sequential_rounds=3,
            stable_head_hierarchical_maximum_rollback_rounds=2,
        )


def test_hierarchical_configuration_parses_all_controls_without_coercion() -> None:
    source = _hierarchical_config()
    mapping = {
        field.name: getattr(source, field.name)
        for field in fields(source)
        if field.name.startswith("stable_head_")
    }

    parsed = SelectionConfig.from_mapping(mapping)

    assert parsed == source
    mapping["stable_head_hierarchical_partition_count"] = True
    with pytest.raises(ConfigurationError, match="must be an integer"):
        SelectionConfig.from_mapping(mapping)


def test_hf_runtime_fails_closed_before_loading_an_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MOSAICKV_ENABLE_V100_GPU_DEBUG", raising=False)
    config = SimpleNamespace(
        execution=SimpleNamespace(backend=SimpleNamespace(value="huggingface")),
        selection=SimpleNamespace(stable_head_hierarchical_logit_safe_eviction=True),
    )

    with pytest.raises(HuggingFaceRuntimeError, match="explicit debug-only"):
        HuggingFaceMosaicKVModel(
            cast(Any, object()),
            cast(Any, config),
            trace_directory=tmp_path,
        )


def test_hf_runtime_accepts_only_exact_debug_authorization_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        execution=SimpleNamespace(backend=SimpleNamespace(value="huggingface")),
        selection=SimpleNamespace(stable_head_hierarchical_logit_safe_eviction=True),
    )
    monkeypatch.setenv("MOSAICKV_ENABLE_V100_GPU_DEBUG", "true")
    with pytest.raises(HuggingFaceRuntimeError, match="explicit debug-only"):
        HuggingFaceMosaicKVModel(
            cast(Any, object()),
            cast(Any, config),
            trace_directory=tmp_path,
        )

    monkeypatch.setenv("MOSAICKV_ENABLE_V100_GPU_DEBUG", "1")
    with pytest.raises(AttributeError, match="attention_implementation"):
        HuggingFaceMosaicKVModel(
            cast(Any, object()),
            cast(Any, config),
            trace_directory=tmp_path,
        )


def test_route_ladders_are_nested_and_preserve_protected_blocks() -> None:
    blocks = _blocks(route_count=2)
    targets = {
        route: sum(block.byte_size for block in blocks if block.route == route) - 18
        for route in {(0, 0), (0, 1)}
    }

    first = build_route_capacity_ladders(blocks, targets)
    second = build_route_capacity_ladders(list(reversed(blocks)), targets)

    assert first == second
    for ladder in first:
        previous = set(ladder.full_level.selected_node_ids)
        for level in ladder.levels:
            selected = set(level.selected_node_ids)
            assert selected <= previous
            assert set(ladder.protected_node_ids) <= selected
            previous = selected


def test_route_ladders_require_a_protected_core() -> None:
    block = RouteBlockCandidate(0, 0, 0, 16, 1.0, 1.0)

    with pytest.raises(HierarchicalSelectionError, match="at least one"):
        build_route_capacity_ladders([block], {(0, 0): 8})


def test_random_route_ladders_preserve_nesting_and_bytes() -> None:
    for seed in range(20):
        generator = random.Random(seed)
        block_count = generator.randint(3, 12)
        blocks = [
            RouteBlockCandidate(
                node_id=index,
                layer=seed % 3,
                kv_head=seed % 5,
                byte_size=generator.randint(1, 32),
                evidence_utility=generator.random(),
                eviction_risk=generator.random(),
                mandatory=index == block_count - 1,
            )
            for index in range(block_count)
        ]
        full_bytes = sum(block.byte_size for block in blocks)
        target = generator.randint(1, full_bytes)
        ladder = build_route_capacity_ladders(blocks, {(seed % 3, seed % 5): target})[0]

        assert all(
            right.active_bytes <= left.active_bytes
            for left, right in zip(ladder.levels, ladder.levels[1:], strict=False)
        )
        assert all(
            set(right.selected_node_ids) <= set(left.selected_node_ids)
            for left, right in zip(ladder.levels, ladder.levels[1:], strict=False)
        )


def test_partitions_are_deterministic_byte_balanced_and_refine_unsafe_groups() -> None:
    blocks = _blocks(route_count=6)
    routes = {(0, head) for head in range(6)}
    targets = {
        route: sum(block.byte_size for block in blocks if block.route == route) - 18
        for route in routes
    }
    ladders = build_route_capacity_ladders(blocks, targets)
    partitions = partition_route_ladders(ladders, 3)

    assert partitions == partition_route_ladders(tuple(reversed(ladders)), 3)
    assert set(route for partition in partitions for route in partition.routes) == routes
    assert max(partition.proposed_removed_bytes for partition in partitions) - min(
        partition.proposed_removed_bytes for partition in partitions
    ) <= max(ladder.proposed_removed_bytes for ladder in ladders)

    outcomes = {partition.partition_id: index == 0 for index, partition in enumerate(partitions)}
    update = update_partition_screening(
        partitions,
        outcomes,
        {ladder.route: ladder for ladder in ladders},
        maximum_refinement_depth=2,
    )
    assert update.accepted_partitions == (partitions[0],)
    assert all(partition.depth == 1 for partition in update.next_partitions)
    assert not update.unresolved_exact_routes


def test_probe_budget_fails_before_overrun_and_rejects_duplicate_candidates() -> None:
    tracker = ProbeBudgetTracker(3, 2)

    assert tracker.reserve_round(["p0", "p1"]).round_index == 0
    with pytest.raises(HierarchicalSelectionError, match="cannot be probed twice"):
        tracker.reserve_round(["p1"])
    assert tracker.reserve_round(["joint"]).round_index == 1
    with pytest.raises(ProbeBudgetExceeded, match="round budget"):
        tracker.reserve_round(["rollback"])
    assert tracker.evaluations_used == 3
    assert tracker.rounds_used == 2


def test_logit_probe_requires_token_kl_difference_and_complete_horizon() -> None:
    safe = LogitProbeResult(True, 0.01, 0.5, 2)

    assert safe.passes(
        maximum_forward_kl=0.02,
        maximum_absolute_logit_difference=1.0,
        required_steps=2,
    )
    assert not replace(safe, greedy_token_match=False).passes(
        maximum_forward_kl=0.02,
        maximum_absolute_logit_difference=1.0,
        required_steps=2,
    )
    assert not replace(safe, evaluated_steps=1).passes(
        maximum_forward_kl=0.02,
        maximum_absolute_logit_difference=1.0,
        required_steps=2,
    )


def test_multiple_choice_knapsack_uses_one_verified_level_per_route() -> None:
    ladders = (
        _ladder((0, 0), 0, (100.0, 80.0, 70.0), (0.0, 5.0, 20.0)),
        _ladder((0, 1), 10, (100.0, 90.0, 10.0), (0.0, 1.0, 2.0)),
    )
    verified = {(0, 0): (0, 1, 2), (0, 1): (0, 1, 2)}

    selected = select_verified_capacity_levels(ladders, verified, 30)

    assert selected.target_reached
    assert selected.active_bytes == 30
    assert selected.level_index_by_route() == {(0, 0): 2, (0, 1): 1}
    assert selected == select_verified_capacity_levels(tuple(reversed(ladders)), verified, 30)


def test_multiple_choice_knapsack_records_an_unavoidable_target_shortfall() -> None:
    ladder = _ladder((0, 0), 0, (3.0, 2.0, 1.0), (0.0, 1.0, 2.0))

    selected = select_verified_capacity_levels([ladder], {(0, 0): (0, 1)}, 15)

    assert not selected.target_reached
    assert selected.active_bytes == 20
    assert selected.target_shortfall_bytes == 5
    assert selected.target_underfill_bytes == 0


def test_conflict_rollback_restores_riskiest_route_and_uses_verified_donor() -> None:
    ladders = (
        _ladder((0, 0), 0, (100.0, 80.0, 70.0), (0.0, 5.0, 20.0)),
        _ladder((0, 1), 10, (100.0, 90.0, 10.0), (0.0, 1.0, 2.0)),
    )
    verified = {(0, 0): (0, 1, 2), (0, 1): (0, 1, 2)}
    current = select_verified_capacity_levels(ladders, verified, 30)

    rollback = plan_conflict_rollback(current, ladders, verified)

    assert rollback.rollback_applied
    assert rollback.rolled_back_routes == ((0, 0),)
    assert rollback.donor_routes == ((0, 1),)
    assert rollback.bytes_added_by_rollback == rollback.bytes_removed_by_donors == 10
    assert rollback.selection.active_bytes == 30
    assert rollback.selection.level_index_by_route() == {(0, 0): 1, (0, 1): 2}


def test_conflict_rollback_fails_closed_when_no_exchange_is_feasible() -> None:
    ladder = _ladder((0, 0), 0, (3.0, 2.0, 1.0), (0.0, 1.0, 2.0))
    verified = {(0, 0): (0, 1, 2)}
    current = select_verified_capacity_levels([ladder], verified, 10)

    rollback = plan_conflict_rollback(current, [ladder], verified)

    assert not rollback.rollback_applied
    assert rollback.selection is current
    assert rollback.reason == "no_target_feasible_verified_rollback_exchange"


def _probe_result(passed: bool) -> LogitProbeResult:
    return LogitProbeResult(
        greedy_token_match=passed,
        maximum_forward_kl=0.01 if passed else 0.5,
        maximum_absolute_logit_difference=0.5 if passed else 5.0,
        evaluated_steps=2,
    )


def test_bounded_hierarchy_selects_jointly_verified_target() -> None:
    ladders = tuple(
        _ladder((0, head), head * 10, (100.0, 80.0, 70.0), (0.0, 5.0, 20.0)) for head in range(4)
    )
    budget = ProbeBudgetTracker(8, 4)

    decision = run_bounded_hierarchical_probe_loop(
        ladders,
        target_active_bytes=40,
        initial_partition_count=2,
        maximum_refinement_depth=1,
        maximum_rollback_rounds=1,
        maximum_forward_kl=0.02,
        maximum_absolute_logit_difference=1.0,
        required_steps=2,
        probe_budget=budget,
        probe=lambda _proposal: _probe_result(True),
    )

    assert decision.target_reached
    assert decision.selected_candidate_passed
    assert decision.final_joint_probe_passed
    assert not decision.used_fullkv_fallback
    assert decision.selection.active_bytes == 40
    assert decision.evaluations_used == 3
    assert decision.rounds_used == 2


def test_bounded_hierarchy_refines_failed_partition_without_exceeding_budget() -> None:
    ladders = tuple(
        _ladder((0, head), head * 10, (100.0, 80.0, 70.0), (0.0, 5.0, 20.0)) for head in range(4)
    )
    budget = ProbeBudgetTracker(4, 4)

    decision = run_bounded_hierarchical_probe_loop(
        ladders,
        target_active_bytes=40,
        initial_partition_count=2,
        maximum_refinement_depth=1,
        maximum_rollback_rounds=0,
        maximum_forward_kl=0.02,
        maximum_absolute_logit_difference=1.0,
        required_steps=2,
        probe_budget=budget,
        probe=lambda proposal: _probe_result(proposal.proposal_id != "p0"),
    )

    assert decision.evaluations_used == 4
    assert decision.rounds_used == 3
    assert decision.unresolved_exact_routes
    assert len(decision.accepted_target_routes) == 3
    assert not decision.target_reached
    assert decision.selected_candidate_passed
    assert not decision.used_fullkv_fallback


def test_bounded_hierarchy_uses_verified_rollback_after_joint_failure() -> None:
    ladders = (
        _ladder((0, 0), 0, (100.0, 80.0, 70.0), (0.0, 5.0, 20.0)),
        _ladder((0, 1), 10, (100.0, 90.0, 10.0), (0.0, 1.0, 2.0)),
    )
    budget = ProbeBudgetTracker(6, 4)

    decision = run_bounded_hierarchical_probe_loop(
        ladders,
        target_active_bytes=40,
        initial_partition_count=2,
        maximum_refinement_depth=0,
        maximum_rollback_rounds=1,
        maximum_forward_kl=0.02,
        maximum_absolute_logit_difference=1.0,
        required_steps=2,
        probe_budget=budget,
        probe=lambda proposal: _probe_result(proposal.proposal_id != "joint-0"),
    )

    assert decision.target_reached
    assert decision.selected_candidate_passed
    assert not decision.final_joint_probe_passed
    assert not decision.used_fullkv_fallback
    assert len(decision.rollback_attempts) == 1
    assert decision.rollback_attempts[0].rollback_applied
    assert decision.evaluations[-1].proposal.proposal_id == "rollback-0"


def test_bounded_hierarchy_falls_back_to_fullkv_after_unsafe_joint_candidates() -> None:
    ladders = (
        _ladder((0, 0), 0, (100.0, 80.0, 70.0), (0.0, 5.0, 20.0)),
        _ladder((0, 1), 10, (100.0, 90.0, 10.0), (0.0, 1.0, 2.0)),
    )
    budget = ProbeBudgetTracker(6, 4)

    decision = run_bounded_hierarchical_probe_loop(
        ladders,
        target_active_bytes=40,
        initial_partition_count=2,
        maximum_refinement_depth=0,
        maximum_rollback_rounds=1,
        maximum_forward_kl=0.02,
        maximum_absolute_logit_difference=1.0,
        required_steps=2,
        probe_budget=budget,
        probe=lambda proposal: _probe_result(proposal.kind.value == "screening"),
    )

    assert not decision.target_reached
    assert not decision.selected_candidate_passed
    assert decision.used_fullkv_fallback
    assert decision.selection.active_bytes == 60
    assert all(choice.level_index == 0 for choice in decision.selection.choices)
