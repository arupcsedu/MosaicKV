from __future__ import annotations

from pathlib import Path

import yaml


def _protocol() -> dict[str, object]:
    package = Path(__file__).parents[2]
    value = yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v104_joint_reallocation_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_v104_keeps_every_scientific_gate_unchanged() -> None:
    protocol = _protocol()
    candidate = protocol["candidate"]
    gate = protocol["preflight_gate"]
    assert isinstance(candidate, dict)
    assert isinstance(gate, dict)

    assert candidate["active_budget_bytes"] == 28_500_000
    assert candidate["hierarchical_maximum_forward_kl"] == 0.02
    assert candidate["hierarchical_maximum_absolute_logit_difference"] == 1.0
    assert candidate["hierarchical_greedy_token_match_required"] is True
    assert gate["maximum_candidate_evaluations"] == 24
    assert gate["maximum_sequential_rounds"] == 6
    assert gate["fullkv_fallback_passes"] is False


def test_v104_reallocation_preserves_dense_physical_capacity() -> None:
    reallocation = _protocol()["joint_reallocation"]
    assert isinstance(reallocation, dict)

    assert reallocation["every_route_active_bytes_must_equal_initial_target_route_bytes"]
    assert reallocation["every_route_physical_positions_must_remain_at_or_below_dense_cap"]
    assert reallocation["global_active_bytes_must_not_change"]
    assert reallocation["atomic_and_mandatory_blocks_must_remain_selected"]
    assert reallocation["every_candidate_requires_isolated_joint_probe"]
    assert reallocation["unprobed_candidate_may_not_be_selected"]


def test_v104_fresh_identity_and_downstream_actions_fail_closed() -> None:
    protocol = _protocol()
    identity = protocol["fresh_identity_plan"]
    downstream = protocol["downstream_actions"]
    assert isinstance(identity, dict)
    assert isinstance(downstream, dict)

    assert identity["sample_count_per_task"] == 2
    assert identity["ledgers_materialized"] is False
    assert identity["ledger_materialization_authorized"] is True
    assert identity["gpu_preflight_authorized"] is True
    assert identity["selection_seed"] == 104_017
    assert len(identity["additionally_excluded_v103_preflight_ledgers"]) == 2
    assert all(value is False for value in downstream.values())


def test_v104_resolves_candidate_sha_from_the_clean_protocol_commit() -> None:
    source = _protocol()["candidate_source"]
    assert isinstance(source, dict)

    assert (
        source["implementation_git_sha_resolution"]
        == "commit_containing_this_protocol_and_runtime"
    )
    assert source["implementation_git_sha_recorded_in_materialized_worker_spec"] is True
    assert source["source_worktree_must_be_clean"] is True
    assert source["runtime_behavior_changes_after_implementation_commit_allowed"] is False
