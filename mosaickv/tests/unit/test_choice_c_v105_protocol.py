from __future__ import annotations

from pathlib import Path

import yaml


def _protocol() -> dict[str, object]:
    package = Path(__file__).parents[2]
    value = yaml.safe_load(
        (package / "configs/experiments/choice_c_v105_reserved_beam_preflight.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def test_v105_retains_v104_memory_and_fidelity_gates() -> None:
    protocol = _protocol()
    candidate = protocol["candidate"]
    gate = protocol["preflight_gate"]
    assert isinstance(candidate, dict)
    assert isinstance(gate, dict)

    assert candidate["active_budget_bytes"] == 28_500_000
    assert candidate["hierarchical_maximum_forward_kl"] == 0.02
    assert candidate["hierarchical_maximum_absolute_logit_difference"] == 1.0
    assert candidate["hierarchical_greedy_token_match_required"] is True
    assert gate["fullkv_fallback_passes"] is False
    assert gate["positive_result_may_authorize_calibration_automatically"] is False
    assert gate["separate_positive_review_and_authorization_required"] is True


def test_v105_preregisters_reserved_bounded_beam() -> None:
    allocation = _protocol()["probe_allocation"]
    beam = _protocol()["label_blind_beam"]
    assert isinstance(allocation, dict)
    assert isinstance(beam, dict)

    assert allocation["total_candidate_evaluations"] == 24
    assert allocation["explicitly_reserved_post_joint_evaluations"] == 12
    assert allocation["maximum_screening_evaluations"] == 11
    assert allocation["reserved_post_joint_rounds"] == 2
    assert allocation["reallocation_beam_width"] == 3
    assert allocation["reallocation_maximum_combination_width"] == 3
    assert beam["widths"] == [2, 3]
    assert beam["labels_references_task_scores_answers_and_failure_identities_accessed"] is False
    assert beam["global_active_bytes_must_not_change"] is True
    assert beam["every_composed_candidate_requires_isolated_joint_probe"] is True


def test_v105_fresh_identity_and_every_downstream_action_remain_blocked() -> None:
    protocol = _protocol()
    identity = protocol["fresh_identity_plan"]
    downstream = protocol["downstream_actions"]
    assert isinstance(identity, dict)
    assert isinstance(downstream, dict)

    assert identity["sample_count_per_task"] == 2
    assert identity["selection_seed"] == 105_017
    assert identity["ledgers_materialized"] is False
    assert identity["ledger_materialization_authorized"] is True
    assert identity["gpu_preflight_authorized"] is True
    assert len(identity["additionally_excluded_v104_preflight_ledgers"]) == 2
    assert all(value is False for value in downstream.values())


def test_v105_resolves_candidate_sha_from_the_clean_authorized_commit() -> None:
    source = _protocol()["candidate_source"]
    assert isinstance(source, dict)

    assert (
        source["implementation_git_sha_resolution"]
        == "commit_containing_this_protocol_and_runtime"
    )
    assert source["implementation_git_sha_recorded_in_materialized_worker_spec"] is True
    assert source["source_worktree_must_be_clean"] is True
    assert source["runtime_behavior_changes_after_implementation_commit_allowed"] is False
