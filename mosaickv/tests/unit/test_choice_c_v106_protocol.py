from __future__ import annotations

from pathlib import Path

import yaml


def _protocol() -> dict[str, object]:
    package = Path(__file__).parents[2]
    value = yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v106_logit_feedback_restore_donor_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_v106_keeps_all_memory_and_fidelity_gates_unchanged() -> None:
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


def test_v106_restore_donor_is_label_blind_exact_byte_and_route_feasible() -> None:
    policy = _protocol()["restore_donor_policy"]
    assert isinstance(policy, dict)

    assert policy["exact_logical_bytes_added_must_equal_bytes_removed"] is True
    assert policy["global_logical_active_bytes_must_not_change"] is True
    assert policy["every_final_proposal_checked_against_every_route_position_limit_before_probe"]
    assert policy["candidate_requires_isolated_joint_logit_probe"] is True
    assert (
        policy["benchmark_labels_references_scores_answers_and_failure_identities_accessed"]
        is False
    )


def test_v106_uses_fresh_ids_but_keeps_every_execution_action_disabled() -> None:
    protocol = _protocol()
    identity = protocol["fresh_identity_plan"]
    downstream = protocol["downstream_actions"]
    assert isinstance(identity, dict)
    assert isinstance(downstream, dict)

    assert identity["selection_seed"] == 106_017
    assert identity["sample_count_per_task"] == 2
    assert len(identity["additionally_excluded_v105_preflight_ledgers"]) == 2
    assert identity["ledger_materialization_authorized"] is False
    assert identity["gpu_preflight_authorized"] is False
    assert all(value is False for value in downstream.values())
