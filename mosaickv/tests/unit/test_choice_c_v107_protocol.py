from __future__ import annotations

from pathlib import Path

import yaml


def _protocol() -> dict[str, object]:
    package = Path(__file__).parents[2]
    value = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v107_feasible_route_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_v107_keeps_memory_and_fidelity_gates_unchanged() -> None:
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


def test_v107_uses_only_feasible_final_route_levels() -> None:
    policy = _protocol()["feasible_route_policy"]
    assert isinstance(policy, dict)

    assert policy["fullkv_eligible_for_final_compressed_proposals"] is False
    assert policy["unresolved_route_baseline"] == "largest_position_feasible_level"
    assert policy["restore_donor_levels_must_belong_to_filtered_feasible_ladders"]
    assert policy["exact_logical_bytes_added_must_equal_bytes_removed"] is True
    assert (
        policy[
            "benchmark_labels_references_scores_answers_and_failure_identities_accessed"
        ]
        is False
    )


def test_v107_authorizes_only_fresh_preflight_materialization_and_execution() -> None:
    protocol = _protocol()
    identity = protocol["fresh_identity_plan"]
    downstream = protocol["downstream_actions"]
    assert isinstance(identity, dict)
    assert isinstance(downstream, dict)

    assert identity["selection_seed"] == 107_019
    assert identity["sample_count_per_task"] == 2
    assert len(identity["additionally_excluded_v106_preflight_ledgers"]) == 2
    assert identity["ledger_materialization_authorized"] is True
    assert identity["gpu_preflight_authorized"] is True
    assert downstream["preflight_materialization_authorized"] is True
    assert downstream["preflight_execution_authorized"] is True
    assert downstream["calibration_ledger_materialization_authorized"] is False
    assert downstream["calibration_execution_authorized"] is False
    assert downstream["candidate_freeze_authorized"] is False
    assert downstream["heldout_materialization_authorized"] is False
    assert downstream["heldout_execution_authorized"] is False
