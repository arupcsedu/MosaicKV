from __future__ import annotations

from pathlib import Path

import yaml


def _protocol() -> dict[str, object]:
    package = Path(__file__).parents[2]
    return yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v110_adaptive_relative_impact_preflight.yaml"
        ).read_text(encoding="utf-8")
    )


def test_v110_reclaims_probes_but_keeps_absolute_gates() -> None:
    protocol = _protocol()
    policy = protocol["adaptive_probe_policy"]
    assert policy["reclaimed_reallocation_evaluations"] == 9
    assert policy["retained_final_reallocation_evaluations"] == 3
    assert policy["maximum_refinement_frontier_per_round"] == 4
    assert policy["relative_impact_changes_candidate_order_only"] is True
    assert policy["final_candidate_uses_absolute_fullkv_gate"] is True
    gates = protocol["unchanged_memory_and_fidelity_gates"]
    assert gates["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["maximum_forward_kl"] == 0.02
    assert gates["maximum_absolute_logit_difference"] == 1.0
    assert gates["greedy_token_match_required"] is True


def test_v110_uses_fresh_ids_and_keeps_calibration_blocked() -> None:
    protocol = _protocol()
    boundary = protocol["scientific_boundary"]
    assert boundary["all_v109_planned_sample_identities_consumed"] is True
    assert boundary[
        "v109_task_labels_answers_scores_or_failure_identities_used_for_v110_design"
    ] is False
    identity = protocol["fresh_identity_plan"]
    assert identity["sample_count_per_task"] == 2
    assert len(identity["additionally_excluded_v109_preflight_ledgers"]) == 2
    assert protocol["downstream_actions"]["calibration_execution_authorized"] is False
    assert protocol["downstream_actions"]["heldout_execution_authorized"] is False
