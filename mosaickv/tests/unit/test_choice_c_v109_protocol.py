from __future__ import annotations

from pathlib import Path

import yaml


def _protocol() -> dict[str, object]:
    package = Path(__file__).parents[2]
    return yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v109_multi_feasible_ladder_preflight.yaml"
        ).read_text(encoding="utf-8")
    )


def test_v109_keeps_gates_and_calibration_blocked() -> None:
    protocol = _protocol()
    gates = protocol["unchanged_memory_and_fidelity_gates"]
    assert gates["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["maximum_forward_kl"] == 0.02
    assert gates["maximum_absolute_logit_difference"] == 1.0
    assert gates["greedy_token_match_required"] is True
    assert protocol["downstream_actions"]["calibration_execution_authorized"] is False
    assert protocol["downstream_actions"]["heldout_execution_authorized"] is False


def test_v109_requires_multiple_feasible_levels_and_dual_accounting() -> None:
    protocol = _protocol()
    candidate = protocol["candidate"]
    assert candidate["dense_prompt_position_cap"] == 54
    assert candidate["refinement_positions_below_baseline"] == 8
    assert candidate["minimum_distinct_ordinary_levels_when_optional_capacity_allows"] == 3
    assert candidate["require_positive_savings_screening_partitions"] is True
    accounting = protocol["fallback_accounting_invariants"]
    assert accounting["hard_budget_fallback_records_all_six_accounting_fields"] is True
    assert accounting["fullkv_fallback_accounting_recorded_separately"] is True
    assert accounting["rejected_compressed_proposal_accounting_recorded_separately"] is True
    assert accounting["rejected_compressed_proposal_id_recorded"] is True


def test_v109_consumes_v108_and_requires_fresh_identities() -> None:
    protocol = _protocol()
    boundary = protocol["scientific_boundary"]
    assert boundary["all_v108_planned_sample_identities_consumed"] is True
    assert boundary[
        "v108_task_labels_answers_scores_or_failure_identities_used_for_v109_design"
    ] is False
    identity = protocol["fresh_identity_plan"]
    assert identity["sample_count_per_task"] == 2
    assert len(identity["additionally_excluded_v108_preflight_ledgers"]) == 2
