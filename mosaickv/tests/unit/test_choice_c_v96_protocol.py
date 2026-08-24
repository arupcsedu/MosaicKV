from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml

from mosaickv.config import load_config


def _spec(package: Path) -> dict:
    return yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v96_pareto_atomic_group_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _protocol(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v96_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v95_negative_seal_blocks_reclassification_and_pins_artifacts() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v95_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["overall_freeze_gate_pass"] is False
    assert seal["memory_gate_pass"] is True
    assert seal["aggregate_fidelity_gate_pass"] is False
    assert seal["robust_fold_fidelity_gate_pass"] is False
    assert seal["atomic_group_utilization_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["heldout_evaluation_authorized"] is False
    for relation in ("protocol", "parent_negative_result_seal"):
        record = seal[relation]
        path = package / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    boundary = seal["scientific_boundary"]
    assert boundary["future_design_must_not_use_v95_answers"] is True
    assert boundary["future_design_must_not_use_v95_failure_identities"] is True
    assert boundary["future_heldout_materialized"] is False


def test_v96_is_fresh_disjoint_and_does_not_use_v95_outcomes() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    boundary = spec["scientific_boundary"]

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["execution_status"] == "prepared_waiting_for_clean_commit"
    assert spec["prior_result_exclusion_boundary"] == "v95_and_all_prior_choice_c_runs"
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v95_negative_calibration_seal.yaml"
    )
    assert boundary["aggregate_v95_mechanism_activity_used_for_design"] is True
    for key in (
        "v95_answers_used_for_design",
        "v95_references_used_for_design",
        "v95_task_scores_used_for_design",
        "v95_failure_identities_used_for_design",
        "v95_generated_token_ids_used_for_design",
        "v95_per_sample_telemetry_used_for_design",
    ):
        assert boundary[key] is False
    assert {task["selection_seed"] for task in spec["tasks"]} == {20281126}
    assert {task["sample_count"] for task in spec["tasks"]} == {32}


def test_v96_uses_pareto_knapsack_and_non_group_output_stability() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    protocol = _protocol(package)
    base = load_config(package / spec["candidate_config"])
    common = protocol.apply_common_cache_policy(base, spec["common_cache_policy"])
    applied = protocol.apply_candidate(common, spec["candidates"][0])
    selection = applied.selection

    assert spec["design"]["shortlist_per_route"] == "none"
    assert "not_dominated" in spec["design"]["frontier_rule"]
    assert "multiple_choice" in spec["design"]["group_knapsack_rule"]
    assert spec["design"]["benchmark_labels_used"] is False
    assert selection.stable_head_query_coverage_layout_group_reservation
    assert selection.stable_head_query_coverage_layout_group_atomic_selection
    assert (
        selection.stable_head_query_coverage_layout_group_atomic_selection_mode == "pareto_knapsack"
    )
    assert selection.stable_head_query_coverage_cross_head_stability_gate
    assert selection.stable_head_query_coverage_cross_head_stability_risk_mode == "output_impact"
    assert selection.stable_head_query_coverage_cross_head_minimum_normalized_gain == 0.000001
    assert selection.stable_head_query_coverage_cross_head_gain_to_prompt_loss_ratio == 1.01
    assert not selection.stable_head_query_coverage_output_impact_reservation
    assert not applied.prototypes.enabled
    assert not applied.residual.enabled
    assert not applied.repair.enabled


def test_v96_keeps_every_gate_and_submits_no_heldout() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    objective = spec["calibration_objective"]
    robust = spec["robust_fidelity_analysis"]
    layout_audit = spec["layout_group_audit"]
    mechanism_audit = spec["pareto_knapsack_audit"]

    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert objective["require_zero_residual_bytes"] is True
    assert robust["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fold_count"] == 4
    assert robust["expected_samples_per_fold"] == 8
    assert layout_audit["minimum_per_sample_reservation_utilization"] == 0.75
    assert layout_audit["require_atomic_membership_through_final_selection"] is True
    assert mechanism_audit["activity_is_additional_freeze_gate"] is True

    gpu = (package / "slurm/choice_c_v96_pareto_atomic_group_a100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:a100:1" in gpu
    assert "#SBATCH --array=0-1%2" in gpu
    assert "#SBATCH --reservation=bi_fox_dgx" in gpu

    submit = (package / "scripts/submit_choice_c_v96.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)
    assert "heldout" not in submitted
    assert "validate_choice_c_v95_negative_seal.py" in submit
    assert "choice_c_v96_materialize_pareto_atomic_group_calibration.sbatch" in submit
    assert "choice_c_v96_pareto_atomic_group_a100.sbatch" in submit
    assert "choice_c_v96_freeze.sbatch" in submit
    assert "assert_clean_worktree.sh" in submit
