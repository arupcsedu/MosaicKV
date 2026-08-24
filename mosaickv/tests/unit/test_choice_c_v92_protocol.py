from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml

from mosaickv.config import load_config


def _spec(package: Path) -> dict:
    return yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v92_output_impact_stability_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _protocol(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v92_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v92_is_fresh_disjoint_and_uses_no_v91_outcomes() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    boundary = spec["scientific_boundary"]

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["execution_status"] == "prepared_waiting_for_clean_commit"
    assert spec["prior_result_exclusion_boundary"] == "v91_and_all_prior_choice_c_runs"
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v91_negative_calibration_seal.yaml"
    )
    assert boundary["aggregate_v91_mechanism_activity_used_for_design"] is True
    for key in (
        "v91_answers_used_for_design",
        "v91_references_used_for_design",
        "v91_task_scores_used_for_design",
        "v91_failure_identities_used_for_design",
        "v91_generated_token_ids_used_for_design",
        "v91_per_sample_telemetry_used_for_design",
    ):
        assert boundary[key] is False
    assert {task["selection_seed"] for task in spec["tasks"]} == {20280822}
    assert {task["sample_count"] for task in spec["tasks"]} == {32}


def test_v92_replaces_only_stability_risk_and_keeps_numeric_gates() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    base = load_config(package / spec["candidate_config"])
    applied = _protocol(package).apply_candidate(
        _protocol(package).apply_common_cache_policy(base, spec["common_cache_policy"]),
        spec["candidates"][0],
    )
    selection = applied.selection

    assert spec["design"]["stability_risk_mode"] == "output_impact"
    assert spec["design"]["minimum_normalized_coverage_gain"] == 0.005
    assert spec["design"]["minimum_gain_to_output_impact_risk_ratio"] == 4.0
    assert selection.stable_head_query_coverage_initial_selection
    assert selection.stable_head_query_coverage_cross_head_allocation
    assert selection.stable_head_query_coverage_cross_head_stability_gate
    assert selection.stable_head_query_coverage_cross_head_stability_risk_mode == "output_impact"
    assert selection.stable_head_query_coverage_cross_head_minimum_normalized_gain == 0.005
    assert selection.stable_head_query_coverage_cross_head_gain_to_prompt_loss_ratio == 4.0
    assert not applied.prototypes.enabled
    assert not applied.residual.enabled
    assert not applied.repair.enabled


def test_v92_keeps_unchanged_gates_and_submits_no_heldout() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    objective = spec["calibration_objective"]
    robust = spec["robust_fidelity_analysis"]

    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert objective["require_zero_residual_bytes"] is True
    assert robust["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fold_count"] == 4
    assert robust["expected_samples_per_fold"] == 8

    gpu = (package / "slurm/choice_c_v92_output_impact_a100.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:a100:1" in gpu
    assert "#SBATCH --array=0-1%2" in gpu
    assert "#SBATCH --reservation=bi_fox_dgx" in gpu

    submit = (package / "scripts/submit_choice_c_v92.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)
    assert "heldout" not in submitted
    assert "validate_choice_c_v91_negative_seal.py" in submit
    assert "choice_c_v92_materialize_output_impact_calibration.sbatch" in submit
    assert "choice_c_v92_output_impact_a100.sbatch" in submit
    assert "choice_c_v92_freeze.sbatch" in submit
    assert "assert_clean_worktree.sh" in submit
