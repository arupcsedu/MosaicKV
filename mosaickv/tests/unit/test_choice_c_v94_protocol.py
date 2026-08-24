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
            package / "configs/experiments/choice_c_v94_coherent_layout_group_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _protocol(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v94_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v94_is_fresh_disjoint_and_uses_no_v93_outcomes() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    boundary = spec["scientific_boundary"]

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["execution_status"] == "prepared_waiting_for_clean_commit"
    assert spec["prior_result_exclusion_boundary"] == "v93_and_all_prior_choice_c_runs"
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v93_negative_calibration_seal.yaml"
    )
    assert boundary["aggregate_v93_mechanism_activity_used_for_design"] is True
    for key in (
        "v93_answers_used_for_design",
        "v93_references_used_for_design",
        "v93_task_scores_used_for_design",
        "v93_failure_identities_used_for_design",
        "v93_generated_token_ids_used_for_design",
        "v93_per_sample_telemetry_used_for_design",
    ):
        assert boundary[key] is False
    assert {task["selection_seed"] for task in spec["tasks"]} == {20280924}
    assert {task["sample_count"] for task in spec["tasks"]} == {32}


def test_v94_protects_complete_layout_groups_before_query_coverage() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    protocol = _protocol(package)
    base = load_config(package / spec["candidate_config"])
    common = protocol.apply_common_cache_policy(base, spec["common_cache_policy"])
    applied = protocol.apply_candidate(common, spec["candidates"][0])
    selection = applied.selection

    assert spec["design"]["protected_group_stage"] == "before_query_coverage_allocation"
    assert spec["design"]["minimum_blocks_per_group"] == 2
    assert spec["design"]["maximum_blocks_per_group"] == 4
    assert spec["design"]["ocr_metadata_claimed"] is False
    assert selection.stable_head_query_coverage_initial_selection
    assert selection.stable_head_query_coverage_cross_head_allocation
    assert selection.stable_head_query_coverage_layout_group_reservation
    assert selection.stable_head_query_coverage_layout_group_fraction == 0.5
    assert selection.stable_head_query_coverage_layout_group_maximum_blocks == 4
    assert selection.stable_head_query_coverage_layout_group_row_bins == 4
    assert not selection.stable_head_query_coverage_output_impact_reservation
    assert not selection.stable_head_query_coverage_cross_head_stability_gate
    assert not applied.prototypes.enabled
    assert not applied.residual.enabled
    assert not applied.repair.enabled


def test_v94_keeps_unchanged_gates_and_submits_no_heldout() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    objective = spec["calibration_objective"]
    robust = spec["robust_fidelity_analysis"]
    activity = spec["layout_group_audit"]

    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert objective["require_zero_residual_bytes"] is True
    assert robust["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fold_count"] == 4
    assert robust["expected_samples_per_fold"] == 8
    assert activity["activity_is_an_additional_freeze_gate"] is True

    gpu = (package / "slurm/choice_c_v94_coherent_layout_a100.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:a100:1" in gpu
    assert "#SBATCH --array=0-1%2" in gpu
    assert "#SBATCH --reservation=bi_fox_dgx" in gpu

    submit = (package / "scripts/submit_choice_c_v94.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)
    assert "heldout" not in submitted
    assert "validate_choice_c_v93_negative_seal.py" in submit
    assert "choice_c_v94_materialize_coherent_layout_calibration.sbatch" in submit
    assert "choice_c_v94_coherent_layout_a100.sbatch" in submit
    assert "choice_c_v94_freeze.sbatch" in submit
    assert "assert_clean_worktree.sh" in submit
