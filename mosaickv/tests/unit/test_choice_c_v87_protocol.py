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
            package / "configs/experiments/choice_c_v87_prompt_query_coverage_development.yaml"
        ).read_text(encoding="utf-8")
    )


def _protocol(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v87_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v87_is_fresh_disjoint_and_does_not_use_v86_labels() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    boundary = spec["scientific_boundary"]

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["prior_result_exclusion_boundary"] == "v86_and_all_prior_choice_c_runs"
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v86_negative_calibration_seal.yaml"
    )
    assert boundary["prior_choice_c_runs_used_only_as_sample_exclusion_boundaries"] is True
    for key in (
        "v86_answers_used_for_design",
        "v86_references_used_for_design",
        "v86_task_scores_used_for_design",
        "v86_failure_identities_used_for_design",
        "v86_generated_token_ids_used_for_design",
        "v86_per_sample_telemetry_used_for_design",
    ):
        assert boundary[key] is False
    assert {task["selection_seed"] for task in spec["tasks"]} == {20280817}
    assert {task["sample_count"] for task in spec["tasks"]} == {32}

    materializer = (
        package / "slurm/choice_c_v87_materialize_query_coverage_calibration.sbatch"
    ).read_text(encoding="utf-8")
    assert "/choice-c-v*/inputs" in materializer
    assert "--seed 20280817" in materializer
    assert "--subset-size 32" in materializer


def test_v87_candidate_is_same_byte_exact_only_query_coverage() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    base = load_config(package / spec["candidate_config"])
    candidate = spec["candidates"][0]
    protocol = _protocol(package)

    common = protocol.apply_common_cache_policy(base, spec["common_cache_policy"])
    applied = protocol.apply_candidate(common, candidate)
    selection = applied.selection

    assert candidate["tier_policy"] == "exact_only_replacement"
    assert candidate["overrides"]["active_budget_bytes"] == 28_500_000
    assert not applied.prototypes.enabled
    assert not applied.residual.enabled
    assert not applied.repair.enabled
    assert selection.stable_head_query_coverage_reranking
    assert selection.stable_head_query_coverage_minimum_prompt_mass_fraction == 0.9975
    assert selection.stable_head_query_coverage_maximum_replacements == 1
    assert selection.stable_head_query_coverage_minimum_normalized_gain == 0.001
    assert not selection.stable_head_adaptive_visual_swaps
    assert not selection.stable_head_future_query_reranking


def test_v87_keeps_memory_and_fidelity_gates_and_submits_no_heldout() -> None:
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

    gpu = (package / "slurm/choice_c_v87_query_coverage_calibration_a100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:a100:1" in gpu
    assert "#SBATCH --array=0-1%2" in gpu
    assert "#SBATCH --reservation=bi_fox_dgx" in gpu

    submit = (package / "scripts/submit_choice_c_v87.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)
    assert "heldout" not in submitted
    assert "choice_c_v87_materialize_query_coverage_calibration.sbatch" in submit
    assert "choice_c_v87_query_coverage_calibration_a100.sbatch" in submit
    assert "choice_c_v87_freeze.sbatch" in submit
    assert "assert_clean_worktree.sh" in submit
