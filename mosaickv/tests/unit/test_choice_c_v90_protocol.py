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
            package
            / "configs/experiments/choice_c_v90_cross_head_query_byte_allocation_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _protocol(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v90_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v90_is_fresh_disjoint_and_does_not_use_v89_outcomes() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    boundary = spec["scientific_boundary"]

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["execution_status"] == "prepared_waiting_for_clean_commit"
    assert spec["prior_result_exclusion_boundary"] == "v89_and_all_prior_choice_c_runs"
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v89_negative_calibration_seal.yaml"
    )
    assert boundary["aggregate_v89_mechanism_activity_used_for_design"] is True
    for key in (
        "v89_answers_used_for_design",
        "v89_references_used_for_design",
        "v89_task_scores_used_for_design",
        "v89_failure_identities_used_for_design",
        "v89_generated_token_ids_used_for_design",
        "v89_per_sample_telemetry_used_for_design",
    ):
        assert boundary[key] is False
    assert {task["selection_seed"] for task in spec["tasks"]} == {20280820}
    assert {task["sample_count"] for task in spec["tasks"]} == {32}


def test_v90_allocates_capped_query_reservation_across_heads() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    base = load_config(package / spec["candidate_config"])
    candidate = spec["candidates"][0]
    protocol = _protocol(package)

    common = protocol.apply_common_cache_policy(base, spec["common_cache_policy"])
    applied = protocol.apply_candidate(common, candidate)
    selection = applied.selection

    assert spec["design"]["formula"] == "sum_q min(tau_q, sum_{i in S} a_qi)"
    assert spec["design"]["allocation_scope"] == (
        "query_conditioned_exact_reservation_bytes_across_stable_heads"
    )
    assert spec["protocol_amendment"]["ragged_physical_head_storage_claimed"] is False
    assert candidate["tier_policy"] == "exact_only_replacement"
    assert candidate["overrides"]["active_budget_bytes"] == 28_500_000
    assert not applied.prototypes.enabled
    assert not applied.residual.enabled
    assert not applied.repair.enabled
    assert selection.stable_head_query_coverage_initial_selection
    assert selection.stable_head_query_coverage_initial_objective == "capped_cumulative"
    assert selection.stable_head_query_coverage_initial_cap_fraction == 0.5
    assert selection.stable_head_query_coverage_cross_head_allocation


def test_v90_keeps_gates_and_submits_no_heldout_campaign() -> None:
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

    gpu = (package / "slurm/choice_c_v90_cross_head_a100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:a100:1" in gpu
    assert "#SBATCH --array=0-1%2" in gpu
    assert "#SBATCH --reservation=bi_fox_dgx" in gpu

    submit = (package / "scripts/submit_choice_c_v90.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)
    assert "heldout" not in submitted
    assert "MOSAICKV_CPU_PARTITION" in submit
    assert "choice_c_v90_materialize_cross_head_calibration.sbatch" in submit
    assert "choice_c_v90_cross_head_a100.sbatch" in submit
    assert "choice_c_v90_freeze.sbatch" in submit
    assert "assert_clean_worktree.sh" in submit
    assert "validate_choice_c_v89_negative_seal.py" in submit
