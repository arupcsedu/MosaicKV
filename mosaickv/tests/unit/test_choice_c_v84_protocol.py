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
            package / "configs/experiments/choice_c_v84_disjoint_risk_gated_hybrid_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _protocol(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v84_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v84_uses_v83_only_as_a_disjoint_exclusion_boundary() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    boundary = spec["scientific_boundary"]

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["parent_negative_result_seal"].endswith("choice_c_v83_negative_heldout_seal.yaml")
    assert boundary["v83_used_only_as_exclusion_boundary"] is True
    assert boundary["v83_answers_used_for_design"] is False
    assert boundary["v83_references_used_for_design"] is False
    assert boundary["v83_task_scores_used_for_design"] is False
    assert boundary["v83_failure_identities_used_for_design"] is False
    assert boundary["v83_generated_token_ids_used_for_design"] is False
    assert boundary["v83_per_sample_telemetry_used_for_design"] is False
    assert boundary["design_evidence_source"] == "v82_label_blind_route_telemetry_only"
    assert {task["selection_seed"] for task in spec["tasks"]} == {20280107}
    assert {task["sample_count"] for task in spec["tasks"]} == {96}

    materializer = (package / "slurm/choice_c_v84_materialize_risk_calibration.sbatch").read_text(
        encoding="utf-8"
    )
    assert "choice-c-v83-v82-frozen-untouched-heldout-quality-memory-v1/inputs" in materializer
    assert "required sealed v83 exclusion directory is missing" in materializer
    assert "choice-c-v82-disjoint-hybrid-divergence-calibration-v1/inputs" in materializer
    assert "--seed 20280107" in materializer
    assert "--subset-size 96" in materializer


def test_v84_candidates_are_exact_only_same_byte_and_risk_gated() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    base = load_config(package / spec["candidate_config"])
    candidates = spec["candidates"]

    assert len(candidates) == 3
    assert {candidate["tier_policy"] for candidate in candidates} == {"exact_only_replacement"}
    assert {candidate["overrides"]["active_budget_bytes"] for candidate in candidates} == {
        27_500_000
    }
    assert {candidate["overrides"]["maximum_prototypes_per_head"] for candidate in candidates} == {
        0
    }

    protocol = _protocol(package)
    common = protocol.apply_common_cache_policy(base, spec["common_cache_policy"])
    applied = [protocol.apply_candidate(common, candidate) for candidate in candidates]
    assert all(not config.prototypes.enabled for config in applied)
    assert all(not config.residual.enabled for config in applied)
    assert all(not config.repair.enabled for config in applied)

    control = candidates[0]["overrides"]
    assert control["stable_head_future_query_reranking"] is False
    assert control["stable_head_future_risk_gate"] is False
    risk_arms = [
        candidate["overrides"]
        for candidate in candidates
        if candidate["overrides"]["stable_head_future_risk_gate"]
    ]
    assert len(risk_arms) == 2
    assert all(arm["stable_head_future_query_reranking"] for arm in risk_arms)
    assert all(arm["stable_head_future_divergence_fail_closed"] for arm in risk_arms)
    assert all(arm["stable_head_future_maximum_normalized_prompt_loss"] < 1 for arm in risk_arms)
    assert all(arm["stable_head_future_minimum_normalized_gain"] > 0 for arm in risk_arms)


def test_v84_keeps_gates_and_submission_chain_has_no_heldout_job() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    objective = spec["calibration_objective"]
    risk = spec["future_rerank_risk_analysis"]

    assert objective["labels_accessed"] is False
    assert objective["official_task_scores_accessed"] is False
    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert objective["maximum_prototype_logical_fraction"] == 0.0
    assert objective["require_zero_residual_bytes"] is True
    assert risk["require_positive_accepted_reranks_each_task"] is True
    assert risk["require_positive_risk_rejections_each_task"] is True
    assert risk["require_not_below_control_each_task"] is True
    assert risk["fullkv_answer_noninferiority_margin_vs_control"] == 0.0

    submit = (package / "scripts/submit_choice_c_v84.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)
    assert "heldout" not in submitted
    assert "choice_c_v84_materialize_risk_calibration.sbatch" in submit
    assert "choice_c_v84_risk_calibration_a100.sbatch" in submit
    assert "choice_c_v84_freeze.sbatch" in submit
