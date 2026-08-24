#!/usr/bin/env python3
"""Fail closed if the prepared V99 protocol weakens a sealed V98 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml
from choice_c_v74_protocol import apply_candidate, apply_common_cache_policy

from mosaickv.config import load_config
from mosaickv.types import MosaicKVMethod


def validate(spec_path: Path, seal_path: Path) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    seal = cast("dict[str, Any]", yaml.safe_load(seal_path.read_text(encoding="utf-8")))
    package_root = Path(__file__).resolve().parents[1]
    if (
        spec.get("experiment_id")
        != "choice_c_v99_progressive_logit_safe_eviction_calibration"
        or spec.get("prior_result_exclusion_boundary")
        != "v98_and_all_prior_choice_c_runs"
        or spec.get("selection_rule_frozen_before_calibration") is not True
        or seal.get("experiment_id") != "choice_c_v98_short_horizon_logit_calibration"
        or seal.get("overall_freeze_gate_pass") is not False
        or seal.get("heldout_evaluation_authorized") is not False
    ):
        raise RuntimeError("V99 does not descend from the sealed negative V98 protocol")
    expected_parent = "configs/ledgers/choice_c_v98_negative_calibration_seal.yaml"
    if spec.get("parent_negative_result_seal") != expected_parent:
        raise RuntimeError("V99 parent negative-result seal changed")

    objective = cast("dict[str, Any]", spec["calibration_objective"])
    expected_objective = {
        "labels_accessed": False,
        "official_task_scores_accessed": False,
        "minimum_active_reduction_fraction_vs_prompt_topk": 0.30,
        "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk": 0.30,
        "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
        "maximum_prototype_logical_fraction": 0.0,
        "require_zero_residual_bytes": True,
        "require_all_rows_complete": True,
    }
    if any(objective.get(key) != value for key, value in expected_objective.items()):
        raise RuntimeError("V99 changed an unchanged task-level memory or fidelity gate")

    robust = cast("dict[str, Any]", spec["robust_fidelity_analysis"])
    expected_robust = {
        "labels_accessed": False,
        "official_task_scores_accessed": False,
        "fold_count": 4,
        "expected_samples_per_fold": 8,
        "require_every_task_and_fold": True,
        "require_base_memory_and_fidelity_gates": True,
        "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
        "fullkv_token_noninferiority_margin_vs_prompt_topk": 0.05,
    }
    if any(robust.get(key) != value for key, value in expected_robust.items()):
        raise RuntimeError("V99 changed an unchanged robust-fold fidelity gate")

    progressive = cast("dict[str, Any]", spec["progressive_logit_safe_eviction_analysis"])
    expected_progressive = {
        "labels_accessed": False,
        "official_task_scores_accessed": False,
        "horizon_steps": 2,
        "maximum_forward_kl": 0.02,
        "maximum_absolute_logit_difference": 1.0,
        "maximum_probes": 128,
        "require_start_from_fullkv": True,
        "require_every_sample_to_reach_target": True,
        "require_every_accepted_checkpoint_to_pass": True,
        "require_greedy_token_match": True,
        "require_fullkv_prefill_reuse": True,
        "forbid_second_prefill": True,
        "require_probe_state_discard_before_measured_decode": True,
    }
    if any(progressive.get(key) != value for key, value in expected_progressive.items()):
        raise RuntimeError("V99 progressive FullKV logit-safety gate changed")
    layout = cast("dict[str, Any]", spec["layout_group_audit"])
    if layout.get("minimum_per_sample_reservation_utilization") != 0.75:
        raise RuntimeError("V99 weakened the atomic-group utilization gate")

    tasks = cast("list[dict[str, Any]]", spec["tasks"])
    if (
        {str(task["task"]) for task in tasks} != {"docvqa", "textvqa"}
        or {int(task["sample_count"]) for task in tasks} != {32}
        or {int(task["selection_seed"]) for task in tasks} != {20300329}
    ):
        raise RuntimeError("V99 fresh calibration task or sample contract changed")
    boundary = cast("dict[str, Any]", spec["scientific_boundary"])
    forbidden_inputs = (
        "v98_answers_used_for_design",
        "v98_references_used_for_design",
        "v98_task_scores_used_for_design",
        "v98_failure_identities_used_for_design",
        "v98_generated_token_ids_used_for_design",
        "v98_per_sample_telemetry_used_for_design",
        "calibration_labels_accessed",
        "official_task_scores_accessed",
    )
    if any(boundary.get(field) is not False for field in forbidden_inputs):
        raise RuntimeError("V99 scientific boundary permits forbidden V98 or label input")

    candidates = cast("list[dict[str, Any]]", spec["candidates"])
    if len(candidates) != 1:
        raise RuntimeError("V99 must evaluate one frozen progressive candidate")
    base = load_config(package_root / str(spec["candidate_config"]))
    common = apply_common_cache_policy(base, cast("dict[str, Any]", spec["common_cache_policy"]))
    config = apply_candidate(common, candidates[0])
    selection = config.selection
    if (
        config.method is not MosaicKVMethod.MOSAICKV_EXACT
        or not selection.stable_head_progressive_logit_safe_eviction
        or selection.stable_head_short_horizon_logit_stability
        or selection.stable_head_progressive_eviction_horizon != 2
        or selection.stable_head_progressive_eviction_maximum_kl != 0.02
        or selection.stable_head_progressive_eviction_maximum_logit_difference != 1.0
        or config.cache.budget_value != 28_500_000
        or config.cache.block_size != 4
        or config.prototypes.enabled
        or config.residual.enabled
        or config.repair.enabled
    ):
        raise RuntimeError("V99 resolved candidate is not the frozen exact-only progressive design")
    return {
        "experiment_id": spec["experiment_id"],
        "candidate_id": candidates[0]["id"],
        "task_count": len(tasks),
        "samples_per_task": 32,
        "active_memory_reduction_gate": 0.30,
        "fidelity_noninferiority_margin": 0.05,
        "group_utilization_gate": 0.75,
        "status": "valid_prepared_protocol",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--seal", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.spec.resolve(), args.seal.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
