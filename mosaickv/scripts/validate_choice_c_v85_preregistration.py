#!/usr/bin/env python3
"""Fail-closed validation for the frozen v85 held-out preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(spec_path: Path, frozen_path: Path) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    frozen = cast("dict[str, Any]", yaml.safe_load(frozen_path.read_text(encoding="utf-8")))
    package_root = Path(__file__).resolve().parents[1]
    if spec.get("experiment_id") != "choice_c_v85_v84_frozen_untouched_heldout_quality_memory":
        raise RuntimeError("unexpected v85 experiment id")
    if spec.get("frozen_before_sample_materialization") is not True:
        raise RuntimeError("v85 candidate was not frozen before sample materialization")
    boundary = cast("dict[str, Any]", spec["scientific_boundary"])
    required_false = (
        "v84_candidate_retuning_allowed",
        "heldout_answers_used_before_execution",
        "heldout_references_used_before_execution",
        "heldout_task_scores_used_before_execution",
        "heldout_failure_identities_used_before_execution",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise RuntimeError("v85 scientific boundary permits held-out tuning or label access")
    if _sha(frozen_path) != spec.get("frozen_candidate_sha256"):
        raise RuntimeError("v85 frozen candidate hash differs from the preregistration")
    if (
        frozen.get("classification") != "frozen_v84_label_blind_risk_gated_hybrid_candidate"
        or frozen.get("labels_accessed") is not False
        or frozen.get("official_task_scores_accessed") is not False
    ):
        raise RuntimeError("v85 frozen candidate lacks label-blind v84 provenance")
    if frozen.get("selected_candidate") != spec.get("selected_candidate"):
        raise RuntimeError("v85 selected candidate differs from the v84 freeze")
    candidate = cast("dict[str, Any]", frozen["candidate"])
    overrides = cast("dict[str, Any]", candidate["overrides"])
    if (
        candidate.get("tier_policy") != "exact_only_replacement"
        or int(overrides.get("maximum_prototypes_per_head", -1)) != 0
        or spec.get("candidate_effective_method") != "mosaickv_exact"
        or spec.get("supports_prototype_or_repair_claim") is not False
    ):
        raise RuntimeError("v85 must preserve the exact-only effective-method label")
    if frozen.get("calibration_result_sha256") != spec.get("v84_calibration_result_sha256"):
        raise RuntimeError("v85 calibration result hash differs from the v84 freeze")
    if frozen.get("calibration_source_snapshot_sha256") != spec.get(
        "runtime_source_snapshot_sha256"
    ):
        raise RuntimeError("v85 runtime snapshot differs from the v84 calibration snapshot")
    if frozen.get("parent_negative_result_seal") != spec.get("parent_negative_result_seal"):
        raise RuntimeError("v85 parent seal path differs from the v84 freeze")
    calibration_spec = package_root / str(spec["calibration_spec"])
    candidate_config = package_root / str(spec["candidate_config"])
    parent_seal = Path(str(spec["parent_negative_result_seal"]))
    if frozen.get("calibration_spec_sha256") != _sha(calibration_spec):
        raise RuntimeError("v85 calibration specification differs from the v84 freeze")
    if frozen.get("candidate_config_sha256") != _sha(candidate_config):
        raise RuntimeError("v85 base candidate configuration differs from the v84 freeze")
    if frozen.get("parent_negative_result_seal_sha256") != _sha(parent_seal):
        raise RuntimeError("v85 parent seal content differs from the v84 freeze")
    tasks = cast("list[dict[str, Any]]", spec["tasks"])
    if (
        {str(task["task"]) for task in tasks} != {"docvqa", "textvqa"}
        or {int(task["sample_count"]) for task in tasks} != {128}
        or {int(task["selection_seed"]) for task in tasks} != {20280113}
    ):
        raise RuntimeError("v85 held-out task matrix differs from the preregistration")
    gate = cast("dict[str, Any]", spec["primary_gate"])
    if (
        float(gate.get("minimum_active_memory_reduction_fraction", -1)) != 0.30
        or float(gate.get("minimum_active_plus_residual_memory_reduction_fraction", -1)) != 0.30
        or float(gate.get("task_score_noninferiority_margin", -1)) != 0.05
        or gate.get("require_nonnegative_mean_score_difference") is not True
        or gate.get("require_bootstrap_lower_bound_above_negative_margin") is not True
        or gate.get("require_each_task_to_pass") is not True
    ):
        raise RuntimeError("v85 memory or paired-quality gate changed")
    return {
        "experiment_id": spec["experiment_id"],
        "frozen_candidate_sha256": _sha(frozen_path),
        "selected_candidate": frozen["selected_candidate"],
        "runtime_source_commit": spec["runtime_source_commit"],
        "tasks": {str(task["task"]): int(task["sample_count"]) for task in tasks},
        "status": "valid",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.spec.resolve(), args.frozen.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
