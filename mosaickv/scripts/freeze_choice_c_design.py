#!/usr/bin/env python3
"""Freeze a label-blind Choice C design candidate without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml

from mosaickv.selection import FidelityObservation, select_fidelity_profile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("sample_ids"), list):
        raise ValueError(f"invalid sample ledger: {path}")
    return cast("dict[str, Any]", value)


def _forecast_regrets(trace_root: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for path in sorted(trace_root.glob("*.json")):
        record = json.loads(path.read_text())
        diagnostic = record.get("forecast_quality_diagnostic", {})
        value = diagnostic.get("mean_forecast_regret")
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise RuntimeError(f"trace has no finite forecast regret: {path}")
        result[str(record["sample_id"])] = float(value)
    return result


def _three_tier_samples(trace_root: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for path in sorted(trace_root.glob("*.json")):
        record = json.loads(path.read_text())
        result[str(record["sample_id"])] = bool(
            record.get("effective_method") == "mosaickv_full"
            and record.get("tier_mode") == "three_tier"
        )
    return result


def freeze(spec_path: Path) -> int:
    project_root = spec_path.parents[2]
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    output_root = Path(str(spec["output_root"])).resolve()
    calibration_path = project_root / str(spec["calibration_ledger"])
    validation_path = project_root / str(spec["calibration_validation_ledger"])
    heldout_path = project_root / str(spec["evaluation_ledger"])
    calibration = _ledger(calibration_path)
    validation = _ledger(validation_path)
    heldout = _ledger(heldout_path)
    calibration_ids = set(cast("list[str]", calibration["sample_ids"]))
    validation_ids = set(cast("list[str]", validation["sample_ids"]))
    heldout_ids = set(cast("list[str]", heldout["sample_ids"]))
    if calibration_ids & validation_ids or calibration_ids & heldout_ids:
        raise RuntimeError("design calibration overlaps a sealed downstream ledger")
    if validation_ids & heldout_ids:
        raise RuntimeError("internal validation overlaps the held-out ledger")

    spec_sha = _sha(spec_path)[:10]
    candidates = cast("dict[str, dict[str, Any]]", spec["choice_c_candidates"])
    observations: list[FidelityObservation] = []
    frames: list[pd.DataFrame] = []
    candidate_eligibility: dict[str, dict[str, object]] = {}
    for variant in sorted(candidates):
        run_id = f"choice-c-calibration-{variant}-{spec_sha}"
        parquet_path = output_root / run_id / "aggregate.parquet"
        frame = pd.read_parquet(parquet_path)
        if len(frame) != len(calibration_ids) or set(frame["sample_id"]) != calibration_ids:
            raise RuntimeError(f"candidate does not cover the design ledger: {variant}")
        if set(frame["status"]) != {"completed"} or frame["error"].notna().any():
            raise RuntimeError(f"candidate contains a failed sample: {variant}")
        regrets = _forecast_regrets(output_root / "traces" / run_id)
        if set(regrets) != calibration_ids:
            raise RuntimeError(f"forecast diagnostics are incomplete: {variant}")
        three_tier = _three_tier_samples(output_root / "traces" / run_id)
        if set(three_tier) != calibration_ids:
            raise RuntimeError(f"tier-mode traces are incomplete: {variant}")
        genuine = all(three_tier.values())
        candidate_eligibility[variant] = {
            "eligible": genuine,
            "genuine_three_tier_rows": sum(three_tier.values()),
            "required_rows": len(calibration_ids),
        }
        tagged = frame.copy()
        tagged["variant"] = variant
        frames.append(tagged)
        if not genuine:
            continue
        for row in frame.itertuples(index=False):
            observations.append(
                FidelityObservation(
                    variant=variant,
                    sample_id=str(row.sample_id),
                    active_kv_bytes=int(row.active_kv_bytes),
                    token_agreement=float(row.token_agreement),
                    logit_kl=float(row.logit_kl),
                    attention_output_error=float(row.attention_output_error),
                    forecast_regret=regrets[str(row.sample_id)],
                )
            )
    calibration_rows = pd.concat(frames, ignore_index=True)
    calibration_rows.to_parquet(output_root / "calibration_rows.parquet", index=False)
    rule = cast("dict[str, Any]", spec["calibration_rule"])
    profile = select_fidelity_profile(
        tuple(observations),
        parameters_by_variant=candidates,
        calibration_dataset_id=str(calibration["dataset_id"]),
        calibration_dataset_revision=str(calibration["dataset_revision"]),
        calibration_split=str(calibration["split"]),
        evaluation_sample_ids=tuple(sorted(heldout_ids)),
        target_active_kv_bytes=math.ceil(
            max(calibration_rows.groupby("variant")["active_kv_bytes"].mean())
        ),
        tail_weight=float(rule["tail_weight"]),
        bootstrap_resamples=int(rule["bootstrap_resamples"]),
        bootstrap_seed=int(rule["bootstrap_seed"]),
        minimum_selection_frequency=float(rule["minimum_selection_frequency"]),
        minimum_calibration_samples=int(rule["minimum_samples"]),
    )
    profile.write(output_root / "frozen_fidelity_profile.json")
    scores = {
        variant: float(frame["task_score"].mean())
        for variant, frame in calibration_rows.groupby("variant")
    }
    gate = {
        "status": "passed",
        "selected_variant": profile.selected_variant,
        "profile_sha256": profile.sha256,
        "bootstrap_selection_frequency": profile.bootstrap_selection_frequency,
        "objective_value": profile.objective_value,
        "task_scores_reported_but_not_used_for_selection": scores,
        "candidate_three_tier_eligibility": candidate_eligibility,
        "labels_or_task_scores_used": False,
        "internal_validation_started": False,
        "heldout_evaluation_started": False,
    }
    (output_root / "design_calibration_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    return freeze(args.spec.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
