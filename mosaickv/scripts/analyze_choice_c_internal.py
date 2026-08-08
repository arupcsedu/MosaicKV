#!/usr/bin/env python3
"""Apply the frozen byte-matched Choice C gate to internal validation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fidelity(frame: pd.DataFrame) -> pd.Series:
    return (
        1.0
        - frame["token_agreement"].astype(float)
        + frame["logit_kl"].astype(float).map(math.log1p)
        + frame["attention_output_error"].astype(float)
    )


def _bootstrap_relative_improvement(
    mosaic: tuple[float, ...],
    competitor: tuple[float, ...],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    values: list[float] = []
    for _replicate in range(resamples):
        indices = tuple(generator.randrange(len(mosaic)) for _index in mosaic)
        mosaic_mean = statistics.fmean(mosaic[index] for index in indices)
        competitor_mean = statistics.fmean(competitor[index] for index in indices)
        values.append(
            (competitor_mean - mosaic_mean) / competitor_mean
            if competitor_mean > 0
            else float("-inf")
        )
    finite = tuple(sorted(value for value in values if math.isfinite(value)))
    if not finite:
        return float("-inf"), float("-inf")

    def quantile(probability: float) -> float:
        offset = probability * (len(finite) - 1)
        lower = math.floor(offset)
        upper = math.ceil(offset)
        if lower == upper:
            return finite[lower]
        weight = offset - lower
        return finite[lower] * (1 - weight) + finite[upper] * weight

    return quantile(0.025), quantile(0.975)


def analyze(spec_path: Path) -> int:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    output_root = Path(
        os.environ.get("MOSAICKV_OUTPUT_ROOT_OVERRIDE", str(spec["output_root"]))
    ).resolve()
    baseline_root = Path(
        os.environ.get("MOSAICKV_BASELINE_OUTPUT_ROOT", str(output_root))
    ).resolve()
    methods = tuple(cast("list[str]", spec["methods"]))
    spec_sha = _sha(spec_path)[:10]
    frames: dict[str, pd.DataFrame] = {}
    for method in methods:
        method_root = output_root if method == "mosaickv_full" else baseline_root
        path = (
            method_root / f"choice-c-internal-validation-{method}-{spec_sha}" / "aggregate.parquet"
        )
        frame = pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True)
        if frame.empty or set(frame["status"]) != {"completed"}:
            raise RuntimeError(f"internal method is incomplete: {method}")
        if frame["error"].notna().any() or frame["sample_id"].duplicated().any():
            raise RuntimeError(f"internal method contains failed or duplicate rows: {method}")
        numeric = (
            "task_score",
            "ttft",
            "prefill_time",
            "compression_time",
            "decode_time",
            "end_to_end_time",
            "active_kv_bytes",
            "residual_kv_bytes",
            "token_agreement",
            "logit_kl",
            "attention_output_error",
        )
        if any(not math.isfinite(float(value)) for name in numeric for value in frame[name]):
            raise RuntimeError(f"internal method has a missing/nonfinite metric: {method}")
        frames[method] = frame
    expected_ids = tuple(frames[methods[0]]["sample_id"])
    if any(tuple(frame["sample_id"]) != expected_ids for frame in frames.values()):
        raise RuntimeError("internal methods do not cover identical paired examples")

    combined = pd.concat(
        [frame.assign(comparison_method=method) for method, frame in frames.items()],
        ignore_index=True,
    )
    combined.to_parquet(output_root / "internal_validation_rows.parquet", index=False)
    mosaic = frames["mosaickv_full"]
    mosaic_bytes = mosaic["active_kv_bytes"].astype(float)
    tolerance = float(spec["gate"]["maximum_relative_byte_mismatch"])
    eligibility: dict[str, dict[str, float | bool]] = {}
    eligible: list[str] = []
    for method in methods:
        if method in {"full_kv", "mosaickv_full"}:
            continue
        mismatch = (
            frames[method]["active_kv_bytes"].astype(float) - mosaic_bytes
        ).abs() / mosaic_bytes
        maximum = float(mismatch.max())
        mean = float(mismatch.mean())
        admitted = maximum <= tolerance
        eligibility[method] = {
            "eligible": admitted,
            "mean_relative_byte_mismatch": mean,
            "maximum_relative_byte_mismatch": maximum,
        }
        if admitted:
            eligible.append(method)
    if eligible:
        nearest = min(
            eligible,
            key=lambda method: (float(_fidelity(frames[method]).mean()), method),
        )
        competitor_error = float(_fidelity(frames[nearest]).mean())
        mosaic_error = float(_fidelity(mosaic).mean())
        relative_improvement = (
            (competitor_error - mosaic_error) / competitor_error
            if competitor_error > 0
            else float("-inf")
        )
        ci_low, ci_high = _bootstrap_relative_improvement(
            tuple(float(value) for value in _fidelity(mosaic)),
            tuple(float(value) for value in _fidelity(frames[nearest])),
            resamples=int(spec["calibration_rule"]["bootstrap_resamples"]),
            seed=int(spec["calibration_rule"]["bootstrap_seed"]),
        )
        nearest_task_score = float(frames[nearest]["task_score"].mean())
    else:
        nearest = None
        competitor_error = relative_improvement = ci_low = ci_high = float("nan")
        mosaic_error = float(_fidelity(mosaic).mean())
        nearest_task_score = float("nan")

    trace_root = output_root / "traces" / (f"choice-c-internal-validation-mosaickv_full-{spec_sha}")
    trace_paths = tuple(sorted(trace_root.glob("*.json")))
    genuine_three_tier = len(trace_paths) == len(expected_ids) and all(
        json.loads(path.read_text()).get("effective_method") == "mosaickv_full"
        and json.loads(path.read_text()).get("tier_mode") == "three_tier"
        for path in trace_paths
    )
    required = float(spec["gate"]["required_relative_improvement_over_nearest_competitor"])
    margin = float(spec["gate"]["task_score_noninferiority_margin"])
    mosaic_task_score = float(mosaic["task_score"].mean())
    fidelity_pass = bool(eligible and relative_improvement >= required)
    task_pass = bool(eligible and mosaic_task_score >= nearest_task_score - margin)
    repair_is_zero = bool(
        all((frame["repair_count"].astype(int) == 0).all() for frame in frames.values())
    )
    passed = bool(fidelity_pass and task_pass and genuine_three_tier and repair_is_zero)

    summary_rows = []
    for method, frame in frames.items():
        summary_rows.append(
            {
                "method": method,
                "samples": len(frame),
                "mean_task_score": float(frame["task_score"].mean()),
                "mean_fidelity_error": float(_fidelity(frame).mean()),
                "mean_active_kv_bytes": float(frame["active_kv_bytes"].mean()),
                "mean_residual_kv_bytes": float(frame["residual_kv_bytes"].mean()),
                "mean_compression_seconds": float(frame["compression_time"].mean()),
                "mean_decode_seconds": float(frame["decode_time"].mean()),
                "mean_end_to_end_seconds": float(frame["end_to_end_time"].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_parquet(
        output_root / "internal_validation_summary.parquet", index=False
    )
    gate = {
        "status": "passed" if passed else "failed",
        "pass": passed,
        "sample_count": len(expected_ids),
        "nearest_eligible_competitor": nearest,
        "byte_eligibility": eligibility,
        "mosaickv_fidelity_error": mosaic_error,
        "nearest_competitor_fidelity_error": (
            competitor_error if math.isfinite(competitor_error) else None
        ),
        "relative_fidelity_improvement": (
            relative_improvement if math.isfinite(relative_improvement) else None
        ),
        "paired_bootstrap_95_ci": [
            ci_low if math.isfinite(ci_low) else None,
            ci_high if math.isfinite(ci_high) else None,
        ],
        "required_relative_improvement": required,
        "fidelity_pass": fidelity_pass,
        "mosaickv_task_score": mosaic_task_score,
        "nearest_competitor_task_score": (
            nearest_task_score if math.isfinite(nearest_task_score) else None
        ),
        "task_score_noninferiority_margin": margin,
        "task_noninferiority_pass": task_pass,
        "fullkv_task_score": float(frames["full_kv"]["task_score"].mean()),
        "fullkv_task_score_delta": (
            mosaic_task_score - float(frames["full_kv"]["task_score"].mean())
        ),
        "genuine_three_tier": genuine_three_tier,
        "repair_count_zero_for_repair_disabled_methods": repair_is_zero,
        "heldout_evaluation_started": False,
        "full_sweep_allowed": False,
    }
    (output_root / "internal_validation_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report = [
        "# Choice C internal validation gate",
        "",
        f"- Status: **{gate['status']}**",
        f"- Samples: {len(expected_ids)} disjoint internal-validation examples",
        f"- Nearest eligible competitor: `{nearest}`",
        f"- MosaicKV fidelity error: {mosaic_error:.6f}",
        f"- Nearest-competitor fidelity error: {competitor_error:.6f}",
        f"- Relative improvement: {relative_improvement:.2%}",
        f"- Paired bootstrap 95% CI: [{ci_low:.2%}, {ci_high:.2%}]",
        f"- Task score (MosaicKV / nearest): {mosaic_task_score:.6f} / {nearest_task_score:.6f}",
        f"- Genuine three-tier rows: {genuine_three_tier}",
        "- Held-out evaluation started: False",
        "",
        "This is an exploratory engineering gate, not a paper result.",
    ]
    (output_root / "internal_validation_report.md").write_text("\n".join(report) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    return analyze(args.spec.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
