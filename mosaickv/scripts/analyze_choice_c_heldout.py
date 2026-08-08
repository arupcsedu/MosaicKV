#!/usr/bin/env python3
"""Analyze the once-only, byte-matched Choice C held-out comparison."""

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
        indices = tuple(generator.randrange(len(mosaic)) for _value in mosaic)
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
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return quantile(0.025), quantile(0.975)


def _bootstrap_score_difference(
    mosaic: tuple[float, ...],
    competitor: tuple[float, ...],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    differences = tuple(
        mosaic_value - competitor_value
        for mosaic_value, competitor_value in zip(mosaic, competitor, strict=True)
    )
    values = sorted(
        statistics.fmean(differences[generator.randrange(len(differences))] for _ in differences)
        for _replicate in range(resamples)
    )
    return values[math.floor(0.025 * (len(values) - 1))], values[
        math.ceil(0.975 * (len(values) - 1))
    ]


def analyze(spec_path: Path) -> int:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    output_root = Path(
        os.environ.get("MOSAICKV_OUTPUT_ROOT_OVERRIDE", str(spec["output_root"]))
    ).resolve()
    methods = tuple(cast("list[str]", spec["methods"]))
    spec_sha = _sha(spec_path)[:10]
    frames: dict[str, pd.DataFrame] = {}
    effective: dict[str, dict[str, str]] = {}
    tier_modes: dict[str, dict[str, str]] = {}
    for method in methods:
        run_id = f"choice-c-heldout-{method}-{spec_sha}"
        path = output_root / run_id / "aggregate.parquet"
        frame = pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True)
        if frame.empty or set(frame["status"]) != {"completed"}:
            raise RuntimeError(f"held-out method is incomplete: {method}")
        if frame["error"].notna().any() or frame["sample_id"].duplicated().any():
            raise RuntimeError(f"held-out method contains failures/duplicates: {method}")
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
            raise RuntimeError(f"held-out method has a missing/nonfinite metric: {method}")
        traces: dict[str, str] = {}
        modes: dict[str, str] = {}
        trace_root = output_root / "traces" / run_id
        for trace_path in sorted(trace_root.glob("*.json")):
            record = json.loads(trace_path.read_text())
            sample_id = str(record["sample_id"])
            traces[sample_id] = str(record.get("effective_method"))
            modes[sample_id] = str(record.get("tier_mode"))
        if set(traces) != set(frame["sample_id"]):
            raise RuntimeError(f"held-out traces are incomplete: {method}")
        effective[method] = traces
        tier_modes[method] = modes
        frames[method] = frame

    expected_ids = tuple(frames[methods[0]]["sample_id"])
    if any(tuple(frame["sample_id"]) != expected_ids for frame in frames.values()):
        raise RuntimeError("held-out methods do not cover identical paired examples")

    combined = pd.concat(
        [
            frame.assign(
                requested_method=method,
                effective_method=frame["sample_id"].map(effective[method]),
                tier_mode=frame["sample_id"].map(tier_modes[method]),
            )
            for method, frame in frames.items()
        ],
        ignore_index=True,
    )
    combined.to_parquet(output_root / "heldout_rows.parquet", index=False)
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
        eligibility[method] = {
            "eligible": maximum <= tolerance,
            "mean_relative_byte_mismatch": float(mismatch.mean()),
            "maximum_relative_byte_mismatch": maximum,
        }
        if maximum <= tolerance:
            eligible.append(method)
    if not eligible:
        raise RuntimeError("no held-out baseline satisfies the measured-byte constraint")
    nearest = min(
        eligible,
        key=lambda method: (float(_fidelity(frames[method]).mean()), method),
    )
    mosaic_error = float(_fidelity(mosaic).mean())
    competitor_error = float(_fidelity(frames[nearest]).mean())
    improvement = (
        (competitor_error - mosaic_error) / competitor_error
        if competitor_error > 0
        else float("-inf")
    )
    ci_low, ci_high = _bootstrap_relative_improvement(
        tuple(float(value) for value in _fidelity(mosaic)),
        tuple(float(value) for value in _fidelity(frames[nearest])),
        resamples=max(2000, int(spec["calibration_rule"]["bootstrap_resamples"])),
        seed=int(spec["calibration_rule"]["bootstrap_seed"]),
    )
    best_task_method = max(
        eligible,
        key=lambda method: (float(frames[method]["task_score"].mean()), method),
    )
    task_ci_low, task_ci_high = _bootstrap_score_difference(
        tuple(float(value) for value in mosaic["task_score"]),
        tuple(float(value) for value in frames[best_task_method]["task_score"]),
        resamples=max(2000, int(spec["calibration_rule"]["bootstrap_resamples"])),
        seed=int(spec["calibration_rule"]["bootstrap_seed"]),
    )
    genuine_three_tier = all(
        effective["mosaickv_full"][sample_id] == "mosaickv_full"
        and tier_modes["mosaickv_full"][sample_id] == "three_tier"
        for sample_id in expected_ids
    )
    required = float(spec["gate"]["required_relative_improvement_over_nearest_competitor"])
    margin = float(spec["gate"]["task_score_noninferiority_margin"])
    fidelity_pass = improvement >= required
    task_pass = task_ci_low >= -margin
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
    pd.DataFrame(summary_rows).to_parquet(output_root / "heldout_summary.parquet", index=False)
    gate = {
        "status": "passed" if passed else "failed",
        "pass": passed,
        "classification": "exploratory_dirty_tree_not_paper_result",
        "sample_count": len(expected_ids),
        "nearest_eligible_competitor": nearest,
        "best_task_score_competitor": best_task_method,
        "byte_eligibility": eligibility,
        "mosaickv_fidelity_error": mosaic_error,
        "nearest_competitor_fidelity_error": competitor_error,
        "relative_fidelity_improvement": improvement,
        "paired_bootstrap_relative_improvement_95_ci": [ci_low, ci_high],
        "required_relative_improvement": required,
        "fidelity_pass": fidelity_pass,
        "mosaickv_task_score": float(mosaic["task_score"].mean()),
        "best_competitor_task_score": float(frames[best_task_method]["task_score"].mean()),
        "paired_bootstrap_task_score_difference_95_ci": [task_ci_low, task_ci_high],
        "task_score_noninferiority_margin": margin,
        "task_noninferiority_pass": task_pass,
        "genuine_three_tier": genuine_three_tier,
        "repair_count_zero_for_repair_disabled_methods": repair_is_zero,
        "full_sweep_allowed": passed,
    }
    (output_root / "heldout_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report = [
        "# Choice C held-out gate",
        "",
        f"- Status: **{gate['status']}**",
        f"- Samples: {len(expected_ids)} once-only held-out examples",
        f"- Nearest byte-matched competitor: `{nearest}`",
        f"- MosaicKV fidelity error: {mosaic_error:.6f}",
        f"- Nearest-competitor fidelity error: {competitor_error:.6f}",
        f"- Relative fidelity improvement: {improvement:.2%}",
        f"- Paired-bootstrap improvement 95% CI: [{ci_low:.2%}, {ci_high:.2%}]",
        f"- Task score (MosaicKV / best comparator): "
        f"{float(mosaic['task_score'].mean()):.6f} / "
        f"{float(frames[best_task_method]['task_score'].mean()):.6f}",
        f"- Paired-bootstrap task-score difference 95% CI: [{task_ci_low:.6f}, {task_ci_high:.6f}]",
        f"- Genuine three-tier rows: {genuine_three_tier}",
        f"- Full sweep allowed: **{passed}**",
        "",
        "This is an exploratory dirty-worktree engineering gate, not a paper result.",
    ]
    (output_root / "heldout_gate_report.md").write_text("\n".join(report) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    return analyze(args.spec.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
