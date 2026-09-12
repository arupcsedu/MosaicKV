#!/usr/bin/env python3
"""Aggregate the measured PrefixKV and VL-Cache fidelity diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

CLASSIFICATION = "canonical_published_baseline_fidelity_diagnostic"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(
    values: np.ndarray[Any, np.dtype[np.float64]], *, seed: int, resamples: int
) -> tuple[float, float]:
    if not len(values):
        raise ValueError("cannot bootstrap an empty paired sample")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = values[draws].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def analyze(input_root: Path, output_root: Path) -> dict[str, Any]:
    completions = sorted(input_root.glob("*/*/*/job_complete.json"))
    if len(completions) != 32:
        raise RuntimeError(f"expected 32 completed shards, found {len(completions)}")
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    source_shas: set[str] = set()
    protocol_shas: set[str] = set()
    for path in completions:
        payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        if payload.get("status") != "completed" or payload.get("classification") != CLASSIFICATION:
            raise RuntimeError(f"invalid completion: {path}")
        source_shas.add(str(payload["source_git_sha"]))
        protocol_shas.add(str(payload["protocol_sha256"]))
        family = str(payload["family"])
        task = str(payload["task"])
        model_id = str(payload["model_id"])
        for record in cast("list[dict[str, Any]]", payload["records"]):
            ratio = float(record.get("retention_ratio", 1.0))
            for row in cast("list[dict[str, Any]]", record["rows"]):
                rows.append(
                    {
                        **row,
                        "family": family,
                        "task": task,
                        "model_id": model_id,
                        "method": str(record["method"]),
                        "retention_ratio": ratio,
                        "profile_forget_ratio": record.get("profile_forget_ratio"),
                        "source_git_sha": payload["source_git_sha"],
                        "protocol_sha256": payload["protocol_sha256"],
                        "job_spec_sha256": payload["job_spec_sha256"],
                    }
                )
        checks.extend(cast("list[dict[str, Any]]", payload["byte_match_checks"]))
    if len(source_shas) != 1 or len(protocol_shas) != 1:
        raise RuntimeError("campaign mixed source or protocol identities")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["family", "task", "method", "retention_ratio", "sample_id"]).any():
        raise RuntimeError("duplicate measured rows")
    if (
        frame.isna()[["task_score", "active_kv_bytes", "decode_time", "end_to_end_time"]]
        .any()
        .any()
    ):
        raise RuntimeError("measured rows contain missing primary values")
    expected_per_group = 64
    counts = frame.groupby(["family", "task", "method", "retention_ratio"]).size()
    if set(counts) != {expected_per_group}:
        raise RuntimeError(f"incomplete measured groups: {counts.to_dict()}")
    check_frame = pd.DataFrame(checks)
    if not bool(check_frame["exact_byte_match"].all()):
        raise RuntimeError("paired physical active-byte matching failed")

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for (family, task, method, ratio), group in frame.groupby(
        ["family", "task", "method", "retention_ratio"], sort=True
    ):
        summary_rows.append(
            {
                "family": family,
                "task": task,
                "model_id": str(group["model_id"].iloc[0]),
                "method": method,
                "retention_ratio": ratio,
                "samples": len(group),
                "task_score_percent": 100 * float(group["task_score"].mean()),
                "active_kv_mib": float(group["active_kv_bytes"].mean()) / 2**20,
                "decode_seconds": float(group["decode_time"].mean()),
                "end_to_end_seconds": float(group["end_to_end_time"].mean()),
            }
        )
        if method == "full_kv":
            continue
        full = frame[
            frame["family"].eq(family) & frame["task"].eq(task) & frame["method"].eq("full_kv")
        ][["sample_id", "task_score", "answer", "active_kv_bytes"]].rename(
            columns={
                "task_score": "full_score",
                "answer": "full_answer",
                "active_kv_bytes": "full_bytes",
            }
        )
        paired = group.merge(full, on="sample_id", validate="one_to_one")
        differences = (paired["task_score"] - paired["full_score"]).to_numpy(dtype=float)
        low, high = _bootstrap(
            differences,
            seed=20271103 + round(float(ratio) * 1000),
            resamples=10000,
        )
        comparison_rows.append(
            {
                "family": family,
                "task": task,
                "model_id": str(group["model_id"].iloc[0]),
                "method": method,
                "retention_ratio": ratio,
                "samples": len(paired),
                "score_percent": 100 * float(paired["task_score"].mean()),
                "full_score_percent": 100 * float(paired["full_score"].mean()),
                "score_delta_points": 100 * float(differences.mean()),
                "score_delta_ci_low_points": 100 * low,
                "score_delta_ci_high_points": 100 * high,
                "answer_agreement_percent": 100
                * float((paired["answer"] == paired["full_answer"]).mean()),
                "active_kv_mib": float(paired["active_kv_bytes"].mean()) / 2**20,
                "active_reduction_from_full_percent": 100
                * float((1 - paired["active_kv_bytes"] / paired["full_bytes"]).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_root / "raw_rows.parquet", index=False)
    summary.to_parquet(output_root / "summary.parquet", index=False)
    comparisons.to_parquet(output_root / "paired_fullkv_comparisons.parquet", index=False)
    check_frame.to_parquet(output_root / "byte_match_validation.parquet", index=False)

    method_rows = comparisons[comparisons["method"].isin(["prefixkv_reimpl", "vl_cache_reimpl"])]
    lines = [
        "# Published-baseline fidelity diagnostic",
        "",
        "These are local reimplementations, not official-code results. PrefixKV uses the "
        "pinned official layer-profile shape under the common hard budget. VL-Cache follows "
        "the paper equations on the paper's LLaVA-v1.6-Mistral family. All comparisons are "
        "paired against FullKV and Prompt Top-K with exact physical active-byte matching.",
        "",
        "| Family | Task | Method | Retention | Score (%) | Delta vs Full (points, 95% CI) | "
        "Agreement (%) | Active KV (MiB) | Reduction (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows.sort_values(["family", "task", "retention_ratio"]).itertuples():
        lines.append(
            f"| {row.family} | {row.task} | `{row.method}` | {row.retention_ratio:.2f} | "
            f"{row.score_percent:.2f} | {row.score_delta_points:+.2f} "
            f"[{row.score_delta_ci_low_points:+.2f}, {row.score_delta_ci_high_points:+.2f}] | "
            f"{row.answer_agreement_percent:.2f} | {row.active_kv_mib:.2f} | "
            f"{row.active_reduction_from_full_percent:.2f} |"
        )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "classification": CLASSIFICATION,
        "source_git_sha": next(iter(source_shas)),
        "protocol_sha256": next(iter(protocol_shas)),
        "completion_count": len(completions),
        "raw_rows": len(frame),
        "raw_rows_sha256": _sha(output_root / "raw_rows.parquet"),
        "summary_sha256": _sha(output_root / "summary.parquet"),
        "comparisons_sha256": _sha(output_root / "paired_fullkv_comparisons.parquet"),
        "byte_match_sha256": _sha(output_root / "byte_match_validation.parquet"),
        "report_sha256": _sha(output_root / "report.md"),
    }
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input_root.resolve(), args.output_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
