#!/usr/bin/env python3
"""Fail closed on a unified, exact-physical-byte comparison artifact.

This validator deliberately checks the per-sample accounting boundary instead
of nominal retention ratios.  FullKV is allowed to exceed a compressed cap;
the three compressed arms must each fit, and must have identical measured
active-KV bytes for every paired task/sample/cap point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

COMPRESSED_METHODS = (
    "prompt_attention_topk",
    "vl_cache_reimpl",
    "mosaickv_exact",
)
REQUIRED_RAW_COLUMNS = {
    "task",
    "sample_id",
    "method",
    "configured_byte_cap",
    "active_kv_bytes",
    "status",
    "error",
    "source_git_sha",
    "model_revision",
    "result_category",
    "result_origin",
}


def _require_columns(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def validate_artifact(
    root: Path,
    *,
    expected_source_sha: str | None = None,
    expected_model_revision: str | None = None,
) -> dict[str, Any]:
    """Validate and summarize one immutable shared-byte curve artifact."""

    raw_path = root / "raw_results.parquet"
    byte_path = root / "byte_validation.parquet"
    summary_path = root / "summary.parquet"
    paired_path = root / "paired_comparisons.parquet"
    for path in (raw_path, byte_path, summary_path, paired_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = pd.read_parquet(raw_path)
    byte = pd.read_parquet(byte_path)
    summary = pd.read_parquet(summary_path)
    paired = pd.read_parquet(paired_path)
    _require_columns(raw, REQUIRED_RAW_COLUMNS, "raw_results")
    _require_columns(
        byte,
        {
            "task",
            "sample_id",
            "configured_byte_cap",
            "prompt_attention_topk_bytes",
            "vl_cache_reimpl_bytes",
            "mosaickv_exact_bytes",
            "exact_byte_match",
            "within_hard_cap",
        },
        "byte_validation",
    )
    _require_columns(summary, {"task", "metric_name", "method", "sample_count"}, "summary")
    _require_columns(
        paired,
        {
            "task",
            "metric_name",
            "reference_method",
            "candidate_method",
            "sample_count",
            "bootstrap_resamples",
        },
        "paired_comparisons",
    )
    if raw.empty:
        raise ValueError("raw_results is empty")
    if set(raw["status"].astype(str)) != {"completed"}:
        raise ValueError("raw_results includes non-completed rows")
    if raw["error"].notna().any():
        raise ValueError("raw_results includes an error payload")
    if raw.duplicated(["task", "sample_id", "method", "configured_byte_cap"]).any():
        raise ValueError("raw_results has duplicate paired method rows")
    if expected_source_sha is not None and set(raw["source_git_sha"].astype(str)) != {
        expected_source_sha
    }:
        raise ValueError("source SHA differs from the immutable experiment revision")
    if expected_model_revision is not None and set(raw["model_revision"].astype(str)) != {
        expected_model_revision
    }:
        raise ValueError("model revision differs from the registered experiment revision")
    if set(raw["result_category"].astype(str)) != {"unified_comparison"}:
        raise ValueError("artifact does not consistently identify itself as a unified comparison")
    if set(raw["result_origin"].astype(str)) != {"locally_measured"}:
        raise ValueError("artifact contains non-local or unlabeled result rows")

    compressed = raw[raw["method"].isin(COMPRESSED_METHODS)].copy()
    if set(compressed["method"].astype(str)) != set(COMPRESSED_METHODS):
        raise ValueError("one or more compressed comparison arms are absent")
    if (compressed["active_kv_bytes"] > compressed["configured_byte_cap"]).any():
        raise ValueError("a compressed row exceeds its declared hard byte cap")
    counts = (
        compressed.groupby(["task", "sample_id", "configured_byte_cap"], sort=False)["method"]
        .nunique()
        .rename("method_count")
    )
    if not (counts == len(COMPRESSED_METHODS)).all():
        raise ValueError("compressed methods do not share identical task/sample/cap identities")
    byte_keys = ["task", "sample_id", "configured_byte_cap"]
    if byte.duplicated(byte_keys).any():
        raise ValueError("byte_validation has duplicate sample checks")
    if not byte["exact_byte_match"].astype(bool).all():
        raise ValueError("at least one sample does not have exact physical-byte matching")
    if not byte["within_hard_cap"].astype(bool).all():
        raise ValueError("at least one exact-byte comparison exceeds its hard cap")
    expected_byte_rows = compressed[byte_keys].drop_duplicates().itertuples(index=False, name=None)
    expected_byte_keys = set(tuple(value) for value in expected_byte_rows)
    observed_byte_rows = byte[byte_keys].itertuples(index=False, name=None)
    observed_byte_keys = set(tuple(value) for value in observed_byte_rows)
    if observed_byte_keys != expected_byte_keys:
        raise ValueError("byte_validation identities differ from the compressed raw rows")

    # Benchmark metrics remain per-task. A mixed metric in one task table is
    # allowed only if it is a named systems measure, never a macro task score.
    task_metrics = {
        str(task): sorted(set(group["metric_name"].astype(str)))
        for task, group in summary.groupby("task", sort=True)
    }
    if len(task_metrics) < 1:
        raise ValueError("summary lacks task-level metrics")
    if (paired["sample_count"] <= 0).any() or (paired["bootstrap_resamples"] < 1).any():
        raise ValueError("paired analysis has an invalid denominator or bootstrap count")

    return {
        "status": "passed",
        "artifact_root": str(root),
        "raw_rows": len(raw),
        "byte_validated_samples": len(byte),
        "tasks": sorted(set(raw["task"].astype(str))),
        "caps_bytes": sorted(int(value) for value in set(compressed["configured_byte_cap"])),
        "task_metrics": task_metrics,
        "compressed_methods": list(COMPRESSED_METHODS),
        "source_git_sha": sorted(set(raw["source_git_sha"].astype(str))),
        "model_revision": sorted(set(raw["model_revision"].astype(str))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--expected-model-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_artifact(
        args.artifact_root,
        expected_source_sha=args.expected_source_sha,
        expected_model_revision=args.expected_model_revision,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        if args.output.exists() and args.output.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(
                f"refusing to overwrite a differing validation result: {args.output}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.output.exists():
            args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
