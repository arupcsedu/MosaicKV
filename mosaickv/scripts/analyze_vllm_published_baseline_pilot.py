#!/usr/bin/env python3
"""Analyze the paired vLLM published-baseline development pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd

TASKS = ("mmstar", "textvqa", "docvqa")
METHODS = (
    "full_kv",
    "prompt_attention_topk",
    "lookm_reimpl",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "mosaickv_exact",
)


def _rows(root: Path, task: str, method: str) -> list[dict[str, Any]]:
    paths = sorted((root / task / method / "vllm").glob("*/results.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {task}/{method} result file, found {paths}")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    if len(rows) != 2:
        raise RuntimeError(f"expected two paired rows for {task}/{method}, found {len(rows)}")
    if any(row.get("status") != "completed" for row in rows):
        raise RuntimeError(f"{task}/{method} contains a failed sample")
    return cast("list[dict[str, Any]]", rows)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def analyze(root: Path) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    by_task_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in TASKS:
        for method in METHODS:
            rows = _rows(root, task, method)
            by_task_method[(task, method)] = rows
            for row in rows:
                all_rows.append({"task": task, "method": method, **row})
            summaries.append(
                {
                    "task": task,
                    "method": method,
                    "sample_count": len(rows),
                    "task_score": _mean([float(row["task_score"]) for row in rows]),
                    "active_kv_bytes": _mean([float(row["active_kv_bytes"]) for row in rows]),
                    "active_kv_mib": _mean([float(row["active_kv_bytes"]) for row in rows])
                    / 2**20,
                    "end_to_end_seconds": _mean(
                        [float(row["end_to_end_time"]) for row in rows]
                    ),
                    "effective_methods": sorted(
                        {str(row["method"]) for row in rows}
                    ),
                    "failed_samples": 0,
                }
            )
        mosaic = {row["sample_id"]: row for row in by_task_method[(task, "mosaickv_exact")]}
        for competitor in METHODS[1:-1]:
            other = {row["sample_id"]: row for row in by_task_method[(task, competitor)]}
            shared = sorted(set(mosaic) & set(other))
            if len(shared) != 2:
                raise RuntimeError(f"{task}/{competitor} is not sample-paired")
            mosaic_bytes = _mean([float(mosaic[key]["active_kv_bytes"]) for key in shared])
            other_bytes = _mean([float(other[key]["active_kv_bytes"]) for key in shared])
            score_delta = _mean(
                [
                    float(mosaic[key]["task_score"]) - float(other[key]["task_score"])
                    for key in shared
                ]
            )
            comparisons.append(
                {
                    "task": task,
                    "competitor": competitor,
                    "sample_count": len(shared),
                    "mosaickv_memory_reduction_fraction": 1.0 - mosaic_bytes / other_bytes,
                    "mosaickv_task_score_delta": score_delta,
                    "answer_agreement_fraction": _mean(
                        [float(mosaic[key]["answer"] == other[key]["answer"]) for key in shared]
                    ),
                    "memory_gate_30pct": 1.0 - mosaic_bytes / other_bytes >= 0.30 - 1e-12,
                    "quality_noninferiority_gate": score_delta >= -1e-12,
                }
            )
    summary_df = pd.DataFrame(summaries)
    comparison_df = pd.DataFrame(comparisons)
    analysis_root = root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_parquet(analysis_root / "raw_rows.parquet", index=False)
    summary_df.to_parquet(analysis_root / "summary.parquet", index=False)
    comparison_df.to_parquet(analysis_root / "comparisons.parquet", index=False)
    prompt = comparison_df[comparison_df["competitor"] == "prompt_attention_topk"]
    gate = bool(prompt["memory_gate_30pct"].all() and prompt["quality_noninferiority_gate"].all())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "classification": "development_pilot_non_paper",
        "model": "llava-hf/llava-1.5-7b-hf",
        "backend": "vllm-0.10.2-requestwide-projection",
        "tasks": list(TASKS),
        "samples_per_task": 2,
        "gate_definition": (
            "MosaicKV active KV bytes at least 30% below Prompt Top-K and paired mean task "
            "score no lower on every task"
        ),
        "gate_passed": gate,
        "summary": summaries,
        "comparisons": comparisons,
        "semantic_limit": (
            "LOOK-M, PrefixKV, and VL-Cache rows are request-wide whole-block projections "
            "of reimplementation selector decisions, not official/native paper layouts"
        ),
    }
    (analysis_root / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# vLLM published-baseline development pilot",
        "",
        f"Overall Prompt Top-K gate: **{'PASS' if gate else 'FAIL'}**.",
        "",
        "This is a two-sample-per-task development pilot, not paper-level evidence.",
        "LOOK-M/PrefixKV/VL-Cache are explicitly labeled request-wide projections.",
        "",
        "## Per-task method results",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## MosaicKV paired comparisons",
        "",
        comparison_df.to_markdown(index=False),
        "",
    ]
    (analysis_root / "report.md").write_text("\n".join(lines))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
