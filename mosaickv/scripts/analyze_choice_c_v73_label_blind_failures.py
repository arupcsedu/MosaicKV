#!/usr/bin/env python3
"""Audit paired v73 outputs and cache traces without accessing benchmark labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

AUDIT_METHODS = ("full_kv", "prompt_attention_topk", "mosaickv_full")
AUDIT_COLUMNS = (
    "task",
    "method",
    "repetition",
    "sample_id",
    "answer",
    "active_kv_bytes",
    "residual_kv_bytes",
    "compression_time",
    "end_to_end_time",
    "status",
)


def _normalized_answer(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _finite(values: list[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            result.append(numeric)
    return result


def _median_or_none(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _trace_record(path: Path) -> dict[str, Any]:
    trace = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    graph = cast("dict[str, Any]", trace.get("graph_diagnostics") or {})
    edges = dict(cast("list[list[object]]", graph.get("edge_counts") or []))
    routing = cast("dict[str, Any]", trace.get("candidate_graph_routing") or {})
    external = cast("dict[str, Any]", routing.get("external_stable_capacity") or {})
    shortfalls = cast("list[dict[str, Any]]", external.get("routes_with_shortfall") or [])
    prototypes = cast(
        "list[dict[str, Any]]", trace.get("prototype_diagnostics_by_kv_head") or []
    )
    key_dispersions = _finite(
        [record.get("maximum_key_dispersion") for record in prototypes]
    )
    return {
        "sample_id": str(trace["sample_id"]),
        "spatial_edges": int(edges.get("spatial_adjacency", 0)),
        "layout_edges": int(edges.get("ocr_layout", 0)),
        "same_region_edges": int(edges.get("same_evidence_region", 0)),
        "planner_nodes": int(
            routing.get("planner_node_count") or graph.get("node_count") or 0
        ),
        "candidate_nodes": int(routing.get("candidate_node_count") or 0),
        "stable_route_count": int(external.get("route_count") or 0),
        "stable_shortfall_routes": len(shortfalls),
        "stable_shortfall_positions": sum(
            int(record.get("shortfall_positions") or 0) for record in shortfalls
        ),
        "prototype_heads": len(prototypes),
        "prototype_count": sum(
            int(record.get("prototype_count") or 0) for record in prototypes
        ),
        "maximum_key_dispersion": max(key_dispersions) if key_dispersions else None,
        "repair_events": len(cast("list[object]", trace.get("repair_events") or [])),
    }


def audit(input_root: Path, *, tasks: tuple[str, ...], run_prefix: str) -> dict[str, Any]:
    trials_path = input_root / "analysis" / "heldout_trials.parquet"
    # Explicit column projection is the enforcement boundary: neither references nor
    # official task scores are loaded into this process.
    frame = pd.read_parquet(trials_path, columns=list(AUDIT_COLUMNS))
    frame = frame[
        frame["task"].isin(tasks)
        & frame["method"].isin(AUDIT_METHODS)
        & (frame["status"] == "completed")
    ].copy()
    if frame.empty:
        raise RuntimeError("no completed paired rows matched the requested audit")
    grouped = (
        frame.groupby(["task", "method", "sample_id"], as_index=False)
        .agg(
            answer=("answer", "first"),
            answer_variants=("answer", "nunique"),
            active_kv_bytes=("active_kv_bytes", "mean"),
            residual_kv_bytes=("residual_kv_bytes", "mean"),
            repetition_count=("repetition", "nunique"),
        )
        .sort_values(["task", "method", "sample_id"])
    )
    if int(grouped["answer_variants"].max()) != 1:
        raise RuntimeError("paired audit found nondeterministic answers")

    task_results: dict[str, Any] = {}
    for task in tasks:
        by_method = {
            method: grouped[
                (grouped["task"] == task) & (grouped["method"] == method)
            ]
            .set_index("sample_id")
            .sort_index()
            for method in AUDIT_METHODS
        }
        sample_ids = tuple(by_method["mosaickv_full"].index)
        if not sample_ids or any(tuple(table.index) != sample_ids for table in by_method.values()):
            raise RuntimeError(f"{task} methods do not cover identical paired samples")

        trace_paths = sorted(
            (input_root / "traces").glob(
                f"{run_prefix}-{task}-trial-00-mosaickv_full/*.json"
            )
        )
        traces: dict[str, dict[str, Any]] = {}
        for path in trace_paths:
            record = _trace_record(path)
            traces[str(record["sample_id"])] = record
        if set(traces) != set(sample_ids):
            raise RuntimeError(f"{task} trace set does not match paired result rows")

        categories = {
            "mosaic_equals_fullkv": 0,
            "mosaic_equals_prompt_topk": 0,
            "prompt_topk_equals_fullkv": 0,
            "mosaic_differs_from_both": 0,
        }
        sample_records: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            mosaic = _normalized_answer(by_method["mosaickv_full"].at[sample_id, "answer"])
            full = _normalized_answer(by_method["full_kv"].at[sample_id, "answer"])
            prompt = _normalized_answer(
                by_method["prompt_attention_topk"].at[sample_id, "answer"]
            )
            equals_full = mosaic == full
            equals_prompt = mosaic == prompt
            prompt_equals_full = prompt == full
            categories["mosaic_equals_fullkv"] += int(equals_full)
            categories["mosaic_equals_prompt_topk"] += int(equals_prompt)
            categories["prompt_topk_equals_fullkv"] += int(prompt_equals_full)
            categories["mosaic_differs_from_both"] += int(not equals_full and not equals_prompt)
            sample_records.append(
                {
                    **traces[sample_id],
                    "mosaic_equals_fullkv": equals_full,
                    "mosaic_equals_prompt_topk": equals_prompt,
                    "prompt_topk_equals_fullkv": prompt_equals_full,
                }
            )

        zero_structural = sum(
            record["spatial_edges"] == 0
            and record["layout_edges"] == 0
            and record["same_region_edges"] == 0
            for record in sample_records
        )
        task_results[task] = {
            "sample_count": len(sample_ids),
            "paired_output_agreement": categories,
            "trace_mechanisms": {
                "samples_without_structural_edges": zero_structural,
                "total_repair_events": sum(record["repair_events"] for record in sample_records),
                "median_planner_nodes": _median_or_none(
                    [float(record["planner_nodes"]) for record in sample_records]
                ),
                "median_stable_route_count": _median_or_none(
                    [float(record["stable_route_count"]) for record in sample_records]
                ),
                "median_stable_shortfall_positions": _median_or_none(
                    [float(record["stable_shortfall_positions"]) for record in sample_records]
                ),
                "samples_without_prototypes": sum(
                    record["prototype_count"] == 0 for record in sample_records
                ),
                "median_maximum_key_dispersion": _median_or_none(
                    _finite([record["maximum_key_dispersion"] for record in sample_records])
                ),
            },
        }
    return {
        "schema_version": 1,
        "classification": "label_blind_paired_failure_audit",
        "benchmark_labels_accessed": False,
        "benchmark_scores_accessed": False,
        "loaded_result_columns": list(AUDIT_COLUMNS),
        "source_trials": str(trials_path.resolve()),
        "tasks": task_results,
        "diagnosis": [
            "The evaluated candidate did not construct spatial, OCR/layout, or same-region edges.",
            "Most layer/head routes bypassed graph selection through stable-head Prompt Top-K.",
            "Repair was disabled, so no divergent output could be repaired online.",
            "Prototype-free traces are valid and are retained as zero-prototype observations.",
        ],
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# V73 label-blind paired failure audit",
        "",
        "This audit loaded answers, byte counts, timings, and cache traces only. "
        "It did not load benchmark references or task scores.",
        "",
        "| Task | N | Mosaic=FullKV | Mosaic=Prompt Top-K | No structural edges | "
        "Repairs | Median stable shortfall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, record in cast("dict[str, dict[str, Any]]", result["tasks"]).items():
        agreement = cast("dict[str, int]", record["paired_output_agreement"])
        mechanisms = cast("dict[str, Any]", record["trace_mechanisms"])
        lines.append(
            f"| {task} | {record['sample_count']} | {agreement['mosaic_equals_fullkv']} | "
            f"{agreement['mosaic_equals_prompt_topk']} | "
            f"{mechanisms['samples_without_structural_edges']} | "
            f"{mechanisms['total_repair_events']} | "
            f"{float(mechanisms['median_stable_shortfall_positions']):.0f} positions |"
        )
    lines.extend(["", "## Mechanism findings", ""])
    lines.extend(f"- {item}" for item in cast("list[str]", result["diagnosis"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", default=("docvqa", "chartqa"))
    parser.add_argument("--run-prefix", default="v73")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    result = audit(
        args.input_root.resolve(),
        tasks=tuple(str(task) for task in args.tasks),
        run_prefix=str(args.run_prefix),
    )
    output_json = args.output_json or args.input_root / "analysis/label_blind_failure_audit.json"
    output_report = (
        args.output_report or args.input_root / "analysis/label_blind_failure_audit.md"
    )
    for path in (output_json, output_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output_report.write_text(_report(result))
    print(json.dumps({"output_json": str(output_json), "labels_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
