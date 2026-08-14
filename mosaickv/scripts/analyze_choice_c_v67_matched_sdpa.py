#!/usr/bin/env python3
"""Analyze the paired Prompt Top-K versus MosaicKV quality-memory control."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

METHODS = ("prompt_attention_topk", "mosaickv_full")


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def analyze(output_root: Path) -> dict[str, Any]:
    telemetry_paths = tuple(sorted((output_root / "telemetry").glob("*.json")))
    if len(telemetry_paths) != 1:
        raise RuntimeError(f"expected one v67 telemetry file, found {len(telemetry_paths)}")
    telemetry = cast("dict[str, Any]", json.loads(telemetry_paths[0].read_text()))
    if telemetry.get("status") != "completed":
        raise RuntimeError(f"v67 telemetry is not complete: {telemetry.get('status')}")
    repetitions = int(telemetry["measured_repetitions_per_method"])
    sample_ids = tuple(cast("list[str]", telemetry["sample_ids"]))
    trials = cast("list[dict[str, Any]]", telemetry["trials"])
    expected_trials = repetitions * len(METHODS)
    if len(trials) != expected_trials:
        raise RuntimeError(f"expected {expected_trials} measured trials, found {len(trials)}")

    rows_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    order_balance: defaultdict[str, list[int]] = defaultdict(list)
    for trial in trials:
        method = str(trial["method"])
        repetition = int(trial["repetition"])
        if method not in METHODS or not 0 <= repetition < repetitions:
            raise RuntimeError("v67 trial contains an unknown method or repetition")
        order_balance[method].append(int(trial["order"]))
        rows = cast("list[dict[str, Any]]", trial["rows"])
        if {str(row["sample_id"]) for row in rows} != set(sample_ids):
            raise RuntimeError("v67 trial does not contain the fixed paired samples")
        for row in rows:
            key = repetition, method, str(row["sample_id"])
            if key in rows_by_key:
                raise RuntimeError(f"duplicate v67 measurement row: {key}")
            rows_by_key[key] = row

    metrics = (
        "active_kv_bytes",
        "residual_kv_bytes",
        "compression_time",
        "decode_time",
        "end_to_end_time",
        "task_score",
    )
    summary: dict[str, Any] = {}
    per_repetition: dict[str, list[float]] = {method: [] for method in METHODS}
    for method in METHODS:
        method_rows = [
            rows_by_key[(repetition, method, sample_id)]
            for repetition in range(repetitions)
            for sample_id in sample_ids
        ]
        method_summary: dict[str, Any] = {"row_count": len(method_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in method_rows]
            method_summary[metric] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": _percentile(values, 95),
            }
        method_summary["answers_by_sample"] = {
            sample_id: sorted(
                {
                    str(rows_by_key[(repetition, method, sample_id)]["answer"])
                    for repetition in range(repetitions)
                }
            )
            for sample_id in sample_ids
        }
        method_summary["order_positions"] = order_balance[method]
        summary[method] = method_summary
        per_repetition[method] = [
            statistics.mean(
                float(rows_by_key[(repetition, method, sample_id)]["end_to_end_time"])
                for sample_id in sample_ids
            )
            for repetition in range(repetitions)
        ]

    prompt = cast("dict[str, Any]", summary["prompt_attention_topk"])
    mosaic = cast("dict[str, Any]", summary["mosaickv_full"])
    paired_e2e_ratios = [
        mosaic_value / prompt_value
        for mosaic_value, prompt_value in zip(
            per_repetition["mosaickv_full"],
            per_repetition["prompt_attention_topk"],
            strict=True,
        )
    ]
    prompt_bytes = float(prompt["active_kv_bytes"]["mean"])
    mosaic_bytes = float(mosaic["active_kv_bytes"]["mean"])
    prompt_score = float(prompt["task_score"]["mean"])
    mosaic_score = float(mosaic["task_score"]["mean"])
    median_ratio = statistics.median(paired_e2e_ratios)
    gates = {
        "quality_noninferior": mosaic_score >= prompt_score,
        "active_memory_reduction_fraction": 1.0 - mosaic_bytes / prompt_bytes,
        "active_memory_at_least_30_percent_smaller": mosaic_bytes <= 0.70 * prompt_bytes,
    }
    gates["quality_memory_gate_pass"] = bool(
        gates["quality_noninferior"]
        and gates["active_memory_at_least_30_percent_smaller"]
    )
    return {
        "schema_version": 2,
        "classification": telemetry["classification"],
        "source_snapshot": telemetry["source_snapshot"],
        "sample_ids": list(sample_ids),
        "warmups_per_method": telemetry["warmups_per_method"],
        "measured_repetitions_per_method": repetitions,
        "summary": summary,
        "gates": gates,
        "latency_report": {
            "policy": "informational_not_a_rejection_criterion",
            "paired_end_to_end_ratios": paired_e2e_ratios,
            "paired_median_end_to_end_improvement_fraction": 1.0 - median_ratio,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    result = analyze(output_root)
    destination = output_root / "analysis" / "matched_quality_memory_gate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["gates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
