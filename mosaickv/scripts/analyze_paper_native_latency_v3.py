#!/usr/bin/env python3
"""Analyze the persistent paired PrefixKV-versus-MosaicKV latency trial."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from mosaickv.evaluation.metrics.storage import load_metric_jsonl
from mosaickv.evaluation.metrics.types import MetricScope
from mosaickv.evaluation.paper_native_latency import (
    canonical_inference_traces,
    load_paper_native_latency_protocol,
    paired_method_order,
)


def _metric(path: Path, name: str, scope: MetricScope) -> float:
    values = [
        row.metric_value
        for row in load_metric_jsonl(path)
        if row.metric_name == name and row.metric_scope is scope
    ]
    if not values:
        raise RuntimeError(f"metric {name}/{scope.value} is absent from {path}")
    return math.fsum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _bootstrap_ratio_interval(
    numerator: list[float],
    denominator: list[float],
    *,
    resamples: int,
) -> tuple[float, float]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap requires equally sized non-empty inputs")
    rng = np.random.default_rng(0)
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    indices = rng.integers(0, len(top), size=(resamples, len(top)))
    ratios = top[indices].mean(axis=1) / bottom[indices].mean(axis=1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def _trial_rows(record: dict[str, Any], expected_samples: int) -> pd.DataFrame:
    path = Path(str(record["result_path"])).resolve(strict=True)
    frame = pd.read_parquet(path)
    timing = ("ttft", "prefill_time", "compression_time", "decode_time", "end_to_end_time")
    if (
        len(frame) != expected_samples
        or set(frame["status"].astype(str)) != {"completed"}
        or frame[list(timing)].isna().any(axis=None)
        or not np.isfinite(frame[list(timing)].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError(f"trial contains incomplete or nonfinite rows: {path}")
    if sorted(frame["sample_id"].astype(str)) != sorted(record["sample_ids"]):
        raise RuntimeError(f"trial sample IDs differ from telemetry: {path}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_paper_native_latency_protocol(args.protocol)
    root = args.output_root.resolve(strict=True)
    telemetry = cast(
        "dict[str, Any]", json.loads((root / "telemetry.json").read_text(encoding="utf-8"))
    )
    if telemetry.get("status") != "completed":
        raise RuntimeError(f"latency telemetry is not complete: {telemetry.get('status')}")
    if (
        telemetry.get("experiment_id") != protocol.experiment_id
        or telemetry.get("source_git_dirty") is not False
        or telemetry.get("persistent_model_process") is not True
        or int(telemetry.get("warmups_per_method", -1)) != protocol.timing.warmups_per_method
        or int(telemetry.get("timed_repetitions_per_method", -1))
        != protocol.timing.timed_repetitions_per_method
    ):
        raise RuntimeError("latency telemetry differs from the registered protocol")

    warmups = cast("list[dict[str, Any]]", telemetry.get("warmups"))
    trials = cast("list[dict[str, Any]]", telemetry.get("trials"))
    if len(warmups) != protocol.timing.warmups_per_method * 2:
        raise RuntimeError("latency warmup coverage is incomplete")
    if len(trials) != protocol.timing.timed_repetitions_per_method * 2:
        raise RuntimeError("latency measured-trial coverage is incomplete")

    rows: list[pd.DataFrame] = []
    quality_by_method: dict[str, list[tuple[float, float]]] = {
        method: [] for method in protocol.timing.methods
    }
    answers_by_method: dict[str, dict[str, set[str]]] = {
        method: {} for method in protocol.timing.methods
    }
    padding_clean = True
    planner_bounded = True
    uniform_capacity = True
    for record in trials:
        repetition = int(record["repetition"])
        order = int(record["order"])
        method = str(record["method"])
        expected_order = paired_method_order(protocol.timing, repetition)
        if not 0 <= repetition < protocol.timing.timed_repetitions_per_method:
            raise RuntimeError("trial repetition lies outside the protocol")
        if order not in (0, 1) or method != expected_order[order]:
            raise RuntimeError("trial method order differs from the alternating protocol")
        frame = _trial_rows(record, protocol.timing.sample_count).copy()
        frame["repetition"] = repetition
        frame["order"] = order
        rows.append(frame)
        for row in frame.itertuples():
            answers_by_method[method].setdefault(str(row.sample_id), set()).add(str(row.answer))
        metric_path = Path(str(record["metric_path"])).resolve(strict=True)
        quality_by_method[method].append(
            (
                _metric(metric_path, "rouge_l_f1", MetricScope.SAMPLE),
                _metric(metric_path, "teacher_forced_ppl", MetricScope.CORPUS),
            )
        )
        traces = canonical_inference_traces(
            tuple(sorted(Path(str(record["trace_root"])).resolve(strict=True).glob("*.json")))
        )
        if len(traces) != protocol.timing.sample_count:
            raise RuntimeError("trial does not have one canonical inference trace per sample")
        for trace in traces:
            padding_clean = padding_clean and int(trace["packed_padding_kv_bytes"]) == 0
            lengths = tuple(int(value) for value in trace["tier_active_cache_lengths"])
            uniform_capacity = uniform_capacity and len(set(lengths)) == 1
            if method == "mosaickv_exact":
                routing = cast("dict[str, Any]", trace["candidate_graph_routing"])
                planner_bounded = planner_bounded and (
                    int(routing["planner_node_count"]) / int(routing["source_node_count"]) <= 0.025
                )

    combined = pd.concat(rows, ignore_index=True)
    if combined.duplicated(["repetition", "method", "sample_id"]).any():
        raise RuntimeError("latency trials contain duplicate paired observations")
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite latency report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    trial_table = destination.with_name("trials.parquet")
    if trial_table.exists():
        raise FileExistsError(f"refusing to overwrite latency trials: {trial_table}")
    combined.to_parquet(trial_table, index=False)

    method_summary: dict[str, Any] = {}
    repetition_means: dict[str, list[float]] = {}
    for method in protocol.timing.methods:
        selected = combined.loc[combined["method"] == method]
        end_to_end = selected["end_to_end_time"].astype(float).tolist()
        repetition_means[method] = [
            float(selected.loc[selected["repetition"] == repetition, "end_to_end_time"].mean())
            for repetition in range(protocol.timing.timed_repetitions_per_method)
        ]
        quality = quality_by_method[method]
        method_summary[method] = {
            "row_count": len(selected),
            "mean_active_kv_bytes": float(selected["active_kv_bytes"].mean()),
            "end_to_end_seconds": {
                "mean": statistics.mean(end_to_end),
                "p50": statistics.median(end_to_end),
                "p95": _percentile(end_to_end, 95),
            },
            "mean_rouge_l_f1": statistics.mean(value[0] for value in quality),
            "mean_teacher_forced_ppl": statistics.mean(value[1] for value in quality),
            "deterministic_answers": all(
                len(values) == 1 for values in answers_by_method[method].values()
            ),
        }

    prefix = cast("dict[str, Any]", method_summary["prefixkv_reimpl"])
    mosaic = cast("dict[str, Any]", method_summary["mosaickv_exact"])
    paired_ratios = [
        mosaic_value / prefix_value
        for mosaic_value, prefix_value in zip(
            repetition_means["mosaickv_exact"],
            repetition_means["prefixkv_reimpl"],
            strict=True,
        )
    ]
    primary_ratio = statistics.median(paired_ratios)
    interval = _bootstrap_ratio_interval(
        repetition_means["mosaickv_exact"],
        repetition_means["prefixkv_reimpl"],
        resamples=protocol.timing.bootstrap_resamples,
    )
    memory_reduction = 1.0 - float(mosaic["mean_active_kv_bytes"]) / float(
        prefix["mean_active_kv_bytes"]
    )
    gates = {
        "active_memory_at_least_30_percent_smaller": memory_reduction >= 0.30,
        "registered_40_875_mib_cache_unchanged": abs(
            float(mosaic["mean_active_kv_bytes"]) - protocol.expected_mosaickv_mean_active_kv_bytes
        )
        <= protocol.expected_mean_active_kv_byte_tolerance,
        "rouge_l_noninferior": float(mosaic["mean_rouge_l_f1"]) >= float(prefix["mean_rouge_l_f1"]),
        "ppl_noninferior": float(mosaic["mean_teacher_forced_ppl"])
        <= float(prefix["mean_teacher_forced_ppl"]),
        "latency_within_5_percent": primary_ratio <= protocol.timing.maximum_latency_ratio,
        "deterministic_answers": all(
            bool(summary["deterministic_answers"]) for summary in method_summary.values()
        ),
        "zero_physical_padding": padding_clean,
        "uniform_physical_capacity": uniform_capacity,
        "bounded_shortlist_planner": planner_bounded,
    }
    payload = {
        "schema_version": 1,
        "experiment_id": protocol.experiment_id,
        "source_git_sha": telemetry["source_git_sha"],
        "sample_count": protocol.timing.sample_count,
        "warmups_per_method": protocol.timing.warmups_per_method,
        "timed_repetitions_per_method": protocol.timing.timed_repetitions_per_method,
        "method_summary": method_summary,
        "active_memory_reduction_fraction": memory_reduction,
        "paired_repetition_mean_latency_ratios": paired_ratios,
        "primary_paired_median_latency_ratio": primary_ratio,
        "paired_bootstrap_95_latency_ratio": list(interval),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "trial_table": str(trial_table),
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
