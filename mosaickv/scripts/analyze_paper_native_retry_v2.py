#!/usr/bin/env python3
"""Evaluate the preregistered memory, quality, layout, and completeness gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict

from mosaickv.evaluation.metrics.storage import load_metric_jsonl
from mosaickv.evaluation.metrics.types import MetricObservation, MetricScope
from mosaickv.evaluation.paper_native_latency import canonical_inference_traces
from mosaickv.evaluation.results import EvaluationResult, ResultStatus
from mosaickv.evaluation.storage import load_jsonl


class CompletedSummary(TypedDict):
    sample_count: int
    completed_samples: int
    failed_samples: int
    mean_active_kv_bytes: float
    mean_end_to_end_seconds: float
    mean_task_score: float | None


class LayoutSummary(TypedDict):
    trace_count: int
    maximum_packed_padding_kv_bytes: int
    uniform_active_positions_per_layer_head: bool
    maximum_planner_node_count: int
    maximum_source_node_count: int
    maximum_planner_node_fraction: float


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _rows(root: Path, lane: str, method: str) -> tuple[EvaluationResult, ...]:
    return load_jsonl(root / lane / f"{method}-raw.jsonl")


def _metrics(root: Path, lane: str, method: str) -> tuple[MetricObservation, ...]:
    return load_metric_jsonl(root / lane / f"{method}-metrics.jsonl")


def _mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    if not collected:
        raise ValueError("cannot average an empty metric")
    return math.fsum(collected) / len(collected)


def _completed_summary(rows: tuple[EvaluationResult, ...]) -> CompletedSummary:
    completed = tuple(row for row in rows if row.status is ResultStatus.COMPLETED)
    failed = tuple(row for row in rows if row.status is ResultStatus.FAILED)
    active_bytes = tuple(row.active_kv_bytes for row in completed)
    if not completed or any(value is None for value in active_bytes):
        raise ValueError("completed paper-native rows require measured active bytes")
    timing = tuple(
        float(row.end_to_end_time) for row in completed if row.end_to_end_time is not None
    )
    scores = tuple(float(row.task_score) for row in completed if row.task_score is not None)
    return {
        "sample_count": len(rows),
        "completed_samples": len(completed),
        "failed_samples": len(failed),
        "mean_active_kv_bytes": _mean(float(value) for value in active_bytes if value is not None),
        "mean_end_to_end_seconds": _mean(timing),
        "mean_task_score": None if not scores else _mean(scores),
    }


def _corpus_metric(rows: tuple[MetricObservation, ...], name: str) -> float:
    selected = tuple(
        row.metric_value
        for row in rows
        if row.metric_scope is MetricScope.CORPUS and row.metric_name == name
    )
    if len(selected) != 1:
        raise ValueError(f"expected one corpus {name!r} metric, observed {len(selected)}")
    return selected[0]


def _sample_metric(rows: tuple[MetricObservation, ...], name: str) -> float:
    return _mean(
        row.metric_value
        for row in rows
        if row.metric_scope is MetricScope.SAMPLE and row.metric_name == name
    )


def _required_task_score(summary: CompletedSummary, label: str) -> float:
    value = summary["mean_task_score"]
    if value is None:
        raise ValueError(f"{label} requires sample-level task scores")
    return value


def _trace_records(
    root: Path,
    lane: str,
    experiment_id: str,
) -> tuple[dict[str, Any], ...]:
    trace_root = root / lane / "traces" / f"{experiment_id}-{lane}-mosaickv_exact"
    paths = sorted(trace_root.glob("*.json"))
    if not paths:
        raise ValueError(f"MosaicKV trace directory is empty: {trace_root}")
    return canonical_inference_traces(tuple(paths))


def _nested(record: Mapping[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"trace lacks required field: {'.'.join(path)}")
        value = value[key]
    return value


def _layout_summary(traces: tuple[dict[str, Any], ...]) -> LayoutSummary:
    padding = tuple(int(_nested(trace, "packed_padding_kv_bytes")) for trace in traces)
    planner_nodes = tuple(
        int(_nested(trace, "candidate_graph_routing", "planner_node_count")) for trace in traces
    )
    source_nodes = tuple(
        int(_nested(trace, "candidate_graph_routing", "source_node_count")) for trace in traces
    )
    if any(value <= 0 for value in source_nodes):
        raise ValueError("trace source_node_count must be positive")
    active_lengths = tuple(
        tuple(int(value) for value in _nested(trace, "tier_active_cache_lengths"))
        for trace in traces
    )
    uniform = all(lengths and len(set(lengths)) == 1 for lengths in active_lengths)
    return {
        "trace_count": len(traces),
        "maximum_packed_padding_kv_bytes": max(padding),
        "uniform_active_positions_per_layer_head": uniform,
        "maximum_planner_node_count": max(planner_nodes),
        "maximum_source_node_count": max(source_nodes),
        "maximum_planner_node_fraction": max(
            planner / source for planner, source in zip(planner_nodes, source_nodes, strict=True)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment-id",
        choices=("paper-native-retry-v2", "paper-native-latency-v3"),
        default="paper-native-retry-v2",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    methods = {
        "prefixkv": ("full_kv", "prefixkv_reimpl", "mosaickv_exact"),
        "milebench": ("full_kv", "lookm_reimpl", "mosaickv_exact"),
    }
    summaries: dict[str, dict[str, CompletedSummary]] = {
        lane: {method: _completed_summary(_rows(root, lane, method)) for method in lane_methods}
        for lane, lane_methods in methods.items()
    }
    prefix_metrics = {method: _metrics(root, "prefixkv", method) for method in methods["prefixkv"]}
    prefix_quality = {
        method: {
            "rouge_l_f1": _sample_metric(rows, "rouge_l_f1"),
            "teacher_forced_ppl": _corpus_metric(rows, "teacher_forced_ppl"),
        }
        for method, rows in prefix_metrics.items()
    }
    gates: dict[str, bool] = {}
    for lane, baseline in (("prefixkv", "prefixkv_reimpl"), ("milebench", "lookm_reimpl")):
        mosaic_bytes = summaries[lane]["mosaickv_exact"]["mean_active_kv_bytes"]
        baseline_bytes = summaries[lane][baseline]["mean_active_kv_bytes"]
        gates[f"{lane}_minimum_30_percent_memory_reduction"] = (
            1.0 - mosaic_bytes / baseline_bytes >= 0.30
        )
        if args.experiment_id == "paper-native-latency-v3":
            mosaic_latency = summaries[lane]["mosaickv_exact"]["mean_end_to_end_seconds"]
            baseline_latency = summaries[lane][baseline]["mean_end_to_end_seconds"]
            gates[f"{lane}_latency_within_5_percent"] = mosaic_latency <= 1.05 * baseline_latency
    gates["prefixkv_rouge_noninferior"] = (
        prefix_quality["mosaickv_exact"]["rouge_l_f1"]
        >= prefix_quality["prefixkv_reimpl"]["rouge_l_f1"]
    )
    gates["prefixkv_ppl_noninferior"] = (
        prefix_quality["mosaickv_exact"]["teacher_forced_ppl"]
        <= prefix_quality["prefixkv_reimpl"]["teacher_forced_ppl"]
    )
    gates["milebench_accuracy_noninferior"] = _required_task_score(
        summaries["milebench"]["mosaickv_exact"],
        "MileBench MosaicKV",
    ) >= _required_task_score(
        summaries["milebench"]["lookm_reimpl"],
        "MileBench LOOK-M",
    )
    gates["milebench_fullkv_accuracy_measurable"] = (
        _required_task_score(
            summaries["milebench"]["full_kv"],
            "MileBench FullKV",
        )
        > 0
    )
    gates["no_failed_samples"] = all(
        summary["failed_samples"] == 0 for lane in summaries.values() for summary in lane.values()
    )
    layouts: dict[str, LayoutSummary] = {
        lane: _layout_summary(_trace_records(root, lane, args.experiment_id)) for lane in methods
    }
    gates["zero_physical_padding"] = all(
        layout["maximum_packed_padding_kv_bytes"] == 0 for layout in layouts.values()
    )
    gates["uniform_physical_capacity"] = all(
        bool(layout["uniform_active_positions_per_layer_head"]) for layout in layouts.values()
    )
    maximum_planner_fraction = 0.025 if args.experiment_id == "paper-native-latency-v3" else 0.10
    gates["bounded_shortlist_planner"] = all(
        layout["maximum_planner_node_fraction"] <= maximum_planner_fraction
        for layout in layouts.values()
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": args.experiment_id.replace("-", "_"),
        "classification": "preregistered_development_validation",
        "summaries": summaries,
        "prefixkv_quality": prefix_quality,
        "layout": layouts,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    _write_once(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
