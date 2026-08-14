#!/usr/bin/env python3
"""Analyze the compact split-planner latency and accounting gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

import numpy as np
from analyze_choice_c_v67_matched_sdpa import analyze as analyze_paired_gate


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(array, 95)),
        "maximum": max(values),
    }


def analyze(output_root: Path) -> dict[str, Any]:
    result = analyze_paired_gate(output_root)
    paired_gates = cast("dict[str, Any]", result["gates"])
    result["quality_memory_gate"] = dict(paired_gates)
    result["quality_memory_gate"]["latency_policy"] = cast(
        "dict[str, Any]", result["latency_report"]
    )["policy"]
    trace_paths = tuple(
        sorted((output_root / "traces").glob("v67-trial-*-mosaickv_full/*.json"))
    )
    if len(trace_paths) != 10:
        raise RuntimeError(f"expected 10 measured MosaicKV traces, found {len(trace_paths)}")
    traces = [cast("dict[str, Any]", json.loads(path.read_text())) for path in trace_paths]
    if any(trace.get("status") != "completed" for trace in traces):
        raise RuntimeError("one or more MosaicKV traces are incomplete")

    preparation = [
        float(cast("dict[str, Any]", trace["timing_breakdown"])["planner_preparation"])
        for trace in traces
    ]
    graph = [
        float(cast("dict[str, Any]", trace["timing_breakdown"])["graph"])
        for trace in traces
    ]
    planner_nodes = [
        int(cast("dict[str, Any]", trace["candidate_graph_routing"])["planner_node_count"])
        for trace in traces
    ]
    source_nodes = [
        int(cast("dict[str, Any]", trace["candidate_graph_routing"])["source_node_count"])
        for trace in traces
    ]
    external_exact = [
        int(
            cast("dict[str, Any]", trace["candidate_graph_routing"])[
                "external_stable_exact_node_count"
            ]
        )
        for trace in traces
    ]
    external_capacity = [
        cast(
            "dict[str, Any]",
            cast("dict[str, Any]", trace["candidate_graph_routing"])[
                "external_stable_capacity"
            ],
        )
        for trace in traces
    ]
    accounting_errors: list[float] = []
    for trace in traces:
        timing = cast("dict[str, Any]", trace["timing_breakdown"])
        expected = sum(
            float(timing[name])
            for name in (
                "cache_state_construction",
                "forecast",
                "planner_preparation",
                "graph",
                "utility_and_selection",
                "tier_construction",
                "cache_packing",
            )
        )
        run_id = str(trace["run_id"])
        sample_id = str(trace["sample_id"])
        rows_path = output_root / run_id / "results.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        matching = [row for row in rows if row["sample_id"] == sample_id]
        if len(matching) != 1:
            raise RuntimeError(f"cannot resolve one result row for {run_id}/{sample_id}")
        accounting_errors.append(abs(float(matching[0]["compression_time"]) - expected))

    preparation_summary = _summary(preparation)
    graph_summary = _summary(graph)
    compaction_fraction = [
        1.0 - planner / source
        for planner, source in zip(planner_nodes, source_nodes, strict=True)
    ]
    split_gates = {
        "mean_planner_preparation_below_0_10_seconds": (
            preparation_summary["mean"] < 0.10
        ),
        "mean_graph_below_0_15_seconds": graph_summary["mean"] < 0.15,
        "compression_accounting_max_error_below_1e_9": max(accounting_errors) < 1e-9,
        "planner_node_fraction_at_most_0_07": max(
            planner / source
            for planner, source in zip(planner_nodes, source_nodes, strict=True)
        )
        <= 0.07,
    }
    split_gates["split_planner_gate_pass"] = all(split_gates.values())
    result["split_planner"] = {
        "planner_preparation_seconds": preparation_summary,
        "graph_seconds": graph_summary,
        "planner_node_count": sorted(set(planner_nodes)),
        "source_node_count": sorted(set(source_nodes)),
        "external_stable_exact_node_count": sorted(set(external_exact)),
        "external_stable_requested_positions": _summary(
            [float(item["requested_positions"]) for item in external_capacity]
        ),
        "external_stable_selected_positions": _summary(
            [float(item["selected_positions"]) for item in external_capacity]
        ),
        "external_stable_capacity_shortfall_positions": _summary(
            [float(item["shortfall_positions"]) for item in external_capacity]
        ),
        "external_stable_selected_bytes": _summary(
            [float(item["selected_bytes"]) for item in external_capacity]
        ),
        "external_stable_routes_with_shortfall": sorted(
            {
                len(cast("list[dict[str, Any]]", item["routes_with_shortfall"]))
                for item in external_capacity
            }
        ),
        "planner_node_reduction_fraction": _summary(compaction_fraction),
        "compression_accounting_absolute_error_seconds": _summary(accounting_errors),
        "gates": split_gates,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    result = analyze(output_root)
    destination = output_root / "analysis" / "split_planner_gate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["split_planner"]["gates"], indent=2, sort_keys=True))
    print(json.dumps(result["quality_memory_gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
