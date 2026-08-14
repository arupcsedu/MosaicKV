#!/usr/bin/env python3
"""Analyze the frozen, counterbalanced full-graph MosaicKV latency gate."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

RUN_PATTERN = re.compile(
    r"^v46-rank(?P<rank>[01])-cycle(?P<cycle>\d+)-order(?P<order>[01])-"
    r"(?P<mode>prompt_attention_topk|mosaickv_full_graph)-"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--latency-speedup", default=0.05, type=float)
    parser.add_argument("--active-memory-reduction", default=0.30, type=float)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("v46-rank*/results.jsonl")):
        match = RUN_PATTERN.match(path.parent.name)
        if match is None:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row.update(
                rank=int(match.group("rank")),
                cycle=int(match.group("cycle")),
                order=int(match.group("order")),
                mode=match.group("mode"),
            )
            rows.append(row)
    if not rows or any(row.get("status") != "completed" or row.get("error") for row in rows):
        raise RuntimeError("v46 has no complete, failure-free measured rows")

    indexed = {(r["rank"], r["cycle"], r["sample_id"], r["mode"]): r for r in rows}
    pair_keys = sorted({key[:3] for key in indexed})
    pairs = [
        (
            indexed[(*key, "prompt_attention_topk")],
            indexed[(*key, "mosaickv_full_graph")],
        )
        for key in pair_keys
    ]
    prompt_seconds = statistics.mean(float(prompt["end_to_end_time"]) for prompt, _ in pairs)
    mosaic_seconds = statistics.mean(float(mosaic["end_to_end_time"]) for _, mosaic in pairs)
    prompt_bytes = statistics.mean(float(prompt["active_kv_bytes"]) for prompt, _ in pairs)
    mosaic_bytes = statistics.mean(float(mosaic["active_kv_bytes"]) for _, mosaic in pairs)
    speedup = 1.0 - mosaic_seconds / prompt_seconds
    memory_reduction = 1.0 - mosaic_bytes / prompt_bytes

    traces = [
        json.loads(path.read_text())
        for path in sorted(root.glob("traces/v46-rank*-mosaickv_full_graph-*/*.json"))
    ]
    genuine = bool(traces) and all(
        trace.get("tier_mode") == "three_tier"
        and trace.get("effective_method") == "mosaickv_full"
        for trace in traces
    )
    snapshots = []
    for path in sorted((root / "telemetry").glob("v46-rank*.json")):
        telemetry = json.loads(path.read_text())
        if telemetry.get("status") != "completed":
            raise RuntimeError(f"incomplete worker telemetry: {path}")
        snapshots.append(telemetry["source_snapshot"]["snapshot_sha256"])
    source_frozen = len(snapshots) == 2 and len(set(snapshots)) == 1
    phase_names = (
        "forecast",
        "graph",
        "utility_and_selection",
        "tier_construction",
        "cache_packing",
        "decode",
        "host_orchestration",
    )
    phase_means = {
        phase: statistics.mean(
            float(trace["timing_breakdown"][phase]) for trace in traces
        )
        for phase in phase_names
    }
    passed = bool(
        speedup >= args.latency_speedup
        and memory_reduction >= args.active_memory_reduction
        and genuine
        and source_frozen
    )
    report = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_same_gpu_counterbalanced_latency_gate",
        "pair_count": len(pairs),
        "prompt_mean_end_to_end_seconds": prompt_seconds,
        "mosaickv_mean_end_to_end_seconds": mosaic_seconds,
        "mosaickv_latency_speedup_fraction": speedup,
        "prompt_mean_active_kv_bytes": prompt_bytes,
        "mosaickv_mean_active_kv_bytes": mosaic_bytes,
        "mosaickv_active_memory_reduction_fraction": memory_reduction,
        "mosaickv_phase_means_seconds": phase_means,
        "genuine_three_tier_rows": genuine,
        "source_frozen_across_workers": source_frozen,
        "light_graph_used": False,
        "latency_threshold_fraction": args.latency_speedup,
        "memory_threshold_fraction": args.active_memory_reduction,
        "passed": passed,
    }
    (root / "v46_latency_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (root / "v46_latency_gate.md").write_text(
        "# Frozen same-GPU Prompt Top-K versus full-graph MosaicKV latency gate\n\n"
        f"- Paired observations: {len(pairs)}\n"
        f"- Prompt Top-K mean end-to-end: {prompt_seconds:.4f} s\n"
        f"- MosaicKV mean end-to-end: {mosaic_seconds:.4f} s\n"
        f"- MosaicKV latency improvement: {100 * speedup:.2f}%\n"
        f"- MosaicKV active-KV reduction: {100 * memory_reduction:.2f}%\n"
        f"- Genuine full-graph three-tier rows: {genuine}\n"
        f"- Source frozen across workers: {source_frozen}\n"
        f"- Gate passed: {passed}\n"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
