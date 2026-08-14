#!/usr/bin/env python3
"""Analyze the counterbalanced same-GPU Prompt Top-K/MosaicKV latency gate."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

RUN_PATTERN = re.compile(
    r"^v43-rank(?P<rank>[01])-cycle(?P<cycle>\d+)-order(?P<order>[01])-"
    r"(?P<mode>prompt_attention_topk|mosaickv_light_graph)-"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--latency-speedup", default=0.05, type=float)
    parser.add_argument("--active-memory-reduction", default=0.30, type=float)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("v43-rank*/results.jsonl")):
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
    if not rows:
        raise RuntimeError("no v43 measured rows found")
    if any(row.get("status") != "completed" or row.get("error") for row in rows):
        raise RuntimeError("v43 contains failed sample rows")

    indexed = {
        (row["rank"], row["cycle"], row["sample_id"], row["mode"]): row for row in rows
    }
    pair_keys = sorted({key[:3] for key in indexed})
    pairs: list[dict[str, float]] = []
    for rank, cycle, sample_id in pair_keys:
        prompt = indexed[(rank, cycle, sample_id, "prompt_attention_topk")]
        mosaic = indexed[(rank, cycle, sample_id, "mosaickv_light_graph")]
        pairs.append(
            {
                "prompt_seconds": float(prompt["end_to_end_time"]),
                "mosaic_seconds": float(mosaic["end_to_end_time"]),
                "prompt_active_bytes": float(prompt["active_kv_bytes"]),
                "mosaic_active_bytes": float(mosaic["active_kv_bytes"]),
            }
        )
    prompt_seconds = statistics.mean(item["prompt_seconds"] for item in pairs)
    mosaic_seconds = statistics.mean(item["mosaic_seconds"] for item in pairs)
    prompt_bytes = statistics.mean(item["prompt_active_bytes"] for item in pairs)
    mosaic_bytes = statistics.mean(item["mosaic_active_bytes"] for item in pairs)
    speedup = 1.0 - mosaic_seconds / prompt_seconds
    memory_reduction = 1.0 - mosaic_bytes / prompt_bytes

    mosaic_traces = []
    for path in sorted(root.glob("traces/v43-rank*-mosaickv_light_graph-*/*.json")):
        mosaic_traces.append(json.loads(path.read_text()))
    genuine = bool(mosaic_traces) and all(
        trace.get("tier_mode") == "three_tier"
        and trace.get("effective_method") == "mosaickv_full"
        for trace in mosaic_traces
    )
    passed = bool(
        speedup >= args.latency_speedup
        and memory_reduction >= args.active_memory_reduction
        and genuine
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
        "genuine_three_tier_rows": genuine,
        "latency_threshold_fraction": args.latency_speedup,
        "memory_threshold_fraction": args.active_memory_reduction,
        "passed": passed,
    }
    (root / "v43_latency_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (root / "v43_latency_gate.md").write_text(
        "# Same-GPU Prompt Top-K versus MosaicKV latency gate\n\n"
        f"- Paired observations: {len(pairs)}\n"
        f"- Prompt Top-K mean end-to-end: {prompt_seconds:.4f} s\n"
        f"- MosaicKV mean end-to-end: {mosaic_seconds:.4f} s\n"
        f"- MosaicKV latency improvement: {100 * speedup:.2f}%\n"
        f"- MosaicKV active-KV reduction: {100 * memory_reduction:.2f}%\n"
        f"- Genuine three-tier rows: {genuine}\n"
        f"- Gate passed: {passed}\n"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
