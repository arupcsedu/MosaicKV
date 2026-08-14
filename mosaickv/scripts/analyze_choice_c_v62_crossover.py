#!/usr/bin/env python3
"""Audit v62 crossover measurements against the preregistered paper gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

LENGTHS = (16, 64, 128)
MODES = ("full_kv", "retention_one", "prompt_attention_topk", "candidate_fast")


def _frame(root: Path, length: int, mode: str) -> pd.DataFrame:
    path = root / f"v62-validation-length{length}-{mode}" / "aggregate.parquet"
    if not path.is_file():
        raise RuntimeError(f"missing v62 aggregate: {path}")
    frame = pd.read_parquet(path)
    if len(frame) != 2 or set(frame["status"]) != {"completed"}:
        raise RuntimeError(f"v62 cell is incomplete: {path}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def _trace(root: Path, length: int, sample_id: str) -> dict[str, Any]:
    directory = root / "traces" / f"v62-validation-length{length}-candidate_fast"
    for path in sorted(directory.glob("*.json")):
        payload = cast("dict[str, Any]", json.loads(path.read_text()))
        if payload.get("sample_id") == sample_id:
            return payload
    raise RuntimeError(f"missing candidate trace for length={length}, sample={sample_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    trace_audits: list[dict[str, Any]] = []
    for length in LENGTHS:
        frames = {mode: _frame(root, length, mode) for mode in MODES}
        full = frames["full_kv"].set_index("sample_id")
        identity = frames["retention_one"].set_index("sample_id")
        prompt = frames["prompt_attention_topk"].set_index("sample_id")
        candidate = frames["candidate_fast"].set_index("sample_id")
        if list(full.index) != list(candidate.index):
            raise RuntimeError("v62 sample pairing changed between methods")
        observed_candidate_methods = set(candidate["method"].astype(str))
        if observed_candidate_methods != {"mosaickv_full"}:
            raise RuntimeError(
                "v62 candidate contains an exact-only or otherwise substituted method: "
                f"{sorted(observed_candidate_methods)}"
            )
        identity_match = bool(
            (full["answer"].astype(str) == identity["answer"].astype(str)).all()
        )
        for mode, frame in frames.items():
            rows.append(
                {
                    "output_length": length,
                    "mode": mode,
                    "sample_count": len(frame),
                    "mean_score": float(frame["task_score"].mean()),
                    "mean_active_kv_bytes": float(frame["active_kv_bytes"].mean()),
                    "mean_compression_seconds": float(frame["compression_time"].mean()),
                    "mean_decode_seconds": float(frame["decode_time"].mean()),
                    "mean_end_to_end_seconds": float(frame["end_to_end_time"].mean()),
                    "p50_end_to_end_seconds": float(frame["end_to_end_time"].median()),
                    "maximum_end_to_end_seconds": float(frame["end_to_end_time"].max()),
                }
            )
        memory_reduction = 1.0 - float(
            candidate["active_kv_bytes"].mean() / prompt["active_kv_bytes"].mean()
        )
        latency_reduction = 1.0 - float(
            candidate["end_to_end_time"].mean() / prompt["end_to_end_time"].mean()
        )
        quality_equal = bool(
            np.isclose(candidate["task_score"].mean(), full["task_score"].mean())
        )
        gates.append(
            {
                "output_length": length,
                "retention_one_answer_identity": identity_match,
                "candidate_quality_equals_fullkv": quality_equal,
                "active_memory_reduction_vs_prompt_topk": memory_reduction,
                "end_to_end_reduction_vs_prompt_topk": latency_reduction,
                "passes_30_memory_5_latency_equal_quality_gate": bool(
                    identity_match
                    and quality_equal
                    and memory_reduction >= 0.30
                    and latency_reduction >= 0.05
                ),
            }
        )
        for sample_id in full.index:
            trace = _trace(root, length, str(sample_id))
            if trace.get("tier_mode") != "three_tier":
                raise RuntimeError(
                    "v62 candidate trace is not a genuine three-tier construction: "
                    f"length={length}, sample={sample_id}, "
                    f"tier_mode={trace.get('tier_mode')}"
                )
            routing = cast("dict[str, Any]", trace["candidate_graph_routing"])
            prototype = cast("dict[str, Any]", trace["prototypes"])
            head_routes = cast(
                "list[Any]",
                cast("dict[str, Any]", trace["online_uncertainty_routing"]).get(
                    "prototype_head_routes", []
                ),
            )
            prototype_count = int(prototype.get("record_count", len(prototype)))
            trace_audits.append(
                {
                    "output_length": length,
                    "sample_id": str(sample_id),
                    "candidate_node_count": int(routing["candidate_node_count"]),
                    "graph_node_count": int(
                        cast("dict[str, Any]", trace["graph_diagnostics"])["node_count"]
                    ),
                    "prototype_head_route_count": len(head_routes),
                    "prototype_count": prototype_count,
                    "prototype_count_at_most_two_per_routed_head": bool(
                        prototype_count <= 2 * len(head_routes)
                    ),
                    "tier_construction_seconds": float(
                        cast("dict[str, Any]", trace["timing_breakdown"])[
                            "tier_construction"
                        ]
                    ),
                    "graph_seconds": float(
                        cast("dict[str, Any]", trace["timing_breakdown"])["graph"]
                    ),
                    "selection_seconds": float(
                        cast("dict[str, Any]", trace["timing_breakdown"])[
                            "utility_and_selection"
                        ]
                    ),
                    "packing_seconds": float(
                        cast("dict[str, Any]", trace["timing_breakdown"])["cache_packing"]
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = root / "v62_crossover_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    payload = {"gates": gates, "trace_audits": trace_audits}
    (root / "v62_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# V62 crossover and divergent-repair audit",
        "",
        "Exploratory dirty-worktree measurements only; these rows are not paper results.",
        "",
        "| Output tokens | Memory reduction vs Prompt Top-K | E2E reduction vs "
        "Prompt Top-K | FullKV quality parity | Gate |",
        "|---:|---:|---:|:---:|:---:|",
    ]
    for gate in gates:
        lines.append(
            f"| {gate['output_length']} | "
            f"{100 * float(gate['active_memory_reduction_vs_prompt_topk']):.2f}% | "
            f"{100 * float(gate['end_to_end_reduction_vs_prompt_topk']):.2f}% | "
            f"{'yes' if gate['candidate_quality_equals_fullkv'] else 'no'} | "
            f"{'PASS' if gate['passes_30_memory_5_latency_equal_quality_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "A paper sweep remains blocked unless the gate passes on held-out examples. "
            "Repair rows are generated only for samples whose candidate answer differs "
            "from the paired FullKV answer; absence of such rows means no divergence was "
            "observed, not zero measured repair cost.",
        ]
    )
    (root / "v62_analysis.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(gates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
