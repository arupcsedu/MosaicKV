#!/usr/bin/env python3
"""Summarize the declared-calibration full-graph candidate screen."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from run_choice_c_v50_full_graph_candidates_worker import RANK_MODES


def _mode(run_id: str) -> str:
    for mode in (item for values in RANK_MODES.values() for item in values):
        if f"-{mode}-" in run_id:
            return mode
    raise ValueError(f"cannot resolve candidate mode from {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    telemetry = tuple(
        json.loads(path.read_text())
        for path in sorted((root / "telemetry").glob("*.json"))
    )
    if len(telemetry) != 2 or any(row.get("status") != "completed" for row in telemetry):
        raise RuntimeError("both frozen v50 workers must complete before analysis")
    snapshots = {row["source_snapshot"]["snapshot_sha256"] for row in telemetry}
    if len(snapshots) != 1:
        raise RuntimeError("v50 workers used different source snapshots")

    rows: list[dict[str, Any]] = []
    traces: dict[tuple[str, str], dict[str, Any]] = {}
    for result_path in sorted(root.glob("v50-calibration-*/results.jsonl")):
        mode = _mode(result_path.parent.name)
        for line in result_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") != "completed" or row.get("error") is not None:
                raise RuntimeError(f"failed v50 result row: {row}")
            rows.append({"mode": mode, **row})
            trace_paths = tuple((root / "traces" / str(row["run_id"])).glob("*.json"))
            matching = tuple(
                json.loads(path.read_text())
                for path in trace_paths
                if json.loads(path.read_text()).get("sample_id") == row["sample_id"]
            )
            if len(matching) != 1:
                raise RuntimeError(
                    f"expected one trace for {row['run_id']} / {row['sample_id']}"
                )
            trace = matching[0]
            traces[(mode, str(row["sample_id"]))] = trace
    expected_modes = tuple(item for values in RANK_MODES.values() for item in values)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row["mode"])].append(row)
    if set(by_mode) != set(expected_modes):
        raise RuntimeError(f"v50 mode coverage is incomplete: {sorted(by_mode)}")
    sample_sets = {
        mode: {str(row["sample_id"]) for row in values}
        for mode, values in by_mode.items()
    }
    if len({frozenset(values) for values in sample_sets.values()}) != 1:
        raise RuntimeError("v50 modes did not evaluate identical calibration samples")

    prompt_active = mean(float(row["active_kv_bytes"]) for row in by_mode["prompt_attention_topk"])
    prompt_latency = mean(float(row["end_to_end_time"]) for row in by_mode["prompt_attention_topk"])
    prompt_score = mean(float(row["task_score"]) for row in by_mode["prompt_attention_topk"])
    summaries: list[dict[str, Any]] = []
    for mode in expected_modes:
        values = by_mode[mode]
        tier_modes = {
            str(traces[(mode, str(row["sample_id"]))].get("tier_mode")) for row in values
        }
        config_paths = tuple((root / "configs").glob(f"*{mode}*.yaml"))
        graph_neighbor_values = {
            int(yaml.safe_load(path.read_text())["graph"]["max_neighbors"])
            for path in config_paths
        }
        light_graph = mode != "prompt_attention_topk" and graph_neighbor_values != {4}
        active = mean(float(row["active_kv_bytes"]) for row in values)
        latency = mean(float(row["end_to_end_time"]) for row in values)
        score = mean(float(row["task_score"]) for row in values)
        summaries.append(
            {
                "mode": mode,
                "sample_count": len(values),
                "mean_score": score,
                "score_delta_vs_prompt": score - prompt_score,
                "mean_end_to_end_seconds": latency,
                "latency_speedup_vs_prompt": 1.0 - latency / prompt_latency,
                "mean_active_kv_bytes": active,
                "active_reduction_vs_prompt": 1.0 - active / prompt_active,
                "tier_modes": sorted(tier_modes),
                "all_three_tier": tier_modes == {"three_tier"},
                "light_graph_detected": light_graph,
            }
        )
    eligible = [
        row
        for row in summaries
        if row["mode"] != "prompt_attention_topk"
        and row["all_three_tier"]
        and not row["light_graph_detected"]
        and float(row["score_delta_vs_prompt"]) >= -0.01
        and float(row["active_reduction_vs_prompt"]) >= 0.30
    ]
    selected = (
        min(eligible, key=lambda row: float(row["mean_end_to_end_seconds"]))
        if eligible
        else None
    )
    payload = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_calibration_candidate_screen",
        "source_frozen_across_workers": len(snapshots) == 1,
        "light_graph_prohibited": True,
        "selection_rule": (
            "lowest calibration latency among full-graph genuine-three-tier candidates with "
            "score delta >= -0.01 and active-KV reduction >= 0.30"
        ),
        "summaries": summaries,
        "selected_candidate": None if selected is None else selected["mode"],
        "requires_fresh_disjoint_heldout_validation": selected is not None,
    }
    (root / "v50_candidate_screen.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Full-graph calibration candidate screen",
        "",
        "This is an exploratory calibration result, not held-out evidence.",
        "",
        "| Mode | Score | Delta vs Prompt | E2E (s) | Speedup | Active reduction | Three-tier |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['mode']} | {row['mean_score']:.6f} | "
            f"{row['score_delta_vs_prompt']:+.6f} | "
            f"{row['mean_end_to_end_seconds']:.4f} | "
            f"{100 * row['latency_speedup_vs_prompt']:+.2f}% | "
            f"{100 * row['active_reduction_vs_prompt']:.2f}% | "
            f"{row['all_three_tier']} |"
        )
    lines.extend(
        [
            "",
            f"Selected for a new disjoint held-out gate: {payload['selected_candidate']}",
            "",
        ]
    )
    (root / "v50_candidate_screen.md").write_text("\n".join(lines))
    print(json.dumps(payload, sort_keys=True))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
