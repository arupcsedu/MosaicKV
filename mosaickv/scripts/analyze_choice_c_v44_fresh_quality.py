#!/usr/bin/env python3
"""Apply the preregistered MosaicKV light-graph held-out quality gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

MODES = ("implementation_only", "light_graph", "prompt_attention_topk")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MODES:
        paths = tuple(root.glob(f"v44-heldout-{mode}-*/results.jsonl"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one held-out result file for {mode}, found {len(paths)}")
        rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        if any(row.get("status") != "completed" or row.get("error") for row in rows):
            raise RuntimeError(f"held-out mode {mode} contains failed samples")
        by_mode[mode] = {str(row["sample_id"]): row for row in rows}
    sample_sets = {tuple(sorted(rows)) for rows in by_mode.values()}
    if len(sample_sets) != 1:
        raise RuntimeError("held-out methods did not evaluate identical sample IDs")
    sample_ids = next(iter(sample_sets))
    implementation = by_mode["implementation_only"]
    light = by_mode["light_graph"]
    prompt = by_mode["prompt_attention_topk"]

    light_identity = all(
        light[sample_id]["answer"] == implementation[sample_id]["answer"]
        and float(light[sample_id]["task_score"])
        == float(implementation[sample_id]["task_score"])
        for sample_id in sample_ids
    )
    trace_tokens: dict[str, dict[str, tuple[int, ...]]] = {}
    for mode in ("implementation_only", "light_graph"):
        trace_tokens[mode] = {}
        for path in root.glob(f"traces/v44-heldout-{mode}-*/*.json"):
            trace = json.loads(path.read_text())
            trace_tokens[mode][str(trace["sample_id"])] = tuple(
                int(value) for value in trace["generated_token_ids"]
            )
    light_token_identity = bool(
        set(trace_tokens["implementation_only"])
        == set(trace_tokens["light_graph"])
        == set(sample_ids)
        and all(
            trace_tokens["implementation_only"][sample_id]
            == trace_tokens["light_graph"][sample_id]
            for sample_id in sample_ids
        )
    )
    light_score = statistics.mean(float(light[item]["task_score"]) for item in sample_ids)
    implementation_score = statistics.mean(
        float(implementation[item]["task_score"]) for item in sample_ids
    )
    prompt_score = statistics.mean(float(prompt[item]["task_score"]) for item in sample_ids)
    light_bytes = statistics.mean(float(light[item]["active_kv_bytes"]) for item in sample_ids)
    prompt_bytes = statistics.mean(float(prompt[item]["active_kv_bytes"]) for item in sample_ids)
    score_delta = light_score - prompt_score
    memory_reduction = 1.0 - light_bytes / prompt_bytes

    traces = [
        json.loads(path.read_text())
        for path in sorted(root.glob("traces/v44-heldout-light_graph-*/*.json"))
    ]
    genuine = len(traces) == len(sample_ids) and all(
        trace.get("tier_mode") == "three_tier"
        and trace.get("effective_method") == "mosaickv_full"
        for trace in traces
    )
    passed = bool(
        light_identity
        and light_token_identity
        and score_delta >= -0.01
        and memory_reduction >= 0.30
        and genuine
    )
    report = {
        "schema_version": 1,
        "sample_count": len(sample_ids),
        "light_graph_exact_answer_and_score_identity": light_identity,
        "light_graph_exact_generated_token_identity": light_token_identity,
        "implementation_only_mean_score": implementation_score,
        "light_graph_mean_score": light_score,
        "prompt_attention_topk_mean_score": prompt_score,
        "light_graph_score_delta_vs_prompt": score_delta,
        "light_graph_active_kv_reduction_vs_prompt": memory_reduction,
        "genuine_three_tier_rows": genuine,
        "passed": passed,
    }
    (root / "v44_quality_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (root / "v44_quality_gate.md").write_text(
        "# Fresh disjoint held-out light-graph quality gate\n\n"
        f"- Samples: {len(sample_ids)}\n"
        f"- Light/full-graph answer and score identity: {light_identity}\n"
        f"- Light/full-graph generated-token identity: {light_token_identity}\n"
        f"- Full-graph MosaicKV mean score: {implementation_score:.6f}\n"
        f"- Light-graph MosaicKV mean score: {light_score:.6f}\n"
        f"- Prompt Top-K mean score: {prompt_score:.6f}\n"
        f"- Light-graph score delta versus Prompt Top-K: {score_delta:+.6f}\n"
        f"- Active-KV reduction versus Prompt Top-K: {100 * memory_reduction:.2f}%\n"
        f"- Genuine three-tier rows: {genuine}\n"
        f"- Gate passed: {passed}\n"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
