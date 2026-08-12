#!/usr/bin/env python3
"""Validate and summarize the small incremental MosaicKV idea test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from run_choice_c_incremental_worker import MODES

TIMING_COLUMNS = ("ttft", "prefill_time", "compression_time", "decode_time", "end_to_end_time")


def _traces(root: Path, run_id: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text()) for path in sorted((root / "traces" / run_id).glob("*.json"))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows: list[pd.DataFrame] = []
    trace_rows: list[dict[str, Any]] = []
    for rank in (0, 1):
        for mode in MODES:
            matches = sorted(
                root.glob(f"incremental-measured-{mode}-rank{rank}-*/aggregate.parquet")
            )
            if len(matches) != 1:
                raise RuntimeError(f"expected one measured result for rank={rank}, mode={mode}")
            frame = pd.read_parquet(matches[0]).copy()
            frame["rank"] = rank
            frame["idea_mode"] = mode
            rows.append(frame)
            run_id = str(frame["run_id"].iloc[0])
            for trace in _traces(root, run_id):
                forecast = trace.get("forecast_statistics", {})
                provenance = forecast.get("provenance", {}) if isinstance(forecast, dict) else {}
                storage = trace.get("tier_storage_accounting", {})
                trace_rows.append(
                    {
                        "rank": rank,
                        "idea_mode": mode,
                        "sample_id": trace["sample_id"],
                        "effective_method": trace.get("effective_method"),
                        "tier_mode": trace.get("tier_mode"),
                        "conditional_draft_decision": provenance.get("conditional_draft_decision"),
                        "prompt_uncertainty": provenance.get("prompt_uncertainty"),
                        "total_active_plus_residual_bytes": storage.get(
                            "total_active_plus_residual_bytes"
                        ),
                        "stored_residual_source_blocks": storage.get(
                            "stored_residual_source_blocks"
                        ),
                    }
                )
    trials = pd.concat(rows, ignore_index=True)
    if len(trials) != len(MODES) * 2 * 2:
        raise RuntimeError("incremental result grid is incomplete")
    if set(trials["status"]) != {"completed"}:
        raise RuntimeError("incremental result grid contains failed rows")
    for column in TIMING_COLUMNS:
        values = pd.to_numeric(trials[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise RuntimeError(f"timing column contains missing/non-finite values: {column}")
    if (trials["repair_count"] != 0).any():
        raise RuntimeError("repair-disabled incremental test unexpectedly repaired a block")
    traces = pd.DataFrame(trace_rows)
    bounded = trials[trials["idea_mode"].isin(("bounded_residual_rr20", "combined_rr14"))]
    if (bounded["residual_kv_bytes"] > 8 * 1024 * 1024).any():
        raise RuntimeError("bounded residual mode exceeded its 8 MiB cap")
    candidate = trials[trials["idea_mode"] == "combined_rr14"].set_index(["rank", "sample_id"])
    prompt = trials[trials["idea_mode"] == "prompt_topk_rr20"].set_index(["rank", "sample_id"])
    if set(candidate.index) != set(prompt.index):
        raise RuntimeError("candidate and Prompt Top-K rows are not paired")
    paired = pd.DataFrame(index=candidate.index)
    paired["active_reduction"] = 1.0 - candidate["active_kv_bytes"] / prompt["active_kv_bytes"]
    paired["total_storage_reduction"] = 1.0 - (
        candidate["active_kv_bytes"] + candidate["residual_kv_bytes"]
    ) / (prompt["active_kv_bytes"] + prompt["residual_kv_bytes"])
    paired["e2e_reduction"] = 1.0 - candidate["end_to_end_time"] / prompt["end_to_end_time"]
    paired["quality_delta"] = candidate["task_score"] - prompt["task_score"]
    summary = (
        trials.groupby("idea_mode", sort=False)
        .agg(
            rows=("sample_id", "size"),
            task_score_mean=("task_score", "mean"),
            active_kv_bytes_median=("active_kv_bytes", "median"),
            residual_kv_bytes_median=("residual_kv_bytes", "median"),
            compression_time_median=("compression_time", "median"),
            decode_time_median=("decode_time", "median"),
            end_to_end_time_median=("end_to_end_time", "median"),
        )
        .reset_index()
    )
    gate = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_small_sample_not_paper_evidence",
        "all_rows_complete": True,
        "bounded_residual_cap_satisfied": True,
        "candidate_vs_prompt_topk": {
            "mean_active_reduction": float(paired["active_reduction"].mean()),
            "mean_total_storage_reduction": float(paired["total_storage_reduction"].mean()),
            "mean_end_to_end_reduction": float(paired["e2e_reduction"].mean()),
            "mean_task_score_delta": float(paired["quality_delta"].mean()),
            "active_30_percent_gate": bool((paired["active_reduction"].mean()) >= 0.30),
            "total_storage_30_percent_gate": bool(
                (paired["total_storage_reduction"].mean()) >= 0.30
            ),
            "latency_5_percent_gate": bool((paired["e2e_reduction"].mean()) >= 0.05),
            "noninferior_observed_mean_score_gate": bool(paired["quality_delta"].mean() >= 0),
        },
        "conditional_draft_decisions": traces["conditional_draft_decision"]
        .dropna()
        .value_counts()
        .to_dict(),
    }
    trials.to_parquet(root / "incremental_trials.parquet", index=False)
    summary.to_parquet(root / "incremental_summary.parquet", index=False)
    paired.reset_index().to_parquet(root / "candidate_prompt_pairs.parquet", index=False)
    (root / "incremental_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    comparison = gate["candidate_vs_prompt_topk"]
    lines = [
        "# Incremental MosaicKV idea smoke test",
        "",
        "This is an exploratory two-sample, two-GPU dirty-tree test; it is not paper evidence.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Combined RR=0.14 candidate versus Prompt Top-K RR=0.20",
        "",
        f"- Mean active-KV reduction: {comparison['mean_active_reduction']:.2%}",
        "- Mean active+residual storage reduction: "
        f"{comparison['mean_total_storage_reduction']:.2%}",
        f"- Mean end-to-end reduction: {comparison['mean_end_to_end_reduction']:.2%}",
        f"- Mean task-score delta: {comparison['mean_task_score_delta']:+.4f}",
        "",
        "The ratio-mismatched row is a Pareto-candidate smoke check only. A later "
        "held-out run must establish quality non-inferiority before any "
        "apples-to-apples Pareto claim.",
    ]
    (root / "incremental_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
