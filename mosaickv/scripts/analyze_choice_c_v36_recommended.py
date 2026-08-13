#!/usr/bin/env python3
"""Validate the tiny retrospective v36 quality/storage/latency comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from run_choice_c_v36_recommended_worker import RANK_MODES

TIMING_COLUMNS = ("ttft", "prefill_time", "compression_time", "decode_time", "end_to_end_time")
COMPETITORS = ("prompt_attention_topk", "lookm_reimpl", "vl_cache_reimpl")


def _read_traces(root: Path, run_id: str) -> list[dict[str, Any]]:
    paths = sorted((root / "traces" / run_id).glob("*.json"))
    return [json.loads(path.read_text()) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    frames: list[pd.DataFrame] = []
    traces: list[dict[str, Any]] = []
    for rank, modes in RANK_MODES.items():
        for mode in modes:
            paths = sorted(root.glob(f"v36-measured-{mode}-rank{rank}-*/aggregate.parquet"))
            if len(paths) != 1:
                raise RuntimeError(f"expected one v36 result for rank={rank}, mode={mode}")
            frame = pd.read_parquet(paths[0]).copy()
            frame["rank"] = rank
            frame["validation_mode"] = mode
            frames.append(frame)
            run_id = str(frame["run_id"].iloc[0])
            run_traces = _read_traces(root, run_id)
            if len(run_traces) != len(frame):
                raise RuntimeError(f"trace/result count mismatch for {run_id}")
            traces.extend(run_traces)
    trials = pd.concat(frames, ignore_index=True)
    expected_rows = sum(len(modes) for modes in RANK_MODES.values()) * 2
    if len(trials) != expected_rows or set(trials["status"]) != {"completed"}:
        raise RuntimeError("v36 comparison is incomplete or contains failed rows")
    if trials["error"].notna().any():
        raise RuntimeError("completed v36 rows unexpectedly contain error records")
    for column in TIMING_COLUMNS:
        values = pd.to_numeric(trials[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all() or (values < 0).any():
            raise RuntimeError(f"v36 has invalid timing values in {column}")
    if (trials["generated_tokens"] != 16).any():
        raise RuntimeError("v36 timed generation length differs across methods")
    if (trials["repair_count"] != 0).any():
        raise RuntimeError("repair-policy-none v36 row unexpectedly repaired a block")
    if trials["manifest_path"].isna().any():
        raise RuntimeError("v36 row is missing its run manifest")
    for manifest_path in sorted(set(trials["manifest_path"].astype(str))):
        manifest = json.loads(Path(manifest_path).read_text())
        if not manifest["model"].get("revision") or not manifest["dataset"].get("revision"):
            raise RuntimeError("v36 manifest is missing model or dataset revision")
        if manifest["source"].get("git_dirty") is not True:
            raise RuntimeError("v36 must remain classified as a dirty-tree exploratory run")

    full = trials[trials["validation_mode"] == "full_kv"]
    if len(full) != 2 or (full["task_score"] != 1.0).any():
        raise RuntimeError("the retrospective non-degenerate FullKV control was not correct")
    mosaic = trials[trials["validation_mode"] == "mosaickv_recommended"]
    if len(mosaic) != 4:
        raise RuntimeError("both V100 workers must replicate the MosaicKV candidate")
    if (mosaic["residual_kv_bytes"] > 1024 * 1024).any():
        raise RuntimeError("MosaicKV exceeded its 1 MiB residual cap")

    mosaic_traces = [
        trace
        for trace in traces
        if str(trace.get("run_id", "")).startswith("v36-measured-mosaickv")
    ]
    if len(mosaic_traces) != len(mosaic):
        raise RuntimeError("MosaicKV traces are incomplete")
    for trace in mosaic_traces:
        if trace.get("effective_method") != "mosaickv_full":
            raise RuntimeError("v36 MosaicKV silently used a fallback method")
        if trace.get("tier_mode") != "three_tier":
            raise RuntimeError("v36 MosaicKV did not construct the three-tier cache")
        if trace.get("packed_cache_all_slots_valid") is not True:
            raise RuntimeError("v36 MosaicKV still required padded decode validity masks")
        active_bytes = trace.get("active_kv_bytes")
        hard_budget = trace.get("hard_active_kv_budget_bytes")
        if (
            not isinstance(active_bytes, int)
            or not isinstance(hard_budget, int)
            or active_bytes > hard_budget
        ):
            raise RuntimeError("v36 MosaicKV active bytes exceed its physical budget")
        if trace.get("packed_padding_kv_bytes") != 0:
            raise RuntimeError("v36 MosaicKV materialized padding bytes")
        packed_capacity = trace.get("packed_prompt_capacity")
        hard_capacity = trace.get("hard_prompt_capacity")
        tier_lengths = trace.get("tier_active_cache_lengths")
        if (
            not isinstance(packed_capacity, int)
            or not isinstance(hard_capacity, int)
            or packed_capacity > hard_capacity
            or tier_lengths != [packed_capacity]
        ):
            raise RuntimeError(
                "v36 MosaicKV planned, constructed, and packed cache lengths disagree"
            )
        if packed_capacity < hard_capacity and (
            f"physical cap from {hard_capacity} to {packed_capacity} positions"
            not in str(trace.get("tier_reason", ""))
        ):
            raise RuntimeError("v36 MosaicKV underfill provenance reports the wrong length")
        quality = trace.get("quality_decoding", {})
        if not isinstance(quality, dict) or quality.get("terminal_token_id") is None:
            raise RuntimeError("v36 answer did not reach a terminal token within the fixed output")

    summary = (
        trials.groupby("validation_mode", sort=False)
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
    mosaic_by_sample = mosaic.groupby("sample_id", as_index=True).agg(
        task_score=("task_score", "mean"),
        active_kv_bytes=("active_kv_bytes", "mean"),
        residual_kv_bytes=("residual_kv_bytes", "mean"),
        end_to_end_time=("end_to_end_time", "mean"),
    )
    comparisons: list[dict[str, Any]] = []
    for competitor in COMPETITORS:
        baseline = trials[trials["validation_mode"] == competitor].set_index("sample_id")
        if set(baseline.index) != set(mosaic_by_sample.index):
            raise RuntimeError(f"MosaicKV and {competitor} samples are not paired")
        active_reduction = 1.0 - mosaic_by_sample["active_kv_bytes"] / baseline["active_kv_bytes"]
        total_reduction = 1.0 - (
            mosaic_by_sample["active_kv_bytes"] + mosaic_by_sample["residual_kv_bytes"]
        ) / (baseline["active_kv_bytes"] + baseline["residual_kv_bytes"])
        latency_reduction = 1.0 - (
            mosaic_by_sample["end_to_end_time"] / baseline["end_to_end_time"]
        )
        score_delta = mosaic_by_sample["task_score"] - baseline["task_score"]
        comparisons.append(
            {
                "competitor": competitor,
                "mean_active_reduction": float(active_reduction.mean()),
                "mean_total_storage_reduction": float(total_reduction.mean()),
                "mean_end_to_end_reduction": float(latency_reduction.mean()),
                "mean_task_score_delta": float(score_delta.mean()),
                "active_30_percent_gate": bool(active_reduction.mean() >= 0.30),
                "total_storage_30_percent_gate": bool(total_reduction.mean() >= 0.30),
                "latency_5_percent_gate": bool(latency_reduction.mean() >= 0.05),
                "observed_score_noninferiority_gate": bool(score_delta.mean() >= 0),
            }
        )
    gate = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_retrospective_non_claim_validation",
        "all_rows_complete": True,
        "fullkv_non_degenerate_control": True,
        "mosaickv_three_tier_verified": True,
        "mosaickv_all_physical_slots_valid": True,
        "comparisons": comparisons,
        "claim_eligible": False,
        "claim_ineligibility_reason": (
            "two samples were selected retrospectively because prior LLaVA FullKV scored 1.0; "
            "this validates implementation behavior only and cannot estimate held-out superiority"
        ),
    }
    trials.to_parquet(root / "v36_trials.parquet", index=False)
    summary.to_parquet(root / "v36_summary.parquet", index=False)
    (root / "v36_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    gpu_names: set[str] = set()
    for telemetry_path in sorted((root / "telemetry").glob("*.json")):
        telemetry = json.loads(telemetry_path.read_text())
        after = telemetry.get("after_measurement", {})
        devices = after.get("devices", []) if isinstance(after, dict) else []
        for device in devices:
            if isinstance(device, dict) and device.get("name"):
                gpu_names.add(str(device["name"]))
    hardware_label = ", ".join(sorted(gpu_names)) if gpu_names else "GPU"
    lines = [
        f"# Recommended-sequence {hardware_label} validation",
        "",
        "This is a two-sample retrospective dirty-tree implementation validation, "
        "not paper evidence.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Paired comparisons",
        "",
    ]
    for comparison in comparisons:
        lines.extend(
            (
                f"### {comparison['competitor']}",
                "",
                f"- Active-KV reduction: {comparison['mean_active_reduction']:.2%}",
                f"- Active+residual reduction: {comparison['mean_total_storage_reduction']:.2%}",
                f"- End-to-end reduction: {comparison['mean_end_to_end_reduction']:.2%}",
                f"- Task-score delta: {comparison['mean_task_score_delta']:+.4f}",
                "",
            )
        )
    lines.extend(("## Scientific limitation", "", gate["claim_ineligibility_reason"]))
    (root / "v36_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
