#!/usr/bin/env python3
"""Analyze the clean v73 FullKV, Prompt Top-K, and MosaicKV validation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml
from analyze_choice_c_v72_heldout_quality_memory import (
    _bootstrap_mean_difference,
    _is_missing_error,
    _timing_summary,
)

METHODS = ("full_kv", "prompt_attention_topk", "mosaickv_full")


def _load_task(
    output_root: Path,
    task: str,
    expected_samples: int,
    expected_repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = tuple(sorted((output_root / "telemetry").glob(f"v73-{task}-*.json")))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {task} telemetry file, found {len(paths)}")
    telemetry = cast("dict[str, Any]", json.loads(paths[0].read_text()))
    if telemetry.get("status") != "completed" or telemetry.get("canonical_eligible") is not True:
        raise RuntimeError(f"{task} telemetry is incomplete or noncanonical")
    sample_ids = tuple(cast("list[str]", telemetry["sample_ids"]))
    if len(sample_ids) != expected_samples or len(set(sample_ids)) != expected_samples:
        raise RuntimeError(f"{task} telemetry has the wrong held-out sample set")
    trials = cast("list[dict[str, Any]]", telemetry["trials"])
    if len(trials) != expected_repetitions * len(METHODS):
        raise RuntimeError(f"{task} telemetry has an incomplete measured trial set")

    rows: list[dict[str, Any]] = []
    keys: set[tuple[int, str, str]] = set()
    for trial in trials:
        repetition = int(trial["repetition"])
        method = str(trial["method"])
        trial_rows = cast("list[dict[str, Any]]", trial["rows"])
        if method not in METHODS or {str(row["sample_id"]) for row in trial_rows} != set(
            sample_ids
        ):
            raise RuntimeError(f"{task} contains an unknown or incomplete trial")
        for row in trial_rows:
            key = repetition, method, str(row["sample_id"])
            if key in keys:
                raise RuntimeError(f"duplicate v73 row: {task}/{key}")
            keys.add(key)
            if row.get("status") != "completed" or not _is_missing_error(row.get("error")):
                raise RuntimeError(f"failed v73 row: {task}/{key}")
            rows.append(
                {
                    **row,
                    "task": task,
                    "method": method,
                    "repetition": repetition,
                    "order": int(trial["order"]),
                    "run_id": str(trial["run_id"]),
                }
            )
    return telemetry, rows


def _fullkv_is_transformation_free(output_root: Path, task: str, sample_ids: set[str]) -> bool:
    traces = []
    for path in sorted((output_root / "traces").glob(f"v73-{task}-trial-*-full_kv/*.json")):
        record = json.loads(path.read_text())
        traces.append(record)
    return bool(traces) and {str(record["sample_id"]) for record in traces} == sample_ids and all(
        record.get("effective_method") == "full_kv"
        and record.get("tier_mode") == "full_cache"
        and int(record.get("residual_kv_bytes", 0)) == 0
        for record in traces
    )


def analyze(spec_path: Path, output_root: Path) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    gate_spec = cast("dict[str, Any]", spec["primary_gate"])
    statistics_spec = cast("dict[str, Any]", spec["statistics"])
    repetitions = int(spec["measured_repetitions_per_method"])
    task_entries = cast("list[dict[str, Any]]", spec["tasks"])
    telemetry_by_task: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for entry in task_entries:
        task = str(entry["task"])
        telemetry, rows = _load_task(
            output_root, task, int(entry["sample_count"]), repetitions
        )
        if telemetry.get("dataset_revision") != entry["dataset_revision"]:
            raise RuntimeError(f"{task} dataset revision differs from the protocol")
        telemetry_by_task[task] = telemetry
        all_rows.extend(rows)
    snapshots = {
        str(cast("dict[str, Any]", item["source_snapshot"])["snapshot_sha256"])
        for item in telemetry_by_task.values()
    }
    if len(snapshots) != 1:
        raise RuntimeError("v73 workers used different source snapshots")

    frame = pd.DataFrame(all_rows).sort_values(
        ["task", "repetition", "method", "sample_id"]
    )
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(analysis_root / "heldout_trials.parquet", index=False)

    minimum_reduction = float(gate_spec["minimum_active_memory_reduction_fraction"])
    margin = float(gate_spec["task_score_noninferiority_margin"])
    resamples = int(statistics_spec["paired_bootstrap_resamples"])
    bootstrap_seed = int(statistics_spec["paired_bootstrap_seed"])
    task_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for task_index, entry in enumerate(task_entries):
        task = str(entry["task"])
        task_frame = frame[frame["task"] == task]
        per_sample: dict[str, pd.DataFrame] = {}
        method_summaries: dict[str, Any] = {}
        for method in METHODS:
            method_frame = task_frame[task_frame["method"] == method]
            if (
                int(method_frame.groupby("sample_id")["answer"].nunique().max()) != 1
                or int(method_frame.groupby("sample_id")["task_score"].nunique().max()) != 1
            ):
                raise RuntimeError(f"{task}/{method} is nondeterministic")
            sample_frame = (
                method_frame.groupby("sample_id", as_index=False)
                .agg(
                    answer=("answer", "first"),
                    task_score=("task_score", "first"),
                    active_kv_bytes=("active_kv_bytes", "mean"),
                    residual_kv_bytes=("residual_kv_bytes", "mean"),
                )
                .sort_values("sample_id")
                .reset_index(drop=True)
            )
            per_sample[method] = sample_frame
            timings = {
                name: _timing_summary([float(value) for value in method_frame[name]])
                for name in ("compression_time", "decode_time", "end_to_end_time")
            }
            method_summaries[method] = {
                "sample_count": len(sample_frame),
                "mean_task_score": float(sample_frame["task_score"].mean()),
                "mean_active_kv_bytes": float(sample_frame["active_kv_bytes"].mean()),
                "mean_residual_kv_bytes": float(sample_frame["residual_kv_bytes"].mean()),
                **timings,
            }
            summary_rows.append(
                {
                    "task": task,
                    "method": method,
                    "sample_count": len(sample_frame),
                    "mean_task_score": float(sample_frame["task_score"].mean()),
                    "mean_active_kv_bytes": float(sample_frame["active_kv_bytes"].mean()),
                    "mean_residual_kv_bytes": float(sample_frame["residual_kv_bytes"].mean()),
                    **{
                        f"{name}_{statistic}": value
                        for name, values in timings.items()
                        for statistic, value in values.items()
                    },
                }
            )

        sample_ids = tuple(per_sample["mosaickv_full"]["sample_id"])
        if any(tuple(per_sample[method]["sample_id"]) != sample_ids for method in METHODS):
            raise RuntimeError(f"{task} methods do not cover identical paired samples")
        mosaic = per_sample["mosaickv_full"]
        prompt = per_sample["prompt_attention_topk"]
        active_reduction = 1.0 - float(mosaic["active_kv_bytes"].mean()) / float(
            prompt["active_kv_bytes"].mean()
        )
        total_reduction = 1.0 - float(
            (mosaic["active_kv_bytes"] + mosaic["residual_kv_bytes"]).mean()
        ) / float(prompt["active_kv_bytes"].mean())
        comparisons: dict[str, Any] = {}
        quality_pass = True
        for comparator_index, comparator in enumerate(("full_kv", "prompt_attention_topk")):
            baseline = per_sample[comparator]
            differences = [
                float(mosaic_value) - float(baseline_value)
                for mosaic_value, baseline_value in zip(
                    mosaic["task_score"], baseline["task_score"], strict=True
                )
            ]
            ci = _bootstrap_mean_difference(
                differences,
                resamples=resamples,
                seed=bootstrap_seed + task_index * 2 + comparator_index,
            )
            mean_difference = statistics.fmean(differences)
            mean_pass = mean_difference >= 0.0
            ci_pass = ci[0] >= -margin
            comparison_pass = bool(mean_pass and ci_pass)
            quality_pass = quality_pass and comparison_pass
            comparisons[comparator] = {
                "paired_mean_task_score_difference": mean_difference,
                "paired_bootstrap_task_score_difference_95_ci": ci,
                "paired_answer_agreement": statistics.fmean(
                    mosaic_answer == baseline_answer
                    for mosaic_answer, baseline_answer in zip(
                        mosaic["answer"], baseline["answer"], strict=True
                    )
                ),
                "nonnegative_mean_pass": mean_pass,
                "noninferiority_ci_pass": ci_pass,
                "quality_pass": comparison_pass,
            }
        fullkv_safe = _fullkv_is_transformation_free(output_root, task, set(sample_ids))
        memory_pass = active_reduction >= minimum_reduction
        task_pass = bool(memory_pass and quality_pass and fullkv_safe)
        paired_ratios = []
        for repetition in range(repetitions):
            repetition_frame = task_frame[task_frame["repetition"] == repetition]
            prompt_latency = float(
                repetition_frame[
                    repetition_frame["method"] == "prompt_attention_topk"
                ]["end_to_end_time"].mean()
            )
            mosaic_latency = float(
                repetition_frame[repetition_frame["method"] == "mosaickv_full"][
                    "end_to_end_time"
                ].mean()
            )
            paired_ratios.append(mosaic_latency / prompt_latency)
        task_results[task] = {
            "dataset_id": entry["dataset_id"],
            "dataset_revision": entry["dataset_revision"],
            "sample_count": len(sample_ids),
            "methods": method_summaries,
            "quality_comparisons": comparisons,
            "active_memory_reduction_fraction_vs_prompt_topk": active_reduction,
            "active_plus_residual_reduction_fraction_vs_prompt_topk": total_reduction,
            "active_memory_gate_pass": memory_pass,
            "fullkv_transformation_free": fullkv_safe,
            "quality_memory_gate_pass": task_pass,
            "latency_report": {
                "policy": "informational_not_a_rejection_criterion",
                "paired_mosaic_prompt_end_to_end_ratios": paired_ratios,
                "paired_median_mosaic_prompt_end_to_end_ratio": statistics.median(
                    paired_ratios
                ),
            },
        }

    overall_pass = all(
        bool(task_result["quality_memory_gate_pass"])
        for task_result in task_results.values()
    )
    result = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "classification": spec["classification"],
        "canonical_eligible": True,
        "source_snapshot_sha256": next(iter(snapshots)),
        "latency_policy": "informational_not_a_rejection_criterion",
        "latency_rejection_applied": False,
        "task_results": task_results,
        "quality_memory_gate_pass": overall_pass,
    }
    (analysis_root / "heldout_quality_memory_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(summary_rows).to_parquet(
        analysis_root / "heldout_summary.parquet", index=False
    )
    report = [
        "# V73 clean held-out quality-memory validation",
        "",
        f"- Overall gate: **{'PASS' if overall_pass else 'FAIL'}**",
        "- Latency is informational and cannot reject the run.",
        "- Task scores are never combined across incompatible benchmarks.",
        "",
        "| Task | Samples | Active reduction | FullKV quality | Prompt quality | Gate |",
        "|---|---:|---:|:---:|:---:|:---:|",
    ]
    for task, task_result in task_results.items():
        comparisons = cast("dict[str, dict[str, Any]]", task_result["quality_comparisons"])
        report.append(
            f"| {task} | {task_result['sample_count']} | "
            f"{float(task_result['active_memory_reduction_fraction_vs_prompt_topk']):.2%} | "
            f"{'PASS' if comparisons['full_kv']['quality_pass'] else 'FAIL'} | "
            f"{'PASS' if comparisons['prompt_attention_topk']['quality_pass'] else 'FAIL'} | "
            f"{'PASS' if task_result['quality_memory_gate_pass'] else 'FAIL'} |"
        )
    (analysis_root / "heldout_quality_memory_report.md").write_text(
        "\n".join(report) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.spec.resolve(), args.output_root.resolve())
    print(json.dumps({"quality_memory_gate_pass": result["quality_memory_gate_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
