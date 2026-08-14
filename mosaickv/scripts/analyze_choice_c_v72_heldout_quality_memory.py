#!/usr/bin/env python3
"""Analyze the preregistered v72 held-out quality-memory validation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

METHODS = ("prompt_attention_topk", "mosaickv_full")


def _is_missing_error(value: object) -> bool:
    """Normalize JSON nulls, pandas missing scalars, and legacy ``"nan"`` strings."""

    if value is None or (isinstance(value, str) and value.casefold() == "nan"):
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, bool | np.bool_) else False


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 95),
    }


def _bootstrap_mean_difference(
    differences: list[float], *, resamples: int, seed: int
) -> list[float]:
    if not differences:
        raise ValueError("paired bootstrap requires at least one difference")
    generator = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=np.float64)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _load_task(
    output_root: Path,
    task: str,
    expected_samples: int,
    expected_repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = tuple(sorted((output_root / "telemetry").glob(f"v72-{task}-*.json")))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {task} telemetry file, found {len(paths)}")
    telemetry = cast("dict[str, Any]", json.loads(paths[0].read_text()))
    if telemetry.get("status") != "completed":
        raise RuntimeError(f"{task} telemetry is not complete: {telemetry.get('status')}")
    sample_ids = tuple(cast("list[str]", telemetry["sample_ids"]))
    if len(sample_ids) != expected_samples or len(set(sample_ids)) != expected_samples:
        raise RuntimeError(f"{task} telemetry has the wrong held-out sample set")
    if int(telemetry["measured_repetitions_per_method"]) != expected_repetitions:
        raise RuntimeError(f"{task} telemetry has the wrong repetition count")
    trials = cast("list[dict[str, Any]]", telemetry["trials"])
    if len(trials) != expected_repetitions * len(METHODS):
        raise RuntimeError(f"{task} telemetry has an incomplete trial set")

    rows: list[dict[str, Any]] = []
    keys: set[tuple[int, str, str]] = set()
    for trial in trials:
        repetition = int(trial["repetition"])
        method = str(trial["method"])
        if method not in METHODS or not 0 <= repetition < expected_repetitions:
            raise RuntimeError(f"{task} has an unknown method or repetition")
        trial_rows = cast("list[dict[str, Any]]", trial["rows"])
        if {str(row["sample_id"]) for row in trial_rows} != set(sample_ids):
            raise RuntimeError(f"{task} trial does not cover its fixed paired samples")
        for row in trial_rows:
            key = repetition, method, str(row["sample_id"])
            if key in keys:
                raise RuntimeError(f"duplicate v72 measurement row: {task}/{key}")
            keys.add(key)
            if row.get("status") != "completed" or not _is_missing_error(row.get("error")):
                raise RuntimeError(f"failed v72 measurement row: {task}/{key}")
            record = dict(row)
            record.update(
                {
                    "task": task,
                    "method": method,
                    "repetition": repetition,
                    "order": int(trial["order"]),
                    "run_id": str(trial["run_id"]),
                }
            )
            rows.append(record)
    return telemetry, rows


def analyze(spec_path: Path, output_root: Path) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    gate_spec = cast("dict[str, Any]", spec["primary_gate"])
    statistics_spec = cast("dict[str, Any]", spec["statistics"])
    repetitions = int(spec["measured_repetitions_per_method"])
    all_rows: list[dict[str, Any]] = []
    telemetry_by_task: dict[str, dict[str, Any]] = {}
    task_entries = cast("list[dict[str, Any]]", spec["tasks"])
    for entry in task_entries:
        task = str(entry["task"])
        telemetry, rows = _load_task(
            output_root,
            task,
            int(entry["sample_count"]),
            repetitions,
        )
        if telemetry.get("dataset_revision") != entry["dataset_revision"]:
            raise RuntimeError(f"{task} dataset revision differs from the frozen protocol")
        telemetry_by_task[task] = telemetry
        all_rows.extend(rows)

    snapshots = {
        str(cast("dict[str, Any]", item["source_snapshot"])["snapshot_sha256"])
        for item in telemetry_by_task.values()
    }
    if len(snapshots) != 1:
        raise RuntimeError("v72 workers did not execute the same source snapshot")

    frame = pd.DataFrame(all_rows).sort_values(
        ["task", "repetition", "method", "sample_id"]
    )
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(analysis_root / "heldout_trials.parquet", index=False)

    minimum_reduction = float(gate_spec["minimum_active_memory_reduction_fraction"])
    margin = float(gate_spec["task_score_noninferiority_margin"])
    bootstrap_resamples = int(statistics_spec["paired_bootstrap_resamples"])
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
            answer_counts = method_frame.groupby("sample_id")["answer"].nunique()
            score_counts = method_frame.groupby("sample_id")["task_score"].nunique()
            if int(answer_counts.max()) != 1 or int(score_counts.max()) != 1:
                raise RuntimeError(f"{task}/{method} is not deterministic across repetitions")
            sample_frame = (
                method_frame.groupby("sample_id", as_index=False)
                .agg(
                    answer=("answer", "first"),
                    reference=("reference", "first"),
                    task_score=("task_score", "first"),
                    active_kv_bytes=("active_kv_bytes", "mean"),
                    residual_kv_bytes=("residual_kv_bytes", "mean"),
                )
                .sort_values("sample_id")
                .reset_index(drop=True)
            )
            per_sample[method] = sample_frame
            method_summaries[method] = {
                "sample_count": len(sample_frame),
                "mean_task_score": float(sample_frame["task_score"].mean()),
                "mean_active_kv_bytes": float(sample_frame["active_kv_bytes"].mean()),
                "mean_residual_kv_bytes": float(sample_frame["residual_kv_bytes"].mean()),
                "compression_time": _timing_summary(
                    [float(value) for value in method_frame["compression_time"]]
                ),
                "decode_time": _timing_summary(
                    [float(value) for value in method_frame["decode_time"]]
                ),
                "end_to_end_time": _timing_summary(
                    [float(value) for value in method_frame["end_to_end_time"]]
                ),
            }
            summary_rows.append({"task": task, "method": method, **method_summaries[method]})

        prompt = per_sample["prompt_attention_topk"]
        mosaic = per_sample["mosaickv_full"]
        if tuple(prompt["sample_id"]) != tuple(mosaic["sample_id"]):
            raise RuntimeError(f"{task} methods do not cover identical paired samples")
        prompt_bytes = float(prompt["active_kv_bytes"].mean())
        mosaic_bytes = float(mosaic["active_kv_bytes"].mean())
        reduction = 1.0 - mosaic_bytes / prompt_bytes
        score_differences = [
            float(mosaic_value) - float(prompt_value)
            for mosaic_value, prompt_value in zip(
                mosaic["task_score"], prompt["task_score"], strict=True
            )
        ]
        mean_score_difference = statistics.fmean(score_differences)
        score_ci = _bootstrap_mean_difference(
            score_differences,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + task_index,
        )
        answer_agreement = statistics.fmean(
            float(mosaic_answer == prompt_answer)
            for mosaic_answer, prompt_answer in zip(
                mosaic["answer"], prompt["answer"], strict=True
            )
        )
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
        active_pass = reduction >= minimum_reduction
        quality_pass = mean_score_difference >= -margin
        task_results[task] = {
            "dataset_id": entry["dataset_id"],
            "dataset_revision": entry["dataset_revision"],
            "sample_count": len(prompt),
            "methods": method_summaries,
            "paired_answer_agreement": answer_agreement,
            "paired_mean_task_score_difference": mean_score_difference,
            "paired_bootstrap_task_score_difference_95_ci": score_ci,
            "task_score_noninferiority_margin": margin,
            "quality_noninferiority_pass": quality_pass,
            "active_memory_reduction_fraction": reduction,
            "minimum_active_memory_reduction_fraction": minimum_reduction,
            "active_memory_gate_pass": active_pass,
            "quality_memory_gate_pass": bool(active_pass and quality_pass),
            "latency_report": {
                "policy": "informational_not_a_rejection_criterion",
                "paired_end_to_end_ratios": paired_ratios,
                "paired_median_end_to_end_ratio": statistics.median(paired_ratios),
            },
        }

    require_each = bool(gate_spec["require_each_task_to_pass"])
    overall_pass = all(
        bool(result["quality_memory_gate_pass"]) for result in task_results.values()
    )
    if not require_each:
        raise RuntimeError("v72 requires the preregistered per-task gate")
    result = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "classification": spec["classification"],
        "canonical_eligible": all(
            bool(item.get("canonical_eligible", False))
            for item in telemetry_by_task.values()
        ),
        "source_snapshot_sha256": next(iter(snapshots)),
        "latency_policy": "informational_not_a_rejection_criterion",
        "latency_rejection_applied": False,
        "task_results": task_results,
        "quality_memory_gate_pass": overall_pass,
    }
    destination = analysis_root / "heldout_quality_memory_gate.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(summary_rows).to_parquet(
        analysis_root / "heldout_summary.parquet", index=False
    )

    report = [
        "# V72 held-out quality-memory validation",
        "",
        f"- Overall quality-memory gate: **{'PASS' if overall_pass else 'FAIL'}**",
        "- Latency policy: informational; latency cannot reject this run.",
        f"- Canonical eligible: {result['canonical_eligible']}",
        "",
        "| Task | Samples | Active reduction | Score difference | Answer agreement "
        "| Gate | Median latency ratio |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for task, task_result in task_results.items():
        score_difference = float(task_result["paired_mean_task_score_difference"])
        latency_ratio = float(
            cast("dict[str, Any]", task_result["latency_report"])[
                "paired_median_end_to_end_ratio"
            ]
        )
        report.append(
            f"| {task} | {task_result['sample_count']} | "
            f"{float(task_result['active_memory_reduction_fraction']):.2%} | "
            f"{score_difference:+.4f} | "
            f"{float(task_result['paired_answer_agreement']):.2%} | "
            f"{'PASS' if task_result['quality_memory_gate_pass'] else 'FAIL'} | "
            f"{latency_ratio:.3f}x |"
        )
    report.extend(
        [
            "",
            "Task scores are reported per task and are not combined into a macro-average.",
            "This dirty-worktree run is an exploratory validation, not a canonical paper result.",
        ]
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
    print(
        json.dumps(
            {
                "quality_memory_gate_pass": result["quality_memory_gate_pass"],
                "latency_policy": result["latency_policy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
