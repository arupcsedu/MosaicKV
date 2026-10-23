#!/usr/bin/env python3
"""Metric-aware aggregation for the complete PrefixKV campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from run_prefixkv_decode_preflight import digest, write_once

STORAGE = (
    "initial_active_kv_storage_bytes",
    "final_active_kv_storage_bytes",
    "peak_active_kv_storage_bytes",
)
EXPECTED_COUNTS = {"llava_description": 1000, "mmvet_prefixkv": 218}


def _observations(record: dict) -> pd.DataFrame:
    return pd.read_parquet(record["metric_observations_parquet_output"])


def _paired_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    ref = reference.sort_values("sample_id").reset_index(drop=True)
    cand = candidate.sort_values("sample_id").reset_index(drop=True)
    if list(ref.sample_id) != list(cand.sample_id):
        raise ValueError("paired bootstrap requires identical sample IDs")
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    for offset in range(0, replicates, 256):
        count = min(256, replicates - offset)
        indices = rng.integers(0, len(ref), size=(count, len(ref)))
        if metric == "rouge_l_f1":
            candidate_values = cand.metric_value.to_numpy()[indices].mean(axis=1)
            reference_values = ref.metric_value.to_numpy()[indices].mean(axis=1)
            values = candidate_values - reference_values
        elif metric == "teacher_forced_ppl":
            ref_nll = ref.numerator.to_numpy(dtype=float)[indices].sum(axis=1)
            ref_tok = ref.denominator.to_numpy(dtype=float)[indices].sum(axis=1)
            can_nll = cand.numerator.to_numpy(dtype=float)[indices].sum(axis=1)
            can_tok = cand.denominator.to_numpy(dtype=float)[indices].sum(axis=1)
            values = np.exp(can_nll / can_tok) - np.exp(ref_nll / ref_tok)
        else:
            raise ValueError(f"unsupported PrefixKV bootstrap metric: {metric}")
        diffs.append(values)
    values = np.concatenate(diffs)
    observed = _metric_value(cand, metric)[0] - _metric_value(ref, metric)[0]
    return (
        observed,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _metric_value(frame: pd.DataFrame, metric: str) -> tuple[float, float, int]:
    if metric == "rouge_l_f1":
        return float(frame.metric_value.mean()), float("nan"), len(frame)
    nll = math.fsum(float(value) for value in frame.numerator)
    tokens = sum(int(value) for value in frame.denominator)
    return math.exp(nll / tokens), nll, tokens


def check_shared_storage(raw: pd.DataFrame, expected_counts: dict[str, int]) -> bool:
    """Only compressed methods share a budget; FullKV is a quality reference."""
    methods = {"prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact"}
    for task, count in expected_counts.items():
        shared = raw[
            (raw.task == task) & (raw.lane == "shared_byte_40mib") & (raw.method != "full_kv")
        ]
        if len(shared) != count * len(methods) or shared.sample_id.nunique() != count:
            return False
        for _sample_id, rows in shared.groupby("sample_id"):
            if len(rows) != len(methods) or set(rows.method) != methods:
                return False
            for field in STORAGE:
                if rows[field].isna().any() or rows[field].nunique() != 1:
                    return False
            if not (rows.initial_active_kv_storage_bytes == 40 * 2**20).all():
                return False
    return True


def analyze(args: argparse.Namespace) -> dict:
    plan_path = args.root / "plan.json"
    if digest(plan_path) != args.plan_sha:
        raise ValueError("PrefixKV campaign plan changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    permit = json.loads(args.authorization.read_text(encoding="utf-8"))
    if (
        permit.get("status") != "authorized"
        or permit.get("plan_sha256") != args.plan_sha
        or permit.get("full_campaign_authorized") is not True
    ):
        raise ValueError("full PrefixKV campaign was not authorized")

    group_ids = sorted({int(job["group_index"]) for job in plan["jobs"]})
    if group_ids != list(range(153)):
        raise ValueError("PrefixKV campaign plan is incomplete")
    records = []
    missing = []
    worker_seconds = []
    for group in group_ids:
        jobs = [job for job in plan["jobs"] if int(job["group_index"]) == group]
        task, shard = jobs[0]["task"], int(jobs[0]["shard"])
        path = args.root / "workers" / task / f"{shard:03d}.json"
        if not path.is_file():
            missing.append(group)
            continue
        worker = json.loads(path.read_text(encoding="utf-8"))
        if (
            worker.get("status") != "passed"
            or worker.get("plan_sha256") != args.plan_sha
            or worker.get("group_index") != group
            or len(worker.get("records", [])) != 7
        ):
            raise ValueError(f"malformed PrefixKV campaign worker: {path}")
        records.extend(worker["records"])
        worker_seconds.append(float(worker["worker_wall_seconds"]))
    if missing:
        raise ValueError(f"PrefixKV campaign is missing groups: {missing}")

    sample_rows = []
    metric_frames: dict[tuple[str, str, str], list[pd.DataFrame]] = {}
    for record in records:
        task = "mmvet_prefixkv" if "mmvet_prefixkv" in record["run_id"] else "llava_description"
        lane, method = record["lane"], record["method"]
        observations = _observations(record)
        observations = observations[observations.sample_id.notna()].copy()
        observations["task"] = task
        observations["lane"] = lane
        observations["method"] = method
        metric_frames.setdefault((task, lane, method), []).append(observations)
        for row in record["rows"]:
            item = dict(row)
            item.update(task=task, lane=lane, method=method)
            sample_rows.append(item)
    raw = pd.DataFrame(sample_rows)
    if raw.duplicated(["task", "lane", "method", "sample_id"]).any():
        raise ValueError("duplicate PrefixKV campaign sample rows")
    failures = raw[raw.error.notna()] if "error" in raw.columns else raw.iloc[0:0]
    expected_rows = sum(count * 7 for count in EXPECTED_COUNTS.values())
    complete = len(raw) == expected_rows and failures.empty

    combined = {key: pd.concat(value, ignore_index=True) for key, value in metric_frames.items()}
    # Every lane has its own precision-matched FullKV execution and reference.
    summaries = []
    comparisons = []
    for task, expected_count in EXPECTED_COUNTS.items():
        for lane, methods in {
            "paper_native": ("prefixkv_reimpl", "mosaickv_exact"),
            "shared_byte_40mib": (
                "prompt_attention_topk",
                "prefixkv_reimpl",
                "mosaickv_exact",
            ),
        }.items():
            keys = [(lane, "full_kv")] + [(lane, method) for method in methods]
            for actual_lane, method in keys:
                frame = combined[(task, actual_lane, method)]
                row_slice = raw[
                    (raw.task == task) & (raw.lane == actual_lane) & (raw.method == method)
                ]
                for metric in ("teacher_forced_ppl", "rouge_l_f1"):
                    metric_frame = frame[
                        (frame.metric_name == metric) & (frame.metric_scope == "sample")
                    ]
                    if len(metric_frame) != expected_count:
                        raise ValueError(f"incomplete {task}/{actual_lane}/{method}/{metric}")
                    value, nll, tokens = _metric_value(metric_frame, metric)
                    summaries.append(
                        {
                            "task": task,
                            "lane": lane,
                            "method": method,
                            "metric_name": metric,
                            "metric_value": value,
                            "negative_log_likelihood_sum": nll,
                            "evaluated_token_count": tokens,
                            "sample_count": expected_count,
                            **{field: float(row_slice[field].mean()) for field in STORAGE},
                            "mean_end_to_end_seconds": (
                                None
                                if method == "full_kv"
                                else float(row_slice.end_to_end_time.mean())
                            ),
                        }
                    )
            pairs = [("mosaickv_exact", method) for method in methods if method != "mosaickv_exact"]
            pairs += [(method, "full_kv") for method in methods]
            for candidate, reference in pairs:
                candidate_lane = lane
                reference_lane = lane
                for metric in ("teacher_forced_ppl", "rouge_l_f1"):
                    cand = combined[(task, candidate_lane, candidate)]
                    ref = combined[(task, reference_lane, reference)]
                    cand = cand[(cand.metric_name == metric) & (cand.metric_scope == "sample")]
                    ref = ref[(ref.metric_name == metric) & (ref.metric_scope == "sample")]
                    mean, low, high = _paired_bootstrap(
                        ref,
                        cand,
                        metric=metric,
                        replicates=int(plan["bootstrap_replicates"]),
                        seed=3,
                    )
                    comparisons.append(
                        {
                            "task": task,
                            "lane": lane,
                            "metric_name": metric,
                            "candidate": candidate,
                            "reference": reference,
                            "difference_candidate_minus_reference": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "bootstrap_replicates": plan["bootstrap_replicates"],
                        }
                    )

    # Fail closed on the shared physical-storage contract at every sample.
    exact_storage = check_shared_storage(raw, EXPECTED_COUNTS)

    args.output.mkdir(parents=True, exist_ok=False)
    raw_path = args.output / "sample_rows.parquet"
    summary_path = args.output / "metrics.parquet"
    comparison_path = args.output / "paired_comparisons.parquet"
    raw.to_parquet(raw_path, index=False)
    pd.DataFrame(summaries).to_parquet(summary_path, index=False)
    pd.DataFrame(comparisons).to_parquet(comparison_path, index=False)

    table = pd.DataFrame(summaries).pivot_table(
        index=["task", "lane", "method"], columns="metric_name", values="metric_value"
    )
    (args.output / "prefixkv_metrics.tex").write_text(
        table.to_latex(float_format="%.4f", na_rep="--"), encoding="utf-8"
    )
    storage_table = pd.DataFrame(summaries).drop_duplicates(["task", "lane", "method"])
    storage_table[["task", "lane", "method", *STORAGE, "mean_end_to_end_seconds"]].to_latex(
        args.output / "prefixkv_storage.tex", index=False, float_format="%.2f"
    )

    # Plotting is derived output and must not make completed inference unusable
    # when an optional matplotlib wheel is absent from the common environment.
    figure_path = args.output / "prefixkv_quality_memory.tex"
    lines = [
        r"\documentclass{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[xlabel={Peak live KV (MiB)},ylabel={ROUGE-L},legend pos=outer north east]",
    ]
    for row in summaries:
        if (
            row["lane"] != "shared_byte_40mib"
            or row["method"] == "full_kv"
            or row["metric_name"] != "rouge_l_f1"
        ):
            continue
        x = row["peak_active_kv_storage_bytes"] / 2**20
        y = row["metric_value"]
        label = (row["task"] + "/" + row["method"]).replace("_", r"\_")
        lines.extend(
            [
                f"\\addplot+[only marks] coordinates {{({x:.9g},{y:.9g})}};",
                f"\\addlegendentry{{{label}}}",
            ]
        )
    lines.extend([r"\end{axis}", r"\end{tikzpicture}", r"\end{document}"])
    figure_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = "passed" if complete and exact_storage else "failed"
    payload = {
        "schema_version": 1,
        "status": status,
        "plan_sha256": args.plan_sha,
        "execution_source_sha": plan["execution_source_sha"],
        "candidate_algorithm_sha": plan["candidate_algorithm_sha"],
        "sample_counts": EXPECTED_COUNTS,
        "completed_rows": len(raw),
        "expected_rows": expected_rows,
        "failed_rows": len(failures),
        "exact_shared_storage_match": bool(exact_storage),
        "ppl_and_rouge_separate": True,
        "fullkv_excluded_from_latency_comparison": True,
        "analysis_script_sha256": digest(Path(__file__)),
        "input_plan_sha256": digest(plan_path),
        "plot_format": "tikz_source_no_matplotlib_required",
        "gate_scope": "execution_completeness_and_compressed_storage_only",
        "paper_superiority_claim_established": False,
        "mean_worker_wall_seconds": statistics.fmean(worker_seconds),
        "max_worker_wall_seconds": max(worker_seconds),
        "artifacts": {
            path.name: {"path": str(path), "sha256": digest(path)}
            for path in (raw_path, summary_path, comparison_path, figure_path)
        },
    }
    write_once(args.output / "report.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())
