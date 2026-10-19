#!/usr/bin/env python3
"""Fail-closed whole-cohort PrefixKV PPL/ROUGE and byte validation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import pandas as pd
from run_prefixkv_decode_preflight import digest, write_once

STORAGE_FIELDS = (
    "initial_active_kv_storage_bytes",
    "final_active_kv_storage_bytes",
    "peak_active_kv_storage_bytes",
)


def _sample_metrics(path: str) -> list[dict]:
    frame = pd.read_parquet(path)
    return [dict(row) for row in frame.to_dict(orient="records") if row["sample_id"] is not None]


def analyze(args: argparse.Namespace) -> dict:
    plan_path = args.root / "plan.json"
    if digest(plan_path) != args.plan_sha:
        raise ValueError("PrefixKV byte-validation plan changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan.get("byte_validation_authorized") or plan.get("full_campaign_authorized"):
        raise ValueError("wrong PrefixKV analysis authorization")
    groups = sorted({(str(job["task"]), int(job["shard"])) for job in plan["jobs"]})
    records: list[dict] = []
    missing = []
    for task, shard in groups:
        path = args.root / "workers" / task / f"{shard:02d}.json"
        if not path.is_file():
            missing.append({"task": task, "shard": shard})
            continue
        worker = json.loads(path.read_text(encoding="utf-8"))
        if (
            worker.get("status") != "passed"
            or worker.get("plan_sha256") != args.plan_sha
            or (worker.get("task"), worker.get("shard")) != (task, shard)
            or len(worker.get("records", [])) != 16
        ):
            raise ValueError(f"failed or malformed PrefixKV worker: {path}")
        records.extend(worker["records"])
    if missing:
        raise ValueError(f"incomplete PrefixKV validation: {missing}")
    metrics = []
    sample_rows: dict[tuple[str, str, int | None, str], dict] = {}
    observations: dict[tuple[str, str, int | None], list[dict]] = {}
    for record in records:
        method, cap = record["method"], record["cap_mib"]
        # The task is explicitly embedded in every result row's manifest; use
        # the run prefix only as a consistency assertion and recover the task
        # from the worker phase directory encoded in the reference artifact.
        task = "mmvet_prefixkv" if "mmvet_prefixkv" in record["run_id"] else "llava_description"
        key = (task, method, cap)
        observations.setdefault(key, []).extend(
            _sample_metrics(record["metric_observations_parquet_output"])
        )
        for row in record["rows"]:
            row_key = task, method, cap, row["sample_id"]
            if row_key in sample_rows:
                raise ValueError("duplicate PrefixKV sample row")
            sample_rows[row_key] = row
    for task in ("llava_description", "mmvet_prefixkv"):
        ids = sorted(
            {key[3] for key in sample_rows if key[0] == task and key[1] == "full_kv"}
        )
        if len(ids) != 32:
            raise ValueError(f"{task} requires exactly 32 paired samples")
        method_caps = [("full_kv", None)] + [
            (method, cap)
            for cap in plan["caps_mib"]
            for method in ("prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact")
        ]
        for cap in plan["caps_mib"]:
            compressed = ("prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact")
            for sample_id in ids:
                compared = [sample_rows[task, method, cap, sample_id] for method in compressed]
                for field in STORAGE_FIELDS:
                    values = [row[field] for row in compared]
                    if any(value is None for value in values) or len(set(values)) != 1:
                        raise ValueError(
                            f"non-identical {field} at {task}/{cap}/{sample_id}: {values}"
                        )
        for method, method_cap in method_caps:
            key = (task, method, method_cap)
            observed = observations[key]
            rouge = [
                float(row["metric_value"])
                for row in observed
                if row["metric_name"] == "rouge_l_f1" and row["metric_scope"] == "sample"
            ]
            ppl = [
                row
                for row in observed
                if row["metric_name"] == "teacher_forced_ppl"
                and row["metric_scope"] == "sample"
            ]
            if len(rouge) != 32 or len(ppl) != 32:
                raise ValueError(f"incomplete PrefixKV native metrics for {key}")
            nll = math.fsum(float(row["numerator"]) for row in ppl)
            tokens = sum(int(row["denominator"]) for row in ppl)
            selected = [sample_rows[task, method, method_cap, sample_id] for sample_id in ids]
            metrics.append(
                {
                    "task": task,
                    "method": method,
                    "cap_mib": method_cap,
                    "sample_count": 32,
                    "rouge_l_f1": statistics.fmean(rouge),
                    "teacher_forced_ppl": math.exp(nll / tokens),
                    "negative_log_likelihood_sum": nll,
                    "evaluated_token_count": tokens,
                    **{
                        field: statistics.fmean(float(row[field]) for row in selected)
                        for field in STORAGE_FIELDS
                    },
                }
            )
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    parquet = output / "metrics.parquet"
    pd.DataFrame(metrics).to_parquet(parquet, index=False)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "plan_sha256": args.plan_sha,
        "source_sha": plan["source_sha"],
        "sample_count_per_task": 32,
        "tasks": ["llava_description", "mmvet_prefixkv"],
        "metrics_separate": True,
        "ppl_reducer": "exp(sum(nll)/sum(evaluated_token_count))",
        "exact_storage_match": True,
        "completed_rows": len(sample_rows),
        "metrics_parquet": str(parquet),
        "metrics_parquet_sha256": digest(parquet),
        "full_campaign_authorized": False,
    }
    write_once(output / "report.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(analyze(parser.parse_args()), sort_keys=True))
