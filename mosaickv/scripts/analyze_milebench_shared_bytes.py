#!/usr/bin/env python3
"""Validate and reduce a paired representative MileBench shared-byte stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from run_coco_shared_bytes import completed_sample
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.evaluation.coco_shared_bytes import STORAGE_FIELDS, dense_capacity_certificate
from mosaickv.evaluation.milebench import (
    MileBenchQuestionType,
    MileBenchSampleScore,
    aggregate_milebench_samples,
    paired_milebench_bootstrap,
    route_milebench_dataset,
)
from mosaickv.evaluation.shared_byte_contract import digest as object_digest
from mosaickv.evaluation.shared_byte_contract import validate_group


def _collect(plan: dict, plan_sha: str, stage: str):
    plan = dict(plan, _plan_sha=plan_sha)
    rows: dict[tuple[str, int | None, str], dict] = {}
    traces: dict[tuple[str, int | None, str], dict] = {}
    samples: dict[str, dict] = {}
    missing: list[dict] = []
    for index, job in enumerate(plan["jobs"]):
        if job["stage"] != stage:
            continue
        if digest(Path(job["ledger"])) != job["ledger_sha"]:
            raise ValueError("input ledger changed")
        payload = json.loads(Path(job["ledger"]).read_text(encoding="utf-8"))
        for sample in payload["samples"]:
            sample_id = sample["sample_id"]
            if sample_id in samples and samples[sample_id] != sample:
                raise ValueError("paired input mismatch")
            samples[sample_id] = sample
            root = Path(job["output"]) / "samples" / hashlib.sha256(sample_id.encode()).hexdigest()
            record = completed_sample(plan, job, sample_id)
            if record is None:
                missing.append(
                    {
                        "index": index,
                        "sample_id": sample_id,
                        "method": job["method"],
                        "cap_mib": job["cap_mib"],
                        "failed_attempts": len(list(root.glob("attempt-*/failure.json"))),
                    }
                )
                continue
            if record["warmups"] != plan["warmups"] or len(record["rows"]) != 1:
                raise ValueError("measurement protocol or result cardinality mismatch")
            row = record["rows"][0]
            if (
                row["sample_id"] != sample_id
                or row["status"] != "completed"
                or row["error"] is not None
                or row["task_score"] is None
            ):
                raise ValueError("failed or incomplete MileBench row")
            trace = json.loads(Path(record["trace"]).read_text(encoding="utf-8"))
            if (
                trace["status"] != "completed"
                or trace["method"] != job["method"]
                or trace.get("effective_method", job["method"]) != job["method"]
                or trace["prefixkv_decode_events"]
                or trace["repair_events"]
            ):
                raise ValueError("trace substitution, eviction, or repair detected")
            for name in (
                "hierarchical_logit_safe_eviction",
                "progressive_logit_safe_eviction",
                "short_horizon_logit_stability",
            ):
                if trace.get(name, {}).get("used_fullkv_fallback"):
                    raise ValueError("FullKV fallback")
            if any(trace["cache_lifecycle"][field] != row[field] for field in STORAGE_FIELDS):
                raise ValueError("cache trace/raw row mismatch")
            route = route_milebench_dataset(
                sample["milebench_dataset"], sample["milebench_question_type"]
            )
            key = job["method"], job["cap_mib"], sample_id
            if key in rows:
                raise ValueError("duplicate method/cap/sample")
            rows[key] = {
                **row,
                "method": job["method"],
                "cap_mib": job["cap_mib"],
                "lane": "lookm",
                "task": "milebench",
                "metric_name": route.metric.value,
                "result_category": "unified_comparison",
                "control_hashes": record["control_hashes"],
                "physical_byte_cap": (
                    None if job["cap_mib"] is None else job["cap_mib"] * 2**20
                ),
                "decode_evictions": 0,
                "used_fullkv_fallback": False,
                "source_sha": record["source_sha"],
                "config_sha": record["config_sha"],
                "trace": record["trace"],
                "label": "reimpl" if job["method"].endswith("_reimpl") else "local",
            }
            traces[key] = trace
    return rows, traces, samples, missing


def _score(rows: dict, samples: dict, method: str, cap: int | None, ids: list[str]):
    scored = []
    for sample_id in ids:
        sample = samples[sample_id]
        route = route_milebench_dataset(
            sample["milebench_dataset"], sample["milebench_question_type"]
        )
        scored.append(
            MileBenchSampleScore(
                sample_id=sample_id,
                dataset=sample["milebench_dataset"],
                question_type=MileBenchQuestionType(sample["milebench_question_type"]),
                metric=route.metric,
                value=float(rows[method, cap, sample_id]["task_score"]),
            )
        )
    return tuple(scored)


def analyze(args) -> None:
    import pandas as pd

    plan_path = args.root / "plan.json"
    if digest(plan_path) != args.plan_sha:
        raise ValueError("plan hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["task"] != "milebench":
        raise ValueError("not a MileBench plan")
    output = args.output
    if output.exists():
        raise FileExistsError("analysis is write-once")
    report = {
        "status": "failed",
        "plan_sha256": args.plan_sha,
        "stage": args.stage,
        "task": "milebench",
        "result_category": "unified_comparison",
        "official_lookm_execution": False,
        "methods": plan["methods"],
        "gates": [],
        "comparisons": [],
    }
    try:
        rows, traces, samples, missing = _collect(plan, args.plan_sha, args.stage)
        report["missing_or_failed"] = missing
        report["retry_indices"] = sorted({item["index"] for item in missing})
        if missing:
            raise ValueError("incomplete paired cohort; metrics intentionally withheld")
        ids = sorted(samples)
        expected_count = 32 if args.stage == "validation" else 256
        if len(ids) != expected_count:
            raise ValueError("wrong preregistered stage denominator")
        caps = [40] if args.stage == "validation" else plan["caps_mib"]
        methods = tuple(plan["methods"])
        full = [rows["full_kv", None, sample_id] for sample_id in ids]
        expected_controls = {
            sample_id: rows["full_kv", None, sample_id]["control_hashes"] for sample_id in ids
        }
        for cap in caps:
            certificates = {}
            for method in (name for name in methods if name != "full_kv"):
                for sample_id in ids:
                    key = method, cap, sample_id
                    source_layout = traces["full_kv", None, sample_id]["source_kv_layout"]
                    if source_layout != traces[key]["source_kv_layout"]:
                        raise ValueError("source geometry differs across methods")
                    certificate = dense_capacity_certificate(source_layout, cap * 2**20)
                    certificates[method, sample_id] = certificate
                    rows[key].update(
                        largest_feasible_initial_bytes=certificate[
                            "largest_feasible_initial_bytes"
                        ],
                        capacity_certificate_sha256=object_digest(certificate),
                        target_shortfall_bytes=certificate["target_shortfall_bytes"],
                    )
            for metric in ("accuracy", "rouge_l_f1"):
                metric_ids = [
                    sample_id
                    for sample_id in ids
                    if route_milebench_dataset(
                        samples[sample_id]["milebench_dataset"],
                        samples[sample_id]["milebench_question_type"],
                    ).metric.value
                    == metric
                ]
                group = [row for row in full if row["sample_id"] in metric_ids]
                group += [
                    rows[method, cap, sample_id]
                    for method in methods
                    if method != "full_kv"
                    for sample_id in metric_ids
                ]
                gate = validate_group(
                    group,
                    sample_ids=metric_ids,
                    methods=methods,
                    cap_bytes=cap * 2**20,
                    metric=metric,
                    expected_controls={key: expected_controls[key] for key in metric_ids},
                    capacity_certificates={
                        key: value
                        for key, value in certificates.items()
                        if key[1] in metric_ids
                    },
                )
                report["gates"].append({"cap_mib": cap, "metric": metric, **gate})
        aggregates = []
        for cap in caps:
            for method in methods:
                method_cap = None if method == "full_kv" else cap
                scores = _score(rows, samples, method, method_cap, ids)
                summary = aggregate_milebench_samples(scores)
                measured = [rows[method, method_cap, sample_id] for sample_id in ids]
                for metric_name, value in summary.items():
                    aggregates.append(
                        {
                            "method": method,
                            "cap_mib": method_cap,
                            "metric_name": f"milebench/{metric_name}",
                            "metric_value": value,
                            "sample_count": len(ids),
                            "label": "reimpl" if method.endswith("_reimpl") else "local",
                            **{
                                field: statistics.mean(row[field] for row in measured)
                                for field in STORAGE_FIELDS
                            },
                            "end_to_end_time": (
                                None
                                if method == "full_kv"
                                else statistics.mean(row["end_to_end_time"] for row in measured)
                            ),
                        }
                    )
                if method != "mosaickv_exact":
                    comparisons = paired_milebench_bootstrap(
                        scores,
                        _score(rows, samples, "mosaickv_exact", cap, ids),
                        resamples=plan["bootstrap_resamples"],
                        seed=plan["bootstrap_seed"],
                    )
                    report["comparisons"].extend(
                        {
                            "cap_mib": cap,
                            "baseline": method,
                            "candidate": "mosaickv_exact",
                            "hierarchy_key": key,
                            **asdict(value),
                        }
                        for key, value in comparisons.items()
                    )
        output.mkdir(parents=True)
        aggregate_path = output / "aggregates.parquet"
        row_path = output / "derived_rows.parquet"
        pd.DataFrame(aggregates).to_parquet(aggregate_path, index=False)
        pd.DataFrame(rows.values()).to_parquet(row_path, index=False)
        report.update(
            status="passed",
            sample_count=len(ids),
            completed_rows=len(rows),
            metrics_separate=True,
            fullkv_latency_comparison=False,
            derived_artifact_hashes={
                str(aggregate_path): digest(aggregate_path),
                str(row_path): digest(row_path),
            },
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        output.mkdir(parents=True, exist_ok=True)
        write_once(output / "report.json", report)
        raise
    write_once(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--stage", choices=("validation", "larger"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())
