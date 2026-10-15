#!/usr/bin/env python3
"""Analyze fresh-cohort prompt-only versus deterministic draft-gated selection."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, replace
from pathlib import Path

from analyze_output_preserving_capacity_sweep import (
    _parquet_comparison_rows,
    worker,
)
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.evaluation.coco_shared_bytes import (
    STORAGE_FIELDS,
    TokenizedCider,
    bootstrap_cider_curve,
)
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.types import MetricSample


def analyze(args: argparse.Namespace) -> int:
    import pandas as pd

    root = args.root.resolve()
    if digest(root / "plan.json") != args.plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads((root / "plan.json").read_text())
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "failed",
        "classification": plan["classification"],
        "heldout_claim": False,
        "paper_claim_authorized": False,
        "plan_sha256": args.plan_sha256,
        "missing_or_failed": [],
        "byte_gates": [],
        "quality_comparisons": [],
        "cross_curve_30_percent_tests": [],
        "draft_gate_telemetry": [],
    }
    try:
        samples = json.loads(Path(plan["ledger"]).read_text())["samples"]
        refs = {sample["sample_id"]: tuple(sample["references"]) for sample in samples}
        ids = tuple(plan["sample_ids"])
        rows, traces = {}, {}
        for job in plan["jobs"]:
            try:
                result = worker(Path(job["output"]))
            except ValueError as error:
                report["missing_or_failed"].append({"index": job["index"], "error": str(error)})
                continue
            point = job["analysis_label"], float(job["cap_mib"])
            for row in result["record"]["rows"]:
                key = (*point, row["sample_id"])
                if key in rows:
                    raise ValueError(f"duplicate row: {key}")
                rows[key] = dict(row, label=point[0], cap_mib=point[1])
                traces[key] = json.loads(Path(result["traces"][row["sample_id"]]).read_text())
        if report["missing_or_failed"]:
            raise ValueError("incomplete draft sweep; metrics withheld")

        points = sorted({(label, cap) for label, cap, _sid in rows}, key=str)
        for cap in plan["aggressive_caps_mib"]:
            labels = sorted(label for label, value in points if value == cap)
            for sid in ids:
                for field in STORAGE_FIELDS:
                    values = {label: rows[label, cap, sid][field] for label in labels}
                    if len(set(values.values())) != 1:
                        raise ValueError(f"non-identical {field} at {cap}: {values}")
            report["byte_gates"].append(
                {
                    "cap_mib": cap,
                    "passed": True,
                    **{field: rows[labels[0], cap, ids[0]][field] for field in STORAGE_FIELDS},
                }
            )

        groups = {
            point: [MetricSample(sid, rows[*point, sid]["answer"], refs[sid]) for sid in ids]
            for point in points
        }
        definition = replace(
            default_metric_registry().resolve("cider"),
            reducer=TokenizedCider([sample for group in groups.values() for sample in group]),
        )
        aggregates = []
        for (label, cap), group in groups.items():
            measured = [rows[label, cap, sid] for sid in ids]
            aggregates.append(
                {
                    "label": label,
                    "cap_mib": cap,
                    "sample_count": len(ids),
                    "cider": definition.aggregate(group),
                    **{
                        field: statistics.mean(row[field] for row in measured)
                        for field in STORAGE_FIELDS
                    },
                    "compression_time": statistics.mean(
                        row["compression_time"] for row in measured
                    ),
                    "decode_time": statistics.mean(row["decode_time"] for row in measured),
                    "end_to_end_time": statistics.mean(row["end_to_end_time"] for row in measured),
                }
            )

        for cap in plan["aggressive_caps_mib"]:
            for label in ("output_preserving_prompt", "output_preserving_draft_gate"):
                audits = [traces[label, cap, sid]["output_reconstruction"] for sid in ids]
                report["draft_gate_telemetry"].append(
                    {
                        "label": label,
                        "cap_mib": cap,
                        "accepted_swaps_mean": statistics.mean(a["accepted_swaps"] for a in audits),
                        "prompt_error_reduction_mean": statistics.mean(
                            a["initial_error"] - a["final_error"] for a in audits
                        ),
                        "draft_error_reduction_mean": statistics.mean(
                            a["draft_initial_error"] - a["draft_final_error"] for a in audits
                        ),
                        "rejected_validation_pairs_mean": statistics.mean(
                            a["rejected_validation_pairs"] for a in audits
                        ),
                    }
                )

        comparisons = []
        for cap in plan["aggressive_caps_mib"]:
            draft = ("output_preserving_draft_gate", cap)
            comparisons.extend(
                (
                    (("prompt_attention_topk", cap), draft),
                    (("output_preserving_prompt", cap), draft),
                    (("prompt_attention_topk", 40.0), draft),
                )
            )
        paired = bootstrap_cider_curve(
            groups,
            comparisons,
            resamples=plan["bootstrap_resamples"],
            seed=plan["bootstrap_seed"],
            workers=min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), plan["analysis_cpus"]),
        )
        by_point = {(row["label"], row["cap_mib"]): row for row in aggregates}
        for reference, candidate in comparisons:
            value = paired[reference, candidate]
            record = {"reference": reference, "candidate": candidate, **asdict(value)}
            report["quality_comparisons"].append(record)
            if reference == ("prompt_attention_topk", 40.0):
                reductions = {
                    phase: 1 - by_point[candidate][f"{phase}_active_kv_storage_bytes"]
                    / by_point[reference][f"{phase}_active_kv_storage_bytes"]
                    for phase in ("initial", "final", "peak")
                }
                report["cross_curve_30_percent_tests"].append(
                    {
                        **record,
                        "quality_noninferior_zero_margin": value.ci_low >= 0,
                        "memory_reductions": reductions,
                        "passes_all_memory_boundaries": min(reductions.values()) >= 0.30,
                        "passes_joint_gate": value.ci_low >= 0 and min(reductions.values()) >= 0.30,
                    }
                )

        pd.DataFrame(rows.values()).to_parquet(output / "paired_samples.parquet", index=False)
        pd.DataFrame(aggregates).to_parquet(output / "aggregates.parquet", index=False)
        pd.DataFrame(_parquet_comparison_rows(report["quality_comparisons"])).to_parquet(
            output / "quality_comparisons.parquet", index=False
        )
        pd.DataFrame(
            _parquet_comparison_rows(report["cross_curve_30_percent_tests"])
        ).to_parquet(output / "cross_curve_tests.parquet", index=False)
        pd.DataFrame(report["draft_gate_telemetry"]).to_parquet(
            output / "draft_gate_telemetry.parquet", index=False
        )
        report.update(
            status="passed",
            interpretation=(
                "development_joint_gate_passed_not_heldout"
                if any(row["passes_joint_gate"] for row in report["cross_curve_30_percent_tests"])
                else "development_joint_gate_not_demonstrated"
            ),
            aggregates=aggregates,
        )
    except Exception as error:
        report.update(
            status="failed",
            interpretation="analysis_failed_metrics_must_not_be_used",
            error=f"{type(error).__name__}: {error}",
        )
    write_once(output / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    raise SystemExit(analyze(parser.parse_args()))
