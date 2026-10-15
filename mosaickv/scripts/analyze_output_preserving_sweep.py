#!/usr/bin/env python3
"""Analyze exact-byte COCO development curves and the 30% cross-curve gate."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import TypeAlias

from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.evaluation.coco_shared_bytes import (
    STORAGE_FIELDS,
    TokenizedCider,
    bootstrap_cider_curve,
)
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.types import MetricSample

Point: TypeAlias = tuple[str, float]
Comparison: TypeAlias = tuple[Point, Point]


def worker(path: Path) -> dict:
    attempts = sorted(path.glob("attempt-*/worker.json"))
    passed = [json.loads(item.read_text()) for item in attempts
              if json.loads(item.read_text()).get("status") == "passed"]
    if len(passed) != 1:
        raise ValueError(f"expected one passed worker in {path}, found {len(passed)}")
    return passed[0]


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
    }
    try:
        samples = json.loads(Path(plan["ledger"]).read_text())["samples"]
        refs = {sample["sample_id"]: tuple(sample["references"]) for sample in samples}
        rows, traces = {}, {}
        for job in plan["jobs"]:
            try:
                result = worker(Path(job["output"]))
            except ValueError as error:
                report["missing_or_failed"].append({"index": job["index"], "error": str(error)})
                continue
            for row in result["record"]["rows"]:
                key = job["label"], job["cap_mib"], row["sample_id"]
                if key in rows:
                    raise ValueError("duplicate result row")
                trace = json.loads(Path(result["traces"][row["sample_id"]]).read_text())
                rows[key] = dict(row, label=job["label"], cap_mib=job["cap_mib"])
                traces[key] = trace
        if report["missing_or_failed"]:
            raise ValueError("incomplete sweep; metrics withheld")
        ids = tuple(plan["sample_ids"])
        labels = [
            "prompt_attention_topk", "vl_cache_reimpl", "frozen_mosaickv_exact",
            "output_preserving_anchor", "output_preserving_pooled",
            "output_preserving_individual",
        ]
        for cap in plan["caps_mib"]:
            for sid in ids:
                measured = {label: rows[label, cap, sid] for label in labels}
                for field in STORAGE_FIELDS:
                    values = {label: measured[label][field] for label in labels}
                    if len(set(values.values())) != 1:
                        raise ValueError(f"non-identical realized {field} at cap {cap}: {values}")
                actual = measured[labels[0]]["initial_active_kv_storage_bytes"]
                if actual > round(cap * 2**20):
                    raise ValueError("realized initial storage exceeds the physical cap")
            report["byte_gates"].append(
                {
                    "cap_mib": cap,
                    "passed": True,
                    **{
                        field: rows[labels[0], cap, ids[0]][field]
                        for field in STORAGE_FIELDS
                    },
                }
            )
        groups = {
            (label, cap): [
                MetricSample(sid, rows[label, cap, sid]["answer"], refs[sid]) for sid in ids
            ]
            for label, cap, _sid in rows
            if label in labels
        }
        union = [sample for group in groups.values() for sample in group]
        definition = replace(
            default_metric_registry().resolve("cider"), reducer=TokenizedCider(union)
        )
        aggregates = []
        for (label, cap), group in sorted(groups.items(), key=str):
            measured = [rows[label, cap, sid] for sid in ids]
            aggregates.append(
                {
                    "label": label,
                    "cap_mib": cap,
                    "sample_count": len(ids),
                    "cider": definition.aggregate(group),
                    **{field: statistics.mean(row[field] for row in measured)
                       for field in STORAGE_FIELDS},
                    "compression_time": statistics.mean(r["compression_time"] for r in measured),
                    "decode_time": statistics.mean(r["decode_time"] for r in measured),
                    "end_to_end_time": statistics.mean(r["end_to_end_time"] for r in measured),
                }
            )
        by_point = {(r["label"], r["cap_mib"]): r for r in aggregates}
        candidate = "output_preserving_individual"
        comparisons: list[Comparison] = []
        for cap in plan["caps_mib"]:
            for baseline in labels[:-1]:
                comparisons.append(((baseline, cap), (candidate, cap)))
        for candidate_cap in plan["caps_mib"]:
            for baseline in (
                "prompt_attention_topk",
                "vl_cache_reimpl",
                "frozen_mosaickv_exact",
            ):
                for baseline_cap in plan["caps_mib"]:
                    comparisons.append(((baseline, baseline_cap), (candidate, candidate_cap)))
        bootstrap_workers = min(
            int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            int(plan.get("analysis_cpus", 1)),
        )
        paired_results = bootstrap_cider_curve(
            groups,
            comparisons,
            resamples=plan["bootstrap_resamples"],
            seed=plan["bootstrap_seed"],
            workers=bootstrap_workers,
        )
        report["bootstrap_execution"] = {
            "strategy": "shared_draw_curve_points_process_pool",
            "workers": bootstrap_workers,
            "unique_curve_points": len(groups),
            "requested_comparisons": len(comparisons),
            "unique_comparisons": len(paired_results),
        }
        for cap in plan["caps_mib"]:
            for baseline in labels[:-1]:
                paired = paired_results[(baseline, cap), (candidate, cap)]
                report["quality_comparisons"].append(
                    {"cap_mib": cap, "baseline": baseline, "candidate": candidate,
                     **asdict(paired)}
                )
        # Cross-cap claim: the competitor is any baseline point whose quality is
        # not higher than the candidate under the zero-margin paired test.
        for candidate_cap in plan["caps_mib"]:
            candidate_point = by_point[candidate, candidate_cap]
            for baseline in (
                "prompt_attention_topk",
                "vl_cache_reimpl",
                "frozen_mosaickv_exact",
            ):
                for baseline_cap in plan["caps_mib"]:
                    paired = paired_results[
                        (baseline, baseline_cap), (candidate, candidate_cap)
                    ]
                    lower = paired.ci_low
                    reductions = {
                        phase: 1 - candidate_point[f"{phase}_active_kv_storage_bytes"]
                        / by_point[baseline, baseline_cap][f"{phase}_active_kv_storage_bytes"]
                        for phase in ("initial", "final", "peak")
                    }
                    report["cross_curve_30_percent_tests"].append(
                        {
                            "candidate_cap_mib": candidate_cap,
                            "baseline": baseline,
                            "baseline_cap_mib": baseline_cap,
                            "quality_difference": paired.candidate_minus_reference,
                            "quality_ci_low": lower,
                            "quality_ci_high": paired.ci_high,
                            "quality_noninferior_zero_margin": lower >= 0,
                            "memory_reductions": reductions,
                            "passes_all_memory_boundaries": min(reductions.values()) >= 0.30,
                            "passes_joint_gate": lower >= 0 and min(reductions.values()) >= 0.30,
                        }
                    )
        passed_pairs = [r for r in report["cross_curve_30_percent_tests"]
                        if r["passes_joint_gate"]]
        report.update(
            status="passed",
            interpretation=(
                "development_joint_gate_passed_not_heldout" if passed_pairs
                else "development_joint_gate_not_demonstrated"
            ),
            joint_gate_pairs=passed_pairs,
            aggregates=aggregates,
        )
        pd.DataFrame(rows.values()).to_parquet(output / "paired_samples.parquet", index=False)
        pd.DataFrame(aggregates).to_parquet(output / "aggregates.parquet", index=False)
        pd.DataFrame(report["quality_comparisons"]).to_parquet(
            output / "quality_comparisons.parquet", index=False
        )
        pd.DataFrame(report["cross_curve_30_percent_tests"]).to_parquet(
            output / "cross_curve_tests.parquet", index=False
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    write_once(output / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    raise SystemExit(analyze(parser.parse_args()))
