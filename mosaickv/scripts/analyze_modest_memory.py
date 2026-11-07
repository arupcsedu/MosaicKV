#!/usr/bin/env python3
"""Fail-closed 128-image budget calibration; confirmation is never submitted here."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from prepare_modest_memory import digest, write_once

try:  # A submission snapshots this exact pure-CPU module alongside the script.
    from cider_resampling import ResampledCider, paired_draw_counts
except ModuleNotFoundError:
    from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts

from mosaickv.evaluation.coco_shared_bytes import (
    METHODS,
    STORAGE_FIELDS,
    dense_capacity_certificate,
)
from mosaickv.evaluation.metrics.types import MetricSample
from mosaickv.evaluation.shared_byte_contract import digest as object_digest
from mosaickv.evaluation.shared_byte_contract import validate_group


def select_pair(values: dict, *, target: float) -> dict:
    """One smallest quality-eligible competitor across BOTH baseline methods."""
    priority = {"prompt_attention_topk": 0, "vl_cache_reimpl": 1}
    baselines = sorted(
        (p for p in values if p[0] in priority and values[p] >= target),
        key=lambda p: (p[1], priority[p[0]]),
    )
    if not baselines:
        return dict(
            status="negative", reason="no_baseline_meets_independent_calibration_fullkv_target"
        )
    baseline = baselines[0]
    candidates = sorted(
        (
            p
            for p in values
            if p[0] == "mosaickv_exact" and p[1] < baseline[1] and values[p] >= values[baseline]
        ),
        key=lambda p: p[1],
        reverse=True,
    )
    if not candidates:
        return dict(
            status="negative",
            reason="no_smaller_candidate_preserves_selected_competitor_quality",
            baseline=list(baseline),
            quality_target=target,
        )
    return dict(
        status="promising_calibration_only",
        baseline=list(baseline),
        candidate=list(candidates[0]),
        quality_target=target,
        calibration_difference=values[candidates[0]] - values[baseline],
    )


def power_size(
    delta: float, bootstrap_se: float, calibration_n: int, *, tests: int = 10
) -> int | None:
    """Planning approximation, not a power guarantee; never changes after confirmation."""
    if bootstrap_se < 0 or calibration_n < 2 or tests < 1:
        raise ValueError("invalid power inputs")
    if delta == 0 and bootstrap_se == 0:
        return 512
    if delta <= 0:
        return None
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * tests)) + NormalDist().inv_cdf(0.8)
    return max(512, math.ceil(calibration_n * (z * bootstrap_se / delta) ** 2))


def analyze(args):
    root = args.root.resolve()
    if digest(root / "plan.json") != args.plan_sha:
        raise ValueError("plan hash mismatch")
    plan = json.loads((root / "plan.json").read_text())
    output = root / "analysis" / "calibration"
    if output.exists():
        raise FileExistsError("analysis is write-once")
    source = Path(plan["source_root"])
    sys.path.insert(0, str(source / "mosaickv/scripts"))
    spec = importlib.util.spec_from_file_location(
        "modest_collector", source / "mosaickv/scripts/analyze_coco_shared_bytes.py"
    )
    assert spec and spec.loader
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    report = dict(
        status="failed",
        classification="prospective_budget_calibration_not_confirmation",
        plan_sha256=args.plan_sha,
        source_sha=plan["source_sha"],
        heldout_paper_claim=False,
        confirmation_submitted=False,
        no_algorithm_changes=True,
    )
    try:
        # The pinned edb6edf collector predates subset-filter keyword options;
        # this plan already contains exactly the intended complete 13-arm grid.
        rows, traces, samples, missing = collector.collect(plan, args.plan_sha, "validation")
        report.update(completed_rows=len(rows), missing_or_failed=missing, planned_samples=128)
        output.mkdir(parents=True)
        if rows:
            pd.DataFrame(
                [
                    dict(r, control_hashes=json.dumps(r["control_hashes"], sort_keys=True))
                    for r in rows.values()
                ]
            ).to_parquet(output / "paired_samples.parquet", index=False)
        if missing or len(rows) != 128 * 13 or len(samples) != 128:
            raise ValueError("incomplete calibration; no scoring, freeze, or confirmation allowed")
        ids = sorted(samples)
        full = [rows["full_kv", None, sid] for sid in ids]
        expected = {sid: rows["full_kv", None, sid]["control_hashes"] for sid in ids}
        gates = []
        for cap in plan["caps_mib"]:
            group = list(full)
            certificates = {}
            for method in METHODS[1:]:
                for sid in ids:
                    layout = traces["full_kv", None, sid]["source_kv_layout"]
                    if layout != traces[method, cap, sid]["source_kv_layout"]:
                        raise ValueError("source cache layouts differ")
                    cert = dense_capacity_certificate(layout, cap * 2**20)
                    certificates[method, sid] = cert
                    row = rows[method, cap, sid]
                    row.update(
                        largest_feasible_initial_bytes=cert["largest_feasible_initial_bytes"],
                        target_shortfall_bytes=cert["target_shortfall_bytes"],
                        capacity_certificate_sha256=object_digest(cert),
                    )
                    group.append(row)
            gates.append(
                dict(
                    cap_mib=cap,
                    **validate_group(
                        group,
                        sample_ids=ids,
                        methods=METHODS,
                        cap_bytes=cap * 2**20,
                        metric="cider",
                        expected_controls=expected,
                        capacity_certificates=certificates,
                    ),
                )
            )
        groups = {
            p: [
                MetricSample(sid, rows[*p, sid]["answer"], tuple(samples[sid]["references"]))
                for sid in ids
            ]
            for p in {(m, c) for m, c, _ in rows}
        }
        scorer = ResampledCider(groups)
        counts = paired_draw_counts(128, 10000, plan["bootstrap_seed"])
        original = np.ones((1, 128), dtype=np.int64)
        values = scorer.score(original)[0]
        error = scorer.validate_official(
            np.vstack([original, counts[:2]]), scorer.score(np.vstack([original, counts[:2]]))
        )
        scores = scorer.score(counts)
        lookup = {p: i for i, p in enumerate(scorer.points)}
        quality = {p: float(v) for p, v in zip(scorer.points, values, strict=True)}
        curve = [
            dict(
                method=p[0],
                cap_mib=p[1],
                cider=quality[p],
                sample_count=128,
                **{
                    field: statistics.mean(rows[*p, sid][field] for sid in ids)
                    for field in STORAGE_FIELDS
                },
                end_to_end_time=None
                if p[0] == "full_kv"
                else statistics.mean(rows[*p, sid]["end_to_end_time"] for sid in ids),
            )
            for p in scorer.points
        ]
        pd.DataFrame(curve).to_parquet(output / "corpus_metrics.parquet", index=False)
        np.savez_compressed(
            output / "bootstrap_corpus_scores.npz",
            counts=counts,
            scores=scores,
            points=np.asarray(scorer.points, dtype=str),
            sample_ids=np.asarray(scorer.sample_ids),
        )
        choice = select_pair(quality, target=quality["full_kv", None])
        report.update(
            status="completed",
            gates=gates,
            aggregates=curve,
            selection=choice,
            official_cider_parity_max_error=error,
            bootstrap_resamples=10000,
            quality_target_rule="calibration_fullkv_corpus_cider",
            frozen_candidate_created=False,
        )
        if choice["status"] == "promising_calibration_only":
            bp, cp = tuple(choice["baseline"]), tuple(choice["candidate"])
            delta = scores[:, lookup[cp]] - scores[:, lookup[bp]]
            n = power_size(choice["calibration_difference"], float(delta.std(ddof=1)), 128)
            storage = {
                field: dict(
                    candidate_mean=statistics.mean(rows[*cp, sid][field] for sid in ids),
                    baseline_mean=statistics.mean(rows[*bp, sid][field] for sid in ids),
                    every_sample_smaller=all(
                        rows[*cp, sid][field] < rows[*bp, sid][field] for sid in ids
                    ),
                )
                for field in STORAGE_FIELDS
            }
            report.update(
                power=dict(
                    required_samples=n,
                    planning_only=True,
                    power=0.8,
                    alpha=0.05,
                    tests=10,
                    paired_bootstrap_se=float(delta.std(ddof=1)),
                    caveat="selected calibration effects can overestimate confirmation effects",
                ),
                storage=storage,
            )
            if not all(s["every_sample_smaller"] for s in storage.values()):
                raise ValueError(
                    "chosen operating point is not physically smaller on every example"
                )
            if n is not None and n <= 512:
                candidate_job = next(j for j in plan["jobs"] if (j["method"], j["cap_mib"]) == cp)
                frozen = dict(
                    choice,
                    schema_version=1,
                    status="frozen_after_calibration_not_paper_pass",
                    source_sha=plan["source_sha"],
                    candidate_content_sha256=plan["candidate_content_sha256"],
                    candidate_config=candidate_job["config"],
                    candidate_config_sha256=candidate_job["config_sha"],
                    confirmation_samples=512,
                    quality_margin=0.0,
                    latency_ratio_upper_limit=1.0,
                    confirmation_controls="modest_memory_v1",
                    calibration_plan_sha256=args.plan_sha,
                    calibration_sample_ids_consumed=ids,
                    algorithm_retuning_authorized=False,
                )
                write_once(
                    output / "frozen_candidate.yaml", frozen
                )  # JSON is valid YAML; immutable, hashed.
                report["frozen_candidate_created"] = True
            else:
                report["confirmation_blocker"] = (
                    "power_requires_more_than_512_new_ID_only_reserved_examples"
                )
        else:
            report["confirmation_blocker"] = choice["reason"]
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            frozen_candidate_created=False,
        )
    report["derived_artifact_hashes"] = (
        {str(p): digest(p) for p in output.iterdir() if p.is_file()} if output.exists() else {}
    )
    write_once(output / "report.json", report)
    print(
        json.dumps(
            dict(
                status=report["status"],
                report=str(output / "report.json"),
                frozen_candidate_created=report.get("frozen_candidate_created", False),
                blocker=report.get("confirmation_blocker", report.get("error")),
            )
        ),
        flush=True,
    )
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha", required=True)
    raise SystemExit(analyze(p.parse_args()))
