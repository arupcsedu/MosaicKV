#!/usr/bin/env python3
"""Validate complete paired shards; recompute cohort CIDEr and paired bootstrap.

Never read/average shard CIDEr scores. Never include FullKV in latency tests.
The write-once report is a stage gate, not evidence of superiority by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

from run_coco_shared_bytes import completed_sample
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import load_config
from mosaickv.evaluation.coco_shared_bytes import (
    METHODS,
    STORAGE_FIELDS,
    TokenizedCider,
    bootstrap_cider_curve,
    control_hashes,
    dense_capacity_certificate,
    validate_caption_rows,
)
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.scorers import aggregate_mean
from mosaickv.evaluation.metrics.statistics import paired_bootstrap
from mosaickv.evaluation.metrics.types import MetricDirection, MetricSample
from mosaickv.evaluation.shared_byte_contract import digest as object_digest
from mosaickv.evaluation.shared_byte_contract import validate_group


def plot_curves(aggregates, output):
    # The locked common environment has no matplotlib. Emit portable vector
    # plots without adding packages or making metric gates depend on a renderer.
    from html import escape

    names = {
        "prompt_attention_topk": ("Prompt Top-K", "#2864b4"),
        "vl_cache_reimpl": ("VL-Cache (reimpl)", "#c96b16"),
        "mosaickv_exact": ("MosaicKV exact", "#22804a"),
    }
    for metric, ylabel, filename in (
        ("metric_value", "Corpus CIDEr (score points)", "cider_memory.svg"),
        ("end_to_end_time", "End-to-end latency (seconds)", "compressed_latency_memory.svg"),
    ):
        values = [r for r in aggregates if r["method"] in names]
        xmax = max(r["initial_active_kv_storage_bytes"] / 2**20 for r in values) * 1.1
        ymax = max(r[metric] for r in aggregates if r[metric] is not None) * 1.15 or 1
        svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="850" height="430" viewBox="0 0 850 430">',
            '<rect width="850" height="430" fill="white"/>',
            '<g font-family="sans-serif" font-size="13" fill="#222">',
        ]

        def point(x, y, xmax=xmax, ymax=ymax):
            return 80 + 490 * x / xmax, 350 - 310 * y / ymax

        for i in range(6):
            x, y = point(xmax * i / 5, ymax * i / 5)
            svg += [
                f'<path d="M80 {y:.2f}H570" stroke="#ddd"/>',
                f'<text x="69" y="{y + 4:.2f}" text-anchor="end">{ymax * i / 5:.2f}</text>',
                f'<text x="{x:.2f}" y="373" text-anchor="middle">{xmax * i / 5:.1f}</text>',
            ]
        svg.append('<path d="M80 40V350H570" fill="none" stroke="#222"/>')
        for index, (method, (name, color)) in enumerate(names.items()):
            points = sorted(
                (r for r in aggregates if r["method"] == method), key=lambda r: r["cap_mib"]
            )
            coordinates = [
                point(r["initial_active_kv_storage_bytes"] / 2**20, r[metric]) for r in points
            ]
            joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
            svg.append(
                f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for x, y in coordinates:
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            svg.append(f'<text x="610" y="{65 + index * 26}" fill="{color}">{escape(name)}</text>')
        if metric == "metric_value":
            full = next(r[metric] for r in aggregates if r["method"] == "full_kv")
            _, y = point(0, full)
            svg.extend(
                [
                    f'<path d="M80 {y:.2f}H570" stroke="gray" stroke-dasharray="5 4"/>',
                    '<text x="610" y="143" fill="gray">FullKV quality reference</text>',
                ]
            )
        svg += [
            '<text x="325" y="408" text-anchor="middle">'
            "Measured initial active KV storage (MiB)</text>",
            '<text transform="translate(20,195) rotate(-90)" text-anchor="middle">'
            f"{escape(ylabel)}</text>",
            "</g></svg>",
        ]
        with (output / filename).open("x") as stream:
            stream.write("\n".join(svg))


def comparison_scope(plan, stage, *, requested_methods=None, requested_caps=None):
    """Resolve an explicitly registered subset without inventing missing arms."""

    registered_methods = tuple(plan.get("methods", METHODS))
    methods = tuple(requested_methods or registered_methods)
    if (
        len(methods) != len(set(methods))
        or any(method not in registered_methods for method in methods)
        or "full_kv" not in methods
        or "mosaickv_exact" not in methods
    ):
        raise ValueError("analysis methods must be a registered subset containing FullKV/MosaicKV")
    registered_caps = tuple(plan.get("caps_mib", (40,)))
    caps = (40,) if stage == "validation" else tuple(requested_caps or registered_caps)
    if (
        len(caps) != len(set(caps))
        or any(cap not in registered_caps for cap in caps)
    ):
        raise ValueError("analysis caps must be a distinct registered subset")
    return methods, caps


def collect(plan, plan_sha, stage, *, methods, caps):
    plan = dict(plan, _plan_sha=plan_sha)
    rows, traces, samples, missing = {}, {}, {}, []
    for index, job in enumerate(plan["jobs"]):
        if job["stage"] != stage:
            continue
        if job["method"] not in methods:
            continue
        if job["method"] != "full_kv" and job["cap_mib"] not in caps:
            continue
        if digest(Path(job["ledger"])) != job["ledger_sha"]:
            raise ValueError("input ledger changed")
        payload = json.loads(Path(job["ledger"]).read_text())
        cfg = load_config(job["config"])
        for sample in payload["samples"]:
            sid = sample["sample_id"]
            if sid in samples and samples[sid] != sample:
                raise ValueError("paired input mismatch")
            samples[sid] = sample
            root = Path(job["output"]) / "samples" / hashlib.sha256(sid.encode()).hexdigest()
            record = completed_sample(plan, job, sid)
            if record is None:
                missing.append(
                    dict(
                        index=index,
                        sample_id=sid,
                        method=job["method"],
                        cap_mib=job["cap_mib"],
                        failed_attempts=len(list(root.glob("attempt-*/failure.json"))),
                    )
                )
                continue
            validate_caption_rows(record["rows"], [sample], plan["output_tokens"])
            row = record["rows"][0]
            expected = control_hashes(
                cfg, sample, plan["processor_signature"], record["gpu_controls"]
            )
            if expected != record["control_hashes"] or record["warmups"] != plan["warmups"]:
                raise ValueError("measurement controls mismatch")
            trace = json.loads(Path(record["trace"]).read_text())
            if (
                trace["status"] != "completed"
                or trace["method"] != job["method"]
                or trace.get("effective_method", job["method"]) != job["method"]
            ):
                raise ValueError("incomplete trace/effective-method substitution")
            layout = trace["source_kv_layout"]
            # Prove append-only logical payload growth independently of
            # potentially larger preallocated storage, and reject fallback.
            geometry = dense_capacity_certificate(layout, 2**62)
            if (
                trace["final_active_kv_bytes"]
                != trace["active_kv_bytes"]
                + (plan["output_tokens"] - 1) * geometry["bytes_per_position"]
            ):
                raise ValueError("decode payload is not append-only")
            if trace["prefixkv_decode_events"] or trace["repair_events"]:
                raise ValueError("unexpected decode eviction/repair")
            for name in (
                "hierarchical_logit_safe_eviction",
                "progressive_logit_safe_eviction",
                "short_horizon_logit_stability",
            ):
                if trace.get(name, {}).get("used_fullkv_fallback"):
                    raise ValueError("FullKV fallback")
            if any(trace["cache_lifecycle"][field] != row[field] for field in STORAGE_FIELDS):
                raise ValueError("cache trace/raw row mismatch")
            key = job["method"], job["cap_mib"], sid
            if key in rows:
                raise ValueError("duplicate method/cap/sample")
            rows[key] = dict(
                row,
                method=job["method"],
                cap_mib=job["cap_mib"],
                lane="vl_cache",
                task="coco_caption",
                metric_name="cider",
                result_category="unified_comparison",
                control_hashes=expected,
                physical_byte_cap=None if job["cap_mib"] is None else job["cap_mib"] * 2**20,
                decode_evictions=0,
                used_fullkv_fallback=False,
                source_sha=record["source_sha"],
                config_sha=record["config_sha"],
                trace=record["trace"],
                label="reimpl" if job["method"].endswith("_reimpl") else "local",
            )
            traces[key] = trace
    return rows, traces, samples, missing


def analyze(args):
    import pandas as pd

    if digest(args.root / "plan.json") != args.plan_sha:
        raise ValueError("plan hash mismatch")
    plan = json.loads((args.root / "plan.json").read_text())
    gate_output = args.root / "analysis" / args.stage / "report.json"
    if args.output is None and gate_output.exists():
        raise FileExistsError("canonical successful analysis is immutable")
    output = args.output or gate_output.parent / f"attempt-{time.time_ns()}"
    if output.exists():
        raise FileExistsError("analysis is write-once; choose a new diagnostic --output")
    report = dict(
        status="failed",
        stage=args.stage,
        plan_sha256=args.plan_sha,
        plan_path=str(args.root / "plan.json"),
        source_sha=plan["source_sha"],
        fullkv_latency_comparison=False,
        historical_bf16_reused=False,
        prior_use=plan["prior_use_disclosures"],
        heldout_claim=False,
        complete_cohort_cider=True,
        bootstrap_resamples=plan["bootstrap_resamples"],
        comparisons=[],
        gates=[],
    )
    try:
        methods, caps = comparison_scope(
            plan,
            args.stage,
            requested_methods=getattr(args, "methods", None),
            requested_caps=getattr(args, "caps", None),
        )
        report["methods"] = list(methods)
        report["caps_mib"] = list(caps)
        rows, traces, samples, missing = collect(
            plan, args.plan_sha, args.stage, methods=methods, caps=caps
        )
        report.update(
            planned_samples=len(samples),
            completed_rows=len(rows),
            missing_or_failed=missing,
            retry_indices=sorted({m["index"] for m in missing}),
        )
        if missing:
            raise ValueError("incomplete paired cohort; metrics intentionally withheld")
        ids = sorted(samples)
        denominator = (
            (32 if args.stage == "validation" else 256)
            if args.stage != "final"
            else plan["final_sample_count"]
        )
        if len(ids) != denominator:
            raise ValueError("wrong preregistered stage denominator")
        full = [rows["full_kv", None, sid] for sid in ids]
        expected_controls = {sid: rows["full_kv", None, sid]["control_hashes"] for sid in ids}
        for cap in caps:
            group, certificates = list(full), {}
            for method in (name for name in methods if name != "full_kv"):
                for sid in ids:
                    key = method, cap, sid
                    source_layout = traces["full_kv", None, sid]["source_kv_layout"]
                    if source_layout != traces[key]["source_kv_layout"]:
                        raise ValueError("source geometry differs across methods")
                    certificate = dense_capacity_certificate(source_layout, cap * 2**20)
                    certificates[method, sid] = certificate
                    rows[key].update(
                        largest_feasible_initial_bytes=certificate[
                            "largest_feasible_initial_bytes"
                        ],
                        capacity_certificate_sha256=object_digest(certificate),
                        target_shortfall_bytes=certificate["target_shortfall_bytes"],
                    )
                    group.append(rows[key])
            report["gates"].append(
                dict(
                    cap_mib=cap,
                    **validate_group(
                        group,
                        sample_ids=ids,
                        methods=methods,
                        cap_bytes=cap * 2**20,
                        metric="cider",
                        expected_controls=expected_controls,
                        capacity_certificates=certificates,
                    ),
                )
            )
            write_once(
                output / f"capacity-{cap}.json",
                {f"{m}/{s}": c for (m, s), c in certificates.items()},
            )
        metric_samples = {
            (method, cap): [
                MetricSample(
                    sample_id=sid,
                    prediction=rows[method, cap, sid]["answer"],
                    references=tuple(samples[sid]["references"]),
                )
                for sid in ids
            ]
            for method, cap in sorted({(m, c) for m, c, _ in rows}, key=str)
        }
        reducer = TokenizedCider([sample for group in metric_samples.values() for sample in group])
        definition = replace(default_metric_registry().resolve("cider"), reducer=reducer)
        aggregates = []
        reference_sha = object_digest({s: samples[s]["references"] for s in ids})
        for (method, cap), contributions in metric_samples.items():
            measured = [rows[method, cap, sid] for sid in ids]
            aggregates.append(
                dict(
                    run_id=f"coco-{args.stage}-{args.plan_sha[:12]}-{method}-{cap}",
                    sample_id=None,
                    task="coco_caption",
                    model=plan["model_id"],
                    model_revision=plan["model_revision"],
                    dataset_id=plan["dataset_id"],
                    dataset_revision=plan["dataset_revision"],
                    backend="huggingface",
                    precision="fp16",
                    source_sha=plan["source_sha"],
                    config_sha=measured[0]["config_sha"],
                    manifest_path=str(output / "report.json"),
                    result_category="unified_comparison",
                    result_origin="locally_measured",
                    metric_direction="higher",
                    evaluated_token_count=None,  # CIDEr scores captions, not teacher-forced tokens.
                    generated_token_count=len(ids) * plan["output_tokens"],
                    method=method,
                    cap_mib=cap,
                    metric_name="cider",
                    metric_value=definition.aggregate(contributions),
                    metric_scope="corpus",
                    metric_unit="score_points",
                    metric_implementation=definition.implementation,
                    metric_version=definition.version,
                    evaluator_commit=plan["source_sha"],
                    reference_artifact_sha=reference_sha,
                    sample_count=len(ids),
                    label="reimpl" if method.endswith("_reimpl") else "local",
                    measurement_type="unified_comparison_metric_measured",
                    **{
                        field: statistics.mean(row[field] for row in measured)
                        for field in STORAGE_FIELDS
                    },
                    end_to_end_time=None
                    if method == "full_kv"
                    else statistics.mean(r["end_to_end_time"] for r in measured),
                    decode_time=None
                    if method == "full_kv"
                    else statistics.mean(r["decode_time"] for r in measured),
                )
            )
        cider_pairs = [
            (
                (reference, None if reference == "full_kv" else cap),
                ("mosaickv_exact", cap),
            )
            for cap in caps
            for reference in methods
            if reference != "mosaickv_exact"
        ]
        cider_comparisons = bootstrap_cider_curve(
            metric_samples,
            cider_pairs,
            resamples=plan["bootstrap_resamples"],
            seed=plan["bootstrap_seed"],
            workers=getattr(args, "bootstrap_workers", 1),
        )
        for cap in caps:
            for reference in (name for name in methods if name != "mosaickv_exact"):
                refcap = None if reference == "full_kv" else cap
                result = cider_comparisons[
                    (reference, refcap), ("mosaickv_exact", cap)
                ]
                report["comparisons"].append(
                    dict(
                        cap_mib=cap,
                        baseline=reference,
                        candidate="mosaickv_exact",
                        **asdict(result),
                    )
                )
                if reference != "full_kv":
                    latency_def = replace(
                        default_metric_registry().resolve("accuracy"),
                        name="end_to_end_time",
                        reducer=aggregate_mean,
                        direction=MetricDirection.LOWER,
                        unit="seconds",
                    )
                    latency = paired_bootstrap(
                        latency_def,
                        [
                            MetricSample(
                                sample_id=sid,
                                prediction="latency",
                                references=("latency",),
                                value=rows[reference, cap, sid]["end_to_end_time"],
                            )
                            for sid in ids
                        ],
                        [
                            MetricSample(
                                sample_id=sid,
                                prediction="latency",
                                references=("latency",),
                                value=rows["mosaickv_exact", cap, sid]["end_to_end_time"],
                            )
                            for sid in ids
                        ],
                        resamples=plan["bootstrap_resamples"],
                        seed=plan["bootstrap_seed"],
                    )
                    report["comparisons"].append(
                        dict(
                            cap_mib=cap,
                            baseline=reference,
                            candidate="mosaickv_exact",
                            **asdict(latency),
                        )
                    )
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                dict(r, control_hashes=json.dumps(r["control_hashes"], sort_keys=True))
                for r in rows.values()
            ]
        ).to_parquet(output / "paired_samples.parquet", index=False)
        pd.DataFrame(aggregates).to_parquet(output / "corpus_metrics.parquet", index=False)
        pd.DataFrame(report["comparisons"]).to_parquet(
            output / "paired_comparisons.parquet", index=False
        )
        plot_curves(aggregates, output)
        report["aggregates"] = aggregates
        report["derived_artifact_hashes"] = {
            str(p): digest(p) for p in output.iterdir() if p.suffix in {".parquet", ".svg"}
        }
        report["status"] = "passed"
        report["interpretation"] = "execution/byte/metric gate only; not a superiority gate"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    write_once(output / "report.json", report)
    if report["status"] == "passed" and args.output is None:
        write_once(
            gate_output,
            dict(
                report,
                analysis_report=str(output / "report.json"),
                analysis_report_sha256=digest(output / "report.json"),
            ),
        )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--stage", choices=("validation", "larger", "final"), required=True)
    parser.add_argument("--methods", nargs="+", help="registered method subset to analyze")
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=1,
        help="processes used by the exact whole-corpus CIDEr bootstrap",
    )
    parser.add_argument("--caps", type=int, nargs="+", help="registered cap subset to analyze")
    parser.add_argument(
        "--output", type=Path, help="new output path for an incomplete diagnostic only"
    )
    raise SystemExit(analyze(parser.parse_args()))
