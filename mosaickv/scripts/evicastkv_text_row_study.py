#!/usr/bin/env python3
"""Bounded consumed-COCO precision ablations; never independent paper evidence."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import evicastkv_precision_rate_distortion as rate
import evicastkv_request_validation as previous

from mosaickv.evaluation import request_precision
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD = Path("/scratch/djy8hg/runs/evicastkv/precision-rate-distortion-20261031-v2")
ROW = Path("/scratch/djy8hg/runs/evicastkv/value-row-precision-consumed-20261031-v1")
POLICY = "head3k2v_rows_text"
STUDY_POLICIES = ("head3k2v_mse", "head3_boundary8")
TEXT = Path("/scratch/djy8hg/runs/evicastkv/text-value-row-study-20261031-v1")
DIAGNOSTICS = (
    Path("/scratch/djy8hg/runs/evicastkv/mse-value-precision-consumed-20261031-v1"),
    Path("/scratch/djy8hg/runs/evicastkv/boundary-precision-consumed-20261031-v1"),
)
METHODS = (
    "mosaickv_exact",
    "lookm_reimpl",
    "vl_cache_reimpl",
    "prefixkv_reimpl",
    "prompt_attention_topk",
)


def compact_row(row):
    """Derived scalar/answer view; full catalogs remain in hash-bound raw traces."""
    fields = (
        "sample_id",
        "task",
        "method",
        "precision",
        "warmup",
        "status",
        "source_sha",
        "plan_sha256",
        "manifest_path",
        "scored_answer",
        "logits_sha256",
        "end_to_end_seconds",
        "used_fullkv_fallback",
        "error",
    )
    result = {k: row[k] for k in fields if k in row}
    if "storage" in row:
        result["storage"] = {
            k: row["storage"][k]
            for k in (
                "initial_active_kv_storage_bytes",
                "final_active_kv_storage_bytes",
                "peak_active_kv_storage_bytes",
            )
            if k in row["storage"]
        }
    return result


def construct(*args, policy, **kwargs):
    from mosaickv.evaluation.head_local_precision import bundle

    if args[-1] != "query3" or policy not in ("union3", POLICY, *STUDY_POLICIES):
        raise ValueError("registered text-row precision required")
    formats, proof = bundle(*args[:-1], policies=(policy,), **kwargs)
    return formats[policy], dict(
        proof,
        precision=policy,
        generated_history="lossless_consolidated_source_append_only",
        explicit_precision_variant_not_graph_exact_paper_method=True,
    )


def worker(root, expected, index):
    from mosaickv.evaluation.head_local_precision import gpu_controls

    plan = previous.common.inputs(root, expected)
    controls = gpu_controls()
    write_once(
        root / f"numerical/{index}.json", dict(source_sha=plan["source_sha"], controls=controls)
    )
    original = request_precision.complete_request
    baseline = json.loads(Path(plan["reused_rows"]).read_text())["rows"]
    reference = {
        (r["sample_id"], r["method"]): r
        for r in baseline
        if not r["warmup"] and r["precision"] == "union3"
    }
    checked = False

    def invoke(*args, **kwargs):
        nonlocal checked
        if (
            not checked
            and kwargs.get("precision") == "query3"
            and kwargs.get("method") != "full_kv"
        ):
            # An unchanged packet control is outside the measured request.
            with patch.object(
                request_precision,
                "construct_request_formats",
                lambda *a, **k: construct(*a, policy="union3", **k),
            ):
                logits, record = original(*args, **kwargs)
            sample_id = previous.read(plan["jobs"][index]["inputs"][0])["sample_id"]
            old = reference[(sample_id, kwargs["method"])]
            digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()
            passed = digest == old["logits_sha256"]
            write_once(
                root / f"union_controls/{index}.json",
                dict(
                    source_sha=plan["source_sha"],
                    sample_id=sample_id,
                    method=kwargs["method"],
                    old_logits_sha256=old["logits_sha256"],
                    new_logits_sha256=digest,
                    bitwise_logit_hash_match=passed,
                    outside_measured_request=True,
                ),
            )
            del logits, record
            if not passed:
                raise RuntimeError("unchanged union3 transport drift")
            checked = True
        return original(*args, **kwargs)

    with (
        patch.object(request_precision, "complete_request", invoke),
        patch.object(rate, "POLICIES", tuple(plan["precision_policies"])),
        patch.object(rate, "construct_policy", construct),
    ):
        rate.worker(root, expected, index)


def prepare(root):
    if root.exists():
        raise ValueError("new write-once output root required")
    old = previous.read(OLD / "plan.json")
    gate = previous.read(OLD / "analysis/GATE.json")
    row = previous.read(ROW / "analysis/decision.json")
    if (
        not gate["complete"]
        or gate["failures"]
        or gate["unchanged_union_transport_drift"]
        or not row["complete"]
        or row["failures"]
    ):
        raise ValueError("complete previous diagnostic evidence required")
    bound = dict(old["bound_files"])
    text_plan = previous.read(TEXT / "plan.json")
    text_gate = previous.read(TEXT / "analysis/GATE.json")
    if not text_gate["complete"] or text_gate["failures"]:
        raise ValueError("complete text-row transport study required")
    for field in (
        "request_policy",
        "ratio",
        "seed",
        "profile_path",
        "references",
        "selected_inputs",
    ):
        if text_plan[field] != old[field]:
            raise ValueError("incompatible reused execution controls")
    bound.update(text_plan["bound_files"])
    for path in (TEXT / "plan.json", TEXT / "analysis/GATE.json", TEXT / "analysis/rows.json"):
        bound[str(path)] = sha(path)
    reused = previous.read(TEXT / "analysis/rows.json")
    compact = [compact_row(r) for r in reused["rows"]]
    reused_provenance = reused["original_producer_hashes"]
    del reused
    for path, digest in reused_provenance.items():
        if sha(Path(path)) != digest:
            raise ValueError("text-row producer changed")
        bound[path] = digest
        if Path(path).name == "result.json":
            for name, artifact_sha in previous.read(path)["outputs"].items():
                artifact = Path(path).parent / name
                if sha(artifact) != artifact_sha:
                    raise ValueError("text-row artifact changed")
                bound[str(artifact)] = artifact_sha
    for diagnostic in DIAGNOSTICS:
        decision = previous.read(diagnostic / "analysis/decision.json")
        if not decision["complete"] or decision["failures"]:
            raise ValueError(
                "complete numerical execution required; quality failures remain negative"
            )
        bound[str(diagnostic / "analysis/decision.json")] = sha(
            diagnostic / "analysis/decision.json"
        )
    for path in (
        OLD / "plan.json",
        OLD / "analysis/GATE.json",
        OLD / "derived/normalized_raw.json",
        ROW / "analysis/decision.json",
        SOURCE / "mosaickv/docs/text_value_row_study_20261031.md",
        SOURCE / "mosaickv/docs/precision_combined_task_study_20261031.md",
    ):
        bound[str(path)] = sha(path)
    # Bind every original producer artifact before deriving reused rows.
    for path, digest in previous.read(OLD / "derived/normalized_raw.json")[
        "original_artifact_hashes"
    ].items():
        if sha(Path(path)) != digest:
            raise ValueError("reused producer changed")
        bound[path] = digest
        for name, artifact_sha in previous.read(path)["outputs"].items():
            artifact = Path(path).parent / name
            if sha(artifact) != artifact_sha:
                raise ValueError("reused artifact changed")
            bound[str(artifact)] = artifact_sha
    chosen = old["selected_inputs"]
    if len(chosen) != 16 or len(set(chosen)) != 16:
        raise ValueError("only the same sixteen consumed inputs permitted")
    work = [("mosaickv_exact", 0)] + [
        (m, s)
        for m in (*METHODS[1:], METHODS[0])
        for s in range(0, 16, 4)
        if (m, s) != ("mosaickv_exact", 0)
    ]
    jobs = [
        dict(
            index=i,
            inputs=chosen[start : start + 4],
            method_arms=[[method, p] for p in STUDY_POLICIES],
        )
        for i, (method, start) in enumerate(work)
    ]
    write_once(
        root / "reused_summary.json",
        dict(
            rows=compact,
            original_artifact_sha256=sha(TEXT / "analysis/rows.json"),
            omitted_catalogs_remain_in_hash_bound_raw=True,
        ),
    )
    bound[str(root / "reused_summary.json")] = sha(root / "reused_summary.json")
    plan = dict(
        old,
        source_sha=clean_source(SOURCE),
        bound_files=bound,
        jobs=jobs,
        precision_policies=list(STUDY_POLICIES),
        reused_rows=str(root / "reused_summary.json"),
        stage="range_fit_and_eight_bit_boundary_consumed_development",
        arms=[],
        reused_worker_zero=None,
        new_expected_rows=160,
        total_expected_rows=400,
        no_fresh_or_paper_promotion=True,
        paper_campaign_authorized=False,
        independent_quality_N=0,
        bootstrap_comparisons=48,
        explicit_per_policy_overrides={
            "head3_boundary8": dict(
                first_last_two_layers_KV_bits=8,
                middle_layers_KV_bits=3,
                boundary_layers_are_not_fully_exact=True,
            ),
            "head3k2v_mse": dict(
                value_range_fit="fixed_cache_local_MSE", no_extra_packet_bytes=True
            ),
        },
    )
    write_once(root / "plan.json", plan)
    print(sha(root / "plan.json"))


def canary_valid(result, rows, control, *, policies=(POLICY,)):
    scored = [r for r in rows if not r["warmup"]]
    return (
        result["status"] == "completed"
        and not result["failures"]
        and len(scored) == 4 * len(policies)
        and len({r["sample_id"] for r in scored}) == 4
        and all(
            r["status"] == "completed"
            and r["precision"] in policies
            and r["method"] == "mosaickv_exact"
            for r in scored
        )
        and {(r["sample_id"], r["precision"]) for r in scored}
        == {(r["sample_id"], p) for r in scored for p in policies}
        and control["bitwise_logit_hash_match"]
    )


def promote(root, expected):
    plan = previous.common.inputs(root, expected)
    out = root / "workers/0"
    passed = False
    if (out / "result.json").exists() and (root / "union_controls/0.json").exists():
        result = previous.read(out / "result.json")
        for name, digest in result["outputs"].items():
            if sha(out / name) != digest:
                raise ValueError("canary artifact changed")
        rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
        passed = canary_valid(
            result,
            rows,
            previous.read(root / "union_controls/0.json"),
            policies=tuple(plan["precision_policies"]),
        )
    write_once(
        root / "canary_decision.json",
        dict(
            passed=passed,
            source_sha=plan["source_sha"],
            execution_control_only=True,
            no_quality_pass_claim=True,
        ),
    )
    if passed:
        submit(root, expected, remaining=True)
    else:
        analyze(root, expected)


def analyze(root, expected):
    import numpy as np
    import pandas as pd

    from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
    from mosaickv.evaluation.metrics.types import MetricSample

    plan = previous.common.inputs(root, expected)
    rows = [r for r in previous.read(plan["reused_rows"])["rows"] if not r["warmup"]]
    failures, provenance = [], {}
    new = []
    for job in plan["jobs"]:
        out = root / f"workers/{job['index']}"
        if not (out / "result.json").exists():
            failures.append(dict(index=job["index"], error="missing producer"))
            continue
        result = previous.read(out / "result.json")
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("producer source mismatch")
        provenance[str(out / "result.json")] = sha(out / "result.json")
        for name, digest in result["outputs"].items():
            if sha(out / name) != digest:
                raise ValueError("producer artifact changed")
        failures.extend(result["failures"])
        for kind in ("numerical", "union_controls"):
            path = root / f"{kind}/{job['index']}.json"
            if not path.exists():
                failures.append(dict(index=job["index"], error=f"missing {kind}"))
            else:
                control = previous.read(path)
                provenance[str(path)] = sha(path)
                if control["source_sha"] != plan["source_sha"] or (
                    kind == "union_controls" and not control["bitwise_logit_hash_match"]
                ):
                    failures.append(dict(index=job["index"], error=f"failed {kind}"))
        if (out / "raw.jsonl").exists():
            new.extend(
                json.loads(line)
                for line in (out / "raw.jsonl").read_text().splitlines()
                if not json.loads(line)["warmup"]
            )
    rows += [compact_row(r) for r in new]
    keys = {(r["sample_id"], r["method"], r["precision"]) for r in rows}
    complete = (
        len(new) == plan["new_expected_rows"]
        and len(rows) == len(keys) == plan["total_expected_rows"]
        and not failures
        and all(r["status"] == "completed" for r in rows)
    )
    output = root / "analysis"
    write_once(
        output / "rows.json",
        dict(rows=rows, original_producer_hashes=provenance, no_original_artifact_changes=True),
    )
    gate = dict(
        complete=complete,
        failures=failures,
        new_rows=len(new),
        total_rows=len(rows),
        independent_quality_N=0,
        development_sample_count=16,
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        no_paper_promotion=True,
        superiority_established=False,
    )
    if complete:
        references = {
            s["sample_id"]: tuple(s["references"])
            for s in previous.read(plan["references"])["samples"]
        }
        groups = {}
        for r in rows:
            groups.setdefault((r["method"] + ":" + r["precision"], 0.5), []).append(
                MetricSample(
                    r["sample_id"], r["scored_answer"]["answer"], references[r["sample_id"]]
                )
            )
        ids = {r["sample_id"] for r in rows}
        for index in (0, 1):
            for line in (rate.PARENT / f"workers/{index}/raw.jsonl").read_text().splitlines():
                r = json.loads(line)
                if not r["warmup"] and r["method"] == "full_kv" and r["sample_id"] in ids:
                    groups.setdefault(("full_kv:ordinary", 0.5), []).append(
                        MetricSample(
                            r["sample_id"], r["scored_answer"]["answer"], references[r["sample_id"]]
                        )
                    )
        scorer = ResampledCider(groups)
        counts = np.vstack(
            (
                np.ones((1, 16), dtype=np.int64),
                paired_draw_counts(16, 10000, plan["bootstrap_seed"]),
            )
        )
        values = scorer.score(counts, batch_size=16)
        error = scorer.validate_official(counts[:4], values[:4])
        metrics, comparisons = [], []
        for i, point in enumerate(scorer.points):
            group = [r for r in rows if r["method"] + ":" + r["precision"] == point[0]]
            item = dict(
                point=point[0],
                cider=float(values[0, i]),
                sample_count=16,
                result_category="unified_comparison",
                result_origin="locally_measured",
                measurement_type="derived",
                reference_artifact_sha256=sha(Path(plan["references"])),
                evaluator_commit=plan["source_sha"],
                plan_sha256=expected,
            )
            if group:
                item["producer_source_shas"] = sorted({r["source_sha"] for r in group})
                item["mean_e2e_seconds"] = float(np.mean([r["end_to_end_seconds"] for r in group]))
                for field in (
                    "initial_active_kv_storage_bytes",
                    "final_active_kv_storage_bytes",
                    "peak_active_kv_storage_bytes",
                ):
                    item[field] = float(np.mean([r["storage"][field] for r in group]))
            metrics.append(item)
        for policy in plan["precision_policies"]:
            candidate = scorer.points.index(("mosaickv_exact:" + policy, 0.5))
            for i, point in enumerate(scorer.points):
                if i == candidate or point[0] == "full_kv:ordinary":
                    continue
                lo, hi = np.quantile(
                    values[1:, candidate] - values[1:, i],
                    [
                        0.05 / (2 * plan["bootstrap_comparisons"]),
                        1 - 0.05 / (2 * plan["bootstrap_comparisons"]),
                    ],
                )
                comparisons.append(
                    dict(
                        candidate="mosaickv_exact:" + policy,
                        baseline=point[0],
                        cider_difference=float(values[0, candidate] - values[0, i]),
                        simultaneous_ci95_bonferroni=[float(lo), float(hi)],
                    )
                )
        pd.DataFrame(metrics).to_parquet(output / "metrics.parquet", index=False)
        write_once(
            output / "comparisons.json",
            dict(
                metrics=metrics,
                comparisons=comparisons,
                official_reducer_max_error=error,
                corpus_recomputed_inside_every_bootstrap=True,
                not_byte_matched=True,
                same_consumed_development_inputs=True,
            ),
        )
    write_once(output / "GATE.json", gate)
    print(json.dumps(gate))


def submit(root, expected, *, remaining=False):
    plan = previous.common.inputs(root, expected)
    cpu = previous.read(root / "cpu.json")
    if (
        cpu["source_sha"] != plan["source_sha"]
        or cpu["returncode"]
        or cpu["failed_or_skipped"]
        or cpu["test_count"] < 2860
        or cpu["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("clean complete CPU suite proof required")
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in line.lower() for line in queue.splitlines()):
        raise ValueError("six-slot admission requires other owned GPU work complete")
    (root / "slurm").mkdir(exist_ok=True)
    script = str(SOURCE / "mosaickv/slurm/evicastkv_text_row_study.sbatch")
    args = [str(SOURCE), str(root), expected]
    gpu = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii-gpu",
            "--account=bii_dsc_community",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:30:00",
            "--array=1-19%6" if remaining else "--array=0",
            "--job-name=mk-text-rows",
            f"--output={root}/slurm/gpu-%A_%a.out",
            f"--error={root}/slurm/gpu-%A_%a.err",
            script,
            *args,
            "worker",
        ],
        text=True,
    ).strip()
    analysis = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:10:00",
            "--job-name=mk-text-analysis",
            f"--dependency=afterany:{gpu}",
            f"--output={root}/slurm/analysis-%j.out",
            f"--error={root}/slurm/analysis-%j.err",
            script,
            *args,
            "analyze" if remaining else "promote",
        ],
        text=True,
    ).strip()
    from control_a100_schedule import tick

    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=30,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[gpu],
    )
    scheduler = root / "scheduler" / ("remaining" if remaining else "canary")
    write_once(scheduler / "policy.json", policy)
    write_once(scheduler / "registration.json", tick(policy, {}, scheduler / "state.json", False))
    write_once(
        root / ("submission.json" if remaining else "canary_submission.json"),
        dict(gpu_array=gpu, cpu_analysis=analysis),
    )
    print(json.dumps(dict(gpu_array=gpu, cpu_analysis=analysis)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "bootstrap", "cpu", "submit", "worker", "promote", "analyze")
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "bootstrap":
        prepare(args.root)
        digest = sha(args.root / "plan.json")
        previous.cpu(args.root, digest)
        submit(args.root, digest)
    elif args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
        submit(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
