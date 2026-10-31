#!/usr/bin/env python3
"""Six-A100, sixteen-consumed-COCO-image development rate/distortion study.

Uses complete requests and actual task scores, not token identity as a quality
substitute. It cannot turn failed consumed-input guards into passes or authorize
a paper/fresh campaign. Both representations are offered to every selector.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import evicastkv_request_validation as previous

from mosaickv.evaluation import request_precision
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/request-validation-20261031-v3-single-pass-r1/cohort")
CONSUMED = Path(
    "/scratch/djy8hg/runs/evicastkv/precision-refit-consumed-20261031-v1/analysis/decision.json"
)
POLICIES = ("union3", "head3k2v_refit")


def construct_policy(
    adapter, prepared, prefill, capture, cfg, method, profile_path, ratio, precision, *, policy
):
    from mosaickv.evaluation.head_local_precision import bundle

    if precision != "query3" or policy not in POLICIES:
        raise ValueError("registered precision policy required")
    formats, proof = bundle(
        adapter, prepared, prefill, capture, cfg, method, profile_path, ratio, policies=(policy,)
    )
    return formats[policy], dict(
        proof,
        precision=policy,
        generated_history="lossless_consolidated_source_append_only",
        explicit_precision_variant_not_graph_exact_paper_method=True,
    )


def worker(root, expected, index):
    original = request_precision.complete_request

    def invoke(*args, **kwargs):
        policy = kwargs.get("precision")
        if policy not in POLICIES:
            return original(*args, **kwargs)
        kwargs["precision"] = "query3"

        def construct(*args, **kwargs):
            return construct_policy(*args, **kwargs, policy=policy)

        with patch.object(request_precision, "construct_request_formats", construct):
            logits, record = original(*args, **kwargs)
        record["development_precision_policy"] = policy
        record["precision"] = policy
        return logits, record

    # Worker process is explicitly single-request-at-a-time. Recursive FullKV
    # retention-1 calls take the ordinary branch, never the low-bit constructor.
    with patch.object(request_precision, "complete_request", invoke):
        previous.worker(root, expected, index)


def prepare(root):
    if root.exists():
        raise ValueError("new write-once root required")
    old = previous.read(PARENT / "plan.json")
    decision = previous.read(CONSUMED)
    if not decision["complete"] or decision["failures"]:
        raise ValueError("consumed numerical execution must be complete")
    # This is an explicitly different empirical question from the strict
    # consumed-token gate. Preserve its negative verdict and do not weaken it.
    bound = dict(old["bound_files"])
    bound[str(PARENT / "plan.json")] = sha(PARENT / "plan.json")
    bound[str(CONSUMED)] = sha(CONSUMED)
    reuse = Path("/scratch/djy8hg/runs/evicastkv/precision-rate-distortion-20261031-v1/workers/0")
    reused_result = previous.read(reuse / "result.json")
    if reused_result["status"] != "completed" or reused_result["failures"]:
        raise ValueError("only the completed label-affected canary may be reused")
    # Numerical code is unchanged; only wrapper output labels and CPU recovery
    # differ from the original producer. Preserve that original producer SHA.
    for relative in (
        "mosaickv/src/mosaickv/evaluation/head_local_precision.py",
        "mosaickv/src/mosaickv/evaluation/request_precision.py",
        "mosaickv/scripts/evicastkv_request_validation.py",
    ):
        previous_content = subprocess.check_output(
            ["git", "show", f"{reused_result['source_sha']}:{relative}"], cwd=SOURCE
        )
        import hashlib

        if sha(SOURCE / relative) != hashlib.sha256(previous_content).hexdigest():
            raise ValueError("cannot reuse canary after numerical or worker changes")
    bound[str(reuse / "result.json")] = sha(reuse / "result.json")
    for name, digest in reused_result["outputs"].items():
        if sha(reuse / name) != digest:
            raise ValueError("reused canary artifact changed")
        bound[str(reuse / name)] = digest
    chosen = [p for j in old["jobs"][:2] for p in j["inputs"]]
    if len(chosen) != 16 or len(set(chosen)) != 16:
        raise ValueError("exactly the first sixteen already-consumed ledger inputs required")
    for index in (0, 1):
        folder = PARENT / f"workers/{index}"
        result = previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("reused reference producer incomplete")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("reused reference artifact changed")
            bound[str(folder / name)] = digest
    protocol = SOURCE / "mosaickv/docs/precision_rate_distortion_20261031.md"
    bound[str(protocol)] = sha(protocol)
    jobs = []
    # Longest selector first for bounded six-worker scheduling.
    for method in (
        "lookm_reimpl",
        "vl_cache_reimpl",
        "prefixkv_reimpl",
        "prompt_attention_topk",
        "mosaickv_exact",
    ):
        for start in range(0, 16, 4):
            jobs.append(
                dict(
                    index=len(jobs),
                    inputs=chosen[start : start + 4],
                    method_arms=[[method, policy] for policy in POLICIES],
                )
            )
    plan = dict(
        old,
        source_sha=clean_source(SOURCE),
        bound_files=bound,
        jobs=jobs,
        stage="precision_rate_distortion_consumed_development",
        arms=[],
        warmups=1,
        timed_repetitions=1,
        precision_policies=POLICIES,
        reused_reference_root=str(PARENT),
        selected_inputs=chosen,
        independent_quality_N=0,
        development_sample_count=16,
        no_fresh_or_paper_promotion=True,
        automatic_bounded_cohort_after_correctness=False,
        paper_campaign_authorized=False,
        maximum_a100s=6,
        cpus_per_worker=12,
        bootstrap_replicates=10000,
        bootstrap_seed=20261031,
        no_shared_encoded_byte_matching_claim=True,
        no_unchanged_128_token_or_FullKV_quality_preservation_claim=True,
        preserved_negative_guard_result=str(CONSUMED),
        reused_worker_zero=dict(
            root=str(reuse),
            source_sha=reused_result["source_sha"],
            plan_sha256=reused_result["plan_sha256"],
        ),
    )
    write_once(root / "plan.json", plan)
    print(sha(root / "plan.json"))


def normalize_precision_row(row):
    """Derived label recovery only; original immutable raw records are retained."""
    if row.get("status") == "failed" and row.get("precision") in POLICIES:
        return row
    policy = row.get("development_precision_policy")
    if (
        policy not in POLICIES
        or row.get("online_variant", {}).get("precision") != policy
        or not row.get("method_label", "").endswith("_" + policy)
    ):
        raise ValueError("ambiguous or conflicting precision provenance")
    if row["precision"] == policy:
        return row
    if row["precision"] != "query3":
        raise ValueError("unregistered precision label cannot be recovered")
    return dict(
        row, precision=policy, original_precision_field="query3", derived_label_only_recovery=True
    )


def analyze(root, expected):
    import numpy as np
    import pandas as pd

    from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
    from mosaickv.evaluation.metrics.types import MetricSample

    plan = previous.common.inputs(root, expected)
    failures, rows, hashes = [], [], {}
    for job in plan["jobs"]:
        reuse = plan.get("reused_worker_zero") if job["index"] == 0 else None
        out = Path(reuse["root"]) if reuse else root / f"workers/{job['index']}"
        if not (out / "result.json").exists():
            failures.append(dict(index=job["index"], error="missing producer"))
            continue
        result = previous.read(out / "result.json")
        if result["source_sha"] != (reuse["source_sha"] if reuse else plan["source_sha"]) or result[
            "plan_sha256"
        ] != (reuse["plan_sha256"] if reuse else expected):
            raise ValueError("producer provenance mismatch")
        hashes[str(out / "result.json")] = sha(out / "result.json")
        for name, digest in result["outputs"].items():
            if sha(out / name) != digest:
                raise ValueError("raw artifact changed")
        failures.extend(result["failures"])
        if (out / "raw.jsonl").exists():
            rows.extend(
                normalize_precision_row(json.loads(line))
                for line in (out / "raw.jsonl").read_text().splitlines()
            )
    scored = [r for r in rows if not r["warmup"]]
    write_once(
        root / "derived/normalized_raw.json",
        dict(
            source_sha=plan["source_sha"],
            original_artifact_hashes=hashes,
            transformation=(
                "precision label only; no outputs, numbers, or producer identities modified"
            ),
            rows=rows,
        ),
    )
    keys = [(r["sample_id"], r["method"], r["precision"]) for r in scored]
    complete = (
        len(scored) == 160
        and len(set(keys)) == 160
        and not failures
        and all(r["status"] == "completed" for r in scored)
    )
    reference_rows = []
    for index in (0, 1):
        reference_rows.extend(
            json.loads(line)
            for line in (PARENT / f"workers/{index}/raw.jsonl").read_text().splitlines()
        )
    references = {
        s["sample_id"]: tuple(s["references"]) for s in previous.read(plan["references"])["samples"]
    }
    chosen_ids = {previous.read(p)["sample_id"] for p in plan["selected_inputs"]}
    old_index = {
        (r["sample_id"], r["method"], r["precision"]): r for r in reference_rows if not r["warmup"]
    }
    drift = [
        dict(sample_id=r["sample_id"], method=r["method"])
        for r in scored
        if r["status"] == "completed"
        and r["precision"] == "union3"
        and r["logits_sha256"]
        != old_index[(r["sample_id"], r["method"], "query3")]["logits_sha256"]
    ]
    gate = dict(
        complete=complete,
        expected_rows=160,
        scored_rows=len(scored),
        failures=failures,
        unchanged_union_transport_drift=drift,
        independent_quality_N=0,
        development_sample_count=16,
        no_paper_promotion=True,
        superiority_established=False,
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        input_hashes=hashes,
    )
    output = root / "analysis"
    if complete and not drift:
        groups = {}
        for r in scored:
            groups.setdefault((r["method"] + ":" + r["precision"], 0.5), []).append(
                MetricSample(
                    r["sample_id"], r["scored_answer"]["answer"], references[r["sample_id"]]
                )
            )
        full = [old_index[(i, "full_kv", "ordinary")] for i in sorted(chosen_ids)]
        groups[("full_kv:ordinary", 0.5)] = [
            MetricSample(r["sample_id"], r["scored_answer"]["answer"], references[r["sample_id"]])
            for r in full
        ]
        scorer = ResampledCider(groups)
        counts = np.vstack(
            (
                np.ones((1, 16), dtype=np.int64),
                paired_draw_counts(16, 10000, plan["bootstrap_seed"]),
            )
        )
        values = scorer.score(counts, batch_size=16)
        error = scorer.validate_official(counts[:4], values[:4])
        metrics = []
        for i, point in enumerate(scorer.points):
            group = [r for r in scored if (r["method"] + ":" + r["precision"]) == point[0]]
            item = dict(
                point=point[0],
                cider=float(values[0, i]),
                sample_count=16,
                evaluator_commit=plan["source_sha"],
                plan_sha256=expected,
                reference_artifact_sha256=sha(Path(plan["references"])),
                result_category="unified_comparison",
                result_origin="locally_measured",
                measurement_type="derived",
                quality_reference_only=not group,
            )
            if group:
                item.update(
                    mean_e2e_seconds=float(np.mean([r["end_to_end_seconds"] for r in group])),
                    **{
                        field: float(np.mean([r["storage"][field] for r in group]))
                        for field in (
                            "initial_active_kv_storage_bytes",
                            "final_active_kv_storage_bytes",
                            "peak_active_kv_storage_bytes",
                        )
                    },
                )
            metrics.append(item)
        candidate = scorer.points.index(("mosaickv_exact:head3k2v_refit", 0.5))
        comparisons = []
        for i, point in enumerate(scorer.points):
            if i == candidate or point[0] == "full_kv:ordinary":
                continue
            delta = values[1:, candidate] - values[1:, i]
            lo, hi = np.quantile(delta, [(0.05 / 9) / 2, 1 - (0.05 / 9) / 2])
            comparisons.append(
                dict(
                    baseline=point[0],
                    cider_difference=float(values[0, candidate] - values[0, i]),
                    simultaneous_ci95_bonferroni=[float(lo), float(hi)],
                )
            )
        output.mkdir(parents=True, exist_ok=False)
        pd.DataFrame(metrics).to_parquet(output / "metrics.parquet", index=False)
        write_once(
            output / "comparisons.json",
            dict(
                comparisons=comparisons,
                metrics=metrics,
                official_reducer_max_error=error,
                corpus_recomputed_inside_every_bootstrap=True,
                not_byte_matched=True,
                same_consumed_development_inputs=True,
            ),
        )
    write_once(output / "GATE.json", gate)
    print(json.dumps(gate))


def canary_valid(result, rows, old_rows):
    scored = [r for r in rows if not r["warmup"]]
    old = {
        (r["sample_id"], r["method"]): r
        for r in old_rows
        if not r["warmup"] and r["precision"] == "query3"
    }
    ids = {r["sample_id"] for r in scored}
    return (
        result["status"] == "completed"
        and not result["failures"]
        and len(scored) == 8
        and len(ids) == 4
        and {(r["sample_id"], r["precision"]) for r in scored}
        == {(i, p) for i in ids for p in POLICIES}
        and {r["method"] for r in scored} == {"lookm_reimpl"}
        and all(r["status"] == "completed" for r in scored)
        and all(
            r["logits_sha256"] == old[(r["sample_id"], r["method"])]["logits_sha256"]
            for r in scored
            if r["precision"] == "union3"
        )
    )


def promote(root, expected):
    plan = previous.common.inputs(root, expected)
    out = root / "workers/0"
    if not (out / "result.json").exists():
        analyze(root, expected)
        return
    result = previous.read(out / "result.json")
    for name, digest in result["outputs"].items():
        if sha(out / name) != digest:
            raise ValueError("canary artifact changed")
    rows = (
        [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
        if (out / "raw.jsonl").exists()
        else []
    )
    old = [json.loads(line) for line in (PARENT / "workers/0/raw.jsonl").read_text().splitlines()]
    passed = canary_valid(result, rows, old)
    write_once(
        root / "canary_decision.json",
        dict(
            passed=passed,
            source_sha=plan["source_sha"],
            result_sha256=sha(out / "result.json"),
            no_quality_success_claim=True,
        ),
    )
    if passed:
        submit(root, expected, remaining=True)
    else:
        analyze(root, expected)


def submit(root, expected, *, remaining=False):
    plan = previous.common.inputs(root, expected)
    if plan.get("reused_worker_zero") and not remaining:
        reused = Path(plan["reused_worker_zero"]["root"])
        recovered = [
            normalize_precision_row(json.loads(line))
            for line in (reused / "raw.jsonl").read_text().splitlines()
        ]
        old = [
            json.loads(line) for line in (PARENT / "workers/0/raw.jsonl").read_text().splitlines()
        ]
        if not canary_valid(previous.read(reused / "result.json"), recovered, old):
            raise ValueError("recovered canary did not pass unchanged execution/hash checks")
        write_once(
            root / "canary_decision.json",
            dict(
                passed=True,
                reused_completed_canary=True,
                original_result_sha256=sha(reused / "result.json"),
                original_producer_sha=plan["reused_worker_zero"]["source_sha"],
                source_sha=plan["source_sha"],
                label_only_recovery=True,
                no_quality_success_claim=True,
            ),
        )
        remaining = True
    cpu = previous.read(root / "cpu.json")
    if (
        cpu["source_sha"] != plan["source_sha"]
        or cpu["returncode"]
        or cpu["failed_or_skipped"]
        or cpu["test_count"] < 2849
        or cpu["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("full clean CPU proof required")
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in line.lower() for line in queue.splitlines()):
        raise ValueError("six-slot admission requires other owned GPU work complete")
    (root / "slurm").mkdir(exist_ok=remaining)
    script = str(SOURCE / "mosaickv/slurm/evicastkv_precision_rate_distortion.sbatch")
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
            "--job-name=mk-rate-dist",
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
            "--job-name=mk-rate-analysis",
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "cpu", "submit", "worker", "analyze", "promote"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    args = p.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
        submit(args.root, args.plan_sha)
    elif args.mode == "submit":
        submit(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.mode == "promote":
        promote(args.root, args.plan_sha)
    else:
        analyze(args.root, args.plan_sha)
