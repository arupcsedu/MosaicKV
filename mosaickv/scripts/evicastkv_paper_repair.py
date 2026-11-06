#!/usr/bin/env python3
"""Bounded guard-bug recovery: preserve predictions; retry only invalid rows.

Never tunes a parameter, changes a tolerance or inspects benchmark scores to
select work. The historical unsuccessful attempts remain in raw_all_attempts.
"""

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

import evicastkv_paper_precision as campaign

from mosaickv.evaluation.fox_campaign import sha, write_once

SOURCE = campaign.SOURCE
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_paper_repair.sbatch"
EXPECTED_ERROR = "transport output not close under unchanged rtol=0.01, atol=2**-6 (commit)"


def submit(root, action, **kwargs):
    return campaign.submit(root, action, script=SCRIPT, **kwargs)


def collect(root, phase):
    """Verify producer hashes, preserving both successes and failures."""
    plan = campaign.read(root / "plan.json")
    jobs = campaign.read(root / phase / "jobs.json")
    rows = []
    for index, job in enumerate(jobs):
        folder = root / phase / f"workers/{index}"
        result = campaign.read(folder / "result.json")
        if (
            result["source_sha"] != plan["source_sha"]
            or result["plan_sha256"] != sha(root / "plan.json")
            or result["job"] != job
        ):
            raise ValueError("producer identity changed")
        if result["raw_sha256"] != sha(folder / "raw.jsonl") or result["manifest_sha256"] != sha(
            folder / "manifest.json"
        ):
            raise ValueError("producer evidence changed")
        records = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
        if any(
            r["source_sha"] != plan["source_sha"]
            or r["plan_sha256"] != sha(root / "plan.json")
            or r["method"] != job["method"]
            or r["sample_index"] not in job["samples"]
            for r in records
        ):
            raise ValueError("row provenance differs from producer")
        rows.extend(records)
    scored = [r for r in rows if not r["warmup"]]
    keys = [(r["sample_index"], r["method"]) for r in scored]
    expected = {(i, job["method"]) for job in jobs for i in job["samples"]}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing or duplicate sample-method rows; this is not guard-only recovery")
    return rows, scored


def failed_keys(rows):
    result = []
    for row in rows:
        if row["status"] == "completed":
            if len(row["generated_token_ids"]) != 128 or row["used_fullkv_fallback"]:
                raise ValueError("incomplete successful prediction")
            continue
        if row["method"] != "mosaickv_exact" or row.get("error", "").strip().splitlines()[-1:] != [
            "AssertionError: " + EXPECTED_ERROR
        ]:
            raise ValueError("failure is not the registered correction/verifier inconsistency")
        result.append((row["sample_index"], row["method"]))
    return sorted(result)


def prepare(root, parent):
    campaign.prepare(root, parent, autostart=False)
    current = sha(root / "plan.json")
    cpu = submit(root, "cpu", minutes=60)
    producer = campaign.read(parent / "quality/submission.json")["array"]
    select = submit(root, "select", dependency=f"{cpu}:{producer}", minutes=60)
    write_once(
        root / "submission.json",
        dict(cpu=cpu, select=select, parent_array=producer, plan_sha256=current),
    )


def cpu(root, expected):
    plan = campaign.checked(root, expected)
    with (root / "cpu_tests.txt").open("x") as stream:
        result = subprocess.run(
            [
                "/scratch/djy8hg/env/mosaickv/bin/python",
                "-m",
                "pytest",
                "mosaickv/tests",
                "-q",
                "-m",
                "not gpu",
            ],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    write_once(
        root / "cpu.json",
        dict(passed=result.returncode == 0, log_sha256=sha(root / "cpu_tests.txt")),
    )
    if result.returncode:
        return
    parent = Path(plan["reused_ledger_parent"])
    write_once(root / "ledger.json", campaign.load_ledger(parent))
    write_once(
        root / "ledger_seal.json",
        dict(campaign.read(parent / "ledger_seal.json"), plan_sha256=expected),
    )
    if sha(root / "ledger.json") != sha(parent / "ledger.json"):
        raise ValueError("cohort identity/reference changed")


def select(root, expected):
    plan = campaign.checked(root, expected)
    if not campaign.read(root / "cpu.json")["passed"]:
        raise ValueError("CPU suite failed; no GPU repair")
    parent = Path(plan["reused_ledger_parent"])
    _, scored = collect(parent, "quality")
    failures = failed_keys(scored)
    if not failures or len(failures) > 32:
        raise ValueError("bounded recovery requires 1--32 known guard failures")
    write_once(
        root / "selection.json",
        dict(
            failed_keys=failures,
            parent=str(parent),
            parent_plan_sha256=sha(parent / "plan.json"),
            selection_uses_only_execution_status_not_answers=True,
            correction_is_generic_numerical_bugfix_not_quality_tuning=True,
        ),
    )
    campaign.launch_phase(
        root, "canary", [dict(method="mosaickv_exact", samples=[failures[0][0]])], submitter=submit
    )


def canary_analysis(root, expected):
    _, scored = collect(root, "canary")
    if (
        len(scored) != 1
        or scored[0]["status"] != "completed"
        or scored[0].get("optimized_parent_128_logit_words_bitwise") is not True
    ):
        raise ValueError("corrected model-level guard/parent canary failed")
    row = scored[0]
    if sha(Path(row["logit_capture"])) != row["logit_capture_sha256"]:
        raise ValueError("diagnostic tensor capture changed")
    write_once(
        root / "canary/analysis/decision.json",
        dict(
            complete=True,
            source_sha=campaign.checked(root, expected)["source_sha"],
            sample_index=row["sample_index"],
            all_128_parent_logit_vectors_bitwise=True,
            thresholds_relaxed=False,
            no_independent_quality_claim=True,
        ),
    )
    keys = campaign.read(root / "selection.json")["failed_keys"]
    jobs = [
        dict(method="mosaickv_exact", samples=[int(k[0]) for k in keys[i : i + 8]])
        for i in range(0, len(keys), 8)
    ]
    campaign.launch_phase(root, "quality", jobs, submitter=submit)


def quality_analysis(root, expected):
    import numpy as np
    import pandas as pd

    from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
    from mosaickv.evaluation.metrics.types import MetricSample

    plan = campaign.checked(root, expected)
    parent = Path(plan["reused_ledger_parent"])
    old_raw, old_scored = collect(parent, "quality")
    new_raw, new_scored = collect(root, "quality")
    required = set(failed_keys(old_scored))
    retries = {(r["sample_index"], r["method"]): r for r in new_scored}
    if set(retries) != required or any(r["status"] != "completed" for r in retries.values()):
        raise ValueError("all and only the failed rows must be recovered")
    selected = [retries.get((r["sample_index"], r["method"]), r) for r in old_scored]
    if len(selected) != 512 or len({(r["sample_index"], r["method"]) for r in selected}) != 512:
        raise ValueError("complete 256-image pairing required")
    out = root / "quality/analysis"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "raw_all_attempts.jsonl").open("x") as stream:
        for row in [*old_raw, *new_raw]:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_once(out / "selected_quality_rows.json", selected)
    ledger = campaign.load_ledger(root)
    refs = {s["sample_id"]: tuple(s["references"]) for s in ledger["samples"]}
    methods = ("full_kv", "mosaickv_exact")
    groups = {
        m: [
            MetricSample(r["sample_id"], r["answer"]["answer"], refs[r["sample_id"]])
            for r in selected
            if r["method"] == m
        ]
        for m in methods
    }
    reducer = ResampledCider(groups)
    draws = paired_draw_counts(256, 10000, plan["protocol"]["seed"])
    scores = reducer.score(draws, batch_size=16)
    reducer.validate_official(draws[:3], scores[:3])
    counts = np.ones((1, 256), dtype=int)
    points = reducer.score(counts)[0]
    reducer.validate_official(counts, points[None])
    a, b = reducer.points.index("mosaickv_exact"), reducer.points.index("full_kv")
    interval = np.quantile(scores[:, a] - scores[:, b], [0.025, 0.975]).tolist()
    metrics = dict(
        candidate_cider=float(points[a]),
        fullkv_cider=float(points[b]),
        difference=float(points[a] - points[b]),
        paired_95_interval=interval,
    )
    pd.DataFrame([metrics]).to_parquet(out / "quality_comparisons.parquet", index=False)
    storage = []
    full = {r["sample_index"]: r for r in selected if r["method"] == "full_kv"}
    for row in selected:
        if row["method"] != "mosaickv_exact":
            continue
        ref = full[row["sample_index"]]
        storage.append(
            dict(
                sample_id=row["sample_id"],
                producer_sha=row["source_sha"],
                manifest_path=row["manifest_path"],
                initial_bytes=row["initial_encoded_bytes"],
                final_bytes=row["final_encoded_bytes"],
                decode_owned_peak_bytes=row["decode_owned_peak_bytes"],
                fullkv_initial_bytes=ref["storage"]["initial_active_kv_storage_bytes"],
                fullkv_final_bytes=ref["storage"]["final_active_kv_storage_bytes"],
                whole_request_peak_allocated=row["whole_request_peak_allocated_bytes"],
                fullkv_whole_request_peak_allocated=ref["whole_request_peak_allocated_bytes"],
            )
        )
    pd.DataFrame(storage).to_parquet(out / "storage.parquet", index=False)
    smaller = all(
        r["initial_bytes"] < r["fullkv_initial_bytes"]
        and r["final_bytes"] < r["fullkv_final_bytes"]
        and r["whole_request_peak_allocated"] < r["fullkv_whole_request_peak_allocated"]
        for r in storage
    )
    decision = dict(
        complete=True,
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        evaluation_N=256,
        observations=512,
        original_failed_rows=len(required),
        recovered_failed_rows=len(retries),
        reused_valid_rows=512 - len(retries),
        producer_shas=sorted({r["source_sha"] for r in selected}),
        reused_timings_and_whole_request_allocator_peaks_are_historical_per_producer=True,
        corrected_implementation_whole_request_peak_on_all_256_certified=False,
        historical_attempts_preserved=True,
        no_previously_valid_quality_prediction_rerun=True,
        bugfix_not_label_tuning=True,
        metrics=metrics,
        quality_preservation_pass=interval[0] >= 0,
        measured_memory_reduced=smaller,
        byte_matched_claim=False,
        whole_request_KV_only_peak_certified=False,
        paper_superiority_established=False,
        compressed_baseline_comparisons_not_executed=True,
    )
    write_once(out / "decision.json", decision)
    if decision["quality_preservation_pass"] and smaller:
        jobs = [
            dict(method="paired_methods", methods=["mosaickv_exact"], samples=[2 + 43 * i])
            for i in range(6)
        ]
        campaign.launch_phase(root, "timing", jobs, submitter=submit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--plan-sha")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.parent)
        return
    campaign.checked(args.root, args.plan_sha)
    try:
        if args.action in ("cpu", "select", "canary_analysis", "quality_analysis"):
            globals()[args.action](args.root, args.plan_sha)
        elif args.action in ("canary", "quality", "timing"):
            campaign.worker(
                args.root, args.plan_sha, args.action, int(os.environ["SLURM_ARRAY_TASK_ID"])
            )
        elif args.action == "timing_analysis":
            campaign.analyze(args.root, args.plan_sha, "timing")
        elif args.action.endswith("_controller"):
            phase = args.action.removesuffix("_controller")
            subprocess.run(
                [
                    "/scratch/djy8hg/env/mosaickv/bin/python",
                    str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                    "--policy",
                    str(args.root / phase / "scheduler/policy.json"),
                    "--state",
                    str(args.root / phase / "scheduler/state.json"),
                    "--apply",
                    "--watch",
                ],
                check=True,
            )
        else:
            raise ValueError("unknown bounded recovery action")
    except Exception:
        write_once(
            args.root / f"{args.action}_blocked.json",
            dict(
                complete=False,
                error=traceback.format_exc(),
                source_sha=campaign.clean_source(SOURCE),
                thresholds_relaxed=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
