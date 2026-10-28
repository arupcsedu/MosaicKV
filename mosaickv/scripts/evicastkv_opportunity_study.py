#!/usr/bin/env python3
"""Bounded consumed-input canary; no automatic fresh-cohort or quality claim."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
from pathlib import Path

import yaml
from evicastkv_analyze import collect, pairing
from evicastkv_campaign import admitted_array_concurrency, environment, load_lock, resolved_config

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
STAGE = "evidence_canary"
REGISTRATION = SOURCE / "mosaickv/configs/experiments/evicastkv_opportunity_demand_v1.yaml"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/fox-review-20261028-v6-coco-scoped")
VARIANTS = ("complete", "evidence-routing-v1", "evidence-coverage-v1", "evidence-both-v1")


def arms():
    return [("full_kv", None, "complete")] + [
        (method, cap, variant)
        for cap in (20, 40)
        for method, variant in [
            ("prompt_attention_topk", "complete"),
            ("vl_cache_reimpl", "complete"),
            *(("mosaickv_exact", variant) for variant in VARIANTS),
        ]
    ]


def prepare(root):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("use a new immutable canary root")
    registration = yaml.safe_load(REGISTRATION.read_text())
    if registration["budgets_mib"] != [20, 40] or registration["variants"] != list(VARIANTS):
        raise ValueError("registered budgets or ablation variants changed")
    if any(
        registration[key]
        for key in (
            "fresh_development_materialization_authorized",
            "confirmation_authorized",
            "paper_campaign_authorized",
        )
    ):
        raise ValueError("this runner only authorizes consumed-input correctness")
    parent_lock_path = PARENT / "protocol_lock.json"
    lock = json.loads(parent_lock_path.read_text())
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    control_path = PARENT / "analysis/correctness/GATE_correctness.json"
    if json.loads(control_path.read_text())["status"] != "passed":
        raise ValueError("reference producer correctness is not passed")
    original = lock["cohorts"]["coco"]["validation"]
    ledger_path = Path(original["path"])
    if sha(ledger_path) != original["sha256"]:
        raise ValueError("consumed input ledger changed")
    ledger = json.loads(ledger_path.read_text())
    if [s["sample_id"] for s in ledger["samples"]] != original["sample_ids"]:
        raise ValueError("parent ledger identity order changed")
    ledger["samples"] = ledger["samples"][:2]
    for sample in ledger["samples"]:
        for media in sample["media"]:
            if sha(Path(media["path"])) != media["sha256"]:
                raise ValueError("consumed media hash changed")
    root.mkdir(parents=True)
    path = root / "manifests/consumed-canary.json"
    write_once(path, ledger)
    lock = copy.deepcopy(lock)
    lock.update(
        name="evicastkv_opportunity_demand_v1_consumed_canary",
        source_root=str(SOURCE),
        source_sha=revision,
        environment=environment(),
        stages=[STAGE],
        resources=registration["resources"],
        cohorts=dict(
            coco=dict(
                consumed_canary=dict(
                    path=str(path),
                    sha256=sha(path),
                    sample_ids=[s["sample_id"] for s in ledger["samples"]],
                )
            )
        ),
        evidence_study=dict(
            policy=registration["policy"],
            registration_sha256=sha(REGISTRATION),
            parent_protocol_sha256=sha(parent_lock_path),
            fresh_confirmation=False,
            diagnostic_timing_not_headline=True,
            fresh_development_authorized=False,
        ),
    )
    lock["historical_bindings"][str(REGISTRATION)] = sha(REGISTRATION)
    lock["historical_bindings"][str(parent_lock_path)] = sha(parent_lock_path)
    lock["historical_bindings"][str(control_path)] = sha(control_path)
    for method, cap, variant in arms():
        resolved_config(lock, "coco", method, cap, variant=variant)
    write_once(root / "protocol_lock.json", lock)
    jobs = [
        dict(workload="coco", stage=STAGE, subset="consumed_canary", indices=[i], arms=arms())
        for i in range(2)
    ]
    write_once(
        root / f"jobs/{STAGE}.json",
        dict(stage=STAGE, protocol_sha256=sha(root / "protocol_lock.json"), jobs=jobs),
    )
    write_once(
        root / "readiness.json",
        dict(
            status="prepared_not_executed",
            requests=26,
            source_sha=revision,
            fresh_confirmation=False,
            quality_claim_allowed=False,
        ),
    )


def analyze(root):
    lock = load_lock(root)
    output = root / "analysis/evidence_canary"
    if output.exists():
        raise FileExistsError("analysis is write-once")
    errors, summaries = [], []
    rows, workers, pairs = [], [], []
    try:
        _, workers, rows, errors = collect(root, STAGE)
        if len(rows) != 26 or len(workers) != 2:
            errors.append(
                dict(error="incomplete registered denominator", actual=len(rows), expected=26)
            )
        pairs = pairing(rows)
        if len(pairs) != 4:
            errors.append(dict(error="expected two samples times two paired budgets"))
        for row in rows:
            if row["status"] != "completed":
                continue
            expected = row["method"]
            if row["variant"].startswith("evidence-"):
                name = {
                    "evidence-routing-v1": "routing_only",
                    "evidence-coverage-v1": "coverage_only",
                    "evidence-both-v1": "both",
                }[row["variant"]]
                expected += f"__evidence_{name}_v1"
            if row["effective_method"] != expected:
                raise ValueError("unexpected fallback or experimental method label")
            if row["emitted_tokens"] != 64 or row["forward_steps"] != 63:
                raise ValueError("fixed emission/append schedule changed")
            trace = json.loads(Path(row["trace_path"]).read_text())
            activity = trace.get("experimental_evidence_policy")
            if row["variant"].startswith("evidence-") and not activity:
                raise ValueError("missing experimental policy trace")
            if row["budget_bytes"] is not None:
                if row["initial_kv_bytes"] != row["budget_bytes"]:
                    raise ValueError("realized initial cache does not equal shared cap")
                key = row["run_id"]
                # Actual trace catalogs may be nested; locate the worker's output directory.
                worker_paths = list(
                    (root / "outputs" / STAGE).glob("*/*/correctness/" + key + ".json")
                )
                if len(worker_paths) != 1:
                    raise ValueError("missing independent packed-cache correctness report")
                correctness = worker_paths[0]
                report = json.loads(correctness.read_text())
                if (
                    report["status"] != "passed"
                    or report["retained_entries"] != report["padded_entries"]
                ):
                    raise ValueError("representation parity or zero-padding gate failed")
            summaries.append(
                dict(
                    sample_id=row["sample_id"],
                    variant=row["variant"],
                    method=row["method"],
                    cap=row["budget_bytes"],
                    initial_bytes=row["initial_kv_bytes"],
                    final_bytes=row["final_kv_bytes"],
                    peak_bytes=row["peak_kv_bytes"],
                    experimental_activity=activity,
                    diagnostic_wall_seconds=row["e2e_wall_seconds"],
                )
            )
    except Exception as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
    status = "passed_consumed_correctness_only" if not errors else "failed"
    write_once(
        output / "GATE_evidence_canary.json",
        dict(
            status=status,
            source_sha=lock["source_sha"],
            protocol_sha256=sha(root / "protocol_lock.json"),
            expected_rows=26,
            actual_rows=len(rows),
            paired_budgets=len(pairs),
            errors=errors,
            quality_claim_allowed=False,
            fresh_confirmation=False,
            calibration_submitted=False,
        ),
    )
    write_once(output / "activity.json", summaries)
    (output / "report.md").write_text(
        f"# Opportunity/demand consumed-input canary\n\nStatus: {status}.\n\n"
        f"Rows: {len(rows)}/26; paired input/budget groups: {len(pairs)}/4.\n\n"
        "This is instrumented representation/activity validation, not benchmark-quality "
        "superiority or headline timing. No fresh cohort or calibration was submitted.\n\n"
        + "\n".join(str(e) for e in errors)
        + "\n"
    )


def submit(root):
    lock = load_lock(root)
    if (root / "jobs/submission.json").exists():
        raise FileExistsError("canary submission already exists")
    active = subprocess.check_output(
        [
            "squeue",
            "-h",
            "-u",
            os.environ["USER"],
            "-t",
            "RUNNING,COMPLETING,PENDING",
            "-o",
            "%i|%b",
        ],
        text=True,
    )
    total = sum(
        int(match[1])
        for line in active.splitlines()
        if (match := re.search(r"gpu(?::[a-zA-Z0-9_-]+)?:(\d+)", line))
    )
    if admitted_array_concurrency(lock["resources"], total) < 2:
        raise ValueError("insufficient aggregate GPU admission; preserve existing jobs")
    (root / "slurm").mkdir(exist_ok=True)
    common = ["sbatch", "--parsable", "--account=bii_dsc_community"]
    spec_sha = sha(root / f"jobs/{STAGE}.json")
    command = [
        *common,
        "--array=0-1%2",
        f"--output={root}/slurm/canary-%A_%a.out",
        f"--error={root}/slurm/canary-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_opportunity_canary.sbatch"),
        str(SOURCE),
        str(root),
        "worker",
        spec_sha,
    ]
    gpu = subprocess.check_output(command, text=True).strip().split(";")[0]
    callback = [
        *common,
        "--partition=bii",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:10:00",
        "--gres=none",
        f"--dependency=afterany:{gpu}",
        f"--output={root}/slurm/analysis-%j.out",
        f"--error={root}/slurm/analysis-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_opportunity_analysis.sbatch"),
        str(SOURCE),
        str(root),
    ]
    cpu = subprocess.check_output(callback, text=True).strip().split(";")[0]
    write_once(
        root / "jobs/submission.json",
        dict(array=gpu, analysis=cpu, command=command, callback=callback),
    )
    print(json.dumps(dict(array=gpu, analysis=cpu)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze", "submit"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--spec-sha")
    args = parser.parse_args()
    if args.mode == "worker":
        from evicastkv_worker import run_worker

        run_worker(args.root, STAGE, int(os.environ["SLURM_ARRAY_TASK_ID"]), args.spec_sha)
    else:
        globals()[args.mode](args.root)
