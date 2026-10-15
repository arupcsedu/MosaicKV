#!/usr/bin/env python3
"""Stage promotion, selective submission, and final sample-size registration.

Print submission commands by default. --submit creates only a HELD array;
it cannot bypass the shared controller's global four-A100 admission policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_coco_shared_bytes import completed_sample, verified_success, verify_plan
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.evaluation.coco_shared_bytes import sample_size_recommendation


def missing_indices(plan, plan_sha, stage):
    plan = dict(plan, _plan_sha=plan_sha)
    indices = []
    for index, job in enumerate(plan["jobs"]):
        if job["stage"] != stage:
            continue
        if any(completed_sample(plan, job, sid) is None for sid in job["sample_ids"]):
            indices.append(index)
    return indices


def passed_report(root, stage, plan_sha):
    path = root / "analysis" / stage / "report.json"
    report = json.loads(path.read_text())
    if (
        report.get("status") != "passed"
        or report.get("plan_sha256") != plan_sha
        or report.get("missing_or_failed")
    ):
        raise ValueError("complete validated stage required")
    for name, sha in report["derived_artifact_hashes"].items():
        if digest(Path(name)) != sha:
            raise ValueError("derived analysis changed")
    return path, report


def main(args):
    root = args.root.resolve()
    plan_path = root / "plan.json"
    if digest(plan_path) != args.plan_sha:
        raise ValueError("plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, plan["source_sha"])
    if args.action == "promote":
        path, _ = passed_report(root, "validation", args.plan_sha)
        write_once(
            root / "larger-authorization.json",
            dict(stage="larger", plan_sha256=args.plan_sha, validation_sha256=digest(path)),
        )
        return
    if args.action == "register-final":
        path, report = passed_report(root, "larger", args.plan_sha)
        comparisons = [
            r
            for r in report["comparisons"]
            if r["metric_name"] == "cider" and r["baseline"] != "full_kv"
        ]
        widest = max(comparisons, key=lambda r: r["ci_high"] - r["ci_low"])
        # Includes all measured arms at all caps, FullKV once, not just MosaicKV.
        seconds_per_sample = sum(a["end_to_end_time"] or 0 for a in report["aggregates"])
        full_times, worker_times = [], []
        for job in plan["jobs"]:
            if job["stage"] != "larger":
                continue
            for worker in Path(job["output"]).glob("attempt-*/worker.json"):
                measured = json.loads(worker.read_text())
                worker_times.append(measured["worker_wall_seconds"])
            if job["method"] == "full_kv":
                for sid in job["sample_ids"]:
                    record = verified_success(
                        Path(job["output"]) / "samples" / hashlib.sha256(sid.encode()).hexdigest(),
                        args.plan_sha,
                        job["config_sha"],
                    )
                    full_times.append(record["rows"][0]["end_to_end_time"])
        seconds_per_sample += statistics.mean(full_times)
        # Wall-cost estimate includes repeated model loads, parity and warmups.
        wall_cost = max(seconds_per_sample, sum(worker_times) / report["planned_samples"])
        recommendation = sample_size_recommendation(
            pilot_n=256,
            ci_low=widest["ci_low"],
            ci_high=widest["ci_high"],
            target_half_width=args.cider_ci_half_width,
            seconds_per_example=wall_cost,
            available=min(4096, plan["full_pool_size"]),
        )
        write_once(
            root / "final-preregistration.json",
            dict(
                schema_version=1,
                plan_sha256=args.plan_sha,
                pilot_report_sha256=digest(path),
                sample_count=min(4096, plan["full_pool_size"]),
                target_ci_half_width_score_points=args.cider_ci_half_width,
                recommendation=recommendation,
                caps_mib=[10, 20, 40, 80],
                cohort_policy="ID-only hash rank; disclosed pool; compatible pilot reuse",
                candidate_retuned=False,
                fullkv_latency_comparison=False,
                heldout_claim=False,
                prior_use=plan["prior_use_disclosures"],
                execution_authorized=False,
                note="precision is an estimate; retain negative results",
            ),
        )
        return
    indices = missing_indices(plan, args.plan_sha, args.stage)
    if not indices:
        print(json.dumps(dict(status="already_complete", submitted=False)))
        return
    verify_plan(root, args.plan_sha, indices[0])
    log_root = root / "slurm"
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--array={','.join(map(str, indices))}%4",
        f"--output={log_root}/%A_%a.out",
        f"--error={log_root}/%A_%a.err",
        str(source / "mosaickv/slurm/coco_shared_bytes.sbatch"),
        str(source),
        str(root),
        args.plan_sha,
    ]
    if args.submit:
        if args.controller_policy is None:
            raise ValueError(
                "shared controller policy required; per-array throttle is not a global cap"
            )
        policy = json.loads(args.controller_policy.read_text())
        if policy.get("maximum_worker_minutes") != 120:
            raise ValueError("unexpected controller policy")
        # Fail closed unless the tested controller actually enforces four at
        # every time, not a historical overnight-eight schedule.
        from datetime import datetime

        import control_a100_schedule as controller

        if controller.limits(datetime.now(controller.ZONE)) != (4, 4):
            raise ValueError("controller does not enforce global four-A100 rule")
        queue = controller.read_queue()
        registered = set(map(str, policy["arrays"]))
        if any(
            controller.gpu_count(r["gres"])
            and r["state"] == "PENDING"
            and r["array"] not in registered
            for r in queue
        ):
            raise ValueError("unregistered pending A100 jobs could exceed aggregate cap")
        log_root.mkdir(parents=True, exist_ok=True)
        job_id = (
            subprocess.run(command, check=True, text=True, capture_output=True)
            .stdout.strip()
            .split(";")[0]
        )
        # No controller mutation races: produce a new policy. A controller
        # operator must atomically replace its policy under its existing lock.
        policy["arrays"] = [*policy["arrays"], job_id]
        policy["adopt_existing_holds"] = [*policy.get("adopt_existing_holds", []), job_id]
        write_once(root / f"controller-policy-{job_id}.json", policy)
        write_once(
            root / f"submission-{job_id}.json",
            dict(
                job_id=job_id,
                indices=indices,
                command=command,
                status="held_pending_shared_controller_registration",
                plan_sha256=args.plan_sha,
            ),
        )
        print(
            json.dumps(
                dict(
                    job_id=job_id,
                    held=True,
                    controller_policy=str(root / f"controller-policy-{job_id}.json"),
                )
            )
        )
    else:
        print(json.dumps(dict(command=command, indices=indices, submitted=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("action", choices=("submit", "promote", "register-final"))
    parser.add_argument("--stage", choices=("validation", "larger", "final"), default="validation")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--controller-policy", type=Path)
    parser.add_argument("--cider-ci-half-width", type=float, default=2.0)
    main(parser.parse_args())
