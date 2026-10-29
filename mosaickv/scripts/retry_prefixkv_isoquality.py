#!/usr/bin/env python3
"""Retry only MM-Vet Top-K/80 MiB after a provenance timeout; reuse 13 workers."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from prefixkv_isoquality_development import (
    SOURCE,
    completed_worker,
    read,
    sbatch,
    validate_protocol,
    validate_rows,
)

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

PARENT_SHA = "dfc2c21623e239ff816b4fd2c8ef235dc81569b2"
PARENT_PLAN_SHA = "5b6db91e3856428fdc4a882a8f082c3ff73917f459cccf51c581479c56dc3c7a"


def prepare(parent, root):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("retry output must be a new immutable root")
    if sha(parent / "plan.json") != PARENT_PLAN_SHA:
        raise ValueError("retry must use the exact failed parent plan")
    original = read(parent / "plan.json")
    validate_protocol(original["protocol"])
    if original["source_sha"] != PARENT_SHA:
        raise ValueError("historical producer SHA changed")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", PARENT_SHA, revision, "--", "mosaickv/src"],
        cwd=SOURCE,
        text=True,
    ).splitlines()
    if changed != ["mosaickv/src/mosaickv/manifest.py"]:
        raise ValueError("retry may change provenance only, never model/cache algorithms")
    for path, expected in original["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"historical bound input changed: {path}")
    imported = {}
    for job in original["jobs"]:
        index = job["index"]
        worker = completed_worker(parent, original, PARENT_PLAN_SHA, index)
        if index == 8:
            if (
                worker is None
                or worker["status"] != "failed"
                or (job["task"], job["method"], job["cap_mib"])
                != ("mmvet_prefixkv", "prompt_attention_topk", 80)
                or "TimeoutExpired" not in worker["error"]
                or "git" not in worker["error"]
                or "status" not in worker["error"]
            ):
                raise ValueError("retry is restricted to the recorded provenance failure")
        elif worker is None or worker["status"] != "completed":
            raise ValueError("all other thirteen workers must already be completed")
        else:
            validate_rows(worker["record"]["rows"], job["sample_ids"], job["cap_mib"])
            for row in worker["record"]["rows"]:
                provenance = read(row["manifest_path"])["source"]
                if (
                    provenance["git_sha"] != PARENT_SHA
                    or provenance["config_sha"] != job["config_sha"]
                    or provenance["git_dirty"]
                    or not provenance["canonical_eligible"]
                ):
                    raise ValueError("import requires clean, correctly bound original manifests")
            path = parent / f"workers/{index:02d}.json"
            imported[str(index)] = dict(
                path=str(path), sha256=sha(path), plan_sha256=PARENT_PLAN_SHA, source_sha=PARENT_SHA
            )
    if len(imported) != 13:
        raise ValueError("exactly thirteen unchanged successful workers must be reused")
    root.mkdir(parents=True)
    revised = copy.deepcopy(original)
    revised.update(
        source_sha=revision,
        imported_workers=imported,
        retry=dict(
            parent_root=str(parent),
            parent_plan_sha256=PARENT_PLAN_SHA,
            retry_indices=[8],
            new_quality_requests=8,
            reused_successful_sample_rows=104,
            reason="bounded fail-closed Git-status retry only",
            experiment_controls_unchanged=True,
        ),
    )
    revised["prefixkv_admission_ceiling"] = 1
    revised["scheduling_amendment"] = dict(
        user_authorized_fox_ceiling=5,
        prefixkv_ceiling=1,
        aggregate_ceiling=6,
        no_running_jobs_cancelled=True,
    )
    revised["jobs"][8]["output"] = str(root / "outputs/08")
    # Correct only the old prospective count annotation: the already registered
    # two tasks x two IDs x twenty repetitions x two methods is 160, not 80.
    revised["maximum_conditional_timing_requests_per_worker"] = 80
    revised["maximum_conditional_timing_requests"] = 160
    revised["bound_files"][str(parent / "plan.json")] = PARENT_PLAN_SHA
    revised["bound_files"][str(parent / "workers/08.json")] = sha(parent / "workers/08.json")
    write_once(root / "plan.json", revised)
    write_once(
        root / "retry_provenance.json",
        dict(
            original_failed_worker=str(parent / "workers/08.json"),
            original_raw_outputs_preserved=True,
            imported_workers=imported,
            old_source_sha=PARENT_SHA,
            retry_source_sha=revision,
            only_changed_package_file=changed[0],
            interpretation="development_only",
            candidate_and_baseline_selection_unchanged=True,
        ),
    )
    print(
        json.dumps(
            dict(
                root=str(root),
                plan_sha=sha(root / "plan.json"),
                retry_indices=[8],
                reused_workers=13,
            )
        )
    )


def submit(root, plan_sha):
    from control_a100_schedule import ACTIVE, gpu_count, read_queue
    from prefixkv_isoquality_development import inputs

    plan = inputs(root, plan_sha)
    if plan["retry"]["retry_indices"] != [8] or len(plan["imported_workers"]) != 13:
        raise ValueError("retry scope changed")
    for index in plan["imported_workers"]:
        completed_worker(root, plan, plan_sha, int(index))
    # No other campaign is submitted. Timing remains conditional on the same
    # registered quality/memory gates and uses at most one additional A100.
    occupied = sum(gpu_count(r["gres"]) for r in read_queue() if r["state"] in ACTIVE)
    if occupied >= 6:
        raise ValueError("six-A100 aggregate ceiling reached")
    write_once(root / "decisions/retry-started.json", dict(plan_sha256=plan_sha, retry=[8]))
    job = sbatch(root, plan_sha, mode="worker", indices=[8])
    callback = sbatch(root, plan_sha, mode="analyze", dependency=job)
    write_once(
        root / "decisions/retry.json",
        dict(
            gpu_array=job,
            cpu_callback=callback,
            reused_workers=13,
            retry_workers=1,
            existing_results_untouched=True,
        ),
    )
    write_once(
        root / "scheduler/policy.json",
        dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            window="all_times",
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[job],
            retry_worker_count=1,
            only_registered_worker_index=8,
        ),
    )
    from control_a100_schedule import tick

    policy = read(root / "scheduler/policy.json")
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    print(json.dumps(dict(gpu_array=job, analysis=callback, retry_workers=1, reused_workers=13)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit"])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        parser.error("scratch-only retry output required")
    if args.action == "prepare":
        if args.parent is None:
            parser.error("--parent required")
        prepare(args.parent, args.root)
    else:
        if args.plan_sha is None:
            parser.error("--plan-sha required")
        submit(args.root, args.plan_sha)
