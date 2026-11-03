#!/usr/bin/env python3
"""One phase-evidence-gated descriptor follow-up; no fresh quality campaign."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import evicastkv_paired_request_readers as driver

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = driver.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/paired-request-readers-consumed-20261102-retry1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_replay_descriptor_followup.sbatch"
DOC = SOURCE / "mosaickv/docs/replay_descriptor_followup_20261102.md"


def parent_dependency(root=None):
    # Prefer the terminal callback, not its still-running predecessor; this
    # avoids racing closeout publication after the all-method analyzer.
    root = PARENT if root is None else root
    for name, key in (
        ("dispatch/registered_closeout.json", "job_id"),
        ("dispatch/timing/submission.json", "cpu_analysis"),
        ("dispatch/submission.json", "cpu_analysis"),
        ("submission.json", "cpu_analysis"),
        ("precheck_submission.json", "cpu_begin"),
    ):
        path = root / name
        if path.exists():
            return str(json.loads(path.read_text())[key])
    raise ValueError("no registered immutable parent dependency")


def schedule(root, revision, dependency, step):
    if not 0 <= step <= 5:
        raise ValueError("bounded five-stage watcher only")
    launch = root.with_name(root.name + "-launch")
    launch.mkdir(exist_ok=True)
    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:20:00",
            f"--dependency=afterany:{dependency}",
            "--job-name=mk-descriptor-watch",
            f"--output={launch}/watch-{step}-%j.out",
            f"--error={launch}/watch-{step}-%j.err",
            str(SCRIPT),
            str(SOURCE),
            str(root),
            revision,
            str(step),
        ],
        text=True,
    ).strip()
    write_once(
        launch / f"watch-{step}.json",
        dict(
            job_id=job,
            dependency=dependency,
            source_sha=revision,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


def opportunity(profiles):
    rows = [r for r in profiles if r["method"] == "mosaickv_exact"]
    if len(rows) != 4 or {r["input_index"] for r in rows} != {0, 1}:
        raise ValueError("both consumed-input paired Replay traces required")
    costs = [
        sum(
            v["exclusive_wall_seconds"]
            for k, v in r["phases"].items()
            if k.endswith("/code_replay") and not k.startswith("prefill/")
        )
        for r in rows
    ]
    if any(not 0.1 <= x < float("inf") for x in costs):
        raise ValueError("insufficient/invalid measured host replay opportunity")
    return costs


def watch(root, revision, step):
    if clean_source(SOURCE) != revision or not 0 <= step <= 5:
        raise ValueError("clean immutable finite descriptor source required")
    launch = root.with_name(root.name + "-launch")
    launch.mkdir(exist_ok=True)
    closeout = PARENT / "engineering_closeout.json"
    if not closeout.exists():
        dependency = parent_dependency()
        seen = [json.loads(p.read_text())["dependency"] for p in launch.glob("watch-*.json")]
        if step < 5 and dependency not in seen:
            schedule(root, revision, dependency, step + 1)
        else:
            write_once(
                launch / "blocked.json",
                dict(
                    reason="upstream incomplete after last published dependency",
                    dependency=dependency,
                    no_repeated_polling=True,
                    GPU_submitted=False,
                ),
            )
        return
    final = driver.common.previous.read(closeout)
    if not final["complete"] or final["failures"]:
        write_once(
            launch / "blocked.json",
            dict(
                reason="upstream numerical/execution failure unresolved",
                GPU_submitted=False,
                parent_closeout_sha256=sha(closeout),
                independent_quality_N=0,
            ),
        )
        return
    if final["all_method_no_slower_gate"]:
        write_once(
            launch / "not_needed.json",
            dict(
                reason="parent engineering no-slower gate already passed",
                GPU_submitted=False,
                independent_quality_N=0,
                paper_superiority_established=False,
            ),
        )
        return
    parent = driver.common.previous.read(PARENT / "plan.json")
    first = driver.common.previous.read(PARENT / "analysis/decision.json")
    if not first["complete"] or first["failures"]:
        raise ValueError("complete phase attribution required")
    costs = opportunity(first["profiles"])
    dispatch_path = PARENT / "dispatch/analysis/decision.json"
    dispatch = driver.common.previous.read(dispatch_path) if dispatch_path.exists() else None
    use_dispatch = bool(
        dispatch
        and dispatch["complete"]
        and not dispatch["failures"]
        and dispatch["dispatch_end_to_end_improvement_gate"]
    )
    bound = dict(parent["bound_files"])
    for path in (PARENT / "plan.json", PARENT / "analysis/decision.json", closeout, DOC):
        bound[str(path)] = sha(path)
    if dispatch_path.exists():
        bound[str(dispatch_path)] = sha(dispatch_path)
    for i in (0, 1):
        folder = PARENT / f"workers/{i}"
        result = driver.common.previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("complete phase producer required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, value in result["outputs"].items():
            if sha(folder / name) != value:
                raise ValueError("phase producer changed")
            bound[str(folder / name)] = value
    write_once(
        root / "plan.json",
        dict(
            parent,
            source_sha=revision,
            bound_files=bound,
            paired_reader_stage="descriptor",
            reader_modes=["original_descriptors", "reused_descriptors"],
            selected_reader=first["selected_reader"],
            base_dispatch=use_dispatch,
            stage="consumed_private_replay_descriptor_reuse",
            study_root=str(root),
            measured_replay_host_seconds=costs,
            maximum_descriptor_attempts=1,
            maximum_worker_minutes=60,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    driver.common.previous.cpu(root, sha(root / "plan.json"))
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in row.lower() for row in queue.splitlines()):
        raise ValueError("wait for other owned GPU jobs before bounded descriptor precheck")
    # Use the same primitive precheck -> begin -> paired model worker chain.
    # No model inference is submitted directly from this watcher.
    (root / "slurm").mkdir(exist_ok=True)
    arguments = [str(SOURCE), str(root), sha(root / "plan.json")]
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
            "--mem=16G",
            "--time=00:10:00",
            "--job-name=mk-descriptor-precheck",
            f"--output={root}/slurm/precheck-%j.out",
            f"--error={root}/slurm/precheck-%j.err",
            str(driver.SCRIPT),
            *arguments,
            "precheck",
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
            f"--dependency=afterany:{gpu}",
            "--job-name=mk-descriptor-begin",
            f"--output={root}/slurm/begin-%j.out",
            f"--error={root}/slurm/begin-%j.err",
            str(driver.SCRIPT),
            *arguments,
            "begin",
        ],
        text=True,
    ).strip()
    write_once(root / "precheck_submission.json", dict(gpu_job=gpu, cpu_begin=analysis))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    watch(args.root, args.source_sha, args.step)
