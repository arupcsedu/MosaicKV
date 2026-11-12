#!/usr/bin/env python3
"""Finite CPU-only continuation of the 506 reviewed historical audit cursors.

Eight execution partitions retain the original scanner ownership (7 of 8).
This tool never samples data, launches a GPU, or promotes partial freshness.
Every round and checkpoint is immutable; an interrupted round is not retried
automatically.  The original global verification/deferred queues are retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

from mosaickv.evaluation import coco_history_resume as core
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
BASE = "8b8db8cb7ef91ad957dbab85948f02613e7dc5fa"
SEED = Path(
    "/scratch/djy8hg/workdir/fox-history-failed7-retry/run-20261111-1/subset/workers/shard-7.json"
)
SEED_SHA = "e2638e7a214d1e2c64c4ea946e4f2ee99dea85410d64c73298c5242e3f505448"
SEED_STATE_SHA = "e39d83833e163d0d3a081705ef3133ac87fa139c423324aa16ac1c00e6b712db"
PRIOR_GATE = SEED.parent.parent / "GATE_COCO_history_subset.json"
DEFERRED, INITIAL_PENDING, PARTS = 4938, 506, 8
MAX_ROUNDS, MAX_WALL = 8, 3600
POLICY = dict(
    original_owner_index=7,
    original_shard_count=8,
    execution_partitions=PARTS,
    scanner_seconds=210,
    scanner_source_bytes=2 * 2**30,
    outer_worker_seconds=420,
    maximum_rounds=MAX_ROUNDS,
    maximum_controller_seconds=MAX_WALL,
    deferred_frontier_items_preserved=DEFERRED,
    initial_reviewed_pending_items=INITIAL_PENDING,
    metadata_only=True,
    global_verification_performed=False,
    answers_or_predictions_exported=False,
    maximum_GPU_workers=0,
    historical_identity_media_coverage_complete=False,
    globally_fresh=False,
    fresh_confirmation_authorized=False,
    materialization_authorized=False,
    automatic_failed_worker_retry=False,
)


def dependencies():
    subset = importlib.import_module("evicastkv_coco_history_subset")
    scanner, empty, history = subset.modules()
    return subset, scanner, empty, history


def read(path, expected=None):
    return dependencies()[0].read(Path(path), expected)


def put(path, value):
    dependencies()[3].bounded_write(Path(path), value, 256 * 2**20, compact=True)


def additive_bridge():
    """Old numerical controls are not relabeled as executions of this reader."""
    raw = subprocess.check_output(
        ["git", "diff", "--name-status", f"{BASE}..HEAD"], cwd=SOURCE, text=True, timeout=15
    )
    names = []
    for line in raw.splitlines():
        status, name = line.split("\t", 1)
        if status != "A" or not name.startswith(
            ("mosaickv/scripts/", "mosaickv/slurm/", "mosaickv/src/", "mosaickv/tests/")
        ):
            raise ValueError("only additive audit tooling may bridge the existing controls")
        names.append(name)
    return dict(base_sha=BASE, additions=names, old_producer_files_unchanged=True)


def register(root, cpu_gate_root):
    revision = clean_source(SOURCE)
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("new immutable scratch audit root required")
    _subset, scanner, _empty, history = dependencies()
    followups = importlib.import_module("evicastkv_review_followups")
    cpu = followups.cpu_gate(Path(cpu_gate_root), revision)
    seed = read(SEED, SEED_SHA)
    if seed.get("state_sha256") != SEED_STATE_SHA or digest(seed["state"]) != SEED_STATE_SHA:
        raise ValueError("saved continuation state digest changed")
    split = core.partition(seed["state"], seed["state"]["frontier"][DEFERRED:], DEFERRED, PARTS)
    if sum(map(len, split["queues"])) != INITIAL_PENDING:
        raise ValueError("registered 506-item frontier differs")
    roots = [str(path.resolve(strict=True)) for path in scanner.ROOTS]
    bound = {str(SEED): SEED_SHA, str(PRIOR_GATE): sha(PRIOR_GATE)}
    # Preserve the full preceding registration, its exact roots, and original
    # successful worker 1.  No successful history worker is repeated.
    prior_path = SEED.parent.parent / "plan.json"
    prior = read(prior_path)
    bound[str(prior_path)] = sha(prior_path)
    if (
        prior["roots"] != roots
        or prior["source_sha"] != BASE
        or prior["original_owner_indices"] != [1, 7]
        or prior["original_shard_count"] != 8
        or prior["execute"] is not True
        or prior["fresh_confirmation_authorized"] is not False
        or prior["materialization_authorized"] is not False
        or prior["empirical_admission"].get("whole_request_engineering_gate_passed") is not True
        or prior["empirical_admission"].get("COCO_boundary_gate_passed") is not True
    ):
        raise ValueError("original root ownership or passed empirical admission differs")
    for name, expected in prior["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("preceding immutable provenance binding changed")
        bound[name] = expected
    for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
        path = Path(cpu_gate_root) / "full-cpu" / name
        bound[str(path)] = sha(path)
    worker1 = SEED.parent / "shard-1.json"
    bound[str(worker1)] = sha(worker1)
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_bridge=history.bridge(),
            additive_bridge=additive_bridge(),
            policy=POLICY,
            seed=str(SEED),
            seed_sha256=SEED_SHA,
            seed_state_sha256=SEED_STATE_SHA,
            original_scope_roots=split["all_scope_roots"],
            original_scope_sha256=digest(split["all_scope_roots"]),
            roots=roots,
            successful_worker1_preserved=str(worker1),
            bindings=bound,
            cpu_gate_root=str(Path(cpu_gate_root).resolve()),
            cpu_gate=cpu,
        ),
    )
    return dict(registered=True, pending_items=INITIAL_PENDING, CPU_partitions=PARTS, GPU_jobs=0)


def load(root):
    plan = read(Path(root) / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["policy"]) != digest(POLICY) or plan["additive_bridge"] != additive_bridge():
        raise ValueError("immutable audit policy/source bridge changed")
    if plan["source_bridge"] != dependencies()[3].bridge():
        raise ValueError("original scanner semantics changed")
    if digest(plan["original_scope_roots"]) != plan["original_scope_sha256"]:
        raise ValueError("original full scope changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable input/control binding changed")
    return plan


def worker(root, number, index):
    root = Path(root).resolve(strict=True)
    plan = load(root)
    path = root / f"round-{number:02d}"
    round_plan = read(path / "plan.json")
    if round_plan["source_sha"] != plan["source_sha"] or not 0 <= index < PARTS:
        raise ValueError("indexed immutable worker required")
    write_once(
        path / "locks" / f"worker-{index}.json",
        dict(
            source_sha=plan["source_sha"],
            job_id=os.getenv("SLURM_JOB_ID"),
            pid=os.getpid(),
            original_owner=7,
            execution_partition=index,
        ),
    )
    base = read(round_plan["base_path"], round_plan["base_file_sha256"])
    queue = round_plan["queues"][index]
    if round_plan["all_scope_roots_sha256"] != plan["original_scope_sha256"]:
        raise ValueError("full reviewed scope was tightened or expanded")
    _subset, scanner, empty, _history = dependencies()
    result = core.advance_partition(
        base,
        queue,
        plan["original_scope_roots"],
        tuple(Path(p) for p in plan["roots"]),
        scanner,
        empty.empty_projection_parquet,
    )
    put(
        path / "workers" / f"part-{index}.json",
        dict(
            source_sha=plan["source_sha"],
            original_owner=7,
            execution_partition=index,
            **result,
        ),
    )
    return result["summary"]


def run_process(root, number, index):
    path = Path(root) / f"round-{number:02d}"
    started = time.monotonic()
    with (
        (path / "logs" / f"worker-{index}.out").open("xb") as out,
        (path / "logs" / f"worker-{index}.err").open("xb") as err,
    ):
        try:
            result = subprocess.run(
                [
                    "/scratch/djy8hg/env/mosaickv/bin/python",
                    str(Path(__file__).resolve()),
                    "worker",
                    "--root",
                    str(root),
                    "--round",
                    str(number),
                    "--index",
                    str(index),
                ],
                cwd=SOURCE,
                stdout=out,
                stderr=err,
                timeout=420,
                check=False,
            )
            status = dict(returncode=result.returncode, timeout=False)
        except subprocess.TimeoutExpired:
            status = dict(returncode=None, timeout=True)
    record = dict(index=index, wall_seconds=time.monotonic() - started, **status)
    write_once(path / "jobs" / f"worker-{index}.json", record)
    return record


def controller(root):
    root = Path(root).resolve(strict=True)
    plan = load(root)
    write_once(
        root / "controller.json",
        dict(
            job_id=os.getenv("SLURM_JOB_ID"),
            host=os.uname().nodename,
            source_sha=plan["source_sha"],
            bounded_rounds=MAX_ROUNDS,
            no_polling=True,
        ),
    )
    started = time.monotonic()
    base = read(SEED, SEED_SHA)["state"]
    seed_processed_files = base["processed_files"]
    previous = base
    final = None
    last_attempted_round = None
    rounds_completed = 0
    for number in range(MAX_ROUNDS):
        if time.monotonic() - started > MAX_WALL - 480:
            break
        last_attempted_round = number
        split = core.partition(base, base["frontier"][DEFERRED:], DEFERRED, PARTS)
        path = root / f"round-{number:02d}"
        for name in ("logs", "jobs", "workers", "locks"):
            (path / name).mkdir(parents=True, exist_ok=False)
        put(path / "base.json", base)
        write_once(
            path / "plan.json",
            dict(
                source_sha=plan["source_sha"],
                base_path=str(path / "base.json"),
                base_file_sha256=sha(path / "base.json"),
                queues=split["queues"],
                all_scope_roots_sha256=plan["original_scope_sha256"],
                round=number,
            ),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARTS) as executor:
            statuses = list(
                executor.map(
                    lambda index, current=number: run_process(root, current, index), range(PARTS)
                )
            )
        if any(row["returncode"] != 0 for row in statuses):
            final = dict(
                complete=False, reason="failed_execution_partition", round=number, statuses=statuses
            )
            break
        results = [read(path / "workers" / f"part-{index}.json") for index in range(PARTS)]
        if any(
            row["source_sha"] != plan["source_sha"]
            or row["all_scope_roots_sha256"] != plan["original_scope_sha256"]
            or row["execution_partition"] != index
            or row["original_owner"] != 7
            for index, row in enumerate(results)
        ):
            raise ValueError("partition producer/whole-scope binding changed")
        merged = core.merge(base, split["queues"], results, split["deferred"])
        put(path / "merged-state.json", merged)
        summary = dict(
            round=number,
            remaining_pending_items=len(merged["frontier"]) - DEFERRED,
            newly_processed_files=merged["processed_files"] - base["processed_files"],
            new_errors=len(merged["errors"]) - len(base["errors"]),
            state_sha256=digest(merged),
            state_file_sha256=sha(path / "merged-state.json"),
            workers={str(i): sha(path / "workers" / f"part-{i}.json") for i in range(PARTS)},
        )
        write_once(path / "decision.json", summary)
        rounds_completed += 1
        print(json.dumps(summary, sort_keys=True), flush=True)
        previous, base = base, merged
        if summary["remaining_pending_items"] == 0:
            final = dict(complete=True, reason="reviewed_scope_traversal_exhausted", round=number)
            break
        if merged == previous:
            final = dict(complete=False, reason="no_checkpoint_progress", round=number)
            break
    if final is None:
        final = dict(
            complete=False, reason="registered_round_or_wall_bound", round=last_attempted_round
        )
    gate = dict(
        source_sha=plan["source_sha"],
        **final,
        initial_reviewed_items=INITIAL_PENDING,
        reviewed_pending_items=len(base["frontier"]) - DEFERRED,
        deferred_frontier_items_preserved=DEFERRED,
        unselected_global_pending_preserved=8481,
        reviewed_scope_frontier_exhausted=final["complete"],
        new_processed_files=base["processed_files"] - seed_processed_files,
        rounds_completed=rounds_completed,
        preserved_error_count=len(base["errors"]),
        scope_traversal_only=True,
        global_verification_performed=False,
        historical_identity_media_coverage_complete=False,
        globally_fresh=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        wall_seconds=time.monotonic() - started,
    )
    write_once(root / "GATE_history_resume_v35.json", gate)
    return gate


def closeout(root):
    """Afterany evidence recovery/admission analysis, never model execution."""
    root = Path(root).resolve(strict=True)
    plan = load(root)
    destination = root / "review"
    destination.mkdir(exist_ok=False)
    media = destination / "media-gap.json"
    admission = destination / "confirmation-admission.json"
    commands = [
        (
            "media-recovery",
            [
                "/scratch/djy8hg/env/mosaickv/bin/python",
                str(SOURCE / "mosaickv/scripts/evicastkv_media_gap_recovery_v35.py"),
                "--output",
                str(media),
            ],
            240,
        ),
        (
            "confirmation-admission",
            [
                "/scratch/djy8hg/env/mosaickv/bin/python",
                str(SOURCE / "mosaickv/scripts/evicastkv_coco_confirmation_admission_v35.py"),
                "--output",
                str(admission),
                "--history-report",
                str(root / "GATE_history_resume_v35.json"),
                "--media-report",
                str(media),
            ],
            60,
        ),
    ]
    statuses = []
    for name, command, timeout in commands:
        started = time.monotonic()
        with (destination / f"{name}.log").open("x") as output:
            try:
                result = subprocess.run(
                    command,
                    cwd=SOURCE,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
                status = dict(returncode=result.returncode, timeout=False)
            except subprocess.TimeoutExpired:
                status = dict(returncode=None, timeout=True)
        statuses.append(
            dict(
                name=name,
                command=command,
                wall_seconds=time.monotonic() - started,
                **status,
            )
        )
    report = dict(
        source_sha=plan["source_sha"],
        commands=statuses,
        complete=all(item["returncode"] == 0 for item in statuses),
        reports={str(path): sha(path) for path in (media, admission) if path.is_file()},
        no_GPU_work=True,
        no_dataset_or_media_materialization=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(destination / "closeout.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "controller", "worker", "closeout"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--cpu-gate-root")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "register":
        value = register(args.root, args.cpu_gate_root)
    elif args.mode == "worker":
        value = worker(args.root, args.round, args.index)
    elif args.mode == "closeout":
        value = closeout(args.root)
    else:
        value = controller(args.root)
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
