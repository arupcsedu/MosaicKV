#!/usr/bin/env python3
"""Finite afterany history continuation; no polling, restarts or freshness pass."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import evicastkv_freshness_empty_parquet_continue as frontier
import evicastkv_freshness_yaml_events as yaml_events
from evicastkv_review_followups import cpu_gate
from evicastkv_review_v8_diagnostics import identity

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1")
PARENT_SHA = "76d4ced1a10c6746bad73f564e84210522355ef9"
SCRATCH = Path("/scratch/djy8hg/workdir")
GATE = "GATE_freshness_empty_parquet_frontier.json"
POLICY = dict(
    maximum_parent_hops=4,
    maximum_new_segments=24,
    workers=8,
    worker_CPUs=1,
    worker_seconds=210,
    YAML_worker_seconds=240,
    worker_minutes=5,
    analyzer_CPUs=4,
    analyzer_minutes=15,
    no_queue_or_log_polling=True,
    no_state_restart=True,
    original_errors_preserved=True,
    fresh_confirmation_authorized=False,
    materialization_authorized=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def load(root):
    plan = read(Path(root) / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    if plan["execute"] is not True or plan["policy"] != POLICY:
        raise ValueError("only the unchanged bounded CPU continuation is executable")
    if any(sha(Path(path)) != value for path, value in plan["bindings"].items()):
        raise ValueError("immutable historical continuation registration changed")
    return plan


def parent_analyzer(parent):
    """Read at most four immutable parent job ledgers, never query the queue."""
    parent = Path(parent)
    candidates = []
    first = parent / "jobs/after-frontier-submission.json"
    if first.is_file():
        candidates.append((1, identity(read(first)["job"]), first))
    for index in range(2, 5):
        path = parent / f"frontier-segment-{index}-jobs.json"
        if path.is_file():
            candidates.append((index, identity(read(path)["analyzer"]), path))
    if not candidates:
        raise ValueError("no actual immutable parent frontier-analyzer job exists")
    return max(candidates, key=lambda row: row[0])


def submit(root, label, mode, *, dependencies=(), array=False, number=0, completion=None):
    root = Path(root)
    dependencies = list(dict.fromkeys(identity(value) for value in dependencies))
    if completion is not None:
        if (
            not completion
            or completion[0] != "sbatch"
            or any(
                value.startswith(("--dependency", "--gres", "--gpus"))
                or value == "--partition=bii-gpu"
                for value in completion
            )
        ):
            raise ValueError(
                "explicit CPU-only sbatch completion without duplicate dependencies required"
            )
        command = [
            completion[0],
            *(["--dependency=afterany:" + ":".join(dependencies)] if dependencies else []),
            *completion[1:],
        ]
    else:
        command = [
            "sbatch",
            "--parsable",
            "--account=bii_dsc_community",
            "--partition=bii",
            "--nodes=1",
            "--ntasks=1",
            f"--job-name=fox13-history-{label}",
            f"--cpus-per-task={1 if array else 4}",
            "--mem=16G",
            f"--time=00:{5 if array else 15:02d}:00",
            f"--output={root}/slurm/{label}-%A_%a.out",
            f"--error={root}/slurm/{label}-%A_%a.err",
        ]
        if array:
            command.append("--array=0-7%8")
        if dependencies:
            command.append("--dependency=afterany:" + ":".join(dependencies))
        command.extend(
            [
                str(SOURCE / "mosaickv/slurm/evicastkv_history_v13.sbatch"),
                str(SOURCE),
                str(root),
                mode,
                str(number),
            ]
        )
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(root / f"jobs/{label}-request.json", dict(command=command, single_attempt=True))
    try:
        job = identity(
            subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=30)
        )
    except Exception as error:
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(error=f"{type(error).__name__}: {error}", automatic_retry=False),
        )
        raise
    write_once(root / f"jobs/{label}-submission.json", dict(job=job, command=command))
    return job


def register(
    root,
    *,
    cpu_gate_root,
    parent_campaign=PARENT,
    execute=True,
    completion_argv=None,
    completion_dependencies=(),
):
    root, parent = Path(root).resolve(), Path(parent_campaign).resolve()
    if root.exists() or root == SCRATCH.resolve() or not root.is_relative_to(SCRATCH.resolve()):
        raise ValueError("new write-once scratch history output root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    parent_plan = read(parent / "registration.json")
    if parent_plan["source_sha"] != PARENT_SHA:
        raise ValueError("only the immutable V12 parent campaign is admissible")
    clean_source(Path(parent_plan["source_root"]), PARENT_SHA)
    index, job, path = parent_analyzer(parent)
    bindings = {
        str(parent / "registration.json"): sha(parent / "registration.json"),
        str(path): sha(path),
        str(Path(frontier.__file__)): sha(Path(frontier.__file__)),
        str(Path(yaml_events.__file__)): sha(Path(yaml_events.__file__)),
        str(Path(frontier.scanner.__file__)): sha(Path(frontier.scanner.__file__)),
        str(Path(__file__)): sha(Path(__file__)),
    }
    if completion_argv is not None and (
        not all(isinstance(v, str) for v in completion_argv)
        or not completion_argv
        or completion_argv[0] != "sbatch"
        or any(
            value.startswith(("--dependency", "--gres", "--gpus", "-G", "-d"))
            or value == "--partition=bii-gpu"
            for value in completion_argv
        )
    ):
        raise ValueError("explicit CPU-only sbatch completion without dependencies required")
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            cpu_gate_root=str(cpu_gate_root),
            cpu_gate=cpu,
            parent_campaign=str(parent),
            bindings=bindings,
            policy=POLICY,
            initial_parent_index=index,
            initial_parent_analyzer=job,
            initial_parent_job_ledger=str(path),
            initial_parent_job_ledger_sha256=sha(path),
            completion_argv=completion_argv,
            completion_dependencies=[identity(d) for d in completion_dependencies],
        ),
    )
    if not execute:
        return dict(status="dry_run", jobs=0, parent_dependency=job)
    dependencies = current_dependency() if (parent / "frontier-selection.json").is_file() else [job]
    callback = submit(root, "attach-0", "attach", dependencies=dependencies, number=0)
    return dict(
        status="bounded_history_registered",
        callback=callback,
        terminal_gate=str(root / "GATE_history_v13.json"),
    )


def current_dependency():
    return [identity(os.environ["SLURM_JOB_ID"])] if os.environ.get("SLURM_JOB_ID") else []


def finish(root, *, reason, last=None, yaml_gate=None, error=None):
    root = Path(root)
    plan = load(root)
    previous = root / "GATE_history_v13.json"
    if previous.exists():
        return read(previous)
    last_gate = (
        read(Path(last) / GATE) if last is not None and (Path(last) / GATE).is_file() else None
    )
    paths = [
        Path(last) / name
        for name in ("plan.json", GATE, "combined_identity_inventory.json")
        if last is not None and (Path(last) / name).is_file()
    ]
    if yaml_gate is not None:
        paths.extend(
            root / "yaml" / name
            for name in (
                "plan.json",
                "GATE_freshness_yaml_events.json",
                "combined_identity_inventory.json",
            )
        )
    gate = dict(
        status="finite_history_continuation_finished_not_freshness_certified"
        if error is None
        else "blocked_history_continuation",
        source_sha=plan["source_sha"],
        stop_reason=reason,
        last_frontier=str(last) if last is not None else None,
        frontier_exhausted=bool(last_gate and last_gate["frontier_exhausted"]),
        missing_shards=last_gate["missing_shards"] if last_gate else list(range(8)),
        original_and_new_coverage_errors=last_gate["original_and_new_coverage_errors"]
        if last_gate
        else None,
        YAML_recovery=yaml_gate,
        error=error,
        bindings={str(p): sha(p) for p in paths},
        original_errors_preserved=True,
        error_waivers=0,
        new_GPU_inference=0,
        complete_historical_inventory_certified=False,
        cross_dataset_media_certified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(previous, gate)
    if plan["completion_argv"] is not None:
        submit(
            root,
            "completion",
            "unused",
            dependencies=[*plan["completion_dependencies"], *current_dependency()],
            completion=plan["completion_argv"],
        )
    return gate


def launch_segment(root, parent, index):
    plan = load(root)
    if not 1 <= index <= POLICY["maximum_new_segments"]:
        raise ValueError("bounded continuation segment index required")
    out = Path(root) / f"segment-{index}"
    frontier.prepare(
        out,
        cpu_gate_root=Path(plan["cpu_gate_root"]),
        parent=Path(parent),
        parent_gate_filename=GATE,
    )
    worker = submit(
        root,
        f"segment-{index}",
        "worker",
        dependencies=current_dependency(),
        array=True,
        number=index,
    )
    analyzer = submit(
        root, f"after-segment-{index}", "advance", dependencies=[worker], number=index
    )
    return dict(segment=index, worker=worker, analyzer=analyzer)


def attach(root, hop):
    root = Path(root)
    plan = load(root)
    if type(hop) is not int or not 0 <= hop < POLICY["maximum_parent_hops"]:
        raise ValueError("bounded parent reattachment index required")
    parent = Path(plan["parent_campaign"])
    selection = parent / "frontier-selection.json"
    if selection.exists():
        selected = read(selection)
        expected = parent / (
            "frontier" if selected["index"] == 1 else f"frontier-{selected['index']}"
        )
        if (
            not 1 <= selected["index"] <= 4
            or Path(selected["path"]).resolve() != expected.resolve()
            or sha(expected / GATE) != selected["gate_sha256"]
        ):
            raise ValueError("immutable parent selected gate differs")
        gate = read(expected / GATE)
        if (
            gate["source_sha"] != PARENT_SHA
            or read(expected / "plan.json")["source_sha"] != PARENT_SHA
        ):
            raise ValueError("selected parent state producer differs from immutable V12")
        write_once(
            root / "parent-selection.json",
            dict(
                path=str(expected),
                selection_sha256=sha(selection),
                gate_sha256=sha(expected / GATE),
                source_sha=gate["source_sha"],
            ),
        )
        if gate["missing_shards"]:
            return finish(
                root, reason="parent_missing_workers", last=expected, error="missing parent states"
            )
        if gate["frontier_exhausted"]:
            return launch_yaml(root, expected, "parent_frontier_exhausted")
        return launch_segment(root, expected, 1)
    index, job, path = parent_analyzer(parent)
    last_index = (
        plan["initial_parent_index"]
        if hop == 0
        else read(root / f"parent-attachments/{hop}.json")["index"]
    )
    if index <= last_index or hop + 1 >= POLICY["maximum_parent_hops"]:
        return finish(
            root,
            reason="parent_selection_unavailable",
            error="no next registered parent analyzer/selection",
        )
    write_once(
        root / f"parent-attachments/{hop + 1}.json",
        dict(index=index, job=job, path=str(path), sha256=sha(path)),
    )
    return dict(
        reattached_to=job,
        callback=submit(root, f"attach-{hop + 1}", "attach", dependencies=[job], number=hop + 1),
    )


def progress(root):
    changed = False
    for index in range(8):
        old = frontier.read(Path(root) / f"imports/shard-{index}.json")["state"]
        new = frontier.read(Path(root) / f"workers/shard-{index}.json")["state"]
        if new["errors"][: len(old["errors"])] != old["errors"]:
            raise ValueError("original historical errors were removed or reordered")
        changed |= digest({k: v for k, v in old.items() if k != "errors"}) != digest(
            {k: v for k, v in new.items() if k != "errors"}
        )
    return changed


def launch_yaml(root, parent, reason):
    plan = load(root)
    yaml_events.prepare(
        Path(root) / "yaml",
        cpu_gate_root=Path(plan["cpu_gate_root"]),
        parent=Path(parent),
        parent_gate_filename=GATE,
    )
    write_once(Path(root) / "yaml-stop.json", dict(parent=str(parent), reason=reason))
    worker = submit(root, "yaml", "yaml-worker", dependencies=current_dependency(), array=True)
    return dict(
        yaml_worker=worker,
        analyzer=submit(root, "after-yaml", "yaml-analyze", dependencies=[worker]),
    )


def advance(root, index):
    load(root)
    if type(index) is not int or not 1 <= index <= POLICY["maximum_new_segments"]:
        raise ValueError("bounded current segment required")
    path = Path(root) / f"segment-{index}"
    gate = frontier.analyze(path)
    changed = not gate["missing_shards"] and progress(path)
    write_once(
        Path(root) / f"decisions/segment-{index}.json",
        dict(
            gate_sha256=sha(path / GATE),
            state_progress=changed,
            missing_shards=gate["missing_shards"],
            frontier_exhausted=gate["frontier_exhausted"],
        ),
    )
    if gate["missing_shards"]:
        return finish(
            root, reason="missing_worker_states", last=path, error="all eight workers required"
        )
    if not gate["frontier_exhausted"] and changed and index < POLICY["maximum_new_segments"]:
        return launch_segment(root, path, index + 1)
    reason = (
        "frontier_exhausted"
        if gate["frontier_exhausted"]
        else "no_state_progress"
        if not changed
        else "maximum_segments"
    )
    return launch_yaml(root, path, reason)


def run(root, mode, number=0):
    root = Path(root)
    load(root)
    write_once(
        root / f"locks/{mode}-{number}-{os.getenv('SLURM_ARRAY_TASK_ID', 'callback')}.json",
        dict(job_id=os.getenv("SLURM_JOB_ID")),
    )
    if mode == "worker":
        return frontier.worker(root / f"segment-{number}", int(os.environ["SLURM_ARRAY_TASK_ID"]))
    if mode == "yaml-worker":
        return yaml_events.worker(root / "yaml", int(os.environ["SLURM_ARRAY_TASK_ID"]))
    try:
        if mode == "attach":
            return attach(root, number)
        if mode == "advance":
            return advance(root, number)
        if mode == "yaml-analyze":
            gate = yaml_events.analyze(root / "yaml")
            stopped = read(root / "yaml-stop.json")
            return finish(
                root, reason=stopped["reason"], last=Path(stopped["parent"]), yaml_gate=gate
            )
        raise ValueError("unknown bounded history callback")
    except Exception as error:
        last = root / f"segment-{number}" if mode == "advance" else None
        return finish(
            root, reason=f"{mode}_failed", last=last, error=f"{type(error).__name__}: {error}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("attach", "worker", "advance", "yaml-worker", "yaml-analyze")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--number", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.mode, args.number)))


if __name__ == "__main__":
    main()
