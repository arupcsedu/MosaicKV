#!/usr/bin/env python3
"""One finite CPU-only continuation round using the unchanged 64be064 scanner.

Existing checkpoint chains are appended, never rewritten. Every summary,
subprocess transcript and derived inventory lives under a new round directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD_SOURCE = Path("/scratch/djy8hg/worktrees/evicastkv-fox-scan-linear-20261110").resolve()
OLD_SHA = "64be064d3e6622890f6fe96a556cedc0161f8f4c"
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
DEFAULT_PARENT = Path("/scratch/djy8hg/audits/fox-review-v7-freshness-linear-20261110")
AUDITS_ROOT = Path("/scratch/djy8hg/audits").resolve()
SHARDS, INVOCATIONS, SECONDS, BYTE_LIMIT = 8, 3, 180, 2 * 2**30
# Directly executing this new script would put its directory ahead of the old
# PYTHONPATH and shadow the pinned scanner. run_path on a FILE preserves the
# original interpreter search path; it does not inject the driver's directory.
CHILD_BOOTSTRAP = (
    "import runpy,sys;driver=sys.argv.pop(1);sys.argv[0]=driver;"
    "runpy.run_path(driver,run_name='__main__')"
)


def read(path):
    return json.loads(Path(path).read_text())


def latest(path):
    files = sorted(Path(path).glob("checkpoint-*.json"))
    if not files:
        raise ValueError("existing hash-bound checkpoint required")
    return files[-1]


def load(root):
    record = read(root / "registration.json")
    clean_source(SOURCE, record["source_sha"])
    clean_source(OLD_SOURCE, OLD_SHA)
    if record["execute"] is not True:
        raise ValueError("dry registration cannot execute a continuation")
    cpu_gate(Path(record["cpu_gate_root"]), record["source_sha"])
    for name, expected in record["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"immutable continuation binding changed: {name}")
    return record


def register(root, parent=DEFAULT_PARENT, *, execute=False, cpu_gate_root=None):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists() or not root.is_relative_to(AUDITS_ROOT):
        raise ValueError("new write-once scratch audits round required")
    revision = clean_source(SOURCE)
    if execute and cpu_gate_root is None:
        raise ValueError("execution requires a verified current-source CPU gate root")
    passed_cpu = cpu_gate(Path(cpu_gate_root), revision) if execute else None
    clean_source(OLD_SOURCE, OLD_SHA)
    old_gate = parent / "freshness/GATE_freshness_workers.json"
    if read(old_gate).get("status") != "blocked_missing_or_incomplete_shards":
        raise ValueError("the preserved negative/incomplete parent gate is required")
    bound = [
        Path(__file__),
        OLD_SOURCE / "mosaickv/scripts/evicastkv_review_freshness_stream.py",
        parent / "registration.json",
        old_gate,
    ]
    if execute:
        bound += [
            Path(cpu_gate_root) / "full-cpu" / name
            for name in ("cpu-tests.json", "cpu-tests.log", "cpu-tests.xml")
        ]
    shards = []
    for index in range(SHARDS):
        path = parent / "freshness/shards" / f"shard-{index:03d}"
        checkpoint = latest(path)
        registration = read(path / "registration.json")
        state = read(checkpoint)["state"]
        if registration["source_sha"] != OLD_SHA or registration["shard_index"] != index:
            raise ValueError("old scanner/source ownership differs")
        if registration["shard_count"] != SHARDS or state["traversal_complete"]:
            raise ValueError("exactly eight unfinished parent shards required")
        shards.append(
            dict(
                index=index,
                path=str(path),
                checkpoint=str(checkpoint),
                baseline_files=state["processed_files"],
                baseline_records=state["projected_records"],
            )
        )
        bound += [checkpoint, path / "registration.json", path / "reviewed_provenance.json"]
    record = dict(
        source_sha=revision,
        old_source=str(OLD_SOURCE),
        old_source_sha=OLD_SHA,
        parent=str(parent),
        shards=shards,
        bindings={str(path): sha(path) for path in bound},
        execute=execute,
        cpu_gate_root=str(Path(cpu_gate_root).resolve()) if cpu_gate_root else None,
        cpu_gate=passed_cpu,
        maximum_GPU_workers=0,
        shard_count=SHARDS,
        concurrency=4,
        maximum_invocations=INVOCATIONS,
        seconds_per_invocation=SECONDS,
        maximum_bytes_per_invocation=BYTE_LIMIT,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "registration.json", record)
    if not execute:
        return dict(status="DRY_RUN_ONLY", jobs_submitted=0)
    (root / "slurm").mkdir()

    def submit(mode, depend=None, array=False):
        command = (
            "export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv && "
            f"source {shlex.quote(str(SOURCE / 'mosaickv/scripts/cache_env.sh'))} && "
            f"export PYTHONPATH={shlex.quote(str(SOURCE / 'mosaickv/src'))}:"
            f"{shlex.quote(str(SOURCE / 'mosaickv/scripts'))} "
            "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' && "
            f"exec {PYTHON} {shlex.quote(str(Path(__file__)))} {mode} "
            f"--root {shlex.quote(str(root))}"
        )
        if array:
            command += ' --index "$SLURM_ARRAY_TASK_ID"'
        arguments = [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=4",
            "--mem=16G",
            "--time=00:15:00",
            f"--job-name=fox8scan-{mode}",
            f"--output={root}/slurm/{mode}-%A_%a.out",
            f"--error={root}/slurm/{mode}-%A_%a.err",
        ]
        if depend:
            arguments.append(f"--dependency={depend}")
        if array:
            arguments.append("--array=0-7%4")
        argv = [*arguments, "--wrap", shlex.join(["bash", "-c", command])]
        write_once(
            root / "jobs" / f"{mode}-request.json", dict(arguments=argv, retry_authorized=False)
        )
        job, stdout, stderr = None, "", ""
        try:
            result = subprocess.run(argv, check=True, capture_output=True, text=True)
            stdout, stderr = result.stdout, getattr(result, "stderr", "")
            candidate = stdout.strip().split(";")[0]
            if not candidate.isdigit():
                raise ValueError("scheduler did not return a numeric job ID")
            job = candidate
            receipt = dict(
                job_id=job, arguments=argv, command=command, stdout=stdout, stderr=stderr
            )
            write_once(root / "jobs" / f"{mode}.json", receipt)
            return dict(job_id=job, uncertain=False, error=None)
        except Exception as exc:
            stdout, stderr = getattr(exc, "stdout", stdout), getattr(exc, "stderr", stderr)
            stdout = (
                stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout or ""
            )
            stderr = (
                stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr or ""
            )
            # An explicit rejection is different from a lost/ambiguous RPC
            # reply. Neither case authorizes retrying this immutable request.
            rejected = isinstance(exc, subprocess.CalledProcessError) and (
                "Batch job submission failed:" in stderr and not stdout.strip()
            )
            failure = dict(
                type=type(exc).__name__,
                message=str(exc),
                stdout=stdout,
                stderr=stderr,
                retry_authorized=False,
                scheduler_outcome_uncertain=not rejected,
            )
            write_once(root / "jobs" / f"{mode}-error.json", failure)
            return dict(job_id=job, uncertain=not rejected, error=failure)

    array = submit("worker", array=True)
    analysis = submit("aggregate", f"afterany:{array['job_id']}") if not array["error"] else None
    result = dict(
        status="blocked_submission"
        if array["error"] or analysis is None or analysis["error"]
        else "submitted_bounded_CPU_continuation",
        workers=array["job_id"],
        analysis=analysis["job_id"] if analysis else None,
        workers_submission_uncertain=array["uncertain"],
        analysis_definitely_not_submitted=analysis is None
        or bool(analysis["error"] and not analysis["uncertain"] and not analysis["job_id"]),
        analysis_submission_uncertain=analysis["uncertain"] if analysis else False,
        submission_error=array["error"] or (analysis["error"] if analysis else None),
    )
    write_once(root / "jobs/submission-result.json", result)
    return result


def child(root, mode, index=None):
    """Executed with old PYTHONPATH/cwd: scanner's own source checks stay valid."""
    record = load(root)
    import evicastkv_review_freshness_stream as stream

    expected = OLD_SOURCE / "mosaickv/scripts/evicastkv_review_freshness_stream.py"
    if Path(stream.__file__).resolve() != expected:
        raise ValueError("child imported a different scanner implementation")
    if mode == "step":
        path = Path(record["shards"][index]["path"])
        return stream.run(
            path,
            shard_index=index,
            shard_count=SHARDS,
            seconds=SECONDS,
            maximum_bytes=BYTE_LIMIT,
            resume=True,
        )
    return stream.aggregate(root / "aggregate", [Path(row["path"]) for row in record["shards"]])


def invoke(root, mode, *, index=None, attempt=0):
    env = dict(
        os.environ,
        PYTHONPATH=f"{OLD_SOURCE}/mosaickv/src:{OLD_SOURCE}/mosaickv/scripts",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    arguments = [
        PYTHON,
        "-c",
        CHILD_BOOTSTRAP,
        str(Path(__file__)),
        f"child-{mode}",
        "--root",
        str(root),
    ]
    if index is not None:
        arguments += ["--index", str(index)]
    try:
        result = subprocess.run(
            arguments, cwd=OLD_SOURCE, env=env, capture_output=True, text=True, timeout=250
        )
        out, err, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        out, err, code = exc.stdout or "", exc.stderr or "", 124
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        err = err.decode("utf-8", "replace") if isinstance(err, bytes) else err
    prefix = root / "transcripts" / f"{mode}-{index}-{attempt}"
    stdout, stderr = Path(str(prefix) + ".stdout.json"), Path(str(prefix) + ".stderr.json")
    write_once(stdout, out)
    write_once(stderr, err)
    if code:
        raise RuntimeError(f"scanner subprocess failed ({code}); see {stderr}")
    gate = json.loads(out)
    return dict(gate=gate, stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))


def worker(root, index):
    root = Path(root).resolve()
    record = load(root)
    if not 0 <= index < SHARDS:
        raise ValueError("registered shard index required")
    row = record["shards"][index]
    if str(latest(row["path"])) != row["checkpoint"]:
        raise ValueError(
            "parent checkpoint already advanced; refusing duplicate/racing continuation"
        )
    write_once(
        root / "locks" / f"worker-{index:03d}.json",
        dict(index=index, job_id=os.environ.get("SLURM_JOB_ID")),
    )
    progress, failure, files, records = [], None, row["baseline_files"], row["baseline_records"]
    try:
        for attempt in range(INVOCATIONS):
            result = invoke(root, "step", index=index, attempt=attempt)
            gate = result["gate"]
            if gate["shard_index"] != index or gate["source_sha"] != OLD_SHA:
                raise ValueError("scanner gate source/ownership differs")
            result.update(
                files_added=gate["processed_files"] - files,
                records_added=gate["projected_records"] - records,
                job_id=os.environ.get("SLURM_JOB_ID", "not_slurm"),
            )
            files, records = gate["processed_files"], gate["projected_records"]
            write_once(root / "progress" / f"worker-{index:03d}-step-{attempt}.json", result)
            progress.append(result)
            if gate["traversal_complete"]:
                break
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), retryable=False)
    summary = dict(
        index=index,
        source_sha=record["source_sha"],
        scanner_source_sha=OLD_SHA,
        job_id=os.environ.get("SLURM_JOB_ID", "not_slurm"),
        progress=progress,
        failure=failure,
        files_added=files - row["baseline_files"],
        records_added=records - row["baseline_records"],
        fresh_confirmation_authorized=False,
    )
    write_once(root / "workers" / f"worker-{index:03d}.json", summary)
    return summary


def aggregate(root):
    root = Path(root).resolve()
    load(root)
    summaries = [root / "workers" / f"worker-{index:03d}.json" for index in range(SHARDS)]
    missing = [str(path) for path in summaries if not path.is_file()]
    try:
        # Partial traversal still yields a conservative, explicitly incomplete
        # inventory. Original scanner verifies every checkpoint/hash chain.
        merged, failure = invoke(root, "merge"), None
    except Exception as exc:
        merged, failure = None, dict(type=type(exc).__name__, message=str(exc))
    report = dict(
        status="derived_inventory_confirmation_blocked",
        missing_workers=missing,
        worker_bindings={str(path): sha(path) for path in summaries if path.is_file()},
        merged=merged,
        failure=failure,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        maximum_GPU_workers=0,
    )
    write_once(root / "FINAL_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("register", "worker", "aggregate", "child-step", "child-merge")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode in {"worker", "child-step"} and args.index is None:
        parser.error("worker/child-step requires --index")
    if args.mode == "register":
        result = register(
            args.root, args.parent, execute=args.execute, cpu_gate_root=args.cpu_gate_root
        )
    elif args.mode == "worker":
        result = worker(args.root, args.index)
    elif args.mode.startswith("child-"):
        result = child(args.root, args.mode.removeprefix("child-"), args.index)
    else:
        result = aggregate(args.root)
    print(json.dumps(result))
