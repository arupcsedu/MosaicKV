#!/usr/bin/env python3
"""Bounded CPU report rescue and event-time Slurm dependency resolution.

Slurm can purge a finished controller ID while accounting retains its terminal
record. Never attach a new dependency to such an ID. This module neither polls
nor restarts inference/scanning: each callback performs one accounting query,
then waits for an existing acknowledged callback or reports a bounded blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v13/run-20261111-1")
PARENT_SOURCE = Path("/scratch/djy8hg/workdir/fox-review-v13-code")
PARENT_SHA = "cc2768b2ba93e13682bc0d58e0e49a7856efd4c1"
GATE = "GATE_completion_v14.json"
MAX_HOPS = 32
JOB = re.compile(r"[1-9][0-9]*(?:_[0-9]+)?\Z")
TERMINAL = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "OOM",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
ACTIVE = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "RESIZING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESV_DEL_HOLD",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
    }
)
STATIC = {
    "lookm": "GATE_lookm_hf_checkpoint_compat.json",
    "streaming": "GATE_streaming_lossless_source.json",
    "ownership": "GATE_source_ownership_v12.json",
    "providers": "GATE_historical_provider_superset.json",
}


class DependencyResolutionError(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def read(path):
    path = Path(path)
    if path.stat().st_size > 32 * 2**20:
        raise ValueError("Bounded report/job metadata exceeds 32 MiB")
    return json.loads(path.read_text())


def identity(value):
    if not isinstance(value, str) or not JOB.fullmatch(value):
        raise ValueError("One concrete Slurm allocation/array-task ID required")
    return value


def resolve_dependencies(ids):
    """One event-time accounting query; return active IDs and a sealable receipt."""
    requested = sorted(set(identity(value) for value in ids))
    if len(requested) > 256:
        raise ValueError("At most 256 known own dependencies per bounded query")
    receipt = dict(requested=requested, queries=0, retained=[], terminal_omitted=[], records=[])
    if not requested:
        return [], dict(receipt, passed=True)
    command = [
        "sacct",
        "--array",
        "--allocations",
        "--duplicates",
        "--noheader",
        "--parsable2",
        "--starttime=1970-01-01T00:00:00",
        "--jobs=" + ",".join(requested),
        "--format=JobID%128,JobIDRaw%128,State%64,UID",
    ]
    receipt.update(queries=1, command=command, expected_UID=os.getuid())
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=30)
        receipt["stdout_sha256"] = hashlib.sha256(output.encode()).hexdigest()
        groups, seen = {value: [] for value in requested}, set()
        for line in output.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split("|")]
            if len(fields) != 4:
                raise ValueError("Ambiguous accounting field layout")
            display, raw, state, uid = fields
            identity(display)
            identity(raw)
            if not uid.isdecimal() or int(uid) != os.getuid():
                raise ValueError("Accounting allocation is not owned by the executing UID")
            key = display, raw
            if key in seen:
                raise ValueError("Duplicate accounting allocation/job-ID incarnation")
            seen.add(key)
            normalized = state.split()[0] if state else ""
            if normalized not in TERMINAL | ACTIVE:
                raise ValueError("Unknown/truncated accounting state")
            row = dict(
                JobID=display, JobIDRaw=raw, state=state, normalized_state=normalized, UID=int(uid)
            )
            targets = [
                value
                for value in requested
                if value in (display, raw) or ("_" not in value and display.startswith(value + "_"))
            ]
            if not targets:
                raise ValueError("Unexpected allocation outside exact requested dependencies")
            receipt["records"].append(row)
            for value in targets:
                groups[value].append(row)
        for value, rows in groups.items():
            if not rows:
                raise ValueError("Requested dependency is missing from accounting")
            (
                receipt["retained"]
                if any(row["normalized_state"] in ACTIVE for row in rows)
                else receipt["terminal_omitted"]
            ).append(value)
        receipt.update(passed=True, expanded_array_allocations=True, terminal_is_not_success=True)
        return receipt["retained"], receipt
    except Exception as error:
        receipt.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise DependencyResolutionError(
            "Fail-closed Slurm dependency resolution", receipt
        ) from error


def job_ledger(path, *, source, campaign, history=False, allow_existing_gpu=False):
    value = read(path)
    job, command = identity(value["job"]), value["command"]
    expected_script = (
        source / f"mosaickv/slurm/evicastkv_{'history_v13' if history else 'review_v13'}.sbatch"
    )
    expected_root = campaign / "history" if history else campaign
    if (
        not isinstance(command, list)
        or any(not isinstance(part, str) for part in command)
        or not command
        or command[0] != "sbatch"
        or (
            not allow_existing_gpu
            and any(
                part.startswith(("--gres", "--gpus", "-G")) or part == "--partition=bii-gpu"
                for part in command
            )
        )
        or str(expected_script) not in command
    ):
        raise ValueError("Only an actual immutable CPU callback job ledger is admissible")
    position = command.index(str(expected_script))
    if command[position + 1 : position + 3] != [str(source), str(expected_root)]:
        raise ValueError("Acknowledged callback targets another source or output root")
    return job, dict(path=str(path), sha256=sha(path), job=job)


def static_errors(campaign, lane):
    """Exact immutable failure records can complete reporting, never a gate."""
    result = []
    for name in (f"prepare-{lane}", f"analysis-{lane}"):
        path = campaign / f"errors/{name}.json"
        if path.is_file():
            value = read(path)
            if (
                value.get("status") != "blocked"
                or not isinstance(value.get("error"), str)
                or not value["error"]
            ):
                raise ValueError("Recorded static-lane failure has invalid provenance shape")
            result.append(dict(path=str(path), sha256=sha(path), evidence=value))
    return result


def frontier(campaign, source):
    """Latest acknowledged callback, not a queue/log search or new scan request."""
    folder = campaign / "history/jobs"
    options = []
    for hop in range(4):
        path = folder / f"attach-{hop}-submission.json"
        if path.is_file():
            options.append((hop, path))
    for segment in range(1, 25):
        path = folder / f"after-segment-{segment}-submission.json"
        if path.is_file():
            options.append((4 + segment, path))
    last = folder / "after-yaml-submission.json"
    if last.is_file():
        options.append((29, last))
    if not options:
        raise ValueError("No acknowledged actual history callback exists")
    rank, path = max(options, key=lambda row: row[0])
    job, evidence = job_ledger(path, source=source, campaign=campaign, history=True)
    return dict(rank=rank, **evidence), job


def parent_evidence(campaign, source):
    registration, history = (
        read(campaign / "registration.json"),
        read(campaign / "history/registration.json"),
    )
    if (
        registration["source_sha"] != PARENT_SHA
        or history["source_sha"] != PARENT_SHA
        or Path(registration["source_root"]).resolve() != source
    ):
        raise ValueError("Exact immutable V13 campaign/source required")
    clean_source(source, PARENT_SHA)
    return {
        str(campaign / name): sha(campaign / name)
        for name in ("registration.json", "history/registration.json")
    }


def register(
    root, *, parent_campaign=PARENT, source=PARENT_SOURCE, execute=True, cpu_gate_root=None
):
    from evicastkv_review_followups import cpu_gate

    root, campaign, source = (
        Path(root).resolve(),
        Path(parent_campaign).resolve(),
        Path(source).resolve(),
    )
    scratch = Path("/scratch/djy8hg/workdir").resolve()
    if root.exists() or root == scratch or not root.is_relative_to(scratch):
        raise ValueError("New write-once scratch completion root required")
    revision = clean_source(SOURCE)
    if cpu_gate_root is None:
        raise ValueError("Complete current immutable CPU suite required")
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    bindings = parent_evidence(campaign, source)
    bindings[str(Path(__file__))] = sha(Path(__file__))
    launcher = SOURCE / "mosaickv/slurm/evicastkv_completion_v14.sbatch"
    bindings[str(launcher)] = sha(launcher)
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            parent_campaign=str(campaign),
            parent_source=str(source),
            parent_source_sha=PARENT_SHA,
            cpu_gate_root=str(cpu_gate_root),
            cpu_gate=cpu,
            bindings=bindings,
            maximum_event_hops=MAX_HOPS,
            no_new_GPU_or_scan=True,
            no_queue_or_log_polling=True,
            automatic_retry=False,
        ),
    )
    return advance(root, 0) if execute else dict(status="disabled", jobs=0)


def load(root):
    plan = read(Path(root) / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if (
        plan["execute"] is not True
        or plan["maximum_event_hops"] != MAX_HOPS
        or plan["no_new_GPU_or_scan"] is not True
    ):
        raise ValueError("Only the bounded report-rescue scope is executable")
    if any(sha(Path(path)) != expected for path, expected in plan["bindings"].items()):
        raise ValueError("Immutable report-rescue source/parent binding changed")
    parent_evidence(Path(plan["parent_campaign"]), Path(plan["parent_source"]))
    return plan


def submit(root, hop, dependencies):
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=00:15:00",
        "--job-name=fox14-report-rescue",
        f"--output={root}/slurm/rescue-{hop}-%j.out",
        f"--error={root}/slurm/rescue-{hop}-%j.err",
    ]
    if dependencies:
        command.append("--dependency=afterany:" + ":".join(dependencies))
    command.extend(
        [
            str(SOURCE / "mosaickv/slurm/evicastkv_completion_v14.sbatch"),
            str(SOURCE),
            str(root),
            str(hop),
        ]
    )
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(root / f"jobs/hop-{hop}-request.json", dict(command=command, single_attempt=True))
    try:
        job = identity(
            subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=30)
            .strip()
            .split(";", 1)[0]
        )
    except Exception as error:
        write_once(
            root / f"jobs/hop-{hop}-submission-error.json",
            dict(error=f"{type(error).__name__}: {error}", automatic_retry=False),
        )
        raise
    write_once(root / f"jobs/hop-{hop}-submission.json", dict(job=job, command=command))
    return job


def verify_report(campaign):
    report = read(campaign / "FINAL_report.json")
    if report["source_sha"] != PARENT_SHA or any(
        report.get(key) is not False
        for key in (
            "superiority_established",
            "fresh_confirmation_established",
            "paper_campaign_authorized",
        )
    ):
        raise ValueError("Parent report source/scope differs")
    expected = {
        **{lane: campaign / lane / name for lane, name in STATIC.items()},
        "history": campaign / "history/GATE_history_v13.json",
    }
    if set(report["gates"]) != set(expected):
        raise ValueError("Complete five-lane parent report required")
    for lane, path in expected.items():
        entry = report["gates"][lane]
        if not path.is_file():
            if (
                lane not in STATIC
                or not static_errors(campaign, lane)
                or entry.get("status") != "missing_gate"
                or Path(entry["path"]).resolve() != path.resolve()
            ):
                raise ValueError("Unexplained missing parent gate cannot complete reporting")
            continue
        if (
            Path(entry["path"]).resolve() != path.resolve()
            or entry["sha256"] != sha(path)
            or entry["evidence"] != read(path)
        ):
            raise ValueError("Parent report gate reference/hash/evidence differs")
    actual_errors = {
        str(path.resolve()): sha(path) for path in (campaign / "errors").glob("*.json")
    }
    recorded = {str(Path(row["path"]).resolve()): row["sha256"] for row in report["errors"]}
    if actual_errors != recorded or any(
        row["evidence"] != read(row["path"]) for row in report["errors"]
    ):
        raise ValueError("Parent report failure inventory differs")
    return dict(
        path=str(campaign / "FINAL_report.json"),
        sha256=sha(campaign / "FINAL_report.json"),
        summary_exists=(campaign / "SUMMARY.md").is_file(),
    )


def invoke_finish(root, campaign, source):
    """A distinct process with the immutable V13 source, never V14 imports."""
    script = (
        'set -euo pipefail\ncd "$1"\nsource mosaickv/scripts/cache_env.sh\n'
        'export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"\n'
        'export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""\n'
        "export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
        "bash mosaickv/scripts/assert_clean_worktree.sh\n"
        "exec /scratch/djy8hg/env/mosaickv/bin/python "
        'mosaickv/scripts/evicastkv_review_v13.py finish --root "$2"'
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [
        "bash",
        "--noprofile",
        "--norc",
        "-c",
        script,
        "v13-report-rescue",
        str(source),
        str(campaign),
    ]
    write_once(
        root / "external-finish-command.json",
        dict(
            command=command,
            parent_source_sha=PARENT_SHA,
            external_process=True,
            V14_PYTHONPATH_not_inherited=True,
        ),
    )
    with (
        (root / "external-finish.stdout").open("x") as stdout,
        (root / "external-finish.stderr").open("x") as stderr,
    ):
        subprocess.run(
            command,
            cwd=source,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            timeout=300,
            check=True,
        )
    return verify_report(campaign)


def advance(root, hop=0):
    root = Path(root)
    plan = load(root)
    if type(hop) is not int or not 0 <= hop < MAX_HOPS:
        raise ValueError("Registered bounded completion event index required")
    event = root / f"events/{hop}"
    write_once(event / "start.json", dict(job_id=os.environ.get("SLURM_JOB_ID"), hop=hop))
    try:
        campaign, source = Path(plan["parent_campaign"]), Path(plan["parent_source"])
        bindings, dependencies, recorded_failures = {}, [], {}
        for lane in STATIC:
            failures = static_errors(campaign, lane)
            if failures:
                recorded_failures[lane] = failures
                bindings.update({row["path"]: row["sha256"] for row in failures})
            acknowledged = False
            for name in (lane, f"after-{lane}"):
                path = campaign / f"jobs/{name}-submission.json"
                if (campaign / f"jobs/{name}-request.json").is_file() and not path.is_file():
                    raise ValueError(
                        "Unacknowledged static scheduler request cannot be declared terminal"
                    )
                if path.is_file():
                    job, evidence = job_ledger(
                        path, source=source, campaign=campaign, allow_existing_gpu=name == lane
                    )
                    dependencies.append(job)
                    bindings[evidence["path"]] = evidence["sha256"]
                    acknowledged = True
            if not acknowledged and not failures:
                raise ValueError("No actual static job ledger or explicit preparation failure")
        history_gate = campaign / "history/GATE_history_v13.json"
        current = None
        if not history_gate.is_file():
            current, job = frontier(campaign, source)
            dependencies.append(job)
            bindings[current["path"]] = current["sha256"]
        original_completion = campaign / "history/jobs/completion-submission.json"
        if any(
            path.name != "completion-submission-error.json"
            and not path.with_name(
                path.name.replace("-submission-error.json", "-submission.json")
            ).is_file()
            for path in (campaign / "history/jobs").glob("*-submission-error.json")
        ):
            raise ValueError("Unacknowledged noncompletion history request remains ambiguous")
        if original_completion.is_file():
            job, evidence = job_ledger(original_completion, source=source, campaign=campaign)
            dependencies.append(job)
            bindings[evidence["path"]] = evidence["sha256"]
        retained, receipt = resolve_dependencies(dependencies)
        receipt.update(
            job_ledger_bindings=bindings,
            history_callback=current,
            recorded_static_failures=recorded_failures,
        )
        write_once(event / "dependency-resolution.json", receipt)
        if current is not None and current["job"] in receipt["terminal_omitted"]:
            raise ValueError("History callback ended without its gate or a newer active callback")
        original_failures = [
            dict(path=str(path), sha256=sha(path), evidence=read(path))
            for path in sorted((campaign / "history/jobs").glob("*-submission-error.json"))
        ]
        if retained:
            if hop + 1 >= MAX_HOPS:
                raise ValueError("Bounded 32-event completion reattachment exhausted")
            job = submit(root, hop + 1, retained)
            write_once(
                event / "continued.json",
                dict(
                    callback=job,
                    dependencies=retained,
                    history_callback=current,
                    original_history_submission_failures=original_failures,
                ),
            )
            return dict(status="attached_to_existing_CPU_callbacks", job=job, hop=hop)
        if not history_gate.is_file():
            raise ValueError("History callback is terminal without a gate or newer active callback")
        history = read(history_gate)
        if history["source_sha"] != PARENT_SHA:
            raise ValueError("History terminal gate has wrong producer")
        for path, expected in history.get("bindings", {}).items():
            if sha(Path(path)) != expected:
                raise ValueError("History terminal evidence changed")
        for lane, name in STATIC.items():
            path = campaign / lane / name
            if not path.is_file():
                if not static_errors(campaign, lane):
                    raise ValueError("Unexplained missing static-lane gate")
                continue
            gate = read(path)
            if gate.get("source_sha") != PARENT_SHA:
                raise ValueError("Missing or differently produced static lane gate")
        existing = (campaign / "FINAL_report.json").exists()
        report = verify_report(campaign) if existing else invoke_finish(root, campaign, source)
        gate = dict(
            status="reused_verified_parent_report"
            if existing
            else "recovered_parent_report_not_paper_ready",
            completed=True,
            source_sha=plan["source_sha"],
            parent_source_sha=PARENT_SHA,
            parent_report=report,
            history_gate=dict(path=str(history_gate), sha256=sha(history_gate)),
            original_history_submission_failures=original_failures,
            completed_with_recorded_static_failures=bool(recorded_failures),
            recorded_static_failures=recorded_failures,
            accounting_receipt=dict(
                path=str(event / "dependency-resolution.json"),
                sha256=sha(event / "dependency-resolution.json"),
            ),
            new_GPU_jobs=0,
            new_scans=0,
            superiority_established=False,
            fresh_confirmation_authorized=False,
            paper_campaign_authorized=False,
        )
        write_once(root / GATE, gate)
        return gate
    except Exception as error:
        if isinstance(error, DependencyResolutionError):
            write_once(event / "dependency-resolution-failure.json", error.receipt)
        gate = dict(
            status="blocked_completion_rescue",
            completed=False,
            source_sha=plan["source_sha"],
            error=f"{type(error).__name__}: {error}",
            hop=hop,
            new_GPU_jobs=0,
            new_scans=0,
            automatic_retry=False,
            superiority_established=False,
            fresh_confirmation_authorized=False,
        )
        write_once(root / GATE, gate)
        return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "advance"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hop", type=int, default=0)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--parent-source", type=Path, default=PARENT_SOURCE)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        register(
            args.root,
            parent_campaign=args.parent,
            source=args.parent_source,
            execute=args.execute,
            cpu_gate_root=args.cpu_gate_root,
        )
        if args.mode == "register"
        else advance(args.root, args.hop)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
