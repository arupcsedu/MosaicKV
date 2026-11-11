#!/usr/bin/env python3
"""At most four afterany CPU frontier segments from verified V15 recovery.

No restart, metadata-scope exemption, data/model access or freshness license.
Old cc2768 producer receipts remain immutable and separate from new workers.
Only checkpoint JSON whitespace changes; canonical state digests do not.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import evicastkv_review_freshness_stream as scanner
from evicastkv_freshness_empty_parquet_continue import empty_projection_parquet
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_history_segment_recovery import IDENTITIES, OLD_SHA
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
WORKDIR = Path("/scratch/djy8hg/workdir").resolve()
PARENT_SOURCE = "279c4f6336bd852d2ccd8408d519e1ddfa2a53b1"
MAX_SEGMENTS, SHARDS, SECONDS = 4, 8, 210
MAX_STATE, MAX_EXPORT = 256 * 2**20, 64 * 2**20
BRIDGE_FILES = (
    "mosaickv/scripts/evicastkv_review_freshness_stream.py",
    "mosaickv/scripts/evicastkv_review_freshness_audit.py",
    "mosaickv/scripts/evicastkv_boundary_operators.py",
    "mosaickv/scripts/evicastkv_freshness_empty_parquet_continue.py",
    "mosaickv/scripts/evicastkv_freshness_producer_canary.py",
)
POLICY = dict(
    maximum_segments=MAX_SEGMENTS,
    worker_CPUs=1,
    workers=SHARDS,
    worker_seconds=SECONDS,
    analyzer_CPUs=4,
    maximum_state_bytes=MAX_STATE,
    maximum_export_bytes=MAX_EXPORT,
    line_bytes=8 * 2**20,
    maximum_source_bytes_per_worker=2 * 2**30,
    no_restart=True,
    automatic_unbounded_resubmission=False,
    fresh_confirmation_authorized=False,
    external_video_journals_unresolved=True,
)


def read(path, expected=None, *, bound=MAX_STATE):
    return read_json(read_bytes(path, bound=bound, expected=expected)[0])


def bridge():
    files, functions = {}, {}
    for name in BRIDGE_FILES:
        raw = subprocess.check_output(
            ["git", "show", f"{OLD_SHA}:{name}"], cwd=SOURCE, timeout=15, stderr=subprocess.DEVNULL
        )
        current = SOURCE / name
        expected = hashlib.sha256(raw).hexdigest()
        if len(raw) > 2**20 or sha(current) != expected:
            raise ValueError("historical scanner/collector dependency changed; bridge refused")
        files[name] = expected
    name = "mosaickv/src/mosaickv/evaluation/fox_campaign.py"
    old = ast.parse(
        subprocess.check_output(
            ["git", "show", f"{OLD_SHA}:{name}"], cwd=SOURCE, timeout=15, stderr=subprocess.DEVNULL
        )
    )
    new = ast.parse((SOURCE / name).read_text())
    for symbol in ("digest", "sha", "write_once"):
        previous = [n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == symbol]
        current = [n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == symbol]
        if len(previous) != 1 or len(current) != 1:
            raise ValueError("historical metadata primitive unavailable")
        left, right = (ast.dump(n[0], include_attributes=False) for n in (previous, current))
        if left != right:
            raise ValueError("historical metadata primitive semantics changed")
        functions[symbol] = hashlib.sha256(right.encode()).hexdigest()
    return dict(
        origin_source_sha=OLD_SHA,
        files=files,
        primitive_AST_sha256s=functions,
        identity_semantics_unchanged=True,
        historical_source_executed=False,
    )


def bounded_write(path, value, limit, *, compact=False):
    """Atomic write-once JSON; a cap failure never exposes a partial checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            encoder = json.JSONEncoder(
                sort_keys=True, allow_nan=False, separators=(",", ":") if compact else None
            )
            count = 0
            for fragment in encoder.iterencode(value):
                raw = fragment.encode()
                count += len(raw)
                if count + 1 > limit:
                    raise ValueError("registered immutable checkpoint/export size exceeded")
                handle.write(raw)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def identity(value):
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError("explicit numeric scheduler identity required")
    return value


def completion_check(command):
    if command is None:
        return
    if (
        not isinstance(command, list)
        or not command
        or command[0] != "sbatch"
        or any(not isinstance(arg, str) for arg in command)
        or any(
            arg.startswith(("--dependency", "--gres", "--gpus", "--array", "-G", "--tres-per-task"))
            or "bii-gpu" in arg
            for arg in command
        )
    ):
        raise ValueError("one explicit CPU-only sbatch completion hook required")


def load(root):
    plan = read(Path(root) / "registration.json", bound=2**20)
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    if plan["policy"] != POLICY or plan["source_bridge"] != bridge():
        raise ValueError("registered scanner/state transport policy changed")
    for name, expected in plan["control_bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable recovery/receipt provenance changed")
    return plan


def submit(root, label, mode, number, *, dependencies=(), array=False, completion=None):
    root = Path(root)
    plan = load(root)
    if not plan["execute"]:
        raise ValueError("dry registration cannot submit jobs")
    dependencies = list(dict.fromkeys(identity(value) for value in dependencies))
    if completion is None:
        command = [
            "sbatch",
            "--parsable",
            "--account=bii_dsc_community",
            "--partition=bii,standard",
            "--nodes=1",
            "--ntasks=1",
            f"--job-name=fox16-history-{label}",
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
                str(SOURCE / "mosaickv/slurm/evicastkv_history_closure_v16.sbatch"),
                str(SOURCE),
                str(root),
                mode,
                str(number),
            ]
        )
    else:
        completion_check(completion)
        command = [
            completion[0],
            *(["--dependency=afterany:" + ":".join(dependencies)] if dependencies else []),
            *completion[1:],
        ]
    write_once(root / "jobs" / f"{label}-request.json", dict(command=command, single_attempt=True))
    try:
        raw = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=30)
        job = identity(raw.strip().split(";", 1)[0])
    except Exception as error:
        write_once(
            root / "jobs" / f"{label}-submission-error.json",
            dict(
                reason=type(error).__name__,
                automatic_retry=False,
                response_uncertain=not isinstance(error, subprocess.CalledProcessError),
            ),
        )
        raise
    write_once(root / "jobs" / f"{label}-submission.json", dict(job=job, command=command))
    return job


def current_dependency():
    value = os.getenv("SLURM_JOB_ID")
    return [identity(value)] if value else []


def register(root, *, cpu_gate_root, parent_recovery, execute=False, completion_argv=None):
    root, parent = Path(root).resolve(), Path(parent_recovery).resolve(strict=True)
    if type(execute) is not bool:
        raise ValueError("explicit boolean execution scope required")
    if root.exists() or not root.is_relative_to(WORKDIR):
        raise ValueError("new immutable workspace history-closure root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    source_bridge = bridge()
    completion_check(completion_argv)
    gate_path, parent_plan_path = (
        parent / "GATE_history_segment_recovery.json",
        parent / "plan.json",
    )
    gate, old = read(gate_path), read(parent_plan_path)
    if (
        gate["source_sha"] != PARENT_SOURCE
        or old["source_sha"] != PARENT_SOURCE
        or gate["historical_worker_source_sha"] != OLD_SHA
        or gate["missing_shards"]
        or gate["status"] != "recovered_completed_segment_nine_metadata_not_freshness_certified"
        or gate["fresh_confirmation_authorized"] is not False
    ):
        raise ValueError("actual complete V15 eight-worker recovery gate required")
    old_root = Path(old["parent"]).resolve(strict=True)
    original_plan_path = old_root / "segment-9/plan.json"
    original_sha = old["bindings"].get(str(original_plan_path))
    if not isinstance(original_sha, str) or re.fullmatch(r"[0-9a-f]{64}", original_sha) is None:
        raise ValueError("original segment plan must be explicitly bound by recovery")
    original = read(original_plan_path, original_sha)
    if original["source_sha"] != OLD_SHA or original["same_scanner_file_sha256"] != sha(
        Path(scanner.__file__)
    ):
        raise ValueError("old worker policy does not match the reviewed scanner bridge")
    inventory = parent / "combined_identity_inventory.json"
    read(inventory, gate["identity_inventory_sha256"], bound=MAX_EXPORT)
    bound = {
        str(gate_path): sha(gate_path),
        str(parent_plan_path): sha(parent_plan_path),
        str(original_plan_path): sha(original_plan_path),
        str(inventory): gate["identity_inventory_sha256"],
    }
    for name, expected in gate["receipt_bindings"].items():
        read(name, expected) if Path(name).suffix == ".json" else read_bytes(
            name, expected=expected
        )
        bound[name] = expected
    rows = []
    for row in old["workers"]:
        if gate["receipt_bindings"].get(row["stdout"]) != row["stdout_sha256"]:
            raise ValueError("original clean-source stdout receipt is not bound by recovery")
        path = str(Path(row["path"]).resolve(strict=True))
        if gate["worker_bindings"].get(path) != row["sha256"]:
            raise ValueError("actual worker is not bound by the recovered gate")
        value = read(path, row["sha256"])
        if (
            value["source_sha"] != OLD_SHA
            or value["index"] != row["index"]
            or value["state_sha256"] != digest(value["state"])
        ):
            raise ValueError("recovered state differs from its original worker")
        rows.append(
            dict(index=row["index"], path=path, sha256=row["sha256"], producer_source_sha=OLD_SHA)
        )
    if sorted(r["index"] for r in rows) != list(range(SHARDS)):
        raise ValueError("all eight unique original worker states required")
    if tuple(Path(p).resolve(strict=True) for p in original["roots"]) != tuple(
        Path(p).resolve(strict=True) for p in scanner.ROOTS
    ):
        raise ValueError("original historical roots differ from the unchanged scanner scope")
    registration = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        source_bridge=source_bridge,
        origin_source_sha=OLD_SHA,
        parent_recovery_source_sha=PARENT_SOURCE,
        parent_recovery=str(parent),
        control_bindings=bound,
        original_worker_bindings=gate["worker_bindings"],
        initial_workers=rows,
        initial_inventory=str(inventory),
        roots=original["roots"],
        policy=POLICY,
        execute=bool(execute),
        completion_argv=completion_argv,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "registration.json", registration)
    if not execute:
        return dict(status="dry_registered_no_submissions", maximum_segments=MAX_SEGMENTS)
    return start(root, 1, rows, inventory)


def start(root, number, parents, inventory):
    root = Path(root)
    plan = load(root)
    if plan["execute"] is not True:
        raise ValueError("dry registration cannot advance history")
    if not 1 <= number <= MAX_SEGMENTS:
        raise ValueError("finite segment count exceeded")
    if sorted(row["index"] for row in parents) != list(range(SHARDS)):
        raise ValueError("eight parent states required before a new segment")
    for row in parents:
        old = read(row["path"], row["sha256"])
        if (
            old["source_sha"] != row["producer_source_sha"]
            or digest(old["state"]) != old["state_sha256"]
        ):
            raise ValueError("parent state/source/hash differs before segment")
    segment = root / f"segment-{number}"
    write_once(
        segment / "plan.json",
        dict(
            source_sha=plan["source_sha"],
            number=number,
            parents=parents,
            parent_inventory=str(inventory),
            parent_inventory_sha256=sha(Path(inventory)),
            registration_sha256=sha(root / "registration.json"),
            origin_source_sha=OLD_SHA,
        ),
    )
    worker = None
    try:
        worker = submit(
            root,
            f"segment-{number}",
            "worker",
            number,
            dependencies=current_dependency(),
            array=True,
        )
        analyzer = submit(root, f"after-segment-{number}", "advance", number, dependencies=[worker])
        return dict(
            status="submitted_finite_segment", segment=number, workers=worker, analysis=analyzer
        )
    except Exception as error:
        return finish(
            root,
            "submission_failed_no_retry",
            number=None,
            error=type(error).__name__,
            dependencies=[worker] if worker else [],
            submission_uncertain=not isinstance(error, subprocess.CalledProcessError),
        )


def worker(root, number, index):
    root = Path(root)
    plan = load(root)
    if plan["execute"] is not True:
        raise ValueError("dry registration cannot scan history")
    if type(index) is not int or not 0 <= index < SHARDS or not 1 <= number <= MAX_SEGMENTS:
        raise ValueError("registered segment/shard required")
    segment = root / f"segment-{number}"
    own = read(segment / "plan.json")
    if own["registration_sha256"] != sha(root / "registration.json") or own["number"] != number:
        raise ValueError("segment registration differs")
    row = next(row for row in own["parents"] if row["index"] == index)
    previous = read(row["path"], row["sha256"])
    if (
        previous["source_sha"] != row["producer_source_sha"]
        or digest(previous["state"]) != previous["state_sha256"]
    ):
        raise ValueError("imported worker state is not immutable")
    write_once(segment / "locks" / f"shard-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    state = copy.deepcopy(previous["state"])
    with (
        patch.object(scanner, "_parquet", empty_projection_parquet),
        patch.object(scanner, "MAX_LINE", POLICY["line_bytes"]),
    ):
        scanner.advance(
            state,
            tuple(Path(p) for p in plan["roots"]),
            shard_index=index,
            shard_count=SHARDS,
            budget=scanner.Budget(
                seconds=SECONDS, maximum_bytes=POLICY["maximum_source_bytes_per_worker"]
            ),
        )
    value = dict(
        source_sha=plan["source_sha"],
        origin_source_sha=OLD_SHA,
        index=index,
        parent_worker_path=row["path"],
        parent_worker_sha256=row["sha256"],
        parent_producer_source_sha=row["producer_source_sha"],
        state=state,
        state_sha256=digest(state),
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
    )
    target = segment / "workers" / f"shard-{index}.json"
    bounded_write(target, value, MAX_STATE, compact=True)
    return dict(index=index, sha256=sha(target), frontier_exhausted=state["traversal_complete"])


def progress_state(state):
    return digest(
        {
            name: state[name]
            for name in (
                "frontier",
                "processed_files",
                "seen_files",
                "seen_directories",
                "verification_index",
                "traversal_complete",
            )
        }
    )


def advance(root, number):
    root = Path(root)
    plan = load(root)
    if plan["execute"] is not True:
        raise ValueError("dry registration cannot advance history")
    if not 1 <= number <= MAX_SEGMENTS:
        raise ValueError("bounded afterany callback index required")
    segment = root / f"segment-{number}"
    own = read(segment / "plan.json")
    if own["registration_sha256"] != sha(root / "registration.json") or own["number"] != number:
        raise ValueError("afterany segment registration differs")
    inventory = read(own["parent_inventory"], own["parent_inventory_sha256"], bound=MAX_EXPORT)
    merged = {
        key: {tuple(v) if isinstance(v, list) else v for v in inventory["identities"][key]}
        for key in IDENTITIES
    }
    missing, parents, errors, summaries = [], [], [], []
    changed = False
    for index in range(SHARDS):
        path = segment / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        value = read(path)
        parent = next(row for row in own["parents"] if row["index"] == index)
        old = read(parent["path"], parent["sha256"])
        state = value["state"]
        if (
            value["source_sha"] != plan["source_sha"]
            or value["origin_source_sha"] != OLD_SHA
            or value["index"] != index
            or value["state_sha256"] != digest(state)
            or value["parent_worker_sha256"] != parent["sha256"]
            or value["parent_worker_path"] != parent["path"]
            or value["parent_producer_source_sha"] != parent["producer_source_sha"]
            or old["source_sha"] != parent["producer_source_sha"]
            or state["errors"][: len(old["state"]["errors"])] != old["state"]["errors"]
        ):
            raise ValueError("new state lineage/digest differs or old errors were dropped")
        if any(
            state[name].get(key) != item
            for name in ("bindings", "directories")
            for key, item in old["state"][name].items()
        ):
            raise ValueError("old source/storage signature proof disappeared")
        if (
            state["processed_files"] < old["state"]["processed_files"]
            or state["verification_index"] < old["state"]["verification_index"]
            or any(
                not set(old["state"][key]) <= set(state[key])
                for key in ("seen_files", "seen_directories")
            )
        ):
            raise ValueError("historical work was reset or forgotten")
        for key in IDENTITIES:
            current = {tuple(v) if isinstance(v, list) else v for v in state[key]}
            previous = {tuple(v) if isinstance(v, list) else v for v in old["state"][key]}
            if not previous <= current:
                raise ValueError("old projected identity disappeared")
            merged[key].update(current)
        changed |= progress_state(old["state"]) != progress_state(state)
        errors.extend(state["errors"])
        summaries.append(
            dict(
                index=index,
                frontier_exhausted=state["traversal_complete"],
                pending_frontier_items=len(state["frontier"]),
                processed_files=state["processed_files"],
            )
        )
        parents.append(
            dict(
                index=index,
                path=str(path),
                sha256=sha(path),
                producer_source_sha=plan["source_sha"],
            )
        )
    if missing:
        return finish(
            root,
            "missing_or_failed_workers",
            number=number,
            error="all eight immutable outputs required",
            missing=missing,
        )
    target = segment / "combined_identity_inventory.json"
    bounded_write(
        target,
        dict(
            source_sha=plan["source_sha"],
            origin_source_sha=OLD_SHA,
            identities={key: sorted(v) for key, v in merged.items()},
            worker_bindings={row["path"]: row["sha256"] for row in parents},
            complete_historical_id_inventory=False,
            fresh_confirmation_authorized=False,
        ),
        MAX_EXPORT,
    )
    bounded_write(segment / "preserved_errors.json", errors, MAX_EXPORT)
    decision = dict(
        source_sha=plan["source_sha"],
        positive_state_progress=changed,
        frontier_exhausted=all(row["frontier_exhausted"] for row in summaries),
        pending_frontier_items=sum(row["pending_frontier_items"] for row in summaries),
        original_and_new_coverage_errors=len(errors),
        shard_summary=summaries,
        worker_bindings={row["path"]: row["sha256"] for row in parents},
        identity_inventory_sha256=sha(target),
        errors_sha256=sha(segment / "preserved_errors.json"),
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
    )
    write_once(segment / "decision.json", decision)
    if decision["frontier_exhausted"]:
        return finish(root, "frontier_exhausted_coverage_still_unresolved", number=number)
    if not changed or number == MAX_SEGMENTS:
        return finish(
            root, "no_state_progress" if not changed else "maximum_four_segments", number=number
        )
    return start(root, number + 1, parents, target)


def finish(
    root,
    reason,
    *,
    number=None,
    error=None,
    missing=(),
    dependencies=(),
    submission_uncertain=False,
):
    root = Path(root)
    plan = load(root)
    last = root / f"segment-{number}" if number else None
    path = last / "decision.json" if last else None
    decision = read(path) if path and path.exists() else None
    gate = dict(
        status="finite_history_closure_stopped_not_freshness_certified",
        source_sha=plan["source_sha"],
        origin_source_sha=OLD_SHA,
        stop_reason=reason,
        error=error,
        last_segment=number,
        last_decision_sha256=sha(path) if decision else None,
        missing_shards=list(missing),
        frontier_exhausted=bool(decision and decision["frontier_exhausted"]),
        pending_frontier_items=decision["pending_frontier_items"] if decision else None,
        original_and_new_coverage_errors=decision["original_and_new_coverage_errors"]
        if decision
        else None,
        identity_inventory_sha256=decision["identity_inventory_sha256"] if decision else None,
        known_outstanding_worker_ids=list(dependencies),
        scheduler_submission_uncertain=submission_uncertain,
        original_errors_preserved=True,
        error_waivers=0,
        roots_restarted=False,
        old_producer_relabelled=False,
        fresh_confirmation_authorized=False,
        external_video_journals_unresolved=True,
        materialization_authorized=False,
    )
    write_once(root / "GATE_history_closure_v16.json", gate)
    if plan["completion_argv"] is not None and plan["execute"] and not submission_uncertain:
        try:
            completion = submit(
                root,
                "completion",
                "unused",
                0,
                dependencies=[*dependencies, *current_dependency()],
                completion=plan["completion_argv"],
            )
            return dict(gate=gate, completion=completion)
        except Exception as error:
            return dict(gate=gate, completion=None, completion_error=type(error).__name__)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "advance"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--number", type=int, required=True)
    args = parser.parse_args()
    try:
        result = (
            worker(args.root, args.number, int(os.environ["SLURM_ARRAY_TASK_ID"]))
            if args.mode == "worker"
            else advance(args.root, args.number)
        )
    except Exception as error:
        if args.mode == "worker":
            raise
        result = finish(
            args.root,
            "analysis_or_transition_failed",
            number=args.number,
            error=type(error).__name__,
        )
    print(json.dumps(result))
