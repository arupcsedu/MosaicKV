#!/usr/bin/env python3
"""One bounded CPU continuation fork, never an automatic freshness license.

Imports exactly eight CP005 states into new write-once roots, preserving the
old producer/policy separately. Identity functions are unchanged byte-for-byte;
only the strict JSONL line bound becomes 8 MiB. The reviewed config overlay is
evidence, not a directory-skip rule. Oversized prior records have a separate
registered queue. No automatic resubmission, model or dataset materialization.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import evicastkv_freshness_line_recovery as lines
import evicastkv_review_freshness_stream as scanner
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = Path(
    "/scratch/djy8hg/audits/fox-review-v8-consumed-diagnostics-20261110/freshness-continuation"
)
OLD_SOURCE_SHA = "64be064d3e6622890f6fe96a556cedc0161f8f4c"
SHARDS, LINE_BYTES, FRONTIER_SECONDS, WORKER_SECONDS = 8, 8 * 2**20, 180, 240


def read(path, expected=None, *, limit=256 * 2**20):
    raw, _actual = read_bytes(path, bound=limit, expected=expected)
    return read_json(raw)


def canonical_bindings(bindings):
    """Resolve explicit evidence aliases, rejecting conflicting or broken keys."""
    result = {}
    for name, expected in bindings.items():
        canonical = str(Path(name).resolve(strict=True))
        if canonical in result and result[canonical] != expected:
            raise ValueError("conflicting aliases in immutable evidence bindings")
        result[canonical] = expected
    return result


def prepare(root, parent=DEFAULT_PARENT, *, cpu_gate_root=None, coverage_root=None):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch frontier-fork root required")
    cpu_gate_root = Path(cpu_gate_root) if cpu_gate_root else root.parent
    coverage_root = Path(coverage_root) if coverage_root else root.parent / "freshness"
    revision = clean_source(SOURCE)
    cpu = cpu_gate(cpu_gate_root, revision)
    inventory_path = parent / "aggregate/inventory.json"
    inventory = read(inventory_path)
    coverage_gate_path = coverage_root / "GATE_freshness_recipe_recovery.json"
    coverage = read(coverage_gate_path)
    if (
        coverage["status"] != "completed_exact_config_recovery_coverage_overlay"
        or coverage["indexed_configs"] != 9403
        or coverage["source_sha"] != revision
        or coverage["fresh_confirmation_authorized"] is not False
    ):
        raise ValueError("current-source exact 9403-config coverage evidence required")
    overlay_path = coverage_root / "indexed_coverage_overlay.json"
    overlay = read(overlay_path, coverage["coverage_overlay_sha256"])
    if (
        len(overlay) != 9403
        or len({row["config_path"] for row in overlay}) != 9403
        or any(
            row["byte_identical"] is not True or row["error_waived"] is not False for row in overlay
        )
    ):
        raise ValueError("coverage overlay rows differ from the reviewed producer proof")
    coverage_plan_path = coverage_root / "plan.json"
    coverage_plan = read(coverage_plan_path, coverage["plan_sha256"])
    if canonical_bindings(coverage_plan["bindings"]).get(str(inventory_path)) != sha(
        inventory_path
    ):
        raise ValueError("coverage overlay belongs to a different original inventory")
    all_offsets = {}
    for row in inventory["errors"]:
        if row.get("error") == lines.PREFIX:
            if type(row.get("offset")) is not int or row["offset"] < 0:
                raise ValueError("registered oversized-record offset invalid")
            all_offsets[row["path"], row["offset"]] = row
    requested = {path for path, _offset in all_offsets}
    source_bindings, old_shards, imported = {}, [], []
    current_scanner_sha = sha(Path(scanner.__file__))
    bound = {
        str(inventory_path): sha(inventory_path),
        str(coverage_gate_path): sha(coverage_gate_path),
        str(overlay_path): sha(overlay_path),
        str(Path(scanner.__file__)): current_scanner_sha,
        str(Path(lines.__file__)): sha(Path(lines.__file__)),
        str(coverage_plan_path): coverage["plan_sha256"],
    }
    for checkpoint, expected in sorted(inventory["shard_bindings"].items()):
        path = Path(checkpoint)
        saved = read(path, expected)
        if path.name != "checkpoint-000005.json" or saved["number"] != 5:
            raise ValueError("only the explicitly reviewed CP005 snapshot can be forked")
        registration_path = path.parent / "registration.json"
        registration = read(registration_path, saved["registration_sha256"])
        if (
            registration["source_sha"] != OLD_SOURCE_SHA
            or registration["shard_count"] != SHARDS
            or registration["source_sha256"] != current_scanner_sha
        ):
            raise ValueError("old scanner semantics/source differs; explicit review required")
        state = saved["state"]
        if digest(state) != saved["state_sha256"]:
            raise ValueError("imported checkpoint state binding changed")
        index = registration["shard_index"]
        old_shards.append(
            dict(
                index=index,
                checkpoint=checkpoint,
                checkpoint_sha256=expected,
                registration=registration,
                state_sha256=saved["state_sha256"],
            )
        )
        imported.append((index, state))
        bound.update({checkpoint: expected, str(registration_path): saved["registration_sha256"]})
        for name in requested.intersection(state["bindings"]):
            value = state["bindings"][name]
            if name in source_bindings and source_bindings[name] != value:
                raise ValueError("conflicting complete bindings for recovery source")
            source_bindings[name] = value
    if sorted(row["index"] for row in old_shards) != list(range(SHARDS)):
        raise ValueError("every registered shard must be imported exactly once")
    roots = [Path(path) for path in inventory["roots"]]
    queue = []
    for (name, offset), original in sorted(all_offsets.items()):
        queue.append(
            dict(
                path=name,
                offset=offset,
                original_error_sha256=digest(original),
                index=scanner.owner(Path(name), roots, SHARDS),
                complete_source_binding_available=name in source_bindings,
            )
        )
    plan = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        old_producer_sha=OLD_SOURCE_SHA,
        old_shards=old_shards,
        roots=[str(p) for p in roots],
        bindings=bound,
        same_scanner_file_sha256=current_scanner_sha,
        identity_projection_and_collection_unchanged=True,
        explicit_policy_difference=dict(old_MAX_LINE=2**20, new_MAX_LINE=LINE_BYTES),
        frontier_seconds=FRONTIER_SECONDS,
        worker_seconds=WORKER_SECONDS,
        maximum_bytes_per_segment=2 * 2**30,
        shard_count=SHARDS,
        maximum_parallel_CPUs=8,
        analysis_CPUs=4,
        maximum_total_CPUs=12,
        segment_count=1,
        automatic_resubmission=False,
        registered_oversize_offsets=len(queue),
        separate_offset_recovery_limits=dict(
            maximum_file_bytes=lines.MAX_BYTES,
            maximum_record_bytes=lines.MAX_BYTES,
            complete_original_source_binding_required=True,
        ),
        coverage_overlay_configs=9403,
        coverage_is_not_a_walk_skip_rule=True,
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "oversize_queue.json", queue)
    write_once(root / "recovery_source_bindings.json", source_bindings)
    for index, state in imported:
        write_once(
            root / "imports" / f"shard-{index:03d}.json",
            dict(state=state, state_sha256=digest(state), old_source_sha=OLD_SOURCE_SHA),
        )
    plan["auxiliary_bindings"] = {
        str(path): sha(path)
        for path in [
            root / "oversize_queue.json",
            root / "recovery_source_bindings.json",
            *(root / "imports" / f"shard-{index:03d}.json" for index in range(SHARDS)),
        ]
    }
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_one_finite_frontier_segment",
        workers=SHARDS,
        plan_sha256=sha(root / "plan.json"),
        oversized_offsets=len(queue),
    )


def load(root):
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in {**plan["bindings"], **plan["auxiliary_bindings"]}.items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable frontier-fork evidence changed")
    return plan


def worker(root, index):
    root = Path(root).resolve()
    started = time.monotonic()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered shard index required")
    saved = read(root / "imports" / f"shard-{index:03d}.json")
    expected = next(row["state_sha256"] for row in plan["old_shards"] if row["index"] == index)
    if saved["state_sha256"] != expected or digest(saved["state"]) != expected:
        raise ValueError("imported state content differs from bound parent")
    write_once(
        root / "locks" / f"worker-{index:03d}.json",
        dict(index=index, job_id=os.environ.get("SLURM_JOB_ID")),
    )
    state = copy.deepcopy(saved["state"])
    before = dict(
        files=state["processed_files"],
        records=state["projected_records"],
        frontier=len(state["frontier"]),
        errors=len(state["errors"]),
    )
    roots = tuple(Path(path) for path in plan["roots"])
    with patch.object(scanner, "MAX_LINE", LINE_BYTES):
        scanner.advance(
            state,
            roots,
            shard_index=index,
            shard_count=SHARDS,
            budget=scanner.Budget(seconds=FRONTIER_SECONDS, maximum_bytes=2 * 2**30),
        )
    frontier_complete = state["traversal_complete"]
    queue = [row for row in read(root / "oversize_queue.json") if row["index"] == index]
    bindings = read(root / "recovery_source_bindings.json")
    recovered, observations, remaining = [], [], []
    deadline = started + WORKER_SECONDS
    for number, row in enumerate(queue):
        if time.monotonic() >= deadline - 5:
            remaining.extend(queue[number:])
            break
        captured = {}
        original_collect = lines.collect

        def capture(
            projected, target, path, *, original_collect=original_collect, captured=captured
        ):
            original_collect(projected, target, path)
            captured.update(target)

        result = dict(
            path=row["path"],
            offset=row["offset"],
            original_error_sha256=row["original_error_sha256"],
        )
        try:
            if row["path"] not in bindings:
                raise ValueError("registered record lacks a complete saved source binding")
            with patch.object(lines, "collect", capture):
                result.update(lines.project_record(row, bindings[row["path"]], roots, deadline - 5))
            # Only a fully verified, complete before/after source transaction
            # exports metadata identities. Original errors remain unchanged.
            recovered.append(dict(path=row["path"], offset=row["offset"], identities=captured))
        except TimeoutError:
            remaining.extend(queue[number:])
            break
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            result.update(status="blocked_not_waived", error=f"{type(exc).__name__}: {exc}")
        observations.append(result)
    output = dict(
        source_sha=plan["source_sha"],
        old_source_sha=OLD_SOURCE_SHA,
        imported_parent_state_sha256=expected,
        state=state,
        state_sha256=digest(state),
        frontier_complete=frontier_complete,
        before=before,
        files_added=state["processed_files"] - before["files"],
        records_added=state["projected_records"] - before["records"],
        recovery_observations=observations,
        recovered_metadata=recovered,
        recovery_queue_pending=remaining,
        wall_seconds=time.monotonic() - started,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    path = root / "workers" / f"shard-{index:03d}.json"
    write_once(path, output)
    return dict(
        index=index,
        frontier_complete=frontier_complete,
        files_added=output["files_added"],
        records_added=output["records_added"],
        recovered_records=len(recovered),
        pending_recovery_records=len(remaining),
        worker_sha256=sha(path),
    )


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    complete, missing, records, bound = [], [], [], {}
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index:03d}.json"
        if not path.is_file():
            missing.append(index)
            continue
        record = read(path)
        bound[str(path)] = sha(path)
        if (
            record["source_sha"] != plan["source_sha"]
            or digest(record["state"]) != record["state_sha256"]
        ):
            raise ValueError("worker fork-state source/hash differs")
        records.append(record)
        if record["frontier_complete"]:
            complete.append(index)
    merged = {
        key: set()
        for key in (
            "sample_ids",
            "coco_ids",
            "media_hashes",
            "source_group_ids",
            "scopes",
            "scoped_identities",
        )
    }
    for record in records:
        observed = [record["state"], *(item["identities"] for item in record["recovered_metadata"])]
        for state in observed:
            for key in merged:
                merged[key].update(
                    tuple(item) if isinstance(item, list) else item for item in state[key]
                )
    write_once(
        root / "combined_identity_inventory.json",
        dict(
            identities={key: sorted(values) for key, values in merged.items()},
            source_sha=plan["source_sha"],
            worker_bindings=bound,
            complete_historical_id_inventory=False,
            cross_dataset_media_certified=False,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
        ),
    )
    report = dict(
        status="bounded_frontier_segment_complete_freshness_blocked",
        source_sha=plan["source_sha"],
        complete_shards=complete,
        missing_shards=missing,
        frontier_exhausted=not missing and len(complete) == SHARDS,
        pending_frontier_items=sum(len(r["state"]["frontier"]) for r in records),
        newly_processed_files=sum(r["files_added"] for r in records),
        newly_projected_records=sum(r["records_added"] for r in records),
        original_and_new_coverage_errors=sum(len(r["state"]["errors"]) for r in records),
        recovered_oversize_records=sum(len(r["recovered_metadata"]) for r in records),
        pending_oversize_records=sum(len(r["recovery_queue_pending"]) for r in records),
        failed_oversize_records=sum(
            sum(v["status"] == "blocked_not_waived" for v in r["recovery_observations"])
            for r in records
        ),
        worker_bindings=bound,
        source_ownership_and_media_alias_certificate=False,
        identity_inventory_sha256=sha(root / "combined_identity_inventory.json"),
        error_waivers=0,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "GATE_freshness_frontier_fork.json", report)
    return report


def launch(root, *, execute=False):
    root = Path(root).resolve()
    load(root)
    commands = []
    for mode, array in (("worker", True), ("analyze", False)):
        shell = (
            f"export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv && source "
            f"{shlex.quote(str(SOURCE / 'mosaickv/scripts/cache_env.sh'))} && export "
            f"PYTHONPATH={SOURCE}/mosaickv/src:{SOURCE}/mosaickv/scripts "
            "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && exec "
            "/scratch/djy8hg/env/mosaickv/bin/python "
            f"{shlex.quote(str(Path(__file__)))} {mode} --root {shlex.quote(str(root))}"
        )
        if array:
            shell += ' --index "$SLURM_ARRAY_TASK_ID"'
        args = [
            "sbatch",
            "--parsable",
            f"--job-name=fox11scan-{mode}",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--nodes=1",
            f"--cpus-per-task={1 if array else 4}",
            "--mem=16G",
            f"--time={'00:05:00' if array else '00:10:00'}",
            f"--output={root}/slurm/{mode}-%A_%a.out",
            f"--error={root}/slurm/{mode}-%A_%a.err",
        ]
        if array:
            args.append("--array=0-7%8")
        commands.append([*args, "--wrap", shlex.join(["bash", "-c", shell])])
    if not execute:
        return dict(status="DRY_RUN_ONLY", commands=commands, jobs_submitted=0)
    (root / "slurm").mkdir(exist_ok=True)
    write_once(root / "submission-request.json", dict(commands=commands, retry_authorized=False))

    def submit(command, label):
        write_once(root / f"{label}-request.json", dict(command=command, retry_authorized=False))
        try:
            output = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
            job = output.stdout.strip().split(";")[0]
            if not job.isdigit():
                raise ValueError("scheduler response ambiguous")
            write_once(root / f"{label}-submission.json", dict(job_id=job))
            return dict(job=job, failure=None, uncertain=False)
        except Exception as exc:
            stderr = getattr(exc, "stderr", "") or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            rejected = (
                isinstance(exc, subprocess.CalledProcessError)
                and "Batch job submission failed:" in stderr
            )
            failure = dict(
                error=f"{type(exc).__name__}: {exc}", stderr=stderr, retry_authorized=False
            )
            write_once(root / f"{label}-submission-error.json", failure)
            return dict(job=None, failure=failure, uncertain=not rejected)

    first = submit(commands[0], "workers")
    second = None
    if first["job"]:
        command = [*commands[1][:-2], f"--dependency=afterany:{first['job']}", *commands[1][-2:]]
        second = submit(command, "analysis")
    result = dict(
        status="submitted_one_bounded_CPU_segment"
        if second and second["job"]
        else "blocked_submission",
        workers=first["job"],
        analysis=second["job"] if second else None,
        submission_error=first["failure"] or (second["failure"] if second else None),
        workers_submission_uncertain=first["uncertain"],
        analysis_submission_uncertain=second["uncertain"] if second else False,
        analysis_definitely_not_submitted=second is None
        or bool(second["failure"] and not second["uncertain"]),
        automatic_retry=False,
    )
    write_once(root / "submission-result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze", "launch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--coverage-root", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(
            args.root,
            args.parent,
            cpu_gate_root=args.cpu_gate_root,
            coverage_root=args.coverage_root,
        )
    elif args.mode == "worker":
        result = worker(args.root, args.index)
    elif args.mode == "analyze":
        result = analyze(args.root)
    else:
        result = launch(args.root, execute=args.execute)
    print(json.dumps(result))
