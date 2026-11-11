#!/usr/bin/env python3
"""Recover only the completed V13 segment-nine CPU aggregation.

The old analyzer's 128-MiB input limit rejected a completed worker artifact.
This separate scope accepts at most 256 MiB per immutable metadata file and
2 GiB total, after verifying all eight clean-producer stdout/hash receipts.
No traversal, source mutation, worker execution or cohort selection is done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_review_freshness_stream as scanner
from evicastkv_freshness_gap_audit import category
from evicastkv_freshness_producer_canary import read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v13/run-20261111-1/history")
OLD_SHA = "cc2768b2ba93e13682bc0d58e0e49a7856efd4c1"
MAX_FILE, MAX_TOTAL, SECONDS = 256 * 2**20, 2 * 2**30, 600
MAX_OUTPUT = 64 * 2**20
IDENTITIES = (
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
)


class Reader:
    def __init__(self):
        self.deadline = time.monotonic() + SECONDS
        self.bytes = 0
        self.bindings = {}

    def raw(self, name, *, expected=None, bound=2**20):
        path = Path(name).resolve(strict=True)
        if not path.is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("recovery metadata must remain within scratch")
        before = scanner.signature(path)
        if before[2] > bound or self.bytes + before[2] > MAX_TOTAL:
            raise ValueError("registered metadata byte limit exceeded")
        value = bytearray()
        with path.open("rb") as stream:
            while True:
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("finite metadata recovery deadline")
                chunk = stream.read(2**20)
                if not chunk:
                    break
                value.extend(chunk)
        actual = hashlib.sha256(value).hexdigest()
        if (
            scanner.signature(path) != before
            or len(value) != before[2]
            or (expected is not None and actual != expected)
        ):
            raise ValueError("immutable metadata source/signature/hash changed")
        self.bytes += len(value)
        self.bindings[str(path)] = actual
        return bytes(value)

    def read(self, name, **kwargs):
        return read_json(self.raw(name, **kwargs))


def prepare(root, *, cpu_gate_root, parent=PARENT):
    root, parent = Path(root).resolve(), Path(parent).resolve(strict=True)
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch metadata-recovery root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    reader = Reader()
    registration = reader.read(parent / "registration.json")
    failure_path = parent / "GATE_history_v13.json"
    failure = reader.read(failure_path)
    segment = parent / "segment-9"
    segment_plan_path = segment / "plan.json"
    source_plan = reader.read(segment_plan_path)
    if (
        registration["source_sha"] != OLD_SHA
        or failure["source_sha"] != OLD_SHA
        or failure["status"] != "blocked_history_continuation"
        or failure["stop_reason"] != "advance_failed"
        or failure["error"] != "ValueError: bound producer artifact exceeds registered byte limit"
        or Path(failure["last_frontier"]).resolve() != segment
        or source_plan["source_sha"] != OLD_SHA
        or source_plan["same_scanner_file_sha256"] != sha(Path(scanner.__file__))
    ):
        raise ValueError("only the exact known immutable segment-nine failure may be recovered")
    if failure["bindings"].get(str(segment_plan_path)) != sha(segment_plan_path):
        raise ValueError("failed gate does not bind the original segment plan")
    submitted = reader.read(parent / "jobs/segment-9-submission.json")
    array = submitted["job"]
    if not isinstance(array, str) or re.fullmatch(r"[1-9][0-9]*", array) is None:
        raise ValueError("exact completed CPU array identity required")
    workers = []
    for index in range(8):
        stdout = parent / "slurm" / f"segment-9-{array}_{index}.out"
        lines = reader.raw(stdout).splitlines()
        if len(lines) != 2 or lines[0] != f"clean_git_sha={OLD_SHA}".encode():
            raise ValueError("exact clean producer receipt required for every shard")
        fields = {"index", "frontier_complete", "files_added", "worker_sha256"}
        with patch.object(scanner, "safe_key", lambda key, fields=fields: key in fields):
            receipt = scanner.identity_projection(
                lines[1], budget=SimpleNamespace(deadline=reader.deadline)
            )
        if (
            set(receipt) != fields
            or type(receipt["index"]) is not int
            or type(receipt["frontier_complete"]) is not bool
            or type(receipt["files_added"]) is not int
            or receipt["index"] != index
            or not re.fullmatch(r"[a-f0-9]{64}", receipt["worker_sha256"])
        ):
            raise ValueError("unexpected worker receipt scope")
        path = segment / "workers" / f"shard-{index}.json"
        # Authenticate the exact body BEFORE accepting the larger JSON scope.
        reader.raw(path, expected=receipt["worker_sha256"], bound=MAX_FILE)
        workers.append(
            dict(
                index=index,
                path=str(path),
                sha256=receipt["worker_sha256"],
                stdout=str(stdout),
                stdout_sha256=sha(stdout),
                receipt=receipt,
                signature=scanner.signature(path),
            )
        )
    original_inventory = Path(source_plan["parent"]) / "combined_identity_inventory.json"
    expected = source_plan["parent_bindings"].get(str(original_inventory))
    if expected is None:
        raise ValueError("original parent identity inventory must remain hash-bound")
    reader.raw(original_inventory, expected=expected, bound=MAX_FILE)
    plan = dict(
        source_sha=revision,
        historical_worker_source_sha=OLD_SHA,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        parent=str(parent),
        workers=workers,
        parent_identity_inventory=str(original_inventory),
        original_failure_gate=str(failure_path),
        bindings=reader.bindings,
        maximum_metadata_file_bytes=MAX_FILE,
        maximum_total_metadata_bytes=MAX_TOTAL,
        maximum_inventory_and_errors_bytes_each=MAX_OUTPUT,
        seconds=SECONDS,
        old_workers_copied=False,
        old_failure_preserved=True,
        new_scan=False,
        new_GPU_inference=0,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_exact_segment_nine_metadata_recovery",
        workers=8,
        maximum_metadata_file_bytes=MAX_FILE,
        plan_sha256=sha(root / "plan.json"),
    )


def run(root, *, cpu_gate_root=None):
    root = Path(root).resolve()
    if not root.exists():
        if cpu_gate_root is None:
            raise ValueError("CPU gate required before metadata recovery")
        prepare(root, cpu_gate_root=cpu_gate_root)
    plan = Reader().read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    if (
        cpu_gate_root is not None
        and Path(cpu_gate_root).resolve() != Path(plan["cpu_gate_root"]).resolve()
    ):
        raise ValueError("registered CPU gate differs")
    if (
        plan["maximum_metadata_file_bytes"] != MAX_FILE
        or plan["maximum_total_metadata_bytes"] != MAX_TOTAL
        or plan["historical_worker_source_sha"] != OLD_SHA
        or plan["maximum_inventory_and_errors_bytes_each"] != MAX_OUTPUT
        or plan["seconds"] != SECONDS
    ):
        raise ValueError("registered bounded historical recovery scope changed")
    reader = Reader()
    parent_inventory = reader.read(
        plan["parent_identity_inventory"],
        expected=plan["bindings"][plan["parent_identity_inventory"]],
        bound=MAX_FILE,
    )
    merged = {
        key: {tuple(v) if isinstance(v, list) else v for v in parent_inventory["identities"][key]}
        for key in IDENTITIES
    }
    errors, summary, frontier, workers = [], [], [], {}
    state_paths = {row["path"] for row in plan["workers"]}
    for name, expected in plan["bindings"].items():
        if name not in state_paths and name != plan["parent_identity_inventory"]:
            reader.raw(name, expected=expected)
    if sorted(row["index"] for row in plan["workers"]) != list(range(8)):
        raise ValueError("every completed worker required exactly once")
    for row in plan["workers"]:
        path = Path(row["path"])
        value = reader.read(path, expected=row["sha256"], bound=MAX_FILE)
        if (
            value["source_sha"] != OLD_SHA
            or value["index"] != row["index"]
            or scanner.signature(path) != row["signature"]
            or value["state_sha256"] != digest(value["state"])
            or value["files_added"] != row["receipt"]["files_added"]
            or value["state"]["traversal_complete"] != row["receipt"]["frontier_complete"]
        ):
            raise ValueError("completed worker differs from its immutable clean-source receipt")
        state = value["state"]
        for key in IDENTITIES:
            merged[key].update(tuple(v) if isinstance(v, list) else v for v in state[key])
        errors.extend(state["errors"])
        frontier.extend(state["frontier"])
        summary.append(
            dict(
                index=row["index"],
                processed_files=state["processed_files"],
                files_added=value["files_added"],
                records_added=value["records_added"],
                frontier_exhausted=state["traversal_complete"],
                pending_frontier_items=len(state["frontier"]),
            )
        )
        workers[str(path)] = row["sha256"]
    inventory = dict(
        source_sha=plan["source_sha"],
        historical_worker_source_sha=OLD_SHA,
        identities={key: sorted(values) for key, values in merged.items()},
        worker_bindings=workers,
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        fresh_confirmation_authorized=False,
    )
    output_sizes = {}
    for name, value in (
        ("combined_identity_inventory.json", inventory),
        ("original_errors.json", errors),
    ):
        size = len(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()) + 1
        if size > MAX_OUTPUT:
            raise ValueError("derived metadata export exceeds its registered 64-MiB bound")
        output_sizes[name] = size
    write_once(root / "combined_identity_inventory.json", inventory)
    write_once(root / "original_errors.json", errors)
    report = dict(
        status="recovered_completed_segment_nine_metadata_not_freshness_certified",
        source_sha=plan["source_sha"],
        historical_worker_source_sha=OLD_SHA,
        missing_shards=[],
        worker_bindings=workers,
        shard_summary=summary,
        frontier_exhausted=all(row["frontier_exhausted"] for row in summary),
        pending_frontier_items=len(frontier),
        frontier_kinds=dict(Counter(r["kind"] for r in frontier)),
        original_and_new_coverage_errors=len(errors),
        error_categories=dict(Counter(category(r.get("error", "")) for r in errors)),
        newly_processed_files=sum(r["files_added"] for r in summary),
        newly_projected_objects_not_independent_examples=sum(r["records_added"] for r in summary),
        identity_inventory_sha256=sha(root / "combined_identity_inventory.json"),
        preserved_errors_sha256=sha(root / "original_errors.json"),
        output_metadata_bytes=output_sizes,
        old_failure_gate_sha256=plan["bindings"][plan["original_failure_gate"]],
        receipt_bindings={r["stdout"]: r["stdout_sha256"] for r in plan["workers"]},
        original_errors_preserved=True,
        error_waivers=0,
        new_scan=False,
        new_GPU_inference=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "GATE_history_segment_recovery.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    args = parser.parse_args()
    result = (
        prepare(args.root, cpu_gate_root=args.cpu_gate_root)
        if args.mode == "prepare"
        else run(args.root, cpu_gate_root=args.cpu_gate_root)
    )
    print(json.dumps(result))
