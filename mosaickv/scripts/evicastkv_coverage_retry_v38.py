#!/usr/bin/env python3
"""One finite, resource-feasible retry of V36 metadata proofs, not a waiver.

The original preparation tried to place more than 25 GB of two-pass Parquet
reads into eight 2-GiB workers and correctly refused submission. This additive
registrar preserves that bound, prioritizes small lexical metadata, and retains
every deferred candidate as an explicit unresolved registration. Workers and
scientific proof checks are the unchanged V36 implementations.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import evicastkv_coverage_resolution_v36 as original

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
GATE = "GATE_coverage_retry_v38.json"
FAILED_RECEIPT = Path(
    "/scratch/djy8hg/workdir/fox-review-v36/run-20261111-1/history/review/coverage-infrastructure-error.json"
)
FAILED_SHA = "847370fb3b385539b8ed06d79c9dc5a9ee712e943aba935063486e55fb416ad7"
PRIORITY = {"single_json_document": 0, "yaml_events": 1, "closed_parquet": 2}
DEFERRED_REASON = "worker_two_pass_byte_budget_deferred"


def pack_wave(rows):
    """Never drop a row or exceed the existing per-worker two-pass allowance."""
    if not isinstance(rows, list) or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("distinct explicitly registered candidates required")
    costs, unconstrained = [0] * original.SHARDS, [0] * original.SHARDS
    queued, deferred = [], []
    for row in rows:
        if row.get("kind") not in PRIORITY or not isinstance(row.get("signature"), list):
            raise ValueError("registered kind and exact file signature required")
        if (
            len(row["signature"]) != 4
            or type(row["signature"][2]) is not int
            or row["signature"][2] < 0
        ):
            raise ValueError("strict nonnegative file size required")
    for row in sorted(rows, key=lambda value: (-value["signature"][2], value["path"])):
        index = min(range(original.SHARDS), key=lambda i: (unconstrained[i], i))
        unconstrained[index] += row["signature"][2]
    for row in sorted(
        rows, key=lambda value: (PRIORITY[value["kind"]], -value["signature"][2], value["path"])
    ):
        index = min(range(original.SHARDS), key=lambda i: (costs[i], i))
        amount = row["signature"][2]
        if 2 * (costs[index] + amount) > original.MAX_WORKER_BYTES:
            deferred.append(
                dict(
                    row=row,
                    reason=DEFERRED_REASON,
                    resource_deferred=True,
                    original_error_unresolved=True,
                    proof_not_available=True,
                )
            )
        else:
            costs[index] += amount
            queued.append(dict(row, index=index))
    if len(queued) + len(deferred) != len(rows):
        raise AssertionError("resource packing lost a registered candidate")
    return dict(
        queue=queued,
        registration_blocked=deferred,
        worker_source_bytes=costs,
        original_unconstrained_two_pass_worker_bytes=[2 * value for value in unconstrained],
        maximum_worker_two_pass_bytes=original.MAX_WORKER_BYTES,
        resource_bounds_raised=False,
    )


def prepare(root, *, cpu_gate_root=None, global_inventory=None, global_inventory_sha256=None):
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(original.OUTPUT_BASE.resolve()):
        raise ValueError("new immutable scratch retry root required")
    revision = clean_source(SOURCE)
    cpu_root = Path(cpu_gate_root or root.parent).resolve()
    original.cpu_gate(cpu_root, revision)
    saved, errors = original.baseline()
    failed = original.read(FAILED_RECEIPT, FAILED_SHA)
    if (
        failed.get("phase") != "prepare"
        or failed.get("error_type") != "ValueError"
        or failed.get("complete") is not False
        or failed.get("error_waivers") != 0
    ):
        raise ValueError("the immutable actual V36 preparation failure is required")
    saved["bindings"][str(FAILED_RECEIPT.resolve(strict=True))] = FAILED_SHA
    covered = set(saved["covered_error_hashes"])
    categories = Counter(error["error"] for key, error in errors.items() if key not in covered)
    if any(categories[name] != expected for name, expected in original.EXPECTED_CATEGORIES.items()):
        raise ValueError("registered original net categories changed")
    roots = [str(Path(path).resolve(strict=True)) for path in original.scanner.ROOTS]
    groups, registrations, blocked = original.nominate(errors, covered), [], []
    deadline = time.monotonic() + original.SECONDS
    for (name, kind), hashes in sorted(groups.items()):
        row = dict(path=name, kind=kind, original_error_sha256s=sorted(hashes))
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("finite registration deadline")
            path = original.canonical(name, roots)
            row.update(canonical_path=str(path), signature=original.scanner.signature(path))
            if kind == "closed_parquet":
                if (
                    path.name not in original.derived.NAMES
                    or path.parent.parent.name != "mechanism-terms"
                ):
                    raise ValueError("no unchanged closed-writer proof for this Parquet layout")
                if row["signature"][2] > original.derived.MAX_ARTIFACT_BYTES:
                    raise ValueError("unchanged per-artifact proof bound exceeded")
                row["current_file_sha256"] = None  # Actual producer map checked by worker.
            else:
                bound = (
                    original.yaml_events.MAX_BYTES if kind == "yaml_events" else original.SMALL_JSON
                )
                _raw, actual = original.read_bytes(path, bound=bound)
                if original.scanner.signature(path) != row["signature"]:
                    raise ValueError("metadata changed across registration")
                row["current_file_sha256"] = actual
            registrations.append(row)
        except (OSError, ValueError, TimeoutError) as error:
            blocked.append(dict(row=row, reason=type(error).__name__, proof_not_available=True))
    packed = pack_wave(registrations)
    for module in (
        Path(__file__),
        Path(original.__file__),
        Path(original.yaml_events.__file__),
        Path(original.derived.__file__),
        Path(original.scanner.__file__),
        Path(original.imported.__file__),
    ):
        saved["bindings"][str(module.resolve())] = sha(module)
    if (global_inventory is None) != (global_inventory_sha256 is None):
        raise ValueError("global inventory path/hash required together")
    if global_inventory is not None:
        path = Path(global_inventory).resolve(strict=True)
        if not path.is_relative_to(original.OUTPUT_BASE.resolve()):
            raise ValueError("global inventory escapes project scratch")
        original.read_bytes(path, bound=128 * 2**20, expected=global_inventory_sha256)
        saved["bindings"][str(path)] = global_inventory_sha256
    plan = dict(
        **saved,
        source_sha=revision,
        cpu_gate_root=str(cpu_root),
        roots=roots,
        queue=packed["queue"],
        registration_blocked=blocked + packed["registration_blocked"],
        nominated_files=len(groups),
        expected_category_errors=original.EXPECTED_CATEGORIES,
        shard_count=original.SHARDS,
        seconds_per_worker=original.SECONDS,
        maximum_bytes_per_worker=original.MAX_WORKER_BYTES,
        worker_source_bytes=packed["worker_source_bytes"],
        resource_deferred_files=len(packed["registration_blocked"]),
        original_unconstrained_two_pass_worker_bytes=packed[
            "original_unconstrained_two_pass_worker_bytes"
        ],
        original_preparation_failure=dict(
            path=str(FAILED_RECEIPT),
            sha256=FAILED_SHA,
            cause="registered two-pass source bytes exceed per-worker bound",
        ),
        worker_and_scientific_proof_implementations_unchanged=True,
        resource_bounds_raised=False,
        original_errors_preserved=True,
        error_waivers=0,
        **original.false_scope(),
    )
    if len(plan["queue"]) + len(plan["registration_blocked"]) != len(groups):
        raise ValueError("original nominated denominator changed")
    write_once(root / "plan.json", plan)
    return dict(
        workers=original.SHARDS,
        registered_files=len(plan["queue"]),
        blocked_files=len(plan["registration_blocked"]),
        resource_deferred_files=plan["resource_deferred_files"],
        plan_sha256=sha(root / "plan.json"),
        resource_bounds_raised=False,
        **original.false_scope(),
    )


def worker(root, index):
    return original.worker(root, index)


def analyze(root):
    root = Path(root).resolve()
    gate = original.analyze(root)
    plan = original.read(root / "plan.json")
    report = dict(
        gate,
        original_proof_gate_filename=original.GATE,
        original_proof_gate_sha256=sha(root / original.GATE),
        resource_deferred_files=plan["resource_deferred_files"],
        resource_deferred_error_hashes=sum(
            len(item["row"]["original_error_sha256s"])
            for item in plan["registration_blocked"]
            if item.get("resource_deferred")
        ),
        original_preparation_failure=plan["original_preparation_failure"],
        resource_bounds_raised=False,
        automatic_resubmission=False,
    )
    write_once(root / GATE, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--global-inventory", type=Path)
    parser.add_argument("--global-inventory-sha256")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    result = (
        prepare(
            args.root,
            cpu_gate_root=args.cpu_gate_root,
            global_inventory=args.global_inventory,
            global_inventory_sha256=args.global_inventory_sha256,
        )
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
