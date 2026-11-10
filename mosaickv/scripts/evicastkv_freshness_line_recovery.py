#!/usr/bin/env python3
"""Bounded recovery of registered oversized JSONL metadata, not a freshness pass.

No directory discovery, answers, model loading, identity selection, or error
waivers. The CLI is dry by default. Real execution requires current-source CPU
suite evidence and an explicit immutable inventory hash from its controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

from evicastkv_freshness_gap_audit import MetadataReader
from evicastkv_review_followups import cpu_gate
from evicastkv_review_freshness_stream import (
    ROOTS,
    ProjectionPaused,
    collect,
    identity_projection,
    signature,
)

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = Path(
    "/scratch/djy8hg/audits/fox-review-v8-consumed-diagnostics-20261110/freshness-continuation"
)
MAX_BYTES = 128 * 2**20
SECONDS_PER_RECORD = 180
TOTAL_SECONDS = 600
PREFIX = "JSONL record exceeds identity-projection line bound"


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise ProjectionPaused("bounded oversized-line recovery deadline reached")


def select_errors(inventory, maximum):
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 4:
        raise ValueError("one to four registered records are permitted")
    selected = {}
    for row in inventory["errors"]:
        if row.get("error") != PREFIX:
            continue
        offset = row.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("oversized record lacks a valid registered byte offset")
        path = row.get("path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("oversized record lacks an absolute registered path")
        selected[path, offset] = dict(path=path, offset=offset, original_error_sha256=digest(row))
    # Freeze locations before any attempt; never keep searching until a success.
    return [selected[key] for key in sorted(selected)[:maximum]]


def canonical(path, roots):
    actual = Path(path).resolve(strict=True)
    if not any(actual.is_relative_to(root) for root in roots):
        raise ValueError("canonical source escapes registered historical run roots")
    if actual.suffix not in {".jsonl", ".ndjson"}:
        raise ValueError("registered record is not a JSONL source")
    return actual


def _descriptor(path, expected):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        actual = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("registered source is not a regular file")
        if actual != expected:
            raise ValueError("registered source signature changed")
        if info.st_size > MAX_BYTES:
            raise ValueError("unsupported source larger than 128 MiB; not silently discarded")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def hash_source(path, binding, deadline):
    """Verify original complete segment hashes and a fresh whole-file digest."""
    expected = binding["signature"]
    segments = binding.get("segments", [])
    if digest(segments) != binding.get("segmented_content_root"):
        raise ValueError("registered segment manifest hash changed")
    cursor = 0
    for item in segments:
        if item["start"] != cursor or item["end"] <= cursor:
            raise ValueError("saved content segments do not completely and uniquely cover source")
        cursor = item["end"]
    if cursor != expected[2]:
        raise ValueError("saved content segments lack complete source coverage")
    whole = hashlib.sha256()
    with _descriptor(path, expected) as stream:
        for item in segments:
            segment = hashlib.sha256()
            remaining = item["end"] - item["start"]
            while remaining:
                check_deadline(deadline)
                chunk = stream.read(min(2**20, remaining))
                if not chunk:
                    raise ValueError("source truncated during registered-segment verification")
                whole.update(chunk)
                segment.update(chunk)
                remaining -= len(chunk)
            if segment.hexdigest() != item["sha256"]:
                raise ValueError("source differs from original recorded segment content")
        if stream.read(1):
            raise ValueError("source grew beyond registered segment content")
    if signature(path) != expected:
        raise ValueError("source signature changed during content verification")
    return whole.hexdigest()


def project_record(row, binding, roots, deadline):
    path = canonical(row["path"], roots)
    before = hash_source(path, binding, deadline)
    check_deadline(deadline)
    offset = row["offset"]
    with _descriptor(path, binding["signature"]) as stream:
        if offset >= binding["signature"][2]:
            raise ValueError("registered offset is outside source")
        if offset:
            stream.seek(offset - 1)
            if stream.read(1) != b"\n":
                raise ValueError("registered offset is not a JSONL record boundary")
        stream.seek(offset)
        raw = stream.readline(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("unsupported record larger than 128 MiB; not discarded")
    if not raw.strip():
        raise ValueError("registered oversized record is empty")
    statistics = {}
    projected = identity_projection(
        raw, budget=SimpleNamespace(deadline=deadline), statistics=statistics
    )
    state = {
        key: []
        for key in (
            "sample_ids",
            "coco_ids",
            "media_hashes",
            "source_group_ids",
            "scopes",
            "scoped_identities",
        )
    }
    state["projected_records"] = 0
    collect(projected, state, path)
    check_deadline(deadline)
    after = hash_source(path, binding, deadline)
    if after != before or canonical(row["path"], roots) != path:
        raise ValueError("registered source changed during metadata recovery")
    return dict(
        status="projected_registered_metadata_only",
        canonical_path=str(path),
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
        line_sha256=hashlib.sha256(raw).hexdigest(),
        line_bytes=len(raw),
        projected_metadata_sha256=digest(projected),
        statistics=statistics,
        counts={key: len(value) for key, value in state.items() if isinstance(value, list)},
        projected_records=state["projected_records"],
        raw_or_answer_values_exported=False,
    )


def preparation(parent, maximum, expected=None):
    parent = Path(parent)
    path = parent / "aggregate/inventory.json" if parent.is_dir() else parent
    reader = MetadataReader()
    inventory = reader.read(path, expected)
    selected = select_errors(inventory, maximum)
    roots = [Path(value).resolve(strict=True) for value in inventory["roots"]]
    if set(roots) != {Path(value).resolve(strict=True) for value in ROOTS}:
        raise ValueError("inventory historical roots differ from the registered scanner scope")
    bindings = {}
    requested = {item["path"] for item in selected}
    for checkpoint, expected_hash in sorted(inventory["shard_bindings"].items()):
        saved = reader.read(checkpoint, expected_hash)
        state = saved["state"]
        if digest(state) != saved["state_sha256"]:
            raise ValueError("saved checkpoint state hash changed")
        for name in requested & set(state["bindings"]):
            value = state["bindings"][name]
            if name in bindings and bindings[name] != value:
                raise ValueError("conflicting recorded source bindings")
            bindings[name] = value
    for row in selected:
        row["source_binding"] = bindings.get(row["path"])
    return dict(
        inventory_path=str(path),
        inventory_sha256=sha(path),
        inventory_source_sha=inventory["source_sha"],
        selected_records=selected,
        original_errors_sha256=digest(inventory["errors"]),
        original_error_count=len(inventory["errors"]),
        metadata_bindings=reader.bindings,
        canonical_roots=[str(value) for value in roots],
        source_traversal_performed=False,
        bounds=dict(
            maximum_records=maximum,
            bytes_per_file=MAX_BYTES,
            seconds_per_record=SECONDS_PER_RECORD,
            total_seconds=TOTAL_SECONDS,
        ),
    )


def run(root, parent=DEFAULT_PARENT, max_records=4, *, cpu_gate_root=None, inventory_sha256=None):
    if cpu_gate_root is None or inventory_sha256 is None:
        raise ValueError("explicit CPU-suite and immutable inventory hash bindings required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    root = Path(root)
    if root.exists():
        raise FileExistsError("a new immutable derived recovery root is required")
    started = time.monotonic()
    plan = preparation(parent, max_records, inventory_sha256)
    plan.update(source_sha=revision, cpu_gate=cpu, mode="bounded_CPU_metadata_canary")
    root.mkdir(parents=True)
    write_once(root / "plan.json", plan)
    results = []
    roots = [Path(value) for value in plan["canonical_roots"]]
    for row in plan["selected_records"]:
        item = {key: value for key, value in row.items() if key != "source_binding"}
        record_started = time.monotonic()
        deadline = min(started + TOTAL_SECONDS, record_started + SECONDS_PER_RECORD)
        try:
            check_deadline(deadline)
            if row["source_binding"] is None:
                raise ValueError("selected record lacks a saved complete source binding")
            item.update(project_record(row, row["source_binding"], roots, deadline))
        except (OSError, ValueError, TypeError, KeyError, RecursionError, TimeoutError) as error:
            # Keep this exact slot and its failure; never replace it with an easier file.
            item.update(
                status="blocked_not_waived", error_type=type(error).__name__, error=str(error)
            )
        item["wall_seconds"] = time.monotonic() - record_started
        results.append(item)
        write_once(root / f"record-{len(results) - 1:02d}.json", item)
    gate = dict(
        status="completed_bounded_metadata_canary"
        if results
        and all(item["status"] == "projected_registered_metadata_only" for item in results)
        else "blocked_bounded_metadata_canary",
        selected=len(results),
        projected=sum(item["status"] == "projected_registered_metadata_only" for item in results),
        results=results,
        plan_sha256=sha(root / "plan.json"),
        source_sha=revision,
        original_errors_preserved=True,
        original_error_count=plan["original_error_count"],
        original_errors_sha256=plan["original_errors_sha256"],
        complete_scan=False,
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        cohort_selected=False,
        models_loaded=False,
        new_GPU_inference=0,
        error_waivers=0,
    )
    write_once(root / "GATE_freshness_line_recovery.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--max-records", type=int, default=4)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--inventory-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = (
        run(
            args.root,
            args.parent,
            args.max_records,
            cpu_gate_root=args.cpu_gate_root,
            inventory_sha256=args.inventory_sha256,
        )
        if args.execute
        else {
            "mode": "dry_no_raw_source_access",
            "execute": False,
            "plan": preparation(args.parent, args.max_records),
            "fresh_confirmation_authorized": False,
        }
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
