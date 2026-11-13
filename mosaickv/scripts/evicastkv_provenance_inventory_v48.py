#!/usr/bin/env python3
"""Inventory only the V47 residual provenance gaps; never reopen source bodies."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import evicastkv_audit_resume_v47 as resume

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.parquet_binding_inventory_v48 import build_inventory

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v47/run-20261112-a05e557-1")
PRODUCER = "a05e557438bc9c47d79c1c1a410baa304b495fcf"
BOUND = {
    "analysis/decision.json": "62b0fd63a41af262c2d9a0c61bcfced3d3bdd09696b56ab9b93ff5434f14ff0e",
    "analysis/reconciled_history.json": (
        "dafb1f61ff07fd938c0a3deb264fa26bb351b44333da758a85fbac4073e7a74b"
    ),
    "registration.json": "465390b5be2a74ebe78d21aba8bd12020f7834b7a424048b90b58507deea1338",
    "bulk.json": "68bf0b7325d9c42aba61205f2efeddc94328296df1e3fa879b2b30a4e9cb0656",
}
SECONDS = resume.reader.SECONDS
BYTE_BOUND = resume.reader.BYTE_BOUND


def _elapsed(started):
    elapsed = time.monotonic() - started
    if elapsed > SECONDS:
        raise TimeoutError("unchanged 210-second metadata-inventory deadline exceeded")
    return elapsed


def _read_parents():
    started = time.monotonic()
    total = sum((PARENT / name).stat().st_size for name in BOUND)
    if total > BYTE_BOUND:
        raise ValueError("unchanged 2-GiB metadata input bound exceeded")
    values = {}
    for name, expected in BOUND.items():
        _elapsed(started)
        values[name] = resume.reader.common.history.read(PARENT / name, expected)
    _elapsed(started)
    return values, total, started


def check_parent(values):
    decision = values["analysis/decision.json"]
    parent = values["analysis/reconciled_history.json"]
    plan = values["registration.json"]
    bulk = values["bulk.json"]
    if (
        decision["source_sha"] != PRODUCER
        or decision["producer_source_sha"] != PRODUCER
        or parent["source_sha"] != PRODUCER
        or plan["source_sha"] != PRODUCER
        or bulk["status"] != "registered_work_complete"
        or decision["newly_resolved_JSON_occurrences"] != 1168
        or len(set(decision["passed_indices"])) != 1168
        or set(decision["passed_indices"]) != set(plan["retry_indices"])
        or decision["reconciled_history_sha256"] != BOUND["analysis/reconciled_history.json"]
        or digest(plan["rows"]) != plan["rows_sha256"]
        or decision["error_counts"] != parent["counts"]
        or parent["counts"]["net_unresolved_error_count"] != 135
        or parent["counts"]["raw_error_count"] != 40760
        or any(
            decision.get(name) is not False
            for name in (
                "fresh_confirmation_authorized",
                "materialization_authorized",
                "historical_media_exposure_resolved",
            )
        )
        or decision["GPU_jobs"] != 0
    ):
        raise ValueError("complete immutable V47 metadata-only outcome required")
    return parent, plan


def inventory(root):
    revision = clean_source(SOURCE)
    resume.reader.common.cpu_gate(root / "cpu", revision)
    values, byte_count, started = _read_parents()
    parent, plan = check_parent(values)
    result = build_inventory(parent, plan["rows"], auditor_source_sha=revision)
    if result["counts"]["parquet_occurrences"] != 134:
        raise ValueError("exactly the 134 still-unresolved Parquet occurrences required")
    result.update(
        schema="evicastkv_remaining_provenance_inventory_v48",
        source_sha=revision,
        parent_auditor_source_sha=PRODUCER,
        bindings={str(PARENT / name): expected for name, expected in BOUND.items()},
        parent_checkpoint_unchanged=True,
        completed_JSON_transactions_replayed=0,
        historical_source_bodies_opened=0,
        parquet_rows_decoded=0,
        credential_bodies_read=False,
        metadata_input_bytes=byte_count,
        limits=dict(
            worker_seconds=SECONDS,
            input_bytes=BYTE_BOUND,
            live_metadata_bytes=8 * 2**20,
            live_metadata_nodes=100000,
            checkpoint_bytes=resume.reader.common.history.MAX_STATE,
            aggregate_artifact_bytes=resume.MAX_ARTIFACT_BYTES,
            CPU_allocation_bytes=32 * 2**30,
        ),
    )
    del values, parent, plan
    gc.collect()
    result["wall_seconds"] = _elapsed(started)
    if len(json.dumps(result).encode()) > 8 * 2**20:
        raise ValueError("unchanged live projected-metadata bound exceeded")
    resume.require_artifact_capacity(root, 8 * 2**20)
    write_once(root / "inventory.json", result)
    return dict(
        inventory_sha256=sha(root / "inventory.json"),
        counts=result["counts"],
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
    )


def analyze(root):
    revision = clean_source(SOURCE)
    resume.reader.common.cpu_gate(root / "cpu", revision)
    path = root / "inventory.json"
    if not path.is_file():
        return dict(
            status="inventory_failed_or_incomplete",
            source_sha=revision,
            GPU_jobs=0,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
        )
    result = resume.reader.common.history.read(path)
    if result["source_sha"] != revision or result["parent_auditor_source_sha"] != PRODUCER:
        raise ValueError("exact inventory source/parent binding required")
    return dict(
        status="provenance_gaps_inventoried_not_resolved",
        source_sha=revision,
        inventory_path=str(path),
        inventory_sha256=sha(path),
        counts=result["counts"],
        original_JSON_proofs_replayed=0,
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        historical_media_exposure_resolved=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "inventory", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only write-once audit root required")
    try:
        if args.mode == "cpu":
            result = resume.cpu(root)
        elif args.mode == "analyze":
            result = analyze(root)
            write_once(root / "decision.json", result)
        else:
            result = inventory(root)
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error="bounded CPU provenance inventory failed closed",
                GPU_jobs=0,
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
