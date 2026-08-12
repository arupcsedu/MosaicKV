#!/usr/bin/env python3
"""Build an immutable overlay campaign for only failed PrefixKV shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import yaml
from prepare_llava_full_quality_8gpu import (
    _canonical_bytes,
    _source_provenance,
    _write_once,
)

from mosaickv.config import config_sha256, load_config


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_spec(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("PrefixKV repair spec must be a schema-version-1 mapping")
    return cast("dict[str, Any]", value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _validated_sample_ids(cell: dict[str, Any]) -> tuple[str, ...]:
    path = Path(str(cell["sample_id_file"])).resolve()
    payload = path.read_bytes()
    if _sha(payload) != cell["sample_id_file_sha"]:
        raise RuntimeError(f"sample ledger SHA mismatch: {path}")
    value = json.loads(payload)
    sample_ids = value.get("sample_ids")
    if not isinstance(sample_ids, list) or not all(isinstance(item, str) for item in sample_ids):
        raise RuntimeError(f"invalid sample ledger: {path}")
    result = tuple(cast("list[str]", sample_ids))
    if not result or len(result) != len(set(result)):
        raise RuntimeError(f"empty or duplicate sample ledger: {path}")
    return result


def _classify_source_cell(
    root: Path,
    cell: dict[str, Any],
    expected_error: str,
) -> str:
    run_root = root / str(cell["run_id"])
    raw = run_root / "results.jsonl"
    manifest = run_root / "manifest.json"
    parquet = run_root / "aggregate.parquet"
    if not (raw.is_file() and manifest.is_file() and parquet.is_file()):
        raise RuntimeError(f"source shard has incomplete artifacts: {cell['run_id']}")
    rows = _jsonl(raw)
    sample_ids = _validated_sample_ids(cell)
    if len(rows) != len(sample_ids) or {str(row.get("sample_id")) for row in rows} != set(
        sample_ids
    ):
        raise RuntimeError(f"source shard sample mismatch: {cell['run_id']}")
    statuses = {row.get("status") for row in rows}
    if statuses == {"completed"}:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("source", {}).get("config_sha") != cell["config_sha"]:
            raise RuntimeError(f"source shard config SHA mismatch: {cell['run_id']}")
        if value.get("artifacts", {}).get("raw_output_sha") != _sha(raw.read_bytes()):
            raise RuntimeError(f"source shard raw SHA mismatch: {cell['run_id']}")
        return "completed"
    if statuses == {"failed"} and {row.get("error") for row in rows} == {expected_error}:
        return "failed"
    raise RuntimeError(
        f"source shard has an unexpected mixed or unknown result state: {cell['run_id']}"
    )


def _check_smoke(spec: dict[str, Any]) -> dict[str, Any]:
    smoke = cast("dict[str, Any]", spec["smoke_evidence"])
    rows = _jsonl(Path(str(smoke["result"])).resolve())
    if len(rows) != 1:
        raise RuntimeError("PrefixKV fix smoke must contain exactly one result")
    row = rows[0]
    if row.get("status") != "completed" or row.get("error") is not None:
        raise RuntimeError("PrefixKV fix smoke did not complete successfully")
    if not str(row.get("method", "")).startswith(str(smoke["method"])):
        raise RuntimeError("PrefixKV fix smoke method label mismatch")
    retention_ratio = row.get("retention_ratio")
    generated_tokens = row.get("generated_tokens")
    if not isinstance(retention_ratio, int | float) or float(retention_ratio) != float(
        smoke["retention_ratio"]
    ):
        raise RuntimeError("PrefixKV fix smoke retention mismatch")
    if not isinstance(generated_tokens, int) or generated_tokens != int(smoke["generated_tokens"]):
        raise RuntimeError("PrefixKV fix smoke output length mismatch")
    return row


def _balance(cells: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    totals = [0.0] * workers
    for cell in sorted(
        cells,
        key=lambda item: (-float(item["estimated_seconds"]), str(item["run_id"])),
    ):
        worker = min(range(workers), key=lambda index: (totals[index], index))
        buckets[worker].append(cell)
        totals[worker] += float(cell["estimated_seconds"])
    return buckets


def _symlink_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"overlay source directory is missing: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace overlay path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, destination, target_is_directory=True)


def materialize(spec_path: Path) -> tuple[Path, Path]:
    spec = _load_spec(spec_path)
    destination = Path(str(spec["campaign_destination"])).resolve()
    output_root = Path(str(spec["repair_output_root"])).resolve()
    if destination.exists() or output_root.exists():
        raise FileExistsError("repair campaign destination and output root must both be new")

    source_index = Path(str(spec["source_continuation_index"])).resolve()
    source_payload = source_index.read_bytes()
    if _sha(source_payload) != spec["source_continuation_index_sha256"]:
        raise RuntimeError("source continuation index SHA mismatch")
    source_manifest = json.loads(
        (source_index.parent / "continuation_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("index_sha256") != _sha(source_payload):
        raise RuntimeError("source continuation manifest disagrees with its index")

    repository = Path(__file__).resolve().parents[1]
    behavior_paths = {
        "hf_runtime_sha256": repository / "src/mosaickv/backends/hf_runtime.py",
        "config_sha256": repository / "src/mosaickv/config.py",
    }
    observed_behavior = {name: _sha(path.read_bytes()) for name, path in behavior_paths.items()}
    if observed_behavior != spec["runtime_identity"]:
        raise RuntimeError(
            f"runtime identity changed: expected={spec['runtime_identity']}, "
            f"observed={observed_behavior}"
        )
    smoke_row = _check_smoke(spec)
    git_sha, git_dirty, source_patch = _source_provenance()
    if not git_dirty:
        raise RuntimeError("repair spec is classified as an exploratory dirty-tree campaign")

    source_jobs = _jsonl(source_index)
    source_root = Path(str(spec["source_output_root"])).resolve()
    expected_error = str(spec["expected_error"])
    completed_cells: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failed_cells: list[dict[str, Any]] = []
    for job in source_jobs:
        for source_cell in cast("list[dict[str, Any]]", job["cells"]):
            state = _classify_source_cell(source_root, source_cell, expected_error)
            if state == "completed":
                completed_cells.append((job, source_cell))
                continue
            config = load_config(source_cell["config_path"])
            if config_sha256(config) != source_cell["config_sha"]:
                raise RuntimeError(f"failed shard config SHA mismatch: {source_cell['run_id']}")
            failed_cells.append(dict(source_cell))

    expected_failed = int(spec["expected_failed_shards"])
    if len(failed_cells) != expected_failed:
        raise RuntimeError(f"failed shard count {len(failed_cells)} != {expected_failed}")
    failed_rows = sum(len(_validated_sample_ids(cell)) for cell in failed_cells)
    if failed_rows != int(spec["expected_failed_rows"]):
        raise RuntimeError("failed sample-row count changed")
    logical_ids = {str(cell["logical_run_id"]) for cell in failed_cells}
    if len(logical_ids) != int(spec["expected_failed_logical_cells"]):
        raise RuntimeError("failed logical-cell count changed")
    if {str(cell["method"]) for cell in failed_cells} != set(spec["expected_methods"]):
        raise RuntimeError("repair set contains an unexpected method")
    if {str(cell["task_key"]) for cell in failed_cells} != set(spec["expected_tasks"]):
        raise RuntimeError("repair set contains an unexpected task")
    if {float(cell["retention_ratio"]) for cell in failed_cells} != {
        float(value) for value in spec["expected_retention_ratios"]
    }:
        raise RuntimeError("repair set contains an unexpected retention ratio")

    destination.mkdir(parents=True, exist_ok=False)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "workers").mkdir()
    (output_root / "traces").mkdir()
    _write_once(destination / "repair_spec.yaml", spec_path.read_bytes())
    _write_once(destination / "source.patch", source_patch)
    source_patch_sha = _sha(source_patch)

    for job, cell in completed_cells:
        run_id = str(cell["run_id"])
        _symlink_directory(source_root / run_id, output_root / run_id)
        _symlink_directory(
            Path(str(job["trace_directory"])).resolve() / run_id,
            output_root / "traces" / run_id,
        )

    workers = int(spec["workers"])
    buckets = _balance(failed_cells, workers)
    repair_jobs: list[dict[str, Any]] = []
    for worker_index, cells in enumerate(buckets):
        repair_jobs.append(
            {
                "schema_version": 1,
                "classification": spec["classification"],
                "worker_index": worker_index,
                "output_root": str(output_root),
                "trace_directory": str(output_root / "traces"),
                "source_git_sha": git_sha,
                "source_git_dirty": git_dirty,
                "source_patch_sha256": source_patch_sha,
                "cells": cells,
            }
        )
    repair_index_payload = b"".join(_canonical_bytes(row) for row in repair_jobs)
    repair_index = destination / "jobs.jsonl"
    _write_once(repair_index, repair_index_payload)
    repair_manifest = {
        "schema_version": 1,
        "classification": spec["classification"],
        "index_sha256": _sha(repair_index_payload),
        "worker_count": workers,
        "worker_shard_counts": [len(bucket) for bucket in buckets],
        "failed_shard_count": len(failed_cells),
        "failed_row_count": failed_rows,
        "failed_logical_cell_count": len(logical_ids),
        "source_continuation_index": str(source_index),
        "source_continuation_index_sha256": _sha(source_payload),
        "source_output_root": str(source_root),
        "output_root": str(output_root),
        "behavior_hashes": observed_behavior,
        "source_git_sha": git_sha,
        "source_git_dirty": git_dirty,
        "source_patch_sha256": source_patch_sha,
        "smoke_result_sha256": _sha(Path(str(spec["smoke_evidence"]["result"])).read_bytes()),
        "smoke_sample_id": smoke_row["sample_id"],
    }
    _write_once(destination / "continuation_manifest.json", _canonical_bytes(repair_manifest))

    overlay_jobs: list[dict[str, Any]] = []
    for source_job in source_jobs:
        overlay_job = dict(source_job)
        overlay_job["classification"] = spec["classification"]
        overlay_job["output_root"] = str(output_root)
        overlay_job["trace_directory"] = str(output_root / "traces")
        overlay_job["source_git_sha"] = git_sha
        overlay_job["source_git_dirty"] = git_dirty
        overlay_job["source_patch_sha256"] = source_patch_sha
        overlay_jobs.append(overlay_job)
    overlay_payload = b"".join(_canonical_bytes(row) for row in overlay_jobs)
    overlay_root = destination / "full_overlay"
    overlay_index = overlay_root / "jobs.jsonl"
    _write_once(overlay_index, overlay_payload)
    overlay_manifest = dict(source_manifest)
    overlay_manifest.update(
        {
            "classification": spec["classification"],
            "index_sha256": _sha(overlay_payload),
            "output_root": str(output_root),
            "source_git_sha": git_sha,
            "source_git_dirty": git_dirty,
            "source_patch_sha256": source_patch_sha,
            "repair_index": str(repair_index),
            "repair_index_sha256": _sha(repair_index_payload),
            "linked_completed_shards": len(completed_cells),
            "reexecuted_failed_shards": len(failed_cells),
        }
    )
    _write_once(
        overlay_root / "continuation_manifest.json",
        _canonical_bytes(overlay_manifest),
    )
    print(
        json.dumps(
            {
                "repair_index": str(repair_index),
                "repair_index_sha256": _sha(repair_index_payload),
                "overlay_index": str(overlay_index),
                "overlay_index_sha256": _sha(overlay_payload),
                "output_root": str(output_root),
                "worker_shard_counts": [len(bucket) for bucket in buckets],
                "linked_completed_shards": len(completed_cells),
                "reexecuted_failed_shards": len(failed_cells),
                "repair_rows": failed_rows,
            },
            sort_keys=True,
        )
    )
    return repair_index, overlay_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    materialize(args.spec.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
