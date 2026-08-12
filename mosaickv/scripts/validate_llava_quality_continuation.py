#!/usr/bin/env python3
"""Validate combined parent and continuation coverage without rewriting raw results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from prepare_llava_quality_continuation import _reusable_cell, _sample_ids

from mosaickv.evaluation.results import EvaluationResult
from mosaickv.evaluation.storage import load_jsonl


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _trace_tokens(trace_root: Path, run_id: str, sample_id: str) -> tuple[int, ...]:
    candidates: list[tuple[int, tuple[int, ...]]] = []
    for path in (trace_root / run_id).glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "completed" or value.get("sample_id") != sample_id:
            continue
        raw = value.get("generated_token_ids")
        if not isinstance(raw, list) or not all(isinstance(token, int) for token in raw):
            continue
        marker = "-attempt-"
        attempt = 0 if marker not in path.stem else int(path.stem.rsplit(marker, 1)[1])
        candidates.append((attempt, tuple(cast("list[int]", raw))))
    if not candidates:
        raise ValueError(f"missing completed trace: {run_id}/{sample_id}")
    return max(candidates, key=lambda item: item[0])[1]


def _validated_rows(
    *,
    run_root: Path,
    config_sha: str,
    expected_ids: tuple[str, ...],
) -> list[EvaluationResult]:
    raw_path = run_root / "results.jsonl"
    manifest_path = run_root / "manifest.json"
    parquet_path = run_root / "aggregate.parquet"
    if not (raw_path.is_file() and manifest_path.is_file() and parquet_path.is_file()):
        raise ValueError(f"incomplete run artifacts: {run_root.name}")
    rows = load_jsonl(raw_path)
    if len(rows) != len(expected_ids) or {row.sample_id for row in rows} != set(expected_ids):
        raise ValueError(f"sample coverage mismatch: {run_root.name}")
    if any(row.status.value != "completed" for row in rows):
        raise ValueError(f"failed sample is present: {run_root.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("config_sha") != config_sha:
        raise ValueError(f"config SHA mismatch: {run_root.name}")
    if (
        manifest.get("artifacts", {}).get("raw_output_sha")
        != hashlib.sha256(raw_path.read_bytes()).hexdigest()
    ):
        raise ValueError(f"raw artifact SHA mismatch: {run_root.name}")
    return list(rows)


def validate(index: Path, output_root: Path) -> tuple[int, dict[str, Any]]:
    continuation_manifest = json.loads(
        (index.parent / "continuation_manifest.json").read_text(encoding="utf-8")
    )
    if hashlib.sha256(index.read_bytes()).hexdigest() != continuation_manifest["index_sha256"]:
        raise RuntimeError("continuation index SHA mismatch")
    continuation_jobs = [
        json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line
    ]
    shard_by_logical: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for job in continuation_jobs:
        for cell in cast("list[dict[str, Any]]", job["cells"]):
            shard_by_logical[str(cell["logical_run_id"])].append((job, cell))

    parent_index = Path(str(continuation_manifest["parent_index"])).resolve()
    parent_jobs = [
        json.loads(line) for line in parent_index.read_text(encoding="utf-8").splitlines() if line
    ]
    parent_output_root = Path(
        json.loads((parent_index.parent / "pilot_manifest.json").read_text())["output_root"]
    )
    issues: list[str] = []
    combined_rows = 0
    reused_cells = 0
    continued_cells = 0
    logical_metadata: dict[str, tuple[str, str, float, int, float]] = {}
    full_tokens: dict[tuple[str, str], tuple[int, ...]] = {}
    retention_one_tokens: list[tuple[str, str, str, tuple[int, ...]]] = []

    worker_paths = sorted((output_root / "workers").glob("worker-*.json"))
    if len(worker_paths) != int(continuation_manifest["worker_count"]):
        issues.append(
            f"worker record count {len(worker_paths)} != {continuation_manifest['worker_count']}"
        )
    for path in worker_paths:
        worker = json.loads(path.read_text(encoding="utf-8"))
        if worker.get("status") != "completed" or worker.get("errors"):
            issues.append(f"worker did not finish cleanly: {path.name}")

    for parent_job in parent_jobs:
        task_key = str(parent_job["task_key"])
        expected_ids = _sample_ids(parent_job)
        for parent_cell in cast("list[dict[str, Any]]", parent_job["cells"]):
            logical_id = str(parent_cell["run_id"])
            method = str(parent_cell["requested_method"])
            ratio = float(parent_cell["retention_ratio"])
            seed = int(parent_cell["seed"])
            rows_with_runs: list[tuple[EvaluationResult, Path, str]] = []
            if _reusable_cell(parent_job, parent_cell, expected_ids):
                reused_cells += 1
                run_root = parent_output_root / logical_id
                try:
                    rows = _validated_rows(
                        run_root=run_root,
                        config_sha=str(parent_cell["config_sha"]),
                        expected_ids=expected_ids,
                    )
                    rows_with_runs.extend(
                        (row, Path(str(parent_job["trace_directory"])), logical_id) for row in rows
                    )
                except ValueError as error:
                    issues.append(str(error))
                    continue
            else:
                continued_cells += 1
                shards = shard_by_logical.get(logical_id, [])
                if len(shards) != int(continuation_manifest["shards_per_logical_cell"]):
                    issues.append(f"continuation shard count mismatch: {logical_id}")
                    continue
                observed_ids: set[str] = set()
                for job, shard in shards:
                    sample_payload = json.loads(
                        Path(str(shard["sample_id_file"])).read_text(encoding="utf-8")
                    )
                    shard_ids = tuple(cast("list[str]", sample_payload["sample_ids"]))
                    overlap = observed_ids.intersection(shard_ids)
                    if overlap:
                        issues.append(f"duplicate continuation samples: {logical_id}/{overlap}")
                    observed_ids.update(shard_ids)
                    try:
                        rows = _validated_rows(
                            run_root=output_root / str(shard["run_id"]),
                            config_sha=str(shard["config_sha"]),
                            expected_ids=shard_ids,
                        )
                        rows_with_runs.extend(
                            (row, Path(str(job["trace_directory"])), str(shard["run_id"]))
                            for row in rows
                        )
                    except ValueError as error:
                        issues.append(str(error))
                if observed_ids != set(expected_ids):
                    issues.append(f"continuation logical coverage mismatch: {logical_id}")
            if len(rows_with_runs) != len(expected_ids):
                issues.append(f"combined logical row count mismatch: {logical_id}")
                continue
            combined_rows += len(rows_with_runs)
            active_values: list[int] = []
            for row, trace_root, run_id in rows_with_runs:
                required = (
                    row.ttft,
                    row.prefill_time,
                    row.compression_time,
                    row.decode_time,
                    row.end_to_end_time,
                )
                if any(value is None or not math.isfinite(float(value)) for value in required):
                    issues.append(f"missing/nonfinite timing: {logical_id}/{row.sample_id}")
                if row.generated_tokens != 64:
                    issues.append(f"output-length mismatch: {logical_id}/{row.sample_id}")
                if row.active_kv_bytes is None:
                    issues.append(f"missing active KV bytes: {logical_id}/{row.sample_id}")
                else:
                    active_values.append(row.active_kv_bytes)
                try:
                    tokens = _trace_tokens(trace_root, run_id, row.sample_id)
                except ValueError as error:
                    issues.append(str(error))
                    continue
                if method == "full_kv":
                    full_tokens[(task_key, row.sample_id)] = tokens
                if ratio == 1.0:
                    retention_one_tokens.append((task_key, method, row.sample_id, tokens))
            if active_values:
                logical_metadata[logical_id] = (
                    task_key,
                    method,
                    ratio,
                    seed,
                    sum(active_values) / len(active_values),
                )

    for task, method, sample_id, tokens in retention_one_tokens:
        reference = full_tokens.get((task, sample_id))
        if reference is None:
            issues.append(f"missing FullKV reference: {task}/{sample_id}")
        elif tokens != reference:
            issues.append(f"retention-1 mismatch: {task}/{method}/{sample_id}")

    curves: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for task, method, ratio, seed, active_bytes in logical_metadata.values():
        curves[(task, method, seed)].append((ratio, active_bytes))
    for identity, points in curves.items():
        for previous, current in pairwise(sorted(points)):
            if current[1] + 1e-6 < previous[1]:
                issues.append(f"active bytes are nonmonotonic: {identity}")

    expected_rows = int(continuation_manifest["expected_combined_rows"])
    if combined_rows != expected_rows:
        issues.append(f"combined rows {combined_rows} != expected {expected_rows}")
    report = {
        "schema_version": 1,
        "classification": continuation_manifest["classification"],
        "status": "passed" if not issues else "failed",
        "reused_logical_cells": reused_cells,
        "continued_logical_cells": continued_cells,
        "combined_logical_cells": reused_cells + continued_cells,
        "combined_rows": combined_rows,
        "expected_combined_rows": expected_rows,
        "retention_one_token_comparisons": len(retention_one_tokens),
        "issues": issues,
    }
    return int(bool(issues)), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    status, report = validate(args.index.resolve(), output_root)
    _atomic_json(output_root / "completion_report.json", report)
    print(json.dumps(report, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
