#!/usr/bin/env python3
"""Validate terminal coverage and critical invariants for the LLaVA campaign."""

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

from mosaickv.evaluation.storage import load_jsonl


def _trace_tokens(trace_root: Path, run_id: str, sample_id: str) -> tuple[int, ...]:
    matches: list[tuple[int, tuple[int, ...]]] = []
    for path in (trace_root / run_id).glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("sample_id") != sample_id or value.get("status") != "completed":
            continue
        raw = value.get("generated_token_ids")
        if not isinstance(raw, list) or not all(isinstance(token, int) for token in raw):
            raise ValueError(f"trace has no generated token IDs: {run_id}/{sample_id}")
        marker = "-attempt-"
        attempt = 0 if marker not in path.stem else int(path.stem.rsplit(marker, 1)[1])
        matches.append((attempt, tuple(cast("list[int]", raw))))
    if not matches:
        raise ValueError(f"missing completed trace: {run_id}/{sample_id}")
    return max(matches, key=lambda item: item[0])[1]


def _atomic_report(path: Path, payload: dict[str, Any]) -> None:
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


def validate(index: Path, output_root: Path) -> tuple[int, dict[str, Any]]:
    manifest_path = index.parent / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(index.read_bytes()).hexdigest() != manifest.get("index_sha"):
        raise RuntimeError("campaign index SHA does not match manifest")
    jobs = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    issues: list[str] = []
    worker_records = sorted((output_root / "workers").glob("*.json"))
    if len(worker_records) != len(jobs):
        issues.append(f"worker record count {len(worker_records)} != {len(jobs)}")
    for path in worker_records:
        value = json.loads(path.read_text(encoding="utf-8"))
        for error in cast("list[dict[str, str]]", value.get("errors", [])):
            issues.append(
                f"worker {value.get('array_index')} {error.get('run_id')}: {error.get('error')}"
            )

    rows_seen = 0
    completed_seen = 0
    cell_means: dict[tuple[str, str, int, float], float] = {}
    full_tokens: dict[tuple[str, str], tuple[int, ...]] = {}
    retention_one: list[tuple[str, str, str, tuple[int, ...]]] = []
    required_timings = (
        "ttft",
        "prefill_time",
        "compression_time",
        "decode_time",
        "end_to_end_time",
    )
    for job in jobs:
        sample_payload = json.loads(Path(job["sample_id_file"]).read_text(encoding="utf-8"))
        expected_ids = tuple(cast("list[str]", sample_payload["sample_ids"]))
        for cell in cast("list[dict[str, Any]]", job["cells"]):
            run_id = str(cell["run_id"])
            run_root = output_root / run_id
            raw_path = run_root / "results.jsonl"
            run_manifest_path = run_root / "manifest.json"
            parquet_path = run_root / "aggregate.parquet"
            if not (raw_path.is_file() and run_manifest_path.is_file() and parquet_path.is_file()):
                issues.append(f"incomplete run artifacts: {run_id}")
                continue
            rows = load_jsonl(raw_path)
            rows_seen += len(rows)
            if len(rows) != len(expected_ids) or {row.sample_id for row in rows} != set(
                expected_ids
            ):
                issues.append(f"sample coverage mismatch: {run_id}")
                continue
            active_values: list[int] = []
            for row in rows:
                if row.status.value != "completed":
                    issues.append(f"failed sample {run_id}/{row.sample_id}: {row.error}")
                    continue
                completed_seen += 1
                for name in required_timings:
                    value = getattr(row, name)
                    if value is None or not math.isfinite(float(value)):
                        issues.append(f"missing/nonfinite {name}: {run_id}/{row.sample_id}")
                if row.generated_tokens != 64:
                    issues.append(f"output-length mismatch: {run_id}/{row.sample_id}")
                if row.active_kv_bytes is None:
                    issues.append(f"missing active KV bytes: {run_id}/{row.sample_id}")
                else:
                    active_values.append(row.active_kv_bytes)
                if row.repair_count != 0:
                    issues.append(
                        f"unexpected repair event in frozen campaign: {run_id}/{row.sample_id}"
                    )
                try:
                    tokens = _trace_tokens(Path(job["trace_directory"]), run_id, row.sample_id)
                except ValueError as error:
                    issues.append(str(error))
                    continue
                if len(tokens) != 64:
                    issues.append(f"trace output-length mismatch: {run_id}/{row.sample_id}")
                identity = (str(job["task_key"]), row.sample_id)
                if cell["requested_method"] == "full_kv":
                    full_tokens[identity] = tokens
                if float(cell["retention_ratio"]) == 1.0:
                    retention_one.append(
                        (str(job["task_key"]), str(cell["requested_method"]), row.sample_id, tokens)
                    )
            if active_values:
                key = (
                    str(job["task_key"]),
                    str(cell["requested_method"]),
                    int(cell["seed"]),
                    float(cell["retention_ratio"]),
                )
                cell_means[key] = sum(active_values) / len(active_values)

    for task_key, method, sample_id, tokens in retention_one:
        reference = full_tokens.get((task_key, sample_id))
        if reference is None:
            issues.append(f"missing FullKV identity reference: {task_key}/{sample_id}")
        elif tokens != reference:
            issues.append(f"retention-1 token mismatch: {task_key}/{method}/{sample_id}")

    grouped: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for (task, method, seed, ratio), active_bytes in cell_means.items():
        grouped[(task, method, seed)].append((ratio, active_bytes))
    for group_identity, points in grouped.items():
        ordered = sorted(points)
        for previous, current in pairwise(ordered):
            if current[1] + 1e-6 < previous[1]:
                issues.append(
                    "active bytes are nonmonotonic: "
                    f"{group_identity} ratio {previous[0]}->{current[0]} "
                    f"bytes {previous[1]}->{current[1]}"
                )

    expected_rows = int(manifest["expected_evaluation_rows"])
    if rows_seen != expected_rows:
        issues.append(f"result row count {rows_seen} != expected {expected_rows}")
    report = {
        "schema_version": 1,
        "classification": manifest["classification"],
        "status": "passed" if not issues else "failed",
        "job_count": len(jobs),
        "worker_record_count": len(worker_records),
        "expected_evaluation_rows": expected_rows,
        "observed_rows": rows_seen,
        "completed_rows": completed_seen,
        "retention_one_comparisons": len(retention_one),
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
    _atomic_report(output_root / "completion_report.json", report)
    print(json.dumps(report, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
