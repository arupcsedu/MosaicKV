"""Append-only JSONL storage, deterministic merging, and Parquet materialization."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

from mosaickv.evaluation.results import RESULT_COLUMNS, EvaluationResult
from mosaickv.types import JsonObject


class ResultConflictError(RuntimeError):
    """Raised when the same run/sample key maps to different observations."""


def result_key(result: EvaluationResult) -> tuple[str, str]:
    """Return the resume and deduplication key."""

    return result.run_id, result.sample_id


def _parse_line(line: str, path: Path, line_number: int) -> EvaluationResult:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path} at line {line_number}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"result in {path} at line {line_number} must be an object")
    try:
        return EvaluationResult.from_json_object(cast("JsonObject", payload))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid result in {path} at line {line_number}: {error}") from error


def load_jsonl(path: str | Path) -> tuple[EvaluationResult, ...]:
    """Load raw rows and reject conflicting duplicate keys."""

    source = Path(path)
    if not source.exists():
        return ()
    by_key: dict[tuple[str, str], EvaluationResult] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            result = _parse_line(line, source, line_number)
            key = result_key(result)
            previous = by_key.get(key)
            if previous is not None and previous != result:
                raise ResultConflictError(
                    f"conflicting duplicate run/sample key {key!r} in {source}"
                )
            by_key[key] = result
    return tuple(by_key.values())


class JsonlResultStore:
    """Append-only result store with lock-protected resume semantics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def results(self, *, run_id: str | None = None) -> tuple[EvaluationResult, ...]:
        """Return validated rows, optionally restricted to one run."""

        rows = load_jsonl(self.path)
        if run_id is None:
            return rows
        return tuple(row for row in rows if row.run_id == run_id)

    def completed_sample_ids(self, run_id: str) -> frozenset[str]:
        """Return all terminal sample IDs already recorded for a run."""

        return frozenset(row.sample_id for row in self.results(run_id=run_id))

    def append(self, result: EvaluationResult) -> bool:
        """Append once; return false for an identical existing row.

        A differing row for an existing ``(run_id, sample_id)`` is rejected so
        resume never silently changes a raw observation.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            result.to_json_object(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                previous = _parse_line(line, self.path, line_number)
                if result_key(previous) != result_key(result):
                    continue
                if previous == result:
                    return False
                raise ResultConflictError(
                    "refusing to replace existing result for "
                    f"run={result.run_id!r}, sample={result.sample_id!r}"
                )
            handle.seek(0, os.SEEK_END)
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True


def _deduplicate(results: Iterable[EvaluationResult]) -> tuple[EvaluationResult, ...]:
    by_key: dict[tuple[str, str], EvaluationResult] = {}
    for result in results:
        key = result_key(result)
        previous = by_key.get(key)
        if previous is not None and previous != result:
            raise ResultConflictError(f"conflicting rows for run/sample key {key!r}")
        by_key[key] = result
    return tuple(by_key[key] for key in sorted(by_key))


def merge_jsonl(inputs: Sequence[str | Path], output: str | Path) -> Path:
    """Merge raw files without duplicate run/sample keys."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite merged raw output: {destination}")
    rows = _deduplicate(row for path in inputs for row in load_jsonl(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(
                    json.dumps(
                        row.to_json_object(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def _parquet_schema(pa: Any) -> Any:
    strings = {
        "run_id",
        "sample_id",
        "task",
        "status",
        "error",
        "model",
        "backend",
        "method",
        "answer",
        "reference",
        "manifest_path",
    }
    integers = {
        "generated_tokens",
        "active_kv_bytes",
        "residual_kv_bytes",
        "peak_gpu_memory",
        "repair_count",
        "repaired_bytes",
        "schema_version",
    }
    fields = []
    for name in RESULT_COLUMNS:
        if name in strings:
            data_type = pa.string()
        elif name in integers:
            data_type = pa.int64()
        else:
            data_type = pa.float64()
        fields.append(pa.field(name, data_type, nullable=name not in {"run_id", "sample_id"}))
    return pa.schema(fields)


def write_parquet_aggregate(
    results: Iterable[EvaluationResult],
    output: str | Path,
) -> Path:
    """Write a deduplicated Parquet materialization of per-sample raw rows.

    This function deliberately performs no statistical reduction: downstream
    tables must retain sample lineage and derive statistics in versioned code.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Parquet output requires the evaluation environment with pyarrow installed"
        ) from error

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Parquet output: {destination}")
    rows = _deduplicate(results)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([row.to_json_object() for row in rows], schema=_parquet_schema(pa))
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".parquet",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        parquet.write_table(table, temporary_name, compression="zstd")
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "JsonlResultStore",
    "ResultConflictError",
    "load_jsonl",
    "merge_jsonl",
    "result_key",
    "write_parquet_aggregate",
]
