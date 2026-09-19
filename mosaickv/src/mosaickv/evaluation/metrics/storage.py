"""Append-only storage for the separate paper-native metric stream."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from mosaickv.evaluation.metrics.types import METRIC_COLUMNS, MetricObservation, MetricScope
from mosaickv.types import JsonObject

MetricKey = tuple[str, str, str, str, str, float, str, str, str]


def metric_key(row: MetricObservation) -> MetricKey:
    return (
        row.run_id,
        row.task,
        row.model,
        row.backend,
        row.method,
        row.retention_ratio,
        row.metric_name,
        row.metric_scope.value,
        row.sample_id or "",
    )


def _parse(line: str, path: Path, line_number: int) -> MetricObservation:
    try:
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError("row is not an object")
        return MetricObservation.from_json_object(cast("JsonObject", payload))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid metric row in {path} at line {line_number}: {error}") from error


def load_metric_jsonl(path: str | Path) -> tuple[MetricObservation, ...]:
    source = Path(path)
    if not source.exists():
        return ()
    rows: dict[MetricKey, MetricObservation] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _parse(line, source, line_number)
            key = metric_key(row)
            previous = rows.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate metric key {key!r}")
            rows[key] = row
    return tuple(rows[key] for key in sorted(rows))


class JsonlMetricStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, row: MetricObservation) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            row.to_json_object(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                previous = _parse(line, self.path, line_number)
                if metric_key(previous) != metric_key(row):
                    continue
                if previous == row:
                    return False
                raise ValueError(f"refusing to replace metric row {metric_key(row)!r}")
            handle.seek(0, os.SEEK_END)
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True


def _schema(pa: Any) -> Any:
    strings = {
        "run_id",
        "task",
        "model",
        "backend",
        "method",
        "metric_name",
        "metric_direction",
        "metric_scope",
        "metric_implementation",
        "metric_version",
        "metric_unit",
        "measurement_type",
        "manifest_path",
        "sample_id",
        "evaluator_commit",
        "reference_artifact_sha",
    }
    integers = {
        "evaluated_token_count",
        "denominator",
        "sample_count",
        "schema_version",
    }
    return pa.schema(
        [
            pa.field(
                name,
                (
                    pa.string()
                    if name in strings
                    else pa.int64()
                    if name in integers
                    else pa.float64()
                ),
                nullable=name not in {"run_id", "task", "metric_name"},
            )
            for name in METRIC_COLUMNS
        ]
    )


def write_metric_parquet(rows: Iterable[MetricObservation], output: str | Path) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("metric Parquet output requires pinned pyarrow") from error
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite metric Parquet: {destination}")
    by_key: dict[MetricKey, MetricObservation] = {}
    for row in rows:
        key = metric_key(row)
        previous = by_key.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"conflicting metric rows for {key!r}")
        by_key[key] = row
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [by_key[key].to_json_object() for key in sorted(by_key)], schema=_schema(pa)
    )
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


def write_metric_parquet_tables(
    rows: Iterable[MetricObservation],
    *,
    observations_output: str | Path,
    aggregates_output: str | Path,
) -> tuple[Path, Path]:
    """Write sample observations and corpus aggregates to distinct Parquet tables."""

    materialized = tuple(rows)
    observations = tuple(row for row in materialized if row.metric_scope is MetricScope.SAMPLE)
    aggregates = tuple(row for row in materialized if row.metric_scope is MetricScope.CORPUS)
    observation_path = Path(observations_output)
    aggregate_path = Path(aggregates_output)
    if observation_path.resolve() == aggregate_path.resolve():
        raise ValueError("metric observation and aggregate Parquet paths must differ")
    existing = tuple(path for path in (observation_path, aggregate_path) if path.exists())
    if existing:
        raise FileExistsError(
            "refusing to overwrite metric Parquet table(s): "
            + ", ".join(str(path) for path in existing)
        )
    return (
        write_metric_parquet(observations, observation_path),
        write_metric_parquet(aggregates, aggregate_path),
    )


__all__ = [
    "JsonlMetricStore",
    "load_metric_jsonl",
    "metric_key",
    "write_metric_parquet",
    "write_metric_parquet_tables",
]
