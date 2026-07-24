#!/usr/bin/env python3
"""Derive the GPU-pilot audit artifacts from immutable per-sample runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from mosaickv.evaluation.results import EvaluationResult
from mosaickv.evaluation.storage import load_jsonl

TIMING_FIELDS = (
    "ttft",
    "prefill_time",
    "compression_time",
    "decode_time",
    "end_to_end_time",
)


@dataclass(frozen=True, slots=True)
class Cell:
    task_key: str
    task: str
    dataset_id: str
    dataset_revision: str
    run_id: str
    requested_method: str
    retention_ratio: float
    role: str
    config_sha: str


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    critical: bool
    detail: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast("dict[str, Any]", value)


def _read_matrix(index: Path) -> tuple[list[Cell], dict[str, Any], dict[str, Any]]:
    manifest = _read_json(index.parent / "pilot_manifest.json")
    spec = _read_json(index.parent / "pilot_spec.json")
    actual_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    if actual_sha != manifest.get("index_sha"):
        raise ValueError("pilot index does not match its immutable manifest SHA")
    cells: list[Cell] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        job = json.loads(line)
        for raw in job["cells"]:
            cells.append(
                Cell(
                    task_key=str(job["task_key"]),
                    task=str(job["task"]),
                    dataset_id=str(job["dataset_id"]),
                    dataset_revision=str(job["dataset_revision"]),
                    run_id=str(raw["run_id"]),
                    requested_method=str(raw["requested_method"]),
                    retention_ratio=float(raw["retention_ratio"]),
                    role=str(raw["role"]),
                    config_sha=str(raw["config_sha"]),
                )
            )
    if not cells:
        raise ValueError("pilot index contains no cells")
    return cells, manifest, spec


def _atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived pilot artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _json_line(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _as_float(value: object, field: str) -> float:
    if not _finite(value):
        raise ValueError(f"{field} must be a finite measured number")
    return float(cast("int | float", value))


def _as_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _mean(rows: Sequence[EvaluationResult], field: str) -> float | None:
    values = [float(value) for row in rows if (value := getattr(row, field)) is not None]
    return statistics.fmean(values) if values else None


def _median(rows: Sequence[EvaluationResult], field: str) -> float | None:
    values = [float(value) for row in rows if (value := getattr(row, field)) is not None]
    return statistics.median(values) if values else None


def _load_cells(
    cells: Sequence[Cell], run_root: Path
) -> tuple[dict[str, tuple[EvaluationResult, ...]], dict[str, dict[str, Any]], list[str]]:
    rows_by_run: dict[str, tuple[EvaluationResult, ...]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for cell in cells:
        run_dir = run_root / cell.run_id
        raw_path = run_dir / "results.jsonl"
        manifest_path = run_dir / "manifest.json"
        if not raw_path.is_file():
            errors.append(f"{cell.run_id}: missing results.jsonl")
            rows_by_run[cell.run_id] = ()
            continue
        try:
            rows = load_jsonl(raw_path)
            rows_by_run[cell.run_id] = rows
        except Exception as error:
            errors.append(f"{cell.run_id}: invalid results: {type(error).__name__}: {error}")
            rows_by_run[cell.run_id] = ()
            continue
        if not manifest_path.is_file():
            errors.append(f"{cell.run_id}: missing immutable manifest")
            continue
        manifest = _read_json(manifest_path)
        manifests[cell.run_id] = manifest
        expected_sha = manifest.get("artifacts", {}).get("raw_output_sha")
        actual_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if expected_sha != actual_sha:
            errors.append(f"{cell.run_id}: raw SHA does not match manifest")
        if manifest.get("source", {}).get("config_sha") != cell.config_sha:
            errors.append(f"{cell.run_id}: config SHA does not match matrix")
    return rows_by_run, manifests, errors


def _aggregate(
    cells: Sequence[Cell], rows_by_run: Mapping[str, tuple[EvaluationResult, ...]]
) -> list[dict[str, object]]:
    aggregates: list[dict[str, object]] = []
    for cell in cells:
        rows = list(rows_by_run.get(cell.run_id, ()))
        completed = [row for row in rows if row.status.value == "completed"]
        failed = [row for row in rows if row.status.value == "failed"]
        aggregates.append(
            {
                "schema_version": 1,
                "classification": "derived_from_measured_diagnostic_gpu_pilot",
                "run_id": cell.run_id,
                "task": cell.task_key,
                "requested_method": cell.requested_method,
                "effective_methods": sorted({row.method for row in rows}),
                "retention_ratio": cell.retention_ratio,
                "role": cell.role,
                "terminal_samples": len(rows),
                "completed_samples": len(completed),
                "failed_samples": len(failed),
                "task_score_mean": _mean(completed, "task_score"),
                "active_kv_bytes_mean": _mean(completed, "active_kv_bytes"),
                "active_kv_bytes_median": _median(completed, "active_kv_bytes"),
                "residual_kv_bytes_mean": _mean(completed, "residual_kv_bytes"),
                "ttft_mean": _mean(completed, "ttft"),
                "prefill_time_mean": _mean(completed, "prefill_time"),
                "compression_time_mean": _mean(completed, "compression_time"),
                "decode_time_mean": _mean(completed, "decode_time"),
                "end_to_end_time_mean": _mean(completed, "end_to_end_time"),
                "repair_count_total": sum(row.repair_count or 0 for row in completed),
                "manifest_paths": sorted({row.manifest_path for row in rows}),
            }
        )
    return aggregates


def _row_map(rows: Iterable[EvaluationResult]) -> dict[str, EvaluationResult]:
    return {row.sample_id: row for row in rows}


def _identity_check(
    cells: Sequence[Cell], rows_by_run: Mapping[str, tuple[EvaluationResult, ...]]
) -> Check:
    mismatches: list[str] = []
    for task in sorted({cell.task_key for cell in cells}):
        reference_cell = next(
            (
                cell
                for cell in cells
                if cell.task_key == task
                and cell.requested_method == "full_kv"
                and cell.role == "primary"
            ),
            None,
        )
        if reference_cell is None:
            mismatches.append(f"{task}: FullKV cell missing")
            continue
        reference = _row_map(rows_by_run.get(reference_cell.run_id, ()))
        for cell in cells:
            if (
                cell.task_key != task
                or cell.retention_ratio != 1.0
                or cell.role != "primary"
                or cell.requested_method == "full_kv"
            ):
                continue
            candidate = _row_map(rows_by_run.get(cell.run_id, ()))
            for sample_id in sorted(set(reference) | set(candidate)):
                left = reference.get(sample_id)
                right = candidate.get(sample_id)
                if (
                    left is None
                    or right is None
                    or left.status.value != "completed"
                    or right.status.value != "completed"
                    or left.answer != right.answer
                    or left.task_score != right.task_score
                    or left.active_kv_bytes != right.active_kv_bytes
                ):
                    mismatches.append(f"{task}/{cell.requested_method}/{sample_id}")
    detail = "all supported 1.0 cells match FullKV exactly"
    if mismatches:
        detail = f"{len(mismatches)} mismatches; first: {', '.join(mismatches[:5])}"
    return Check("retention_1_identity", not mismatches, True, detail)


def _monotonic_check(
    cells: Sequence[Cell], rows_by_run: Mapping[str, tuple[EvaluationResult, ...]]
) -> Check:
    violations: list[str] = []
    grouped: dict[tuple[str, str], list[Cell]] = defaultdict(list)
    for cell in cells:
        if cell.role == "primary":
            grouped[(cell.task_key, cell.requested_method)].append(cell)
    for (task, method), method_cells in grouped.items():
        if len(method_cells) < 2:
            continue
        maps = {
            cell.retention_ratio: _row_map(rows_by_run.get(cell.run_id, ()))
            for cell in method_cells
        }
        sample_ids = set.intersection(*(set(rows) for rows in maps.values())) if maps else set()
        if len(sample_ids) != 20:
            violations.append(f"{task}/{method}: only {len(sample_ids)} common samples")
        for sample_id in sorted(sample_ids):
            values: list[tuple[float, int]] = []
            for ratio in sorted(maps):
                row = maps[ratio][sample_id]
                if row.status.value != "completed" or row.active_kv_bytes is None:
                    violations.append(f"{task}/{method}/{sample_id}: incomplete bytes")
                    values = []
                    break
                values.append((ratio, row.active_kv_bytes))
            if any(right[1] < left[1] for left, right in pairwise(values)):
                violations.append(f"{task}/{method}/{sample_id}: {values}")
    detail = "active bytes are nondecreasing with retention for every common sample"
    if violations:
        detail = f"{len(violations)} violations; first: {', '.join(violations[:5])}"
    return Check("active_bytes_monotonic", not violations, True, detail)


def _ordinary_check(
    cells: Sequence[Cell],
    rows_by_run: Mapping[str, tuple[EvaluationResult, ...]],
    run_root: Path,
    revision: int,
) -> Check:
    failures: list[str] = []
    for task in sorted({cell.task_key for cell in cells}):
        full = next(
            (
                cell
                for cell in cells
                if cell.task_key == task
                and cell.requested_method == "full_kv"
                and cell.role == "primary"
            ),
            None,
        )
        ordinary_path = (
            run_root
            / "ordinary_reference"
            / f"ordinary-generate-reference-v{revision}-{task}"
            / "results.jsonl"
        )
        if full is None or not ordinary_path.is_file():
            failures.append(f"{task}: ordinary or FullKV reference missing")
            continue
        ordinary = _row_map(load_jsonl(ordinary_path))
        reference = _row_map(rows_by_run.get(full.run_id, ()))
        for sample_id in sorted(set(reference) | set(ordinary)):
            left = reference.get(sample_id)
            right = ordinary.get(sample_id)
            if (
                left is None
                or right is None
                or left.status.value != "completed"
                or right.status.value != "completed"
                or left.answer != right.answer
                or left.task_score != right.task_score
                or right.generated_tokens != 32
            ):
                failures.append(f"{task}/{sample_id}")
    detail = "FullKV answers and scores match fixed-length model.generate"
    if failures:
        detail = f"{len(failures)} parity failures; first: {', '.join(failures[:5])}"
    return Check("ordinary_fullkv_parity", not failures, True, detail)


def _trace_selection(trace_root: Path, run_id: str) -> dict[str, object]:
    selected: dict[str, object] = {}
    for path in sorted((trace_root / run_id).glob("*.json")):
        trace = _read_json(path)
        if trace.get("status") == "completed" and isinstance(trace.get("sample_id"), str):
            selected[str(trace["sample_id"])] = trace.get("selected_blocks")
    return selected


def _random_reproducibility(
    cells: Sequence[Cell], rows_by_run: Mapping[str, tuple[EvaluationResult, ...]], trace_root: Path
) -> Check:
    primary = next(
        (
            cell
            for cell in cells
            if cell.task_key == "mmstar"
            and cell.requested_method == "random_kv"
            and cell.retention_ratio == 0.5
            and cell.role == "primary"
        ),
        None,
    )
    repeat = next((cell for cell in cells if cell.role == "random_reproducibility_duplicate"), None)
    failures: list[str] = []
    if primary is None or repeat is None:
        failures.append("duplicate cell missing")
    else:
        left = _row_map(rows_by_run.get(primary.run_id, ()))
        right = _row_map(rows_by_run.get(repeat.run_id, ()))
        left_traces = _trace_selection(trace_root, primary.run_id)
        right_traces = _trace_selection(trace_root, repeat.run_id)
        for sample_id in sorted(set(left) | set(right)):
            a = left.get(sample_id)
            b = right.get(sample_id)
            if (
                a is None
                or b is None
                or a.status != b.status
                or a.answer != b.answer
                or a.active_kv_bytes != b.active_kv_bytes
                or left_traces.get(sample_id) != right_traces.get(sample_id)
            ):
                failures.append(sample_id)
    detail = "seeded random rows and selected-block traces reproduce exactly"
    if failures:
        detail = f"{len(failures)} failures; first: {', '.join(failures[:5])}"
    return Check("random_seed_reproducibility", not failures, True, detail)


def _basic_checks(
    cells: Sequence[Cell],
    rows_by_run: Mapping[str, tuple[EvaluationResult, ...]],
    manifests: Mapping[str, dict[str, Any]],
    load_errors: Sequence[str],
    spec: Mapping[str, Any],
) -> list[Check]:
    coverage = [
        f"{cell.run_id}:{len(rows_by_run.get(cell.run_id, ()))}"
        for cell in cells
        if len(rows_by_run.get(cell.run_id, ())) != 20
    ]
    coverage.extend(load_errors)
    checks = [
        Check(
            "complete_cell_coverage",
            not coverage,
            True,
            "all matrix cells contain 20 immutable terminal rows"
            if not coverage
            else f"{len(coverage)} failures; first: {', '.join(coverage[:5])}",
        )
    ]
    invalid_numbers: list[str] = []
    missing_timings: list[str] = []
    truncated: list[str] = []
    error_rows: list[str] = []
    nonrepair: list[str] = []
    mosaic_overhead: list[str] = []
    for cell in cells:
        for row in rows_by_run.get(cell.run_id, ()):
            for field in TIMING_FIELDS:
                value = getattr(row, field)
                if value is not None and not _finite(value):
                    invalid_numbers.append(f"{cell.run_id}/{row.sample_id}/{field}")
                if value is None:
                    missing_timings.append(f"{cell.run_id}/{row.sample_id}/{field}")
            if row.task_score is not None and not _finite(row.task_score):
                invalid_numbers.append(f"{cell.run_id}/{row.sample_id}/task_score")
            if row.status.value == "completed" and row.generated_tokens != 32:
                truncated.append(f"{cell.run_id}/{row.sample_id}:{row.generated_tokens}")
            if row.status.value == "failed" and not (row.error and row.error.strip()):
                error_rows.append(f"{cell.run_id}/{row.sample_id}")
            if (
                row.status.value == "completed"
                and cell.requested_method != "mosaickv_full"
                and row.repair_count != 0
            ):
                nonrepair.append(f"{cell.run_id}/{row.sample_id}:{row.repair_count}")
            if (
                row.status.value == "completed"
                and cell.requested_method.startswith("mosaickv_")
                and row.compression_time is None
            ):
                mosaic_overhead.append(f"{cell.run_id}/{row.sample_id}")
    checks.extend(
        [
            Check(
                "finite_complete_timings",
                not invalid_numbers and not missing_timings,
                True,
                "all rows have finite timing fields"
                if not invalid_numbers and not missing_timings
                else (
                    f"{len(invalid_numbers)} nonfinite and {len(missing_timings)} missing; "
                    f"first: {', '.join((invalid_numbers + missing_timings)[:5])}"
                ),
            ),
            Check(
                "fixed_output_length",
                not truncated,
                True,
                "all completed outputs contain exactly 32 generated tokens"
                if not truncated
                else f"{len(truncated)} truncated rows; first: {', '.join(truncated[:5])}",
            ),
            Check(
                "failed_samples_have_errors",
                not error_rows,
                True,
                "every failed row has an explicit error"
                if not error_rows
                else f"{len(error_rows)} failed rows lack errors",
            ),
            Check(
                "nonrepair_methods_do_not_repair",
                not nonrepair,
                True,
                "repair_count is zero for every method without repair"
                if not nonrepair
                else f"{len(nonrepair)} violations; first: {', '.join(nonrepair[:5])}",
            ),
            Check(
                "mosaickv_compression_overhead_recorded",
                not mosaic_overhead,
                True,
                "every MosaicKV row records compression_time"
                if not mosaic_overhead
                else f"{len(mosaic_overhead)} rows lack compression_time",
            ),
        ]
    )
    provenance_errors: list[str] = []
    git_shas: set[str] = set()
    for cell in cells:
        manifest = manifests.get(cell.run_id)
        if manifest is None:
            continue
        source = manifest.get("source", {})
        git_shas.add(str(source.get("git_sha")))
        if source.get("git_dirty") is not False or source.get("canonical_eligible") is not True:
            provenance_errors.append(f"{cell.run_id}: dirty or ineligible source")
        if manifest.get("dataset", {}).get("id") != cell.dataset_id:
            provenance_errors.append(f"{cell.run_id}: dataset ID mismatch")
        if manifest.get("dataset", {}).get("revision") != cell.dataset_revision:
            provenance_errors.append(f"{cell.run_id}: dataset revision mismatch")
        if manifest.get("model", {}).get("revision") in {None, "", "main"}:
            provenance_errors.append(f"{cell.run_id}: model revision is not immutable")
    if len(git_shas) > 1:
        provenance_errors.append(f"multiple source git SHAs: {sorted(git_shas)}")
    checks.append(
        Check(
            "immutable_model_dataset_provenance",
            not provenance_errors and len(manifests) == len(cells),
            True,
            "all manifests share one clean git SHA and exact model/dataset revisions"
            if not provenance_errors and len(manifests) == len(cells)
            else f"{len(provenance_errors)} errors; first: {', '.join(provenance_errors[:5])}",
        )
    )
    excluded = cast("Mapping[str, str]", spec.get("excluded", {}))
    requested = {str(value) for value in cast("Sequence[object]", spec.get("methods", ()))}
    materialized = {cell.requested_method for cell in cells if cell.role == "primary"}
    missing_requested = sorted(requested - materialized)
    accidentally_materialized = sorted(set(excluded).intersection(materialized))
    checks.append(
        Check(
            "all_requested_methods_executed",
            not missing_requested and not accidentally_materialized,
            True,
            "all supported requested methods executed; unsupported combinations were excluded "
            "before submission"
            if not missing_requested and not accidentally_materialized
            else (
                f"missing={missing_requested}, accidentally_materialized="
                f"{accidentally_materialized}"
            ),
        )
    )
    expected_effective = {
        str(method): {str(label) for label in labels}
        for method, labels in cast(
            "Mapping[str, Sequence[object]]", spec.get("expected_effective_methods", {})
        ).items()
    }
    fallbacks: list[str] = []
    for cell in cells:
        if cell.role != "primary" or cell.requested_method not in {
            "mosaickv_proto",
            "mosaickv_full",
            "prefixkv_reimpl",
        }:
            continue
        effective = {row.method for row in rows_by_run.get(cell.run_id, ())}
        allowed = {cell.requested_method, *expected_effective.get(cell.requested_method, set())}
        if effective and not effective.issubset(allowed):
            fallbacks.append(f"{cell.task_key}/{cell.requested_method}: {sorted(effective)}")
    checks.append(
        Check(
            "requested_method_identity",
            not fallbacks,
            True,
            "effective runtime labels equal the requested method or a predeclared safe label"
            if not fallbacks
            else f"{len(fallbacks)} generalized/fallback cells; first: {', '.join(fallbacks[:5])}",
        )
    )
    return checks


def _trace_metric(trace_root: Path, run_id: str, field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for path in sorted((trace_root / run_id).glob("*.json")):
        trace = _read_json(path)
        sample_id = trace.get("sample_id")
        value = trace.get(field)
        if (
            trace.get("status") == "completed"
            and isinstance(sample_id, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 1
        ):
            values[sample_id] = value
    return values


def _byte_budget_check(
    cells: Sequence[Cell],
    rows_by_run: Mapping[str, tuple[EvaluationResult, ...]],
    trace_root: Path,
) -> Check:
    mismatches: list[str] = []
    missing: list[str] = []
    for task in sorted({cell.task_key for cell in cells}):
        for ratio in (0.5, 0.2, 0.1):
            method_maps: dict[str, dict[str, EvaluationResult]] = {}
            trace_maps: dict[str, dict[str, int]] = {}
            for cell in cells:
                if (
                    cell.task_key == task
                    and cell.retention_ratio == ratio
                    and cell.role == "primary"
                ):
                    method_maps[cell.requested_method] = _row_map(rows_by_run.get(cell.run_id, ()))
                    trace_maps[cell.requested_method] = _trace_metric(
                        trace_root,
                        cell.run_id,
                        "packed_prefill_active_kv_bytes",
                    )
            sample_ids = (
                set.intersection(*(set(rows) for rows in method_maps.values()))
                if method_maps
                else set()
            )
            for sample_id in sorted(sample_ids):
                values = {
                    method: rows[sample_id].active_kv_bytes
                    for method, rows in method_maps.items()
                    if rows[sample_id].status.value == "completed"
                }
                packed_values = {
                    method: traces.get(sample_id) for method, traces in trace_maps.items()
                }
                if any(value is None for value in packed_values.values()):
                    missing.append(f"{task}/{ratio}/{sample_id}")
                if len(set(values.values())) > 1 or len(set(packed_values.values())) > 1:
                    mismatches.append(f"{task}/{ratio}/{sample_id}")
    detail = "compressed methods use identical measured physical prefill active-KV bytes"
    if mismatches or missing:
        detail = (
            f"{len(mismatches)} unequal and {len(missing)} missing sample/ratio groups; "
            f"first: {', '.join((mismatches + missing)[:5])}. "
            "Ratio-only quality comparisons are invalid."
        )
    return Check(
        "active_byte_budget_fairness",
        not mismatches and not missing,
        True,
        detail,
    )


def _write_parquet(
    path: Path, aggregates: Sequence[Mapping[str, object]], metadata: Mapping[str, str]
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived pilot artifact: {path}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(list(aggregates))
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata.update({key.encode(): value.encode() for key, value in metadata.items()})
    table = table.replace_schema_metadata(schema_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _plot_pdf(
    path: Path,
    aggregates: Sequence[Mapping[str, object]],
    *,
    x_key: str,
    title: str,
    x_label: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived pilot artifact: {path}")
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#17becf",
    )
    methods = sorted(
        {
            str(row["requested_method"])
            for row in aggregates
            if row["role"] == "primary" and row["completed_samples"]
        }
    )
    color_by_method = {method: colors[index % len(colors)] for index, method in enumerate(methods)}
    draw.text((55, 30), title, fill="black", font=font)
    draw.text(
        (55, 55),
        "Diagnostic measured means; x-axis uses actual cache/timing, never nominal retention.",
        fill="#444444",
        font=font,
    )
    legend_y = 82
    for index, method in enumerate(methods):
        x = 55 + (index % 5) * 330
        y = legend_y + (index // 5) * 22
        draw.line((x, y + 6, x + 22, y + 6), fill=color_by_method[method], width=4)
        draw.text((x + 28, y), method, fill="black", font=font)
    tasks = sorted({str(row["task"]) for row in aggregates})
    panel_top = 145
    panel_width = (width - 80) // max(1, len(tasks))
    for task_index, task in enumerate(tasks):
        left = 55 + task_index * panel_width
        right = left + panel_width - 35
        top = panel_top
        bottom = height - 90
        points = [
            row
            for row in aggregates
            if row["task"] == task
            and row["role"] == "primary"
            and row["completed_samples"]
            and row[x_key] is not None
            and row["task_score_mean"] is not None
        ]
        draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
        draw.text((left + 8, top + 8), task, fill="black", font=font)
        if not points:
            draw.text(
                (left + 20, top + 50), "No completed measured points", fill="#aa0000", font=font
            )
            continue
        x_values = [_as_float(row[x_key], x_key) for row in points]
        if x_key == "active_kv_bytes_mean":
            x_values = [value / (1024 * 1024) for value in x_values]
        x_min, x_max = min(x_values), max(x_values)
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        plot_left, plot_right = left + 45, right - 18
        plot_top, plot_bottom = top + 45, bottom - 38
        for tick in range(6):
            fraction = tick / 5
            tick_y = plot_bottom - fraction * (plot_bottom - plot_top)
            draw.line((plot_left, tick_y, plot_right, tick_y), fill="#dddddd", width=1)
            draw.text((left + 5, tick_y - 5), f"{fraction:.1f}", fill="#555555", font=font)
        draw.text((plot_left, bottom - 24), x_label, fill="#333333", font=font)
        for method in methods:
            method_points: list[tuple[float, float, float]] = []
            for row in points:
                if row["requested_method"] != method:
                    continue
                x_value = _as_float(row[x_key], x_key)
                if x_key == "active_kv_bytes_mean":
                    x_value /= 1024 * 1024
                method_points.append(
                    (
                        x_value,
                        _as_float(row["task_score_mean"], "task_score_mean"),
                        _as_float(row["retention_ratio"], "retention_ratio"),
                    )
                )
            method_points.sort()
            pixels: list[tuple[float, float]] = []
            for x_value, score, ratio in method_points:
                px = plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left)
                py = plot_bottom - max(0.0, min(1.0, score)) * (plot_bottom - plot_top)
                pixels.append((px, py))
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color_by_method[method])
                draw.text((px + 6, py - 9), f"{ratio:g}", fill=color_by_method[method], font=font)
            if len(pixels) > 1:
                draw.line(pixels, fill=color_by_method[method], width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".pdf", delete=False
        ) as handle:
            temporary = handle.name
        image.save(temporary, "PDF", resolution=150.0)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _report(
    checks: Sequence[Check],
    aggregates: Sequence[Mapping[str, object]],
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    run_root: Path,
    git_shas: Sequence[str],
) -> str:
    critical_failures = [check for check in checks if check.critical and not check.passed]
    verdict = (
        "CRITICAL INVARIANTS FAILED — MAIN EXPERIMENT BLOCKED"
        if critical_failures
        else "ALL CRITICAL PILOT INVARIANTS PASSED"
    )
    lines = [
        "# GPU pilot sanity report",
        "",
        f"**Verdict: {verdict}.**",
        "",
        "This is a diagnostic pilot, not a paper-result table. Every number below is derived by "
        "this script from immutable per-sample observations; no failed or missing row is imputed.",
        "",
        "## Provenance",
        "",
        f"- Classification: `{manifest['classification']}`",
        f"- Pilot revision: `{spec['pilot_revision']}`",
        f"- Matrix index SHA-256: `{manifest['index_sha']}`",
        f"- Source git SHA(s): `{', '.join(git_shas) if git_shas else 'missing'}`",
        f"- Immutable run root: `{run_root}`",
        "- Model: `Qwen/Qwen2.5-VL-3B-Instruct` at the revision recorded in each manifest",
        "- Backend: Hugging Face eager, BF16, deterministic greedy decoding, 32 fixed tokens",
        "- Cache and temporary roots: `/scratch/djy8hg/cache/mosaickv`",
        "",
        "## Invariant checks",
        "",
        "| Check | Result | Critical | Evidence |",
        "|---|---:|---:|---|",
    ]
    for check in checks:
        lines.append(
            f"| `{check.name}` | {'PASS' if check.passed else 'FAIL'} | "
            f"{'yes' if check.critical else 'no'} | {check.detail.replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Measured row coverage",
            "",
            "| Task | Terminal rows | Completed | Failed |",
            "|---|---:|---:|---:|",
        ]
    )
    for task in sorted({str(row["task"]) for row in aggregates}):
        task_rows = [row for row in aggregates if row["task"] == task and row["role"] == "primary"]
        lines.append(
            f"| `{task}` | "
            f"{sum(_as_int(row['terminal_samples'], 'terminal_samples') for row in task_rows)} | "
            f"{sum(_as_int(row['completed_samples'], 'completed_samples') for row in task_rows)} | "
            f"{sum(_as_int(row['failed_samples'], 'failed_samples') for row in task_rows)} |"
        )
    excluded = cast("Mapping[str, str]", spec.get("excluded", {}))
    fallbacks = cast("Mapping[str, str]", spec.get("expected_safety_fallbacks", {}))
    lines.extend(["", "## Unsupported and non-equivalent method labels", ""])
    if excluded:
        for method, reason in excluded.items():
            lines.append(f"- `{method}`: unsupported in this Qwen pilot. {reason}")
    if fallbacks:
        for method, reason in fallbacks.items():
            lines.append(f"- `{method}`: requested label uses a safety fallback. {reason}")
    lines.extend(
        [
            "- `prefixkv_reimpl` on Qwen is a generalized reimplementation, not official or "
            "LLaVA-parity PrefixKV.",
            "- `vl_cache_reimpl` remains diagnostic until checkpoint-specific calibration and "
            "paper-trend validation are documented.",
            "",
            "## Remaining threats to fair comparison",
            "",
            "- Nominal retention ratios are not the primary comparison budget. Only measured "
            "physical prefill active KV bytes are plotted and the invariant gate requires exact "
            "sample-level byte equality.",
            "- Paper-derived methods use different mandatory-token and block-granularity rules. "
            "Those design differences are disclosed, not normalized away.",
            "- These are single-pass quality/timing observations, not the configured 5-warmup, "
            "20-repetition systems microbenchmark protocol.",
            "- Qwen cached keys are post-RoPE, so prototype merging and residual repair are "
            "disabled rather than executed unsafely.",
            "- Any failed sample remains in raw JSONL and prevents a complete-cohort claim.",
            "",
            "## Derived artifacts",
            "",
            "- `raw.jsonl`: exact merged primary and reproducibility rows.",
            "- `aggregate.parquet`: one derived group per immutable run, with lineage metadata.",
            "- `pilot_memory_quality.pdf`: score against measured physical prefill active-KV "
            "bytes.",
            "- `pilot_latency_quality.pdf`: score against measured end-to-end latency.",
            "",
            "The main experiment was not launched by this workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(index: Path, run_root: Path, output_dir: Path, figure_dir: Path) -> list[Check]:
    cells, matrix_manifest, spec = _read_matrix(index)
    rows_by_run, manifests, load_errors = _load_cells(cells, run_root)
    all_rows = [row for cell in cells for row in rows_by_run.get(cell.run_id, ())]
    if not all_rows:
        raise RuntimeError("pilot contains no per-sample rows to finalize")
    duplicate_keys: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for row in all_rows:
        key = (row.run_id, row.sample_id)
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)
    if duplicate_keys:
        raise RuntimeError(
            f"duplicate run/sample IDs in merged pilot: {sorted(duplicate_keys)[:5]}"
        )
    aggregates = _aggregate(cells, rows_by_run)
    revision = int(spec["pilot_revision"])
    checks = _basic_checks(cells, rows_by_run, manifests, load_errors, spec)
    checks.extend(
        [
            _identity_check(cells, rows_by_run),
            _monotonic_check(cells, rows_by_run),
            _ordinary_check(cells, rows_by_run, run_root, revision),
            _random_reproducibility(cells, rows_by_run, run_root / "traces"),
            _byte_budget_check(cells, rows_by_run, run_root / "traces"),
        ]
    )
    raw_path = output_dir / "raw.jsonl"
    raw_text = "".join(
        _json_line(row.to_json_object()) + "\n"
        for row in sorted(all_rows, key=lambda item: (item.run_id, item.sample_id))
    )
    _atomic_text(raw_path, raw_text)
    git_shas = sorted(
        {
            str(manifest.get("source", {}).get("git_sha"))
            for manifest in manifests.values()
            if manifest.get("source", {}).get("git_sha")
        }
    )
    _write_parquet(
        output_dir / "aggregate.parquet",
        aggregates,
        {
            "classification": "derived_from_measured_diagnostic_gpu_pilot",
            "matrix_index_sha256": str(matrix_manifest["index_sha"]),
            "source_run_root": str(run_root),
            "source_git_shas": json.dumps(git_shas, separators=(",", ":")),
        },
    )
    _atomic_text(
        output_dir / "sanity_report.md",
        _report(checks, aggregates, matrix_manifest, spec, run_root, git_shas),
    )
    _plot_pdf(
        figure_dir / "pilot_memory_quality.pdf",
        aggregates,
        x_key="active_kv_bytes_mean",
        title="MosaicKV diagnostic pilot: memory-quality",
        x_label="mean active KV (MiB)",
    )
    _plot_pdf(
        figure_dir / "pilot_latency_quality.pdf",
        aggregates,
        x_key="end_to_end_time_mean",
        title="MosaicKV diagnostic pilot: latency-quality",
        x_label="mean end-to-end latency (s)",
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/gpu-pilot-matrix-v5/jobs.jsonl"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/gpu-pilot-v5"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("results/pilot"))
    parser.add_argument("--figure-directory", type=Path, default=Path("figures"))
    args = parser.parse_args()
    checks = finalize(
        args.index.resolve(),
        args.run_root.resolve(),
        args.output_directory.resolve(),
        args.figure_directory.resolve(),
    )
    failed = [check.name for check in checks if check.critical and not check.passed]
    print(_json_line({"critical_failures": failed, "main_experiment_blocked": bool(failed)}))
    # A blocked pilot must also be a failed batch job. Returning success here
    # previously made the revision-4 finalizer look healthy in Slurm even
    # though its report contained critical invariant failures.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
