#!/usr/bin/env python3
"""Compare two methods in an immutable paper-native metric JSONL stream."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

from mosaickv.evaluation.metric_export import cider_samples_from_results
from mosaickv.evaluation.metrics.comparison import compare_paired_methods
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.storage import load_metric_jsonl
from mosaickv.evaluation.metrics.types import MetricObservation, MetricScope
from mosaickv.evaluation.milebench import (
    milebench_samples_from_observations,
    paired_milebench_bootstrap,
)
from mosaickv.evaluation.paper_native_lanes import (
    cache_operating_point,
    validate_byte_matched_operating_points,
)
from mosaickv.evaluation.results import EvaluationResult
from mosaickv.evaluation.storage import load_jsonl
from mosaickv.types import JsonValue


def _cohort(
    rows: tuple[MetricObservation, ...],
    *,
    task: str,
    model: str,
    backend: str,
    method: str,
    metric: str,
    retention_ratio: float,
) -> tuple[MetricObservation, ...]:
    registry = default_metric_registry()
    canonical_metric = registry.canonical_name(metric)
    return tuple(
        row
        for row in rows
        if row.task == task
        and row.model == model
        and row.backend == backend
        and row.method == method
        and registry.canonical_name(row.metric_name) == canonical_metric
        and row.retention_ratio == retention_ratio
    )


def _method_rows(
    rows: tuple[MetricObservation, ...],
    *,
    task: str,
    model: str,
    backend: str,
    method: str,
    retention_ratio: float,
) -> tuple[MetricObservation, ...]:
    return tuple(
        row
        for row in rows
        if row.task == task
        and row.model == model
        and row.backend == backend
        and row.method == method
        and row.retention_ratio == retention_ratio
        and row.metric_scope is MetricScope.SAMPLE
        and row.metric_name in {"accuracy", "rouge_l_f1"}
    )


def _raw_cohort(
    rows: tuple[EvaluationResult, ...],
    *,
    task: str,
    model: str,
    backend: str,
    method: str,
    retention_ratio: float,
) -> tuple[EvaluationResult, ...]:
    return tuple(
        row
        for row in rows
        if row.task == task
        and row.model == model
        and row.backend == backend
        and row.method == method
        and row.retention_ratio == retention_ratio
    )


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
            encoding="utf-8",
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _cache_payload(
    raw_rows: tuple[EvaluationResult, ...],
    *,
    task: str,
    model: str,
    backend: str,
    reference_method: str,
    candidate_method: str,
    reference_ratio: float,
    candidate_ratio: float,
    require_byte_match: bool,
    byte_match_tolerance: float,
) -> dict[str, object] | None:
    if not raw_rows:
        if require_byte_match:
            raise ValueError("--require-byte-match requires --raw-results")
        return None
    reference_rows = _raw_cohort(
        raw_rows,
        task=task,
        model=model,
        backend=backend,
        method=reference_method,
        retention_ratio=reference_ratio,
    )
    candidate_rows = _raw_cohort(
        raw_rows,
        task=task,
        model=model,
        backend=backend,
        method=candidate_method,
        retention_ratio=candidate_ratio,
    )
    if not reference_rows or not candidate_rows:
        raise ValueError(
            "raw results lack the reference or candidate cohort needed for byte accounting"
        )
    if require_byte_match:
        reference_point, candidate_point = validate_byte_matched_operating_points(
            reference_rows,
            candidate_rows,
            maximum_relative_difference=byte_match_tolerance,
        )
    else:
        reference_point = cache_operating_point(reference_rows)
        candidate_point = cache_operating_point(candidate_rows)
    relative_mean_difference = (
        candidate_point.mean_active_kv_bytes - reference_point.mean_active_kv_bytes
    ) / reference_point.mean_active_kv_bytes
    return {
        "reference": reference_point.to_json_object(),
        "candidate": candidate_point.to_json_object(),
        "candidate_minus_reference_mean_active_bytes_fraction": relative_mean_difference,
        "byte_match_required": require_byte_match,
        "byte_match_tolerance": byte_match_tolerance if require_byte_match else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--raw-results",
        type=Path,
        action="append",
        default=[],
        help="raw result JSONL input; required for paired CIDEr",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--reference-method", default="full_kv")
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--reference-ratio", type=float, default=1.0)
    parser.add_argument("--candidate-ratio", type=float, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--require-byte-match",
        action="store_true",
        help="reject paired samples whose measured active K/V bytes differ beyond tolerance",
    )
    parser.add_argument("--byte-match-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    rows = load_metric_jsonl(args.metrics)
    raw_rows = tuple(row for path in args.raw_results for row in load_jsonl(path))
    cache_payload = _cache_payload(
        raw_rows,
        task=args.task,
        model=args.model,
        backend=args.backend,
        reference_method=args.reference_method,
        candidate_method=args.candidate_method,
        reference_ratio=args.reference_ratio,
        candidate_ratio=args.candidate_ratio,
        require_byte_match=args.require_byte_match,
        byte_match_tolerance=args.byte_match_tolerance,
    )
    identity = {
        "task": args.task,
        "model": args.model,
        "backend": args.backend,
        "metric": args.metric,
    }
    if args.metric == "milebench":
        reference_rows = _method_rows(
            rows,
            task=args.task,
            model=args.model,
            backend=args.backend,
            method=args.reference_method,
            retention_ratio=args.reference_ratio,
        )
        candidate_rows = _method_rows(
            rows,
            task=args.task,
            model=args.model,
            backend=args.backend,
            method=args.candidate_method,
            retention_ratio=args.candidate_ratio,
        )
        comparisons = paired_milebench_bootstrap(
            milebench_samples_from_observations(reference_rows),
            milebench_samples_from_observations(candidate_rows),
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            confidence=args.confidence,
        )
        payload = {
            "task": args.task,
            "model": args.model,
            "backend": args.backend,
            "reference_method": args.reference_method,
            "candidate_method": args.candidate_method,
            "metric_name": "milebench_hierarchy",
            "comparisons": {key: asdict(comparison) for key, comparison in comparisons.items()},
            "input_metrics": str(args.metrics.resolve()),
            "cache_operating_points": cache_payload,
        }
        _write_once(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    reference = _cohort(
        rows,
        **identity,
        method=args.reference_method,
        retention_ratio=args.reference_ratio,
    )
    candidate = _cohort(
        rows,
        **identity,
        method=args.candidate_method,
        retention_ratio=args.candidate_ratio,
    )
    if not reference or not candidate:
        raise ValueError(
            f"missing metric cohort: reference={len(reference)}, candidate={len(candidate)}"
        )
    reference_samples = None
    candidate_samples = None
    if default_metric_registry().canonical_name(args.metric) == "cider":
        if not args.raw_results:
            raise ValueError("paired CIDEr requires at least one --raw-results JSONL")
        reference_samples = cider_samples_from_results(
            _raw_cohort(
                raw_rows,
                task=args.task,
                model=args.model,
                backend=args.backend,
                method=args.reference_method,
                retention_ratio=args.reference_ratio,
            )
        )
        candidate_samples = cider_samples_from_results(
            _raw_cohort(
                raw_rows,
                task=args.task,
                model=args.model,
                backend=args.backend,
                method=args.candidate_method,
                retention_ratio=args.candidate_ratio,
            )
        )
    comparison = compare_paired_methods(
        reference,
        candidate,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        confidence=args.confidence,
        reference_samples=reference_samples,
        candidate_samples=candidate_samples,
    )
    payload = comparison.to_json_object()
    payload["input_metrics"] = str(args.metrics.resolve())
    payload["cache_operating_points"] = cast("JsonValue", cache_payload)
    _write_once(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
