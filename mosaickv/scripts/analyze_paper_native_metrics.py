#!/usr/bin/env python3
"""Compare two methods in an immutable paper-native metric JSONL stream."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from mosaickv.evaluation.metrics.comparison import compare_paired_methods
from mosaickv.evaluation.metrics.storage import load_metric_jsonl
from mosaickv.evaluation.metrics.types import MetricObservation
from mosaickv.types import JsonObject


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
    return tuple(
        row
        for row in rows
        if row.task == task
        and row.model == model
        and row.backend == backend
        and row.method == method
        and row.metric_name == metric
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
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
    args = parser.parse_args()

    rows = load_metric_jsonl(args.metrics)
    identity = {
        "task": args.task,
        "model": args.model,
        "backend": args.backend,
        "metric": args.metric,
    }
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
    comparison = compare_paired_methods(
        reference,
        candidate,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    payload: JsonObject = comparison.to_json_object()
    payload["input_metrics"] = str(args.metrics.resolve())
    _write_once(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
