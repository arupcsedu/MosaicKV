"""Translate legacy evaluation rows into the separate paper-native metric stream."""

from __future__ import annotations

import json
from collections.abc import Sequence

from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.types import MetricObservation, MetricSample, MetricScope
from mosaickv.evaluation.results import EvaluationResult, ResultStatus
from mosaickv.evaluation.tasks import TaskSpec
from mosaickv.types import ResultCategory, ResultOrigin

_LMMS_EVALUATOR_ID = "lmms-eval==0.7.2"


def _references(serialized: str) -> tuple[str, ...]:
    try:
        value = json.loads(serialized)
    except json.JSONDecodeError:
        return (serialized,)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return tuple(value)
    if isinstance(value, str):
        return (value,)
    return (serialized,)


def _sample_observation(
    row: EvaluationResult,
    *,
    metric_name: str,
    manifest_path: str,
    result_category: ResultCategory,
) -> MetricObservation:
    if row.task_score is None:
        raise ValueError("sample metric requires task_score")
    definition = default_metric_registry().resolve(metric_name)
    value = row.task_score * 100.0 if metric_name == "mathvista_accuracy" else row.task_score
    return MetricObservation(
        run_id=row.run_id,
        sample_id=row.sample_id,
        task=row.task,
        model=row.model,
        backend=row.backend,
        method=row.method,
        retention_ratio=row.retention_ratio,
        metric_name=definition.name,
        metric_value=value,
        metric_direction=definition.direction,
        metric_scope=MetricScope.SAMPLE,
        metric_implementation=definition.implementation,
        metric_version=definition.version,
        metric_unit=definition.unit,
        measurement_type=(
            "paper_reproduction_metric_measured"
            if result_category is ResultCategory.PAPER_REPRODUCTION
            else "unified_comparison_metric_measured"
        ),
        manifest_path=manifest_path,
        evaluator_commit=_LMMS_EVALUATOR_ID,
        reference_artifact_sha="not_used",
        result_category=result_category,
        result_origin=(
            ResultOrigin.LOCALLY_REPRODUCED
            if result_category is ResultCategory.PAPER_REPRODUCTION
            else ResultOrigin.LOCALLY_MEASURED
        ),
    )


def _cider_observation(
    rows: Sequence[EvaluationResult],
    *,
    manifest_path: str,
    result_category: ResultCategory,
) -> MetricObservation:
    completed = [
        row for row in rows if row.status is ResultStatus.COMPLETED and row.answer is not None
    ]
    if len(completed) != len(rows):
        raise ValueError("CIDEr corpus aggregation requires every selected sample to complete")
    identities = {(row.run_id, row.task, row.model, row.backend, row.method) for row in completed}
    retentions = {row.retention_ratio for row in completed}
    if len(identities) != 1 or len(retentions) != 1:
        raise ValueError(
            "CIDEr cannot aggregate mixed run/task/model/backend/effective-method rows"
        )
    samples = cider_samples_from_results(completed)
    definition = default_metric_registry().resolve("cider")
    run_id, task, model, backend, method = next(iter(identities))
    return MetricObservation(
        run_id=run_id,
        sample_id=None,
        task=task,
        model=model,
        backend=backend,
        method=method,
        retention_ratio=next(iter(retentions)),
        metric_name=definition.name,
        metric_value=definition.aggregate(samples),
        metric_direction=definition.direction,
        metric_scope=MetricScope.CORPUS,
        metric_implementation=definition.implementation,
        metric_version=definition.version,
        metric_unit=definition.unit,
        measurement_type=(
            "paper_reproduction_metric_measured"
            if result_category is ResultCategory.PAPER_REPRODUCTION
            else "unified_comparison_metric_measured"
        ),
        manifest_path=manifest_path,
        evaluator_commit=_LMMS_EVALUATOR_ID,
        reference_artifact_sha="not_used",
        sample_count=len(samples),
        result_category=result_category,
        result_origin=(
            ResultOrigin.LOCALLY_REPRODUCED
            if result_category is ResultCategory.PAPER_REPRODUCTION
            else ResultOrigin.LOCALLY_MEASURED
        ),
    )


def cider_samples_from_results(
    rows: Sequence[EvaluationResult],
) -> tuple[MetricSample, ...]:
    """Build raw CIDEr contributions without copying captions into metric rows."""

    if not rows:
        raise ValueError("CIDEr requires at least one raw result")
    if any(row.status is not ResultStatus.COMPLETED or row.answer is None for row in rows):
        raise ValueError("CIDEr raw contributions require completed answers")
    samples = tuple(
        MetricSample(
            sample_id=row.sample_id,
            prediction=str(row.answer),
            references=_references(row.reference),
        )
        for row in rows
    )
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("CIDEr raw contributions require unique sample IDs")
    return samples


def export_task_metrics(
    rows: Sequence[EvaluationResult],
    *,
    specs: Sequence[TaskSpec],
    manifest_path: str,
    result_category: ResultCategory = ResultCategory.UNIFIED_COMPARISON,
) -> tuple[tuple[MetricObservation, ...], list[str]]:
    """Export only metrics derivable without inventing missing observations."""

    expected_origin = (
        ResultOrigin.LOCALLY_REPRODUCED
        if result_category is ResultCategory.PAPER_REPRODUCTION
        else ResultOrigin.LOCALLY_MEASURED
    )
    invalid_rows = [
        row.sample_id
        for row in rows
        if (row.result_category, row.result_origin) != (result_category, expected_origin)
    ]
    if invalid_rows:
        raise ValueError(
            "raw result category/origin does not match metric export request: "
            + ", ".join(sorted(invalid_rows))
        )
    spec_by_name = {spec.name: spec for spec in specs}
    rows_by_task: dict[str, list[EvaluationResult]] = {}
    for row in rows:
        if row.task in spec_by_name:
            rows_by_task.setdefault(row.task, []).append(row)
    observations: list[MetricObservation] = []
    errors: list[str] = []
    for task, task_rows in sorted(rows_by_task.items()):
        spec = spec_by_name[task]
        for metric_name in spec.paper_metrics:
            if metric_name == "cider":
                try:
                    observations.append(
                        _cider_observation(
                            task_rows,
                            manifest_path=manifest_path,
                            result_category=result_category,
                        )
                    )
                except (RuntimeError, ValueError) as error:
                    errors.append(f"{task}/cider: {type(error).__name__}: {error}")
                continue
            if metric_name in {"teacher_forced_ppl", "perplexity", "rouge_l_f1"}:
                requirement = "requires the PrefixKV teacher-forcing/reference protocol"
                errors.append(f"{task}/{metric_name}: {requirement}")
                continue
            for row in task_rows:
                if row.status is not ResultStatus.COMPLETED or row.task_score is None:
                    continue
                observations.append(
                    _sample_observation(
                        row,
                        metric_name=metric_name,
                        manifest_path=manifest_path,
                        result_category=result_category,
                    )
                )
    return tuple(observations), errors


__all__ = ["cider_samples_from_results", "export_task_metrics"]
