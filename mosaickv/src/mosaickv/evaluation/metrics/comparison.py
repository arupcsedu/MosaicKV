"""Fail-closed comparisons for paper-native metric streams."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import cast

from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.statistics import paired_bootstrap
from mosaickv.evaluation.metrics.types import MetricObservation, MetricSample, MetricScope
from mosaickv.types import JsonObject, ResultCategory, ResultOrigin


@dataclass(frozen=True, slots=True)
class MethodMetricSummary:
    """One method's reduced score with complete identity metadata."""

    task: str
    model: str
    backend: str
    method: str
    retention_ratio: float
    metric_name: str
    metric_value: float
    sample_count: int
    result_category: ResultCategory
    result_origin: ResultOrigin


@dataclass(frozen=True, slots=True)
class PairedMethodComparison:
    """Paired candidate-versus-reference comparison for one metric."""

    task: str
    model: str
    backend: str
    metric_name: str
    reference_method: str
    candidate_method: str
    reference_retention_ratio: float
    candidate_retention_ratio: float
    reference_value: float
    candidate_value: float
    candidate_minus_reference: float
    relative_difference: float | None
    ci_low: float
    ci_high: float
    relative_ci_low: float | None
    relative_ci_high: float | None
    confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int
    sample_count: int
    result_category: ResultCategory
    result_origin: ResultOrigin

    def to_json_object(self) -> JsonObject:
        payload = asdict(self)
        payload["result_category"] = self.result_category.value
        payload["result_origin"] = self.result_origin.value
        return cast("JsonObject", payload)


def _sample_contributions(rows: Sequence[MetricObservation]) -> tuple[MetricSample, ...]:
    contributions: list[MetricSample] = []
    for row in rows:
        if row.metric_scope is not MetricScope.SAMPLE or row.sample_id is None:
            continue
        contributions.append(
            MetricSample(
                sample_id=row.sample_id,
                prediction="",
                references=("metric-contribution",),
                value=row.metric_value,
                numerator=row.numerator,
                denominator=row.denominator,
            )
        )
    return tuple(contributions)


def _identity(
    rows: Sequence[MetricObservation],
) -> tuple[str, str, str, str, float, str, ResultCategory, ResultOrigin]:
    registry = default_metric_registry()
    identities = {
        (
            row.task,
            row.model,
            row.backend,
            row.method,
            row.retention_ratio,
            registry.canonical_name(row.metric_name),
            row.result_category,
            row.result_origin,
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError(
            "metric cohort must have one task/model/backend/method/ratio/metric/category/origin"
        )
    return next(iter(identities))


def summarize_metric(rows: Sequence[MetricObservation]) -> MethodMetricSummary:
    """Reduce sample contributions, checking any recorded corpus row for equality."""

    if not rows:
        raise ValueError("metric cohort cannot be empty")
    (
        task,
        model,
        backend,
        method,
        retention_ratio,
        metric_name,
        result_category,
        result_origin,
    ) = _identity(rows)
    definition = default_metric_registry().resolve(metric_name)
    samples = _sample_contributions(rows)
    corpus = [row for row in rows if row.metric_scope is MetricScope.CORPUS]
    if len(corpus) > 1:
        raise ValueError("metric cohort contains multiple corpus observations")
    if samples:
        value = definition.aggregate(samples)
        if corpus and abs(value - corpus[0].metric_value) > 1e-9:
            raise ValueError("corpus observation disagrees with sample contributions")
    elif corpus:
        value = corpus[0].metric_value
    else:
        raise ValueError("metric cohort contains neither sample nor corpus observations")
    sample_count = len(samples) if samples else corpus[0].sample_count
    return MethodMetricSummary(
        task=task,
        model=model,
        backend=backend,
        method=method,
        retention_ratio=retention_ratio,
        metric_name=metric_name,
        metric_value=value,
        sample_count=sample_count,
        result_category=result_category,
        result_origin=result_origin,
    )


def compare_paired_methods(
    reference: Sequence[MetricObservation],
    candidate: Sequence[MetricObservation],
    *,
    resamples: int = 10_000,
    seed: int = 2027,
    confidence: float = 0.95,
    reference_samples: Sequence[MetricSample] | None = None,
    candidate_samples: Sequence[MetricSample] | None = None,
) -> PairedMethodComparison:
    """Compute a paired interval without averaging incompatible metrics."""

    reference_summary = summarize_metric(reference)
    candidate_summary = summarize_metric(candidate)
    comparable_reference = (
        reference_summary.task,
        reference_summary.model,
        reference_summary.backend,
        reference_summary.metric_name,
        reference_summary.result_category,
        reference_summary.result_origin,
    )
    comparable_candidate = (
        candidate_summary.task,
        candidate_summary.model,
        candidate_summary.backend,
        candidate_summary.metric_name,
        candidate_summary.result_category,
        candidate_summary.result_origin,
    )
    if comparable_reference != comparable_candidate:
        raise ValueError(
            "paired metric cohorts differ in task, model, backend, metric, category, or origin"
        )
    selected_reference_samples = (
        tuple(reference_samples)
        if reference_samples is not None
        else _sample_contributions(reference)
    )
    selected_candidate_samples = (
        tuple(candidate_samples)
        if candidate_samples is not None
        else _sample_contributions(candidate)
    )
    if not selected_reference_samples or not selected_candidate_samples:
        raise ValueError("paired bootstrap requires sample-level sufficient statistics")
    if len(selected_reference_samples) != reference_summary.sample_count:
        raise ValueError("reference sample count disagrees with recorded metric cohort")
    if len(selected_candidate_samples) != candidate_summary.sample_count:
        raise ValueError("candidate sample count disagrees with recorded metric cohort")
    definition = default_metric_registry().resolve(reference_summary.metric_name)
    if reference_samples is not None:
        reduced = definition.aggregate(selected_reference_samples)
        if abs(reduced - reference_summary.metric_value) > 1e-9:
            raise ValueError("reference sample contributions disagree with corpus observation")
    if candidate_samples is not None:
        reduced = definition.aggregate(selected_candidate_samples)
        if abs(reduced - candidate_summary.metric_value) > 1e-9:
            raise ValueError("candidate sample contributions disagree with corpus observation")
    comparison = paired_bootstrap(
        definition,
        selected_reference_samples,
        selected_candidate_samples,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )
    return PairedMethodComparison(
        task=reference_summary.task,
        model=reference_summary.model,
        backend=reference_summary.backend,
        metric_name=reference_summary.metric_name,
        reference_method=reference_summary.method,
        candidate_method=candidate_summary.method,
        reference_retention_ratio=reference_summary.retention_ratio,
        candidate_retention_ratio=candidate_summary.retention_ratio,
        reference_value=comparison.reference_value,
        candidate_value=comparison.candidate_value,
        candidate_minus_reference=comparison.candidate_minus_reference,
        relative_difference=comparison.relative_difference,
        ci_low=comparison.ci_low,
        ci_high=comparison.ci_high,
        relative_ci_low=comparison.relative_ci_low,
        relative_ci_high=comparison.relative_ci_high,
        confidence=comparison.confidence,
        bootstrap_resamples=comparison.bootstrap_resamples,
        bootstrap_seed=comparison.bootstrap_seed,
        sample_count=comparison.sample_count,
        result_category=reference_summary.result_category,
        result_origin=reference_summary.result_origin,
    )


__all__ = [
    "MethodMetricSummary",
    "PairedMethodComparison",
    "compare_paired_methods",
    "summarize_metric",
]
