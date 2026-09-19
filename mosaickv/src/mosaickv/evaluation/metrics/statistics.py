"""Metric-aware paired bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from mosaickv.evaluation.metrics.registry import MetricDefinition
from mosaickv.evaluation.metrics.types import MetricSample


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    location = probability * (len(sorted_values) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return sorted_values[lower]
    fraction = location - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


@dataclass(frozen=True, slots=True)
class PairedMetricComparison:
    metric_name: str
    reference_value: float
    candidate_value: float
    candidate_minus_reference: float
    ci_low: float
    ci_high: float
    confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int
    sample_count: int


def paired_bootstrap(
    definition: MetricDefinition,
    reference: Sequence[MetricSample],
    candidate: Sequence[MetricSample],
    *,
    resamples: int = 2000,
    seed: int = 2027,
    confidence: float = 0.95,
) -> PairedMetricComparison:
    """Compare paired cohorts while rerunning the metric's corpus reducer."""

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    reference_by_id = {sample.sample_id: sample for sample in reference}
    candidate_by_id = {sample.sample_id: sample for sample in candidate}
    if len(reference_by_id) != len(reference) or len(candidate_by_id) != len(candidate):
        raise ValueError("paired bootstrap sample IDs must be unique")
    if reference_by_id.keys() != candidate_by_id.keys():
        raise ValueError("reference and candidate must contain identical sample IDs")
    identifiers = tuple(sorted(reference_by_id))
    if len(identifiers) < 2:
        raise ValueError("paired bootstrap requires at least two samples")

    reference_value = definition.aggregate([reference_by_id[key] for key in identifiers])
    candidate_value = definition.aggregate([candidate_by_id[key] for key in identifiers])
    generator = random.Random(seed)
    differences: list[float] = []
    for _index in range(resamples):
        indices = [generator.randrange(len(identifiers)) for _ in identifiers]
        reference_draw = [reference_by_id[identifiers[index]] for index in indices]
        candidate_draw = [candidate_by_id[identifiers[index]] for index in indices]
        differences.append(
            definition.aggregate(candidate_draw) - definition.aggregate(reference_draw)
        )
    differences.sort()
    tail = (1.0 - confidence) / 2.0
    return PairedMetricComparison(
        metric_name=definition.name,
        reference_value=reference_value,
        candidate_value=candidate_value,
        candidate_minus_reference=candidate_value - reference_value,
        ci_low=_percentile(differences, tail),
        ci_high=_percentile(differences, 1.0 - tail),
        confidence=confidence,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
        sample_count=len(identifiers),
    )


__all__ = ["PairedMetricComparison", "paired_bootstrap"]
