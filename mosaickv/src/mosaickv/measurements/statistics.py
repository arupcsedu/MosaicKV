"""Deterministic distribution summaries for repeated measurements."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Sequence

from mosaickv.measurements.types import (
    FullKVAggregate,
    FullKVTrialMeasurement,
    SummaryStatistics,
)


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile for finite observations."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("statistics require finite values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(
    values: Sequence[float | int],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> SummaryStatistics:
    """Summarize values and bootstrap a percentile interval for their mean."""

    if not values:
        raise ValueError("summary requires at least one value")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    observations = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in observations):
        raise ValueError("measurement values must be finite and nonnegative")
    rng = random.Random(seed)
    means = []
    count = len(observations)
    for _index in range(bootstrap_samples):
        means.append(statistics.fmean(observations[rng.randrange(count)] for _ in range(count)))
    tail = (1.0 - confidence_level) / 2.0
    return SummaryStatistics(
        count=count,
        median=statistics.median(observations),
        p5=percentile(observations, 0.05),
        p95=percentile(observations, 0.95),
        mean=statistics.fmean(observations),
        standard_deviation=statistics.pstdev(observations),
        bootstrap_mean_ci_low=percentile(means, tail),
        bootstrap_mean_ci_high=percentile(means, 1.0 - tail),
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
    )


def aggregate_trials(
    run_id: str,
    trials: Iterable[FullKVTrialMeasurement],
    *,
    warmups: int,
    repeated_trials: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> FullKVAggregate:
    """Derive statistics while retaining every raw trial key."""

    rows = tuple(trials)
    completed = tuple(row for row in rows if row.status == "completed")
    metrics: dict[str, list[float | int]] = {
        "language_model_prefill_seconds": [],
        "compression_seconds": [],
        "ttft_seconds": [],
        "decode_total_seconds": [],
        "repair_seconds": [],
        "total_latency_seconds": [],
        "host_total_latency_seconds": [],
        "max_memory_allocated_bytes": [],
        "max_memory_reserved_bytes": [],
        "active_kv_bytes": [],
        "cpu_residual_bytes": [],
    }
    optional_metrics: dict[str, list[float]] = {
        "image_video_encoder_seconds": [],
        "projector_seconds": [],
    }
    decode_rows: list[tuple[float, ...]] = []
    for row in completed:
        if row.timings is None or row.memory is None:  # guarded by schema, helps type narrowing
            raise RuntimeError("completed trial is missing timing or memory measurements")
        timings = row.timings
        memory = row.memory
        metrics["language_model_prefill_seconds"].append(timings.language_model_prefill)
        metrics["compression_seconds"].append(timings.compression)
        metrics["ttft_seconds"].append(timings.ttft)
        metrics["decode_total_seconds"].append(timings.decode_total)
        metrics["repair_seconds"].append(timings.repair)
        metrics["total_latency_seconds"].append(timings.total_latency)
        metrics["host_total_latency_seconds"].append(timings.host_total_latency)
        metrics["max_memory_allocated_bytes"].append(memory.max_memory_allocated)
        metrics["max_memory_reserved_bytes"].append(memory.max_memory_reserved)
        metrics["active_kv_bytes"].append(memory.active_kv_bytes)
        metrics["cpu_residual_bytes"].append(memory.cpu_residual_bytes)
        if timings.image_video_encoder is not None:
            optional_metrics["image_video_encoder_seconds"].append(timings.image_video_encoder)
        if timings.projector is not None:
            optional_metrics["projector_seconds"].append(timings.projector)
        decode_rows.append(timings.per_token_decode)

    summaries = {
        name: summarize(
            values,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + index,
        )
        for index, (name, values) in enumerate(metrics.items())
        if values
    }
    for index, (name, values) in enumerate(optional_metrics.items(), start=len(summaries)):
        if values:
            summaries[name] = summarize(
                values,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed + index,
            )

    positions = max((len(row) for row in decode_rows), default=0)
    per_token = tuple(
        summarize(
            [row[position] for row in decode_rows if position < len(row)],
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 1000 + position,
        )
        for position in range(positions)
    )
    sequences_by_sample: dict[str, list[tuple[int, ...]]] = {}
    for row in completed:
        sequences_by_sample.setdefault(row.sample_id, []).append(row.generated_token_ids)
    comparable = bool(sequences_by_sample) and all(
        len(sequences) >= 2 for sequences in sequences_by_sample.values()
    )
    deterministic_match = (
        all(len(set(sequences)) == 1 for sequences in sequences_by_sample.values())
        if comparable
        else None
    )
    return FullKVAggregate(
        run_id=run_id,
        warmups=warmups,
        repeated_trials=repeated_trials,
        completed_trials=len(completed),
        failed_trials=len(rows) - len(completed),
        deterministic_token_match=deterministic_match,
        trial_keys=tuple(f"{row.sample_id}:{row.trial_index}" for row in rows),
        metrics=summaries,
        per_token_decode=per_token,
    )


__all__ = ["aggregate_trials", "percentile", "summarize"]
