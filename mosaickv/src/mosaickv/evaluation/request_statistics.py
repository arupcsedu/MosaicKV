"""Paired complete-request summaries. No output scores enter a selector.

These ordinary 95% paired summaries are not a new paper-admission contract;
the existing comparison scope and no-slower threshold remain unchanged.
"""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

STORAGE_FIELDS = (
    "initial_active_kv_storage_bytes",
    "final_active_kv_storage_bytes",
    "peak_active_kv_storage_bytes",
)
MAXIMUM_REPRESENTABLE_BYTES = int(np.iinfo(np.int64).max)


def _indexed(rows: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, (list, tuple)):
        raise ValueError("complete unique paired request identities required")
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("explicit complete-request observation mappings required")
        sample = row.get("sample_id")
        if not isinstance(sample, str) or not sample.strip() or sample in indexed:
            raise ValueError("complete unique paired request identities required")
        indexed[sample] = row
    return indexed


def _seconds(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("positive finite numeric complete-request latency required")
    try:
        seconds = float(value)
    except (OverflowError, ValueError):
        raise ValueError("positive finite numeric complete-request latency required") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive finite numeric complete-request latency required")
    return seconds


def _bytes(value: Any) -> int:
    if type(value) is not int or not 0 < value <= MAXIMUM_REPRESENTABLE_BYTES:
        raise ValueError("positive plain-integer representable measured storage bytes required")
    return value


def _observations(
    rows: Mapping[str, Mapping[str, Any]], ids: list[str]
) -> tuple[Any, dict[str, Any], Any]:
    seconds: list[float] = []
    peak: list[int] = []
    storage: dict[str, list[int]] = {field: [] for field in STORAGE_FIELDS}
    for sample in ids:
        row = rows[sample]
        seconds.append(_seconds(row.get("end_to_end_seconds")))
        measured = row.get("storage")
        if not isinstance(measured, Mapping):
            raise ValueError("explicit measured backing-storage fieldmap required")
        for field in STORAGE_FIELDS:
            storage[field].append(_bytes(measured.get(field)))
        peak.append(_bytes(row.get("whole_request_peak_allocated_bytes")))
    return (
        np.asarray(seconds, dtype=float),
        {field: np.asarray(values, dtype=np.int64) for field, values in storage.items()},
        np.asarray(peak, dtype=np.int64),
    )


def compare_requests(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]], *, resamples: int, seed: int
) -> dict[str, Any]:
    """Validate raw observations before any numeric coercion or bootstrap draw."""
    if type(resamples) is not int or resamples < 1:
        raise ValueError("positive plain-integer bootstrap resamples required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative plain-integer bootstrap seed required")
    left, right = _indexed(candidate), _indexed(baseline)
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("complete unique paired request identities required")
    ids = sorted(left)
    c, candidate_storage, candidate_peak = _observations(left, ids)
    b, baseline_storage, baseline_peak = _observations(right, ids)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _summarize(
                c,
                b,
                candidate_storage,
                baseline_storage,
                candidate_peak,
                baseline_peak,
                resamples=resamples,
                seed=seed,
            )
    except FloatingPointError:
        raise ValueError("finite representable paired statistics required") from None


def _summarize(
    c: Any,
    b: Any,
    candidate_storage: Mapping[str, Any],
    baseline_storage: Mapping[str, Any],
    candidate_peak: Any,
    baseline_peak: Any,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    draws = np.random.default_rng(seed).integers(0, len(c), (resamples, len(c)))

    def interval(values: Any) -> list[float]:
        if not np.isfinite(values).all():
            raise ValueError("finite representable bootstrap distribution required")
        return [float(v) for v in np.quantile(values, [0.025, 0.975])]

    ratio = c.mean() / b.mean()
    ratio_ci = interval(c[draws].mean(1) / b[draws].mean(1))
    memory = {}
    for field in STORAGE_FIELDS:
        a, z = candidate_storage[field], baseline_storage[field]
        memory[field] = dict(
            candidate_mean_bytes=float(a.mean()),
            baseline_mean_bytes=float(z.mean()),
            reduction_fraction=float(1 - a.mean() / z.mean()),
            all_inputs_smaller=bool((a < z).all()),
            smaller_input_count=int((a < z).sum()),
            paired_reduction_ci95=interval(1 - a[draws].mean(1) / z[draws].mean(1)),
        )
    a, z = candidate_peak, baseline_peak
    return dict(
        unique_sample_count=len(c),
        mean_latency_ratio=float(ratio),
        paired_mean_latency_ratio_ci95=ratio_ci,
        paired_latency_delta_seconds=float((c - b).mean()),
        paired_latency_delta_ci95=interval((c - b)[draws].mean(1)),
        latency_no_slower=ratio_ci[1] <= 1,
        candidate_mean_e2e_seconds=float(c.mean()),
        baseline_mean_e2e_seconds=float(b.mean()),
        active_storage=memory,
        whole_request_allocated_peak=dict(
            candidate_mean_bytes=float(a.mean()),
            baseline_mean_bytes=float(z.mean()),
            paired_delta_ci95=interval((a - z)[draws].mean(1)),
            all_inputs_smaller=bool((a < z).all()),
        ),
        not_a_shared_encoded_byte_comparison=True,
        not_a_whole_request_KV_only_peak=True,
    )
