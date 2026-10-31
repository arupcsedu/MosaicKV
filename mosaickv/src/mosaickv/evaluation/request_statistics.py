"""Paired complete-request summaries. No output scores enter a selector."""

from typing import Any

import numpy as np


def compare_requests(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]], *, resamples: int, seed: int
) -> dict[str, Any]:
    left = {r["sample_id"]: r for r in candidate}
    right = {r["sample_id"]: r for r in baseline}
    if (
        len(left) != len(candidate)
        or len(right) != len(baseline)
        or set(left) != set(right)
        or len(left) < 2
        or resamples < 1
    ):
        raise ValueError("complete unique paired request identities required")
    ids = sorted(left)
    draws = np.random.default_rng(seed).integers(0, len(ids), (resamples, len(ids)))

    def interval(values: Any) -> list[float]:
        return [float(v) for v in np.quantile(values, [0.025, 0.975])]

    c = np.asarray([left[i]["end_to_end_seconds"] for i in ids], dtype=float)
    b = np.asarray([right[i]["end_to_end_seconds"] for i in ids], dtype=float)
    if not np.isfinite(c).all() or not np.isfinite(b).all() or min(c.min(), b.min()) <= 0:
        raise ValueError("positive finite complete-request latency required")
    ratio = c.mean() / b.mean()
    ratio_ci = interval(c[draws].mean(1) / b[draws].mean(1))
    memory = {}
    for field in (
        "initial_active_kv_storage_bytes",
        "final_active_kv_storage_bytes",
        "peak_active_kv_storage_bytes",
    ):
        a = np.asarray([left[i]["storage"][field] for i in ids], dtype=np.int64)
        z = np.asarray([right[i]["storage"][field] for i in ids], dtype=np.int64)
        if min(a.min(), z.min()) <= 0:
            raise ValueError("positive measured backing storage required")
        memory[field] = dict(
            candidate_mean_bytes=float(a.mean()),
            baseline_mean_bytes=float(z.mean()),
            reduction_fraction=float(1 - a.mean() / z.mean()),
            all_inputs_smaller=bool((a < z).all()),
            smaller_input_count=int((a < z).sum()),
            paired_reduction_ci95=interval(1 - a[draws].mean(1) / z[draws].mean(1)),
        )
    a = np.asarray([left[i]["whole_request_peak_allocated_bytes"] for i in ids], dtype=float)
    z = np.asarray([right[i]["whole_request_peak_allocated_bytes"] for i in ids], dtype=float)
    return dict(
        unique_sample_count=len(ids),
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
