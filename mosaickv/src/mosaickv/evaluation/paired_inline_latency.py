"""Paired engineering latency, never a two-image benchmark-superiority test."""

import numpy as np


def validate_rows(rows, *, warmups=5, repetitions=20):
    arms = ("original_probe", "inline")
    expected = {
        (phase, trial, arm)
        for phase, count in (("warmup", warmups), ("timed", repetitions))
        for trial in range(count)
        for arm in arms
    }
    keyed = {(r["phase"], r["trial"], r["arm"]): r for r in rows}
    if len(rows) != len(expected) or len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError("complete unique warmup and timed pairs required")
    if len({r["sample_id"] for r in rows}) != 1 or any(
        r["status"] != "completed" or r.get("all_128_logits_bitwise") is not True for r in rows
    ):
        raise ValueError("one identical sample and complete bitwise predictions required")
    for phase, count in (("warmup", warmups), ("timed", repetitions)):
        for trial in range(count):
            a, b = (keyed[phase, trial, arm] for arm in arms)
            if a["method_order"] != b["method_order"] or sorted(a["method_order"]) != sorted(arms):
                raise ValueError("recorded pair order is inconsistent")
            for field in (
                "initial_encoded_bytes",
                "final_encoded_bytes",
                "reported_counter_peak_bytes",
            ):
                if (
                    type(a["detail"][field]) is not int
                    or a["detail"][field] < 1
                    or a["detail"][field] != b["detail"][field]
                ):
                    raise ValueError("measured storage boundaries differ")
    return {arm: [keyed["timed", trial, arm] for trial in range(repetitions)] for arm in arms}


def summarize(old, new, *, seed, replicates=10000):
    a, b = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or len(a) < 20
        or not (np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all())
    ):
        raise ValueError("at least twenty finite positive paired trials required")
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(a), size=(replicates, len(a)))
    ratios = b[ids].mean(1) / a[ids].mean(1)
    low, high = np.quantile(ratios, (0.025, 0.975))

    def stats(t):
        return dict(
            mean=float(t.mean()),
            std=float(t.std(ddof=1)),
            p5=float(np.quantile(t, 0.05)),
            p50=float(np.median(t)),
            p95=float(np.quantile(t, 0.95)),
        )

    return dict(
        original=stats(a),
        inline=stats(b),
        paired_mean_ratio=float(b.mean() / a.mean()),
        paired_ratio_interval=[float(low), float(high)],
        adjusted_one_sided_upper_ratio=float(high),
        two_input_familywise_alpha=0.05,
        adjustment="Bonferroni, one-sided alpha .025 per input",
        paired_latency_engineering_gate=bool(high <= 1.0),
        adjusted_four_percent_faster_gate=bool(high <= 0.96),
        paired_mean_difference_seconds=float((b - a).mean()),
        paired_difference_interval_seconds=np.quantile(
            (b[ids] - a[ids]).mean(1), (0.025, 0.975)
        ).tolist(),
        bootstrap_units="paired_trials_on_a_fixed_consumed_input_not_independent_examples",
        superiority_over_other_baselines_established=False,
    )
