"""Complete paired storage/hybrid trials; no benchmark-quality inference."""

import numpy as np

ARMS = ("storage_only", "codec_reference", "codec_fast", "hybrid_reference", "hybrid_fast")
BOUNDARIES = ("initial_encoded_bytes", "final_encoded_bytes", "reported_counter_peak_bytes")
COMPARISONS = (
    ("storage_only", "codec_fast"),
    ("storage_only", "hybrid_fast"),
    ("codec_fast", "hybrid_fast"),
    ("codec_reference", "codec_fast"),
    ("hybrid_reference", "hybrid_fast"),
)


def ordered_trials(rows, *, warmups=5, repetitions=20, arms=ARMS):
    if tuple(arms) not in (ARMS, ("storage_only", "codec_fast", "hybrid_fast")):
        raise ValueError("registered complete factorial or storage-controlled follow-up required")
    expected = {
        (phase, trial, arm)
        for phase, n in (("warmup", warmups), ("timed", repetitions))
        for trial in range(n)
        for arm in arms
    }
    keyed = {(r["phase"], r["trial"], r["arm"]): r for r in rows}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError("complete unique trials for all storage/hybrid arms required")
    if len({r["sample_id"] for r in rows}) != 1 or any(
        r["status"] != "completed" or r.get("all_128_expected_logits_bitwise") is not True
        for r in rows
    ):
        raise ValueError("one sample, bitwise model verification and no failures required")
    for (phase, trial, _), _row in keyed.items():
        group = [keyed[phase, trial, a] for a in arms]
        if any(sorted(r["method_order"]) != sorted(arms) for r in group) or any(
            r["method_order"] != group[0]["method_order"] for r in group
        ):
            raise ValueError("balanced recorded method order required")
        for r in group:
            if not np.isfinite(r["request_wall_seconds"]) or r["request_wall_seconds"] <= 0:
                raise ValueError("finite positive synchronized request times required")
            if any(type(r["detail"][k]) is not int or r["detail"][k] <= 0 for k in BOUNDARIES):
                raise ValueError("all measured live-storage boundaries required")
        for old, new in (("codec_reference", "codec_fast"),):
            if old not in arms:
                continue
            if any(
                keyed[phase, trial, old]["detail"][k] != keyed[phase, trial, new]["detail"][k]
                for k in BOUNDARIES
            ):
                raise ValueError("construction-only optimization changed storage")
    return {a: [keyed["timed", t, a] for t in range(repetitions)] for a in arms}


def paired_summary(old, new, *, seed, replicates=10000, comparisons=10):
    from mosaickv.evaluation.paired_inline_latency import summarize

    result = summarize(old, new, seed=seed, replicates=replicates)
    a, b = np.asarray(old), np.asarray(new)
    indices = np.random.default_rng(seed).integers(0, len(a), (replicates, len(a)))
    upper = float(np.quantile(b[indices].mean(1) / a[indices].mean(1), 1 - 0.05 / comparisons))
    result.update(
        adjusted_one_sided_upper_ratio=upper,
        adjustment=f"Bonferroni one-sided across {comparisons} registered input/comparison pairs",
        paired_latency_engineering_gate=upper <= 1.0,
        adjusted_four_percent_faster_gate=upper <= 0.96,
    )
    result.pop("two_input_familywise_alpha")
    return result
