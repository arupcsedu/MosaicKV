"""Paired consumed-input timing of two bit-identical history representations.

Prompt selection and formats are common. These are decode/storage-construction
timings, not whole-request latency or independently validated task quality.
"""

import gc
import math
import random
import time

import numpy as np
import torch

POLICIES = {
    "raw_history": dict(sealed=False, sequential_lease=True),
    "exact_source_history": dict(
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        source_history_constructor="direct_batched_checks",
        consolidated_history=True,
    ),
}


def trial_order(*, seed, warmups=5, repetitions=20):
    if (
        type(seed) is not int
        or type(warmups) is not int
        or warmups < 5
        or type(repetitions) is not int
        or repetitions < 20
    ):
        raise ValueError("integer seed, at least five warmups and twenty paired trials required")
    rng = random.Random(seed)
    for trial in range(-warmups, repetitions):
        order = list(POLICIES)
        rng.shuffle(order)
        for index, variant in enumerate(order):
            yield dict(trial=trial, warmup=trial < 0, order_index=index, variant=variant)


def measure(adapter, prefill, formats, reference, *, seed, record, runner=None):
    from mosaickv.evaluation.storage_runs import run

    execute = run if runner is None else runner
    if tuple(reference.shape)[0:1] != (128,) or reference.device.type != "cpu":
        raise ValueError("complete 128-prefix CPU reference required")
    rows = []
    for entry in trial_order(seed=seed):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begun = time.monotonic()
        logits, details = execute(
            adapter, prefill, formats, tokens=128, audit=False, **POLICIES[entry["variant"]]
        )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begun
        if not torch.equal(logits, reference):
            raise RuntimeError("timed history changed an original greedy logit bit")
        row = dict(
            **entry,
            decode_wall_seconds=details["decode_wall_seconds"],
            decode_cuda_ms=details["decode_cuda_ms"],
            private_cache_and_decode_wall_seconds=elapsed,
            torch_peak_allocated=torch.cuda.max_memory_allocated(),
            torch_peak_reserved=torch.cuda.max_memory_reserved(),
            logits_bitwise=True,
            tensor_result_auditing=False,
            storage_record_not_peak_evidence=True,
            whole_request_latency=False,
        )
        if any(
            not math.isfinite(row[k]) or row[k] <= 0
            for k in (
                "decode_wall_seconds", "decode_cuda_ms", "private_cache_and_decode_wall_seconds"
            )
        ):
            raise RuntimeError("invalid synchronized timing")
        record(row)
        rows.append(row)
    return rows


def summarize(rows, *, seed):
    """Within-input paired timing uncertainty, not a task-quality confidence bound."""
    expected = {(x["trial"], x["variant"]) for x in trial_order(seed=seed)}
    pairs = [(r["trial"], r["variant"]) for r in rows]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError("complete unique warmup and timed pairs required")
    if any(not r["logits_bitwise"] or r["tensor_result_auditing"] for r in rows):
        raise ValueError("only bit-identical unaudited timings are admissible")
    timed = {k: [] for k in POLICIES}
    by_pair = {(r["trial"], r["variant"]): r for r in rows}
    for trial in range(20):
        for name in POLICIES:
            value = by_pair[trial, name]["decode_wall_seconds"]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive decode time required")
            timed[name].append(value)
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in timed.items()}
    old, new = arrays["raw_history"], arrays["exact_source_history"]
    ids = np.random.default_rng(seed).integers(0, 20, size=(10000, 20))
    ratios = new[ids].mean(1) / old[ids].mean(1)
    differences = (new[ids] - old[ids]).mean(1)
    return dict(
        timed_pairs=20,
        warmup_pairs=5,
        summaries={
            k: dict(
                mean=float(a.mean()),
                standard_deviation=float(a.std(ddof=1)),
                p5=float(np.quantile(a, 0.05)),
                p50=float(np.median(a)),
                p95=float(np.quantile(a, 0.95)),
            )
            for k, a in arrays.items()
        },
        mean_paired_difference_seconds=float((new - old).mean()),
        mean_time_ratio=float(new.mean() / old.mean()),
        paired_difference_CI95=np.quantile(differences, [0.025, 0.975]).tolist(),
        paired_time_ratio_CI95=np.quantile(ratios, [0.025, 0.975]).tolist(),
        independent_quality_N=0,
        whole_request_latency=False,
    )
