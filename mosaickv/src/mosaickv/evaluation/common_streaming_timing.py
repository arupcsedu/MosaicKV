"""Randomized, same-engine complete requests for frozen precision variants.

These are consumed, different-encoded-byte operating points, not new algorithm
curves or independent paper evidence. Every method gets streaming construction,
lossless history and a single layer lease. No benchmark labels are inputs.
"""

import gc
import random
import time
from dataclasses import replace

import numpy as np
import torch

METHODS = (
    "mosaickv_exact",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
)


def order(seed):
    rng = random.Random(seed)
    for trial in range(-5, 20):
        methods = list(METHODS)
        rng.shuffle(methods)
        for position, method in enumerate(methods):
            yield dict(trial=trial, warmup=trial < 0, order_index=position, method=method)


@torch.inference_mode()
def request(adapter, messages, profile, method, *, minimum_only=False):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.storage_runs import run
    from mosaickv.evaluation.streaming_baselines import SelectedRetiringCache, StreamingBaseline
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention
    from mosaickv.evaluation.streaming_source_constructor import StreamingDirectSource

    if method not in METHODS:
        raise ValueError("five frozen local precision variants only")
    torch.cuda.synchronize()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    started = time.monotonic()
    begin.record()
    prepared = adapter.prepare_inputs(messages)
    prepared_at = time.monotonic()
    cache = (
        RetiringPrefillCache(len(profile), audit=False)
        if method == "mosaickv_exact"
        else SelectedRetiringCache(len(profile), audit=False)
    )
    encoder = (
        StreamingDirectSource(adapter, prepared, cache, profile, minimum_only=minimum_only)
        if method == "mosaickv_exact"
        else StreamingBaseline(prepared, cache, profile, method, minimum_only=minimum_only)
    )
    fresh = replace(prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache))
    with SealAfterAttention(adapter, cache, encoder):
        pref = adapter.prefill(fresh, capture_queries=False, capture_attentions=False)
    encoder.finish() if method == "mosaickv_exact" else encoder.finish(adapter)
    packets = cache.formats()
    runtime = replace(pref, state=replace(pref.state, past_key_values=None), attention_weights=())
    prepared.model_inputs.clear()
    del pref, fresh, cache, encoder, prepared
    gc.collect()
    torch.cuda.synchronize()
    constructed_at = time.monotonic()
    values, details = run(
        adapter,
        runtime,
        packets,
        tokens=128,
        audit=False,
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        source_history_constructor="direct_batched_checks",
        consolidated_history=True,
    )
    end.record()
    torch.cuda.synchronize()
    return (
        values,
        packets,
        dict(
            total_wall_seconds=time.monotonic() - started,
            total_cuda_ms=begin.elapsed_time(end),
            preprocessing_wall_seconds=prepared_at - started,
            prefill_construct_retire_wall_seconds=constructed_at - prepared_at,
            decode_wall_seconds=details["decode_wall_seconds"],
            decode_cuda_ms=details["decode_cuda_ms"],
            initial_encoded_bytes=backing_bytes(tuple(t for p in packets for t in p.tensors())),
            tensor_result_auditing=False,
            memory_counters_not_peak_evidence=True,
            one_prefill_per_request=True,
            full_image_and_language_prefill_included=True,
            warm_cache_PIL_media_loading_excluded=True,
            common_gc_and_synchronization_included=True,
            prediction_hash_and_score_evaluation_excluded=True,
        ),
    )


def summarize(rows, seed):
    expected = {(r["trial"], r["method"]) for r in order(seed)}
    keys = [(r["trial"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("complete unique five-warmup/twenty-trial five-method requests required")
    if any(not r["logits_bitwise"] or r["tensor_result_auditing"] for r in rows):
        raise ValueError("all requests must retain bitwise parent logits without timing audits")
    by_key = {(r["trial"], r["method"]): r for r in rows}
    arrays = {
        method: np.asarray([by_key[i, method]["total_wall_seconds"] for i in range(20)])
        for method in METHODS
    }
    if any(not np.isfinite(a).all() or not (a > 0).all() for a in arrays.values()):
        raise ValueError("finite synchronized complete-request time required")
    ids = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    candidate = arrays["mosaickv_exact"]
    # Eight comparisons: four baselines times the two consumed inputs. These
    # are engineering trial intervals, NOT intervals over independent images.
    tail = 0.05 / (2 * 8)
    pairs = {}
    for method, other in arrays.items():
        if method == "mosaickv_exact":
            continue
        ratios = candidate[ids].mean(1) / other[ids].mean(1)
        bounds = np.quantile(ratios, [tail, 1 - tail]).tolist()
        pairs[method] = dict(
            mean_ratio=float(candidate.mean() / other.mean()),
            paired_ratio_CI95=np.quantile(ratios, [0.025, 0.975]).tolist(),
            familywise_eight_comparison_ratio_bounds=bounds,
            paired_difference_CI95=np.quantile(
                (candidate[ids] - other[ids]).mean(1), [0.025, 0.975]
            ).tolist(),
            similar_time_gate_upper_ratio_1_05=bool(bounds[1] <= 1.05),
        )
    return dict(
        timed_trials_per_method=20,
        warmups_per_method=5,
        distributions={
            method: dict(
                mean=float(a.mean()),
                p5=float(np.quantile(a, 0.05)),
                p50=float(np.median(a)),
                p95=float(np.quantile(a, 0.95)),
                standard_deviation=float(a.std(ddof=1)),
            )
            for method, a in arrays.items()
        },
        pairs=pairs,
        multiple_testing="Bonferroni eight engineering comparisons",
        independent_quality_N=0,
        equal_encoded_byte_comparison=False,
        paper_superiority_established=False,
    )
