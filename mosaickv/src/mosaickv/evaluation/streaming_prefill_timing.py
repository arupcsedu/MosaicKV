"""Own-parent paired end-to-end source-variant timing, no task labels.

Use the minimal equivalent window/prior capture in the offline control. Neither
arm builds discarded codecs, saves predictions inside timing, or audits tensor
results. Memory evidence must come from the separate audited canary.
"""

import gc
import random
import time
from dataclasses import replace

import numpy as np
import torch


def order(seed):
    rng = random.Random(seed)
    for trial in range(-5, 20):
        variants = ["offline", "streaming"]
        rng.shuffle(variants)
        for index, variant in enumerate(variants):
            yield dict(trial=trial, warmup=trial < 0, order_index=index, variant=variant)


@torch.inference_mode()
def request(adapter, messages, profile, cap, variant):
    from mosaickv.evaluation.direct_source_constructor import DirectSourcePolicy, construct
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.evaluation.storage_runs import run
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention
    from mosaickv.evaluation.streaming_source_constructor import StreamingDirectSource
    from mosaickv.evaluation.text_source_storage import TextSourceCapture

    if variant not in ("offline", "streaming"):
        raise ValueError("only the frozen offline/streaming source execution policies permitted")
    torch.cuda.synchronize()
    starts, ends = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begun = time.monotonic()
    starts.record()
    prepared = adapter.prepare_inputs(messages)
    prepared_at = time.monotonic()
    if variant == "offline":
        with (
            SourceCapture(adapter) as source,
            TextSourceCapture(adapter, prepared) as text,
            PhaseCapture(adapter) as phase,
            PromptQueryCapture(adapter, summaries=True, normalized=False, prior=True) as queries,
        ):
            pref = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
        pref = replace(
            pref, attention_weights=tuple(queries.windows[i] for i in range(len(profile)))
        )
        formats, _ = construct(
            adapter,
            prepared,
            pref,
            source,
            text,
            phase,
            profile,
            cap,
            policy=DirectSourcePolicy(head_group=4, lossless_minimum=False),
            priors=queries.priors,
        )
        runtime = replace(
            pref, state=replace(pref.state, past_key_values=None), attention_weights=()
        )
        source.records.clear()
        text.records.clear()
        queries.windows.clear()
        queries.priors.clear()
        del pref, source, text, phase, queries
    else:
        cache = RetiringPrefillCache(len(profile), audit=False)
        encoder = StreamingDirectSource(adapter, prepared, cache, profile)
        fresh = replace(prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache))
        with SealAfterAttention(adapter, cache, encoder):
            pref = adapter.prefill(fresh, capture_queries=False, capture_attentions=False)
        encoder.finish()
        formats = cache.formats()
        runtime = replace(
            pref, state=replace(pref.state, past_key_values=None), attention_weights=()
        )
        del pref, fresh, cache, encoder
    prepared.model_inputs.clear()
    del prepared
    gc.collect()
    torch.cuda.synchronize()
    constructed_at = time.monotonic()
    logits, details = run(
        adapter,
        runtime,
        formats,
        tokens=128,
        audit=False,
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        source_history_constructor="direct_batched_checks",
        consolidated_history=True,
    )
    ends.record()
    torch.cuda.synchronize()
    record = dict(
        total_wall_seconds=time.monotonic() - begun,
        total_cuda_ms=starts.elapsed_time(ends),
        preprocessing_wall_seconds=prepared_at - begun,
        prefill_construct_retire_wall_seconds=constructed_at - prepared_at,
        decode_wall_seconds=details["decode_wall_seconds"],
        decode_cuda_ms=details["decode_cuda_ms"],
        initial_encoded_bytes=backing_bytes(tuple(t for p in formats for t in p.tensors())),
        tensor_result_auditing=False,
        memory_counters_not_peak_evidence=True,
        full_image_and_language_prefill_included=True,
        one_prefill_per_request=True,
        warm_cache_PIL_media_loading_excluded=True,
        common_gc_and_synchronization_included=True,
        prediction_hash_and_score_evaluation_excluded=True,
    )
    return logits, record, formats


def summary(rows, *, seed):
    keys = [(r["trial"], r["variant"]) for r in rows]
    expected = {(r["trial"], r["variant"]) for r in order(seed)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("complete unique five-warmup/twenty-trial pairs required")
    if any(not r["logits_bitwise"] or r["tensor_result_auditing"] for r in rows):
        raise ValueError("every unaudited timing must retain the same 128-logit trajectory")
    by_key = {(r["trial"], r["variant"]): r for r in rows}
    arrays = {
        name: np.asarray([by_key[i, name]["total_wall_seconds"] for i in range(20)], dtype=float)
        for name in ("offline", "streaming")
    }
    if any(not np.isfinite(a).all() or not (a > 0).all() for a in arrays.values()):
        raise ValueError("positive finite synchronized total timings required")
    old, new = arrays["offline"], arrays["streaming"]
    ids = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    ratio = new[ids].mean(1) / old[ids].mean(1)
    return dict(
        timed_pairs=20,
        warmup_pairs=5,
        distributions={
            k: dict(
                mean=float(a.mean()),
                standard_deviation=float(a.std(ddof=1)),
                p5=float(np.quantile(a, 0.05)),
                p50=float(np.median(a)),
                p95=float(np.quantile(a, 0.95)),
            )
            for k, a in arrays.items()
        },
        mean_time_ratio=float(new.mean() / old.mean()),
        paired_ratio_CI95=np.quantile(ratio, [0.025, 0.975]).tolist(),
        paired_difference_CI95=np.quantile((new[ids] - old[ids]).mean(1), [0.025, 0.975]).tolist(),
        similar_time_gate_upper_ratio_1_05=bool(np.quantile(ratio, 0.975) <= 1.05),
        independent_quality_N=0,
        baseline_comparison=False,
    )
