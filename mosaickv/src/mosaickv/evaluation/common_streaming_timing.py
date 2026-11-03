"""Randomized, same-engine complete requests for frozen precision variants.

These are consumed, different-encoded-byte operating points, not new algorithm
curves or independent paper evidence. Every method gets streaming construction,
lossless history and a single layer lease. No benchmark labels are inputs.
"""

import gc
import random
import time
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from unittest.mock import patch

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
def request(
    adapter,
    messages,
    profile,
    method,
    *,
    minimum_only=False,
    componentwise_fitting=False,
    code_delta=False,
    input_packet_validator=None,
    source_geometry=None,
    source_num_warps=None,
    construction_geometry=None,
    reconstruction_readers="original",
    reuse_kernel_dispatch=False,
    audit_decode_storage=False,
    reuse_code_descriptors=False,
    source_hotpath="original",
    full_prompt_native=None,
):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.native_execution_plan import use as native_plan
    from mosaickv.evaluation.reconstruction_policy import use, validate
    from mosaickv.evaluation.replay_descriptor_policy import use as use_descriptors
    from mosaickv.evaluation.request_kernel_dispatch import use_dispatch
    from mosaickv.evaluation.source_hotpath import use as source_plan
    from mosaickv.evaluation.source_hotpath import validate as validate_source_plan
    from mosaickv.evaluation.storage_runs import run
    from mosaickv.evaluation.streaming_baselines import SelectedRetiringCache, StreamingBaseline
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention
    from mosaickv.evaluation.streaming_source_constructor import StreamingDirectSource

    validate(reconstruction_readers)
    validate_source_plan(source_hotpath)
    if full_prompt_native is not None:
        from mosaickv.evaluation.full_prompt_source import validate as validate_native

        validate_native(full_prompt_native)
    if type(audit_decode_storage) is not bool:
        raise ValueError("explicit decode storage audit boolean required")

    if method not in METHODS:
        raise ValueError("five frozen local precision variants only")
    if type(componentwise_fitting) is not bool or type(code_delta) is not bool:
        raise ValueError("explicit common constructor and code-storage flags required")
    if input_packet_validator is not None and not callable(input_packet_validator):
        raise ValueError("optional first-warmup packet validator must be callable")
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
        StreamingDirectSource(
            adapter,
            prepared,
            cache,
            profile,
            minimum_only=minimum_only,
            construction_geometry=construction_geometry,
        )
        if method == "mosaickv_exact"
        else StreamingBaseline(prepared, cache, profile, method, minimum_only=minimum_only)
    )
    fresh = replace(prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache))
    with ExitStack() as stack:
        if componentwise_fitting:
            from mosaickv.evaluation import shared_source_format

            stack.enter_context(
                patch.object(
                    shared_source_format,
                    "fit_transport",
                    partial(shared_source_format.fit_transport, componentwise=True),
                )
            )
        with SealAfterAttention(adapter, cache, encoder):
            pref = adapter.prefill(fresh, capture_queries=False, capture_attentions=False)
        encoder.finish() if method == "mosaickv_exact" else encoder.finish(adapter)
    packets = cache.formats()
    runtime = replace(pref, state=replace(pref.state, past_key_values=None), attention_weights=())
    prepared.model_inputs.clear()
    del pref, fresh, cache, encoder, prepared
    gc.collect()
    original_packet_bytes = backing_bytes(tuple(t for p in packets for t in p.tensors()))
    geometry_layers = 0
    if source_geometry is not None:
        from mosaickv.evaluation.source_row_geometry import apply, eligible

        geometry_layers = sum(eligible(p) for p in packets)
        packets = apply(packets, source_geometry, num_warps=source_num_warps)
    elif source_num_warps is not None:
        raise ValueError("warp overrides require an explicit source geometry")
    if input_packet_validator is not None:
        # Caller permits this ONLY on the excluded first warmup. No hash/copy
        # is inserted into a measured repetition's request clock.
        input_packet_validator(packets)
    native_report = None
    native_conversion_wall_seconds = 0.0
    if full_prompt_native is not None:
        from mosaickv.evaluation.full_prompt_source import convert as convert_native

        conversion_started = time.monotonic()
        packets, native_report = convert_native(
            packets,
            tuple(layer.self_attn for layer in adapter._language_layers()),
            full_prompt_native,
        )
        native_conversion_wall_seconds = time.monotonic() - conversion_started
    if code_delta:
        from mosaickv.evaluation.layer_code_delta import convert

        packets, code_report = convert(packets)
        if not code_report["enabled"]:
            raise RuntimeError("registered beneficial source codec did not reduce storage")
    torch.cuda.synchronize()
    constructed_at = time.monotonic()
    with (
        use(reconstruction_readers),
        use_dispatch(reuse_kernel_dispatch) as dispatch,
        use_descriptors(reuse_code_descriptors) as descriptors,
        source_plan(source_hotpath) as hotpath,
        native_plan(full_prompt_native == "kv_fused_prepared") as prepared_native,
    ):
        values, details = run(
            adapter,
            runtime,
            packets,
            tokens=128,
            audit=audit_decode_storage,
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
            initial_encoded_bytes=details["storage"]["initial_active_kv_storage_bytes"],
            final_encoded_bytes=details["storage"]["final_active_kv_storage_bytes"],
            reported_counter_peak_bytes=details["storage"]["peak_active_kv_storage_bytes"],
            peak_measurement_admissible=details["storage"][
                "peak_admissible_only_for_declared_tensor_boundary"
            ],
            decode_owned_peak_bytes=details["storage"]["peak_active_kv_storage_bytes"]
            if details["storage"]["peak_admissible_only_for_declared_tensor_boundary"]
            else None,
            persistent_packet_bytes=backing_bytes(tuple(t for p in packets for t in p.tensors())),
            original_precompression_packet_bytes=original_packet_bytes,
            source_code_delta=code_delta,
            common_componentwise_fitting=componentwise_fitting,
            source_geometry=source_geometry,
            source_num_warps=source_num_warps,
            construction_geometry=construction_geometry,
            reconstruction_readers=reconstruction_readers,
            reuse_kernel_dispatch=reuse_kernel_dispatch,
            reuse_code_descriptors=reuse_code_descriptors,
            source_hotpath=source_hotpath,
            full_prompt_native=full_prompt_native,
            native_conversion=native_report,
            native_conversion_wall_seconds=native_conversion_wall_seconds,
            native_conversion_wall_is_not_pure_kernel_time=True,
            native_full_source_decodes=details["storage"].get("native_full_source_decodes", 0),
            native_full_component_fills=details["storage"].get("native_full_component_fills", 0),
            native_fused_restore_fills=details["storage"].get("native_fused_restore_fills", 0),
            native_plan_weight_hits=0
            if prepared_native is None
            else prepared_native["weight_hits"],
            native_plan_weight_misses=0
            if prepared_native is None
            else prepared_native["weight_misses"],
            native_plan_restore_hits=0
            if prepared_native is None
            else prepared_native["restore_hits"],
            native_plan_restore_misses=0
            if prepared_native is None
            else prepared_native["restore_misses"],
            source_launch_hits=0 if hotpath is None else hotpath["source_hits"],
            source_launch_misses=0 if hotpath is None else hotpath["source_misses"],
            fused_restore_calls=0 if hotpath is None else hotpath["fused_restore_calls"],
            fused_nonempty_calls=0 if hotpath is None else hotpath["fused_nonempty_calls"],
            skipped_prefix_views=0 if hotpath is None else hotpath["skipped_prefix_views"],
            replay_descriptor_preparations=0
            if descriptors is None
            else descriptors["prepared_formats"],
            replay_descriptor_reuses=0 if descriptors is None else descriptors["reused_formats"],
            dispatch_hits=0 if dispatch is None else dispatch["hits"],
            dispatch_misses=0 if dispatch is None else dispatch["misses"],
            source_geometry_applied_layers=geometry_layers,
            first_warmup_packet_audit=input_packet_validator is not None,
            tensor_result_auditing=audit_decode_storage,
            memory_counters_not_peak_evidence=not audit_decode_storage,
            whole_request_GPU_peak_certified=False,
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
