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
    hybrid_reader_policy=None,
    prompt_prefetch=False,
    text_overlay_policy=None,
    direct_text_policy=None,
    final_packet_validator=None,
    defer_transport_validation=False,
    coalesce_transport_nonzero=False,
    batch_source_validation=False,
    full_predicate_span=False,
    prompt_graph_reader=False,
    prompt_graph_policy=None,
    lt_source_reader=False,
    lt_fit_policy="full",
    selective_text_rows=False,
    defer_exact_encoding=False,
):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.native_execution_plan import use as native_plan
    from mosaickv.evaluation.prompt_prefetch import use as prefetch_plan
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
    if type(defer_exact_encoding) is not bool or (
        defer_exact_encoding and direct_text_policy is None
    ):
        raise ValueError("deferred exact encoding requires frozen direct-text construction")
    if type(selective_text_rows) is not bool or (
        selective_text_rows and direct_text_policy is None
    ):
        raise ValueError("selective text rows require explicit frozen direct-text construction")
    if (
        type(lt_fit_policy) is not str
        or lt_fit_policy not in ("full", "direct")
        or (not lt_source_reader and lt_fit_policy != "full")
    ):
        raise ValueError("registered native-source constructor policy requires its enabled reader")
    if type(lt_source_reader) is not bool or (
        lt_source_reader
        and (not prompt_graph_reader or full_prompt_native is not None or prompt_prefetch)
    ):
        raise ValueError("native Lt source reader requires its separate private graph policy")
    if prompt_graph_policy is not None:
        from mosaickv.evaluation.prompt_graph_policy import PromptGraphPolicy

        PromptGraphPolicy.parse(prompt_graph_policy)
        if not prompt_graph_reader:
            raise ValueError("explicit graph reader required for its optimization policy")
    validate_source_plan(source_hotpath)
    if full_prompt_native is not None:
        from mosaickv.evaluation.full_prompt_source import validate as validate_native

        validate_native(full_prompt_native)
    if type(audit_decode_storage) is not bool:
        raise ValueError("explicit decode storage audit boolean required")
    if any(
        type(v) is not bool
        for v in (
            defer_transport_validation,
            coalesce_transport_nonzero,
            batch_source_validation,
            full_predicate_span,
            prompt_graph_reader,
        )
    ):
        raise ValueError("explicit private transport validation policy required")
    if (
        coalesce_transport_nonzero or batch_source_validation or full_predicate_span
    ) and not defer_transport_validation:
        raise ValueError("construction optimizations require the private validation transaction")
    if hybrid_reader_policy is not None and full_prompt_native is not None:
        raise ValueError("hybrid and native projection policies are separate variants")
    if type(prompt_prefetch) is not bool or (
        prompt_prefetch and (hybrid_reader_policy is not None or full_prompt_native is not None)
    ):
        raise ValueError("prefetch requires a separate original-representation policy")
    if text_overlay_policy is not None and (
        hybrid_reader_policy is not None or full_prompt_native is not None or prompt_prefetch
    ):
        raise ValueError("cached text requires a separate original-representation policy")
    if (
        direct_text_policy is not None
        and any(
            p is not None for p in (text_overlay_policy, hybrid_reader_policy, full_prompt_native)
        )
    ) or (direct_text_policy is not None and prompt_prefetch):
        raise ValueError("direct construction requires a separate frozen text policy")
    if final_packet_validator is not None and not callable(final_packet_validator):
        raise ValueError("explicit optional constructed-packet validator required")

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
        from mosaickv.evaluation import deferred_exact_encoding as deferred_exact

        deferred_state = stack.enter_context(deferred_exact.use(defer_exact_encoding))
        from mosaickv.evaluation import selective_text_construction as selective_text

        selective_state = stack.enter_context(selective_text.use(selective_text_rows))
        from mosaickv.evaluation.deferred_transport_checks import use as transport_checks
        from mosaickv.evaluation.direct_text_construction import use as direct_text

        transport_state = stack.enter_context(
            transport_checks(
                defer_transport_validation,
                coalesce_nonzero=coalesce_transport_nonzero,
                batch_source_validation=batch_source_validation,
                full_predicate_span=full_predicate_span,
            )
        )
        direct_state = stack.enter_context(direct_text(direct_text_policy))
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
    direct_report = direct_state.finish(packets) if direct_state is not None else None
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
    hybrid_report = None
    if hybrid_reader_policy is not None:
        from mosaickv.evaluation.hybrid_reader import convert as hybrid_convert

        packets, hybrid_report = hybrid_convert(
            packets,
            tuple(layer.self_attn for layer in adapter._language_layers()),
            hybrid_reader_policy,
        )
    text_report = None
    if text_overlay_policy is not None:
        from mosaickv.evaluation.cached_text_reader import convert as text_convert

        packets, text_report = text_convert(
            packets,
            tuple(layer.self_attn for layer in adapter._language_layers()),
            text_overlay_policy,
        )
    if final_packet_validator is not None:
        final_packet_validator(packets)
    lt_report = None
    if lt_source_reader:
        from mosaickv.evaluation.lt_source_format import convert as convert_lt

        packets, lt_report = convert_lt(
            packets,
            tuple(layer.self_attn for layer in adapter._language_layers()),
            policy=lt_fit_policy,
            audit=audit_decode_storage,
        )
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
        prefetch_plan(prompt_prefetch) as prefetched,
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
            prompt_graph_reader=prompt_graph_reader,
            prompt_graph_policy=prompt_graph_policy,
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
            prompt_graph_reader=details["storage"].get("prompt_graph_reader", dict(enabled=False)),
            lt_source_reader=lt_source_reader,
            lt_fit_policy=lt_fit_policy,
            deferred_exact_encoding=deferred_exact.report(deferred_state),
            selective_text_construction=dict(
                enabled=selective_text_rows,
                cpu_index_metadata_bytes=selective_text.cpu_metadata_bytes(packets),
                CPU_indices_are_NOT_residual_KV=True,
                **(selective_state or {}),
            ),
            lt_source_conversion=lt_report,
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
            hybrid_conversion=hybrid_report,
            cached_text_conversion=text_report,
            direct_text_construction=direct_report,
            transport_validation=transport_state.report()
            if transport_state is not None
            else dict(enabled=False),
            prompt_prefetch=prompt_prefetch,
            prefetch_hits=prefetched["hits"],
            prefetch_misses=prefetched["misses"],
            prefetch_launches=prefetched["launches"],
            prefetch_buffer_bytes=prefetched["buffer_bytes"],
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
