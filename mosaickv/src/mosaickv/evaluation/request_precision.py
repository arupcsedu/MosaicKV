"""Complete, isolated HF requests for an explicit profiled precision variant.

Not the frozen graph-based paper method, official PrefixKV, or official KIVI.
No saved selections, prefills, draft tokens, or annotation answers are inputs.
"""

from contextlib import ExitStack, nullcontext
from dataclasses import replace
from typing import Any

import torch

METHODS = (
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
    "mosaickv_exact",
)
REQUEST_POLICY: dict[str, Any] = dict(
    variant="online_profiled_shared_prior_precision_v1",
    prompt_bits=3,
    quant_group=128,
    prompt_window=16,
    shared_prior_weight=0.5,
    correction_fraction=0.005,
    exact_boundary_layers=2,
    exact_first_selected=4,
    exact_last_selected=8,
    protect_all_selected_text=True,
    normalized_capture_window=16,
    generated_history_lossless=True,
    source_history_constructor="direct",
    consolidate_history=True,
    output_predictions=128,
    block_size=1,
    mandatory_policy="last_token",
    source_KV_must_be_retired=True,
    common_codec_offered_to_every_baseline=True,
    tensor_workspace_auditing_for_timing=False,
)


def ranked_positions(
    scores: Any, count: int, mandatory: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    if scores.ndim != 2 or type(count) is not int or not len(mandatory) <= count <= scores.shape[1]:
        raise ValueError("finite head/token scores, feasible capacity and mandatory IDs required")
    if not bool(torch.isfinite(scores).all()) or any(
        not 0 <= p < scores.shape[1] for p in mandatory
    ):
        raise ValueError("invalid scores or mandatory source position")
    values = scores.clone()
    values[:, list(mandatory)] = float("inf")
    # Source index is the deterministic secondary ordering, also at ties.
    order = torch.argsort(values, dim=-1, descending=True, stable=True)[:, :count]
    return tuple(tuple(sorted(h)) for h in order.cpu().tolist())


def online_frozen(
    prefill: Any,
    profile: tuple[float, ...],
    ratio: float,
    *,
    method: str | None = None,
    priors: Any = None,
    allocation_mode: str = "profile",
) -> tuple[dict[str, Any], dict[str, Any]]:
    from mosaickv.selection.profile_lifetime import allocate_profile_lifetime

    if type(ratio) not in (int, float) or not 0 < ratio < 1:
        raise ValueError("compressed source-slot ratio must be strictly between zero and one")
    probabilities = prefill.attention_weights
    if not probabilities or len(probabilities) != len(profile):
        raise ValueError("complete eager prompt attention and pinned layer profile required")
    lengths = tuple(int(p.shape[-1]) for p in probabilities)
    costs = tuple(int(p.shape[1]) * 128 * 2 * 2 for p in probabilities)
    # This is a source-slot capacity control, NOT an encoded-physical-byte match.
    cap = int(sum(n * c for n, c in zip(lengths, costs, strict=True)) * ratio)
    allocator: Any = allocate_profile_lifetime
    capacity = allocator(
        profile, lengths, (1,) * len(profile), costs, cap_bytes=cap, generated_capacity=0
    )
    counts = capacity.prompt_positions
    if allocation_mode == "balanced":
        if len(set(lengths)) != 1 or len(set(costs)) != 1:
            raise ValueError("balanced source allocation requires equal layer geometry")
        lower, extra = divmod(sum(counts), len(counts))
        counts = tuple(lower + int(i < extra) for i in range(len(counts)))
        if not all(1 <= c <= lengths[i] for i, c in enumerate(counts)):
            raise ValueError("balanced allocation exceeds source or mandatory capacity")
    elif allocation_mode != "profile":
        raise ValueError("registered profile or balanced allocation required")
    topk, mixed = [], []
    for i, (probabilities_, count) in enumerate(zip(probabilities, counts, strict=True)):
        placeholder = tuple(tuple(range(count)) for _ in range(probabilities_.shape[1]))
        if method is not None and method not in ("prompt_attention_topk", "mosaickv_exact"):
            topk.append(placeholder)
            mixed.append(placeholder)
            continue
        window = probabilities_[0, :, -REQUEST_POLICY["prompt_window"] :, :].float().mean(-2)
        # Keep the pinned BF16 head-mean -> FP32 query-mean reduction order.
        last = (probabilities_.shape[-1] - 1,)
        topk.append(
            ranked_positions(window, count, last) if method != "mosaickv_exact" else placeholder
        )
        if method != "prompt_attention_topk":
            prior = (
                priors[i]
                if priors is not None
                else probabilities_.mean(1, keepdim=True).float().mean(-2)[0]
            )
            weight = REQUEST_POLICY["shared_prior_weight"]
            mixed.append(ranked_positions((1 - weight) * window + weight * prior, count, last))
        else:
            mixed.append(placeholder)
    return {
        "window_topk_control": {"anchor": {"positions": tuple(topk)}},
        "uniform_0.5_control": {"anchor": {"positions": tuple(mixed)}},
    }, dict(
        variant="online_profiled_shared_prior_precision_v1_not_frozen_graph_paper_method",
        prompt_window=16,
        full_prompt_shared_prior_weight=0.5,
        mandatory_policy="last_token",
        block_size=1,
        actual_source_slot_cap=sum(counts[i] * costs[i] for i in range(len(costs))),
        per_layer_capacity=counts,
        allocation_mode=allocation_mode,
        original_profile_capacity=capacity.prompt_positions,
        same_total_source_slots=sum(counts) == sum(capacity.prompt_positions),
        nominal_source_ratio=ratio,
        encoded_byte_matching=False,
        profile_source="pinned_PrefixKV_profile_shape_not_official_PrefixKV_selection",
        labels_or_reference_outputs_used=False,
    )


class PromptQueryCapture:
    """Only the 16 prompt vectors needed by the already frozen correction rule."""

    def __init__(
        self, adapter: Any, *, summaries: bool = False, normalized: bool = True, prior: bool = True
    ) -> None:
        self.adapter, self.stack = adapter, ExitStack()
        self.records: dict[int, dict[str, Any]] = {}
        self.summaries, self.normalized, self.prior = summaries, normalized, prior
        self.windows: dict[int, Any] = {}
        self.priors: dict[int, Any] = {}

    def __enter__(self) -> Any:
        for i, layer in enumerate(self.adapter._language_layers()):

            def hook(module: Any, args: Any, kwargs: Any, *, index: int = i) -> None:
                del module
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                rotary = kwargs.get("position_embeddings")
                if index in self.records or hidden is None or rotary is None:
                    raise ValueError("one explicit prompt projection invocation required")
                self.records[index] = dict(
                    normalized=hidden[:, -16:].detach().clone(),
                    cos=rotary[0][:, -16:].detach().clone(),
                    sin=rotary[1][:, -16:].detach().clone(),
                )

            if self.normalized:
                handle = layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
                self.stack.callback(handle.remove)
            if self.summaries:

                def summarize(module: Any, args: Any, output: Any, *, index: int = i) -> None:
                    del module, args
                    p = output[1]
                    if p is None or index in self.windows:
                        raise ValueError("one pinned eager prompt probability output required")
                    self.windows[index] = p[..., -16:, :].detach().clone()
                    if self.prior:
                        self.priors[index] = p.mean(1, keepdim=True).float().mean(-2)[0].detach()

                handle = layer.self_attn.register_forward_hook(summarize)
                self.stack.callback(handle.remove)
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.stack.__exit__(*args)


def construct_request_formats(
    adapter: Any,
    prepared: Any,
    prefill: Any,
    capture: Any,
    cfg: Any,
    method: str,
    profile_path: str,
    ratio: float,
    precision: str,
) -> tuple[Any, dict[str, Any]]:
    import json
    from pathlib import Path

    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.precision_corrections import fund_corrections, prompt_weights
    from mosaickv.evaluation.precision_tier import construct
    from mosaickv.evaluation.storage_baselines import baseline_payload

    if method not in METHODS or precision not in ("lossless", "query3"):
        raise ValueError("only preregistered baseline/precision variants permitted")
    profile = tuple(json.loads(Path(profile_path).read_text()))
    frozen, online = online_frozen(
        prefill,
        profile,
        ratio,
        method=method,
        priors=capture.priors if capture is not None and capture.summaries else None,
    )
    positions, transformed, proof = baseline_payload(
        adapter, prepared, prefill, cfg, method, frozen, profile_path, borrow_source=True
    )
    memberships = (
        positions if positions is not None else proof["selected_source_positions_by_layer"]
    )
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    text = {
        p
        for span in prepared.modality_map
        if span.modality == "text"
        for p in range(span.start, span.end)
    }
    formats, records = [], []
    fmt: Any
    for i, ((key, value), heads) in enumerate(zip(source, memberships, strict=True)):
        if transformed is not None:
            pair = transformed[i]
        else:
            index = torch.tensor(heads, device=key.device)[None, :, :, None].expand(
                1, key.shape[1], len(heads[0]), key.shape[-1]
            )
            pair = key.gather(2, index), value.gather(2, index)
        if (
            precision == "lossless"
            or i < REQUEST_POLICY["exact_boundary_layers"]
            or i >= len(source) - REQUEST_POLICY["exact_boundary_layers"]
        ):
            fmt = ExactFormat("kv", tuple(pair[0].shape), pair=tuple(encode(t) for t in pair))
        else:
            protected = tuple(
                sorted({j for head in heads for j, p in enumerate(head) if p in text})
            )
            fmt = construct(
                pair, bits=REQUEST_POLICY["prompt_bits"], protected=protected, encoded_exact=True
            )
            weights = prompt_weights(
                adapter._language_layers()[i].self_attn,
                capture.records[i],
                prefill.attention_weights[i],
                heads,
                window=REQUEST_POLICY["prompt_window"],
            )
            fmt, record = fund_corrections(
                fmt,
                pair,
                fraction=REQUEST_POLICY["correction_fraction"],
                key_weight=weights[0],
                value_weight=weights[1],
            )
            records.append(dict(layer=i, **record))
        formats.append(fmt)
    return tuple(formats), dict(
        selection=proof,
        online=online,
        precision=precision,
        corrections=records,
        boundary_layers_exact=2,
        source_codec="lossless_BF16_KV_not_projection_search",
        generated_history="lossless_consolidated_source_append_only",
    )


def complete_request(
    adapter: Any,
    messages: Any,
    cfg: Any,
    *,
    method: str,
    precision: str,
    profile_path: str,
    ratio: float,
    audit: bool,
) -> tuple[Any, dict[str, Any]]:
    """Timer starts before processor; end is synchronized CPU logits export.

    Model loading and artifact I/O are excluded. No other request/prefill/cache
    is retained. Whole-process allocated/reserved peaks include ALL live CUDA
    allocations (weights, activations, captures, original/replacement KV).
    """
    import gc
    import time
    import weakref

    from mosaickv.evaluation import storage_runs

    if method != "full_kv" and method not in METHODS:
        raise ValueError("unregistered method")
    if type(ratio) not in (int, float) or not 0 < ratio <= 1:
        raise ValueError("finite source ratio in (0,1] required")
    if precision not in (("ordinary",) if method == "full_kv" else ("lossless", "query3")):
        raise ValueError("unsupported method/precision combination")
    if ratio == 1 and method != "full_kv":
        logits, identity = complete_request(
            adapter,
            messages,
            cfg,
            method="full_kv",
            precision="ordinary",
            profile_path=profile_path,
            ratio=1.0,
            audit=audit,
        )
        identity.update(
            requested_method=method,
            requested_precision=precision,
            retention_one_identity=True,
            quantization_disabled=True,
        )
        return logits, identity
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # Identical, explicit allocator policy for every request.
    baseline = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    event_factory: Any = torch.cuda.Event
    whole_start, whole_end = (event_factory(enable_timing=True) for _ in range(2))
    whole_start.record()
    started = time.monotonic()
    phases = {}

    def boundary(name: str, action: Any) -> Any:
        torch.cuda.synchronize()
        t = time.monotonic()
        result = action()
        torch.cuda.synchronize()
        phases[name] = time.monotonic() - t
        return result

    resolved_messages = boundary("load_media", messages) if callable(messages) else messages
    prepared = boundary("prepare_inputs", lambda: adapter.prepare_inputs(resolved_messages))
    summary_method = method in ("prompt_attention_topk", "mosaickv_exact")
    cap = (
        PromptQueryCapture(
            adapter,
            summaries=summary_method,
            normalized=precision == "query3",
            prior=method == "mosaickv_exact",
        )
        if method != "full_kv" and (precision == "query3" or summary_method)
        else None
    )
    with cap if cap is not None else nullcontext():
        prefill = boundary(
            "prefill",
            lambda: adapter.prefill(
                prepared,
                capture_queries=False,
                capture_attentions=method != "full_kv" and not summary_method,
            ),
        )
    if summary_method:
        assert cap is not None
        prefill = replace(
            prefill, attention_weights=tuple(cap.windows[i] for i in range(len(cap.windows)))
        )
    first_logit_ready = time.monotonic() - started
    record: dict[str, Any]
    if method == "full_kv":

        def ordinary() -> Any:
            logits, token = [prefill.logits.detach().reshape(-1).float()], prefill.next_token_id
            for _ in range(127):
                out = adapter.decode_one_token(token, prefill.state, capture_queries=False)
                logits.append(out.logits.detach().reshape(-1).float())
                token = out.next_token_id
            return torch.stack(logits).cpu()

        initial = sum(
            t.numel() * t.element_size()
            for pair in adapter._legacy_layers(prefill.state.past_key_values)[0]
            for t in pair
        )
        logits = boundary("decode", ordinary)
        final = sum(
            t.numel() * t.element_size()
            for pair in adapter._legacy_layers(prefill.state.past_key_values)[0]
            for t in pair
        )
        record = dict(
            storage=dict(
                initial_active_kv_storage_bytes=initial,
                final_active_kv_storage_bytes=final,
                peak_active_kv_storage_bytes=None,
                boundary="ordinary_fullKV_KV_transaction_peak_not_certified",
            )
        )
        proof = dict(full_kv=True, no_compression=True)
        retired = None
    else:

        def original_tensor_refs() -> Any:
            refs, seen = [], set()
            for pair in adapter._legacy_layers(prefill.state.past_key_values)[0]:
                for tensor in pair:
                    while tensor is not None and id(tensor) not in seen:
                        seen.add(id(tensor))
                        refs.append(weakref.ref(tensor))
                        tensor = tensor._base
            return refs

        original_refs = original_tensor_refs()
        formats, proof = boundary(
            "select_and_construct",
            lambda: construct_request_formats(
                adapter, prepared, prefill, cap, cfg, method, profile_path, ratio, precision
            ),
        )
        owner = weakref.ref(prefill.state.past_key_values)

        def retire() -> Any:
            nonlocal prefill, prepared, cap
            prefill.state.past_key_values = None
            prefill = replace(prefill, attention_weights=())
            prepared.model_inputs.clear()
            if cap is not None:
                cap.records.clear()
                cap.windows.clear()
                cap.priors.clear()
            prepared, cap = None, None
            if owner() is not None or any(r() is not None for r in original_refs):
                gc.collect()
            if owner() is not None or any(r() is not None for r in original_refs):
                raise RuntimeError("original prefill cache or tensor owner survived retirement")

        boundary("retire_source", retire)
        retired = True
        logits, record = boundary(
            "decode",
            lambda: storage_runs.run(
                adapter,
                prefill,
                formats,
                tokens=128,
                audit=audit,
                source_history=True,
                source_history_constructor="direct",
                consolidated_history=True,
            ),
        )

    def detokenize() -> Any:
        from mosaickv.evaluation.precision_quality import answer_prefix

        eos = getattr(adapter.model.generation_config, "eos_token_id", None)
        eos_ids = (eos,) if type(eos) is int else tuple(eos or ())
        return answer_prefix(
            logits.argmax(-1).tolist(),
            eos=eos_ids,
            cap=128,
            decode=lambda ids: adapter.processor.tokenizer.decode(ids, skip_special_tokens=True),
        )

    scored_answer = boundary("detokenize", detokenize)
    whole_end.record()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    memory = dict(
        baseline_model_allocated_bytes=baseline,
        whole_request_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        whole_request_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        whole_request_incremental_peak_allocated_bytes=int(torch.cuda.max_memory_allocated())
        - baseline,
        whole_request_allocator_peak_certified=True,
        whole_request_KV_only_peak_certified=False,
        excludes="model_load,artifact_IO,metric_scoring,input_hash_verification",
        includes="model_resident_weights,processor_GPU_tensors,prefill_KV,activations,attention_captures,selection_construction,decode_workspace",
        allocator_pool_emptied_before_every_request=True,
    )
    measured: dict[str, Any] = dict(record)
    measured.update(
        method=method,
        precision=precision,
        online_variant=proof,
        phase_wall_seconds=phases,
        end_to_end_seconds=elapsed,
        first_logit_ready_seconds=first_logit_ready,
        output_streamed=False,
        first_complete_output_ready_seconds=elapsed,
        end_to_end_cuda_ms=whole_start.elapsed_time(whole_end),
        **memory,
        source_KV_owner_retired_before_decode=retired,
        used_fullkv_fallback=False,
        output_tokens=128,
        timed_output_policy="128_predictions_ignore_EOS_identical_for_every_method",
        scored_answer=scored_answer,
        end_to_end_includes_processor_prefill_selection_constructor_retirement_decode_detokenization=True,
    )
    return logits, measured
