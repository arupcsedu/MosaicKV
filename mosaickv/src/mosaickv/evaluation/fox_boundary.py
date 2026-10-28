"""Evaluation-only held-back-original-token control; never an online policy."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mosaickv.cache_state import Modality


def hold_back_last_text_token(prepared):
    """Only expanded-token LLaVA text tails are currently adapter-valid.

    Media tensors are unchanged. No full-prompt queries/logits are produced by
    this operation, and an unknown sequence-shaped field is rejected.
    """
    from mosaickv.adapters.huggingface import PreparedInputs

    inputs = prepared.model_inputs
    ids = inputs.get("input_ids")
    if ids is None or tuple(ids.shape[:1]) != (1,) or ids.shape[-1] < 2:
        raise NotImplementedError("boundary B requires batch-one explicit expanded input IDs")
    length = int(ids.shape[-1])
    if prepared.logical_sequence_length != length:
        raise NotImplementedError("adapter has not exposed expanded logical token IDs")
    if (
        not prepared.modality_map
        or prepared.modality_map[-1].end != length
        or prepared.modality_map[-1].modality is not Modality.TEXT
    ):
        raise NotImplementedError(
            "final original token is not an independently removable text token"
        )
    known = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    media = {"pixel_values", "image_sizes", "pixel_attention_mask", "image_grid_thw"}
    prefix = {}
    for name, value in inputs.items():
        if name in known:
            if not hasattr(value, "shape") or value.shape[-1] != length:
                raise NotImplementedError(f"unsupported sequence metadata: {name}")
            prefix[name] = value[..., :-1].clone()
        else:
            if name not in media and hasattr(value, "shape") and length in value.shape:
                raise NotImplementedError(f"ambiguous unsliced sequence metadata: {name}")
            prefix[name] = value
    spans = tuple(
        replace(span, end=min(span.end, length - 1))
        for span in prepared.modality_map
        if span.start < length - 1
    )
    return PreparedInputs(prefix, spans, length - 1), ids[..., -1:].clone()


def run_boundary_b(adapter, config, messages, request, output: Path):
    """One prefix prefill, compress, held token once, then 64 answer decisions.

    Uses the production planner up to its existing evaluation-only early-stop
    hook, then performs the explicit independent continuation. Not a timing row.
    """
    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.evaluation.fox_campaign import unique_storage_bytes
    from mosaickv.generation import decode_quality_output, terminal_token_ids

    if adapter.capabilities.model_family not in {"llava_1_5", "llava_next", "llava", "llava15"}:
        # Accept actual architecture names below, not a guessed capability label.
        architecture = type(adapter.model).__name__
        if architecture not in {
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        }:
            raise NotImplementedError("boundary B has not been validated for this adapter")
    prepared = adapter.prepare_inputs(messages)
    prefix, held = hold_back_last_text_token(prepared)
    reserved_bytes = 0
    original_cap = None
    if not config.method.is_full_cache:
        if config.cache.budget_unit.value != "bytes":
            raise NotImplementedError("boundary B reserved-step control requires byte budgets")
        for layer in adapter._language_layers():
            attention = layer.self_attn
            reserved_bytes += (
                attention.k_proj.out_features + attention.v_proj.out_features
            ) * attention.k_proj.weight.element_size()
        original_cap = config.cache.budget_value
        if original_cap <= reserved_bytes:
            raise ValueError("infeasible held-back-step capacity reservation")
        config = replace(
            config, cache=replace(config.cache, budget_value=original_cap - reserved_bytes)
        )
    captures = {}
    original_prefill = adapter.prefill
    original_state = runtime._state_from_packed

    def prefill(*args, **kwargs):
        result = original_prefill(*args, **kwargs)
        actual = (
            sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for k, v in adapter._legacy_layers(result.state.past_key_values)[0]
            )
            // prefix.logical_sequence_length
        )
        if actual != reserved_bytes:
            raise NotImplementedError(
                "projection geometry does not match actual cache append bytes"
            )
        captures["prefill"] = result
        return result

    def packed(local_adapter, source, selection):
        state = original_state(local_adapter, source, selection)
        captures["state"] = state
        raise runtime.PrefillDiagnosticComplete(
            dict(boundary="B", prefix_only=True, held_token_seen_by_forecast=False)
        )

    if config.method.is_full_cache:
        result = adapter.prefill(prefix, capture_queries=False, capture_attentions=False)
        state = result.state
    else:
        with ExitStack() as stack:
            stack.enter_context(patch.object(adapter, "prepare_inputs", return_value=prefix))
            stack.enter_context(patch.object(adapter, "prefill", prefill))
            stack.enter_context(patch.object(runtime, "_state_from_packed", packed))
            model = runtime.HuggingFaceMosaicKVModel(adapter, config, trace_directory=output)
            try:
                model.generate(request)
            except runtime.PrefillDiagnosticComplete:
                pass
            else:
                raise RuntimeError("production path bypassed the registered boundary capture")
        state = captures["state"]
    if state.next_decode_position != prefix.logical_sequence_length:
        raise AssertionError("held-back original token has wrong logical decode position")
    result = adapter.decode_one_token(held, state, capture_queries=False, capture_attentions=False)
    first_logits = result.logits.detach().float().cpu()
    state = result.state
    initial_bytes = unique_storage_bytes(
        [t for pair in adapter._legacy_layers(state.past_key_values)[0] for t in pair]
    )
    if original_cap is not None and initial_bytes > original_cap:
        raise AssertionError("held-back step exceeded registered boundary capacity")
    ids = [int(result.next_token_id.item())]
    token = result.next_token_id
    for _step in range(63):
        result = adapter.decode_one_token(
            token, state, capture_queries=False, capture_attentions=False
        )
        token = result.next_token_id
        ids.append(int(token.item()))
    final_bytes = unique_storage_bytes(
        [t for pair in adapter._legacy_layers(state.past_key_values)[0] for t in pair]
    )
    quality = decode_quality_output(
        adapter.processor, tuple(ids), terminal_token_ids(adapter.model, adapter.processor)
    )
    return dict(
        boundary="B",
        status="completed",
        measurement_type="boundary_diagnostic_not_timing",
        answer=quality.text,
        token_ids=ids,
        first_token_id=ids[0],
        first_logits=first_logits,
        prefix_logical_length=prefix.logical_sequence_length,
        original_logical_length=prepared.logical_sequence_length,
        held_token_forward_count=1,
        observation_window=min(config.forecasting.prompt_window, prefix.logical_sequence_length),
        answer_token_forwards=63,
        initial_kv_bytes=initial_bytes,
        final_kv_bytes=final_bytes,
        peak_kv_bytes=max(initial_bytes, final_bytes),
        capacity_stratum="B_prefix_cap_plus_held_step_no_preallocation",
        directly_byte_matched_to_A=False,
        held_step_reserved_bytes=reserved_bytes,
        configured_answer_boundary_cap=original_cap,
    )
