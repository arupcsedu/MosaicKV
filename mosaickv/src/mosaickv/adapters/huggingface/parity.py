"""Real-checkpoint correctness gates for explicit HF adapter decoding."""

from __future__ import annotations

import copy
from typing import Any

from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter, _torch
from mosaickv.adapters.huggingface.types import ParityReport, PreparedInputs
from mosaickv.cache_state import FullKVState, MosaicKVState


def compare_with_generate(
    adapter: HuggingFaceMultimodalAdapter,
    prepared: PreparedInputs,
    *,
    max_new_tokens: int = 16,
) -> ParityReport:
    """Compare explicit greedy decoding to GenerationMixin as a reference only.

    ``model.generate`` is deliberately confined to this validation function;
    the candidate MosaicKV path is ``adapter.greedy_decode`` and never invokes
    GenerationMixin.
    """

    if max_new_tokens < 16:
        raise ValueError("the adapter acceptance gate requires at least 16 generated tokens")
    torch = _torch()
    generation_config = copy.deepcopy(adapter.model.generation_config)
    # A copied model-derived GenerationConfig is otherwise refreshed from the
    # model config inside generate(), which can silently restore EOS stopping.
    generation_config._from_model_config = False
    generation_config.do_sample = False
    generation_config.max_new_tokens = max_new_tokens
    generation_config.min_new_tokens = None
    generation_config.eos_token_id = None
    generation_config.forced_eos_token_id = None
    generation_config.forced_bos_token_id = None
    generation_config.suppress_tokens = None
    generation_config.begin_suppress_tokens = None
    generation_config.bad_words_ids = None
    generation_config.sequence_bias = None
    generation_config.repetition_penalty = 1.0
    generation_config.no_repeat_ngram_size = 0
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = 0

    reference_inputs = dict(prepared.model_inputs)
    with torch.inference_mode():
        reference = adapter.model.generate(
            **reference_inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
    if len(reference.scores) != max_new_tokens:
        raise RuntimeError(
            f"generate returned {len(reference.scores)} scores; expected {max_new_tokens}"
        )
    reference_tokens = reference.sequences[:, -max_new_tokens:]
    candidate = adapter.greedy_decode(prepared, max_new_tokens=max_new_tokens)
    return _build_report(
        "explicit_full_cache_vs_model_generate",
        reference_tokens,
        tuple(reference.scores),
        candidate.token_ids,
        candidate.step_logits,
    )


def compare_cache_reinjection(
    adapter: HuggingFaceMultimodalAdapter,
    prepared: PreparedInputs,
    *,
    max_new_tokens: int = 16,
) -> ParityReport:
    """Compare untouched full cache with extraction/reinjection at ratio 1.0."""

    if max_new_tokens < 16:
        raise ValueError("the retention-ratio-1.0 gate requires at least 16 generated tokens")
    untouched = adapter.greedy_decode(prepared, max_new_tokens=max_new_tokens)
    reinjected = adapter.greedy_decode(
        prepared, max_new_tokens=max_new_tokens, reinject_after_prefill=True
    )
    return _build_report(
        "retention_1_reinjected_vs_untouched",
        untouched.token_ids,
        untouched.step_logits,
        reinjected.token_ids,
        reinjected.step_logits,
    )


def compare_mosaickv_retention_one(
    adapter: HuggingFaceMultimodalAdapter,
    prepared: PreparedInputs,
    *,
    max_new_tokens: int = 16,
    block_size: int = 4,
) -> ParityReport:
    """Compare untouched decoding with core MosaicKV 100%-exact reinjection."""

    if max_new_tokens < 16:
        raise ValueError("the MosaicKV retention-ratio-1.0 gate requires at least 16 tokens")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    torch = _torch()
    untouched = adapter.greedy_decode(prepared, max_new_tokens=max_new_tokens)

    prefill = adapter.prefill(prepared)
    snapshot = adapter.extract_past_key_values(prefill.state.past_key_values)
    input_ids = prepared.model_inputs.get("input_ids")
    if input_ids is None:
        raise RuntimeError("prepared inputs have no input_ids for cache-state provenance")
    full = FullKVState.from_cache_snapshot(
        snapshot,
        modality_spans=prepared.modality_map,
        token_ids=input_ids,
        block_size=block_size,
        original_logical_sequence_length=prefill.state.logical_sequence_length,
        next_decode_position=prefill.state.next_decode_position,
        mandatory_logical_positions=(0, prefill.state.logical_sequence_length - 1),
    )
    mosaic = MosaicKVState.retention_one(full)
    reconstructed = mosaic.reconstruct_full_state(full)
    state = prefill.state
    state.past_key_values = adapter.inject_past_key_values(reconstructed.to_cache_snapshot())
    state.active_cache_length = reconstructed.active_sequence_length
    state.logical_sequence_length = reconstructed.original_logical_sequence_length
    state.next_decode_position = reconstructed.next_decode_position
    tokens = [prefill.next_token_id]
    logits = [prefill.logits]
    token = prefill.next_token_id
    for _index in range(max_new_tokens - 1):
        step = adapter.decode_one_token(token, state)
        state = step.state
        token = step.next_token_id
        tokens.append(token)
        logits.append(step.logits)
    candidate_tokens = torch.cat(tokens, dim=-1)
    return _build_report(
        "mosaickv_core_retention_1_vs_untouched",
        untouched.token_ids,
        untouched.step_logits,
        candidate_tokens,
        tuple(logits),
    )


def _build_report(
    comparison: str,
    reference_tokens: Any,
    reference_logits: tuple[Any, ...],
    candidate_tokens: Any,
    candidate_logits: tuple[Any, ...],
) -> ParityReport:
    torch = _torch()
    if len(reference_logits) != len(candidate_logits):
        raise RuntimeError("reference and candidate have different logit-step counts")
    reference_flat = reference_tokens.detach().cpu().reshape(-1)
    candidate_flat = candidate_tokens.detach().cpu().reshape(-1)
    if reference_flat.numel() != candidate_flat.numel():
        raise RuntimeError("reference and candidate have different token counts")
    agreement = float((reference_flat == candidate_flat).float().mean().item())
    maximum = 0.0
    for reference_step, candidate_step in zip(reference_logits, candidate_logits, strict=True):
        difference = torch.max(
            torch.abs(reference_step.detach().float() - candidate_step.detach().float())
        )
        maximum = max(maximum, float(difference.item()))
    return ParityReport(
        comparison=comparison,
        generated_tokens=int(reference_flat.numel()),
        token_agreement=agreement,
        maximum_logit_difference=maximum,
        reference_token_ids=tuple(int(item) for item in reference_flat.tolist()),
        candidate_token_ids=tuple(int(item) for item in candidate_flat.tolist()),
    )


__all__ = [
    "compare_cache_reinjection",
    "compare_mosaickv_retention_one",
    "compare_with_generate",
]
