"""Evaluation-only causal controls: identical LOOK-M selection, different K/V.

These are not alternative official baselines or an online compressor. The
complete merged plan is computed once, unchanged, before any control runs.
Every decode receives private cache tensors, with original post-RoPE positions.
"""

from dataclasses import replace
from typing import Any

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.native_storage_control import NativePayload

ARMS = ("selection_only", "key_merge_only", "value_merge_only", "key_value_merge")


def make_payload(
    adapter: Any,
    prefill: Any,
    merged: Any,
    positions: Any,
    arm: str,
) -> NativePayload:
    """Use exactly the same membership/capacity; only the K/V source changes."""
    from mosaickv.adapters.huggingface.types import CacheLayerSnapshot

    if arm not in ARMS:
        raise ValueError("unknown LOOK-M merge ablation")
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    if not (len(source) == len(merged) == len(positions)):
        raise ValueError("source, merged payload, and membership layer counts differ")
    layers = []
    normalized = []
    for (key, value), (mk, mv), heads in zip(source, merged, positions, strict=True):
        heads = tuple(tuple(p) for p in heads)
        if len(heads) != key.shape[1] or not heads or len({len(h) for h in heads}) != 1:
            raise ValueError("equal nonempty physical head capacity required per layer")
        for p in heads:
            if (
                not p
                or any(type(i) is not int for i in p)
                or tuple(sorted(set(p))) != p
                or p[0] < 0
                or p[-1] != prefill.state.logical_sequence_length - 1
                or p[-1] >= key.shape[2]
            ):
                raise ValueError("sorted source positions must retain the mandatory last token")
        index = torch.tensor(heads, device=key.device)[None, :, :, None].expand(
            key.shape[0], key.shape[1], len(heads[0]), key.shape[-1]
        )
        sk, sv = key.gather(2, index), value.gather(2, index)
        for m, s in ((mk, sk), (mv, sv)):
            if m.shape != s.shape or m.dtype != s.dtype or m.device != s.device:
                raise ValueError("merge changed cache capacity, dtype, or device")
            if not bool(torch.isfinite(m).all()):
                raise ValueError("nonfinite merged cache")
        k = mk.detach().clone() if arm in ("key_merge_only", "key_value_merge") else sk
        v = mv.detach().clone() if arm in ("value_merge_only", "key_value_merge") else sv
        layers.append(CacheLayerSnapshot(k, v, 2))
        normalized.append(heads)
    template = adapter.borrow_past_key_values(prefill.state.past_key_values)
    snapshot = replace(
        template,
        layers=tuple(layers),
        active_sequence_length=max(layer.key.shape[2] for layer in layers),
        independent_layer_lengths=True,
    )
    return NativePayload(snapshot, tuple(normalized))


def decode(
    adapter: Any,
    prefill: Any,
    payload: NativePayload,
    *,
    predictions: int,
    forced_prefix: tuple[int, ...] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Include the common prefill logit, then use greedy or fixed FullKV tokens.

    Forced tokens are evaluation-only evidence, never passed to selection. A
    33-prediction forced run measures 32 cache-dependent logits at identical
    conditioning prefixes; the common prefill logit is excluded from the guard.
    """
    from mosaickv.backends.hf_runtime import PackedRuntimeCache, _state_from_packed

    if type(predictions) is not int or predictions < 2:
        raise ValueError("at least two predictions required")
    if forced_prefix is not None and (
        len(forced_prefix) != predictions - 1
        or any(type(i) is not int or i < 0 for i in forced_prefix)
    ):
        raise ValueError("one nonnegative forced token per cache-dependent prediction required")
    snapshot = payload.snapshot
    if snapshot is None:
        raise ValueError("payload already consumed")
    packed = PackedRuntimeCache(
        snapshot,
        (),
        snapshot.active_sequence_length,
        True,
        (),
        (),
        (),
        tuple(tuple(heads[0]) for heads in payload.head_positions),
    )
    state = _state_from_packed(adapter, prefill.state, packed)
    state.model_state["native_prompt_positions_by_layer_head"] = payload.head_positions
    logical, next_position = state.logical_sequence_length, state.next_decode_position
    if (logical, next_position) != (
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    ):
        raise ValueError("logical positions changed during injection")
    payload.snapshot = None
    del snapshot, packed

    def allocated() -> int:
        pairs, _ = adapter._legacy_layers(state.past_key_values)
        return backing_bytes(tuple(t for pair in pairs for t in pair))

    initial = peak = allocated()
    values = [prefill.logits.detach().reshape(-1).float().cpu()]
    token = prefill.next_token_id.detach().clone()
    for step in range(predictions - 1):
        if forced_prefix is not None:
            token = torch.full_like(prefill.next_token_id, forced_prefix[step])
        output = adapter.decode_one_token(token, state, capture_queries=False)
        values.append(output.logits.detach().reshape(-1).float().cpu())
        token = output.next_token_id
        peak = max(peak, allocated())
    logits = torch.stack(values)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite diagnostic logits")
    return logits, dict(
        initial_active_kv_storage_bytes=initial,
        final_active_kv_storage_bytes=allocated(),
        peak_active_kv_storage_bytes=peak,
        boundary="native_KV_at_completed_decode_boundaries_not_whole_request_peak",
        physical_padding_bytes=0,
        original_logical_sequence_length=logical,
        initial_next_decode_position=next_position,
        final_next_decode_position=state.next_decode_position,
        fixed_FullKV_prefix_evaluation_only=forced_prefix is not None,
        cache_dependent_predictions=predictions - 1,
        original_prefill_retained_for_diagnostic=True,
        no_end_to_end_or_memory_superiority_claim=True,
    )


def comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    """FP64 diagnostic KL; registered FP32 guard is retained alongside it."""
    if reference.shape != candidate.shape or not bool(torch.isfinite(candidate).all()):
        raise ValueError("finite identical logit shapes required")
    a, b = reference.double().log_softmax(-1), candidate.double().log_softmax(-1)
    kl = (a.exp() * (a - b)).sum(-1)
    fa, fb = reference.float().log_softmax(-1), candidate.float().log_softmax(-1)
    registered = (fa.exp() * (fa - fb)).sum(-1)
    delta = (reference.double() - candidate.double()).abs().amax(-1)
    same = reference.argmax(-1) == candidate.argmax(-1)
    mismatches = torch.nonzero(~same).flatten().tolist()
    return dict(
        per_prefix_KL_FP64=kl.tolist(),
        per_prefix_KL_FP32=registered.tolist(),
        per_prefix_maximum_logit_difference=delta.tolist(),
        per_prefix_greedy_agreement=same.tolist(),
        mean_KL_FP64=float(kl.mean()),
        maximum_KL_FP32=float(registered.max()),
        maximum_logit_difference=float(delta.max()),
        token_agreement=float(same.double().mean()),
        first_divergent_prediction=None if not mismatches else mismatches[0],
        registered_guard_pass=bool(same.all() and registered.max() <= 0.02 and delta.max() <= 1.0),
    )
