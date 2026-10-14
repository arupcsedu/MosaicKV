"""Opt-in ragged PrefixKV transport for pinned HF LLaVA-1.5 eager attention.

Each layer owns dense [1, Hkv, N_layer, D] tensors. No cross-layer padding,
key re-rotation, global cache renumbering, or model-weight mutation is used.
Only single-token decode is admitted. This is a local reimplementation, not
the official legacy LLaVA/Transformers PrefixKV execution environment.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from mosaickv.adapters.huggingface.types import CacheLayerSnapshot, CacheSnapshot

if TYPE_CHECKING:
    from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter
    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.baselines.prefixkv import PrefixKVCompressionPlan
    from mosaickv.types import JsonObject


def assert_ragged_support(adapter: HuggingFaceMultimodalAdapter) -> None:
    """Fail closed against uninspected kernels, models, and cache APIs."""
    import transformers
    from transformers.models.llama.modeling_llama import LlamaAttention

    if transformers.__version__ != "4.55.2":
        raise RuntimeError("ragged PrefixKV requires audited Transformers 4.55.2")
    layers = tuple(adapter._language_layers())
    if adapter.capabilities.model_family != "llava_1_5" or not layers:
        raise RuntimeError("ragged PrefixKV supports only the LLaVA-1.5 adapter")
    if any(type(layer.self_attn) is not LlamaAttention for layer in layers):
        raise RuntimeError("ragged PrefixKV requires the audited LlamaAttention class")
    if any(layer.self_attn.config._attn_implementation != "eager" for layer in layers):
        raise RuntimeError("ragged PrefixKV supports eager attention only")


def build_ragged_snapshot(
    adapter: HuggingFaceMultimodalAdapter,
    plan: PrefixKVCompressionPlan,
    original_cache: Any,
) -> CacheSnapshot:
    """Gather exact selected positions once per layer, with no padding."""
    import torch
    from transformers.cache_utils import DynamicCache

    assert_ragged_support(adapter)
    if type(original_cache) is not DynamicCache:
        raise RuntimeError("ragged PrefixKV requires an ordinary DynamicCache prefill")
    legacy, _ = adapter._legacy_layers(original_cache)
    if len(legacy) != len(plan.layers):
        raise RuntimeError("PrefixKV plan/cache layer count mismatch")
    layers = []
    for (key, value), layer in zip(legacy, plan.layers, strict=True):
        if key.ndim != 4 or key.shape[0] != 1 or key.shape != value.shape:
            raise RuntimeError("ragged PrefixKV requires batch-one matching BHSD tensors")
        if int(key.shape[-2]) != layer.source_positions:
            raise RuntimeError("PrefixKV source length mismatch")
        indices = torch.tensor(layer.selected_physical_positions, device=key.device)
        layers.append(
            CacheLayerSnapshot(
                key.index_select(-2, indices), value.index_select(-2, indices.to(value.device)), 2
            )
        )
    snapshot = CacheSnapshot(
        tuple(layers),
        DynamicCache,
        "cache_object",
        max(plan.layer_cache_sizes),
        adapter.capabilities.cached_key_state,
        independent_layer_lengths=True,
    )
    actual_bytes = sum(
        tensor.untyped_storage().nbytes()
        for layer in snapshot.layers
        for tensor in (layer.key, layer.value)
    )
    if actual_bytes != plan.retained_bytes:
        raise RuntimeError("ragged PrefixKV physical storage differs from planned bytes")
    return snapshot


def ragged_decode_masks(
    adapter: HuggingFaceMultimodalAdapter, state: DecodeState
) -> tuple[Any, ...]:
    """All retained keys precede this one query; give each layer its own extent."""
    import torch

    assert_ragged_support(adapter)
    if getattr(state.past_key_values, "_mosaickv_independent_layer_lengths", False) is not True:
        raise RuntimeError("ragged PrefixKV cache is missing its explicit layout marker")
    legacy, _ = adapter._legacy_layers(state.past_key_values)
    positions = state.model_state.get("prefixkv_logical_positions")
    if not isinstance(positions, tuple) or len(positions) != len(legacy):
        raise RuntimeError("ragged PrefixKV logical positions are missing")
    masks = []
    for (key, value), logical in zip(legacy, positions, strict=True):
        if key.shape != value.shape or int(key.shape[-2]) != len(logical):
            raise RuntimeError("ragged PrefixKV layer length metadata is inconsistent")
        if not logical or logical[-1] >= state.next_decode_position:
            raise RuntimeError("ragged PrefixKV logical positions reach a future token")
        if any(a >= b for a, b in pairwise(logical)):
            raise RuntimeError("ragged PrefixKV logical positions are not strictly increasing")
        masks.append(torch.zeros((1, 1, 1, len(logical) + 1), dtype=key.dtype, device=key.device))
    return tuple(masks)


def evict_ragged_prefixkv(
    adapter: HuggingFaceMultimodalAdapter,
    state: DecodeState,
    plan: PrefixKVCompressionPlan,
    *,
    step_index: int,
) -> tuple[DecodeState, tuple[JsonObject, ...]]:
    """At-most-one eviction per layer; question protection is an explicit variant."""
    import torch

    legacy, _ = adapter._legacy_layers(state.past_key_values)
    all_positions = state.model_state["prefixkv_logical_positions"]
    positions_after = []
    layers = []
    events: list[JsonObject] = []
    for (key, value), logical, layer in zip(legacy, all_positions, plan.layers, strict=True):
        if key.shape[-2] != len(logical):
            raise RuntimeError("ragged PrefixKV eviction position metadata mismatch")
        target = (
            layer.retained_positions + state.logical_sequence_length - layer.source_positions
            if plan.config.decode_policy == "append_only"
            else state.logical_sequence_length * layer.decode_retention_ratio
        )
        offset = None
        removed_logical = None
        if (
            plan.config.decode_policy != "append_only"
            and math.trunc(len(logical) - target) > 0
            and len(logical) > 1
        ):
            offset = min(layer.eviction_offset, len(logical) - 1)
            if plan.prompt_reservation_policy == "question_and_boundaries":
                protected = {
                    logical_id
                    for physical_id, logical_id in zip(
                        layer.selected_physical_positions,
                        layer.selected_logical_positions,
                        strict=True,
                    )
                    if physical_id in layer.protected_positions
                }
                if not protected <= set(logical):
                    raise RuntimeError("PrefixKV lost protected question evidence before eviction")
                eligible = [
                    i
                    for i, p in enumerate(logical)
                    if p not in protected and i < len(logical) - plan.config.protect_size
                ]
                if not eligible:
                    raise RuntimeError("PrefixKV has no unprotected eviction candidate")
                # Explicit variant: nearest eligible offset, deterministic lower-index tie break.
                requested_offset = offset
                offset = min(eligible, key=lambda i: (abs(i - requested_offset), i))
            removed_logical = logical[offset]
            if plan.config.decode_boundary_policy == "protected" and (
                offset < plan.config.start_size
                or offset >= len(logical) - plan.config.protect_size
                or removed_logical in layer.selected_logical_positions[: plan.config.start_size]
            ):
                raise RuntimeError("PrefixKV decode would evict a protected start/tail boundary")
            keep = torch.cat(
                (
                    torch.arange(offset, device=key.device),
                    torch.arange(offset + 1, len(logical), device=key.device),
                )
            )
            key = key.index_select(-2, keep)
            value = value.index_select(-2, keep.to(value.device))
            logical = logical[:offset] + logical[offset + 1 :]
        layers.append(CacheLayerSnapshot(key, value, 2))
        positions_after.append(logical)
        events.append(
            {
                "step": step_index,
                "layer": layer.layer,
                "logical_sequence_length": state.logical_sequence_length,
                "target_active_positions": target,
                "eviction_offset": layer.eviction_offset,
                "removed_physical_position": offset,
                "removed_logical_position": removed_logical,
                "active_positions_after": len(logical),
                "physical_bytes_after": key.untyped_storage().nbytes()
                + value.untyped_storage().nbytes(),
                "decode_boundary_policy": plan.config.decode_boundary_policy,
                "decode_policy": plan.config.decode_policy,
                "prompt_reservation_policy": plan.prompt_reservation_policy,
                "protected_start_present": all(
                    position in logical
                    for position in layer.selected_logical_positions[: plan.config.start_size]
                ),
            }
        )
    # Rebuild the lightweight cache container; unmodified layers alias their
    # owned tensors. Changed layers have exact-size index_select storage.
    snapshot = CacheSnapshot(
        tuple(layers),
        type(state.past_key_values),
        "cache_object",
        max(len(positions) for positions in positions_after),
        adapter.capabilities.cached_key_state,
        independent_layer_lengths=True,
    )
    state.past_key_values = adapter.inject_past_key_values(snapshot)
    state.active_cache_length = snapshot.active_sequence_length  # maximum, not logical length
    state.model_state["prefixkv_logical_positions"] = tuple(positions_after)
    state.attention_mask = torch.ones(
        (1, state.active_cache_length), dtype=state.attention_mask.dtype, device=key.device
    )
    return state, tuple(events)
