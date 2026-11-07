"""Audited layer-ragged exact-cache transport for LLaVA-NeXT Mistral.

Every layer remains dense across all KV heads.  Only complete layer slots may
move between layers, so this transport never relies on hidden per-head padding.
The retained tensors contain original K/V values in logical-position order.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter
    from mosaickv.adapters.huggingface.types import DecodeState


def assert_layerwise_exact_support(adapter: HuggingFaceMultimodalAdapter) -> None:
    """Fail closed outside the inspected eager LLaVA-NeXT/Mistral path."""

    import transformers
    from transformers.models.mistral.modeling_mistral import MistralAttention

    if transformers.__version__ != "4.55.2":
        raise RuntimeError("layerwise exact cache requires audited Transformers 4.55.2")
    layers = tuple(adapter._language_layers())
    if adapter.capabilities.model_family != "llava_next_mistral" or not layers:
        raise RuntimeError("layerwise exact cache supports only LLaVA-NeXT Mistral")
    if any(type(layer.self_attn) is not MistralAttention for layer in layers):
        raise RuntimeError("layerwise exact cache requires the audited MistralAttention class")
    if any(layer.self_attn.config._attn_implementation != "eager" for layer in layers):
        raise RuntimeError("layerwise exact cache supports eager attention only")


def layerwise_exact_decode_masks(
    adapter: HuggingFaceMultimodalAdapter,
    state: DecodeState,
) -> tuple[Any, ...]:
    """Give each dense layer its exact independent key extent during decode."""

    import torch

    assert_layerwise_exact_support(adapter)
    if getattr(state.past_key_values, "_mosaickv_independent_layer_lengths", False) is not True:
        raise RuntimeError("layerwise exact cache lacks its independent-length marker")
    legacy, _ = adapter._legacy_layers(state.past_key_values)
    logical_by_layer = state.model_state.get("mosaickv_layerwise_exact_logical_positions")
    if not isinstance(logical_by_layer, tuple) or len(logical_by_layer) != len(legacy):
        raise RuntimeError("layerwise exact logical-position metadata is missing")
    masks = []
    for layer_index, ((key, value), logical_by_head) in enumerate(
        zip(legacy, logical_by_layer, strict=True)
    ):
        if not isinstance(logical_by_head, tuple) or not logical_by_head:
            raise RuntimeError(f"layer {layer_index} lacks per-head logical positions")
        layer_length = int(key.shape[-2])
        if key.shape != value.shape or any(len(row) != layer_length for row in logical_by_head):
            raise RuntimeError("layerwise exact length metadata is inconsistent")
        for row in logical_by_head:
            if not row or row[-1] >= state.next_decode_position:
                raise RuntimeError("layerwise exact positions reach a future token")
            if any(a >= b for a, b in pairwise(row)):
                raise RuntimeError("layerwise exact positions are not strictly increasing")
        masks.append(
            torch.zeros(
                (1, 1, 1, layer_length + 1),
                dtype=key.dtype,
                device=key.device,
            )
        )
    return tuple(masks)


__all__ = ["assert_layerwise_exact_support", "layerwise_exact_decode_masks"]
