"""Fixed-storage exact prompt and generated sink/recent K/V transport.

Only the audited Transformers 4.55.2 LLaVA-1.5 eager decoder is admitted.
Post-RoPE keys are copied unchanged; physical ring slots NEVER become RoPE
positions. The circular suffix is deliberately not chronological. Single-token
attention is permutation invariant when K, V and their validity move together;
no multi-token/batched probe API is supported by this transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter
    from mosaickv.adapters.huggingface.types import DecodeState


@dataclass(frozen=True, slots=True)
class BoundedHistoryBudget:
    cap_bytes: int
    bytes_per_position: int
    generated_positions: int
    reserved_generated_bytes: int
    prompt_budget_bytes: int
    lattice_shortfall_bytes: int


def bounded_history_budget(
    cap_bytes: int,
    bytes_per_position: int,
    generated_capacity: int,
    decode_updates: int,
) -> BoundedHistoryBudget:
    """Reserve lifetime capacity, on the real all-layer/head storage lattice."""

    for name, value, minimum in (
        ("cap_bytes", cap_bytes, 1),
        ("bytes_per_position", bytes_per_position, 1),
        ("generated_capacity", generated_capacity, 1),
        ("decode_updates", decode_updates, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    generated = min(generated_capacity, decode_updates)
    slots, shortfall = divmod(cap_bytes, bytes_per_position)
    if slots <= generated:
        raise ValueError("lifetime byte cap cannot fit generated reserve plus one prompt position")
    return BoundedHistoryBudget(
        cap_bytes=cap_bytes,
        bytes_per_position=bytes_per_position,
        generated_positions=generated,
        reserved_generated_bytes=generated * bytes_per_position,
        prompt_budget_bytes=(slots - generated) * bytes_per_position,
        lattice_shortfall_bytes=shortfall,
    )


class BoundedHistoryLayer:
    """One fixed allocation: immutable prompt, generated sinks, recent ring."""

    is_sliding = False
    is_compileable = False

    def __init__(
        self,
        keys: Any,
        values: Any,
        *,
        generated_capacity: int,
        sink_tokens: int,
        next_logical_position: int,
    ) -> None:
        if keys.ndim != 4 or keys.shape != values.shape or int(keys.shape[0]) != 1:
            raise ValueError("bounded history requires matching [1, KV heads, sequence, dim] K/V")
        if keys.dtype != values.dtype or keys.device != values.device:
            raise ValueError("bounded history requires matching K/V dtype and device")
        if type(generated_capacity) is not int or generated_capacity < 1:
            raise ValueError("generated_capacity must be a positive integer")
        if type(sink_tokens) is not int or not 0 <= sink_tokens < generated_capacity:
            raise ValueError("sink_tokens must be in [0, generated_capacity)")
        if type(next_logical_position) is not int or next_logical_position < int(keys.shape[-2]):
            raise ValueError("next logical position cannot precede the prompt cache length")
        self.prompt_length = int(keys.shape[-2])
        if self.prompt_length < 1:
            raise ValueError("bounded history requires a nonempty protected prompt")
        self.generated_capacity = generated_capacity
        self.sink_tokens = sink_tokens
        self._capacity = self.prompt_length + generated_capacity
        self._length = self.prompt_length
        self._updates = 0
        self._next_logical_position = next_logical_position
        self._generated_positions: list[int] = []
        shape = (*keys.shape[:-2], self._capacity, int(keys.shape[-1]))
        self._key_storage = keys.new_empty(shape)
        self._value_storage = values.new_empty(shape)
        self._key_storage[..., : self.prompt_length, :].copy_(keys)
        self._value_storage[..., : self.prompt_length, :].copy_(values)
        self.keys = self._key_storage[..., : self._length, :]
        self.values = self._value_storage[..., : self._length, :]

    def update(
        self,
        key_states: Any,
        value_states: Any,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        del cache_kwargs  # Already-rotated K/V; positions come from adapter inputs.
        expected = (*self.keys.shape[:-2], 1, self.keys.shape[-1])
        if tuple(key_states.shape) != expected or tuple(value_states.shape) != expected:
            raise ValueError("bounded history admits exactly one token and unchanged batch/heads")
        if (
            key_states.dtype != self.keys.dtype
            or value_states.dtype != self.values.dtype
            or key_states.device != self.keys.device
            or value_states.device != self.values.device
        ):
            raise ValueError("bounded history updates must preserve dtype and device")
        if self._updates < self.generated_capacity:
            generated_slot = self._updates
            self._generated_positions.append(self._next_logical_position)
        else:
            recent = self.generated_capacity - self.sink_tokens
            generated_slot = self.sink_tokens + (self._updates - self.sink_tokens) % recent
            self._generated_positions[generated_slot] = self._next_logical_position
        slot = self.prompt_length + generated_slot
        self._key_storage[..., slot : slot + 1, :].copy_(key_states)
        self._value_storage[..., slot : slot + 1, :].copy_(value_states)
        self._updates += 1
        self._next_logical_position += 1
        self._length = min(self._capacity, self.prompt_length + self._updates)
        self.keys = self._key_storage[..., : self._length, :]
        self.values = self._value_storage[..., : self._length, :]
        return self.keys, self.values

    @property
    def generated_logical_positions(self) -> tuple[int, ...]:
        """Physical-slot order, not a renumbered or sorted logical sequence."""

        return tuple(self._generated_positions)

    @property
    def evicted_positions(self) -> int:
        return max(0, self._updates - self.generated_capacity)

    def get_seq_length(self, cache_position: Any | None = None) -> int:
        del cache_position
        return self._length

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position: Any) -> tuple[int, int]:
        if int(cache_position.numel()) != 1:
            raise ValueError("bounded history masks require single-token decode")
        return min(self._capacity, self._length + 1), 0

    def crop(self, max_length: int) -> None:
        del max_length
        raise NotImplementedError(
            "bounded ring rollback requires an isolated cache; crop is unsafe"
        )

    def reorder_cache(self, beam_idx: Any) -> None:
        del beam_idx
        raise NotImplementedError("bounded history does not support beam/batch reordering")


def assert_bounded_history_support(adapter: HuggingFaceMultimodalAdapter) -> None:
    import transformers
    from transformers.models.llama.modeling_llama import LlamaAttention

    layers = tuple(adapter._language_layers())
    if (
        transformers.__version__ != "4.55.2"
        or adapter.capabilities.model_family != "llava_1_5"
        or not layers
        or any(type(layer.self_attn) is not LlamaAttention for layer in layers)
        or any(layer.self_attn.config._attn_implementation != "eager" for layer in layers)
    ):
        raise RuntimeError("bounded history requires audited Transformers 4.55.2 LLaVA-1.5 eager")


def install_bounded_history(
    adapter: HuggingFaceMultimodalAdapter,
    state: DecodeState,
    *,
    generated_capacity: int,
    sink_tokens: int,
    cap_bytes: int,
) -> dict[str, Any]:
    """Install after prompt selection, before accounting or measured decoding."""

    import torch
    from transformers import DynamicCache

    from mosaickv.measurements.cache_lifecycle import CacheLifecycle

    assert_bounded_history_support(adapter)
    if (
        state.model_state.get("mosaickv_packed_all_valid") is not True
        or state.model_state.get("mosaickv_validity_masks") is not None
        or state.model_state.get("prefixkv_ragged")
        or state.model_state.get("mosaickv_layerwise_exact")
        or state.model_state.get("mosaickv_prototype_slots")
        or state.model_state.get("mosaickv_bounded_history")
    ):
        raise RuntimeError("bounded history requires an unpadded exact prompt and no prior ring")
    legacy, _kind = adapter._legacy_layers(state.past_key_values)
    lengths = {int(key.shape[-2]) for key, _value in legacy}
    if lengths != {state.active_cache_length}:
        raise RuntimeError("bounded history requires uniform per-layer physical lengths")
    bytes_per_position = sum(
        (int(key.numel()) * key.element_size() + int(value.numel()) * value.element_size())
        // state.active_cache_length
        for key, value in legacy
    )
    required = (state.active_cache_length + generated_capacity) * bytes_per_position
    if required > cap_bytes:
        raise RuntimeError(f"bounded cache would exceed lifetime cap: {required} > {cap_bytes}")
    cache = DynamicCache()
    cache_layers = [
        BoundedHistoryLayer(
            key,
            value,
            generated_capacity=generated_capacity,
            sink_tokens=sink_tokens,
            next_logical_position=state.next_decode_position,
        )
        for key, value in legacy
    ]
    cast("Any", cache)._mosaickv_bounded_history = True
    cache.layers = cast("Any", cache_layers)
    payload, storage = CacheLifecycle.count(tuple((x.keys, x.values) for x in cache_layers))
    if storage != required:
        raise RuntimeError("bounded cache storage does not match its physical allocation")
    state.past_key_values = cache
    state.model_state["mosaickv_bounded_history"] = True
    state.model_state["mosaickv_bounded_history_capacity"] = (
        state.active_cache_length + generated_capacity
    )
    state.model_state["mosaickv_bounded_history_masks"] = tuple(
        torch.zeros((1, 1, 1, layer._capacity), dtype=layer.keys.dtype, device=layer.keys.device)
        for layer in cache_layers
    )
    state.model_state["mosaickv_decode_cache_storage"] = "bounded_generated_ring_v1"
    return {
        "status": "active_exact_prompt_sink_recent_variant",
        "lifetime_cap_bytes": cap_bytes,
        "protected_prompt_positions_per_head": state.active_cache_length,
        "generated_capacity": generated_capacity,
        "sink_tokens": sink_tokens,
        "recent_tokens": generated_capacity - sink_tokens,
        "initial_payload_bytes": payload,
        "initial_storage_bytes": storage,
        "unused_cap_bytes": cap_bytes - storage,
        "physical_suffix_order": "circular_use_recorded_original_logical_positions",
        "rope_keys_transformed": False,
    }


def bounded_history_masks(state: DecodeState) -> tuple[Any, ...]:
    capacity = int(state.model_state["mosaickv_bounded_history_capacity"])
    length = min(capacity, state.active_cache_length + 1)
    return tuple(mask[..., :length] for mask in state.model_state["mosaickv_bounded_history_masks"])


def bounded_history_record(state: DecodeState) -> dict[str, Any]:
    layers = state.past_key_values.layers
    positions = [layer.generated_logical_positions for layer in layers]
    if not positions or any(row != positions[0] for row in positions):
        raise RuntimeError("bounded history layer positions diverged")
    return {
        "evicted_generated_positions_per_head": layers[0].evicted_positions,
        "retained_generated_logical_positions_physical_order": list(positions[0]),
        "next_decode_position": state.next_decode_position,
        "logical_sequence_length": state.logical_sequence_length,
        "active_cache_length": state.active_cache_length,
        "prompt_slots_unchanged": True,
    }
