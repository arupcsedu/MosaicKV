"""Bounded feasibility transport: lossless exact prompt, raw append-only history.

Decode only the current layer into temporary BF16 K/V, run the unmodified HF
eager attention, then release that layer's temporary buffers. No weights or
attention arithmetic are changed. This is not a production cache backend.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from transformers.cache_utils import CacheLayerMixin, DynamicCache

from mosaickv.evaluation.lossless_bf16 import (
    PackedBF16,
    pack_cpu,
    unpack_cpu,
    unpack_into,
    unpack_pair_into,
)


def backing_bytes(tensors: tuple[Any, ...]) -> int:
    """Unique backing allocations, not views or logical tensor payloads."""
    owners = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        owners[(str(tensor.device), storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(owners.values())


class PromptStorageLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(
        self, key: PackedBF16, value: PackedBF16, *, observe: Any, fused_unpack: bool = False
    ) -> None:
        import torch

        super().__init__()
        if key.shape != value.shape or len(key.shape) != 4 or key.shape[0] != 1:
            raise ValueError("same-shape batch-one prompt storage required")
        self.prompt = key, value
        self.prompt_length = key.shape[-2]
        self.history_length = 0
        self.history_capacity = 0
        shape = (*key.shape[:2], 0, key.shape[-1])
        device = key.buffers[0].device
        self.history = tuple(
            torch.empty(shape, dtype=torch.bfloat16, device=device) for _ in range(2)
        )
        self.leased: tuple[Any, ...] = ()
        self.resident_bytes = key.storage_bytes + value.storage_bytes
        self.history_bytes = 0
        self.lease_bytes = 0
        self.observe = observe
        self.eviction_count = 0
        self.fused_unpack = fused_unpack

    def owned_tensors(self) -> tuple[Any, ...]:
        return (*self.prompt[0].buffers, *self.prompt[1].buffers, *self.history, *self.leased)

    def update(self, key_states: Any, value_states: Any, cache_kwargs: Any = None) -> Any:
        import torch

        del cache_kwargs
        expected = (*self.prompt[0].shape[:2], 1, self.prompt[0].shape[-1])
        if (
            self.leased
            or key_states.shape != expected
            or value_states.shape != expected
            or key_states.dtype != torch.bfloat16
            or value_states.dtype != torch.bfloat16
        ):
            raise ValueError("one BF16 token and released previous materialization required")
        if self.history_length >= 32:
            raise ValueError("this feasibility transport is restricted to 32 decode updates")
        needed = self.history_length + 1
        if needed > self.history_capacity:
            capacity = min(32, ((needed + 15) // 16) * 16)
            shape = (*key_states.shape[:2], capacity, key_states.shape[-1])
            newer = tuple(
                torch.empty(shape, dtype=key_states.dtype, device=key_states.device)
                for _ in range(2)
            )
            self.observe(newer)  # old and new history coexist during growth
            for dest, source in zip(newer, self.history, strict=True):
                dest[..., : self.history_length, :].copy_(source[..., : self.history_length, :])
            self.history, self.history_capacity = newer, capacity
            newer_bytes = backing_bytes(newer)
            self.resident_bytes += newer_bytes - self.history_bytes
            self.history_bytes = newer_bytes
        for dest, source in zip(self.history, (key_states, value_states), strict=True):
            dest[..., self.history_length : self.history_length + 1, :].copy_(source)
        self.history_length += 1
        shape = (*key_states.shape[:2], self.get_seq_length(), key_states.shape[-1])
        self.leased = tuple(
            torch.empty(shape, dtype=key_states.dtype, device=key_states.device) for _ in range(2)
        )
        self.lease_bytes = backing_bytes(self.leased)
        self.observe(())
        if self.fused_unpack:
            unpack_pair_into(self.prompt, self.history, self.leased)
        else:
            for body, dest, history in zip(self.prompt, self.leased, self.history, strict=True):
                unpack_into(body, dest)
                dest[..., self.prompt_length :, :].copy_(history[..., : self.history_length, :])
        return self.leased

    def release(self) -> None:
        self.leased = ()
        self.lease_bytes = 0

    def get_seq_length(self, cache_position: Any = None) -> int:
        del cache_position
        return self.prompt_length + self.history_length

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position: Any) -> tuple[int, int]:
        return self.get_seq_length() + cache_position.shape[0], 0


class PromptStorageCache(DynamicCache):
    """Count compressed bodies, raw histories and temporary materializations."""

    def __init__(self) -> None:
        super().__init__()
        self.layers.clear()
        self.initial_bytes = self.peak_bytes = 0
        self.observations = 0
        self.build_records: list[dict[str, Any]] = []

    def observe(self, extras: tuple[Any, ...] = ()) -> None:
        current = sum(layer.resident_bytes + layer.lease_bytes for layer in self.layers)
        self.peak_bytes = max(self.peak_bytes, current + backing_bytes(extras))
        self.observations += 1

    def storage_record(self) -> dict[str, Any]:
        self.observe()
        actual = backing_bytes(tuple(t for layer in self.layers for t in layer.owned_tensors()))
        if actual != sum(layer.resident_bytes + layer.lease_bytes for layer in self.layers):
            raise RuntimeError("cached accounting differs from independent storage enumeration")
        return dict(
            initial_active_kv_storage_bytes=self.initial_bytes,
            final_active_kv_storage_bytes=backing_bytes(
                tuple(t for layer in self.layers for t in layer.owned_tensors())
            ),
            peak_active_kv_storage_bytes=self.peak_bytes,
            boundary="post_construction_active_decode_backing_including_layer_materialization",
            includes="encoded_headers_and_exponents,raw_fallbacks,raw_generated_history,history_growth_overlap,temporary_full_layer_KV",
            excludes="inactive_full_prefill_source,construction_workspaces,weights,attention_activations,allocator_reserved",
            construction_peak_certified=False,
            whole_request_peak_certified=False,
            generated_history_lossless_and_append_only=True,
            history_updates_per_layer=[layer.history_length for layer in self.layers],
            outstanding_layer_leases=sum(bool(layer.leased) for layer in self.layers),
            observations=self.observations,
            prompt_construction=self.build_records,
        )


def construct_cache(
    adapter: Any,
    source_cache: Any,
    positions: Any,
    *,
    encode: bool,
    gpu_encoder: bool = False,
    fused_unpack: bool = False,
) -> PromptStorageCache:
    """Gather and encode ONE source layer at a time; never change source values."""
    import torch

    from mosaickv.adapters.huggingface.bounded_history_cache import assert_bounded_history_support

    assert_bounded_history_support(adapter)
    source, _ = adapter._legacy_layers(source_cache)
    if len(source) != len(positions):
        raise ValueError("complete saved per-layer/head positions required")
    cache = PromptStorageCache()
    for layer_index, ((key, value), heads) in enumerate(zip(source, positions, strict=True)):
        if (
            key.dtype != torch.bfloat16
            or key.shape != value.shape
            or key.shape[0] != 1
            or len(heads) != key.shape[1]
            or len({len(row) for row in heads}) != 1
            or any(
                not row
                or list(sorted(set(row))) != list(row)
                or row[0] < 0
                or row[-1] >= key.shape[-2]
                for row in heads
            )
        ):
            raise ValueError("sorted unique unpadded BF16 source selections required")
        index = torch.tensor(heads, device=key.device).unsqueeze(0).unsqueeze(-1)
        index = index.expand(1, key.shape[1], len(heads[0]), key.shape[-1])
        gathered = (key.gather(2, index), value.gather(2, index))
        bodies = []
        for tensor in gathered:
            if encode:
                if gpu_encoder:
                    from mosaickv.evaluation.lossless_bf16_kernel import encode as pack_gpu

                    body = pack_gpu(tensor)
                else:
                    cpu = tensor.cpu()
                    packed = pack_cpu(cpu)
                    if not torch.equal(unpack_cpu(packed).view(torch.int16), cpu.view(torch.int16)):
                        raise RuntimeError("CPU codec changed a source bit")
                    body = packed.to(tensor.device)
            else:
                body = PackedBF16(tuple(tensor.shape), True, (tensor,))
            bodies.append(body)
        cache.layers.append(
            PromptStorageLayer(*bodies, observe=cache.observe, fused_unpack=fused_unpack)
        )
        cache.build_records.append(
            dict(
                layer=layer_index,
                original_prompt_bytes=sum(t.numel() * t.element_size() for t in gathered),
                stored_prompt_bytes=sum(b.storage_bytes for b in bodies),
                raw_fallback_tensors=sum(b.raw for b in bodies),
                cpu_roundtrip_bitwise=not gpu_encoder,
                constructor="gpu_integer_v1" if gpu_encoder else "cpu_reference_v1",
            )
        )
    cache.initial_bytes = backing_bytes(
        tuple(t for layer in cache.layers for t in layer.owned_tensors())
    )
    cache.observe()
    return cache


class IsolatedStorageTransport:
    """Scope cache-length dispatch and prompt-buffer release to one diagnostic."""

    def __init__(self, adapter: Any, cache: PromptStorageCache) -> None:
        self.adapter, self.cache, self.stack = adapter, cache, ExitStack()

    def __enter__(self) -> IsolatedStorageTransport:
        original = self.adapter._cache_length

        def length(cache: Any) -> int:
            if cache is self.cache:
                return max(layer.get_seq_length() for layer in cache.layers)
            return int(original(cache))

        self.stack.enter_context(patch.object(self.adapter, "_cache_length", length))
        for layer, attention in zip(
            self.cache.layers, self.adapter._language_layers(), strict=True
        ):

            def release(_module: Any, _args: Any, _output: Any, *, target: Any = layer) -> None:
                target.release()

            handle = attention.self_attn.register_forward_hook(release)
            self.stack.callback(handle.remove)
        return self

    def __exit__(self, *args: Any) -> Any:
        try:
            return self.stack.__exit__(*args)
        finally:
            for layer in self.cache.layers:
                layer.release()
