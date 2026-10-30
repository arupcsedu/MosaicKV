"""Request-isolated experimental HF transport for integer-coded K/V attention.

No full K/V attention lease is created. Raw generated history stays append-only;
this variant changes neither selected evidence nor positions nor model weights.
"""

from contextlib import ExitStack
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

import torch

from mosaickv.evaluation.lossless_bf16 import PackedBF16, pack_cpu
from mosaickv.evaluation.lossless_prompt_cache import (
    IsolatedStorageTransport,
    PromptStorageCache,
    PromptStorageLayer,
    backing_bytes,
)
from mosaickv.evaluation.packed_attention import AttentionPacket, attend, packet


class PackedAttentionCache(PromptStorageCache):
    def storage_record(self) -> dict[str, Any]:
        record = super().storage_record()
        record.update(
            boundary="post_construction_packed_KV_history_backing_and_growth_overlap",
            includes="encoded_payload,raw_fallbacks,literal_checkpoints,raw_history,history_growth_overlap",
            excludes="inactive_prefill_and_selected_reference,weights,attention_activations,internal_kernel_workspace,allocator_reserved,constructor_temporaries",
            full_KV_lease_bytes=0,
            attention_arithmetic="experimental_reduction_order_requires_separate_numerical_gate",
        )
        return record


class PackedAttentionLayer(PromptStorageLayer):
    history: tuple[Any, ...]
    history_capacity: int
    history_bytes: int

    def __init__(
        self,
        key: AttentionPacket,
        value: AttentionPacket,
        *,
        observe: Any,
        maximum_decode_updates: int,
    ) -> None:
        super().__init__(
            key.payload,
            value.payload,
            observe=observe,
            maximum_decode_updates=maximum_decode_updates,
        )
        self.packets = key, value
        self.resident_bytes += backing_bytes((key.checkpoints, value.checkpoints))

    def owned_tensors(self) -> tuple[Any, ...]:
        return (*super().owned_tensors(), *(p.checkpoints for p in self.packets))

    def update(self, key_states: Any, value_states: Any, cache_kwargs: Any = None) -> Any:
        del cache_kwargs
        expected = (*self.prompt_shape[:2], 1, 128)
        if (
            self.leased
            or key_states.shape != expected
            or value_states.shape != expected
            or any(
                t.dtype != torch.bfloat16 or t.device != self.history[0].device
                for t in (key_states, value_states)
            )
            or self.history_length >= self.maximum_decode_updates
        ):
            raise ValueError(
                "one same-device BF16 MHA token within the declared history bound required"
            )
        needed = self.history_length + 1
        if needed > self.history_capacity:
            capacity = min(self.maximum_decode_updates, ((needed + 15) // 16) * 16)
            shape = (*key_states.shape[:2], capacity, 128)
            newer = tuple(
                torch.empty(shape, dtype=t.dtype, device=t.device)
                for t in (key_states, value_states)
            )
            self.observe(newer)
            for target, source in zip(newer, self.history, strict=True):
                target[..., : self.history_length, :].copy_(source[..., : self.history_length, :])
            del target, source
            self.history = newer
            new_bytes = backing_bytes(newer)
            self.resident_bytes += new_bytes - self.history_bytes
            self.history_bytes, self.history_capacity = new_bytes, capacity
        for target, source in zip(self.history, (key_states, value_states), strict=True):
            target[..., self.history_length : self.history_length + 1, :].copy_(source)
        self.history_length += 1
        if self is getattr(self, "cache_first_layer", None):
            self.mask_certificates.clear()
        self.observe()
        # Only the explicitly gated attention dispatch may consume this return.
        # It reads the packet/history owners, not these one-token placeholders.
        return key_states, value_states

    def materialize(self) -> None:
        raise RuntimeError("packed attention must never materialize a full K/V lease")


def construct(pairs: Any, *, encoded: bool, updates: int) -> PromptStorageCache:
    if type(encoded) is not bool or type(updates) is not int or not 1 <= updates <= 4096:
        raise ValueError("explicit encoded flag and positive bounded updates required")
    cache = PackedAttentionCache()
    for index, (key, value) in enumerate(pairs):
        if (
            key.shape != value.shape
            or key.ndim != 4
            or key.shape[0] != 1
            or key.shape[-1] != 128
            or key.dtype != torch.bfloat16
            or value.dtype != key.dtype
            or value.device != key.device
        ):
            raise ValueError("same-shape batch-one BF16 BHN128 K/V required")
        components = []
        for tensor in (key, value):
            tensor = tensor.contiguous()
            if encoded:
                if tensor.device.type == "cuda":
                    from mosaickv.evaluation.lossless_bf16_kernel import encode

                    body = encode(tensor)
                else:
                    body = pack_cpu(tensor)
            else:
                body = PackedBF16(tuple(tensor.shape), True, (tensor.clone(),))
            components.append(packet(body))
        layer = PackedAttentionLayer(
            *components, observe=cache.observe, maximum_decode_updates=updates
        )
        cache.layers.append(layer)
        cache.build_records.append(
            dict(
                layer=index,
                kind="packed_attention" if encoded else "raw_attention_control",
                stored_prompt_bytes=layer.resident_bytes,
            )
        )
    if not cache.layers:
        raise ValueError("at least one layer required")
    certificates: set[Any] = set()
    for layer in cache.layers:
        layer.cache_first_layer = cache.layers[0]
        layer.mask_certificates = certificates
    cache.initial_bytes = backing_bytes(
        tuple(t for layer in cache.layers for t in layer.owned_tensors())
    )
    cache.observe()
    return cache


class PackedAttentionTransport:
    def __init__(self, adapter: Any, cache: PromptStorageCache) -> None:
        self.adapter, self.cache = adapter, cache
        self.stack = ExitStack()

    def __enter__(self) -> "PackedAttentionTransport":
        import transformers
        from transformers.models.llama import modeling_llama

        if transformers.__version__ != "4.55.2":
            raise NotImplementedError("pinned Transformers 4.55.2 required")
        modules = [layer.self_attn for layer in self.adapter._language_layers()]
        if (
            len(modules) != len(self.cache.layers)
            or any(
                type(m) is not modeling_llama.LlamaAttention
                or m.training
                or m.config._attn_implementation != "eager"
                or m.num_key_value_groups != 1
                or m.head_dim != 128
                for m in modules
            )
            or any(not isinstance(layer, PackedAttentionLayer) for layer in self.cache.layers)
        ):
            raise NotImplementedError(
                "eval eager full-MHA Llama128 and complete packed layers required"
            )
        original_eager = modeling_llama.eager_attention_forward
        active: ContextVar[tuple[Any, ...]] = ContextVar("packed_attention_request", default=())
        mapping = dict(zip(modules, self.cache.layers, strict=True))

        def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if module not in active.get():
                return original_eager(
                    module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
                )
            layer = mapping[module]
            if dropout != 0.0:
                raise ValueError(
                    "unmasked batch-one exact prompt/history only; masks cannot be ignored"
                )
            if attention_mask is not None:
                certificate = (
                    attention_mask.data_ptr(),
                    tuple(attention_mask.shape),
                    tuple(attention_mask.stride()),
                )
                if certificate not in layer.mask_certificates:
                    valid = (attention_mask == 0).all()
                    if attention_mask.device.type == "cuda":
                        torch._assert_async(valid, "packed attention requires a fully valid mask")
                    elif not bool(valid):
                        raise ValueError("packed attention requires a fully valid mask")
                    layer.mask_certificates.add(certificate)
            return attend(
                query.contiguous(),
                *layer.packets,
                layer.history,
                used=layer.history_length,
                scaling=scaling,
            )

        try:
            self.stack.enter_context(IsolatedStorageTransport(self.adapter, self.cache))
            self.stack.enter_context(patch.object(modeling_llama, "eager_attention_forward", eager))
            for module in modules:
                old = module.forward

                def forward(*args, _module=module, _old=old, **kwargs):
                    if kwargs.get("past_key_value") is not self.cache:
                        return _old(*args, **kwargs)
                    if _module in active.get():
                        raise RuntimeError("reentrant packed attention is not supported")
                    token = active.set((*active.get(), _module))
                    try:
                        return _old(*args, **kwargs)
                    finally:
                        active.reset(token)

                self.stack.enter_context(patch.object(module, "forward", forward))
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.stack.__exit__(*args)
