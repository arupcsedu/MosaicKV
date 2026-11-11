"""Isolated, full-token native attention with a fixed head-sized KV workspace.

Consumed-input engineering diagnostic only. No source quantization, token
tiling, changed softmax, model weights, or hidden full-layer KV materialization.
The bounded workspace is common to raw and original-source representations.
"""

from contextlib import ExitStack
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

import torch
from transformers.cache_utils import CacheLayerMixin

from mosaickv.evaluation.lossless_prompt_cache import (
    IsolatedStorageTransport,
    PromptStorageCache,
    PromptStorageLayer,
    backing_bytes,
)


class RawHeadPacket:
    """Original selected BF16 K/V with CPU-only original-position metadata."""

    def __init__(self, pair):
        if (
            len(pair) != 2
            or pair[0].shape != pair[1].shape
            or pair[0].ndim != 4
            or pair[0].shape[0] != 1
            or pair[0].shape[-1] != 128
            or any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in pair)
        ):
            raise ValueError("complete contiguous batch-one BF16 BHN128 raw pair required")
        self.pair = tuple(pair)
        self.shape = tuple(pair[0].shape)

    def tensors(self):
        return self.pair

    def fill_group(self, attention, component, start, stop, target, **kwargs):
        del attention, kwargs
        target.copy_(self.pair[component][:, start:stop])


class BoundedNativeLayer(PromptStorageLayer):
    def __init__(self, packet, *, observe, updates):
        CacheLayerMixin.__init__(self)
        self.packet = packet
        self._initialize_storage(
            packet.shape,
            packet.tensors()[0].device,
            0,
            observe=observe,
            fused_unpack=False,
            maximum_decode_updates=updates,
        )
        self.incoming = ()
        self.calls = 0

    def owned_tensors(self):
        return (*self.history, *self.incoming)

    def update(self, key_states, value_states, cache_kwargs=None):
        del cache_kwargs
        expected = (*self.prompt_shape[:2], 1, self.prompt_shape[-1])
        if (
            self.incoming
            or key_states.shape != expected
            or value_states.shape != expected
            or any(
                t.dtype != torch.bfloat16 or t.device != self.history[0].device
                for t in (key_states, value_states)
            )
            or self.history_length >= self.maximum_decode_updates
        ):
            raise ValueError("one isolated same-device BF16 MHA update within bound required")
        self.incoming = key_states, value_states
        self.lease_bytes = backing_bytes(self.incoming)
        needed = self.history_length + 1
        if needed > self.history_capacity:
            capacity = min(self.maximum_decode_updates, ((needed + 15) // 16) * 16)
            newer = tuple(
                torch.empty((*expected[:2], capacity, expected[-1]), dtype=t.dtype, device=t.device)
                for t in self.incoming
            )
            self.observe(newer)
            for target, source in zip(newer, self.history, strict=True):
                target[..., : self.history_length, :].copy_(source[..., : self.history_length, :])
            del target, source
            self.history = newer
            new_bytes = backing_bytes(newer)
            self.resident_bytes += new_bytes - self.history_bytes
            self.history_bytes, self.history_capacity = new_bytes, capacity
        for target, source in zip(self.history, self.incoming, strict=True):
            target[..., self.history_length : self.history_length + 1, :].copy_(source)
        self.history_length += 1
        self.observe()
        return key_states, value_states

    def materialize(self):
        raise RuntimeError("bounded reader forbids full-layer KV materialization")

    def release(self):
        self.incoming = ()
        self.lease_bytes = 0


class BoundedNativeCache(PromptStorageCache):
    def __init__(self, packets, *, updates=128, slots=2048, metadata=()):
        from mosaickv.evaluation.bounded_native_attention import NativeHeadWorkspace

        super().__init__()
        packets = tuple(packets)
        if not packets or len({p.shape for p in packets}) != 1:
            raise ValueError("uniform physical prompt geometry required")
        device = packets[0].tensors()[0].device
        self.workspace = NativeHeadWorkspace(
            head_dim=packets[0].shape[-1], slots=slots, device=device
        )
        self.shared_tensors = (
            *metadata,
            *self.workspace.owned_tensors(),
            *(t for p in packets for t in p.tensors()),
        )
        self.shared_bytes = backing_bytes(self.shared_tensors)
        self.layers.extend(
            BoundedNativeLayer(p, observe=self.observe, updates=updates) for p in packets
        )
        self.initial_bytes = self.shared_bytes
        self.observe()

    def storage_record(self):
        record = super().storage_record()
        record.update(
            boundary="constructed_decode_owned_packet_metadata_history_incoming_and_fixed_KV_workspace",
            includes="all_packet_backing,shared_RoPE,patches,raw_cached_heads,logical_position_buffer,raw_history,growth_overlap,incoming_KV,fixed_KV_workspace",
            excludes="original_verification_prefill_and_capture,weights,attention_scores_probabilities_outputs,opaque_cuBLAS_workspace,construction_temporaries,allocator_reserved",
            construction_peak_certified=False,
            whole_request_peak_certified=False,
            full_layer_KV_materializations=0,
            workspace_bytes=self.workspace.backing_bytes,
            attention_calls_per_layer=[layer.calls for layer in self.layers],
            outstanding_incoming_KV=sum(bool(layer.incoming) for layer in self.layers),
        )
        return record


class BoundedNativeTransport:
    """Only one explicitly bound cache is dispatched; other requests stay native."""

    def __init__(self, adapter, cache):
        self.adapter, self.cache, self.stack = adapter, cache, ExitStack()

    def __enter__(self):
        import transformers
        from transformers.models.llama import modeling_llama

        from mosaickv.evaluation.bounded_native_attention import attention

        if transformers.__version__ != "4.55.2":
            raise NotImplementedError("pinned Transformers 4.55.2 required")
        modules = [layer.self_attn for layer in self.adapter._language_layers()]
        if len(modules) != len(self.cache.layers) or any(
            type(m) is not modeling_llama.LlamaAttention
            or m.training
            or m.config._attn_implementation != "eager"
            or m.num_key_value_groups != 1
            or m.head_dim != 128
            for m in modules
        ):
            raise NotImplementedError("eval eager batch-one MHA Llama128 only")
        original_eager = modeling_llama.eager_attention_forward
        original_mask = self.adapter._append_decode_attention_mask
        active: ContextVar[tuple[Any, ...]] = ContextVar("bounded_native_request", default=())
        mapping = dict(zip(modules, self.cache.layers, strict=True))

        def mask(state):
            if state.past_key_values is self.cache:
                if state.attention_mask is not None or not state.model_state.get(
                    "mosaickv_cache_all_valid"
                ):
                    raise ValueError(
                        "private all-valid sentinel required; masks cannot be discarded"
                    )
                return None
            return original_mask(state)

        def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if module not in active.get():
                return original_eager(
                    module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
                )
            layer = mapping[module]
            if dropout != 0.0 or not layer.incoming:
                raise ValueError("unmasked one-token bounded update required")
            if key is not layer.incoming[0] or value is not layer.incoming[1]:
                raise RuntimeError("attention consumed a different request/update")

            def fill(start, stop, keys, values):
                prompt_k, prompt_v = (
                    keys[..., : layer.prompt_length, :],
                    values[..., : layer.prompt_length, :],
                )
                layer.packet.fill_group(module, 0, start, stop, prompt_k, rope_scratch=prompt_v)
                layer.packet.fill_group(module, 1, start, stop, prompt_v)
                for target, history in zip((keys, values), layer.history, strict=True):
                    target[..., layer.prompt_length :, :].copy_(
                        history[:, start:stop, : layer.history_length, :]
                    )

            output, probabilities, _ = attention(
                query.contiguous(),
                fill,
                self.cache.workspace,
                length=layer.get_seq_length(),
                scaling=scaling,
                attention_mask=attention_mask,
            )
            layer.calls += 1
            return output, probabilities

        try:
            self.stack.enter_context(IsolatedStorageTransport(self.adapter, self.cache))
            self.stack.enter_context(
                patch.object(self.adapter, "_append_decode_attention_mask", mask)
            )
            self.stack.enter_context(patch.object(modeling_llama, "eager_attention_forward", eager))
            for module in modules:
                old = module.forward

                def forward(*args, _module=module, _old=old, **kwargs):
                    if kwargs.get("past_key_value") is not self.cache:
                        return _old(*args, **kwargs)
                    if _module in active.get():
                        raise RuntimeError("reentrant bounded attention forbidden")
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

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


@torch.inference_mode()
def run(adapter, prefill, packets, *, tokens=128, forced=None, slots=2048):
    import time

    from mosaickv.adapters.huggingface.types import DecodeState

    if not 2 <= tokens <= 128 or (forced is not None and len(forced) != tokens):
        raise ValueError("complete 2--128 prediction diagnostic required")
    original_position = prefill.state.next_decode_position
    positions = torch.arange(
        max(512, original_position * 2), dtype=torch.long, device=adapter.device
    )
    cache = BoundedNativeCache(packets, updates=tokens, slots=slots, metadata=(positions,))
    length = packets[0].shape[-2]
    model_state = dict(prefill.state.model_state)
    # Private all-valid transport has no physical-length mask allocation.
    for field in tuple(model_state):
        if field.startswith("mosaickv_decode_"):
            model_state.pop(field)
    model_state.update(
        mosaickv_cache_all_valid=True,
        mosaickv_packed_all_valid=True,
        mosaickv_decode_position_buffer=positions,
        mosaickv_decode_position_capacity=positions.numel(),
    )
    state = DecodeState(
        cache,
        None,
        length,
        prefill.state.logical_sequence_length,
        original_position,
        prefill.state.modality_map,
        model_state,
    )
    logits = [prefill.logits.detach().reshape(-1).float().cpu()]
    token = prefill.next_token_id
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    with BoundedNativeTransport(adapter, cache):
        torch.cuda.synchronize()
        started = time.monotonic()
        begin.record()
        for step in range(tokens - 1):
            if forced is not None:
                token = torch.tensor([[forced[step]]], dtype=torch.long, device=adapter.device)
            output = adapter.decode_one_token(token, state, capture_queries=False)
            logits.append(output.logits.detach().reshape(-1).float().cpu())
            token = output.next_token_id
        end.record()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
    if state.next_decode_position != original_position + tokens - 1:
        raise RuntimeError("original logical position changed")
    storage = cache.storage_record()
    if storage["outstanding_incoming_KV"] or storage["attention_calls_per_layer"] != [
        tokens - 1
    ] * len(packets):
        raise RuntimeError("incomplete bounded attention or outstanding borrowed update")
    return torch.stack(logits), dict(
        storage=storage,
        decode_wall_seconds=elapsed,
        decode_cuda_ms=begin.elapsed_time(end),
        predictions=tokens,
        used_fullkv_fallback=False,
        logical_positions_preserved=True,
        original_KV_values_preserved=True,
        timing_excludes_prefill_and_construction=True,
    )
