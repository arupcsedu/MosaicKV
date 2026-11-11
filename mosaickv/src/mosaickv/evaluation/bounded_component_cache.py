"""Native full-width source projection with one shared component workspace.

Not a full K/V pair lease: K is consumed into scores before V overwrites the
same backing. Ragged raw prompts store no padding; temporary component padding
and the fixed pool are explicitly included in measured decode-owned storage.
"""

import weakref
from contextlib import ExitStack
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

import torch

from mosaickv.evaluation.bounded_native_cache import BoundedNativeLayer
from mosaickv.evaluation.lossless_prompt_cache import (
    IsolatedStorageTransport,
    PromptStorageCache,
    backing_bytes,
)


class ComponentLayer(BoundedNativeLayer):
    def __init__(self, packet, *, owner, observe, updates):
        super().__init__(packet, observe=observe, updates=updates)
        self.owner = weakref.ref(owner)

    def update(self, key_states, value_states, cache_kwargs=None):
        owner = self.owner()
        if owner is None or not owner._mask_dispatch_active:
            raise RuntimeError("component KV updates require the isolated active transport")
        return super().update(key_states, value_states, cache_kwargs)


class ComponentCache(PromptStorageCache):
    _mosaickv_internal_padding_masks = True

    def __init__(self, packets, *, updates=128, metadata=(), workspace=None, allocator_audit=False):
        from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace

        super().__init__()
        self.allocator_ledger = None
        self._mask_dispatch_active = False
        packets = tuple(packets)
        if not packets or any(p.shape[:2] != (1, 32) or p.shape[-1] != 128 for p in packets):
            raise ValueError("complete batch-one MHA32/128 source or ragged raw packets required")
        self.workspace = (
            ComponentWorkspace(device=packets[0].tensors()[0].device)
            if workspace is None
            else workspace
        )
        if (
            not isinstance(self.workspace, ComponentWorkspace)
            or self.workspace.heads != 32
            or self.workspace.head_dim != 128
            or self.workspace.slots != 1024
            or self.workspace.component.shape != (1024, 4096)
            or self.workspace.head.shape != (1024, 128)
            or any(
                t.dtype != torch.bfloat16
                or not t.is_contiguous()
                or t.device != packets[0].tensors()[0].device
                for t in self.workspace.owned_tensors()
            )
            or self.workspace.component.untyped_storage().data_ptr()
            == self.workspace.head.untyped_storage().data_ptr()
        ):
            raise ValueError("same-device registered reusable component workspace required")
        self.shared_tensors = (
            *metadata,
            *self.workspace.owned_tensors(),
            *(t for p in packets for t in p.tensors()),
        )
        self.shared_bytes = backing_bytes(self.shared_tensors)
        self.layers.extend(
            ComponentLayer(p, owner=self, observe=self.observe, updates=updates) for p in packets
        )
        self.initial_bytes = self.shared_bytes
        if type(allocator_audit) is not bool:
            raise ValueError("explicit allocator audit boolean required")
        if allocator_audit:
            from mosaickv.evaluation.kv_allocator_ledger import AllocatorLedger

            self.allocator_ledger = AllocatorLedger(device=packets[0].tensors()[0].device)
        self.observe()

    def observe(self, extras=()):
        super().observe(extras)
        if self.allocator_ledger is not None:
            owners = (
                *self.shared_tensors,
                *(t for layer in self.layers for t in layer.owned_tensors()),
                *extras,
            )
            active = [(i, layer) for i, layer in enumerate(self.layers) if layer.incoming]
            if len(active) > 1:
                raise RuntimeError("concurrent cache updates are unsupported by this ledger")
            if active:
                index, layer = active[0]
                update = layer.history_length + bool(extras)
                action = "growth_overlap" if extras else "copied_history"
                phase = f"decode_update_{update}_layer_{index}_{action}"
            else:
                phase = (
                    "final_cache" if any(x.history_length for x in self.layers) else "initial_cache"
                )
            self.allocator_ledger.observe(owners, phase=phase)

    def storage_record(self):
        record = super().storage_record()
        record.update(
            boundary="constructed_decode_owned_source_or_ragged_raw_metadata_history_and_component_pool",
            includes="packet_backing,phases,patches,logical_position_buffer,raw_history,growth_overlap,incoming_KV,one_component_pool,one_head_scratch",
            excludes="verification_prefill_and_captures,weights,attention_scores_probabilities_outputs,opaque_vendor_workspace,construction_temporaries,allocator_reserved",
            full_KV_pair_leases=0,
            workspace_bytes=self.workspace.backing_bytes,
            full_width_native_projection=True,
            native_dispatch="B32_zero_stride",
            stored_prompt_padding_bytes=0,
            attention_calls_per_layer=[layer.calls for layer in self.layers],
            outstanding_incoming_KV=sum(bool(layer.incoming) for layer in self.layers),
            construction_peak_certified=False,
            whole_request_peak_certified=False,
            allocator_blocks=(
                None if self.allocator_ledger is None else self.allocator_ledger.summary()
            ),
        )
        return record


class ComponentTransport:
    def __init__(self, adapter, cache):
        self.adapter, self.cache, self.stack = adapter, cache, ExitStack()

    def __enter__(self):
        import transformers
        from transformers.models.llama import modeling_llama

        from mosaickv.evaluation.bounded_component_attention import attention

        modules = [layer.self_attn for layer in self.adapter._language_layers()]
        if (
            transformers.__version__ != "4.55.2"
            or len(modules) != len(self.cache.layers)
            or any(
                type(m) is not modeling_llama.LlamaAttention
                or m.training
                or m.config._attn_implementation != "eager"
                or m.num_key_value_groups != 1
                or m.head_dim != 128
                for m in modules
            )
        ):
            raise NotImplementedError("pinned eager Llama full-MHA32/128 required")
        original_eager = modeling_llama.eager_attention_forward
        original_mask = self.adapter._append_decode_attention_mask
        active: ContextVar[tuple[Any, ...]] = ContextVar("component_reader_request", default=())
        mapping = dict(zip(modules, self.cache.layers, strict=True))

        def mask(state):
            if state.past_key_values is self.cache:
                if state.attention_mask is not None or not state.model_state.get(
                    "mosaickv_internal_padding_masks"
                ):
                    raise ValueError("explicit private all-valid transport sentinel required")
                return None
            return original_mask(state)

        def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if module not in active.get():
                return original_eager(
                    module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
                )
            layer = mapping[module]
            if dropout != 0.0 or not layer.incoming:
                raise ValueError("one isolated zero-dropout decode update required")
            if key is not layer.incoming[0] or value is not layer.incoming[1]:
                raise RuntimeError("cross-request KV dispatch forbidden")

            def fill(component, target, scratch):
                layer.packet.fill_component(
                    module,
                    component,
                    target[..., : layer.prompt_length, :],
                    rope_scratch=scratch[0, 0],
                )
                target[..., layer.prompt_length :, :].copy_(
                    layer.history[component][..., : layer.history_length, :]
                )

            output, probabilities, _ = attention(
                query.contiguous(),
                fill,
                self.cache.workspace,
                length=layer.get_seq_length(),
                scaling=scaling,
                attention_mask=attention_mask,
                prompt_lengths=getattr(
                    layer.packet, "prompt_lengths", (layer.prompt_length,) * layer.prompt_shape[1]
                ),
                prompt_max_length=layer.prompt_length,
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
                        raise RuntimeError("reentrant component reader forbidden")
                    token = active.set((*active.get(), _module))
                    try:
                        return _old(*args, **kwargs)
                    finally:
                        active.reset(token)

                self.stack.enter_context(patch.object(module, "forward", forward))
        except BaseException:
            self.stack.close()
            raise
        self.cache._mask_dispatch_active = True
        return self

    def __exit__(self, *args):
        try:
            return self.stack.__exit__(*args)
        finally:
            self.cache._mask_dispatch_active = False


@torch.inference_mode()
def run(
    adapter,
    prefill,
    packets,
    *,
    tokens=128,
    forced=None,
    workspace=None,
    positions=None,
    allocator_audit=False,
):
    import time

    from mosaickv.adapters.huggingface.types import DecodeState

    if not 2 <= tokens <= 128 or forced is None or len(forced) != tokens:
        raise ValueError("bounded complete FullKV-generated evaluation prefix required")
    original_position = prefill.state.next_decode_position
    if positions is None:
        positions = torch.arange(
            max(512, original_position * 2), dtype=torch.long, device=adapter.device
        )
    if (
        positions.dtype != torch.long
        or positions.device != adapter.device
        or positions.ndim != 1
        or positions.numel() < original_position + tokens
    ):
        raise ValueError("same-device complete original logical-position buffer required")
    cache = ComponentCache(
        packets,
        updates=tokens,
        metadata=(positions,),
        workspace=workspace,
        allocator_audit=allocator_audit,
    )
    model_state = {
        k: v for k, v in prefill.state.model_state.items() if not k.startswith("mosaickv_decode_")
    }
    all_valid = all(
        all(n == p.shape[-2] for n in getattr(p, "prompt_lengths", (p.shape[-2],))) for p in packets
    )
    model_state.update(
        mosaickv_cache_all_valid=all_valid,
        mosaickv_packed_all_valid=False,
        mosaickv_internal_padding_masks=True,
        mosaickv_decode_position_buffer=positions,
        mosaickv_decode_position_capacity=positions.numel(),
    )
    state = DecodeState(
        cache,
        None,
        max(p.shape[-2] for p in packets),
        prefill.state.logical_sequence_length,
        original_position,
        prefill.state.modality_map,
        model_state,
    )
    values = [prefill.logits.detach().reshape(-1).float().cpu()]
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    with ComponentTransport(adapter, cache):
        torch.cuda.synchronize()
        started = time.monotonic()
        begin.record()
        for step in range(tokens - 1):
            token = torch.tensor([[forced[step]]], dtype=torch.long, device=adapter.device)
            output = adapter.decode_one_token(token, state, capture_queries=False)
            values.append(output.logits.detach().reshape(-1).float().cpu())
        end.record()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
    storage = cache.storage_record()
    if (
        state.next_decode_position != original_position + tokens - 1
        or storage["outstanding_incoming_KV"]
        or storage["attention_calls_per_layer"] != [tokens - 1] * len(packets)
    ):
        raise RuntimeError("logical position, incoming owner or incomplete attention failure")
    return torch.stack(values), dict(
        storage=storage,
        decode_wall_seconds=elapsed,
        decode_cuda_ms=begin.elapsed_time(end),
        predictions=tokens,
        used_fullkv_fallback=False,
        original_KV_values_preserved=True,
        logical_positions_preserved=True,
        timing_excludes_prefill_and_construction=True,
    )
