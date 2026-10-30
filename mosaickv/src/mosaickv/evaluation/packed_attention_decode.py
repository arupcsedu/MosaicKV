"""Bounded consumed-input decode, with real ordinary HF-cache control."""

from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.packed_attention_cache import PackedAttentionTransport, construct


def kv_owners(cache: Any) -> tuple[Any, ...]:
    return tuple(
        t
        for layer in cache.layers
        for t in (layer.keys, layer.values)
        if isinstance(t, torch.Tensor)
    )


class AppendAudit(TorchDispatchMode):
    """Charge old/new native append storage at the actual cat boundary."""

    def __init__(self, cache: Any) -> None:
        super().__init__()
        self.cache = cache
        self.peak = backing_bytes(kv_owners(cache))
        self.appends = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        native = False
        if str(func) == "aten.cat.default" and args and len(args[0]) == 2:
            first = args[0][0]
            native = (
                first.ndim == 4
                and first.dtype == torch.bfloat16
                and any(
                    first.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                    for t in kv_owners(self.cache)
                )
            )
        output = func(*args, **(kwargs or {}))
        if native:
            self.peak = max(self.peak, backing_bytes((*kv_owners(self.cache), output)))
            self.appends += 1
        return output


def run(
    adapter: Any,
    prefill: Any,
    pairs: Any,
    *,
    variant: str,
    tokens: int = 32,
    audit: bool = False,
    forced: Any = None,
) -> tuple[Any, dict[str, Any]]:
    import time

    from mosaickv.adapters.huggingface.types import CacheLayerSnapshot, DecodeState
    from mosaickv.backends.hf_runtime import PackedRuntimeCache, _state_from_packed

    if variant not in {"ordinary", "raw_kernel", "packed_kernel"} or not 2 <= tokens <= 128:
        raise ValueError("registered variant and bounded 2--128-token schedule required")
    if forced is not None and len(forced) != tokens:
        raise ValueError("complete optional FullKV-generated prefix required")
    torch.cuda.synchronize()
    started = time.monotonic()
    transport: Any
    if variant == "ordinary":
        template = adapter.extract_past_key_values(prefill.state.past_key_values)
        layers = tuple(
            CacheLayerSnapshot(k.detach().clone(), v.detach().clone(), 2) for k, v in pairs
        )
        snapshot = replace(
            template,
            layers=layers,
            active_sequence_length=max(k.shape[-2] for k, _ in pairs),
            independent_layer_lengths=True,
        )
        packed = PackedRuntimeCache(snapshot, (), snapshot.active_sequence_length, True, (), (), ())
        state = _state_from_packed(adapter, prefill.state, packed)
        cache = state.past_key_values
        initial = backing_bytes(kv_owners(cache))
        transport = nullcontext()
    else:
        cache = construct(pairs, encoded=variant == "packed_kernel", updates=tokens)
        cache._mosaickv_independent_layer_lengths = True
        length = max(k.shape[-2] for k, _ in pairs)
        state = DecodeState(
            cache,
            torch.ones(1, length, dtype=torch.long, device=adapter.device),
            length,
            prefill.state.logical_sequence_length,
            prefill.state.next_decode_position,
            prefill.state.modality_map,
            dict(
                prefill.state.model_state,
                mosaickv_packed_all_valid=True,
                mosaickv_cache_all_valid=True,
            ),
        )
        initial = cache.initial_bytes
        transport = PackedAttentionTransport(adapter, cache)
    torch.cuda.synchronize()
    constructor = time.monotonic() - started
    values = [prefill.logits.detach().reshape(-1).float()]
    token = prefill.next_token_id
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    monitor = AppendAudit(cache) if variant == "ordinary" and audit else None
    with transport, monitor if monitor is not None else nullcontext():
        torch.cuda.synchronize()
        started = time.monotonic()
        begin.record()
        for step in range(tokens - 1):
            if forced is not None:
                token = torch.tensor([[forced[step]]], dtype=torch.long, device=adapter.device)
            output = adapter.decode_one_token(token, state, capture_queries=False)
            values.append(output.logits.detach().reshape(-1).float())
            token = output.next_token_id
        end.record()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
    logits = torch.stack(values).cpu()
    if (
        not torch.isfinite(logits).all()
        or state.next_decode_position != prefill.state.next_decode_position + tokens - 1
    ):
        raise RuntimeError("nonfinite logits or changed logical decode progression")
    if variant == "ordinary":
        final = backing_bytes(kv_owners(cache))
        storage = dict(
            initial_active_kv_storage_bytes=initial,
            final_active_kv_storage_bytes=final,
            peak_active_kv_storage_bytes=monitor.peak if monitor is not None else None,
            native_append_transactions=monitor.appends if monitor is not None else None,
            boundary="native_KV_backing_including_old_new_append_overlap",
            whole_request_peak_certified=False,
        )
        if monitor is not None and monitor.appends != 2 * len(pairs) * (tokens - 1):
            raise RuntimeError("ordinary cache append boundaries were not completely observed")
    else:
        storage = cache.storage_record()
        if storage["outstanding_layer_leases"] or storage["history_updates_per_layer"] != [
            tokens - 1
        ] * len(pairs):
            raise RuntimeError("incomplete history or leaked attention lease")
    return logits, dict(
        variant=variant,
        output_tokens=tokens,
        constructor_seconds=constructor,
        decode_wall_seconds=elapsed,
        decode_cuda_ms=begin.elapsed_time(end),
        constructor_plus_decode_seconds=constructor + elapsed,
        timing_excludes_prefill_and_selection=True,
        storage=storage,
        used_fullkv_fallback=False,
    )
