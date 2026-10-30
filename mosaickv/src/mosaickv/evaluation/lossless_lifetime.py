"""Consumed-input lifetime controls; no benchmark labels or new selection policy."""

from __future__ import annotations

from typing import Any


def standard_greedy_reference(
    adapter: Any, prefill: Any, positions: Any, *, tokens: int = 32
) -> Any:
    """Own-token decode from ordinary unchanged exact DynamicCache, not labels."""
    import torch

    from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot
    from mosaickv.backends.hf_runtime import PackedRuntimeCache, _state_from_packed

    if type(tokens) is not int or not 1 <= tokens <= 128:
        raise ValueError("bounded reference requires an integer token count in [1, 128]")
    snapshot = gather_layer_local_snapshot(
        adapter,
        prefill.state.past_key_values,
        tuple(tuple(tuple(head) for head in layer) for layer in positions),
    )
    choice = PackedRuntimeCache(snapshot, (), snapshot.active_sequence_length, True, (), (), ())
    state = _state_from_packed(adapter, prefill.state, choice)
    logits, token = [prefill.logits.detach().reshape(-1).float().cpu()], prefill.next_token_id
    for _step in range(tokens - 1):
        output = adapter.decode_one_token(token, state, capture_queries=False)
        logits.append(output.logits.detach().reshape(-1).float().cpu())
        token = output.next_token_id
    return torch.stack(logits)


def history_codec_measurement(cache: Any) -> dict[str, Any]:
    """Measure sealed-prefix bytes, NOT an installed compressed-history runtime."""
    import torch

    from mosaickv.evaluation.lossless_bf16 import pack_cpu, unpack_into

    rows = []
    for index, layer in enumerate(cache.layers):
        sealed = layer.history_length // 16 * 16
        if sealed == 0:
            raise ValueError("at least one complete 16-token history chunk required")
        sizes, source_bytes = [], 0
        for tensor in layer.history:
            prefix = tensor[..., :sealed, :].contiguous()
            if tensor.device.type == "cuda":
                from mosaickv.evaluation.lossless_bf16_kernel import encode

                packed = encode(prefix)
            else:
                packed = pack_cpu(prefix)
            destination = torch.empty_like(prefix)
            unpack_into(packed, destination)
            if not torch.equal(destination.view(torch.int16), prefix.view(torch.int16)):
                raise RuntimeError("history codec changed a BF16 bit")
            sizes.append(packed.storage_bytes)
            source_bytes += prefix.numel() * prefix.element_size()
        rows.append(
            dict(
                layer=index,
                completed_tokens=sealed,
                raw_completed_prefix_payload_bytes=source_bytes,
                encoded_completed_prefix_backing_bytes=sum(sizes),
                existing_mutable_tail_reserved_bytes=layer.history_bytes - source_bytes,
                raw_history_backing_bytes=layer.history_bytes,
                bitwise_roundtrip=True,
            )
        )
    return dict(
        layers=rows,
        not_installed_in_decode=True,
        active_cache_or_latency_improvement_measured=False,
        interpretation="physical_codec_component_sizes_only_not_live_cache_savings",
    )


def long_control(
    adapter: Any, prefill: Any, positions: Any, *, tokens: int = 128
) -> dict[str, Any]:
    import hashlib

    import torch

    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport, construct_cache
    from mosaickv.evaluation.worker_support import settle

    if type(tokens) is not int or tokens not in (64, 128):
        raise ValueError("registered long control requires 64 or 128 tokens")
    reference = standard_greedy_reference(adapter, prefill, positions, tokens=tokens)
    settle(adapter)
    result: dict[str, Any] = dict(
        tokens=tokens,
        own_greedy=True,
        reference="ordinary_exact_DynamicCache",
        reference_logits_sha256=hashlib.sha256(reference.numpy().tobytes()).hexdigest(),
        selection_changed=False,
        no_history_eviction=True,
        whole_request_peak_certified=False,
        no_benchmark_quality_or_latency_claim=True,
        modes={},
    )
    for encoded in (False, True):
        cache = construct_cache(
            adapter,
            prefill.state.past_key_values,
            positions,
            encode=encoded,
            gpu_encoder=encoded and adapter.device.type == "cuda",
            fused_unpack=encoded,
            maximum_decode_updates=tokens,
        )
        length = max(len(heads[0]) for heads in positions)
        state = DecodeState(
            cache,
            torch.ones((1, length), device=adapter.device, dtype=torch.long),
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
        values = [prefill.logits.detach().reshape(-1).float()]
        token = prefill.next_token_id
        with IsolatedStorageTransport(adapter, cache):
            for _ in range(tokens - 1):
                output = adapter.decode_one_token(token, state, capture_queries=False)
                token = output.next_token_id
                values.append(output.logits.detach().reshape(-1).float())
        logits = torch.stack(values).cpu()
        if not torch.equal(logits, reference):
            raise RuntimeError("long trajectory changed ordinary exact-cache reference logits")
        if state.next_decode_position != prefill.state.next_decode_position + tokens - 1:
            raise RuntimeError("long trajectory changed logical positions")
        storage = cache.storage_record()
        if storage["outstanding_layer_leases"] != 0 or storage["history_updates_per_layer"] != [
            tokens - 1
        ] * len(positions):
            raise RuntimeError("long trajectory has incomplete history or leaked buffers")
        mode = "encoded_fused" if encoded else "raw_separate"
        result["modes"][mode] = dict(
            storage=storage,
            bitwise_logits_equal_reference=True,
            maximum_logit_difference=0.0,
            logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
            generated_token_ids=logits.argmax(-1).tolist(),
        )
        if encoded:
            result["history_codec_measurement"] = history_codec_measurement(cache)
        del output, values, logits, state, cache
        settle(adapter)
    return result
