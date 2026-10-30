"""Paired consumed-input timing diagnostic, not end-to-end or task-quality evidence."""

from __future__ import annotations

import random
import time
from typing import Any

VARIANTS = ((False, False), (False, True), (True, False), (True, True))


def randomized_orders(*, seed: int, warmups: int = 5, repetitions: int = 20) -> list[Any]:
    if warmups < 5 or repetitions < 20:
        raise ValueError("at least five warmups and twenty timed repetitions required")
    generator = random.Random(seed)
    orders = []
    for block in range(warmups + repetitions):
        values = list(VARIANTS)
        generator.shuffle(values)
        orders.append((block - warmups, values))
    return orders


def benchmark(
    adapter: Any, prefill: Any, positions: Any, reference: Any, *, seed: int
) -> list[Any]:
    """Time identical own-greedy updates; transfer logits only after synchronization."""
    import gc

    import torch

    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport, construct_cache

    if reference.shape[0] != 32:
        raise ValueError("the unchanged 32-token exact reference is required")
    rows = []
    for block, order in randomized_orders(seed=seed):
        for order_index, (encoded, fused) in enumerate(order):
            torch.cuda.synchronize()
            before = time.monotonic()
            cache = construct_cache(
                adapter,
                prefill.state.past_key_values,
                positions,
                encode=encoded,
                gpu_encoder=encoded,
                fused_unpack=fused,
            )
            torch.cuda.synchronize()
            construction = time.monotonic() - before
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
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            before = time.monotonic()
            with IsolatedStorageTransport(adapter, cache):
                start.record()
                for _ in range(31):
                    output = adapter.decode_one_token(token, state, capture_queries=False)
                    values.append(output.logits.detach().reshape(-1).float())
                    token = output.next_token_id
                end.record()
            torch.cuda.synchronize()
            wall = time.monotonic() - before
            cuda_ms = start.elapsed_time(end)
            logits = torch.stack(values).cpu()
            if not torch.equal(logits, reference):
                raise RuntimeError("timed decode changed exact reference logits/tokens")
            if state.next_decode_position != prefill.state.next_decode_position + 31:
                raise RuntimeError("timed decode changed logical positions")
            storage = cache.storage_record()
            if storage["outstanding_layer_leases"] != 0:
                raise RuntimeError("timed decode leaked materialized layer storage")
            rows.append(
                dict(
                    block=block,
                    warmup=block < 0,
                    order=order_index,
                    encoded=encoded,
                    fused=fused,
                    seed=seed,
                    construction_seconds=construction,
                    decode_wall_seconds=wall,
                    decode_cuda_ms=cuda_ms,
                    bitwise_reference_parity=True,
                    storage={
                        key: storage[key]
                        for key in (
                            "initial_active_kv_storage_bytes",
                            "final_active_kv_storage_bytes",
                            "peak_active_kv_storage_bytes",
                        )
                    },
                    boundary="shared_prefill_post_selection_constructor_and_31_decode_updates",
                    excludes="prefill,selection,trace_IO,reference_construction,logit_host_transfer",
                    not_request_latency=True,
                )
            )
            del output, state, cache, values, logits
            # Layer observers intentionally refer to their owning cache. Collect
            # that cycle outside measurement rather than retaining old trials.
            gc.collect()
    return rows
