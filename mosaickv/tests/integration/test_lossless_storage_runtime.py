"""CPU random-weight BF16 model transport; no downloaded model or task scores."""

import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

import mosaickv.backends.hf_runtime as runtime
from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot
from mosaickv.adapters.huggingface.types import DecodeState
from mosaickv.evaluation.delimiter_control import source_sha
from mosaickv.evaluation.lossless_prompt_cache import (
    IsolatedStorageTransport,
    construct_cache,
)


def test_raw_and_encoded_transport_match_standard_exact_cache_for_32_prefixes():
    torch.manual_seed(9)
    adapter = _llava()
    adapter.model.to(torch.bfloat16)
    saved = {}
    for mode in ("standard", "raw_control", "lossless"):
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request(mode).messages))
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        before = source_sha(pairs)
        positions = tuple(((0, 3),) for _ in pairs)
        if mode == "standard":
            snapshot = gather_layer_local_snapshot(
                adapter, prefill.state.past_key_values, positions
            )
            choice = runtime.PackedRuntimeCache(snapshot, (), 2, True, (), (), ())
            state = runtime._state_from_packed(adapter, prefill.state, choice)
            from contextlib import nullcontext

            context = nullcontext()
        else:
            cache = construct_cache(
                adapter, prefill.state.past_key_values, positions, encode=mode == "lossless"
            )
            state = DecodeState(
                cache,
                torch.ones((1, 2), dtype=torch.long),
                2,
                4,
                4,
                prefill.state.modality_map,
                {"mosaickv_packed_all_valid": True, "mosaickv_cache_all_valid": True},
            )
            context = IsolatedStorageTransport(adapter, cache)
        original_length = adapter._cache_length
        logits = []
        with context:
            for step in range(31):
                out = adapter.decode_one_token(
                    torch.tensor([[3 + step % 5]]), state, capture_queries=False
                )
                logits.append(out.logits.clone())
            assert state.logical_sequence_length == state.next_decode_position == 35
            if mode != "standard":
                record = cache.storage_record()
                assert record["outstanding_layer_leases"] == 0
                assert record["history_updates_per_layer"] == [31, 31]
                assert (
                    record["peak_active_kv_storage_bytes"]
                    >= record["final_active_kv_storage_bytes"]
                )
        assert adapter._cache_length == original_length
        assert source_sha(pairs) == before
        saved[mode] = torch.stack(logits)
    assert torch.equal(saved["standard"], saved["raw_control"])
    assert torch.equal(saved["standard"], saved["lossless"])


def test_integer_codec_compresses_large_synthetic_source_without_source_mutation():
    from transformers import DynamicCache

    torch.manual_seed(10)
    adapter = _llava(kv_heads=2)
    cache = DynamicCache()
    for i in range(2):
        key, value = (torch.randn(1, 2, 512, 8).bfloat16() for _ in range(2))
        cache.update(key, value, i)
    original, _ = adapter._legacy_layers(cache)
    before = source_sha(original)
    positions = tuple(tuple(tuple(range(256)) for _ in range(2)) for _ in range(2))
    raw = construct_cache(adapter, cache, positions, encode=False)
    packed = construct_cache(adapter, cache, positions, encode=True)
    assert packed.initial_bytes < raw.initial_bytes
    assert source_sha(original) == before
    assert all(r["cpu_roundtrip_bitwise"] for r in packed.build_records)


def test_transport_releases_temporary_buffers_and_restores_dispatch_on_failure():
    adapter = _llava()
    adapter.model.to(torch.bfloat16)
    prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("failure").messages))
    positions = (((0, 3),), ((0, 3),))
    cache = construct_cache(adapter, prefill.state.past_key_values, positions, encode=True)
    original = adapter._cache_length
    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        IsolatedStorageTransport(adapter, cache),
    ):
        cache.layers[0].leased = (torch.empty((4,)),)
        raise RuntimeError("synthetic failure")
    assert adapter._cache_length == original
    assert not any(layer.leased for layer in cache.layers)
