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


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_raw_and_encoded_transport_match_standard_exact_cache_for_32_prefixes(greedy, fused):
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
                adapter,
                prefill.state.past_key_values,
                positions,
                encode=mode == "lossless",
                fused_unpack=fused,
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
        token = prefill.next_token_id
        with context:
            for step in range(31):
                out = adapter.decode_one_token(
                    token if greedy else torch.tensor([[3 + step % 5]]),
                    state,
                    capture_queries=False,
                )
                logits.append(out.logits.clone())
                token = out.next_token_id
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


@pytest.mark.parametrize("updates", [31, 127])
def test_cached_accounting_matches_independent_owner_scan_at_every_boundary(updates):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    adapter = _llava()
    adapter.model.to(torch.bfloat16)
    prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("audit").messages))
    cache = construct_cache(
        adapter,
        prefill.state.past_key_values,
        (((0, 3),), ((0, 3),)),
        encode=True,
        maximum_decode_updates=128,
    )
    observed = [cache.initial_bytes]
    original = cache.observe

    def audit(extras=()):
        actual = backing_bytes(
            tuple(t for layer in cache.layers for t in layer.owned_tensors()) + extras
        )
        observed.append(actual)
        original(extras)
        assert cache.peak_bytes == max(observed)

    for layer in cache.layers:
        layer.observe = audit
    for _step in range(updates):
        for layer in cache.layers:
            key = torch.ones((1, 1, 1, 8), dtype=torch.bfloat16)
            layer.update(key, key)
            layer.release()
    assert cache.storage_record()["peak_active_kv_storage_bytes"] == max(observed)


def test_full_retention_greedy_helper_matches_standard_cache():
    import evicastkv_lossless_storage as runner

    torch.manual_seed(19)
    adapter = _llava()
    adapter.model.to(torch.bfloat16)
    prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("identity").messages))
    positions = (((0, 1, 2, 3),), ((0, 1, 2, 3),))
    reference = runner.standard_greedy_reference(adapter, prefill, positions)
    candidate = runner.encoded_greedy_control(adapter, prefill, positions)
    assert torch.equal(reference, candidate)


def test_explicit_long_transport_reproduces_128_token_reference_without_history_eviction():
    from mosaickv.evaluation.lossless_lifetime import long_control

    torch.manual_seed(22)
    adapter = _llava()
    adapter.model.to(torch.bfloat16)
    prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("long").messages))
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    before = source_sha(pairs)
    report = long_control(adapter, prefill, (((0, 3),), ((0, 3),)), tokens=128)
    assert source_sha(pairs) == before
    for mode in report["modes"].values():
        assert mode["bitwise_logits_equal_reference"]
        assert len(mode["generated_token_ids"]) == 128
        assert mode["storage"]["history_updates_per_layer"] == [127, 127]
    assert report["history_codec_measurement"]["not_installed_in_decode"]


@pytest.mark.parametrize("bound", [0, True, 4097, 2.5])
def test_explicit_history_bound_is_strict(bound):
    from mosaickv.evaluation.lossless_bf16 import pack_cpu
    from mosaickv.evaluation.lossless_prompt_cache import PromptStorageLayer

    body = pack_cpu(torch.ones(1, 1, 2, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="maximum_decode_updates"):
        PromptStorageLayer(body, body, observe=lambda *_: None, maximum_decode_updates=bound)


def test_non_multiple_history_limit_counts_real_backing_and_fails_without_mutation():
    from mosaickv.evaluation.lossless_bf16 import pack_cpu
    from mosaickv.evaluation.lossless_prompt_cache import PromptStorageLayer

    body = pack_cpu(torch.ones(1, 1, 2, 8, dtype=torch.bfloat16))
    layer = PromptStorageLayer(
        body, body, observe=lambda *_: None, maximum_decode_updates=33, fused_unpack=True
    )
    token = torch.ones(1, 1, 1, 8, dtype=torch.bfloat16)
    for step in range(33):
        layer.update(token, token)
        assert layer.history_length == step + 1
        assert layer.history_capacity in (16, 32, 33)
        layer.release()
    before = tuple(t.clone() for t in layer.history)
    with pytest.raises(ValueError, match="maximum decode updates"):
        layer.update(token, token)
    assert layer.history_length == 33
    assert layer.history_bytes == 33 * 8 * 2 * 2
    assert all(torch.equal(a, b) for a, b in zip(before, layer.history, strict=True))


def test_history_growth_releases_old_owners_before_full_layer_lease(monkeypatch):
    import weakref

    from mosaickv.evaluation.lossless_bf16 import pack_cpu
    from mosaickv.evaluation.lossless_prompt_cache import PromptStorageLayer

    body = pack_cpu(torch.ones(1, 1, 2, 8, dtype=torch.bfloat16))
    layer = PromptStorageLayer(body, body, observe=lambda *_: None, maximum_decode_updates=128)
    token = torch.ones(1, 1, 1, 8, dtype=torch.bfloat16)
    for _ in range(16):
        layer.update(token, token)
        layer.release()
    previous = tuple(weakref.ref(t) for t in layer.history)
    original_empty = torch.empty
    checked = []

    def allocate(shape, *args, **kwargs):
        if tuple(shape) == (1, 1, 19, 8):
            assert all(ref() is None for ref in previous)
            checked.append(True)
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    layer.update(token, token)
    assert len(checked) == 2
    layer.release()
