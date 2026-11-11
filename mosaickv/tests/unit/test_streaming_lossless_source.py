"""Actual tiny HF Llama CPU checks, not pretrained quality evidence."""

import gc
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from mosaickv.evaluation.lossless_source_control import FeasibilityDeclined
from mosaickv.evaluation.streaming_lossless_source import StreamingLosslessSource
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def tiny():
    torch.manual_seed(192)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=512,
        intermediate_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).bfloat16().eval()
    adapter = SimpleNamespace(_language_layers=lambda: model.model.layers)
    prepared = SimpleNamespace(logical_sequence_length=8)
    return model, adapter, prepared, torch.arange(8)[None]


def bits(left, right):
    return torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


def streamed(tiny, *, tokens=None, audit=True, caps=None):
    model, adapter, prepared, default = tiny
    cache = RetiringPrefillCache(2, audit=audit)
    encoder = StreamingLosslessSource(adapter, prepared, cache, layer_byte_caps=caps)
    with torch.inference_mode(), SealAfterAttention(adapter, cache, encoder):
        result = model(default if tokens is None else tokens, past_key_values=cache, use_cache=True)
    report = encoder.finish()
    return result, cache, report


def test_actual_hf_fullprefill_logits_and_every_layer_kv_are_bitwise_identical(tiny):
    model, _, _, tokens = tiny
    with torch.inference_mode():
        reference = model(tokens, past_key_values=DynamicCache(), use_cache=True)
    actual, cache, report = streamed(tiny)
    assert actual.past_key_values is cache
    assert bits(reference.logits, actual.logits)
    assert bits(reference.logits[:, -1], actual.logits[:, -1])
    for index, packet in enumerate(cache.formats()):
        restored = packet.materialize(model.model.layers[index].self_attn)
        ordinary = reference.past_key_values.layers[index]
        assert bits(ordinary.keys, restored[0])
        assert bits(ordinary.values, restored[1])
        assert torch.equal(packet.logical_positions, torch.arange(8))
        assert cache.get_seq_length(index) == reference.past_key_values.get_seq_length(index) == 8
        assert cache.layers[index].keys is cache.layers[index].values is None
    gc.collect()
    assert all(ref() is None for ref in cache.original_refs)
    assert report["prefill_storage"]["original_KV_tensor_objects_alive"] == 0
    assert report["persistent_bytes"] < report["original_raw_KV_bytes"]
    assert report["prefill_storage"]["logical_lengths"] == [8, 8]
    assert report["original_KV_all_word_bitwise_validation"]
    assert not report["global_source_or_query_capture"]
    assert not report["whole_request_gpu_peak_certified"]


def test_source_accounting_charges_all_headers_patches_and_constructor_overlap(tiny):
    _, cache, report = streamed(tiny)
    storage = report["prefill_storage"]
    assert report["persistent_bytes"] == storage["final_prefill_cache_bytes"]
    assert storage["observed_prefill_cache_peak_bytes"] >= report["persistent_bytes"]
    assert storage["tensor_results_observed"]
    assert storage["observations"] > 2
    for row, packet in zip(report["layer_records"], cache.formats(), strict=True):
        assert row["persistent_bytes"] == (
            row["source_bytes"] + row["phase_and_map_bytes"] + row["patch_bytes"]
        )
        assert row["persistent_bytes"] == packet.component_bytes
        assert row["persistent_bytes"] < row["raw_kv_bytes"]
        assert row["patch_scalars"] == [0, 0]


def test_completed_layer_is_retired_before_next_attention_starts(tiny):
    model, adapter, prepared, tokens = tiny
    cache = RetiringPrefillCache(2, audit=True)
    encoder = StreamingLosslessSource(adapter, prepared, cache)
    observations = []

    def before_second(_module, _args):
        gc.collect()
        observations.append(
            (
                cache.next_layer,
                cache.layers[0].keys is None,
                cache.layers[0].values is None,
                cache.layers[0].packet is not None,
                all(ref() is None for ref in cache.original_refs),
            )
        )

    handle = model.model.layers[1].register_forward_pre_hook(before_second)
    try:
        with torch.inference_mode(), SealAfterAttention(adapter, cache, encoder):
            model(tokens, past_key_values=cache, use_cache=True)
    finally:
        handle.remove()
    encoder.finish()
    assert observations == [(1, True, True, True, True)]


def test_failed_cap_keeps_private_original_cache_and_rolls_back_hooks(tiny):
    model, adapter, prepared, tokens = tiny
    cache = RetiringPrefillCache(2, audit=True)
    encoder = StreamingLosslessSource(adapter, prepared, cache, layer_byte_caps=(1, 1))
    with (
        pytest.raises(FeasibilityDeclined),
        torch.inference_mode(),
        SealAfterAttention(adapter, cache, encoder),
    ):
        model(tokens, past_key_values=cache, use_cache=True)
    assert encoder.failed and encoder.failure == {"layer": 0, "error_type": "FeasibilityDeclined"}
    assert cache.next_layer == 0 and cache.layers[0].keys is not None
    assert cache.layers[0].packet is None and cache.layers[1].keys is None
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
    with pytest.raises(RuntimeError):
        encoder.finish()
    with pytest.raises(RuntimeError):
        cache.formats()
    # Failure is request-private, not an attention mutation or shared capture.
    with torch.inference_mode():
        ordinary = model(tokens, use_cache=True).logits
    actual, _, _ = streamed(tiny)
    assert bits(actual.logits, ordinary)


def test_later_layer_failure_never_exports_completed_partial_source_cache(tiny):
    model, adapter, prepared, tokens = tiny
    cache = RetiringPrefillCache(2, audit=True)
    encoder = StreamingLosslessSource(adapter, prepared, cache, layer_byte_caps=(2**20, 1))
    with (
        pytest.raises(FeasibilityDeclined),
        torch.inference_mode(),
        SealAfterAttention(adapter, cache, encoder),
    ):
        model(tokens, past_key_values=cache, use_cache=True)
    assert cache.next_layer == 1
    assert cache.layers[0].keys is None and cache.layers[0].packet is not None
    assert cache.layers[1].keys is not None and cache.layers[1].packet is None
    assert encoder.failure == {"layer": 1, "error_type": "FeasibilityDeclined"}
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
    with pytest.raises(RuntimeError):
        encoder.finish()
    with pytest.raises(RuntimeError):
        cache.formats()
    record = cache.record(require_complete=False)
    assert record["original_native_layers_remaining"] == 1
    assert record["logical_lengths"] == [8, 8]


def test_request_a_b_a_uses_private_source_owners_and_preserves_outputs(tiny):
    _, adapter, _, tokens = tiny
    first, first_cache, _ = streamed(tiny)
    middle, middle_cache, _ = streamed(tiny, tokens=tokens.flip(-1))
    replay, replay_cache, _ = streamed(tiny)
    assert bits(first.logits, replay.logits)
    assert not bits(first.logits, middle.logits)
    owners = [
        {
            t.untyped_storage()._cdata
            for packet in c.formats()
            for t in packet.tensors()
            if t.numel()
        }
        for c in (first_cache, middle_cache, replay_cache)
    ]
    assert not owners[0].intersection(owners[1] | owners[2])
    assert not owners[1].intersection(owners[2])
    assert all(not layer.self_attn._forward_hooks for layer in adapter._language_layers())


@pytest.mark.parametrize("caps", [1, (), (1,), (True, 100), (0, 100)])
def test_invalid_per_layer_caps_fail_before_model_execution(tiny, caps):
    _, adapter, prepared, _ = tiny
    with pytest.raises(ValueError, match="predetermined"):
        StreamingLosslessSource(adapter, prepared, RetiringPrefillCache(2), layer_byte_caps=caps)


def test_consumed_cache_and_incomplete_export_are_rejected(tiny):
    _, adapter, prepared, _ = tiny
    _, cache, _ = streamed(tiny)
    with pytest.raises(ValueError, match="fresh"):
        StreamingLosslessSource(adapter, prepared, cache)
    fresh = StreamingLosslessSource(adapter, prepared, RetiringPrefillCache(2))
    with pytest.raises(RuntimeError, match="complete"):
        fresh.finish()
