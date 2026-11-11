"""CPU transport controls; these are not pretrained/CUDA parity evidence."""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama

from mosaickv.evaluation.bounded_native_cache import (
    BoundedNativeCache,
    BoundedNativeTransport,
)
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class RawFixturePacket:
    """Original selected BF16 KV with the packet's explicit head-fill API."""

    def __init__(self, key, value, *, fail=False):
        self.pair = key, value
        self.shape = tuple(key.shape)
        self.fail = fail

    def tensors(self):
        return self.pair

    def fill_group(self, attention, component, start, stop, target, *, rope_scratch=None):
        del attention, rope_scratch
        if self.fail:
            raise RuntimeError("injected bounded packet fill failure")
        target.copy_(self.pair[component][:, start:stop])


def tiny_model():
    torch.manual_seed(771)
    return (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=31,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=128,
                attn_implementation="eager",
            )
        )
        .bfloat16()
        .eval()
    )


def mock_adapter(model):
    return SimpleNamespace(
        _language_layers=lambda: model.model.layers,
        _cache_length=lambda cache: cache.get_seq_length(),
        _append_decode_attention_mask=lambda state: state.attention_mask,
    )


def test_history_accounting_no_full_leases_and_borrowed_release():
    key = torch.ones(1, 2, 4, 128, dtype=torch.bfloat16)
    value = key.clone()
    packet = RawFixturePacket(key, value)
    positions = torch.arange(512)
    cache = BoundedNativeCache((packet,), updates=32, metadata=(positions,))
    initial = cache.storage_record()
    assert initial["initial_active_kv_storage_bytes"] == backing_bytes(
        (*packet.tensors(), positions, *cache.workspace.owned_tensors())
    )
    assert initial["workspace_bytes"] == 1048576
    layer = cache.layers[0]
    source_snapshot = key.clone(), value.clone()
    for index in range(20):
        k = torch.full((1, 2, 1, 128), index, dtype=torch.bfloat16)
        v = torch.full_like(k, -index)
        incoming = layer.update(k, v)
        assert incoming[0] is k and incoming[1] is v
        assert layer.leased == ()
        assert layer.lease_bytes == backing_bytes((k, v))
        with pytest.raises(ValueError, match="isolated"):
            layer.update(k, v)
        layer.release()
        assert layer.incoming == () and layer.lease_bytes == 0
    record = cache.storage_record()
    assert record["history_updates_per_layer"] == [20]
    assert record["outstanding_incoming_KV"] == record["outstanding_layer_leases"] == 0
    assert record["full_layer_KV_materializations"] == 0
    assert layer.history_capacity == 32
    assert torch.equal(layer.history[0][0, 0, :20, 0], torch.arange(20).bfloat16())
    assert torch.equal(layer.history[1][0, 0, :20, 0], -torch.arange(20).bfloat16())
    assert record["final_active_kv_storage_bytes"] == backing_bytes(
        (*cache.shared_tensors, *layer.owned_tensors())
    )
    assert record["peak_active_kv_storage_bytes"] >= record["final_active_kv_storage_bytes"]
    assert all(
        torch.equal(before, after)
        for before, after in zip(source_snapshot, packet.pair, strict=True)
    )
    with pytest.raises(RuntimeError, match="forbids full-layer"):
        layer.materialize()


def test_tiny_native_same_selected_tokens_and_request_isolation():
    model = tiny_model()
    adapter = mock_adapter(model)
    eager_before = modeling_llama.eager_attention_forward
    length_before = adapter._cache_length
    mask_before = adapter._append_decode_attention_mask
    with torch.inference_mode():
        prefill = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        packets = tuple(
            RawFixturePacket(layer.keys.clone(), layer.values.clone())
            for layer in prefill.past_key_values.layers
        )
        snapshots = tuple(t.clone() for packet in packets for t in packet.tensors())
        cache = BoundedNativeCache(packets, updates=16)
        with BoundedNativeTransport(adapter, cache):
            for position, token in enumerate((5, 6, 7), 4):
                expected = model(
                    torch.tensor([[token]]),
                    past_key_values=prefill.past_key_values,
                    cache_position=torch.tensor([position]),
                    use_cache=True,
                ).logits
                actual = model(
                    torch.tensor([[token]]),
                    past_key_values=cache,
                    cache_position=torch.tensor([position]),
                    use_cache=True,
                ).logits
                assert torch.equal(actual, expected)
                assert all(not layer.incoming and not layer.leased for layer in cache.layers)
            ordinary = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True).logits
            assert torch.equal(ordinary, prefill.logits)
            sentinel = SimpleNamespace(
                past_key_values=cache,
                attention_mask=None,
                model_state={"mosaickv_cache_all_valid": True},
            )
            assert adapter._append_decode_attention_mask(sentinel) is None
            sentinel.attention_mask = torch.zeros(1, 1)
            with pytest.raises(ValueError, match="sentinel"):
                adapter._append_decode_attention_mask(sentinel)
    assert modeling_llama.eager_attention_forward is eager_before
    assert adapter._cache_length is length_before
    assert adapter._append_decode_attention_mask is mask_before
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            snapshots, (t for packet in packets for t in packet.tensors()), strict=True
        )
    )
    record = cache.storage_record()
    assert record["attention_calls_per_layer"] == [3, 3]
    assert record["history_updates_per_layer"] == [3, 3]
    assert record["outstanding_incoming_KV"] == 0


def test_private_dispatch_and_borrowed_updates_restore_after_exception():
    model = tiny_model()
    adapter = mock_adapter(model)
    eager_before = modeling_llama.eager_attention_forward
    mask_before = adapter._append_decode_attention_mask
    length_before = adapter._cache_length
    forwards_before = [layer.self_attn.forward for layer in model.model.layers]
    hook_counts_before = [len(layer.self_attn._forward_hooks) for layer in model.model.layers]
    with torch.inference_mode():
        prefill = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        packets = tuple(
            RawFixturePacket(layer.keys.clone(), layer.values.clone(), fail=True)
            for layer in prefill.past_key_values.layers
        )
        cache = BoundedNativeCache(packets, updates=16)
        with (
            pytest.raises(RuntimeError, match="injected bounded"),
            BoundedNativeTransport(adapter, cache),
        ):
            model(
                torch.tensor([[5]]),
                past_key_values=cache,
                cache_position=torch.tensor([4]),
                use_cache=True,
            )
    assert modeling_llama.eager_attention_forward is eager_before
    assert adapter._append_decode_attention_mask is mask_before
    assert adapter._cache_length is length_before
    assert all(not layer.incoming and not layer.leased for layer in cache.layers)
    assert [layer.self_attn.forward for layer in model.model.layers] == forwards_before
    assert [
        len(layer.self_attn._forward_hooks) for layer in model.model.layers
    ] == hook_counts_before
    assert cache.storage_record()["outstanding_incoming_KV"] == 0
