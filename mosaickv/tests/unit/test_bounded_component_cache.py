"""CPU ownership, ragged masking and isolated one-component HF transport."""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama

from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter
from mosaickv.evaluation.bounded_component_cache import ComponentCache, ComponentTransport
from mosaickv.evaluation.bounded_ragged_raw import RaggedRawPacket
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@pytest.fixture(autouse=True)
def bounded_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class SourceFixture:
    """Real BF16 source/phase owners, no stored full K/V pair."""

    def __init__(self, length, *, phases=None):
        self.shape = (1, 32, length, 128)
        self.source = torch.zeros((1, length, 4096), dtype=torch.bfloat16)
        self.phases = phases or tuple(
            torch.ones((1, length, 128), dtype=torch.bfloat16) for _ in range(2)
        )

    def tensors(self):
        return self.source, *self.phases


def ragged_packet(length=3, *, fail=False):
    heads, positions = [], []
    for head in range(32):
        count = length - (head % 2 and length > 1)
        heads.append(tuple(torch.zeros(count, 128, dtype=torch.bfloat16) for _ in range(2)))
        positions.append(torch.arange(count, dtype=torch.long))
    packet = RaggedRawPacket(heads, positions, original_length=length, prompt_capacity=length)
    if fail:

        def reject(*args, **kwargs):
            raise RuntimeError("injected component fill failure")

        packet.fill_component = reject
    return packet


def test_complete_storage_counts_shared_pool_metadata_histories_and_growth_overlap():
    packets = (ragged_packet(3), SourceFixture(5))
    metadata = torch.arange(512, dtype=torch.long)
    cache = ComponentCache(packets, updates=32, metadata=(metadata,))
    initial = cache.storage_record()
    expected_initial = backing_bytes(
        (metadata, *cache.workspace.owned_tensors(), *(t for p in packets for t in p.tensors()))
    )
    assert initial["initial_active_kv_storage_bytes"] == expected_initial
    assert initial["workspace_bytes"] == 8 * 2**20 + 256 * 2**10
    assert initial["stored_prompt_padding_bytes"] == 0
    assert [layer.prompt_length for layer in cache.layers] == [3, 5]
    # This unit isolates accounting from model dispatch; model tests below
    # exercise the real context-managed activation and restoration.
    cache._mask_dispatch_active = True
    for step in range(20):
        for layer in cache.layers:
            key = torch.full((1, 32, 1, 128), step, dtype=torch.bfloat16)
            value = torch.full_like(key, -step)
            result = layer.update(key, value)
            assert result[0] is key and result[1] is value
            assert layer.leased == ()
            assert layer.lease_bytes == backing_bytes((key, value))
            layer.release()
    record = cache.storage_record()
    expected_final = backing_bytes(
        (*cache.shared_tensors, *(t for layer in cache.layers for t in layer.owned_tensors()))
    )
    assert record["final_active_kv_storage_bytes"] == expected_final
    assert expected_final == expected_initial + 2 * (32 * 32 * 128 * 2 * 2)
    # Last layer's 16->32 growth: one 32-slot history, one old 16-slot history,
    # its new 32-slot history, and the currently borrowed one-token K/V pair.
    assert record["peak_active_kv_storage_bytes"] == expected_initial + 1327104
    assert record["history_updates_per_layer"] == [20, 20]
    assert record["outstanding_incoming_KV"] == record["outstanding_layer_leases"] == 0
    assert record["full_KV_pair_leases"] == 0
    assert record["whole_request_peak_certified"] is False
    cache._mask_dispatch_active = False
    for layer in cache.layers:
        with pytest.raises(RuntimeError, match="forbids full-layer"):
            layer.materialize()


def test_shared_source_phases_are_deduplicated_by_backing_not_tensor_view_sizes():
    first = SourceFixture(3)
    second = SourceFixture(3, phases=first.phases)
    cache = ComponentCache((first, second), metadata=first.phases)
    assert cache.shared_bytes == backing_bytes(
        (*cache.workspace.owned_tensors(), first.source, second.source, *first.phases)
    )
    record = cache.storage_record()
    assert record["initial_active_kv_storage_bytes"] == record["final_active_kv_storage_bytes"]


def test_inactive_transport_rejects_updates_before_allocating_history():
    cache = ComponentCache((ragged_packet(),), updates=16)
    layer = cache.layers[0]
    token = torch.zeros((1, 32, 1, 128), dtype=torch.bfloat16)
    before = cache.storage_record()
    with pytest.raises(RuntimeError, match="isolated active transport"):
        layer.update(token, token.clone())
    after = cache.storage_record()
    assert layer.history_length == layer.history_capacity == 0
    assert layer.incoming == ()
    assert after["final_active_kv_storage_bytes"] == before["final_active_kv_storage_bytes"]


@pytest.mark.parametrize("cache_flag", [False, None, "true", 1])
def test_internal_mask_flag_cannot_authorize_an_unowned_or_malformed_cache(cache_flag):
    state = SimpleNamespace(
        past_key_values=SimpleNamespace(_mosaickv_internal_padding_masks=cache_flag),
        active_cache_length=2,
        logical_sequence_length=3,
        model_state={"mosaickv_internal_padding_masks": True},
    )
    with pytest.raises(ValueError, match="owning cache transport"):
        HuggingFaceMultimodalAdapter.decode_one_token(object(), torch.tensor([[1]]), state)


@pytest.mark.parametrize("model_state", [{}, {"mosaickv_internal_padding_masks": "true"}])
def test_default_compressed_adapter_path_keeps_its_existing_safety_rejection(model_state):
    state = SimpleNamespace(
        past_key_values=object(),
        active_cache_length=2,
        logical_sequence_length=3,
        model_state=model_state,
    )
    with pytest.raises(NotImplementedError, match="explicit logical-position"):
        HuggingFaceMultimodalAdapter.decode_one_token(object(), torch.tensor([[1]]), state)


@pytest.fixture(scope="module")
def zero_model():
    """One 32-head layer; zero weights make this an exact dispatch control."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=4096,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=32,
                num_key_value_heads=32,
                head_dim=128,
                attn_implementation="eager",
            )
        )
        .bfloat16()
        .eval()
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    yield model
    torch.set_num_threads(previous)


def adapter_for(model):
    return SimpleNamespace(
        _language_layers=lambda: model.model.layers,
        _cache_length=lambda cache: cache.get_seq_length(),
        _append_decode_attention_mask=lambda state: state.attention_mask,
    )


def test_internal_padding_scope_native_dispatch_and_other_requests_remain_unchanged(zero_model):
    model = zero_model
    adapter = adapter_for(model)
    packet = ragged_packet(3)
    before = tuple(t.clone() for t in packet.tensors())
    cache = ComponentCache((packet,), updates=16)
    eager_before = modeling_llama.eager_attention_forward
    length_before = adapter._cache_length
    mask_before = adapter._append_decode_attention_mask
    forwards_before = [layer.self_attn.forward for layer in model.model.layers]
    hooks_before = [len(layer.self_attn._forward_hooks) for layer in model.model.layers]
    sentinel = SimpleNamespace(
        past_key_values=cache,
        attention_mask=None,
        model_state={"mosaickv_internal_padding_masks": True},
    )
    with torch.inference_mode():
        ordinary = model(torch.tensor([[1, 2, 3]]), use_cache=True)
        with ComponentTransport(adapter, cache):
            assert cache._mask_dispatch_active is True
            assert cache._mosaickv_internal_padding_masks is True
            assert adapter._append_decode_attention_mask(sentinel) is None
            for state in (
                SimpleNamespace(past_key_values=cache, attention_mask=None, model_state={}),
                SimpleNamespace(
                    past_key_values=cache,
                    attention_mask=torch.ones(1, 3),
                    model_state={"mosaickv_internal_padding_masks": True},
                ),
            ):
                with pytest.raises(ValueError):
                    adapter._append_decode_attention_mask(state)
            other = SimpleNamespace(
                past_key_values=object(), attention_mask="original", model_state={}
            )
            assert adapter._append_decode_attention_mask(other) == "original"
            for position, token in enumerate((4, 5, 6), 3):
                expected = model(
                    torch.tensor([[token]]),
                    past_key_values=ordinary.past_key_values,
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
            assert torch.equal(
                model(torch.tensor([[1, 2, 3]]), use_cache=True).logits, ordinary.logits
            )
    assert modeling_llama.eager_attention_forward is eager_before
    assert cache._mask_dispatch_active is False
    assert adapter._cache_length is length_before
    assert adapter._append_decode_attention_mask is mask_before
    assert [layer.self_attn.forward for layer in model.model.layers] == forwards_before
    assert [len(layer.self_attn._forward_hooks) for layer in model.model.layers] == hooks_before
    assert all(torch.equal(t, old) for t, old in zip(packet.tensors(), before, strict=True))
    assert cache.storage_record()["attention_calls_per_layer"] == [3]


def test_exception_restores_eager_forward_masks_hooks_and_borrowed_incoming(zero_model):
    model = zero_model
    adapter = adapter_for(model)
    cache = ComponentCache((ragged_packet(3, fail=True),), updates=16)
    eager_before = modeling_llama.eager_attention_forward
    mask_before = adapter._append_decode_attention_mask
    length_before = adapter._cache_length
    forwards_before = [layer.self_attn.forward for layer in model.model.layers]
    hooks_before = [len(layer.self_attn._forward_hooks) for layer in model.model.layers]
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="injected component fill"),
        ComponentTransport(adapter, cache),
    ):
        model(
            torch.tensor([[4]]),
            past_key_values=cache,
            cache_position=torch.tensor([3]),
            use_cache=True,
        )
    assert modeling_llama.eager_attention_forward is eager_before
    assert cache._mask_dispatch_active is False
    assert adapter._append_decode_attention_mask is mask_before
    assert adapter._cache_length is length_before
    assert [layer.self_attn.forward for layer in model.model.layers] == forwards_before
    assert [len(layer.self_attn._forward_hooks) for layer in model.model.layers] == hooks_before
    assert all(not layer.incoming and not layer.leased for layer in cache.layers)
    assert cache.storage_record()["outstanding_incoming_KV"] == 0
