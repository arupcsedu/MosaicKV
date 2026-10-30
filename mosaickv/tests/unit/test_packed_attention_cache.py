from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama

from mosaickv.evaluation.packed_attention_cache import PackedAttentionTransport, construct


@pytest.mark.parametrize("encoded", [False, True])
def test_tiny_model_position_state_and_same_output(encoded):
    torch.manual_seed(771)
    model = (
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
    adapter = SimpleNamespace(
        _language_layers=lambda: model.model.layers, _cache_length=lambda c: c.get_seq_length()
    )
    original_eager = modeling_llama.eager_attention_forward
    with torch.inference_mode():
        prefill = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        pairs = tuple(
            (layer.keys.clone(), layer.values.clone()) for layer in prefill.past_key_values.layers
        )
        reference = model(
            torch.tensor([[5]]),
            past_key_values=prefill.past_key_values,
            cache_position=torch.tensor([4]),
            use_cache=True,
        ).logits
        cache = construct(pairs, encoded=encoded, updates=32)
        initial = cache.storage_record()
        with PackedAttentionTransport(adapter, cache):
            actual = model(
                torch.tensor([[5]]),
                past_key_values=cache,
                cache_position=torch.tensor([4]),
                use_cache=True,
            ).logits
            assert torch.equal(actual, reference)
            assert all(layer.history_length == 1 and not layer.leased for layer in cache.layers)
            # An ordinary request in the same context must keep original eager.
            ordinary = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True).logits
        assert modeling_llama.eager_attention_forward is original_eager
        assert torch.equal(ordinary, prefill.logits)
        record = cache.storage_record()
        assert record["final_active_kv_storage_bytes"] > initial["initial_active_kv_storage_bytes"]
        assert record["outstanding_layer_leases"] == 0
        assert record["history_updates_per_layer"] == [1, 1]


def test_history_growth_storage_and_no_alias():
    source = torch.ones(1, 2, 3, 128, dtype=torch.bfloat16)
    cache = construct(((source, source),), encoded=True, updates=32)
    original = source.clone()
    for i in range(20):
        token = torch.full((1, 2, 1, 128), i, dtype=torch.bfloat16)
        cache.layers[0].update(token, token)
        assert cache.layers[0].leased == ()
    record = cache.storage_record()
    assert record["history_updates_per_layer"] == [20]
    assert torch.equal(source, original)
    assert cache.layers[0].history_capacity == 32
    with pytest.raises(RuntimeError, match="never materialize"):
        cache.layers[0].materialize()
