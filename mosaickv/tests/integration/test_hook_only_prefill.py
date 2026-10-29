"""CPU equivalence against the pinned model's original public collection path."""

import torch
from test_hf_tiny_models import _llava, _runtime_request


def test_private_hooks_preserve_raw_weights_logits_and_kv_without_model_collection(monkeypatch):
    adapter = _llava()
    prepared = adapter.prepare_inputs(_runtime_request("collector").messages)
    forward = adapter._prefill_forward
    outputs = []

    def legacy(inputs):
        result = forward(dict(inputs, output_attentions=True))
        outputs.append(result)
        return result

    monkeypatch.setattr(adapter, "_prefill_forward", legacy)
    old = adapter.prefill(prepared, capture_attentions=True, attention_query_window=2)
    assert outputs[-1].attentions is not None
    outputs.clear()

    def audited(inputs):
        assert inputs["output_attentions"] is False
        result = forward(inputs)
        assert result.attentions is None
        return result

    monkeypatch.setattr(adapter, "_prefill_forward", audited)
    new = adapter.prefill(prepared, capture_attentions=True, attention_query_window=2)
    assert torch.equal(old.logits, new.logits)
    assert old.state.active_cache_length == new.state.active_cache_length
    for a, b in zip(old.attention_weights, new.attention_weights, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(
        old.state.past_key_values.layers, new.state.past_key_values.layers, strict=True
    ):
        assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
    assert new.state.model_state["mosaickv_prefill_attention_collection"][
        "audited_hook_only_llama_4_55_2"
    ]
