from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mosaickv.adapters.huggingface.types import (
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
    PreparedInputs,
)
from mosaickv.cache_state import Modality, ModalitySpan
from mosaickv.evaluation.fox_shape_transport import (
    padded_attention,
    padded_linear,
    padded_prefix,
    remove_dummy_cache_slot,
)


def prepared(n=4):
    return PreparedInputs(
        dict(
            input_ids=torch.arange(n)[None],
            attention_mask=torch.ones((1, n), dtype=torch.long),
            position_ids=torch.arange(n)[None],
            pixel_values=torch.zeros(1, 3, 2, 2),
        ),
        (ModalitySpan(0, n, Modality.TEXT),),
        n,
    )


def test_prefix_dummy_never_reads_held_id_and_preserves_media_and_valid_positions():
    prefix = prepared()
    result = padded_prefix(prefix, placeholder_id=123)
    assert result.logical_sequence_length == 5
    assert result.model_inputs["input_ids"].tolist() == [[0, 1, 2, 3, 123]]
    assert result.model_inputs["attention_mask"].tolist() == [[1, 1, 1, 1, 0]]
    assert result.model_inputs["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert result.model_inputs["pixel_values"] is prefix.model_inputs["pixel_values"]
    assert prefix.model_inputs["input_ids"].shape[-1] == 4


def test_incompatible_prefix_holes_fail_explicitly():
    p = prepared()
    p.model_inputs["attention_mask"][0, 1] = 0
    with pytest.raises(NotImplementedError):
        padded_prefix(p)


def test_linear_uses_original_operation_and_shape_without_weight_changes():
    torch.manual_seed(1)
    original = torch.nn.Linear(4, 3).eval()
    weight = original.weight.clone()
    calls = []

    def call(x):
        calls.append(tuple(x.shape))
        return original(x)

    x = torch.randn(1, 1, 4)
    result = padded_linear(call, x, 7)
    padded = x.new_zeros(1, 7, 4)
    padded[:, -1:] = x
    assert torch.equal(result, original(padded)[:, -1:])
    assert calls == [(1, 7, 4)]
    assert torch.equal(original.weight, weight)


def test_native_prefill_linear_has_identity_inputs():
    x = torch.ones(1, 6, 3)
    assert padded_linear(lambda value: value, x, 6) is x


def eager(module, q, k, v, mask, **kwargs):
    del module, kwargs
    weights = torch.softmax(q @ k.transpose(-1, -2) + mask, dim=-1)
    return (weights @ v).transpose(1, 2).contiguous(), weights


@pytest.mark.parametrize("masked", [False, True])
def test_attention_padding_returns_only_actual_query_and_never_adds_keys(masked):
    torch.manual_seed(2)
    q = torch.randn(1, 2, 1, 3)
    k = torch.randn(1, 2, 5, 3)
    v = torch.randn_like(k)
    mask = torch.zeros(1, 1, 1, 5)
    if masked:
        mask[..., 0] = float("-inf")
    shape_template = torch.empty(1, 7, 2, 3).transpose(1, 2)
    o, p = padded_attention(eager, None, q, k, v, mask, rows=7, stride=shape_template.stride())
    qpad = torch.zeros_like(shape_template)
    qpad[:, :, -1:] = q
    expected, prob = eager(None, qpad, k, v, mask.expand(1, 1, 7, 5).contiguous())
    assert torch.equal(o, expected[:, -1:])
    assert torch.equal(p, prob[..., -1:, :])
    assert k.shape[-2] == 5


def test_cache_trim_restores_original_type_and_positions_without_dummy_alias():
    p = prepared()
    key = torch.randn(1, 2, 5, 3)
    value = torch.randn_like(key)
    snapshot = CacheSnapshot(
        (CacheLayerSnapshot(key, value, 2),), tuple, "tuple", 5, CachedKeyState.POST_ROPE
    )
    adapter = SimpleNamespace(
        borrow_past_key_values=lambda _: snapshot,
        inject_past_key_values=lambda s: tuple((x.key, x.value) for x in s.layers),
    )
    state = DecodeState(((key, value),), torch.ones(1, 5), 5, 5, 5, ())
    result = remove_dummy_cache_slot(adapter, state, p)
    assert (
        result.active_cache_length
        == result.logical_sequence_length
        == result.next_decode_position
        == 4
    )
    assert torch.equal(result.past_key_values[0][0], key[:, :, :4])
    assert (
        result.past_key_values[0][0].untyped_storage().data_ptr()
        != key.untyped_storage().data_ptr()
    )
    assert (
        result.past_key_values[0][0].untyped_storage().nbytes()
        == 1 * 2 * 4 * 3 * key.element_size()
    )


def test_cache_trim_rejects_unexpected_lengths():
    with pytest.raises(ValueError):
        remove_dummy_cache_slot(
            None,
            SimpleNamespace(
                active_cache_length=4, logical_sequence_length=5, next_decode_position=5
            ),
            prepared(),
        )


def test_tiny_random_weight_hf_api_cpu_shape_control_and_restoration():
    from transformers import MistralConfig, MistralForCausalLM
    from transformers.models.mistral import modeling_mistral

    from mosaickv.evaluation.fox_shape_transport import shape_matched_transport

    torch.manual_seed(8)
    config = MistralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        sliding_window=None,
    )
    model = MistralForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    adapter = SimpleNamespace(_language_layers=lambda: model.model.layers)
    original = modeling_mistral.eager_attention_forward
    weights = tuple(p.detach().clone() for p in model.parameters())
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.inference_mode(), shape_matched_transport(adapter, 6) as operations:
        full = model(ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
        dummy = ids.clone()
        dummy[:, -1] = 0
        mask = torch.ones_like(ids)
        mask[:, -1] = 0
        prefix = model(dummy, attention_mask=mask, use_cache=True, logits_to_keep=1)
        prefix.past_key_values.crop(5)
        split = model(
            ids[:, -1:],
            attention_mask=torch.ones_like(ids),
            past_key_values=prefix.past_key_values,
            cache_position=torch.tensor([5]),
            use_cache=True,
            logits_to_keep=1,
        )
        assert float((full.logits - split.logits).abs().max()) <= 1e-6
        assert operations["linear_padded_calls"] > 0
        assert operations["attention_padded_calls"] == 2
        assert split.past_key_values.get_seq_length() == 6
    assert modeling_mistral.eager_attention_forward is original
    assert all(torch.equal(p, w) for p, w in zip(model.parameters(), weights, strict=True))
