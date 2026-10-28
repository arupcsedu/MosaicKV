from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.fox_valid_slots import valid_slot_forward


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize("groups", [1, 2])
def test_masked_holes_replay_native_math_and_original_probability_axis(dtype, groups):
    from transformers.models.llama.modeling_llama import eager_attention_forward

    torch.manual_seed(4)
    owner = SimpleNamespace(num_key_value_groups=groups, training=False)
    q = torch.randn(1, 2 * groups, 1, 4, dtype=dtype)
    k = torch.randn(1, 2, 5, 4, dtype=dtype)
    v = torch.randn_like(k)
    expected = eager_attention_forward(
        owner, q, k, v, torch.zeros(1, 2 * groups, 1, 5, dtype=dtype), scaling=0.5
    )
    ids = torch.tensor([0, 1, 4, 6, 7])
    pk, pv = torch.zeros(1, 2, 8, 4, dtype=dtype), torch.zeros(1, 2, 8, 4, dtype=dtype)
    pk.index_copy_(2, ids, k)
    pv.index_copy_(2, ids, v)
    mask = torch.full((1, 2 * groups, 1, 8), -torch.inf, dtype=dtype)
    mask[..., ids] = 0
    original = pk.clone(), pv.clone()
    got = valid_slot_forward(eager_attention_forward, owner, q, pk, pv, mask, scaling=0.5)
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1][..., ids], expected[1])
    assert bool((got[1][..., [2, 3, 5]] == 0).all())
    assert torch.equal(pk, original[0]) and torch.equal(pv, original[1])


def test_all_valid_fast_path_is_exact_object_identity():
    q, k, v, mask = (
        torch.ones(1, 1, 1, 2),
        torch.ones(1, 1, 3, 2),
        torch.ones(1, 1, 3, 2),
        torch.zeros(1, 1, 1, 3),
    )
    sentinel = object()

    def original(module, query, key, value, attention_mask, **kwargs):
        assert query is q and key is k and value is v and attention_mask is mask
        return sentinel

    assert valid_slot_forward(original, None, q, k, v, mask) is sentinel


def test_head_dependent_masks_and_biases_are_explicitly_unsupported():
    q, k = torch.ones(1, 2, 1, 2), torch.ones(1, 2, 3, 2)
    mask = torch.zeros(1, 2, 1, 3)
    mask[0, 0, 0, 0] = -torch.inf
    with pytest.raises(NotImplementedError, match="head-dependent"):
        valid_slot_forward(None, None, q, k, k, mask)
    mask[0, 1, 0, 1] = -0.5
    with pytest.raises(NotImplementedError, match="biases"):
        valid_slot_forward(None, None, q, k, k, mask)


def test_scoped_transport_restores_installed_function_and_preserves_fullkv(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from transformers.models.llama import modeling_llama
    from verify_hf_adapters import tiny_adapters

    from mosaickv.evaluation.fox_numerics import unpruned_padding_control
    from mosaickv.evaluation.fox_valid_slots import valid_slot_diagnostic
    from mosaickv.evaluation.messages import build_multimodal_messages

    _, adapter = tiny_adapters()[0]
    prepared = adapter.prepare_inputs(
        build_multimodal_messages("same consumed-style synthetic control")
    )
    untouched = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    original = modeling_llama.eager_attention_forward
    with valid_slot_diagnostic(adapter):
        observed = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
        assert torch.equal(observed.logits, untouched.logits)
        rows = unpruned_padding_control(
            adapter, untouched.state, int(untouched.next_token_id.item()), steps=3
        )
        assert all(r["absolute"] == 0 for r in rows)
    assert modeling_llama.eager_attention_forward is original
    with pytest.raises(RuntimeError), valid_slot_diagnostic(adapter):
        raise RuntimeError("restore on failure")
    assert modeling_llama.eager_attention_forward is original
