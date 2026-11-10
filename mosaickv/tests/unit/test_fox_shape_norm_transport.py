"""CPU-only contracts for opt-in held-token native normalization geometry.

Tiny random-weight fixtures are implementation checks, not pretrained evidence.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.fox_shape_transport import (
    ShapeTransportOperations,
    padded_normalization,
    shape_matched_transport,
)


@pytest.fixture
def tiny():
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(718)
    config = MistralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        sliding_window=None,
        rms_norm_eps=1e-5,
    )
    model = MistralForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    adapter = SimpleNamespace(model=model, _language_layers=lambda: model.model.layers)
    norms = {
        name: module
        for name, module in model.named_modules()
        if name.endswith(("input_layernorm", "post_attention_layernorm", ".norm"))
    }
    assert len(norms) == 5
    return model, adapter, norms


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_norm_helper_uses_full_buffer_same_values_weights_epsilon_dtype(dtype):
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    norm = MistralRMSNorm(8, eps=1e-5).to(dtype=dtype).eval()
    norm.weight.data.copy_(torch.arange(1, 9, dtype=dtype))
    inputs = torch.arange(8, dtype=dtype).reshape(1, 1, 8)
    original_inputs, original_weight = inputs.clone(), norm.weight.detach().clone()
    calls = []

    def original(value):
        calls.append(value)
        return norm(value)

    output = padded_normalization(original, inputs, 17)
    assert len(calls) == 1 and calls[0].shape == (1, 17, 8)
    assert calls[0].is_contiguous()
    assert not calls[0][:, :-1].count_nonzero()
    assert torch.equal(calls[0][:, -1:], inputs)
    assert calls[0].data_ptr() != inputs.data_ptr()
    assert torch.equal(output, norm(calls[0])[:, -1:])
    assert output.dtype == dtype and output.is_contiguous()
    assert output.untyped_storage().nbytes() == output.numel() * output.element_size()
    assert torch.equal(inputs, original_inputs)
    assert torch.equal(norm.weight, original_weight)
    assert norm.variance_epsilon == 1e-5


@pytest.mark.parametrize("rows", [True, 1, 0, 4097, 2.0, "2"])
def test_norm_helper_rejects_unregistered_rows(rows):
    with pytest.raises(ValueError, match="row count"):
        padded_normalization(lambda x: x, torch.ones(1, 1, 4), rows)


@pytest.mark.parametrize(
    "inputs",
    [None, torch.ones(1, 4), torch.ones(2, 1, 4), torch.ones(1, 2, 4), torch.ones(1, 1, 4).long()],
)
def test_norm_helper_refuses_prefill_malformed_or_nonfloating_input(inputs):
    with pytest.raises(NotImplementedError, match="one batch-one token"):
        padded_normalization(lambda x: x, inputs, 6)


@pytest.mark.parametrize("call", [lambda x: x[:, :1], lambda x: x.double(), lambda x: None])
def test_norm_helper_rejects_changed_output_contract(call):
    with pytest.raises(ValueError, match="output geometry"):
        padded_normalization(call, torch.ones(1, 1, 4), 6)


def test_historical_default_has_identical_three_scalar_counters_and_no_norm_patch(tiny):
    _, adapter, norms = tiny
    original = {name: module.forward for name, module in norms.items()}
    with shape_matched_transport(adapter, 6) as operations:
        assert dict(operations) == dict(
            linear_padded_calls=0, attention_padded_calls=0, observed_prefill_layers=0
        )
        assert all(norms[name].forward == forward for name, forward in original.items())
        with pytest.raises(ValueError, match="opt-in"), operations.boundary_normalization():
            pass


def test_norm_phase_is_explicit_single_use_not_nested_and_counts_are_json_scalars():
    operations = ShapeTransportOperations(native_norm_boundary=True, norm_names=("final_norm",))
    with operations.boundary_normalization():
        assert operations._norm_active
        with pytest.raises(ValueError, match="single-use"), operations.boundary_normalization():
            pass
        operations._norm_calls["final_norm"] = 1
    assert not operations._norm_active
    assert operations["normalization_phase_entries"] == 1
    assert operations["normalization_phase_completions"] == 1
    assert all(type(value) is int for value in operations.values())
    assert json.loads(json.dumps(dict(operations))) == dict(operations)
    with pytest.raises(ValueError, match="single-use"), operations.boundary_normalization():
        pass


def test_partial_norm_forward_cannot_be_recorded_as_complete():
    operations = ShapeTransportOperations(native_norm_boundary=True, norm_names=("final_norm",))
    with (
        pytest.raises(ValueError, match="one native normalization"),
        operations.boundary_normalization(),
    ):
        pass
    assert operations["normalization_phase_completions"] == 0
    assert not operations._norm_active


def test_norm_transport_calls_every_language_norm_and_never_vision(tiny, monkeypatch):
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm

    model, adapter, norms = tiny
    model.vision_norm = MistralRMSNorm(16).eval()
    calls = []
    native = MistralRMSNorm.forward

    def observe(self, inputs):
        calls.append((id(self), tuple(inputs.shape)))
        return native(self, inputs)

    monkeypatch.setattr(MistralRMSNorm, "forward", observe)
    singleton = torch.randn(1, 1, 16)
    full = torch.randn(1, 6, 16)
    weights = tuple(p.detach().clone() for p in model.parameters())
    epsilons = tuple(norm.variance_epsilon for norm in norms.values())
    with (
        torch.inference_mode(),
        shape_matched_transport(adapter, 6, native_norm_boundary=True) as operations,
    ):
        for norm in norms.values():
            norm(full)  # Ordinary prefill forwards are untouched.
        with operations.boundary_normalization():
            for norm in norms.values():
                norm(singleton)
            model.vision_norm(singleton)  # Not a decoder-owned normalization.
        for norm in norms.values():
            norm(singleton)  # Later forwards use native one-row geometry.
        assert operations["normalization_padded_calls"] == 5
        assert operations["normalization_expected_calls"] == 5
        assert operations["normalization_input_calls"] == 2
        assert operations["normalization_post_attention_calls"] == 2
        assert operations["normalization_final_calls"] == 1
    assert [shape[1] for _, shape in calls] == [6] * 10 + [1] * 6
    assert calls[10][0] == id(model.vision_norm)
    assert all(torch.equal(p, w) for p, w in zip(model.parameters(), weights, strict=True))
    assert tuple(norm.variance_epsilon for norm in norms.values()) == epsilons


def test_norm_exception_restores_native_methods_and_closes_handle(tiny):
    from transformers.models.mistral import modeling_mistral

    _, adapter, norms = tiny
    forwards = {name: norm.forward for name, norm in norms.items()}
    attention = modeling_mistral.eager_attention_forward
    with (
        pytest.raises(RuntimeError, match="sentinel"),
        shape_matched_transport(adapter, 6, native_norm_boundary=True) as operations,
        operations.boundary_normalization(),
    ):
        next(iter(norms.values()))(torch.ones(1, 1, 16))
        raise RuntimeError("sentinel")
    assert not operations._norm_active
    assert operations["normalization_phase_completions"] == 0
    assert all(norms[name].forward == forward for name, forward in forwards.items())
    assert modeling_mistral.eager_attention_forward is attention
    with pytest.raises(ValueError, match="open opt-in"), operations.boundary_normalization():
        pass


def test_phase_cannot_cover_two_forwards_or_prefill(tiny):
    _, adapter, norms = tiny
    norm = next(iter(norms.values()))
    with (
        shape_matched_transport(adapter, 6, native_norm_boundary=True) as operations,
        pytest.raises(ValueError, match="second forward"),
        operations.boundary_normalization(),
    ):
        norm(torch.ones(1, 1, 16))
        norm(torch.ones(1, 1, 16))
    with (
        shape_matched_transport(adapter, 6, native_norm_boundary=True) as operations,
        pytest.raises(NotImplementedError, match="one batch-one token"),
        operations.boundary_normalization(),
    ):
        norm(torch.ones(1, 6, 16))


@pytest.mark.parametrize("bad_flag", [1, "true", None])
def test_norm_opt_in_must_be_boolean(tiny, bad_flag):
    _, adapter, _ = tiny
    with (
        pytest.raises(ValueError, match="explicit boolean"),
        shape_matched_transport(adapter, 6, native_norm_boundary=bad_flag),
    ):
        pass


@pytest.mark.parametrize(
    "scope",
    ["missing_model", "wrong_final_norm", "shared_norm", "changed_forward", "training", "epsilon"],
)
def test_norm_transport_rejects_unaudited_scope(tiny, scope):
    model, adapter, _ = tiny
    if scope == "missing_model":
        del adapter.model
    elif scope == "wrong_final_norm":
        model.model.norm = torch.nn.LayerNorm(16).eval()
    elif scope == "shared_norm":
        model.model.norm = model.model.layers[0].input_layernorm
    elif scope == "changed_forward":
        model.model.norm.forward = lambda x: x
    elif scope == "training":
        model.model.norm.train()
    else:
        model.model.norm.variance_epsilon = float("nan")
    with (
        pytest.raises(NotImplementedError),
        shape_matched_transport(adapter, 6, native_norm_boundary=True),
    ):
        pass


@pytest.mark.parametrize(
    "bad_input", [torch.ones(1, 1, 16, dtype=torch.float64), torch.ones(1, 1, 8)]
)
def test_norm_transport_rejects_precision_or_hidden_dimension_change(tiny, bad_input):
    model, adapter, _ = tiny
    with (
        shape_matched_transport(adapter, 6, native_norm_boundary=True) as operations,
        pytest.raises(ValueError),
        operations.boundary_normalization(),
    ):
        model.model.norm(bad_input)


def test_tiny_mistral_parity_private_cache_positions_continuation_and_norm_scoping(tiny):
    model, adapter, norms = tiny
    weights = tuple(p.detach().clone() for p in model.parameters())
    original_forwards = {name: norm.forward for name, norm in norms.items()}
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with (
        torch.inference_mode(),
        shape_matched_transport(adapter, ids.shape[1], native_norm_boundary=True) as operations,
    ):
        full = model(ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
        dummy = ids.clone()
        dummy[:, -1] = 0
        mask = torch.ones_like(ids)
        mask[:, -1] = 0
        prefix = model(dummy, attention_mask=mask, use_cache=True, logits_to_keep=1)
        prefix.past_key_values.crop(5)
        assert operations["normalization_padded_calls"] == 0
        with operations.boundary_normalization():
            split = model(
                ids[:, -1:],
                attention_mask=torch.ones_like(ids),
                past_key_values=prefix.past_key_values,
                cache_position=torch.tensor([5]),
                position_ids=torch.tensor([[5]]),
                use_cache=True,
                logits_to_keep=1,
            )
        assert operations["normalization_phase_completions"] == 1
        assert operations["normalization_padded_calls"] == 5
        assert split.past_key_values.get_seq_length() == 6
        assert full.past_key_values is not split.past_key_values
        assert torch.allclose(full.logits, split.logits, atol=1e-6, rtol=0)
        for aa, bb in zip(full.past_key_values.layers, split.past_key_values.layers, strict=True):
            assert aa.keys.shape == bb.keys.shape == (1, 2, 6, 4)
            assert aa.values.shape == bb.values.shape
            assert aa.keys.data_ptr() != bb.keys.data_ptr()
            assert aa.values.data_ptr() != bb.values.data_ptr()
            assert torch.allclose(aa.keys, bb.keys, atol=1e-6, rtol=0)
            assert torch.allclose(aa.values, bb.values, atol=1e-6, rtol=0)
        operations_at_boundary = dict(operations)
    assert all(norms[name].forward == value for name, value in original_forwards.items())
    # The new transport ends at the boundary. All continuation forwards—not
    # merely normalization—are the unmodified installed model operations.
    with torch.inference_mode():
        aa, bb, next_id = full, split, full.logits[:, -1].argmax(-1, keepdim=True)
        for step in range(6, 22):
            kwargs = dict(
                attention_mask=torch.ones(1, step + 1, dtype=torch.long),
                cache_position=torch.tensor([step]),
                position_ids=torch.tensor([[step]]),
                use_cache=True,
                logits_to_keep=1,
            )
            aa = model(next_id, past_key_values=aa.past_key_values, **kwargs)
            bb = model(next_id, past_key_values=bb.past_key_values, **kwargs)
            assert torch.allclose(aa.logits, bb.logits, atol=1e-6, rtol=0)
            assert torch.equal(aa.logits.argmax(-1), bb.logits.argmax(-1))
            assert bb.past_key_values.get_seq_length() == step + 1
            next_id = aa.logits[:, -1].argmax(-1, keepdim=True)
    assert dict(operations) == operations_at_boundary
    assert all(norms[name].forward == value for name, value in original_forwards.items())
    assert all(torch.equal(p, w) for p, w in zip(model.parameters(), weights, strict=True))
