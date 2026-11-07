from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_last_layer_readout import (
    ReadoutCapture,
    attention_output,
    compare_logits,
    record_component,
    suffix,
)


def test_exact_suffix_uses_the_real_nonlinear_frozen_modules():
    torch.manual_seed(338)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=torch.nn.Linear(8, 8, bias=False).double()),
        post_attention_layernorm=torch.nn.LayerNorm(8).double(),
        mlp=torch.nn.Sequential(
            torch.nn.Linear(8, 12), torch.nn.SiLU(), torch.nn.Linear(12, 8)
        ).double(),
    )
    norm, head = torch.nn.LayerNorm(8).double(), torch.nn.Linear(8, 19).double()
    residual, output = torch.randn(1, 1, 8).double(), torch.randn(1, 1, 8).double()
    original = residual + layer.self_attn.o_proj(output)
    expected = head(norm(original + layer.mlp(layer.post_attention_layernorm(original))))
    with torch.inference_mode():
        actual = suffix(layer, norm, head, residual, output)
    assert torch.equal(actual, expected.float().reshape(-1))


def test_attention_output_preserves_native_scaling_mask_softmax_and_head_order():
    torch.manual_seed(18)
    q, k, v = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4)
    mask = torch.tensor([0.0, 0.0, -torch.inf, 0.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 7)
    p = (q @ k.transpose(2, 3) * 0.5 + mask).softmax(-1)
    expected = (p @ v).transpose(1, 2).contiguous().reshape(1, 1, -1)
    assert torch.equal(attention_output(q, k, v, mask, 0.5), expected)


def test_read_only_observers_capture_owned_cpu_values_and_restore_patches(monkeypatch):
    from mosaickv.evaluation.post_attention_segment import Transport
    from mosaickv.evaluation.uniform_core_transport import Binding

    def score(binding, *args):
        binding.calls += 1
        return torch.ones(1)

    def values(binding, length):
        return torch.zeros(1, 1, 32, 128)

    monkeypatch.setattr(Binding, "scores", score)
    monkeypatch.setattr(Binding, "values", values)
    query = torch.randn(1, 32, 1, 128).bfloat16()
    cache = torch.randn(1, 32, 19, 128).bfloat16()
    layer = SimpleNamespace(
        attention=SimpleNamespace(layer_idx=31), leased=(cache,), get_seq_length=lambda: 17
    )
    capture = ReadoutCapture(2)
    with capture.use():
        binding = SimpleNamespace(calls=31)
        Binding.scores(binding, query, cache, None, 128**-0.5)
        Binding.values(binding, 17)
        record_component(layer, 0, parent=True)
        record_component(layer, 0, parent=False)
    before = capture.data["active_key"].clone()
    cache.zero_()
    assert torch.equal(capture.data["active_key"], before)
    assert capture.data["parent_key"].shape == (1, 1, 17, 128)
    assert Binding.scores is score and Binding.values is values
    assert Transport.run.__name__ == "run"
    record_component(layer, 1, parent=False)
    assert "active_value" not in capture.data  # scope cannot leak


def test_missing_observations_and_unsupported_layers_fail_closed():
    with pytest.raises(ValueError):
        ReadoutCapture(0, layer=30)
    with pytest.raises(RuntimeError, match="missing real"):
        ReadoutCapture(0).validate(603)


def test_cpu_owned_probe_logits_compare_without_a_gpu_device_assumption():
    actual = torch.tensor([2.0, 3.0, 1.0])
    result = compare_logits(actual.reshape(1, -1), actual)
    assert result == dict(logits_bitwise=True, maximum_logit_difference=0.0, token_agreement=True)
    assert not compare_logits(actual + 1.0, actual)["logits_bitwise"]
    with pytest.raises(ValueError, match="finite"):
        compare_logits(actual, actual[:2])
    with pytest.raises(ValueError, match="finite"):
        compare_logits(actual, torch.tensor([float("nan"), 0.0, 1.0]))
