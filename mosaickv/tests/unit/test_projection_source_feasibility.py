import pytest
import torch

from mosaickv.evaluation.projection_source_feasibility import (
    aggregate,
    attention_output,
    errors,
    project,
)


def test_causal_mask_prevents_future_prompt_leakage_even_with_large_future_value():
    q = torch.ones(1, 1, 2, 128).bfloat16()
    k = torch.ones(1, 1, 4, 128).bfloat16()
    v = torch.zeros_like(k)
    v[..., 3, :] = 100
    out = attention_output(q, k, v, torch.arange(4)[None], torch.tensor([1, 3]))
    assert (out[..., 0, :] == 0).all()
    assert (out[..., 1, :] == 25).all()
    with pytest.raises(ValueError):
        attention_output(
            q, k[..., 2:, :], v[..., 2:, :], torch.tensor([[2, 3]]), torch.tensor([0, 1])
        )


def test_projection_and_aggregation_are_deterministic_and_weights_unchanged():
    torch.manual_seed(144)
    x = torch.randn(1, 17, 256).bfloat16()
    weight = torch.randn(256, 256).bfloat16()
    original = weight.clone()
    phase = torch.ones(1, 17, 128).bfloat16(), torch.zeros(1, 17, 128).bfloat16()
    a = project(x, weight, heads=2)
    b = project(x, weight, heads=2, phases=phase)
    assert torch.equal(a, b)
    assert torch.equal(weight, original)
    result = aggregate([errors(a, b)])
    assert result["relative_mse"] == 0 and result["p95_per_query_head_relative_mse"] == 0
    with pytest.raises(ValueError):
        project(x.float(), weight, heads=2)


def test_headwise_original_positions_are_not_renumbered():
    q = torch.ones(1, 2, 1, 128).bfloat16()
    k = torch.ones(1, 2, 2, 128).bfloat16()
    v = torch.zeros_like(k)
    v[..., 1, :] = 2
    out = attention_output(q, k, v, torch.tensor([[0, 5], [0, 2]]), torch.tensor([3]))
    assert (out[:, 0] == 0).all()
    assert (out[:, 1] == 1).all()
