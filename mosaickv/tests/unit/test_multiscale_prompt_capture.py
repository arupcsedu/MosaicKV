"""Synthetic mixture math and unchanged forward outputs, not quality measurements."""

import pytest
import torch

from mosaickv.adapters.huggingface.base import AttentionCapture
from mosaickv.config import ConfigurationError, IncrementalHistoryConfig


class AttentionFixture(torch.nn.Module):
    def forward(self, hidden, weights):
        return hidden, weights


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("pooling", ["per_head", "shared_heads"])
def test_mixture_preserves_forward_and_per_query_differences(weight, pooling):
    module = AttentionFixture()
    raw = torch.arange(1, 65, dtype=torch.float32).reshape(1, 4, 4, 4)
    raw = raw / raw.sum(-1, keepdim=True)
    hidden = torch.ones(1, 4)
    old = raw.clone()
    with AttentionCapture(
        [module],
        query_window=2,
        kv_heads=(2,),
        full_prompt_prior_weight=weight,
        full_prompt_prior_head_pooling=pooling,
    ) as capture:
        result = module(hidden, raw)
    actual = capture.result()[0]
    prior = (
        raw.mean(1, keepdim=True).float().mean(-2, keepdim=True)
        if pooling == "shared_heads"
        else raw.mean(-2, keepdim=True)
    )
    expected = (1 - weight) * raw[..., -2:, :] + weight * prior
    expected = expected.reshape(1, 2, 2, 2, 4).mean(2)
    assert result[0] is hidden and result[1] is raw
    assert torch.equal(raw, old)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert torch.allclose(actual.sum(-1), torch.ones_like(actual.sum(-1)), atol=1e-7)
    assert actual.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr()
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()


@pytest.mark.parametrize("weight", [-1, 1.1, float("nan"), True, "0.5"])
def test_invalid_prior_fails_usefully(weight):
    with pytest.raises(ConfigurationError, match="prior"):
        IncrementalHistoryConfig(full_prompt_prior_weight=weight)
    with pytest.raises(ValueError, match="prior"):
        AttentionCapture([AttentionFixture()], query_window=2, full_prompt_prior_weight=weight)


def test_nonzero_prior_requires_bounded_query_window():
    with pytest.raises(ValueError, match="bounded"):
        AttentionCapture([AttentionFixture()], full_prompt_prior_weight=0.5)


def test_shared_prior_uses_the_native_bfloat16_head_reduction_order():
    module = AttentionFixture()
    raw = torch.arange(1, 65, dtype=torch.float32).reshape(1, 4, 4, 4)
    raw = (raw / raw.sum(-1, keepdim=True)).to(torch.bfloat16)
    with AttentionCapture(
        [module],
        query_window=2,
        full_prompt_prior_weight=1.0,
        full_prompt_prior_head_pooling="shared_heads",
    ) as capture:
        module(torch.ones(1, 4), raw)
    expected = raw.mean(1, keepdim=True).float().mean(-2, keepdim=True).expand(1, 4, 2, 4)
    assert torch.equal(capture.result()[0], expected)


@pytest.mark.parametrize("pooling", [None, False, "unknown"])
def test_invalid_head_pooling_fails_closed(pooling):
    with pytest.raises(ConfigurationError, match="pooling"):
        IncrementalHistoryConfig(full_prompt_prior_head_pooling=pooling)


def test_shared_prior_requires_nonzero_weight():
    with pytest.raises(ConfigurationError, match="positive"):
        IncrementalHistoryConfig(full_prompt_prior_head_pooling="shared_heads")
