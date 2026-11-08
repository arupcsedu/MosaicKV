from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mosaickv.evaluation.tiled_softmax import probabilities, use


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("tile", (1, 7, 2048))
def test_row_tiling_preserves_full_softmax_exactly(dtype, tile):
    torch.manual_seed(41)
    scores = torch.randn(1, 4, 31, 31).to(dtype)
    scores.masked_fill_(torch.ones(31, 31, dtype=torch.bool).triu(1), -torch.inf)
    expected = F.softmax(scores, -1, dtype=torch.float32).to(dtype)
    actual = probabilities(scores, rows=tile)
    assert torch.equal(actual, expected)
    assert not torch.isnan(actual).any()


def test_attention_preserves_queries_gqa_mask_outputs_and_restores_module():
    from transformers.models.llama import modeling_llama as llama

    torch.manual_seed(4)
    args = (
        SimpleNamespace(training=False, num_key_value_groups=2),
        torch.randn(1, 4, 5, 8),
        torch.randn(1, 2, 7, 8),
        torch.randn(1, 2, 7, 8),
        torch.zeros(1, 1, 5, 7),
        8**-0.5,
    )
    original = llama.eager_attention_forward
    expected = original(*args)
    with use(rows=3) as report:
        actual = llama.eager_attention_forward(*args)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert llama.eager_attention_forward is original and report["calls"] == 1
    with pytest.raises(ValueError):
        probabilities(torch.zeros(2, 4), rows=3)
