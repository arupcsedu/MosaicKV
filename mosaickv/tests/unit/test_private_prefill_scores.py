from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.tiled_softmax import probabilities, use


@pytest.mark.parametrize("rows", [1, 3, 64, 2048])
def test_private_probability_alias_preserves_complete_rows(rows):
    scores = torch.randn(1, 2, 17, 63).bfloat16()
    scores[:, :, 0, :] = float("-inf")
    expected = probabilities(scores, rows=rows)
    candidate = scores.clone()
    actual = probabilities(candidate, rows=rows, out=candidate)
    assert actual is candidate
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("case", ["foreign_output", "dtype_promotion", "gradient"])
def test_invalid_score_overwrite_fails(case):
    x = torch.randn(1, 1, 2, 32).bfloat16()
    out = x.clone() if case == "foreign_output" else x
    if case == "gradient":
        x.requires_grad_(True)
    with pytest.raises(ValueError, match="private-score overwrite"):
        probabilities(x, out=out, dtype=torch.float32 if case == "dtype_promotion" else None)


def test_full_attention_preserves_original_operators_and_caller_inputs():
    from transformers.models.llama import modeling_llama as llama

    module = SimpleNamespace(training=False, num_key_value_groups=1)
    q, k, v = (torch.randn(1, 2, n, 128).bfloat16() for n in (17, 63, 63))
    mask = torch.zeros(1, 1, 17, 63, dtype=torch.bfloat16)
    mask[:, :, :, -1] = float("-inf")
    originals = tuple(t.clone() for t in (q, k, v, mask))
    with torch.inference_mode():
        with use(rows=3):
            reference = llama.eager_attention_forward(module, q, k, v, mask, 0.125)
        with use(rows=3, inplace=True) as report:
            actual = llama.eager_attention_forward(module, q, k, v, mask, 0.125)
    assert report["private_score_buffer_reused"] and report["calls"] == 1
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(reference, actual, strict=True)
    )
    assert all(a.equal(b) for a, b in zip(originals, (q, k, v, mask), strict=True))


def test_mask_promotion_and_autograd_never_silently_change():
    from transformers.models.llama import modeling_llama as llama

    module = SimpleNamespace(training=False, num_key_value_groups=1)
    x = torch.randn(1, 1, 2, 128).bfloat16()
    with use(inplace=True):
        with pytest.raises(ValueError, match="nongrad"):
            llama.eager_attention_forward(module, x, x, x, None, 0.125)
        with torch.inference_mode(), pytest.raises(ValueError, match="mask promotion"):
            llama.eager_attention_forward(module, x, x, x, torch.zeros(1, 1, 2, 2), 0.125)
