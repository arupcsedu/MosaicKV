import pytest
import torch

from mosaickv.evaluation.logit_composition_audit import screen


def test_actual_directions_can_cancel_and_order_is_deterministic():
    reference = torch.tensor([[0.0, 2.0, 1.0]]).repeat(32, 1)
    anchor = reference + torch.tensor([1.0, 0.0, 0.0])
    heads = {0: anchor + torch.tensor([-2.0, 0.0, 0.0])}
    tokens = {3: anchor + torch.tensor([1.0, 0.0, 0.0])}
    result = screen(reference, anchor, heads, tokens)
    assert result[0]["statistics"]["maximum_logit_difference"] == 0
    assert result[0]["statistics"]["guard_pass"]
    assert screen(reference, anchor, heads, tokens) == result
    with pytest.raises(ValueError):
        screen(reference, anchor, {}, tokens)
    with pytest.raises(ValueError):
        screen(reference[:1], anchor, heads, tokens)
