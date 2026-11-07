import pytest
import torch

from mosaickv.evaluation.precision_gain import exact_restoration_gain, repair_risk


def test_signed_gain_matches_individual_actual_attention_recomputations():
    torch.manual_seed(32)
    q = torch.randn(5, 8)
    original = tuple(torch.randn(17, 8) for _ in range(2))
    parent = tuple(t + 0.2 * torch.randn_like(t) for t in original)
    gain, baseline = exact_restoration_gain(q, original, parent, chunk=4)
    target = torch.softmax(q @ original[0].T / 8**0.5, -1) @ original[1]
    expected = []
    for i in range(17):
        actual = tuple(t.clone() for t in parent)
        for t, full in zip(actual, original, strict=True):
            t[i] = full[i]
        output = torch.softmax(q @ actual[0].T / 8**0.5, -1) @ actual[1]
        expected.append(baseline - (target - output).square().mean())
    assert torch.allclose(gain, torch.stack(expected), atol=2e-7, rtol=1e-5)
    assert bool((gain > 0).any())
    assert bool((gain < 0).any())  # restoring a high-norm value is NOT always beneficial


def test_zero_error_has_zero_gain_and_no_repair_risk():
    q, k, v = torch.zeros(2, 8), torch.ones(5, 8), torch.ones(5, 8)
    gain, base = exact_restoration_gain(q, (k, v), (k, v))
    assert base == 0 and torch.allclose(gain, torch.zeros_like(gain), atol=1e-12)
    assert repair_risk(0.0, 1.0, 2.0) == 0
    assert repair_risk(0.5, 4.0, 2.0) == 0.5
    with pytest.raises(ValueError):
        repair_risk(float("nan"), 1.0, 1.0)
