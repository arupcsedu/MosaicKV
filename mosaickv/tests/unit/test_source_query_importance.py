import pytest
import torch

from mosaickv.evaluation.source_query_importance import importance


def example():
    torch.manual_seed(571)
    h, n, d = 2, 8, 128
    q = torch.randn(1, h, 3, d).bfloat16()
    k = torch.randn(1, h, n, d).bfloat16()
    v = torch.randn_like(k)
    w = torch.randn(h * d, h * d) * 0.01
    return (
        q,
        k,
        v,
        w,
        w.clone(),
        torch.arange(n).repeat(h, 1),
        torch.arange(5, 8),
        (torch.ones(1, n, d).bfloat16(), torch.zeros(1, n, d).bfloat16()),
        torch.zeros(h, n, dtype=torch.bool),
    )


def test_source_sensitivity_is_finite_and_fully_exact_rows_have_no_influence():
    args = list(example())
    args[-1][:, 2] = True
    weights, proof = importance(*args)
    assert weights.shape == (1, 1, 8, 256) and (weights >= 0).all()
    assert (weights[:, :, 2] == 0).all() and proof["not_full_Jacobian"]
    args[-1][:] = True
    assert (importance(*args)[0] == 0).all()


def test_zero_queries_remove_key_effect_and_zero_projections_give_zero_weights():
    args = list(example())
    args[0].zero_()
    first = importance(*args)[0]
    args[3].mul_(100)
    assert torch.equal(first, importance(*args)[0])
    args[4].zero_()
    assert (importance(*args)[0] == 0).all()
    args[5][0, 0] = 1
    with pytest.raises(ValueError):
        importance(*args)
