"""Analytic signed addition error, independent heads and no label access."""

import torch

from mosaickv.evaluation.draft_addition import layer_addition_plans, per_head_addition_gains


def test_signed_gains_match_exhaustive_renormalized_output():
    torch.manual_seed(77)
    p = torch.rand(1, 2, 3, 6)
    p = p / p.sum(-1, keepdim=True)
    values = torch.randn(1, 2, 6, 4)
    positions = ((0, 1), (0, 2))
    actual = per_head_addition_gains(p, values, positions, 4)
    for h, selected in enumerate(positions):
        kept = (*selected, 4, 5)
        full = p[0, h] @ values[0, h]
        old = (p[0, h, :, kept] @ values[0, h, list(kept)]) / p[0, h, :, kept].sum(-1, keepdim=True)
        for i in set(range(4)) - set(selected):
            add = (*kept, i)
            new = (p[0, h, :, add] @ values[0, h, list(add)]) / p[0, h, :, add].sum(
                -1, keepdim=True
            )
            expected = (
                ((full - old).square().sum(-1) - (full - new).square().sum(-1))
                / full.square().sum(-1)
            ).mean()
            assert torch.allclose(actual[h, i], expected, atol=1e-6)
        assert torch.isneginf(actual[h, list(selected)]).all()


def test_layer_plan_is_a_deterministic_superset_at_exact_one_layer_capacity():
    positions = (((0, 1), (0, 2)), ((0, 1), (0, 2)))
    gains = (
        torch.tensor(
            [[-float("inf"), -float("inf"), 0.3, 0.4], [-float("inf"), 0.2, -float("inf"), 0.1]]
        ),
        torch.tensor(
            [[-float("inf"), -float("inf"), 0.9, 0.1], [-float("inf"), 0.8, -float("inf"), 0.1]]
        ),
    )
    plans = layer_addition_plans(positions, gains)
    assert [p["layer"] for p in plans] == [1, 0]
    assert layer_addition_plans(positions, gains)[0]["added_positions"] == [2, 1]
    for plan in plans:
        for layer, (old, new) in enumerate(zip(positions, plan["positions"], strict=True)):
            for a, b in zip(old, new, strict=True):
                assert set(a).issubset(b)
                assert len(b) == len(a) + int(layer == plan["layer"])
