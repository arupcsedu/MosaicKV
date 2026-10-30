import pytest
import torch

from mosaickv.evaluation.precision_corrections import fund_corrections, stable_largest
from mosaickv.evaluation.precision_tier import construct


def test_ranking_ties_zero_scores_and_determinism():
    score = torch.tensor([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
    assert stable_largest(score, 1).tolist() == [2]
    assert stable_largest(score, 3).tolist() == [1, 2, 3]
    assert stable_largest(score, 6).tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("mode", ["mse", "query"])
def test_exceptions_are_exact_funded_and_reduce_source_error(mode):
    generator = torch.Generator().manual_seed(8)
    pair = tuple(torch.randn(1, 2, 140, 128, generator=generator).bfloat16() for _ in range(2))
    base = construct(pair, bits=3, encoded_exact=True)
    raw = construct(pair, bits=3)
    kwargs = (
        dict(key_weight=torch.ones(1, 2, 1, 128), value_weight=torch.ones(1, 2, 140, 1))
        if mode == "query"
        else {}
    )
    corrected, report = fund_corrections(base, pair, fraction=0.005, **kwargs)
    assert corrected.component_bytes <= raw.component_bytes
    assert report["exception_bytes"] <= report["saved_text_and_map_bytes"]
    assert report["exception_bytes"] > 0
    before, after = base.materialize(), corrected.materialize()
    for original, a, b in zip(pair, before, after, strict=True):
        assert (original.float() - b.float()).square().sum() <= (
            original.float() - a.float()
        ).square().sum()
    target = tuple(torch.full((1, 2, 151, 128), -5, dtype=torch.bfloat16) for _ in range(2))
    corrected.fill(None, target, direct=True)
    assert all(torch.equal(a, b[..., :140, :]) for a, b in zip(after, target, strict=True))
    assert all((t[..., 140:, :] == -5).all() for t in target)


@pytest.mark.parametrize("bad", [-1.0, 0.01, float("nan"), True])
def test_invalid_exception_fraction_rejected(bad):
    pair = (torch.zeros(1, 2, 20, 128, dtype=torch.bfloat16),) * 2
    with pytest.raises(ValueError, match="fraction"):
        fund_corrections(construct(pair, bits=3, encoded_exact=True), pair, fraction=bad)


def test_prompt_weights_read_only_and_no_label_interface():
    import inspect
    from types import SimpleNamespace

    from mosaickv.evaluation.precision_corrections import prompt_weights

    projection = torch.nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(256).bfloat16())
    before = projection.weight.detach().clone()
    record = dict(
        normalized=torch.ones(1, 25, 256, dtype=torch.bfloat16),
        cos=torch.ones(1, 25, 128, dtype=torch.bfloat16),
        sin=torch.zeros(1, 25, 128, dtype=torch.bfloat16),
    )
    key, value = prompt_weights(
        SimpleNamespace(q_proj=projection, head_dim=128),
        record,
        torch.ones(1, 2, 25, 25) / 25,
        (tuple(range(19)),) * 2,
    )
    assert key.shape == (1, 2, 1, 128) and value.shape == (1, 2, 19, 1)
    assert not key.requires_grad and not value.requires_grad
    assert torch.equal(projection.weight, before)
    assert torch.allclose(value, torch.full_like(value, 1 / 25))
    assert set(inspect.signature(fund_corrections).parameters) == {
        "fmt",
        "original",
        "fraction",
        "key_weight",
        "value_weight",
    }
