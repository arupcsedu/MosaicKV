import pytest
import torch

from mosaickv.evaluation.shared_source_format import assert_transport_close


def original(output, expected, chunk):
    for start in range(0, output.numel(), chunk):
        torch.testing.assert_close(
            output.flatten()[start : start + chunk],
            expected.flatten()[start : start + chunk],
            rtol=0.01,
            atol=2**-6,
        )


def accepted(function):
    try:
        function()
    except AssertionError:
        return False
    return True


@pytest.mark.parametrize("chunk", [1, 13, 127, 524288])
@pytest.mark.parametrize("offset", [0, 0.015625, 0.02, 0.1])
def test_same_acceptance_as_original_for_bf16_rounding_boundaries(chunk, offset):
    expected = torch.linspace(-4, 4, 257).bfloat16()
    output = (expected.float() + offset).bfloat16()
    assert accepted(lambda: original(output, expected, chunk)) == accepted(
        lambda: assert_transport_close(output, expected, chunk_scalars=chunk)
    )


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [-float("inf")], [-0.0, 0.0]])
def test_same_nan_infinity_signed_zero_rules(values):
    x = torch.tensor(values, dtype=torch.bfloat16)
    assert accepted(lambda: original(x, x.clone(), 1)) == accepted(
        lambda: assert_transport_close(x, x.clone(), chunk_scalars=1)
    )


def test_success_keeps_every_scalar_but_avoids_per_chunk_assertion(monkeypatch):
    x = torch.arange(51, dtype=torch.bfloat16).reshape(3, 17).T
    assert not x.is_contiguous()
    isclose = torch.isclose
    counts = []

    def counted(a, b, **kwargs):
        counts.append(a.numel())
        return isclose(a, b, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("successful compatible check should not synchronize through assert_close")

    monkeypatch.setattr(torch, "isclose", counted)
    monkeypatch.setattr(torch.testing, "assert_close", forbidden)
    assert_transport_close(x, x.clone(), chunk_scalars=7)
    assert sum(counts) == x.numel()
    assert max(counts) <= 7


def test_dtype_validation_and_failure_still_use_upstream_assertions():
    x = torch.ones(17, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="dtype"):
        assert_transport_close(x, x.float(), chunk_scalars=7)
    y = x.clone()
    y[-1] = 2
    with pytest.raises(AssertionError, match="not close"):
        assert_transport_close(y, x, chunk_scalars=7)


def test_empty_and_invalid_chunks_preserve_prior_behavior():
    x = torch.empty(0, dtype=torch.bfloat16)
    assert_transport_close(x, x)
    for invalid in (0, -1, True, 0.5):
        with pytest.raises(ValueError):
            assert_transport_close(x, x, chunk_scalars=invalid)
