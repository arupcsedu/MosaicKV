import pytest
import torch

from mosaickv.evaluation.bounded_transport_checks import use as bounded
from mosaickv.evaluation.deferred_transport_checks import Validation, use
from mosaickv.evaluation.shared_source_format import assert_transport_close, transport_corrections


@pytest.mark.parametrize("chunk", [1, 31, 128, 513])
def test_identical_sparse_patch_order_bits_and_same_rejection(chunk):
    generator = torch.Generator().manual_seed(27)
    reference = torch.randn((1, 2, 17, 128), generator=generator).bfloat16()
    actual = reference.clone()
    actual.flatten()[::127] += 0.25
    protected = torch.empty(0, dtype=torch.int32)
    old = transport_corrections((reference,), (actual,), protected)[0]
    new = transport_corrections((reference,), (actual,), protected, chunk_scalars=chunk)[0]
    assert all(torch.equal(a, b) for a, b in zip(old.tensors, new.tensors, strict=True))
    for kwargs in ({}, dict(chunk_scalars=chunk)):
        with pytest.raises(RuntimeError, match="protected"):
            transport_corrections((reference,), (actual,), torch.tensor([0]), **kwargs)
    assert_transport_close(reference, reference.clone(), chunk_scalars=chunk)
    with pytest.raises(AssertionError):
        assert_transport_close(actual, reference, chunk_scalars=chunk)


def test_finite_controls_and_bad_policy_remain_fail_closed():
    x = torch.ones((1, 1, 8, 128), dtype=torch.bfloat16)
    for n in (0, True, 1.5):
        with pytest.raises(ValueError):
            transport_corrections((x,), (x,), torch.tensor([]), chunk_scalars=n)
        with pytest.raises(ValueError):
            assert_transport_close(x, x, chunk_scalars=n)
    y = x.clone()
    y[..., -1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        transport_corrections((x,), (y,), torch.tensor([]), chunk_scalars=128)


@pytest.mark.parametrize("span", [17, 128, 262144])
@pytest.mark.parametrize("coalesce", [False, True])
def test_identical_patch_words_and_full_coverage(span, coalesce):
    x = torch.randn(1, 2, 69, 128, generator=torch.Generator().manual_seed(77)).bfloat16()
    y = x.clone()
    y.flatten()[::127] += 0.25
    ids = torch.empty(0, dtype=torch.int32)
    with use(True, coalesce_nonzero=coalesce, full_predicate_span=True):
        expected = transport_corrections((x,), (y,), ids)[0]
    original = Validation.predicate_span
    with (
        bounded(scalars=span) as report,
        use(True, coalesce_nonzero=coalesce, full_predicate_span=True) as state,
    ):
        actual = transport_corrections((x,), (y,), ids)[0]
        assert_transport_close(x, x.clone())
    assert Validation.predicate_span is original
    assert all(torch.equal(a, b) for a, b in zip(expected.tensors, actual.tensors, strict=True))
    assert report["calls"] == 2
    assert report["maximum_actual_span"] <= span
    assert state.report()["counts"] == dict(finite=1, protected=1, close=1)


@pytest.mark.parametrize(
    "kind,error", [("finite", ValueError), ("protected", RuntimeError), ("close", AssertionError)]
)
def test_failed_predicates_still_abort_before_commit(kind, error):
    x = torch.ones(1, 1, 8, 128, dtype=torch.bfloat16)
    y = x.clone()
    y.flatten()[-1] = float("nan") if kind == "finite" else 2.0
    original = Validation.predicate_span
    with (
        pytest.raises(error),
        bounded(scalars=17),
        use(True, coalesce_nonzero=True, full_predicate_span=True),
    ):
        if kind == "close":
            assert_transport_close(y, x)
        else:
            transport_corrections((x,), (y,), torch.tensor([7], dtype=torch.int32))
    assert Validation.predicate_span is original


@pytest.mark.parametrize("span", [True, 0, -1, 524289])
def test_bad_bounds_are_rejected(span):
    with pytest.raises(ValueError), bounded(scalars=span):
        pass
