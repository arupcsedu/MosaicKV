import pytest
import torch

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
