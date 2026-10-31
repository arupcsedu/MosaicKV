import pytest
import torch

from mosaickv.evaluation.mse_value_precision import pack_mse_values
from mosaickv.evaluation.precision_tier import pack, unpack_cpu


@pytest.mark.parametrize("kind", ["normal", "outliers", "constant"])
def test_mse_range_uses_identical_bytes_and_never_increases_row_error(kind):
    generator = torch.Generator().manual_seed(924)
    x = torch.randn(1, 3, 17, 128, generator=generator).bfloat16()
    if kind == "outliers":
        x[..., 0] *= 10
    elif kind == "constant":
        x.fill_(2.5)
    original = x.clone()
    baseline, candidate = pack(x, bits=2, axis=3), pack_mse_values(x)
    old, new = unpack_cpu(baseline), unpack_cpu(candidate)
    old_error = (old.float() - x.float()).square().sum(-1)
    new_error = (new.float() - x.float()).square().sum(-1)
    assert bool((new_error <= old_error).all())
    if kind != "constant":
        assert new_error.sum() < old_error.sum()
    else:
        assert torch.equal(new, x)
    assert all(
        a.nbytes == b.nbytes for a, b in zip(baseline.tensors(), candidate.tensors(), strict=True)
    )
    again = pack_mse_values(x)
    assert all(torch.equal(a, b) for a, b in zip(candidate.tensors(), again.tensors(), strict=True))
    assert torch.equal(x, original)


def test_mse_range_rejects_bad_shape_or_dtype():
    with pytest.raises(ValueError, match="128"):
        pack_mse_values(torch.ones(1, 2, 3, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="BF16"):
        pack_mse_values(torch.ones(1, 2, 3, 128))
