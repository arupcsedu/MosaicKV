import pytest
import torch

from mosaickv.evaluation.conditional_precision import pack_unprotected


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("axis", [2, 3])
@pytest.mark.parametrize(
    "pattern", ["random", "constant", "signed_zero", "subnormal", "all_protected"]
)
def test_range_preparation_preserves_all_header_and_code_bits(bits, axis, pattern):
    generator = torch.Generator().manual_seed(189)
    backing = torch.randn(1, 2, 19, 128, generator=generator).bfloat16()
    values = backing[:, :, 1:18, :]
    rows = torch.zeros(2, 17, dtype=torch.bool)
    rows[0, :4] = True
    rows[1, -6:] = True
    if pattern == "constant":
        values.fill_(-2.75)
    if pattern == "signed_zero":
        values.zero_()
        values[:, :, ::2, :] = -0.0
    if pattern == "subnormal":
        values.view(torch.int16)[0, 0, 8, 0] = 1
    if pattern == "all_protected":
        rows.fill_(True)
    snapshot = values.clone()
    old = pack_unprotected(values, rows, bits=bits, axis=axis, bf16_prepare=False)
    new = pack_unprotected(values, rows, bits=bits, axis=axis)
    for a, b in zip(old.tensors(), new.tensors(), strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert torch.equal(snapshot.view(torch.int16), values.view(torch.int16))


def test_preparation_policy_is_strict():
    with pytest.raises(ValueError, match="explicit BF16"):
        pack_unprotected(
            torch.ones(1, 2, 16, 128, dtype=torch.bfloat16),
            torch.zeros(2, 16, dtype=torch.bool),
            bits=3,
            axis=2,
            bf16_prepare=1,
        )
