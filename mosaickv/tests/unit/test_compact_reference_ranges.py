import pytest
import torch

from mosaickv.evaluation.conditional_precision import pack_unprotected


@pytest.mark.parametrize("n", [17, 128, 129, 600, 604])
@pytest.mark.parametrize("pattern", ["random", "signed_zero", "subnormal", "protected"])
@pytest.mark.parametrize("axis", [2, 3])
def test_float_reference_keeps_original_padded_reduction_and_every_packet_bit(n, pattern, axis):
    generator = torch.Generator().manual_seed(310)
    values = torch.randn((1, 2, n + 2, 128), generator=generator).bfloat16()[:, :, 1:-1]
    rows = torch.rand((2, n), generator=generator) > 0.8
    if pattern == "signed_zero":
        values.zero_()
        values[:, :, ::2] = -0.0
    elif pattern == "subnormal":
        values.view(torch.int16)[0, 0, n // 2, 0] = 1
    elif pattern == "protected":
        rows.fill_(True)
    snapshot = values.clone()
    original = pack_unprotected(
        values, rows, bits=3, axis=axis, bf16_prepare=False, compact_ranges=False
    )
    for bf16 in (False, True):
        actual = pack_unprotected(values, rows, bits=3, axis=axis, bf16_prepare=bf16)
        assert all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(original.tensors(), actual.tensors(), strict=True)
        )
    assert torch.equal(values.view(torch.int16), snapshot.view(torch.int16))
