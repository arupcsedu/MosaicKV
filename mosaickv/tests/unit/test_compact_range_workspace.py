import pytest
import torch

from mosaickv.evaluation.conditional_precision import _token_group_preparation, pack_unprotected


@pytest.mark.parametrize("n", [1, 17, 128, 129, 256, 600, 604])
@pytest.mark.parametrize("axis", [2, 3])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_unpadded_ranges_match_original_float_headers_and_codes(n, axis, bits):
    generator = torch.Generator().manual_seed(221)
    storage = torch.randn((1, 2, n + 2, 128), generator=generator).bfloat16()
    values = storage[:, :, 1:-1]
    rows = torch.rand((2, n), generator=generator) > 0.7
    rows[0] = True
    snapshot = storage.clone()
    expected = pack_unprotected(values, rows, bits=bits, axis=axis, bf16_prepare=False)
    actual = pack_unprotected(values, rows, bits=bits, axis=axis)
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(expected.tensors(), actual.tensors(), strict=True)
    )
    assert torch.equal(snapshot.view(torch.uint8), storage.view(torch.uint8))


def test_zero_minimum_declines_without_modifying_original_and_policy_is_strict():
    values = torch.zeros((1, 1, 129, 128), dtype=torch.bfloat16)
    values[:, :, ::2] = -0.0
    rows = torch.zeros((1, 129), dtype=torch.bool)
    snapshot = values.clone()
    assert _token_group_preparation(values, rows) is None
    assert torch.equal(values.view(torch.int16), snapshot.view(torch.int16))
    with pytest.raises(ValueError, match="explicit compact"):
        pack_unprotected(values, rows, bits=3, axis=2, compact_ranges=1)


def test_private_output_and_row_sized_masks_only():
    from torch.utils._python_dispatch import TorchDispatchMode

    values = torch.randn((1, 2, 600, 128)).bfloat16()
    rows = torch.zeros((2, 600), dtype=torch.bool)
    rows[:, ::4] = True
    masks = []

    class Audit(TorchDispatchMode):
        def __torch_dispatch__(self, fn, types, args=(), kwargs=None):
            if str(fn) in ("aten.masked_fill.Scalar", "aten.where.self_out"):
                mask = args[1] if "masked_fill" in str(fn) else args[0]
                masks.append(mask.untyped_storage().nbytes())
            return fn(*args, **(kwargs or {}))

    with Audit():
        result = _token_group_preparation(values, rows)
    assert masks and max(masks) == rows.untyped_storage().nbytes()
    assert result.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
    assert result.is_contiguous() and torch.equal(result[0][~rows], values[0][~rows])
