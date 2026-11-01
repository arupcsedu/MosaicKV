import pytest
import torch

from mosaickv.evaluation.selection_precision_causal import selected_original


def test_original_selected_bits_and_private_owners_and_retention_one():
    torch.manual_seed(66)
    source = tuple(torch.randn(1, 2, 7, 128).bfloat16() for _ in range(2))
    selected = ((0, 3, 6), (1, 2, 6))
    (fmt,) = selected_original((source,), (selected,))
    for actual, original in zip(fmt.materialize(None), source, strict=True):
        for h, ids in enumerate(selected):
            assert torch.equal(actual[0, h], original[0, h, list(ids)])
        assert actual.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    (full,) = selected_original((source,), ((tuple(range(7)),) * 2,))
    assert all(torch.equal(a, b) for a, b in zip(full.materialize(None), source, strict=True))
    with pytest.raises(ValueError):
        selected_original((source,), (((0, 3), (1, 3)),))
