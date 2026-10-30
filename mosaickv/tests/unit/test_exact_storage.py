"""Synthetic-only membership, bit patch and strided storage checks."""

import pytest
import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch, token_partition
from mosaickv.evaluation.lossless_bf16 import pack_cpu, unpack_into


def test_token_partition_covers_each_pair_once_and_does_not_change_membership():
    positions = ((0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 5, 6))
    source, direct = token_partition(positions)
    assert source == (0, 1)
    assert direct == (2, 5, 8, 10, 11)
    assert len(direct) + sum(p in source for h in positions for p in h) == 12
    assert token_partition(positions) == (source, direct)
    with pytest.raises(ValueError, match="uniform"):
        token_partition(((0,), (0, 1)))


def test_patch_preserves_every_bit_including_signed_zero_and_nan_payload():
    words = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    original = words.view(torch.bfloat16)
    altered = (words.to(torch.int32) ^ 1).to(torch.int16).view(torch.bfloat16)
    patch = SparseBitPatch.build(original, altered)
    assert patch.indices.numel() == 65536
    assert patch.indices.dtype == torch.int32
    patch.apply(altered)
    assert torch.equal(original.view(torch.int16), altered.view(torch.int16))
    assert SparseBitPatch.build(original, altered).indices.numel() == 0
    with pytest.raises(ValueError, match="contiguous"):
        patch.apply(torch.ones(5, 5, dtype=torch.bfloat16).T)


@pytest.mark.parametrize("raw", [False, True])
def test_strided_decode_preserves_other_heads_prefix_and_suffix(raw):
    tensor = torch.randn(1, 3, 7, 16).bfloat16()
    packed = pack_cpu(tensor, raw_fallback=raw)
    if raw:
        from mosaickv.evaluation.lossless_bf16 import PackedBF16

        packed = PackedBF16(tuple(tensor.shape), True, (tensor.clone(),))
    backing = torch.full((1, 3, 19, 16), -71.0, dtype=torch.bfloat16)
    target = backing[..., 4:11, :]
    assert not target.is_contiguous()
    unpack_into(packed, target)
    assert torch.equal(target.view(torch.int16), tensor.view(torch.int16))
    assert bool((backing[..., :4, :] == -71).all())
    assert bool((backing[..., 11:, :] == -71).all())
    with pytest.raises(ValueError, match="nonoverlapping"):
        unpack_into(packed, torch.zeros(1, 1, 7, 16, dtype=torch.bfloat16).expand(1, 3, 7, 16))
