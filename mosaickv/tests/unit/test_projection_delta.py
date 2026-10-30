"""Lossless correction covers all BF16 bits; no downloaded models."""

import pytest
import torch

from mosaickv.evaluation.projection_delta import ProjectionDelta


@pytest.mark.parametrize("shift", [0, 1, -1, 2, -2, 17, 32767])
def test_all_bf16_patterns_and_signed_wraparound_are_exact(shift):
    words = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    original = words.view(torch.bfloat16).reshape(1, 4, 1024, 16)
    candidate = (
        (words.to(torch.int32) + shift).to(torch.int16).view(torch.bfloat16).reshape_as(original)
    )
    patch = ProjectionDelta.build(original, candidate)
    backing = torch.full((1, 4, 1030, 16), -71, dtype=torch.bfloat16)
    target = backing[..., 3:1027, :]
    target.copy_(candidate)
    patch.apply(target)
    assert torch.equal(target.view(torch.int16), original.view(torch.int16))
    assert bool((backing[..., :3, :] == -71).all())
    assert bool((backing[..., 1027:, :] == -71).all())
    assert patch.storage_bytes <= original.numel() * 2
    assert patch.kind == ("zero" if shift == 0 else "two_bit" if abs(shift) == 1 else "raw")


def test_strict_shape_and_destination_validation():
    source = torch.zeros(1, 2, 7, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="matching"):
        ProjectionDelta.build(source, source.float())
    patch = ProjectionDelta.build(source, source)
    with pytest.raises(ValueError, match="nonoverlapping"):
        patch.apply(torch.zeros(1, 1, 7, 8, dtype=torch.bfloat16).expand(1, 2, 7, 8))
