"""CPU-only format, boundary, untouched-history and storage regressions."""

import pytest
import torch

from mosaickv.evaluation.lossless_bf16 import PackedBF16, pack_cpu
from mosaickv.evaluation.packed_attention import attend, packet


@pytest.mark.parametrize("length", [1, 7, 8, 9, 31])
@pytest.mark.parametrize("used", [0, 1, 5])
def test_reference_exact(length, used):
    g = torch.Generator().manual_seed(133)
    key, value = [torch.randn(1, 2, length, 128, generator=g).bfloat16() for _ in range(2)]
    query = torch.randn(1, 2, 1, 128, generator=g).bfloat16()
    history = tuple(torch.randn(1, 2, 8, 128, generator=g).bfloat16() for _ in range(2))
    snapshot = tuple(t.clone() for t in history)
    a, p = attend(
        query, packet(pack_cpu(key)), packet(pack_cpu(value)), history, used=used, scaling=128**-0.5
    )
    expected_k = torch.cat((key, history[0][..., :used, :]), -2)
    expected_v = torch.cat((value, history[1][..., :used, :]), -2)
    expected_p = torch.softmax(
        query @ expected_k.transpose(-2, -1) * 128**-0.5, -1, dtype=torch.float32
    ).bfloat16()
    assert torch.equal(p, expected_p)
    assert torch.equal(a, (expected_p @ expected_v).transpose(1, 2))
    assert all(torch.equal(a, b) for a, b in zip(history, snapshot, strict=True))


def test_metadata_is_charged():
    p = packet(pack_cpu(torch.ones(1, 2, 8, 128, dtype=torch.bfloat16)))
    assert p.storage_bytes == p.payload.storage_bytes + p.checkpoints.untyped_storage().nbytes()
    assert p.checkpoints.numel() == 16
    assert p.storage_bytes < 2 * 2 * 8 * 128


def test_rejects_unsupported_shape_and_window():
    with pytest.raises(ValueError, match="BHN128"):
        packet(pack_cpu(torch.ones(1, 1, 3, 64, dtype=torch.bfloat16)))
    p = pack_cpu(torch.ones(1, 1, 3, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="window"):
        packet(PackedBF16(p.shape, p.raw, p.buffers, shared_literal_window=True))


@pytest.mark.parametrize("used,scale", [(-1, 1.0), (9, 1.0), (0, float("nan")), (0, 0.0)])
def test_bad_history_and_scaling(used, scale):
    p = packet(pack_cpu(torch.ones(1, 2, 8, 128, dtype=torch.bfloat16)))
    history = (torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16),) * 2
    with pytest.raises(ValueError):
        attend(
            torch.ones(1, 2, 1, 128, dtype=torch.bfloat16), p, p, history, used=used, scaling=scale
        )
