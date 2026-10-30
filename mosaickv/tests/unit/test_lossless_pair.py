"""CPU-only bit-exact pair transactions and malformed-transaction rejection."""

import pytest
import torch

from mosaickv.evaluation.lossless_bf16 import PackedBF16, pack_cpu, unpack_pair_into


@pytest.mark.parametrize(
    "raw_key,raw_value", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("used", [1, 16, 31])
def test_pair_preserves_all_bits_and_copies_only_used_history(raw_key, raw_value, used):
    torch.manual_seed(5)
    # Include every NaN payload, infinity and signed zero, with partial codec blocks.
    source = torch.arange(65536).to(torch.int16).view(torch.bfloat16)
    sources = (
        source[:65520].reshape(1, 3, 1680, 13),
        source.flip(0)[:65520].reshape(1, 3, 1680, 13),
    )
    bodies = tuple(
        PackedBF16(tuple(s.shape), True, (s.clone(),)) if raw else pack_cpu(s, raw_fallback=False)
        for s, raw in zip(sources, (raw_key, raw_value), strict=True)
    )
    histories = tuple(torch.randn(1, 3, 32, 13).bfloat16() for _ in range(2))
    targets = tuple(torch.empty(1, 3, 1680 + used, 13, dtype=torch.bfloat16) for _ in range(2))
    before = tuple(t.clone() for t in (*sources, *histories))
    unpack_pair_into(bodies, histories, targets)
    for s, h, target in zip(sources, histories, targets, strict=True):
        expected = torch.cat((s, h[..., :used, :]), 2)
        assert torch.equal(target.view(torch.int16), expected.view(torch.int16))
    for a, b in zip(before, (*sources, *histories), strict=True):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_pair_rejects_invalid_suffix_before_any_write():
    tensor = torch.ones(1, 2, 9, 8, dtype=torch.bfloat16)
    body = pack_cpu(tensor)
    targets = (tensor.clone(), tensor.clone())
    with pytest.raises(ValueError, match="nonempty"):
        unpack_pair_into((body, body), targets, targets)
    target = torch.full((1, 2, 12, 8), -99, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="compatible"):
        unpack_pair_into((body, body), (tensor[..., :2, :].contiguous(),) * 2, (target,) * 2)
    assert bool((target == -99).all())


def test_fused_kernel_specializes_only_layout_not_sequence_lengths():
    from mosaickv.evaluation.lossless_bf16_kernel import _decode_pair

    assert {_decode_pair.arg_names[i] for i in _decode_pair.constexprs} == {
        "DIM",
        "K_RAW",
        "V_RAW",
        "BLOCK",
    }
