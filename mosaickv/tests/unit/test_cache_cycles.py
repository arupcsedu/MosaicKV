import pytest
import torch

from mosaickv.evaluation.cache_cycles import compile_cycles


@pytest.mark.parametrize("length", [3, 17, 129])
def test_inplace_uint16_permutation_all_bits_and_strided_history(length):
    g = torch.Generator().manual_seed(443)
    positions = tuple(tuple(torch.randperm(length, generator=g).tolist()) for _ in range(4))
    plan = compile_cycles(positions, device="cpu")
    original = (
        torch.arange(4 * (length + 17) * 128)
        .short()
        .view(torch.bfloat16)
        .reshape(1, 4, length + 17, 128)
    )
    index = torch.tensor([sorted(range(length), key=lambda j: p[j]) for p in positions])
    expected = original.view(torch.int16)[:, :, :length].gather(
        2, index[None, :, :, None].expand(1, 4, length, 128)
    )
    a, b = original.clone(), original.clone()
    plan.apply((a, b))
    for target in (a, b):
        assert torch.equal(target[:, :, :length].view(torch.int16), expected.view(torch.int16))
        assert torch.equal(
            target[:, :, length:].view(torch.int16), original[:, :, length:].view(torch.int16)
        )
    assert sum(t.numel() * t.element_size() for t in plan.tensors()) < 4 * length * 128 * 2


def test_identity_needs_no_cycles_and_duplicate_positions_fail():
    assert not compile_cycles(((1, 3, 8),) * 2, device="cpu").heads.numel()
    with pytest.raises(ValueError):
        compile_cycles(((1, 1, 3),), device="cpu")
