import numpy as np
import pytest
import torch

from mosaickv.evaluation.exact_scale_codec import (
    decode_cpu,
    encode_cpu,
    pack_indices,
    unpack_indices,
    unpack_into,
)


@pytest.mark.parametrize("bits", range(1, 17))
@pytest.mark.parametrize("count", [1, 7, 129])
def test_exact_nonbyte_index_streams(bits, count):
    values = np.random.default_rng(91).integers(0, 1 << bits, count, dtype=np.uint32)
    packed = pack_indices(values, bits)
    assert packed.numel() == (count * bits + 7) // 8
    assert np.array_equal(unpack_indices(packed, count, bits), values)


@pytest.mark.parametrize("kind", ["raw", "word", "low16"])
def test_all_scale_word_formats_roundtrip_and_count_every_tensor(kind):
    generator = torch.Generator().manual_seed(31)
    if kind == "raw":
        scale = torch.rand(129, generator=generator).reshape(3, 43)
    elif kind == "word":
        scale = (torch.arange(4096) % 7 + 1).float().reshape(32, 128) / 7
    else:
        scale = torch.randint(200, 1800, (4096,), generator=generator).float().reshape(32, 128) / 7
    packet = encode_cpu(scale)
    assert packet.kind == kind
    assert packet.storage_bytes <= scale.untyped_storage().nbytes()
    assert torch.equal(decode_cpu(packet).view(torch.int32), scale.view(torch.int32))
    target = torch.full((scale.numel() + 17,), -99.0)
    unpack_into(packet, target)
    assert torch.equal(target[: scale.numel()].view(torch.int32), scale.flatten().view(torch.int32))
    assert bool((target[scale.numel() :] == -99).all())
    assert packet.storage_bytes == sum(t.untyped_storage().nbytes() for t in packet.tensors())


def test_bad_scale_metadata_and_workspace_reject():
    for scale in (
        torch.tensor([0.0]),
        torch.tensor([float("nan")]),
        torch.tensor([-1.0]),
        torch.ones(4).bfloat16(),
    ):
        with pytest.raises(ValueError):
            encode_cpu(scale)
    packet = encode_cpu(torch.ones(4096))
    with pytest.raises(ValueError):
        unpack_into(packet, torch.empty(4096).bfloat16())
    with pytest.raises(ValueError):
        unpack_into(packet, torch.empty(1))


def test_lossless_no_hidden_minimum_or_scale_rounding():
    scale = torch.arange(1, 8193).float().div(7).reshape(32, 256)
    original = scale.clone()
    packet = encode_cpu(scale)
    assert torch.equal(scale, original)
    assert packet.payload.untyped_storage().data_ptr() != scale.untyped_storage().data_ptr()
    assert torch.equal(decode_cpu(packet).view(torch.int32), original.view(torch.int32))
