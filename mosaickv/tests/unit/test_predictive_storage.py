import pytest
import torch

from mosaickv.evaluation.lossless_bf16 import pack_cpu, unpack_cpu
from mosaickv.evaluation.predictive_storage import (
    decode_prediction,
    encode_prediction,
    pack_sparse_high,
    unpack_sparse_high,
)


@pytest.mark.parametrize("kind", ["none", "xor16", "ordered_delta16"])
def test_all_bf16_patterns_roundtrip_through_prediction_and_codec(kind):
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    source = words.view(torch.bfloat16).reshape(1, 2, 256, 128)
    predicted = encode_prediction(source, kind)
    packed = pack_cpu(predicted)
    reconstructed = decode_prediction(unpack_cpu(packed), kind)
    assert torch.equal(reconstructed.view(torch.int16), source.view(torch.int16))
    assert torch.equal(words, torch.arange(65536, dtype=torch.int32).to(torch.int16))
    sparse = pack_sparse_high(predicted)
    decoded = decode_prediction(unpack_sparse_high(sparse), kind)
    assert torch.equal(decoded.view(torch.int16), source.view(torch.int16))


@pytest.mark.parametrize("length", [1, 7, 15, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("kind", ["xor16", "ordered_delta16"])
def test_partial_groups_heads_and_strides_never_mix(length, kind):
    torch.manual_seed(length)
    source = torch.randn(1, 3, length * 2, 12).bfloat16()[..., ::2, :]
    reference = source.clone()
    recovered = decode_prediction(encode_prediction(source, kind), kind)
    assert torch.equal(recovered.view(torch.int16), reference.view(torch.int16))
    assert torch.equal(source, reference)


def test_rejects_unregistered_or_non_bf16_input():
    with pytest.raises(ValueError):
        encode_prediction(torch.ones(1, 2, 3, 4), "xor16")
    with pytest.raises(ValueError):
        decode_prediction(torch.ones(1, 2, 3, 4, dtype=torch.bfloat16), "adaptive")


def test_sparse_packet_counts_corruption_and_partial_blocks():
    from dataclasses import replace

    source = torch.randint(-32768, 32767, (1, 3, 17, 11), dtype=torch.int16).view(torch.bfloat16)
    packet = pack_sparse_high(source)
    assert torch.equal(unpack_sparse_high(packet).view(torch.int16), source.view(torch.int16))
    with pytest.raises(ValueError, match="offsets"):
        unpack_sparse_high(replace(packet, offsets=packet.offsets + 1))
