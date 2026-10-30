"""Synthetic bit-level codec tests, not empirical model-compression results."""

import pytest
import torch

from mosaickv.evaluation.lossless_bf16 import pack_cpu, unpack_cpu, unpack_into


def test_every_bf16_bit_pattern_including_signed_zero_and_nan_payloads():
    tensor = torch.arange(65536).to(torch.int16).view(torch.bfloat16)
    for fallback in (False, True):
        packed = pack_cpu(tensor, raw_fallback=fallback)
        assert torch.equal(unpack_cpu(packed).view(torch.int16), tensor.view(torch.int16))
    torch.manual_seed(0)
    scattered = tensor[torch.randperm(tensor.numel())]
    assert pack_cpu(scattered).raw  # Uniform exponents within each block are incompressible.


@pytest.mark.parametrize("length", [1, 7, 8, 127, 1023, 1024, 1025, 3000])
def test_partial_blocks_escape_offsets_and_exact_backing(length):
    torch.manual_seed(length)
    tensor = torch.randn(length).bfloat16()
    packed = pack_cpu(tensor, raw_fallback=False)
    assert torch.equal(unpack_cpu(packed).view(torch.int16), tensor.view(torch.int16))
    assert sum(x.numel() * x.element_size() for x in packed.buffers) == packed.storage_bytes


def test_prompt_decode_preserves_history_and_original_input():
    torch.manual_seed(4)
    prompt = torch.randn(1, 2, 64, 16).bfloat16()
    before = prompt.view(torch.int16).clone()
    packed = pack_cpu(prompt)
    destination = torch.full((1, 2, 70, 16), -99, dtype=torch.bfloat16)
    unpack_into(packed, destination)
    assert torch.equal(destination[..., :64, :].view(torch.int16), before)
    assert torch.equal(prompt.view(torch.int16), before)
    assert (destination[..., 64:, :] == -99).all()


def test_compressible_storage_and_raw_fallback_never_expand():
    # Deliberately narrow exponent range in a synthetic fixture, not a model claim.
    tensor = torch.linspace(-2, 2, 32768).bfloat16()
    packed = pack_cpu(tensor)
    assert not packed.raw and packed.storage_bytes < tensor.numel() * tensor.element_size()
    assert torch.equal(unpack_cpu(packed).view(torch.int16), tensor.view(torch.int16))


def test_invalid_dtype_rejected():
    with pytest.raises(ValueError, match="BF16"):
        pack_cpu(torch.randn(32))
