"""Experimental, lossless BF16 storage. No eviction, quantization or fitting.

One sign/mantissa byte plus a three-bit exponent code, a per-1024-value
exponent base and literal escape exponents. Every bit, including NaN payloads,
is reconstructed. Raw fallback prevents expansion for incompressible tensors.
This is a storage feasibility component, not an activated production method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BLOCK = 1024


@dataclass(frozen=True)
class PackedBF16:
    shape: tuple[int, ...]
    raw: bool
    buffers: tuple[Any, ...]

    @property
    def storage_bytes(self) -> int:
        return sum(int(t.untyped_storage().nbytes()) for t in self.buffers)

    def to(self, device: Any) -> PackedBF16:
        return PackedBF16(self.shape, self.raw, tuple(t.to(device) for t in self.buffers))


def pack_cpu(tensor: Any, *, raw_fallback: bool = True) -> PackedBF16:
    """Deterministic CPU constructor; no labels/queries or output-based tuning."""
    import torch

    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu" or tensor.numel() == 0:
        raise ValueError("nonempty CPU BF16 tensor required")
    shape, count = tuple(tensor.shape), tensor.numel()
    words = tensor.contiguous().view(torch.int16).reshape(-1).to(torch.int32) & 65535
    exponents = (words >> 7) & 255
    blocks = (count + BLOCK - 1) // BLOCK
    block_ids = torch.arange(count, dtype=torch.int64) // BLOCK
    histogram = torch.bincount(block_ids * 256 + exponents, minlength=blocks * 256)
    histogram = histogram.reshape(blocks, 256)
    prefix = torch.nn.functional.pad(histogram.cumsum(-1), (1, 0))
    # Seven ordinary exponent values; code 7 means a literal escape. Ties
    # choose the lowest base, independently of model outputs.
    bases = (prefix[:, 7:] - prefix[:, :-7]).argmax(-1)
    difference = exponents - bases[block_ids]
    escaped = (difference < 0) | (difference > 6)
    codes = torch.where(escaped, 7, difference).to(torch.int32)
    codes = torch.nn.functional.pad(codes, (0, blocks * BLOCK - count)).reshape(-1, 8)
    packed_words = (codes << (3 * torch.arange(8, dtype=torch.int32))).sum(-1, dtype=torch.int32)
    packed_codes = (
        torch.stack((packed_words & 255, (packed_words >> 8) & 255, (packed_words >> 16) & 255), -1)
        .to(torch.uint8)
        .reshape(-1)
    )
    counts = torch.bincount(block_ids[escaped], minlength=blocks).to(torch.int32)
    offsets = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    sign_mantissa = ((words & 127) | ((words >> 8) & 128)).to(torch.uint8)
    buffers = (
        sign_mantissa,
        packed_codes,
        bases.to(torch.uint8),
        offsets,
        exponents[escaped].to(torch.uint8),
    )
    result = PackedBF16(shape, False, buffers)
    if raw_fallback and result.storage_bytes >= tensor.numel() * tensor.element_size():
        return PackedBF16(shape, True, (tensor.contiguous().clone(),))
    return result


def unpack_cpu(packed: PackedBF16) -> Any:
    import torch

    if any(t.device.type != "cpu" for t in packed.buffers):
        raise ValueError("CPU buffers required")
    if packed.raw:
        return packed.buffers[0].clone()
    sm, byte_codes, bases, offsets, literals = packed.buffers
    byte_codes = byte_codes.reshape(-1, 3).to(torch.int32)
    words = byte_codes[:, 0] | (byte_codes[:, 1] << 8) | (byte_codes[:, 2] << 16)
    codes = ((words[:, None] >> (3 * torch.arange(8, dtype=torch.int32))) & 7).reshape(-1)[
        : sm.numel()
    ]
    exponents = bases.repeat_interleave(BLOCK)[: sm.numel()].to(torch.int32) + codes
    escaped = codes == 7
    if int(escaped.sum()) != literals.numel() or int(offsets[-1]) != literals.numel():
        raise ValueError("invalid literal-exponent accounting")
    exponents[escaped] = literals.to(torch.int32)
    result = (sm.to(torch.int32) & 127) | ((sm.to(torch.int32) & 128) << 8) | (exponents << 7)
    return result.to(torch.int16).view(torch.bfloat16).reshape(packed.shape)


def unpack_into(packed: PackedBF16, destination: Any) -> None:
    """Decode directly into a BH(P+history)D prompt prefix; no full-cache decode."""
    import torch

    if (
        len(packed.shape) != 4
        or packed.shape[0] != 1
        or destination.ndim != 4
        or destination.dtype != torch.bfloat16
        or not destination.is_contiguous()
        or tuple(destination.shape[:2]) != packed.shape[:2]
        or destination.shape[-1] != packed.shape[-1]
        or destination.shape[-2] < packed.shape[-2]
    ):
        raise ValueError("matching contiguous batch-one BHSD destination required")
    if packed.raw:
        destination[..., : packed.shape[-2], :].copy_(packed.buffers[0])
    elif destination.device.type == "cpu":
        destination[..., : packed.shape[-2], :].copy_(unpack_cpu(packed))
    else:
        if any(t.device != destination.device for t in packed.buffers):
            raise ValueError("packed buffers and destination must share one device")
        from mosaickv.evaluation.lossless_bf16_kernel import decode

        decode(packed, destination)


def unpack_pair_into(prompt: Any, history: Any, destination: Any) -> None:
    """Validate a complete pair transaction before fused CUDA or reference CPU copy."""
    import torch

    if len(prompt) != 2 or len(history) != 2 or len(destination) != 2:
        raise ValueError("exactly two prompt, history and destination components required")
    shape = prompt[0].shape
    if len(shape) != 4 or shape[0] != 1 or prompt[1].shape != shape:
        raise ValueError("same-shape batch-one BHSD prompts required")
    device = destination[0].device
    length = destination[0].shape[-2]
    used = length - shape[-2]
    if used <= 0:
        raise ValueError("nonempty generated suffix required")
    for body, suffix, target in zip(prompt, history, destination, strict=True):
        if (
            target.shape != destination[0].shape
            or target.shape[:2] != shape[:2]
            or target.shape[-1] != shape[-1]
            or suffix.ndim != 4
            or suffix.shape[:2] != shape[:2]
            or suffix.shape[-1] != shape[-1]
            or suffix.shape[-2] < used
            or suffix.shape != history[0].shape
            or any(t.device != device for t in (*body.buffers, suffix, target))
            or any(t.dtype != torch.bfloat16 for t in (suffix, target))
            or not suffix.is_contiguous()
            or not target.is_contiguous()
        ):
            raise ValueError("compatible contiguous same-device BHSD pair required")
    if device.type == "cuda":
        from mosaickv.evaluation.lossless_bf16_kernel import decode_pair

        decode_pair(prompt, history, destination)
    else:
        for body, suffix, target in zip(prompt, history, destination, strict=True):
            unpack_into(body, target)
            target[..., shape[-2] :, :].copy_(suffix[..., :used, :])
