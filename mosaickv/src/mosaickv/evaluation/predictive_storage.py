"""CPU feasibility only: reversible integer row predictors, never quantization."""

from dataclasses import dataclass
from typing import Any

import torch

KINDS = ("none", "xor16", "ordered_delta16")


@dataclass(frozen=True)
class SparseHighByte:
    """Eight low bits, high-byte presence bitmap, and literal nonzero highs."""

    shape: tuple[int, ...]
    low: Any
    mask: Any
    offsets: Any
    high: Any

    @property
    def storage_bytes(self) -> int:
        return sum(
            t.untyped_storage().nbytes() for t in (self.low, self.mask, self.offsets, self.high)
        )


def pack_sparse_high(tensor: Any) -> SparseHighByte:
    validate(tensor, "none")
    words = tensor.contiguous().view(torch.int16).flatten().to(torch.int32) & 65535
    if words.numel() >= 2**31:
        raise ValueError("int32-addressable sparse packet required")
    high = (words >> 8).to(torch.uint8)
    present = high != 0
    flags = torch.nn.functional.pad(present.to(torch.int32), (0, (-words.numel()) % 8)).reshape(
        -1, 8
    )
    mask = (flags << torch.arange(8)).sum(-1).to(torch.uint8)
    block_ids = torch.arange(words.numel()) // 1024
    counts = torch.bincount(block_ids[present], minlength=(words.numel() + 1023) // 1024)
    offsets = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    return SparseHighByte(
        tuple(tensor.shape), (words & 255).to(torch.uint8), mask, offsets, high[present]
    )


def unpack_sparse_high(packet: SparseHighByte) -> Any:
    count = packet.low.numel()
    flags = ((packet.mask.to(torch.int32)[:, None] >> torch.arange(8)) & 1).flatten()[:count].bool()
    if (
        count < 1
        or packet.offsets.numel() != (count + 1023) // 1024 + 1
        or int(flags.sum()) != packet.high.numel()
    ):
        raise ValueError("invalid sparse high-byte accounting")
    block_ids = torch.arange(count) // 1024
    counts = torch.bincount(block_ids[flags], minlength=(count + 1023) // 1024)
    expected = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    if not torch.equal(expected, packet.offsets):
        raise ValueError("invalid sparse high-byte offsets")
    high = torch.zeros(count, dtype=torch.int32)
    high[flags] = packet.high.to(torch.int32)
    words = (high << 8) | packet.low.to(torch.int32)
    return words.to(torch.int16).view(torch.bfloat16).reshape(packet.shape)


def validate(tensor: Any, kind: str) -> None:
    if (
        kind not in KINDS
        or tensor.dtype != torch.bfloat16
        or tensor.device.type != "cpu"
        or tensor.ndim != 4
        or tensor.shape[0] != 1
        or not all(tensor.shape)
    ):
        raise ValueError(
            "registered predictor and nonempty CPU batch-one BF16 BHSD tensor required"
        )


def encode_prediction(tensor: Any, kind: str) -> Any:
    validate(tensor, kind)
    if kind == "none":
        return tensor.clone()
    bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 65535
    if kind == "ordered_delta16":
        ordered = torch.where((bits & 32768) != 0, (~bits) & 65535, bits ^ 32768)
        delta = (ordered[..., 1:, :] - ordered[..., :-1, :]) & 65535
        signed = torch.where(delta >= 32768, delta - 65536, delta)
        result = ordered.clone()
        result[..., 1:, :] = ((signed << 1) ^ (signed >> 15)) & 65535
        result[..., ::16, :] = ordered[..., ::16, :]
    else:
        result = bits.clone()
        result[..., 1:, :] = bits[..., 1:, :] ^ bits[..., :-1, :]
        result[..., ::16, :] = bits[..., ::16, :]
    return result.to(torch.int16).view(torch.bfloat16)


def decode_prediction(tensor: Any, kind: str) -> Any:
    validate(tensor, kind)
    if kind == "none":
        return tensor.clone()
    bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 65535
    length = bits.shape[-2]
    if kind == "ordered_delta16":
        delta = (bits >> 1) ^ (-(bits & 1))
        delta[..., ::16, :] = bits[..., ::16, :]
        grouped = torch.nn.functional.pad(delta, (0, 0, 0, (-length) % 16)).reshape(
            *bits.shape[:2], -1, 16, bits.shape[-1]
        )
        ordered = (grouped.cumsum(-2, dtype=torch.int32) & 65535).reshape(
            *bits.shape[:2], -1, bits.shape[-1]
        )[..., :length, :]
        result = torch.where(ordered >= 32768, ordered ^ 32768, (~ordered) & 65535)
    else:
        grouped = torch.nn.functional.pad(bits, (0, 0, 0, (-length) % 16)).reshape(
            *bits.shape[:2], -1, 16, bits.shape[-1]
        )
        for shift in (1, 2, 4, 8):
            grouped[..., shift:, :] = grouped[..., shift:, :] ^ grouped[..., :-shift, :].clone()
        result = grouped.reshape(*bits.shape[:2], -1, bits.shape[-1])[..., :length, :]
    return result.to(torch.int16).view(torch.bfloat16).contiguous()
