"""Lossless CUDA serialization of existing quantizer bytes, never quantization.

All offsets, payloads and replay scratch are GPU tensors. Encoding synchronizes
only the final allocation length. No code bytes or K/V move to CPU. Kernels are
imported lazily so CPU-only configuration and tests remain usable.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceCodes:
    length: int
    payload: torch.Tensor
    offsets: torch.Tensor | None = None

    @property
    def raw(self):
        return self.offsets is None

    def tensors(self):
        return (self.payload,) if self.raw else (self.payload, self.offsets)

    @property
    def serialized_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())


def validate_bytes(value):
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        or not value.numel()
        or value.numel() % 128
        or not value.is_contiguous()
        or value.device.type != "cuda"
        or value.numel() >= 2**31
    ):
        raise ValueError("nonempty contiguous CUDA uint8 complete 128-byte blocks required")
    return value


def encode(previous, current):
    validate_bytes(current)
    if previous is None:
        return DeviceCodes(current.numel(), current)
    validate_bytes(previous)
    if previous.device != current.device:
        raise ValueError("same-device predecessor required")
    if previous.numel() != current.numel():
        return DeviceCodes(current.numel(), current)
    from mosaickv.evaluation.lossless_code_delta_kernel import count, pack

    groups = current.numel() // 128
    lengths = torch.empty(groups, dtype=torch.int32, device=current.device)
    count[(groups,)](previous, current, lengths)
    ends = torch.cumsum(lengths, 0, dtype=torch.int32)
    total = int(ends[-1].item())
    if total + 4 * (groups + 1) >= current.numel():
        return DeviceCodes(current.numel(), current)
    payload = torch.empty(total, dtype=torch.uint8, device=current.device)
    offsets = torch.empty(groups + 1, dtype=torch.int32, device=current.device)
    pack[(groups,)](previous, current, lengths, ends, payload, offsets, GROUPS=groups)
    return DeviceCodes(current.numel(), payload, offsets)


def decode(previous, packet, destination=None):
    if not isinstance(packet, DeviceCodes):
        raise ValueError("explicit GPU byte packet required")
    if packet.raw:
        validate_bytes(packet.payload)
        if packet.payload.numel() != packet.length:
            raise ValueError("raw packet length differs")
        if destination is None:
            return packet.payload
        validate_bytes(destination)
        if destination.shape != packet.payload.shape or destination.device != packet.payload.device:
            raise ValueError("complete same-device destination required")
        destination.copy_(packet.payload)
        return destination
    validate_bytes(previous)
    if previous.numel() != packet.length or previous.device != packet.payload.device:
        raise ValueError("complete same-device predecessor required")
    if (
        packet.offsets.dtype != torch.int32
        or packet.offsets.shape != (packet.length // 128 + 1,)
        or packet.offsets.device != previous.device
        or packet.payload.dtype != torch.uint8
        or packet.payload.ndim != 1
        or not packet.offsets.is_contiguous()
        or not packet.payload.is_contiguous()
    ):
        raise ValueError("encoder-produced GPU offsets and payload required")
    output = torch.empty_like(previous) if destination is None else validate_bytes(destination)
    if output.shape != previous.shape or output.device != previous.device:
        raise ValueError("complete same-device destination required")
    if any(
        output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
        for t in packet.tensors()
    ):
        raise ValueError("decode destination must not alias serialized packet")
    # In-place predecessor replay is safe: every program reads and writes only
    # its own 128-byte block, and each lane owns one distinct result byte.
    from mosaickv.evaluation.lossless_code_delta_kernel import unpack

    unpack[(packet.length // 128,)](previous, packet.payload, packet.offsets, output)
    return output
