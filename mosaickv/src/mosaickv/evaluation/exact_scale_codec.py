"""Lossless FP32 scale-bit packing. No cast/quantization of scale values.

Choose the complete smallest raw, full-word dictionary, or split-high16 plus
low16 dictionary packet. CPU encoding is a feasibility/reference implementation;
its online synchronization cost is not hidden by GPU-only decoder timings.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch


def pack_indices(values, bits):
    values = np.asarray(values, dtype=np.uint32)
    if type(bits) is not int or not 1 <= bits <= 16 or np.any(values >= 1 << bits):
        raise ValueError("bounded exact dictionary indices required")
    offset = np.arange(values.size, dtype=np.int64) * bits
    output = np.zeros((values.size * bits + 7) // 8 + 2, dtype=np.uint8)
    shifted = values << (offset % 8).astype(np.uint32)
    for byte in range(3):
        np.bitwise_or.at(
            output, offset // 8 + byte, ((shifted >> (byte * 8)) & 255).astype(np.uint8)
        )
    return torch.from_numpy(output[:-2].copy())


def unpack_indices(payload, count, bits):
    if (
        payload.device.type != "cpu"
        or payload.dtype != torch.uint8
        or payload.numel() != (count * bits + 7) // 8
    ):
        raise ValueError("complete CPU packed index stream required")
    source = np.pad(payload.numpy().astype(np.uint32), (0, 2))
    offset = np.arange(count, dtype=np.int64) * bits
    word = sum(source[offset // 8 + byte] << (byte * 8) for byte in range(3))
    return ((word >> (offset % 8).astype(np.uint32)) & ((1 << bits) - 1)).astype(np.int64)


@dataclass(frozen=True)
class ScalePacket:
    shape: tuple
    kind: str
    bits: int
    payload: object
    palette: object = None
    high: object = None
    selector: object = None

    def tensors(self):
        return tuple(
            t for t in (self.payload, self.palette, self.high, self.selector) if t is not None
        )

    @property
    def storage_bytes(self):
        return sum(t.untyped_storage().nbytes() for t in self.tensors())

    def to(self, device):
        return replace(
            self,
            **{
                name: value.to(device)
                for name in ("payload", "palette", "high", "selector")
                if (value := getattr(self, name)) is not None
            },
        )


def encode_cpu(scale):
    if (
        scale.device.type != "cpu"
        or scale.dtype != torch.float32
        or not scale.numel()
        or not bool((torch.isfinite(scale) & (scale > 0)).all())
    ):
        raise ValueError("nonempty finite positive CPU FP32 scales required")
    words = scale.contiguous().view(torch.int32).flatten().numpy().view(np.uint32)
    raw = ScalePacket(tuple(scale.shape), "raw", 0, scale.contiguous().clone())
    candidates = [raw]
    for kind, source in (("word", words), ("low16", words & 65535)):
        unique, inverse = np.unique(source, return_inverse=True)
        bits = max(1, (len(unique) - 1).bit_length())
        if bits > 16:
            continue
        palette = torch.from_numpy(unique.copy().view(np.int32))
        if kind == "low16":
            # Two-byte palette words, not a four-byte charge hidden by dtype.
            palette = torch.from_numpy(unique.astype(np.uint16))
        candidates.append(
            ScalePacket(
                tuple(scale.shape),
                kind,
                bits,
                pack_indices(inverse, bits),
                palette,
                torch.from_numpy((words >> 16).astype(np.uint16)) if kind == "low16" else None,
                torch.tensor([1 if kind == "word" else 2, bits], dtype=torch.uint8),
            )
        )
    chosen = min(enumerate(candidates), key=lambda p: (p[1].storage_bytes, p[0]))[1]
    if not torch.equal(decode_cpu(chosen).view(torch.int32), scale.view(torch.int32)):
        raise RuntimeError("scale codec changed an original FP32 bit")
    return chosen


def decode_cpu(packet):
    if any(t.device.type != "cpu" for t in packet.tensors()):
        raise ValueError("CPU scale oracle only")
    if packet.kind == "raw":
        return packet.payload.clone()
    count = int(np.prod(packet.shape))
    indices = unpack_indices(packet.payload, count, packet.bits)
    if packet.kind == "word":
        words = packet.palette.numpy()[indices].view(np.uint32)
    elif packet.kind == "low16":
        words = (packet.high.numpy().astype(np.uint32) << 16) | packet.palette.numpy()[
            indices
        ].astype(np.uint32)
    else:
        raise ValueError("registered scale codec required")
    return torch.from_numpy(words.copy().view(np.float32)).reshape(packet.shape)


def unpack_into(packet, output):
    if (
        output.dtype != torch.float32
        or not output.is_contiguous()
        or output.numel() < int(np.prod(packet.shape))
        or output.device != packet.payload.device
    ):
        raise ValueError("same-device contiguous sufficient FP32 scale workspace required")
    if output.device.type == "cpu":
        output.flatten()[: int(np.prod(packet.shape))].copy_(decode_cpu(packet).flatten())
    elif packet.kind == "raw":
        output.flatten()[: packet.payload.numel()].copy_(packet.payload.flatten())
    else:
        from mosaickv.evaluation.exact_scale_kernel import unpack

        count = int(np.prod(packet.shape))
        compiled = unpack[((count + 255) // 256,)](
            packet.payload,
            packet.palette,
            packet.high if packet.high is not None else packet.palette,
            output,
            count,
            packet.payload.numel(),
            BITS=packet.bits,
            SPLIT=packet.kind == "low16",
            BLOCK=256,
        )
        if compiled.metadata.global_scratch_size or compiled.n_spills:
            raise RuntimeError("scale decoder introduced undeclared scratch or spills")
