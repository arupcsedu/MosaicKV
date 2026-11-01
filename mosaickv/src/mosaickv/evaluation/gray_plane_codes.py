"""CPU-only exact recoding of existing THREE-bit quantizer indices.

No re-quantization, changed scale, prompt eviction or value fitting. This is a
storage-feasibility codec, NOT an implemented GPU cache format. Explicit headers,
uint32 offsets and sparse indices count in serialized bytes. Raw fallback never
inflates the original code payload. Offer it to every selector, not EviCastKV only.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneCodes:
    raw: bool
    value_count: int
    headers: np.ndarray
    offsets: np.ndarray
    payload: np.ndarray

    @property
    def serialized_bytes(self):
        return sum(a.nbytes for a in (self.headers, self.offsets, self.payload))


def unpack_indices(codes):
    codes = np.asarray(codes)
    if codes.dtype != np.uint8 or codes.ndim != 1 or not codes.size or codes.size % 3:
        raise ValueError("flat nonempty original three-bit byte payload required")
    data = codes.reshape(-1, 3).astype(np.uint32)
    words = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
    return ((words[:, None] >> (np.arange(8, dtype=np.uint32) * 3)) & 7).astype(np.uint8).flatten()


def pack_indices(indices):
    q = np.asarray(indices)
    if q.dtype != np.uint8 or q.ndim != 1 or not q.size or q.size % 8 or np.any(q > 7):
        raise ValueError("eight-aligned uint8 original indices in [0,7] required")
    words = np.sum(
        q.reshape(-1, 8).astype(np.uint32) << (np.arange(8, dtype=np.uint32) * 3),
        axis=1,
        dtype=np.uint32,
    )
    return np.stack(
        tuple(((words >> (8 * i)) & 255).astype(np.uint8) for i in range(3)), -1
    ).flatten()


def encode(codes):
    original = np.asarray(codes)
    q = unpack_indices(original)
    if q.size % 128:
        raise ValueError("complete 128-index groups required; padding is not silently omitted")
    gray = (q ^ (q >> 1)).reshape(-1, 128)
    groups = gray.shape[0]
    headers = np.zeros(groups, dtype=np.uint8)
    offsets, pieces, position = [0], [], 0
    planes = [((gray >> b) & 1).astype(np.uint8) for b in range(3)]
    counts = [p.sum(1, dtype=np.uint16) for p in planes]
    raw = [np.packbits(p, axis=1, bitorder="little") for p in planes]
    for group in range(groups):
        for bit in range(3):
            ones = int(counts[bit][group])
            if ones == 0:
                mode, body = 1, np.empty(0, dtype=np.uint8)
            elif ones == 128:
                mode, body = 2, np.empty(0, dtype=np.uint8)
            elif min(ones, 128 - ones) + 1 < 16:
                mode, dominant = 3, int(ones > 64)
                exceptions = np.flatnonzero(planes[bit][group] != dominant).astype(np.uint8)
                body = np.concatenate(
                    (np.asarray([(dominant << 7) | len(exceptions)], dtype=np.uint8), exceptions)
                )
            else:
                mode, body = 0, raw[bit][group]
            headers[group] |= mode << (2 * bit)
            pieces.append(body)
            position += body.size
        offsets.append(position)
    if position >= 2**32:
        raise ValueError("codec offset range exceeds uint32")
    packet = PlaneCodes(
        False, q.size, headers, np.asarray(offsets, dtype=np.uint32), np.concatenate(pieces)
    )
    if packet.serialized_bytes >= original.nbytes:
        return PlaneCodes(
            True, q.size, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint32), original.copy()
        )
    return packet


def decode(packet):
    if (
        not isinstance(packet, PlaneCodes)
        or type(packet.raw) is not bool
        or packet.value_count < 128
        or packet.value_count % 128
    ):
        raise ValueError("explicit complete-group plane packet required")
    if packet.raw:
        if (
            packet.headers.size
            or packet.offsets.size
            or packet.payload.dtype != np.uint8
            or packet.payload.size * 8 != packet.value_count * 3
        ):
            raise ValueError("raw fallback has unexpected metadata or bytes")
        return packet.payload.copy()
    groups = packet.value_count // 128
    if (
        packet.headers.dtype != np.uint8
        or packet.headers.shape != (groups,)
        or packet.offsets.dtype != np.uint32
        or packet.offsets.shape != (groups + 1,)
        or packet.payload.dtype != np.uint8
        or packet.payload.ndim != 1
        or packet.offsets[0] != 0
        or packet.offsets[-1] != packet.payload.size
        or np.any(packet.offsets[1:] < packet.offsets[:-1])
        or np.any(packet.headers >> 6)
    ):
        raise ValueError("invalid charged header/offset/payload layout")
    gray = np.zeros((groups, 128), dtype=np.uint8)
    for group in range(groups):
        cursor, end = int(packet.offsets[group]), int(packet.offsets[group + 1])
        for bit in range(3):
            mode = (int(packet.headers[group]) >> (2 * bit)) & 3
            if mode == 0:
                if cursor + 16 > end:
                    raise ValueError("truncated raw plane")
                plane = np.unpackbits(packet.payload[cursor : cursor + 16], bitorder="little")
                cursor += 16
            elif mode in (1, 2):
                plane = np.full(128, mode - 1, dtype=np.uint8)
            else:
                if cursor >= end:
                    raise ValueError("missing sparse plane control")
                control = int(packet.payload[cursor])
                count, dominant = control & 127, control >> 7
                cursor += 1
                if not 1 <= count <= 14 or cursor + count > end:
                    raise ValueError("invalid sparse exception count")
                ids = packet.payload[cursor : cursor + count]
                if np.any(ids >= 128) or np.any(ids[1:] <= ids[:-1]):
                    raise ValueError("sparse exception indices must be unique and ordered")
                plane = np.full(128, dominant, dtype=np.uint8)
                plane[ids] = 1 - dominant
                cursor += count
            gray[group] |= plane << bit
        if cursor != end:
            raise ValueError("unexpected trailing plane bytes")
    binary = gray ^ (gray >> 1) ^ (gray >> 2)
    return pack_indices(binary.flatten())
