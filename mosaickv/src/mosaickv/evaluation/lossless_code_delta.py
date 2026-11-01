"""CPU reference for exact layer-to-layer code-byte compression, not GPU support.

128-byte blocks use either raw CURRENT bytes or a 16-byte nonzero-XOR bitmap
and literal deltas. A charged uint32 offset carries its raw-block flag. Packet-
level raw fallback avoids inflation; different lengths require raw fallback.
No quantizer, scale, index, K/V value, selection or model weight changes.
"""

from dataclasses import dataclass

import numpy as np

RAW = 1 << 31
MASK = RAW - 1


@dataclass(frozen=True)
class DeltaCodes:
    raw: bool
    length: int
    offsets: np.ndarray
    payload: np.ndarray

    @property
    def serialized_bytes(self):
        return self.offsets.nbytes + self.payload.nbytes


def validate_bytes(value):
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1 or not array.size or array.size % 128:
        raise ValueError("nonempty uint8 bytes in complete 128-byte groups required")
    return array


def encode(previous, current):
    current = validate_bytes(current)
    if previous is None:
        return DeltaCodes(True, current.size, np.empty(0, dtype=np.uint32), current.copy())
    previous = validate_bytes(previous)
    if previous.size != current.size:
        return DeltaCodes(True, current.size, np.empty(0, dtype=np.uint32), current.copy())
    a, b = previous.reshape(-1, 128), current.reshape(-1, 128)
    delta = np.bitwise_xor(a, b)
    nonzero = delta != 0
    count = nonzero.sum(1)
    masks = np.packbits(nonzero, axis=1, bitorder="little")
    pieces, offsets, cursor = [], [], 0
    for group in range(delta.shape[0]):
        compressed = count[group] + 16 < 128
        offsets.append(cursor | (0 if compressed else RAW))
        body = (
            np.concatenate((masks[group], delta[group, nonzero[group]])) if compressed else b[group]
        )
        pieces.append(body)
        cursor += body.size
    if cursor >= RAW:
        raise ValueError("delta stream exceeds its explicitly charged 31-bit address space")
    offsets.append(cursor)
    packet = DeltaCodes(
        False, current.size, np.asarray(offsets, dtype=np.uint32), np.concatenate(pieces)
    )
    if packet.serialized_bytes >= current.nbytes:
        return DeltaCodes(True, current.size, np.empty(0, dtype=np.uint32), current.copy())
    return packet


def decode(previous, packet):
    if (
        not isinstance(packet, DeltaCodes)
        or type(packet.raw) is not bool
        or type(packet.length) is not int
        or packet.length < 128
        or packet.length % 128
    ):
        raise ValueError("explicit complete-group delta packet required")
    if (
        packet.payload.dtype != np.uint8
        or packet.payload.ndim != 1
        or packet.offsets.dtype != np.uint32
    ):
        raise ValueError("exact byte payload and uint32 offsets required")
    if packet.raw:
        if packet.offsets.size or packet.payload.size != packet.length:
            raise ValueError("raw fallback has unexpected offsets or length")
        return packet.payload.copy()
    previous = validate_bytes(previous)
    groups = packet.length // 128
    offsets = packet.offsets & MASK
    if (
        previous.size != packet.length
        or packet.offsets.shape != (groups + 1,)
        or offsets[0] != 0
        or offsets[-1] != packet.payload.size
        or packet.offsets[-1] & RAW
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("invalid predecessor or charged offset layout")
    result = np.empty((groups, 128), dtype=np.uint8)
    old = previous.reshape(groups, 128)
    for group in range(groups):
        start, end = int(offsets[group]), int(offsets[group + 1])
        if packet.offsets[group] & RAW:
            if end - start != 128:
                raise ValueError("raw CURRENT block must contain exactly 128 bytes")
            result[group] = packet.payload[start:end]
        else:
            if end - start < 16:
                raise ValueError("truncated XOR bitmap")
            flags = np.unpackbits(packet.payload[start : start + 16], bitorder="little").astype(
                bool
            )
            if end - start != 16 + int(flags.sum()):
                raise ValueError("XOR bitmap and literal counts differ")
            delta = np.zeros(128, dtype=np.uint8)
            literals = packet.payload[start + 16 : end]
            if np.any(literals == 0):
                raise ValueError("noncanonical zero literal in nonzero XOR stream")
            delta[flags] = literals
            result[group] = np.bitwise_xor(old[group], delta)
    return result.flatten()
