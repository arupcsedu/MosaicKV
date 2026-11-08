"""Reversible three-bit Gray delta coding, CPU feasibility only initially.

Original quantizer symbols/precision and every logical position are unchanged.
The wider 384-byte groups are aligned to three-bit packing and 128-byte source
blocks. A native decoder can invert each group without a cross-group halo or
another decoded-source buffer. That is a design constraint, not GPU evidence.
"""

from dataclasses import dataclass

import numpy as np

from mosaickv.evaluation import all_head_source_codec as symbol_codec
from mosaickv.evaluation import lossless_code_delta as byte_codec
from mosaickv.evaluation.source_dictionary import common_encode

RAW = np.uint32(1 << 31)
MASK = np.uint32((1 << 31) - 1)
GROUP = 384
BITMAP = GROUP // 8


def _validate(value):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 1 or not value.size or value.size % 3:
        raise ValueError("complete contiguous three-bit source words required")
    if not value.flags.c_contiguous:
        raise ValueError("contiguous original source bytes required")
    return value


def transform(value, *, inverse=False):
    """Map every original 3-bit symbol independently, not entire bytes."""
    value = _validate(value)
    following = np.empty_like(value)
    following[:-1], following[-1] = value[1:], 0
    low_two = np.resize(np.array([0xDB, 0xB6, 0x6D], dtype=np.uint8), value.size)
    result = value ^ (((value >> 1) | (following << 7)) & low_two)
    if inverse:
        low_one = np.resize(np.array([0x49, 0x92, 0x24], dtype=np.uint8), value.size)
        result ^= ((value >> 2) | (following << 6)) & low_one
    return result


@dataclass(frozen=True)
class GrayCodes:
    length: int
    offsets: np.ndarray
    payload: np.ndarray

    @property
    def serialized_bytes(self):
        return self.offsets.nbytes + self.payload.nbytes


def encode(previous, current, *, gray=True):
    """Keep the complete proposal visible; selection is separate and strict."""
    current = _validate(current)
    if current.size % GROUP or type(gray) is not bool:
        raise ValueError("whole 384-byte groups and explicit Gray activation required")
    if previous is None:
        return byte_codec.DeltaCodes(
            True, current.size, np.empty(0, dtype=np.uint32), current.copy()
        )
    previous = _validate(previous)
    if previous.shape != current.shape:
        raise ValueError("same original predecessor source layout required")
    delta = previous ^ current
    if gray:
        delta = transform(delta)
    values = delta.reshape(-1, GROUP)
    nonzero = values != 0
    counts = nonzero.sum(1)
    bitmaps = np.packbits(nonzero, axis=1, bitorder="little")
    parts, offsets, cursor = [], [], 0
    for i in range(len(values)):
        raw = BITMAP + counts[i] >= GROUP
        offsets.append(cursor | (int(RAW) if raw else 0))
        body = (
            current.reshape(-1, GROUP)[i].copy()
            if raw
            else np.concatenate((bitmaps[i], values[i][nonzero[i]]))
        )
        parts.append(body)
        cursor += len(body)
    if cursor >= int(RAW):
        raise ValueError("source payload exceeds charged 31-bit offset space")
    offsets.append(cursor)
    return GrayCodes(current.size, np.array(offsets, dtype=np.uint32), np.concatenate(parts))


def decode(previous, packet, *, gray=True):
    if isinstance(packet, symbol_codec.SymbolCodes):
        return symbol_codec.decode(previous, packet)
    if isinstance(packet, byte_codec.DeltaCodes):
        return byte_codec.decode(previous, packet)
    if not isinstance(packet, GrayCodes) or type(gray) is not bool:
        raise ValueError("explicit Gray source descriptor and mode required")
    if (
        type(packet.length) is not int
        or packet.length <= 0
        or packet.length % GROUP
        or packet.offsets.dtype != np.uint32
        or packet.payload.dtype != np.uint8
        or packet.offsets.shape != (packet.length // GROUP + 1,)
        or packet.payload.ndim != 1
    ):
        raise ValueError("complete Gray packet geometry required")
    previous = _validate(previous)
    offsets = packet.offsets & MASK
    if (
        previous.size != packet.length
        or offsets[0] != 0
        or offsets[-1] != packet.payload.size
        or np.any(offsets[1:] < offsets[:-1])
        or packet.offsets[-1] & RAW
    ):
        raise ValueError("invalid Gray predecessor or payload offsets")
    result = np.empty_like(previous)
    for i in range(packet.length // GROUP):
        body = packet.payload[offsets[i] : offsets[i + 1]]
        begin = i * GROUP
        if packet.offsets[i] & RAW:
            if len(body) != GROUP:
                raise ValueError("raw CURRENT block must be exactly 384 bytes")
            result[begin : begin + GROUP] = body
            continue
        if len(body) < BITMAP:
            raise ValueError("truncated Gray bitmap")
        flags = np.unpackbits(body[:BITMAP], bitorder="little").astype(bool)
        literals = body[BITMAP:]
        if len(literals) != flags.sum() or np.any(literals == 0):
            raise ValueError("Gray literal count/value differs from its nonzero bitmap")
        delta = np.zeros(GROUP, dtype=np.uint8)
        delta[flags] = literals
        if gray:
            delta = transform(delta, inverse=True)
        result[begin : begin + GROUP] = previous[begin : begin + GROUP] ^ delta
    return result


def screen(previous, current):
    common = common_encode(previous, current)
    if not np.array_equal(decode(previous, common), current):
        raise RuntimeError("common source-code control changed an original bit")
    gray, wider = encode(previous, current), encode(previous, current, gray=False)
    for packet, active in ((gray, True), (wider, False)):
        if not np.array_equal(decode(previous, packet, gray=active), current):
            raise RuntimeError("reversible source format changed an original bit")
    accepted = gray.serialized_bytes < common.serialized_bytes
    return gray if accepted else common, dict(
        original_code_bits_exact=True,
        common_bytes=common.serialized_bytes + 1,
        gray384_bytes=gray.serialized_bytes + 1,
        ordinary384_bytes=wider.serialized_bytes + 1,
        selected_bytes=min(gray.serialized_bytes, common.serialized_bytes) + 1,
        gray_selected=accepted,
        charged_selector_bytes=1,
        CPU_source_stream_bytes_not_GPU_KV=True,
        model_or_latency_verified=False,
    )
