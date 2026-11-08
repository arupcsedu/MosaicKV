"""Lossless modulo-eight residuals with optional two-bit rare-symbol literals.

This is a common representation optimization, NOT a six-component algorithm.
The original two-symbol masks remain; a group uses a two-byte rare palette only
if at most four rare symbols occur and its complete body is strictly smaller.
Headers, offsets, raw groups and full-stream raw fallback all count.
"""

from dataclasses import dataclass

import numpy as np

from mosaickv.evaluation import all_head_source_codec as base


@dataclass(frozen=True)
class CompactRareCodes:
    length: int
    payload: np.ndarray
    offsets: np.ndarray
    raw: bool

    @property
    def serialized_bytes(self):
        return self.payload.nbytes + self.offsets.nbytes


def pack_small(values, bits):
    values = np.asarray(values, dtype=np.uint8)
    if bits not in (2, 3) or np.any(values >= 1 << bits):
        raise ValueError("original two/three-bit literals required")
    if not len(values):
        return np.empty(0, dtype=np.uint8)
    offsets = np.arange(len(values)) * bits
    out = np.zeros((len(values) * bits + 7) // 8 + 1, dtype=np.uint8)
    np.bitwise_or.at(
        out, offsets // 8, (values.astype(np.uint16) << (offsets % 8)).astype(np.uint8)
    )
    np.bitwise_or.at(
        out, offsets // 8 + 1, (values.astype(np.uint16) >> (8 - offsets % 8)).astype(np.uint8)
    )
    return out[:-1]


def unpack_small(payload, count, bits):
    if len(payload) != (count * bits + 7) // 8:
        raise ValueError("truncated or trailing rare literal payload")
    offsets = np.arange(count) * bits
    padded = np.pad(payload.astype(np.uint16), (0, 1))
    return (
        ((padded[offsets // 8] >> (offsets % 8)) | (padded[offsets // 8 + 1] << (8 - offsets % 8)))
        & ((1 << bits) - 1)
    ).astype(np.uint8)


def encode(previous, current):
    current = base.validate(current)
    if previous is None:
        return CompactRareCodes(len(current), current.copy(), np.empty(0, dtype=np.uint32), True)
    previous = base.validate(previous)
    if previous.shape != current.shape:
        raise ValueError("same original predecessor required")
    residual = (base.symbols(current).astype(np.int16) - base.symbols(previous)) & 7
    groups = residual.astype(np.uint8).reshape(-1, 512)
    first, second, lengths = base._plan(groups)
    parts, offsets, cursor = [], [], 0
    for i, values in enumerate(groups):
        a, b = int(first[i]), int(second[i])
        other = values != a
        rare_mask = values[other] != b
        rare = values[other][rare_mask]
        palette = np.unique(rare)
        narrow_size = 67 + (int(other.sum()) + 7) // 8 + (len(rare) * 2 + 7) // 8
        narrow = len(palette) <= 4 and narrow_size < int(lengths[i])
        size = narrow_size if narrow else int(lengths[i])
        if size >= 192:
            offsets.append(cursor | int(base.RAW))
            body = base.pack_symbols(values)
        else:
            offsets.append(cursor)
            mode = a | b << 3 | (128 if narrow else 0)
            header = np.array([mode], dtype=np.uint8)
            if narrow:
                word = sum(int(v) << (3 * j) for j, v in enumerate(palette))
                header = np.array([mode, word & 255, word >> 8], dtype=np.uint8)
                literals = np.searchsorted(palette, rare).astype(np.uint8)
            else:
                literals = rare
            body = np.concatenate(
                (
                    header,
                    np.packbits(other, bitorder="little"),
                    np.packbits(rare_mask, bitorder="little"),
                    pack_small(literals, 2 if narrow else 3),
                )
            )
            if len(body) != size:
                raise RuntimeError("actual compact group differs from byte plan")
        parts.append(body)
        cursor += len(body)
    offsets.append(cursor)
    payload = np.concatenate(parts)
    offsets = np.array(offsets, dtype=np.uint32)
    if payload.nbytes + offsets.nbytes >= current.nbytes:
        return CompactRareCodes(len(current), current.copy(), np.empty(0, dtype=np.uint32), True)
    return CompactRareCodes(len(current), payload, offsets, False)


def decode(previous, packet):
    if (
        not isinstance(packet, CompactRareCodes)
        or type(packet.raw) is not bool
        or type(packet.length) is not int
        or packet.length <= 0
        or packet.length % 192
        or packet.payload.dtype != np.uint8
        or packet.payload.ndim != 1
        or packet.offsets.dtype != np.uint32
        or packet.offsets.ndim != 1
    ):
        raise ValueError("complete compact-rare descriptor required")
    if packet.raw:
        if packet.offsets.size or len(packet.payload) != packet.length:
            raise ValueError("invalid complete raw fallback")
        return packet.payload.copy()
    previous = base.validate(previous)
    if len(previous) != packet.length or len(packet.offsets) != packet.length // 192 + 1:
        raise ValueError("complete predecessor and group offsets required")
    starts = packet.offsets & base.OFFSET
    if starts[0] != 0 or starts[-1] != len(packet.payload) or np.any(starts[1:] < starts[:-1]):
        raise ValueError("invalid compact group offsets")
    out = []
    for group in range(len(starts) - 1):
        raw = bool(packet.offsets[group] & base.RAW)
        body = packet.payload[int(starts[group]) : int(starts[group + 1])]
        if raw:
            if len(body) != 192:
                raise ValueError("invalid raw residual group")
            out.append(base.symbols(body))
            continue
        if len(body) < 65 or int(body[0]) & 64:
            raise ValueError("invalid additive group header")
        mode, narrow = int(body[0]), bool(int(body[0]) & 128)
        header = 3 if narrow else 1
        if len(body) < header + 64:
            raise ValueError("truncated compact palette/mask")
        a, b = mode & 7, mode >> 3 & 7
        if a == b:
            raise ValueError("two distinct primary symbols required")
        other = np.unpackbits(body[header : header + 64], bitorder="little").astype(bool)
        count = int(other.sum())
        start = header + 64
        mask_size = (count + 7) // 8
        if len(body) < start + mask_size:
            raise ValueError("truncated rare mask")
        rare_mask = np.unpackbits(body[start : start + mask_size], bitorder="little")[
            :count
        ].astype(bool)
        literals = unpack_small(body[start + mask_size :], int(rare_mask.sum()), 2 if narrow else 3)
        if narrow:
            word = int(body[1]) | int(body[2]) << 8
            if word >> 12:
                raise ValueError("palette exceeds original three-bit alphabet")
            palette = np.array([(word >> (3 * j)) & 7 for j in range(4)], dtype=np.uint8)
            literals = palette[literals]
        if np.any((literals == a) | (literals == b)):
            raise ValueError("rare literals conflict with primary symbols")
        values = np.full(512, a, dtype=np.uint8)
        expanded = np.full(count, b, dtype=np.uint8)
        expanded[rare_mask] = literals
        values[other] = expanded
        out.append(values)
    residual = np.concatenate(out)
    return base.pack_symbols((residual + base.symbols(previous)) & 7)
