"""Native integer-only source palette/residual codec. No attention changes."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _symbols(pointer, group, lane):
    bit = (group * 512 + lane) * 3
    shift = bit % 8
    first = tl.load(pointer + bit // 8).to(tl.int32)
    second = tl.load(pointer + bit // 8 + 1, shift > 5, other=0).to(tl.int32)
    return ((first >> shift) | (second << (8 - shift))) & 7


@triton.jit
def _analyze(value):
    first, second, best, next_best = 0, 1, -1, -1
    for symbol in tl.static_range(8):
        count = tl.sum((value == symbol).to(tl.int32), 0)
        replace = count > best
        second = tl.where(replace, first, tl.where(count > next_best, symbol, second))
        next_best = tl.where(replace, best, tl.maximum(next_best, count))
        first = tl.where(replace, symbol, first)
        best = tl.maximum(best, count)
    other = value != first
    rare = other & (value != second)
    count = tl.sum(other.to(tl.int32), 0)
    escapes = tl.sum(rare.to(tl.int32), 0)
    length = 65 + tl.cdiv(count, 8) + tl.cdiv(escapes * 3, 8)
    return first | (second << 3), length


@triton.jit
def count(
    previous, current, lengths, modes, HAS_PREVIOUS: tl.constexpr, ADDITIVE: tl.constexpr = False
):
    group, lane = tl.program_id(0), tl.arange(0, 512)
    value = _symbols(current, group, lane)
    if ADDITIVE:
        value = (value - _symbols(previous, group, lane)) & 7
    mode, size = _analyze(value)
    if HAS_PREVIOUS and not ADDITIVE:
        other_mode, other_size = _analyze(value ^ _symbols(previous, group, lane))
        better = other_size < size
        mode = tl.where(better, other_mode | 64, mode)
        size = tl.minimum(size, other_size)
    tl.store(lengths + group, tl.minimum(size, 192))
    tl.store(modes + group, tl.where(size >= 192, -1, mode))


@triton.jit
def count_choices(previous, current, lengths, modes, GROUPS: tl.constexpr):
    """Count parent and additive packets from one source read/launch."""
    group, lane = tl.program_id(0), tl.arange(0, 512)
    value, old = _symbols(current, group, lane), _symbols(previous, group, lane)
    mode, size = _analyze(value)
    delta_mode, delta_size = _analyze(value ^ old)
    better = delta_size < size
    tl.store(lengths + group, tl.minimum(tl.minimum(size, delta_size), 192))
    chosen_size = tl.minimum(size, delta_size)
    chosen_mode = tl.where(better, delta_mode | 64, mode)
    tl.store(modes + group, tl.where(chosen_size >= 192, -1, chosen_mode))
    add_mode, add_size = _analyze((value - old) & 7)
    tl.store(lengths + GROUPS + group, tl.minimum(add_size, 192))
    tl.store(modes + GROUPS + group, tl.where(add_size >= 192, -1, add_mode))


@triton.jit
def pack(
    previous,
    current,
    lengths,
    modes,
    ends,
    payload,
    offsets,
    GROUPS: tl.constexpr,
    HAS_PREVIOUS: tl.constexpr,
    ADDITIVE: tl.constexpr = False,
):
    group, lane = tl.program_id(0), tl.arange(0, 512)
    mode = tl.load(modes + group)
    length, end = tl.load(lengths + group), tl.load(ends + group)
    start, raw = end - length, mode == -1
    tl.store(offsets + group, start.to(tl.uint32) | tl.where(raw, 0x80000000, 0).to(tl.uint32))
    if group == GROUPS - 1:
        tl.store(offsets + GROUPS, end)
    original = tl.load(current + group * 192 + lane, lane < 192, other=0)
    if not ADDITIVE:
        tl.store(payload + start + lane, original, raw & (lane < 192))
    value = _symbols(current, group, lane)
    if ADDITIVE:
        value = (value - _symbols(previous, group, lane)) & 7
        # Raw groups also store residual symbols, matching the CPU packet.
        raw_word = tl.sum(tl.reshape(value, (64, 8)) << (tl.arange(0, 8)[None, :] * 3), 1)
        for b in tl.static_range(3):
            tl.store(
                payload + start + tl.arange(0, 64) * 3 + b, (raw_word >> (8 * b)).to(tl.uint8), raw
            )
    if HAS_PREVIOUS and not ADDITIVE:
        value ^= tl.where((mode & 64) != 0, _symbols(previous, group, lane), 0)
    first, second = mode & 7, (mode >> 3) & 7
    other = value != first
    rare = other & (value != second)
    count = tl.sum(other.to(tl.int32), 0)
    escapes = tl.sum(rare.to(tl.int32), 0)
    bitmap0 = tl.sum(tl.reshape(other.to(tl.int32), (64, 8)) << tl.arange(0, 8)[None, :], 1)
    tl.store(payload + start, mode.to(tl.uint8), ~raw)
    tl.store(payload + start + 1 + tl.arange(0, 64), bitmap0.to(tl.uint8), ~raw)
    ordered = tl.sort(tl.where(other, lane, lane + 512), descending=False)
    ordered_rare = tl.gather(rare.to(tl.int32), ordered % 512, 0) & (lane < count)
    bitmap1 = tl.sum(tl.reshape(ordered_rare, (64, 8)) << tl.arange(0, 8)[None, :], 1)
    tl.store(
        payload + start + 65 + tl.arange(0, 64),
        bitmap1.to(tl.uint8),
        (~raw) & (tl.arange(0, 64) < tl.cdiv(count, 8)),
    )
    rare_order = tl.sort(tl.where(rare, lane, lane + 512), descending=False)
    literal = tl.where(lane < escapes, tl.gather(value, rare_order % 512, 0), 0)
    word = tl.sum(tl.reshape(literal, (64, 8)) << (tl.arange(0, 8)[None, :] * 3), 1)
    for b in tl.static_range(3):
        i = tl.arange(0, 64) * 3 + b
        tl.store(
            payload + start + 65 + tl.cdiv(count, 8) + i,
            (word >> (8 * b)).to(tl.uint8),
            (~raw) & (i < tl.cdiv(escapes * 3, 8)),
        )


@triton.jit
def _bitmap_counts(payload, start, count, enabled):
    lane = tl.arange(0, 64)
    byte = tl.load(payload + start + lane, enabled & (lane < tl.cdiv(count, 8)), other=0)
    byte = byte.to(tl.int32)
    counts = libdevice.popc(byte)
    return byte, tl.cumsum(counts, 0), tl.sum(counts, 0)


@triton.jit
def _bitmap_rank(byte, cumulative, slot):
    selected = tl.gather(byte, slot // 8, 0)
    prefix = tl.gather(cumulative, slot // 8, 0) - libdevice.popc(selected)
    before = selected & ((1 << (slot % 8)) - 1)
    return ((selected >> (slot % 8)) & 1) != 0, prefix + libdevice.popc(before)


@triton.jit
def unpack(
    previous,
    payload,
    offsets,
    output,
    HAS_PREVIOUS: tl.constexpr,
    BITMAP_PREFIX: tl.constexpr = False,
    ADDITIVE: tl.constexpr = False,
    PACKED_PREVIOUS: tl.constexpr = False,
):
    group, lane = tl.program_id(0), tl.arange(0, 512)
    offset = tl.load(offsets + group).to(tl.uint32)
    raw = (offset & 0x80000000) != 0
    start = (offset & 0x7FFFFFFF).to(tl.int32)
    mode = tl.load(payload + start, ~raw, other=0).to(tl.int32)
    if BITMAP_PREFIX:
        byte0, prefix0, count = _bitmap_counts(payload, start + 1, 512, ~raw)
        flag0, rank0 = _bitmap_rank(byte0, prefix0, lane)
        byte1, prefix1, _ = _bitmap_counts(payload, start + 65, count, ~raw)
        second, rank1 = _bitmap_rank(byte1, prefix1, rank0)
        flag1 = flag0 & second
    else:
        flag0 = (
            (tl.load(payload + start + 1 + lane // 8, ~raw, other=0).to(tl.int32) >> (lane % 8)) & 1
        ) != 0
        rank0 = tl.maximum(tl.cumsum(flag0.to(tl.int32), 0) - 1, 0)
        count = tl.sum(flag0.to(tl.int32), 0)
        flag1 = flag0 & (
            (
                (
                    tl.load(payload + start + 65 + rank0 // 8, (~raw) & flag0, other=0).to(tl.int32)
                    >> (rank0 % 8)
                )
                & 1
            )
            != 0
        )
        rank1 = tl.maximum(tl.cumsum(flag1.to(tl.int32), 0) - 1, 0)
    bit, shift = rank1 * 3, (rank1 * 3) % 8
    literal = start + 65 + tl.cdiv(count, 8) + bit // 8
    a = tl.load(payload + literal, (~raw) & flag1, other=0).to(tl.int32)
    b = tl.load(payload + literal + 1, (~raw) & flag1 & (shift > 5), other=0).to(tl.int32)
    code = ((a >> shift) | (b << (8 - shift))) & 7
    value = tl.where(flag0, tl.where(flag1, code, (mode >> 3) & 7), mode & 7)
    if HAS_PREVIOUS and not ADDITIVE and not PACKED_PREVIOUS:
        value ^= tl.where((mode & 64) != 0, _symbols(previous, group, lane), 0)
    if ADDITIVE and not PACKED_PREVIOUS:
        bit = lane * 3
        low = tl.load(payload + start + bit // 8, raw, other=0).to(tl.int32)
        high = tl.load(payload + start + bit // 8 + 1, raw & (bit % 8 > 5), other=0).to(tl.int32)
        raw_value = ((low >> (bit % 8)) | (high << (8 - bit % 8))) & 7
        value = (tl.where(raw, raw_value, value) + _symbols(previous, group, lane)) & 7
    word = tl.sum(tl.reshape(value, (64, 8)) << (tl.arange(0, 8)[None, :] * 3), 1)
    if HAS_PREVIOUS and PACKED_PREVIOUS:
        index = group * 192 + tl.arange(0, 64) * 3
        needed = ADDITIVE | ((~raw) & ((mode & 64) != 0))
        old = tl.load(previous + index, needed, other=0).to(tl.int32)
        old |= tl.load(previous + index + 1, needed, other=0).to(tl.int32) << 8
        old |= tl.load(previous + index + 2, needed, other=0).to(tl.int32) << 16
        if ADDITIVE:
            literal = tl.load(payload + start + tl.arange(0, 64) * 3, raw, other=0).to(tl.int32)
            literal |= (
                tl.load(payload + start + tl.arange(0, 64) * 3 + 1, raw, other=0).to(tl.int32) << 8
            )
            literal |= (
                tl.load(payload + start + tl.arange(0, 64) * 3 + 2, raw, other=0).to(tl.int32) << 16
            )
            residual = tl.where(raw, literal, word)
            word = (
                ((residual & 0x6DB6DB) + (old & 0x6DB6DB)) ^ ((residual ^ old) & 0x924924)
            ) & 0xFFFFFF
        else:
            word ^= old
    for j in tl.static_range(3):
        index = tl.arange(0, 64) * 3 + j
        direct = tl.load(payload + start + index, raw, other=0)
        if ADDITIVE:
            result = (word >> (8 * j)).to(tl.uint8)
        else:
            result = tl.where(raw, direct, (word >> (8 * j)).to(tl.uint8))
        tl.store(output + group * 192 + index, result)
