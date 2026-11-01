"""128-byte XOR bitmap codec. No attention, floating-point or model changes."""

import triton
import triton.language as tl


@triton.jit
def count(previous, current, lengths):
    group = tl.program_id(0)
    lane = tl.arange(0, 128)
    delta = tl.load(previous + group * 128 + lane) ^ tl.load(current + group * 128 + lane)
    nonzero = tl.sum((delta != 0).to(tl.int32), 0)
    tl.store(lengths + group, tl.minimum(128, 16 + nonzero))


@triton.jit
def pack(previous, current, lengths, ends, payload, offsets, GROUPS: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, 128)
    now = tl.load(current + group * 128 + lane)
    delta = tl.load(previous + group * 128 + lane) ^ now
    flags = delta != 0
    rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    length = tl.load(lengths + group)
    end = tl.load(ends + group)
    start = end - length
    raw = length == 128
    tl.store(offsets + group, start.to(tl.uint32) | tl.where(raw, 0x80000000, 0).to(tl.uint32))
    if group == GROUPS - 1:
        tl.store(offsets + GROUPS, end)
    tl.store(payload + start + lane, now, mask=raw)
    bitmap = tl.sum(tl.reshape(flags.to(tl.uint32), (16, 8)) << tl.arange(0, 8)[None, :], 1)
    tl.store(payload + start + tl.arange(0, 16), bitmap.to(tl.uint8), mask=~raw)
    tl.store(payload + start + 16 + rank, delta, mask=(~raw) & flags)


@triton.jit
def unpack(previous, payload, offsets, output):
    group = tl.program_id(0)
    lane = tl.arange(0, 128)
    header = tl.load(offsets + group).to(tl.uint32)
    raw = (header & 0x80000000) != 0
    start = (header & 0x7FFFFFFF).to(tl.int32)
    bitmap = tl.load(payload + start + lane // 8, mask=~raw, other=0).to(tl.uint32)
    flags = ((bitmap >> (lane % 8)) & 1) != 0
    rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    literal = tl.load(payload + start + 16 + rank, mask=(~raw) & flags, other=0)
    old = tl.load(previous + group * 128 + lane, mask=~raw, other=0)
    direct = tl.load(payload + start + lane, mask=raw, other=0)
    tl.store(output + group * 128 + lane, tl.where(raw, direct, old ^ literal))
