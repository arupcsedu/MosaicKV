"""Restore original FP32 words without floating-point scale arithmetic."""

import triton
import triton.language as tl


@triton.jit
def unpack(
    Payload,
    Palette,
    High,
    Output,
    n,
    length,
    BITS: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bit = i * BITS
    pos, shift = bit // 8, bit % 8
    a = tl.load(Payload + pos, (i < n) & (pos < length), 0).to(tl.uint32)
    b = tl.load(Payload + pos + 1, (i < n) & (pos + 1 < length), 0).to(tl.uint32)
    c = tl.load(Payload + pos + 2, (i < n) & (pos + 2 < length), 0).to(tl.uint32)
    index = ((a | (b << 8) | (c << 16)) >> shift) & ((1 << BITS) - 1)
    word = tl.load(Palette + index, i < n, 0).to(tl.uint32)
    if SPLIT:
        word = word | (tl.load(High + i, i < n, 0).to(tl.uint32) << 16)
    tl.store(Output + i, word.to(tl.float32, bitcast=True), i < n)
