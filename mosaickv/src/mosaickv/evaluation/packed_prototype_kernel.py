"""One-pass exact 3-bit source gather/prototype serializer; no full unpack."""

import triton
import triton.language as tl


@triton.jit
def copy(Codes, Retained, Anchors, Centers, Out, n, stored, width, WORDS: tl.constexpr):
    word_id = tl.program_id(0) * WORDS + tl.arange(0, WORDS)
    flat = word_id[:, None] * 8 + tl.arange(0, 8)[None, :]
    valid = flat < stored * width
    channel, slot = flat // stored, flat % stored
    row = tl.load(Retained + slot, valid, 0)
    old = channel * n + row
    byte = old // 8 * 3
    value = (
        tl.load(Codes + byte, valid, 0).to(tl.int32)
        | (tl.load(Codes + byte + 1, valid, 0).to(tl.int32) << 8)
        | (tl.load(Codes + byte + 2, valid, 0).to(tl.int32) << 16)
    )
    value = (value >> ((old % 8) * 3)) & 7
    group = tl.load(Anchors + slot, valid, -1)
    center = tl.load(Centers + tl.maximum(group, 0) * width + channel, valid & (group >= 0), 0)
    value = tl.where(group >= 0, center, value)
    packed = tl.sum(value << (tl.arange(0, 8)[None, :] * 3), 1)
    for index in range(3):
        tl.store(
            Out + word_id * 3 + index, (packed >> (8 * index)) & 255, word_id * 8 < stored * width
        )
