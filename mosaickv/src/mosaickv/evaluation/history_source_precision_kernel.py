"""Bounded original-order FP32 affine dequantization to BF16 source scratch."""

import triton
import triton.language as tl


@triton.jit
def unpack(Codes, Minimum, Scale, Out, n, GROUP: tl.constexpr = 128, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    code = tl.load(Codes + i, i < n, 0).to(tl.float32)
    scale = tl.load(Scale + i // GROUP, i < n, 1)
    minimum = tl.load(Minimum + i // GROUP, i < n, 0)
    value = code * scale
    value = value + minimum
    tl.store(Out + i, value.to(tl.bfloat16), i < n)
