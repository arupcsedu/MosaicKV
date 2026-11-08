"""Bounded original-order FP32 affine dequantization to BF16 source scratch."""

import triton
import triton.language as tl


@triton.jit
def unpack(Codes, Minimum, Scale, Out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    code = tl.load(Codes + i, i < n, 0).to(tl.float32)
    scale = tl.load(Scale + i // 128, i < n, 1)
    minimum = tl.load(Minimum + i // 128, i < n, 0)
    value = tl.mul(code, scale)
    value = tl.add(value, minimum)
    tl.store(Out + i, value.to(tl.bfloat16), i < n)
