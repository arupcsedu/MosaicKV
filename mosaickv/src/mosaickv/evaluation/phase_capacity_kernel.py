"""In-place paired-half rotation into an existing strided BF16 prompt lease."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "SH", "SN"])
def _rotate(OUT, POS, COS, SIN, N, SH, SN, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < H * N * 64
    d, n, h = i % 64, i // 64 % N, i // (64 * N)
    p = tl.load(POS + h * N + n, valid, other=0)
    a = tl.load(OUT + h * SH + n * SN + d, valid, other=0).to(tl.float32)
    b = tl.load(OUT + h * SH + n * SN + d + 64, valid, other=0).to(tl.float32)
    c = tl.load(COS + p * 128 + d, valid, other=0).to(tl.float32)
    s = tl.load(SIN + p * 128 + d, valid, other=0).to(tl.float32)
    # Match HF BF16 multiply, multiply, add. Do not fuse into an FP32 FMA.
    ac = (a * c).to(tl.bfloat16).to(tl.float32)
    bs = (-b * s).to(tl.bfloat16).to(tl.float32)
    bc = (b * c).to(tl.bfloat16).to(tl.float32)
    ass = (a * s).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + h * SH + n * SN + d, ac + bs, valid)
    tl.store(OUT + h * SH + n * SN + d + 64, bc + ass, valid)


def rotate(target, positions, cos, sin):
    h, n = positions.shape
    if (
        target.shape[:2] != (1, h)
        or target.shape[-1] != 128
        or target.shape[-2] < n
        or target.stride(-1) != 1
    ):
        raise ValueError("feature-contiguous BF16 prompt lease required")
    _rotate[(triton.cdiv(h * n * 64, 1024),)](
        target,
        positions,
        cos,
        sin,
        n,
        target.stride(1),
        target.stride(2),
        H=h,
        BLOCK=1024,
        enable_fp_fusion=False,
    )
