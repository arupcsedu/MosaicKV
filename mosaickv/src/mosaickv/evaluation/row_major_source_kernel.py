"""Integer-only layout conversion; no floating-point reconstruction here."""

import triton
import triton.language as tl


@triton.jit
def _code(CODES, index, valid):
    bit = index * 3
    byte, shift = bit // 8, bit % 8
    a = tl.load(CODES + byte, valid, other=0).to(tl.int32)
    b = tl.load(CODES + byte + 1, valid & (shift + 3 > 8), other=0).to(tl.int32)
    return ((a >> shift) | (b << (8 - shift))) & 7


@triton.jit
def _pack(CODES, OUT, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    pair = tl.program_id(1) * 64 + tl.arange(0, 64)
    valid = (row[:, None] < ROWS) & (pair[None, :] < WIDTH // 2)
    first = pair[None, :] * 2 * ROWS + row[:, None]
    low = _code(CODES, first, valid)
    high = _code(CODES, first + ROWS, valid)
    tl.store(OUT + row[:, None] * (WIDTH // 2) + pair[None, :], low | (high << 4), valid)


def pack(codes, scratch, rows, width):
    compiled = _pack[(triton.cdiv(rows, 16), triton.cdiv(width // 2, 64))](
        codes, scratch, ROWS=rows, WIDTH=width, num_warps=4
    )
    if getattr(compiled.metadata, "global_scratch_size", None) != 0:
        raise RuntimeError("integer source transport cannot introduce opaque global scratch")
