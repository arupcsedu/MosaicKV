"""One bounded mixed INT8/exact-BF16 source decode; no projection changes."""

import triton
import triton.language as tl


@triton.jit
def exact(SM, CODES, BASE, OFFSETS, LITERALS, block, RAW: tl.constexpr):
    local = tl.arange(0, 1024)
    index = block * 1024 + local
    if RAW:
        return tl.load(SM + index).to(tl.uint16, bitcast=True)
    else:
        bit = local * 3
        byte, shift = bit // 8, bit % 8
        first = tl.load(CODES + block * 384 + byte).to(tl.int32)
        second = tl.load(CODES + block * 384 + byte + 1, byte + 1 < 384, other=0).to(tl.int32)
        code = ((first >> shift) | (second << (8 - shift))) & 7
        escaped = code == 7
        rank = tl.cumsum(escaped.to(tl.int32)) - 1
        offset = tl.load(OFFSETS + block)
        literal = tl.load(LITERALS + offset + rank, escaped, other=0).to(tl.int32)
        base = tl.load(BASE + block).to(tl.int32)
        exponent = tl.where(escaped, literal, base + code)
        sm = tl.load(SM + index).to(tl.int32)
        return ((sm & 127) | ((sm & 128) << 8) | (exponent << 7)).to(tl.uint16)


@triton.jit
def unpack(
    OLD,
    MINIMUM,
    SCALE,
    A0,
    A1,
    A2,
    A3,
    A4,
    B0,
    B1,
    B2,
    B3,
    B4,
    C0,
    C1,
    C2,
    C3,
    C4,
    D0,
    D1,
    D2,
    D3,
    D4,
    OUTPUT,
    OLD_ROWS: tl.constexpr,
    GROUP: tl.constexpr,
    RAW_A: tl.constexpr,
    RAW_B: tl.constexpr,
    RAW_C: tl.constexpr,
    RAW_D: tl.constexpr,
):
    block = tl.program_id(0)
    local = tl.arange(0, 1024)
    index = block * 1024 + local
    if block < OLD_ROWS * 4:
        q = tl.load(OLD + index).to(tl.float32)
        minimum = tl.load(MINIMUM + index // GROUP).to(tl.float32)
        scale = tl.load(SCALE + index // GROUP).to(tl.float32)
        # Match registered unpack's separate FP32 multiply/add and BF16 cast.
        value = (q * scale + minimum).to(tl.bfloat16)
    else:
        relative = block - OLD_ROWS * 4
        if relative < 64:
            word = exact(A0, A1, A2, A3, A4, relative, RAW_A)
        elif relative < 128:
            word = exact(B0, B1, B2, B3, B4, relative - 64, RAW_B)
        elif relative < 192:
            word = exact(C0, C1, C2, C3, C4, relative - 128, RAW_C)
        else:
            word = exact(D0, D1, D2, D3, D4, relative - 192, RAW_D)
        value = word.to(tl.bfloat16, bitcast=True)
    tl.store(OUTPUT + index, value)
