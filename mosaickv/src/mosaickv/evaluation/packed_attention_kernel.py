"""Experimental exact-codec attention. No source projection or full K/V lease.

Codec reconstruction is bit exact; attention reduction order is deliberately
tested separately against HF eager and must not be advertised as bit identical.
Only batch-one full-MHA, BF16, head dimension 128, one-token decode is supported.
"""

import triton
import triton.language as tl


@triton.jit
def _rows(SM, CODES, BASE, OFFSETS, LITERALS, CHECK, index, valid, RAW: tl.constexpr):
    col = tl.arange(0, 128)
    flat = index[:, None] * 128 + col[None, :]
    if RAW:
        return tl.load(SM + flat, valid[:, None], other=0)
    else:
        local = flat % 1024
        byte, shift = local * 3 // 8, local * 3 % 8
        first = tl.load(CODES + flat // 1024 * 384 + byte, valid[:, None], other=0)
        second = tl.load(
            CODES + flat // 1024 * 384 + byte + 1,
            valid[:, None] & (byte + 1 < 384),
            other=0,
        )
        code = ((first.to(tl.int32) >> shift) | (second.to(tl.int32) << (8 - shift))) & 7
        escape = (code == 7) & valid[:, None]
        rank = tl.cumsum(escape.to(tl.int32), 1) - 1
        block = index // 8
        prefix = tl.load(CHECK + block * 8 + index % 8, valid, other=0).to(tl.int32)
        start = tl.load(OFFSETS + block, valid, other=0).to(tl.int32) + prefix
        literal = tl.load(LITERALS + start[:, None] + rank, escape, other=0).to(tl.int32)
        base = tl.load(BASE + block, valid, other=0).to(tl.int32)
        exponent = tl.where(escape, literal, base[:, None] + code)
        sm = tl.load(SM + flat, valid[:, None], other=0).to(tl.int32)
        bits = ((sm & 127) | ((sm & 128) << 8) | (exponent << 7)).to(tl.uint16)
        return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def _sequence(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    CHECK,
    HISTORY,
    head,
    tokens,
    PROMPT,
    LENGTH,
    CAPACITY,
    RAW: tl.constexpr,
):
    original = _rows(
        SM, CODES, BASE, OFFSETS, LITERALS, CHECK, head * PROMPT + tokens, tokens < PROMPT, RAW
    )
    col = tl.arange(0, 128)
    history = tl.load(
        HISTORY + (head * CAPACITY + tokens[:, None] - PROMPT) * 128 + col[None, :],
        (tokens[:, None] >= PROMPT) & (tokens[:, None] < LENGTH),
        other=0,
    )
    return tl.where(tokens[:, None] < PROMPT, original, history)


@triton.jit(do_not_specialize=["PROMPT", "LENGTH", "CAPACITY"])
def _scores(
    Q,
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    CHECK,
    HISTORY,
    SCORES,
    PROMPT,
    LENGTH,
    CAPACITY,
    SCALE: tl.constexpr,
    RAW: tl.constexpr,
    TILE: tl.constexpr,
):
    head, block = tl.program_id(0), tl.program_id(1)
    tokens = block * TILE + tl.arange(0, TILE)
    col = tl.arange(0, 128)
    key = _sequence(
        SM,
        CODES,
        BASE,
        OFFSETS,
        LITERALS,
        CHECK,
        HISTORY,
        head,
        tokens,
        PROMPT,
        LENGTH,
        CAPACITY,
        RAW,
    )
    query = tl.load(Q + head * 128 + col)
    # Explicit BF16 boundaries match eager's matmul, then scalar multiply.
    product = tl.sum(key.to(tl.float32) * query[None, :].to(tl.float32), 1).to(tl.bfloat16)
    scaled = (product.to(tl.float32) * SCALE).to(tl.bfloat16)
    tl.store(SCORES + head * LENGTH + tokens, scaled, tokens < LENGTH)


@triton.jit(do_not_specialize=["PROMPT", "LENGTH", "CAPACITY"])
def _values(
    P,
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    CHECK,
    HISTORY,
    OUTPUT,
    PROMPT,
    LENGTH,
    CAPACITY,
    RAW: tl.constexpr,
    TILE: tl.constexpr,
):
    head = tl.program_id(0)
    tokens = tl.arange(0, TILE)
    col = tl.arange(0, 128)
    result = tl.full((128,), 0, tl.float32)
    for offset in range(tl.cdiv(LENGTH, TILE)):
        rows = offset * TILE + tokens
        value = _sequence(
            SM,
            CODES,
            BASE,
            OFFSETS,
            LITERALS,
            CHECK,
            HISTORY,
            head,
            rows,
            PROMPT,
            LENGTH,
            CAPACITY,
            RAW,
        )
        prob = tl.load(P + head * LENGTH + rows, rows < LENGTH, other=0).to(tl.float32)
        result += tl.sum(value.to(tl.float32) * prob[:, None], 0)
    tl.store(OUTPUT + head * 128 + col, result.to(tl.bfloat16))


def arguments(packed):
    buffers = packed.payload.buffers
    if packed.payload.raw:
        return (buffers[0],) * 5 + (packed.checkpoints,)
    return (*buffers, packed.checkpoints)


def attention(query, key, value, history, used, scaling):
    import torch

    heads, prompt = key.payload.shape[1:3]
    length = prompt + used
    capacity = history[0].shape[-2]
    scores = torch.empty((1, heads, 1, length), dtype=query.dtype, device=query.device)
    _scores[(heads, triton.cdiv(length, 32))](
        query,
        *arguments(key),
        history[0],
        scores,
        prompt,
        length,
        capacity,
        SCALE=scaling,
        RAW=key.payload.raw,
        TILE=32,
    )
    probabilities = torch.softmax(scores, -1, dtype=torch.float32).to(query.dtype)
    output = torch.empty((1, heads, 1, 128), dtype=query.dtype, device=query.device)
    _values[(heads,)](
        probabilities,
        *arguments(value),
        history[1],
        output,
        prompt,
        length,
        capacity,
        RAW=value.payload.raw,
        TILE=32,
    )
    return output.transpose(1, 2).contiguous(), probabilities
