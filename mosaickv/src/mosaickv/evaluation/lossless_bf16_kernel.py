"""Integer-only GPU BF16 reconstruction. No attention or model arithmetic changes."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT", "PROMPT", "LENGTH"])
def _decode(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    OUTPUT,
    COUNT,
    PROMPT,
    LENGTH,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    index = block * BLOCK + local
    bit = local * 3
    byte = bit // 8
    shift = bit % 8
    first = tl.load(CODES + block * (BLOCK * 3 // 8) + byte)
    second = tl.load(
        CODES + block * (BLOCK * 3 // 8) + byte + 1, mask=byte + 1 < BLOCK * 3 // 8, other=0
    )
    code = ((first.to(tl.int32) >> shift) | (second.to(tl.int32) << (8 - shift))) & 7
    escaped = (code == 7) & (index < COUNT)
    rank = tl.cumsum(escaped.to(tl.int32)) - 1
    offset = tl.load(OFFSETS + block)
    literal = tl.load(LITERALS + offset + rank, mask=escaped, other=0).to(tl.int32)
    base = tl.load(BASE + block).to(tl.int32)
    exponent = tl.where(escaped, literal, base + code)
    sm = tl.load(SM + index, mask=index < COUNT, other=0).to(tl.int32)
    bits = ((sm & 127) | ((sm & 128) << 8) | (exponent << 7)).to(tl.uint16)
    head = index // (PROMPT * DIM)
    destination = head * LENGTH * DIM + index % (PROMPT * DIM)
    tl.store(OUTPUT + destination, bits, mask=index < COUNT)


def decode(packed, destination):
    import torch

    count = packed.buffers[0].numel()
    _decode[(triton.cdiv(count, 1024),)](
        *packed.buffers,
        destination.view(torch.uint16),
        COUNT=count,
        PROMPT=packed.shape[-2],
        LENGTH=destination.shape[-2],
        DIM=packed.shape[-1],
        BLOCK=1024,
    )
