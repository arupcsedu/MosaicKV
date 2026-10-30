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


@triton.jit(do_not_specialize=["COUNT"])
def _encode_headers(INPUT, SM, CODES, BASES, COUNTS, COUNT, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    index = block * BLOCK + local
    valid = index < COUNT
    word = tl.load(INPUT + index, mask=valid, other=0).to(tl.int32)
    exponent = (word >> 7) & 255
    histogram = tl.histogram(exponent, 256, mask=valid)
    prefix = tl.cumsum(histogram)
    bins = tl.arange(0, 256)
    lower = tl.gather(prefix, tl.maximum(bins - 1, 0), 0)
    upper = tl.gather(prefix, tl.minimum(bins + 6, 255), 0)
    scores = tl.where(bins <= 249, upper - tl.where(bins > 0, lower, 0), -1)
    maximum = tl.max(scores, 0)
    base = tl.min(tl.where(scores == maximum, bins, 256), 0)
    delta = exponent - base
    escaped = ((delta < 0) | (delta > 6)) & valid
    code = tl.where(valid, tl.where(escaped, 7, delta), 0).to(tl.uint32)
    parts = code << (3 * (local % 8))
    words = tl.sum(tl.reshape(parts, (BLOCK // 8, 8)), 1)
    group = tl.arange(0, BLOCK // 8)
    for byte in tl.static_range(3):
        tl.store(
            CODES + block * (BLOCK * 3 // 8) + group * 3 + byte,
            ((words >> (byte * 8)) & 255).to(tl.uint8),
        )
    tl.store(SM + index, ((word & 127) | ((word >> 8) & 128)).to(tl.uint8), mask=valid)
    tl.store(BASES + block, base.to(tl.uint8))
    tl.store(COUNTS + block, tl.sum(escaped.to(tl.int32), 0))


@triton.jit(do_not_specialize=["COUNT"])
def _encode_literals(INPUT, BASES, OFFSETS, LITERALS, COUNT, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    index = block * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    word = tl.load(INPUT + index, mask=valid, other=0).to(tl.int32)
    exponent = (word >> 7) & 255
    base = tl.load(BASES + block).to(tl.int32)
    escaped = ((exponent < base) | (exponent > base + 6)) & valid
    rank = tl.cumsum(escaped.to(tl.int32)) - 1
    offset = tl.load(OFFSETS + block)
    tl.store(LITERALS + offset + rank, exponent.to(tl.uint8), mask=escaped)


def encode(tensor, *, raw_fallback=True):
    import torch

    from mosaickv.evaluation.lossless_bf16 import PackedBF16

    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda" or not tensor.is_contiguous():
        raise ValueError("contiguous CUDA BF16 tensor required")
    count = tensor.numel()
    if count == 0:
        raise ValueError("nonempty source required")
    blocks = triton.cdiv(count, 1024)
    sm = torch.empty(count, dtype=torch.uint8, device=tensor.device)
    codes = torch.empty(blocks * 384, dtype=torch.uint8, device=tensor.device)
    bases = torch.empty(blocks, dtype=torch.uint8, device=tensor.device)
    counts = torch.empty(blocks, dtype=torch.int32, device=tensor.device)
    _encode_headers[(blocks,)](
        tensor.view(torch.uint16), sm, codes, bases, counts, count, BLOCK=1024
    )
    offsets = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    total = int(offsets[-1].item())  # One allocation-size synchronization, never a model probe.
    encoded_size = count + blocks * 384 + blocks + (blocks + 1) * 4 + total
    if raw_fallback and encoded_size >= count * 2:
        return PackedBF16(tuple(tensor.shape), True, (tensor.clone(),))
    literals = torch.empty(total, dtype=torch.uint8, device=tensor.device)
    _encode_literals[(blocks,)](
        tensor.view(torch.uint16), bases, offsets, literals, count, BLOCK=1024
    )
    return PackedBF16(tuple(tensor.shape), False, (sm, codes, bases, offsets, literals))
