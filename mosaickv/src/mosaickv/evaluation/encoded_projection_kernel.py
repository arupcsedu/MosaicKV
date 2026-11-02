"""Project directly from exact encoded BF16 source; no decoded source tensor."""

from typing import Any

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT"])
def _checkpoints(CODES, OUT, COUNT, BLOCK: tl.constexpr, PACK_COUNTS: tl.constexpr = False):
    local = tl.arange(0, BLOCK)
    bit = local * 3
    byte = bit // 8
    first = tl.load(CODES + tl.program_id(0) * 384 + byte)
    second = tl.load(CODES + tl.program_id(0) * 384 + byte + 1, byte + 1 < 384, other=0)
    code = ((first.to(tl.int32) >> (bit % 8)) | (second.to(tl.int32) << (8 - bit % 8))) & 7
    escaped = (code == 7) & (tl.program_id(0) * BLOCK + local < COUNT)
    counts = tl.sum(tl.reshape(escaped.to(tl.int32), (8, 128)), 1)
    prefix = tl.cumsum(counts) - counts
    if PACK_COUNTS:
        groups = tl.arange(0, 8)
        halves = tl.sum(tl.reshape(escaped.to(tl.int32), (16, 64)), 1)
        first = tl.gather(halves, groups * 2, 0)
        # Group zero's prefix is always zero. Reuse its ten free low bits
        # for the eight 64-escape overflow flags. No metadata is added.
        overflow = tl.sum((first == 64).to(tl.int32) << groups, 0)
        prefix = tl.where(groups == 0, overflow, prefix) | ((first & 63) << 10)
    tl.store(OUT + tl.program_id(0) * 8 + tl.arange(0, 8), prefix.to(tl.int16))


@triton.jit
def _codes(CODES, index, valid):
    local = index % 1024
    byte = local * 3 // 8
    shift = local * 3 % 8
    base = index // 1024 * 384
    first = tl.load(CODES + base + byte, valid, other=0).to(tl.int32)
    second = tl.load(CODES + base + byte + 1, valid & (byte + 1 < 384), other=0).to(tl.int32)
    return ((first >> shift) | (second << (8 - shift))) & 7


@triton.jit
def _read(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    CHECK,
    row,
    start,
    ROWS,
    WIDTH: tl.constexpr,
    RAW: tl.constexpr,
    PACK_COUNTS: tl.constexpr = False,
):
    col = tl.arange(0, 64)
    begin = row * WIDTH + start
    index = begin[:, None] + col[None, :]
    valid = row[:, None] < ROWS
    if RAW:
        return tl.load(SM + index, valid, other=0).to(tl.uint16)
    else:
        code = _codes(CODES, index, valid)
        escaped = (code == 7) & valid
        rank = tl.cumsum(escaped.to(tl.int32), 1) - 1
        block = begin // 1024
        group = (begin % 1024) // 128
        prefix = tl.load(CHECK + block * 8 + group, row < ROWS, other=0).to(tl.int32)
        if PACK_COUNTS:
            flags = tl.load(CHECK + block * 8, row < ROWS, other=0).to(tl.int32) & 255
            extra = ((prefix >> 10) & 63) + (((flags >> group) & 1) << 6)
            extra = tl.where((start % 128) >= 64, extra, 0)
            prefix = tl.where(group == 0, 0, prefix & 1023)
        else:
            previous = _codes(CODES, index - 64, valid & ((start % 128) >= 64))
            extra = tl.sum((previous == 7).to(tl.int32), 1)
        offset = tl.load(OFFSETS + block, row < ROWS, other=0) + prefix + extra
        literal = tl.load(LITERALS + offset[:, None] + rank, escaped, other=0).to(tl.int32)
        base = tl.load(BASE + block, row < ROWS, other=0).to(tl.int32)
        exponent = tl.where(escaped, literal, base[:, None] + code)
        low = tl.load(SM + index, valid, other=0).to(tl.int32)
        return ((low & 127) | ((low & 128) << 8) | (exponent << 7)).to(tl.uint16)


@triton.jit
def _read_carried(
    SM, CODES, BASE, OFFSETS, LITERALS, CHECK, row, start, ROWS, WIDTH: tl.constexpr, previous_count
):
    col = tl.arange(0, 64)
    begin = row * WIDTH + start
    index = begin[:, None] + col[None, :]
    valid = row[:, None] < ROWS
    code = _codes(CODES, index, valid)
    escaped = (code == 7) & valid
    rank = tl.cumsum(escaped.to(tl.int32), 1) - 1
    block = begin // 1024
    group = (begin % 1024) // 128
    prefix = tl.load(CHECK + block * 8 + group, row < ROWS, other=0).to(tl.int32)
    extra = tl.where(start % 128 >= 64, previous_count, 0)
    offset = tl.load(OFFSETS + block, row < ROWS, other=0) + prefix + extra
    literal = tl.load(LITERALS + offset[:, None] + rank, escaped, other=0).to(tl.int32)
    base = tl.load(BASE + block, row < ROWS, other=0).to(tl.int32)
    exponent = tl.where(escaped, literal, base[:, None] + code)
    low = tl.load(SM + index, valid, other=0).to(tl.int32)
    return (
        ((low & 127) | ((low & 128) << 8) | (exponent << 7)).to(tl.uint16),
        tl.sum(escaped.to(tl.int32), 1),
    )


@triton.jit
def _materialize_carried(SM, CODES, BASE, OFFSETS, LITERALS, CHECK, OUT, ROWS, WIDTH: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    previous = tl.full((16,), 0, tl.int32)
    for start in range(WIDTH // 64):
        words, previous = _read_carried(
            SM, CODES, BASE, OFFSETS, LITERALS, CHECK, row, start * 64, ROWS, WIDTH, previous
        )
        offset = row[:, None] * WIDTH + start * 64 + tl.arange(0, 64)[None, :]
        tl.store(OUT + offset, words, row[:, None] < ROWS)


@triton.jit(do_not_specialize=["ROWS", "HEAD_OUT", "MAP_STRIDE"])
def _project(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    CHECK,
    WK,
    WV,
    UNION,
    SLOTS,
    COS,
    SIN,
    OK,
    OV,
    ROWS,
    HEAD_OUT,
    MAP_STRIDE,
    WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    RAW: tl.constexpr,
    PACK_COUNTS: tl.constexpr = False,
    COMPONENT: tl.constexpr = -1,
    CARRY_LITERALS: tl.constexpr = False,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    head = tl.program_id(1)
    col = tl.arange(0, DIM)
    reduction = tl.arange(0, 64)
    slot = tl.load(SLOTS + head * MAP_STRIDE + row, row < ROWS, other=-1).to(tl.int32)
    if tl.sum((slot >= 0).to(tl.int32), 0) > 0:
        ak = tl.full((16, DIM), 0, tl.float32)
        av = tl.full((16, DIM), 0, tl.float32)
        previous_count = tl.full((16,), 0, tl.int32)
        for start in range(WIDTH // 64):
            if CARRY_LITERALS and not RAW and not PACK_COUNTS:
                words, previous_count = _read_carried(
                    SM,
                    CODES,
                    BASE,
                    OFFSETS,
                    LITERALS,
                    CHECK,
                    row,
                    start * 64,
                    ROWS,
                    WIDTH,
                    previous_count,
                )
            else:
                words = _read(
                    SM,
                    CODES,
                    BASE,
                    OFFSETS,
                    LITERALS,
                    CHECK,
                    row,
                    start * 64,
                    ROWS,
                    WIDTH,
                    RAW,
                    PACK_COUNTS,
                )
            x = words.to(tl.bfloat16, bitcast=True)
            k = start * 64 + reduction
            if COMPONENT != 1:
                wk = tl.load(WK + (head * DIM + col[None, :]) * WIDTH + k[:, None])
                ak = tl.dot(x, wk, ak)
            if COMPONENT != 0:
                wv = tl.load(WV + (head * DIM + col[None, :]) * WIDTH + k[:, None])
                av = tl.dot(x, wv, av)
        key = ak.to(tl.bfloat16).to(tl.float32)
        rotated = tl.gather(key, tl.broadcast_to(((col + DIM // 2) % DIM)[None, :], (16, DIM)), 1)
        rotated = tl.where(col[None, :] < DIM // 2, -rotated, rotated)
        logical = tl.load(UNION + row, row < ROWS, other=0).to(tl.int32)
        cos = tl.load(COS + logical[:, None] * DIM + col[None, :], row[:, None] < ROWS, other=0).to(
            tl.float32
        )
        sin = tl.load(SIN + logical[:, None] * DIM + col[None, :], row[:, None] < ROWS, other=0).to(
            tl.float32
        )
        first = (key * cos).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sin).to(tl.bfloat16).to(tl.float32)
        pointer = head * HEAD_OUT + slot[:, None] * DIM + col[None, :]
        mask = (row[:, None] < ROWS) & (slot[:, None] >= 0)
        if COMPONENT != 1:
            tl.store(OK + pointer, (first + second).to(tl.bfloat16), mask)
        if COMPONENT != 0:
            tl.store(OV + pointer, av.to(tl.bfloat16), mask)


@triton.jit(do_not_specialize=["ROWS"])
def _materialize(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    CHECK,
    OUT,
    ROWS,
    WIDTH: tl.constexpr,
    RAW: tl.constexpr,
    PACK_COUNTS: tl.constexpr = False,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    start = tl.program_id(1) * 64
    words = _read(
        SM, CODES, BASE, OFFSETS, LITERALS, CHECK, row, start, ROWS, WIDTH, RAW, PACK_COUNTS
    )
    index = row[:, None] * WIDTH + start + tl.arange(0, 64)[None, :]
    tl.store(OUT + index, words, row[:, None] < ROWS)


def checkpoints(body: Any, *, packed_counts: bool = False) -> Any:
    import torch

    count = 1
    for dimension in body.shape:
        count *= dimension
    result = torch.empty(
        (0,) if body.raw else (triton.cdiv(count, 1024), 8),
        device=body.buffers[0].device,
        dtype=torch.int16,
    )
    if not body.raw:
        _checkpoints[(triton.cdiv(count, 1024),)](
            body.buffers[1], result, count, BLOCK=1024, PACK_COUNTS=packed_counts
        )
    return result


def pointers(body: Any) -> tuple[Any, ...]:
    import torch

    return (body.buffers[0].view(torch.uint16),) * 5 if body.raw else body.buffers


def project(
    body: Any,
    check: Any,
    attention: Any,
    archive: Any,
    slots: Any,
    targets: Any,
    *,
    packed_counts: bool = False,
    component: int = -1,
    inspect_kernel: bool = False,
) -> Any:
    import torch

    if type(component) is not int or component not in (-1, 0, 1):
        raise ValueError("explicit both/key/value projection component required")
    heads, kept = archive.selected.shape
    width = body.shape[-1]
    if (
        body.buffers[0].device.type != "cuda"
        or body.shape[:2] != (1, 1)
        or width != heads * 128
        or attention.head_dim != 128
        or body.shape[-2] != archive.union.numel()
        or attention.k_proj.bias is not None
        or attention.v_proj.bias is not None
        or any(
            t.dtype != torch.bfloat16
            or t.shape != (1, heads, kept, 128)
            or t.stride(-2) != 128
            or t.stride(-1) != 1
            or t.stride(-3) < kept * 128
            or t.device != body.buffers[0].device
            for t in targets
        )
        or targets[0].stride(-3) != targets[1].stride(-3)
    ):
        raise NotImplementedError("audited bias-free BF16 full-MHA source/lease required")
    if any(
        not layer.weight.is_contiguous()
        or layer.weight.dtype != torch.bfloat16
        or layer.weight.device != body.buffers[0].device
        or layer.weight.shape != (width, width)
        for layer in (attention.k_proj, attention.v_proj)
    ):
        raise NotImplementedError("contiguous unmodified BF16 square K/V weights required")
    from mosaickv.evaluation.reconstruction_policy import carry_literals
    from mosaickv.evaluation.request_kernel_dispatch import launch

    carry = carry_literals()
    compiled = launch(
        _project,
        (triton.cdiv(archive.union.numel(), 16), heads),
        (
            *pointers(body),
            check,
            attention.k_proj.weight,
            attention.v_proj.weight,
            archive.union,
            slots,
            archive.cos,
            archive.sin,
            *targets,
            archive.union.numel(),
            targets[0].stride(-3),
            slots.stride(0),
        ),
        WIDTH=width,
        DIM=128,
        RAW=body.raw,
        PACK_COUNTS=packed_counts,
        COMPONENT=component,
        CARRY_LITERALS=carry,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if inspect_kernel:
        return dict(
            registers=getattr(compiled, "n_regs", None),
            spills=getattr(compiled, "n_spills", None),
            shared_memory_bytes=getattr(compiled.metadata, "shared", None),
            launch_constants=dict(CARRY_LITERALS=carry, RAW=body.raw, PACK_COUNTS=packed_counts),
            compiled_reader_branch="raw"
            if body.raw
            else ("packed_counts" if packed_counts else "carried" if carry else "original"),
        )


def gpu_controls() -> list[dict[str, Any]]:
    import torch

    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_bf16_kernel import encode

    words = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    tensor = words.view(torch.bfloat16).reshape(1, 1, 256, 256)
    results = []
    for raw in (False, True):
        body = (
            PackedBF16(tuple(tensor.shape), True, (tensor.clone(),))
            if raw
            else encode(tensor, raw_fallback=False)
        )
        for packed_counts in (False, True):
            check = checkpoints(body, packed_counts=packed_counts)
            output = torch.empty_like(tensor)
            _materialize[(16, 4)](
                *pointers(body),
                check,
                output.view(torch.uint16),
                256,
                WIDTH=256,
                RAW=raw,
                PACK_COUNTS=packed_counts,
            )
            if not torch.equal(output.view(torch.int16), tensor.view(torch.int16)):
                raise RuntimeError("encoded-source read changed a BF16 pattern")
            results.append(
                dict(
                    raw=raw,
                    packed_counts=packed_counts,
                    all_65536_BF16_patterns_bitwise=True,
                    checkpoint_bytes=check.numel() * 2,
                )
            )
    # All-literal blocks exercise the 64-escape overflow in every half,
    # including group-zero flag ownership and prefixes up to 896.
    count = tensor.numel()
    bits = tensor.view(torch.int16).flatten().to(torch.int32) & 65535
    body = PackedBF16(
        tuple(tensor.shape),
        False,
        (
            ((bits & 127) | ((bits >> 8) & 128)).to(torch.uint8),
            torch.full((count // 8 * 3,), 255, dtype=torch.uint8, device="cuda"),
            torch.zeros(count // 1024, dtype=torch.uint8, device="cuda"),
            torch.arange(0, count + 1, 1024, dtype=torch.int32, device="cuda"),
            ((bits >> 7) & 255).to(torch.uint8),
        ),
    )
    check = checkpoints(body, packed_counts=True)
    output = torch.empty_like(tensor)
    _materialize[(16, 4)](
        *pointers(body),
        check,
        output.view(torch.uint16),
        256,
        WIDTH=256,
        RAW=False,
        PACK_COUNTS=True,
    )
    if not torch.equal(output.view(torch.int16), tensor.view(torch.int16)):
        raise RuntimeError("packed checkpoint overflow changed a BF16 pattern")
    results.append(
        dict(
            all_literal_overflow_bitwise=True,
            packed_counts=True,
            checkpoint_bytes=check.numel() * 2,
        )
    )
    return results
