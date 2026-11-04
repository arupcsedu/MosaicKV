"""Direct encoded-input K/V projection into the existing charged layer lease.

Derived from this repository's encoded_projection_kernel; no decoded full
source or separate K/V temporary is allocated in GPU global memory.
"""

import triton
import triton.language as tl

from mosaickv.evaluation.prompt_source_subphases import phase as source_phase


@triton.jit
def _read(
    CODES,
    MINIMUM,
    SCALE,
    row,
    start,
    ROWS,
    WIDTH: tl.constexpr,
    BITS: tl.constexpr,
    REUSE_HEADERS: tl.constexpr = False,
    ROW4: tl.constexpr = False,
):
    col = start + tl.arange(0, 64)
    index = col[None, :] * ROWS + row[:, None]
    valid = row[:, None] < ROWS
    if ROW4:
        flat = row[:, None] * WIDTH + col[None, :]
        byte = tl.load(CODES + flat // 2, valid, other=0).to(tl.int32)
        code = (byte >> ((flat % 2) * 4)) & 7
    else:
        bit = index * BITS
        byte, shift = bit // 8, bit % 8
        a = tl.load(CODES + byte, valid, other=0).to(tl.int32)
        b = tl.load(CODES + byte + 1, valid & (shift + BITS > 8), other=0).to(tl.int32)
        code = ((a >> shift) | (b << (8 - shift))) & ((1 << BITS) - 1)
    if REUSE_HEADERS:
        # The fixed 16/32/64-row tiles divide the 128-row quantizer group.
        # Every valid row in this CTA therefore shares these exact headers.
        group = col * tl.cdiv(ROWS, 128) + tl.program_id(0) * row.shape[0] // 128
        minimum = tl.load(MINIMUM + group)[None, :]
        scale = tl.load(SCALE + group)[None, :]
        result = (code.to(tl.float32) * scale + minimum).to(tl.bfloat16)
        return tl.where(valid, result, 0)
    else:
        group = col[None, :] * tl.cdiv(ROWS, 128) + row[:, None] // 128
        minimum = tl.load(MINIMUM + group, valid, other=0)
        scale = tl.load(SCALE + group, valid, other=1)
        return (code.to(tl.float32) * scale + minimum).to(tl.bfloat16)


@triton.jit
def _read_coherent(
    CODES,
    MINIMUM,
    SCALE,
    MAP,
    VALUES,
    row,
    start,
    ROWS,
    WIDTH: tl.constexpr,
    BITS: tl.constexpr,
    HAS_COLUMNS: tl.constexpr,
    REUSE_HEADERS: tl.constexpr = False,
    ROW4: tl.constexpr = False,
):
    x = _read(CODES, MINIMUM, SCALE, row, start, ROWS, WIDTH, BITS, REUSE_HEADERS, ROW4)
    if HAS_COLUMNS:
        col = start + tl.arange(0, 64)
        slot = tl.load(MAP + col).to(tl.int32)
        restore = (slot[None, :] != 255) & (row[:, None] < ROWS)
        original = tl.load(VALUES + slot[None, :] * ROWS + row[:, None], restore, other=0)
        x = tl.where(restore, original, x)
    return x


@triton.jit(do_not_specialize=["ROWS", "HEAD_OUT"])
def _project(
    CODES,
    MINIMUM,
    SCALE,
    COLUMN_MAP,
    COLUMN_VALUES,
    WK,
    WV,
    COS,
    SIN,
    OK,
    OV,
    ROWS,
    HEAD_OUT,
    WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    BITS: tl.constexpr,
    COMPONENT: tl.constexpr = -1,
    HEAD_GROUP: tl.constexpr = 1,
    HAS_COLUMNS: tl.constexpr = False,
    ROW_TILE: tl.constexpr = 16,
    REUSE_HEADERS: tl.constexpr = False,
    ROW4: tl.constexpr = False,
):
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    head = tl.program_id(1) * HEAD_GROUP
    col = tl.arange(0, DIM * HEAD_GROUP)
    reduction = tl.arange(0, 64)
    ak = tl.full((ROW_TILE, DIM * HEAD_GROUP), 0, tl.float32)
    av = tl.full((ROW_TILE, DIM * HEAD_GROUP), 0, tl.float32)
    for start in range(WIDTH // 64):
        x = _read_coherent(
            CODES,
            MINIMUM,
            SCALE,
            COLUMN_MAP,
            COLUMN_VALUES,
            row,
            start * 64,
            ROWS,
            WIDTH,
            BITS,
            HAS_COLUMNS,
            REUSE_HEADERS,
            ROW4,
        )
        k = start * 64 + reduction
        if COMPONENT != 1:
            wk = tl.load(WK + (head * DIM + col[None, :]) * WIDTH + k[:, None])
            ak = tl.dot(x, wk, ak)
        if COMPONENT != 0:
            wv = tl.load(WV + (head * DIM + col[None, :]) * WIDTH + k[:, None])
            av = tl.dot(x, wv, av)
    valid = row[:, None] < ROWS
    pointer = (head + col[None, :] // DIM) * HEAD_OUT + row[:, None] * DIM + col[None, :] % DIM
    if COMPONENT != 1:
        key = ak.to(tl.bfloat16).to(tl.float32)
        rotate_index = col // DIM * DIM + (col + DIM // 2) % DIM
        rotated = tl.gather(
            key, tl.broadcast_to(rotate_index[None, :], (ROW_TILE, DIM * HEAD_GROUP)), 1
        )
        rotated = tl.where(col[None, :] % DIM < DIM // 2, -rotated, rotated)
        cos = tl.load(COS + row[:, None] * DIM + col[None, :] % DIM, valid, other=0).to(tl.float32)
        sin = tl.load(SIN + row[:, None] * DIM + col[None, :] % DIM, valid, other=0).to(tl.float32)
        first = (key * cos).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sin).to(tl.bfloat16).to(tl.float32)
        tl.store(OK + pointer, first + second, valid)
    if COMPONENT != 0:
        tl.store(OV + pointer, av.to(tl.bfloat16), valid)


def project(packet, attention, targets, *, component=-1, inspect_kernel=False):
    import torch

    with source_phase("validation"):
        if type(component) is not int or component not in (-1, 0, 1):
            raise ValueError("explicit both/key/value projection component required")
        h, n, d = packet.shape[1:]
        head_group = getattr(packet, "source_head_group", 1)
        row_tile = getattr(packet, "source_row_tile", 16)
        if type(row_tile) is not int or row_tile not in (16, 32, 64):
            raise ValueError("explicit supported row tile required")
        if type(head_group) is not int or head_group not in (1, 2, 4) or h % head_group:
            raise ValueError("divisible audited head group of one, two or four required")
        warps = getattr(packet, "source_num_warps", None)
        if warps is None:
            warps = 8 if head_group == 4 else 4
        if type(warps) is not int or warps not in (4, 8, 16):
            raise ValueError("explicit supported warp count required")
        body = packet.source
        from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

        column_map, column_values, has_columns = kernel_arguments(packet)
        width = body.shape[-1]
        if body.bits != 3 or body.axis != 2 or body.group != 128 or width != h * d or d != 128:
            raise ValueError("audited three-bit per-channel full-MHA source required")
        if any(t.stride(-1) != 1 or t.stride(-2) != d for t in targets) or targets[0].stride(
            1
        ) != targets[1].stride(1):
            raise ValueError("compatible existing strided K/V leases required")
        if (
            attention.k_proj.bias is not None
            or attention.v_proj.bias is not None
            or not attention.k_proj.weight.is_contiguous()
            or not attention.v_proj.weight.is_contiguous()
            or any(
                weight.shape != (h * d, width)
                or weight.dtype != torch.bfloat16
                or weight.device != body.codes.device
                for weight in (attention.k_proj.weight, attention.v_proj.weight)
            )
        ):
            raise ValueError("unchanged contiguous bias-free native projection weights required")
    from mosaickv.evaluation.reconstruction_policy import reuse_headers
    from mosaickv.evaluation.row_major_source_codes import lookup
    from mosaickv.evaluation.source_hotpath import source_launch

    with source_phase("indexing_views"):
        row4 = lookup(body)
        grid = (triton.cdiv(n, row_tile), h // head_group)
        arguments = (
            body.codes if row4 is None else row4,
            body.minimum,
            body.scale,
            column_map,
            column_values,
            attention.k_proj.weight,
            attention.v_proj.weight,
            packet.cos,
            packet.sin,
            *targets,
            n,
            targets[0].stride(1),
        )
        options = dict(
            WIDTH=width,
            DIM=d,
            BITS=3,
            COMPONENT=component,
            HEAD_GROUP=head_group,
            HAS_COLUMNS=has_columns,
            ROW_TILE=row_tile,
            REUSE_HEADERS=reuse_headers(),
            ROW4=row4 is not None,
            num_warps=warps,
            num_stages={1: 3, 2: 2, 4: 1}[head_group],
            enable_fp_fusion=False,
        )
    with source_phase("projection_dispatch"):
        compiled = source_launch(_project, grid, arguments, options, attention)
    if inspect_kernel:
        return dict(
            registers=getattr(compiled, "n_regs", None),
            spills=getattr(compiled, "n_spills", None),
            shared_memory_bytes=getattr(compiled.metadata, "shared", None),
            head_group=head_group,
            row_tile=row_tile,
            num_warps=warps,
            num_stages={1: 3, 2: 2, 4: 1}[head_group],
        )


@triton.jit(do_not_specialize=["ROWS"])
def _decoded_source(
    CODES,
    MINIMUM,
    SCALE,
    MAP,
    VALUES,
    OUT,
    ROWS,
    WIDTH: tl.constexpr,
    HAS_COLUMNS: tl.constexpr,
    REUSE_HEADERS: tl.constexpr = False,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    start = tl.program_id(1) * 64
    values = _read_coherent(
        CODES, MINIMUM, SCALE, MAP, VALUES, row, start, ROWS, WIDTH, 3, HAS_COLUMNS, REUSE_HEADERS
    )
    offsets = row[:, None] * WIDTH + start + tl.arange(0, 64)[None, :]
    tl.store(OUT + offsets, values, row[:, None] < ROWS)


def decode_source_for_construction(packet):
    """Charged temporary for numerical patch fitting ONLY, never decode."""
    import torch

    n, width = packet.source.shape[2:]
    from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

    column_map, column_values, has_columns = kernel_arguments(packet)
    out = torch.empty((1, n, width), device=packet.source.codes.device, dtype=torch.bfloat16)
    from mosaickv.evaluation.reconstruction_policy import reuse_headers

    _decoded_source[(triton.cdiv(n, 16), width // 64)](
        packet.source.codes,
        packet.source.minimum,
        packet.source.scale,
        column_map,
        column_values,
        out,
        n,
        WIDTH=width,
        HAS_COLUMNS=has_columns,
        REUSE_HEADERS=reuse_headers(),
        enable_fp_fusion=False,
    )
    return out


@triton.jit(do_not_specialize=["ROWS", "FIRST", "TAKE"])
def _decode_chunk(CODES, MINIMUM, SCALE, OUT, ROWS, FIRST, TAKE, WIDTH: tl.constexpr):
    local = tl.program_id(0) * 16 + tl.arange(0, 16)
    start = tl.program_id(1) * 64
    values = _read(CODES, MINIMUM, SCALE, FIRST + local, start, ROWS, WIDTH, 3)
    offset = local[:, None] * WIDTH + start + tl.arange(0, 64)[None, :]
    tl.store(OUT + offset, values, local[:, None] < TAKE)


def decode_chunk(packet, target, first):
    n, width = packet.source.shape[2:]
    take = target.shape[1]
    if not 0 <= first < first + take <= n or not target.is_contiguous():
        raise ValueError("bounded contiguous source chunk required")
    _decode_chunk[(triton.cdiv(take, 16), width // 64)](
        packet.source.codes,
        packet.source.minimum,
        packet.source.scale,
        target,
        n,
        first,
        take,
        WIDTH=width,
        enable_fp_fusion=False,
    )


@triton.jit(do_not_specialize=["ROWS", "FIRST", "TAKE", "HEAD_OUT"])
def _rotate_native(KEY, COS, SIN, ROWS, FIRST, TAKE, HEAD_OUT, COUNT, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    head = index // (TAKE * 64)
    row = index // 64 % TAKE + FIRST
    col = index % 64
    ptr = head * HEAD_OUT + row * 128 + col
    left = tl.load(KEY + ptr, valid, other=0).to(tl.float32)
    right = tl.load(KEY + ptr + 64, valid, other=0).to(tl.float32)
    cos = tl.load(COS + row * 128 + col, valid, other=0).to(tl.float32)
    sin = tl.load(SIN + row * 128 + col, valid, other=0).to(tl.float32)
    a = (left * cos).to(tl.bfloat16).to(tl.float32)
    b = (-right * sin).to(tl.bfloat16).to(tl.float32)
    c = (right * cos).to(tl.bfloat16).to(tl.float32)
    d = (left * sin).to(tl.bfloat16).to(tl.float32)
    tl.store(KEY + ptr, a + b, valid)
    tl.store(KEY + ptr + 64, c + d, valid)


def rotate_native(target, cos, sin, first, take):
    n = cos.shape[1]
    count = target.shape[1] * take * 64
    _rotate_native[(triton.cdiv(count, 256),)](
        target, cos, sin, n, first, take, target.stride(1), count, BLOCK=256, enable_fp_fusion=False
    )
