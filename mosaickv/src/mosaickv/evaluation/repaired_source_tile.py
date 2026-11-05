"""Default-off native-style source tile; exact frozen numerical repairs required.

Source dequantization is derived from shared_source_kernel._read_coherent.
The only changed numerical primitive is GEMM tiling/reduction; original
headers, values, phases, K/V and positions are not retuned or replaced.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _read32(
    CODES,
    MINIMUM,
    SCALE,
    MAP,
    VALUES,
    row,
    start,
    ROWS,
    WIDTH: tl.constexpr,
    HAS_COLUMNS: tl.constexpr,
):
    col = start + tl.arange(0, 32)
    index = col[None, :] * ROWS + row[:, None]
    valid = row[:, None] < ROWS
    bit = index * 3
    byte, shift = bit // 8, bit % 8
    a = tl.load(CODES + byte, valid, other=0).to(tl.int32)
    b = tl.load(CODES + byte + 1, valid & (shift > 5), other=0).to(tl.int32)
    code = ((a >> shift) | (b << (8 - shift))) & 7
    group = col[None, :] * tl.cdiv(ROWS, 128) + row[:, None] // 128
    minimum = tl.load(MINIMUM + group, valid, other=0)
    scale = tl.load(SCALE + group, valid, other=1)
    x = (code.to(tl.float32) * scale + minimum).to(tl.bfloat16)
    if HAS_COLUMNS:
        slot = tl.load(MAP + col).to(tl.int32)
        restore = (slot[None, :] != 255) & valid
        original = tl.load(VALUES + slot[None, :] * ROWS + row[:, None], restore, other=0)
        x = tl.where(restore, original, x)
    return x


@triton.jit(do_not_specialize=["ROWS", "HEAD_OUT"])
def _project(
    CODES,
    MINIMUM,
    SCALE,
    MAP,
    VALUES,
    WEIGHT,
    OUT,
    ROWS,
    HEAD_OUT,
    WIDTH: tl.constexpr,
    HAS_COLUMNS: tl.constexpr,
):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)
    head = tl.program_id(1)
    col = tl.arange(0, 128)
    reduction = tl.arange(0, 32)
    accumulator = tl.full((64, 128), 0, tl.float32)
    for block in range(WIDTH // 32):
        x = _read32(CODES, MINIMUM, SCALE, MAP, VALUES, row, block * 32, ROWS, WIDTH, HAS_COLUMNS)
        k = block * 32 + reduction
        weight = tl.load(WEIGHT + (head * 128 + col[None, :]) * WIDTH + k[:, None])
        accumulator = tl.dot(x, weight, accumulator)
    pointer = head * HEAD_OUT + row[:, None] * 128 + col[None, :]
    tl.store(OUT + pointer, accumulator.to(tl.bfloat16), row[:, None] < ROWS)


@triton.jit(do_not_specialize=["ROWS"])
def _decode(
    CODES, MINIMUM, SCALE, MAP, VALUES, OUT, ROWS, WIDTH: tl.constexpr, HAS_COLUMNS: tl.constexpr
):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)
    col = tl.program_id(1) * 32
    x = _read32(CODES, MINIMUM, SCALE, MAP, VALUES, row, col, ROWS, WIDTH, HAS_COLUMNS)
    offset = row[:, None] * WIDTH + col + tl.arange(0, 32)[None, :]
    tl.store(OUT + offset, x, row[:, None] < ROWS)


def validate(packet, attention, component, target):
    if type(component) is not int or component not in (0, 1):
        raise ValueError("explicit key/value tile component required")
    _, heads, n, d = packet.shape
    body = packet.source
    weight = (attention.k_proj, attention.v_proj)[component]
    if (
        packet.shape[0] != 1
        or d != 128
        or body.shape != (1, 1, n, heads * d)
        or body.bits != 3
        or body.axis != 2
        or body.group != 128
        or body.codes.dtype != torch.uint8
        or not body.codes.is_contiguous()
        or body.codes.device != target.device
        or target.dtype != torch.bfloat16
        or target.ndim != 4
        or target.shape[:2] != (1, heads)
        or target.shape[-1] != 128
        or target.shape[-2] < n
        or target.stride(-1) != 1
        or target.stride(-2) != 128
        or target.untyped_storage().data_ptr() == body.codes.untyped_storage().data_ptr()
        or weight.bias is not None
        or weight.weight.dtype != torch.bfloat16
        or weight.weight.shape != (heads * d, heads * d)
        or not weight.weight.is_contiguous()
        or weight.weight.device != target.device
        or any(
            t.shape != (1, n, 128) or t.dtype != torch.bfloat16 or t.device != target.device
            for t in (packet.cos, packet.sin)
        )
    ):
        raise ValueError(
            "original source, full-MHA BF16 weights and private strided lease required"
        )


def project_raw(packet, attention, component, target, *, inspect_kernel=False):
    """Writes native tile + original RoPE; caller MUST fit/apply frozen repair."""
    from mosaickv.evaluation.shared_source_kernel import rotate_native
    from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

    validate(packet, attention, component, target)
    mapping, values, columns = kernel_arguments(packet)
    compiled = _project[(triton.cdiv(packet.shape[2], 64), packet.shape[1])](
        packet.source.codes,
        packet.source.minimum,
        packet.source.scale,
        mapping,
        values,
        (attention.k_proj, attention.v_proj)[component].weight,
        target,
        packet.shape[2],
        target.stride(1),
        WIDTH=packet.shape[1] * 128,
        HAS_COLUMNS=columns,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    if compiled.metadata.global_scratch_size != 0 or compiled.n_spills != 0:
        raise RuntimeError("native-style source tile introduced scratch or spills")
    if component == 0:
        rotate_native(target, packet.cos, packet.sin, 0, packet.shape[2])
    if inspect_kernel:
        return dict(
            registers=compiled.n_regs,
            spills=compiled.n_spills,
            shared_memory_bytes=compiled.metadata.shared,
            no_full_source_or_output_workspace=True,
        )


def decode_control(packet, target):
    """Control-only full dequantized output; never a serving workspace."""
    from mosaickv.evaluation.source_channel_exceptions import kernel_arguments

    n, width = packet.source.shape[2:]
    if (
        target.shape != (1, n, width)
        or not target.is_contiguous()
        or target.dtype != torch.bfloat16
    ):
        raise ValueError("control-only full BF16 source target required")
    mapping, values, columns = kernel_arguments(packet)
    compiled = _decode[(triton.cdiv(n, 64), width // 32)](
        packet.source.codes,
        packet.source.minimum,
        packet.source.scale,
        mapping,
        values,
        target,
        n,
        WIDTH=width,
        HAS_COLUMNS=columns,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if compiled.metadata.global_scratch_size != 0 or compiled.n_spills != 0:
        raise RuntimeError("source-word control introduced scratch or spills")
