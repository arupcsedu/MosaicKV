"""Integer serialization ONLY, after original FP32 quantization is complete.

Read exact integer-valued floats directly from padded private workspace; avoid
allocating a full int32 tensor or a contiguous unpadded copy. Never recompute a
minimum, scale, division or rounding in Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def words_kernel(
    Q,
    OUT,
    COUNT: tl.constexpr,
    LENGTH: tl.constexpr,
    PAD: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    word = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 8)
    flat = word[:, None] * 8 + lane[None, :]
    q = tl.load(Q + (flat // LENGTH) * PAD + flat % LENGTH, mask=word[:, None] < COUNT, other=0).to(
        tl.int32
    )
    packed = tl.sum((q.to(tl.uint32) << (lane[None, :] * BITS)), 1)
    byte = tl.arange(0, 4)
    value = (packed[:, None] >> (byte[None, :] * 8)) & 255
    tl.store(
        OUT + word[:, None] * BITS + byte[None, :],
        value.to(tl.uint8),
        mask=(word[:, None] < COUNT) & (byte[None, :] < BITS),
    )


@triton.jit
def bytes_kernel(
    Q, OUT, COUNT: tl.constexpr, LENGTH: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr
):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + (flat // LENGTH) * PAD + flat % LENGTH, mask=flat < COUNT, other=0)
    tl.store(OUT + flat, q.to(tl.int32).to(tl.uint8), mask=flat < COUNT)


def serialize(quantized, *, bits, length, value_count):
    if (
        quantized.dtype != torch.float32
        or quantized.device.type != "cuda"
        or not quantized.is_contiguous()
        or bits not in (2, 3, 4, 8)
        or type(bits) is not int
        or type(length) is not int
        or type(value_count) is not int
        or value_count < 1
        or length < 1
        or quantized.numel() >= 2**31
        or value_count >= 2**31
        or (bits != 8 and value_count % 8)
    ):
        raise ValueError(
            "contiguous CUDA FP32 quantized workspace and aligned code geometry required"
        )
    padded = triton.cdiv(length, 128) * 128
    if quantized.numel() != (value_count // length) * padded:
        raise ValueError("complete original quantizer row layout required")
    size = value_count if bits == 8 else (value_count // 8) * bits
    output = torch.empty(size, dtype=torch.uint8, device=quantized.device)
    if bits == 8:
        bytes_kernel[(triton.cdiv(value_count, 1024),)](
            quantized, output, value_count, length, padded, 1024
        )
    else:
        count = value_count // 8
        words_kernel[(triton.cdiv(count, 128),)](
            quantized, output, count, length, padded, bits, 128
        )
    return output


def gpu_controls():
    from mosaickv.evaluation.conditional_precision import pack_unprotected
    from mosaickv.evaluation.precision_tier import pack

    generator = torch.Generator(device="cuda").manual_seed(2441)
    rows = []
    for bits in (2, 3, 4, 8):
        for axis in (2, 3):
            values = torch.randn((1, 2, 33, 128), device="cuda", generator=generator).bfloat16()
            snapshot = values.clone()
            old = pack(values, bits=bits, axis=axis, compact_workspace=False)
            actual = pack(values, bits=bits, axis=axis)
            if not all(
                a.dtype == b.dtype
                and a.shape == b.shape
                and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(old.tensors(), actual.tensors(), strict=True)
            ) or not torch.equal(snapshot.view(torch.int16), values.view(torch.int16)):
                raise RuntimeError("compact quantizer changed an original header/code/input bit")
            rows.append(dict(bits=bits, axis=axis, all_bits_identical=True, odd_length=33))
    for bits in (2, 3, 4, 8):
        for axis in (2, 3):
            values = torch.randn((1, 2, 129, 128), device="cuda", generator=generator).bfloat16()
            protected = torch.zeros((2, 129), device="cuda", dtype=torch.bool)
            protected[0, :128] = True
            protected[1, ::4] = True
            snapshot = values.clone()
            old = pack_unprotected(values, protected, bits=bits, axis=axis, bf16_prepare=False)
            actual = pack_unprotected(values, protected, bits=bits, axis=axis)
            if not all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(old.tensors(), actual.tensors(), strict=True)
            ) or not torch.equal(snapshot.view(torch.int16), values.view(torch.int16)):
                raise RuntimeError("compact ranges changed an original header/code/input bit")
            rows.append(
                dict(bits=bits, axis=axis, compact_ranges_bitwise=True, partial_group_length=129)
            )
    return rows
