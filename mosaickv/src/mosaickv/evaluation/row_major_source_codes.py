"""Lossless layout conversion of existing 3-bit codes into row-major nibbles.

No requantization, header, weight, K/V, position or rounding change. The charged
scratch is half a byte per code. Defaults remain the original packed layout.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_active = ContextVar("evicastkv_row_major_code_scratch", default=None)


def geometry(rows, width):
    if any(type(x) is not int or x < 1 for x in (rows, width)) or width % 8:
        raise ValueError("positive rows and eight-code aligned width required")
    if rows * width >= 2**31:
        raise ValueError("bounded int32 source indexing required")
    return rows * width // 2


def reference(codes, rows, width):
    size = geometry(rows, width)
    if (
        codes.device.type != "cpu"
        or codes.dtype != torch.uint8
        or codes.ndim != 1
        or codes.numel() != rows * width * 3 // 8
    ):
        raise ValueError("exact CPU three-bit column-major code payload required")
    index = torch.arange(rows * width, dtype=torch.long)
    byte, shift = (index * 3) // 8, (index * 3) % 8
    first = codes.long()[byte]
    following = torch.cat((codes.long(), torch.zeros(1, dtype=torch.long)))[byte + 1]
    values = ((first >> shift) | (following << (8 - shift))) & 7
    pairs = values.reshape(width, rows).t().contiguous().reshape(size, 2)
    return (pairs[:, 0] | (pairs[:, 1] << 4)).to(torch.uint8)


def pack_into(body, scratch):
    if body.row_map is not None:
        raise NotImplementedError("row-major scratch has no compact source encoding")
    if (
        body.bits != 3
        or body.axis != 2
        or body.group != 128
        or body.codes.device.type != "cuda"
        or body.shape[:2] != (1, 1)
    ):
        raise ValueError("original three-bit full-source quantizer layout required")
    rows, width = body.shape[2:]
    size = geometry(rows, width)
    if (
        scratch.dtype != torch.uint8
        or scratch.device != body.codes.device
        or scratch.ndim != 1
        or not scratch.is_contiguous()
        or scratch.numel() < size
        or any(
            scratch.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in body.tensors()
        )
    ):
        raise ValueError("separate charged contiguous nibble workspace required")
    from mosaickv.evaluation.row_major_source_kernel import pack

    pack(body.codes, scratch, rows, width)


@contextmanager
def use(body, scratch):
    if _active.get() is not None:
        raise RuntimeError("row-major source view is private and nonnested")
    rows, width = body.shape[2:]
    if (
        scratch.numel() < geometry(rows, width)
        or scratch.dtype != torch.uint8
        or scratch.device != body.codes.device
        or scratch.ndim != 1
        or not scratch.is_contiguous()
    ):
        raise ValueError("complete charged nibble workspace required")
    token = _active.set((body.codes.data_ptr(), tuple(body.shape), scratch))
    try:
        yield
    finally:
        _active.reset(token)


def lookup(body):
    state = _active.get()
    if state is None:
        return None
    if state[:2] != (body.codes.data_ptr(), tuple(body.shape)):
        raise ValueError("row-major scratch does not belong to this frozen source binding")
    return state[2]
