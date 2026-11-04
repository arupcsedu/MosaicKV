"""Reuse a charged idle prompt buffer, never a persistent packet's backing."""

import math

import torch


def partition(buffer, shape):
    if (
        len(shape) != 4
        or shape[0] != 1
        or shape[-1] != 128
        or any(type(x) is not int or x < 1 for x in shape)
        or buffer.dtype != torch.bfloat16
        or not buffer.is_contiguous()
    ):
        raise ValueError("bounded contiguous BF16 history verification geometry required")
    size = math.prod(shape)
    if 4 * size > buffer.numel():
        return None
    flat = buffer.view(-1)
    views = tuple(flat.narrow(0, i * size, size).view(shape) for i in range(4))
    return views[:2], views[2:]


def verify(layer, candidate, formats, shape, bit_flags, scratch=None):
    reference = (
        tuple(
            torch.empty(shape, dtype=torch.bfloat16, device=candidate.archive.cos.device)
            for _ in range(2)
        )
        if scratch is None
        else scratch[0]
    )
    if scratch is not None:
        for t in reference:
            t.fill_(float("nan"))
    offset = 0
    for f in formats:
        end = offset + f.shape[-2]
        f.fill(layer.attention, tuple(t[..., offset:end, :] for t in reference), direct=True)
        offset = end
    for _ in range(3):
        if scratch is None:
            actual = candidate.materialize(layer.attention)
        else:
            actual = scratch[1]
            # Every pass must fully overwrite its outputs. Reusing a correct
            # prior result must not conceal an incomplete subsequent writer.
            for t in actual:
                t.fill_(float("nan"))
            candidate.fill(layer.attention, actual, direct=True)
        if bit_flags is not None:
            bit_flags.extend(
                (a.view(torch.int16) == b.view(torch.int16)).all()
                for a, b in zip(reference, actual, strict=True)
            )
        elif not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(reference, actual, strict=True)
        ):
            raise RuntimeError("consolidated projection changed original history K/V bits")
        # Also remove the old verification-pair overlap for ordinary baselines.
        del actual
    if scratch is not None:
        owner = scratch[0][0].untyped_storage().data_ptr()
        if any(t.numel() and t.untyped_storage().data_ptr() == owner for t in candidate.private()):
            raise RuntimeError("persistent history packet aliases borrowed verification scratch")
