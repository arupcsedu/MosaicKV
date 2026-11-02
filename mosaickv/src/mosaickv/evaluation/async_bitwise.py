"""Bounded exact BF16 comparison flags; no host synchronization on CUDA."""

import torch


def match_flags(reference, candidate):
    if (
        reference.dtype != torch.bfloat16
        or candidate.dtype != reference.dtype
        or reference.shape != candidate.shape
        or reference.device != candidate.device
        or reference.ndim != 4
        or reference.shape[0] != 1
    ):
        raise ValueError("matching batch-one BF16 tensors required")
    if reference.device.type == "cuda" and reference.numel():
        from mosaickv.evaluation.async_bitwise_kernel import match_flags as gpu_flags

        return gpu_flags(reference, candidate)
    return torch.tensor(
        [torch.equal(reference.view(torch.int16), candidate.view(torch.int16))],
        dtype=torch.bool,
        device=reference.device,
    )


def gpu_controls():
    """Actual strided/bit-pattern controls; only called in explicit GPU jobs."""
    generator = torch.Generator(device="cuda").manual_seed(20261102)
    cases = []
    for rows in (1, 17, 65):
        owner = torch.randn((1, 2, rows * 2, 128), device="cuda", generator=generator).bfloat16()
        owner.flatten()[:4].view(torch.int16).copy_(
            torch.tensor([0, -32768, 32704, -64], device="cuda", dtype=torch.int16)
        )
        a = owner[:, :, ::2, :]
        for change in (False, True):
            b = a.clone()
            if change:
                b.view(torch.int16).flatten()[-1] ^= 1
            flags = match_flags(a, b)
            actual = bool(flags.all())
            expected = torch.equal(a.view(torch.int16), b.view(torch.int16))
            if actual != expected:
                raise RuntimeError("asynchronous bit comparison changed the exact predicate")
            cases.append(
                dict(rows=rows, changed=change, same=actual, strided=True, flag_bytes=flags.numel())
            )
    return cases
