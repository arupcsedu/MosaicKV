"""Operator-only controls. Never changes production arithmetic or model guards."""

from contextlib import contextmanager

import torch


@contextmanager
def projection_precision(*, reduced, fp32=False):
    """Restore every backend flag even if a diagnostic projection raises."""
    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
        if fp32:
            torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced


def comparison(reference, actual):
    if reference.shape != actual.shape or not torch.isfinite(actual).all():
        raise ValueError("finite identical-geometry operator outputs required")
    close = torch.isclose(actual, reference, rtol=0.01, atol=2**-6)
    bad = (~close).nonzero()
    errors = (reference.float() - actual.float()).abs()
    return dict(
        registered_tolerance_pass=bool(close.all()),
        mismatched_elements=len(bad),
        element_count=actual.numel(),
        maximum_absolute_difference=float(errors.max()),
        examples=[
            dict(
                index=idx, reference=float(reference[tuple(idx)]), actual=float(actual[tuple(idx)])
            )
            for idx in bad[:8].tolist()
        ],
    )


def projected(attention, x, phases, *, precision):
    if precision not in ("bf16_default", "bf16_no_truncated_reduction", "fp32", "fp64"):
        raise ValueError("registered projection precision required")
    dtype = dict(fp32=torch.float32, fp64=torch.float64).get(precision, torch.bfloat16)
    with projection_precision(
        reduced=precision == "bf16_default", fp32=precision in ("fp32", "fp64")
    ):
        pair = tuple(
            torch.nn.functional.linear(x.to(dtype), projection.weight.to(dtype)).bfloat16()
            for projection in (attention.k_proj, attention.v_proj)
        )
    n, width = x.shape[1:]
    heads = width // 128
    k, v = (t.reshape(1, n, heads, 128).transpose(1, 2) for t in pair)
    cos, sin = phases
    k = k * cos[:, None] + torch.cat((-k[..., 64:], k[..., :64]), -1) * sin[:, None]
    return k.contiguous(), v.contiguous()
