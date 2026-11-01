"""Bounded training-free affine quantizer refit on current prompt tensors only.

Two alternating least-squares/assignment steps, with per-group BF16-error
rollback. No model weights, task annotations or reference outputs are inputs.
Separate explicit CPU and experimental CUDA entry points. CUDA requires the
registered numerical and model-level verification before any performance claim.
"""

from dataclasses import replace

import torch

from mosaickv.evaluation.precision_tier import QuantizedBody, unpack_cpu


@torch.inference_mode()
def refit(body, reference, *, exact_rows=None, importance=None):
    return _refit(body, reference, exact_rows=exact_rows, importance=importance, device_type="cpu")


@torch.inference_mode()
def refit_cuda(body, reference, *, exact_rows=None, importance=None):
    return _refit(body, reference, exact_rows=exact_rows, importance=importance, device_type="cuda")


def _refit(body, reference, *, exact_rows, importance, device_type):
    if (
        not isinstance(body, QuantizedBody)
        or body.codebook is not None
        or body.bits not in (2, 3, 4, 8)
        or body.axis not in (2, 3)
        or body.group != 128
        or reference.device.type != device_type
        or body.codes.device != reference.device
        or body.minimum.device != reference.device
        or body.scale.device != reference.device
        or reference.dtype != torch.bfloat16
        or tuple(reference.shape) != body.shape
        or body.minimum.dtype != torch.float32
        or body.scale.dtype != torch.float32
        or not torch.isfinite(reference).all()
        or not torch.isfinite(body.minimum).all()
        or not torch.isfinite(body.scale).all()
        or not (body.scale > 0).all()
    ):
        raise ValueError(
            "finite compatible explicit-device BF16 data and affine FP32 quantizer required"
        )
    if exact_rows is not None and (
        exact_rows.shape != reference.shape[1:3]
        or exact_rows.dtype != torch.bool
        or exact_rows.device != reference.device
    ):
        raise ValueError("same-shape CPU per-head exact-row mask required")
    if importance is not None and (
        importance.shape != reference.shape
        or importance.device != reference.device
        or not importance.is_floating_point()
        or not torch.isfinite(importance).all()
        or (importance < 0).any()
    ):
        raise ValueError("finite nonnegative same-shape CPU importance required")
    raw = body.codes.reshape(-1, body.bits).long()
    words = sum(raw[:, i] << (8 * i) for i in range(body.bits))
    shift = torch.arange(8, dtype=torch.long, device=reference.device) * body.bits
    q = ((words[:, None] >> shift) & ((1 << body.bits) - 1)).double()
    layout = list(body.shape)
    length = layout.pop(body.axis)
    layout.append(length)
    q = q.reshape(layout)
    y = reference.double().movedim(body.axis, -1).contiguous()
    active = (
        torch.ones_like(reference, dtype=torch.bool)
        if exact_rows is None
        else (~exact_rows)[None, :, :, None].expand_as(reference)
    )
    active = active.movedim(body.axis, -1).contiguous()
    weights = (
        active.double()
        if importance is None
        else importance.double().movedim(body.axis, -1).contiguous() * active
    )
    pad = (-length) % 128
    if pad:
        q = torch.nn.functional.pad(q, (0, pad))
        y = torch.nn.functional.pad(y, (0, pad))
        active = torch.nn.functional.pad(active, (0, pad), value=False)
        weights = torch.nn.functional.pad(weights, (0, pad))
    q = q.reshape(*q.shape[:-1], -1, 128)
    y = y.reshape_as(q)
    active = active.reshape_as(q)
    weights = weights.reshape_as(q)
    count = active.sum(-1)
    minimum = body.minimum.double()
    scale = body.scale.double()

    def sse(codes, m, s, *, unweighted=False):
        predicted = (
            (codes.float() * s.float()[..., None] + m.float()[..., None]).bfloat16().double()
        )
        return ((predicted - y).square() * (active if unweighted else weights)).sum(-1)

    before = sse(q, minimum, scale)
    input_before = sse(q, minimum, scale, unweighted=True)
    accepted = []
    for _ in range(2):
        mass = weights.sum(-1)
        den = torch.where(mass > 0, mass, torch.ones_like(mass))
        mean_q = (q * weights).sum(-1) / den
        mean_y = (y * weights).sum(-1) / den
        centered_q = q - mean_q[..., None]
        centered_y = y - mean_y[..., None]
        variance = (centered_q.square() * weights).sum(-1)
        covariance = (centered_q * centered_y * weights).sum(-1)
        new_scale = (covariance / variance.clamp_min(1e-30)).float().double()
        new_minimum = (mean_y - new_scale * mean_q).float().double()
        valid = (
            (count > 1)
            & (mass > 0)
            & (variance > 0)
            & torch.isfinite(new_scale)
            & (new_scale > 0)
            & torch.isfinite(new_minimum)
        )
        new_scale = torch.where(valid, new_scale, scale)
        new_minimum = torch.where(valid, new_minimum, minimum)
        proposed = (
            ((y - new_minimum[..., None]) / new_scale[..., None])
            .round()
            .clamp(0, (1 << body.bits) - 1)
        )
        proposed = torch.where(active, proposed, q)
        old_sse = sse(q, minimum, scale)
        new_sse = sse(proposed, new_minimum, new_scale)
        keep = valid & (new_sse < old_sse)
        accepted.append(int(keep.sum()))
        q = torch.where(keep[..., None], proposed, q)
        minimum = torch.where(keep, new_minimum, minimum)
        scale = torch.where(keep, new_scale, scale)
    values = q.flatten(-2)[..., :length].contiguous().long().reshape(-1, 8)
    words = (values << shift).sum(-1)
    codes = torch.stack(
        tuple(((words >> (8 * i)) & 255).to(torch.uint8) for i in range(body.bits)), -1
    ).flatten()
    candidate = replace(
        body, codes=codes, minimum=minimum.float().contiguous(), scale=scale.float().contiguous()
    )
    after = sse(q, minimum, scale)
    if (after > before + 1e-12).any():
        raise RuntimeError("BF16 quantizer-error rollback failed")
    if sum(t.numel() * t.element_size() for t in candidate.tensors()) != sum(
        t.numel() * t.element_size() for t in body.tensors()
    ):
        raise RuntimeError("affine refit changed payload bytes")
    if not torch.isfinite(after).all() or (
        device_type == "cpu" and not torch.isfinite(unpack_cpu(candidate)).all()
    ):
        raise RuntimeError("nonfinite affine fit")
    return candidate, dict(
        steps=2,
        accepted_groups=accepted,
        active_source_SSE_before=float(input_before.sum()),
        active_source_SSE_after=float(sse(q, minimum, scale, unweighted=True).sum()),
        fitted_objective_before=float(before.sum()),
        fitted_objective_after=float(after.sum()),
        importance_weighted=importance is not None,
        same_payload_bytes=True,
        model_weights_unchanged=True,
        no_benchmark_label_access=True,
        CPU_only=device_type == "cpu",
    )
