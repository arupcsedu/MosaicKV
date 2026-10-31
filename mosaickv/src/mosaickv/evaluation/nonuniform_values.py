"""Current-request scalar quantization, not model training or official KVQuant.

Four BF16 centroids replace two FP32 affine parameters: identical header bytes.
Fixed eight Lloyd updates start from the old min/max decoder and retain only
actual post-BF16 per-row MSE improvements. No calibration labels or answers.
"""

from dataclasses import replace
from typing import Any

import torch

from mosaickv.evaluation.precision_tier import QuantizedBody, pack


def pack_nonuniform_values(values: Any) -> QuantizedBody:
    if values.ndim != 4 or values.shape[-1] != 128:
        raise ValueError("finite BF16 value rows of width 128 required")
    baseline = pack(values, bits=2, axis=3)
    x = values.float()
    centers = (
        baseline.minimum + baseline.scale * torch.arange(4, device=x.device).float()
    ).bfloat16()
    if not bool(torch.isfinite(centers).all()):
        raise ValueError("finite BF16 centroids required; affine range overflow unsupported")

    def evaluate(c):
        q = (x[..., None] - c.float()[..., None, :]).square().argmin(-1)
        y = c.gather(-1, q).float()
        return q, (x - y).square().sum(-1)

    best_q, best_error = evaluate(centers)
    for _ in range(8):
        proposed = []
        for level in range(4):
            mask = best_q == level
            count = mask.sum(-1)
            mean = torch.where(mask, x, 0).sum(-1) / count.clamp_min(1)
            proposed.append(torch.where(count > 0, mean.bfloat16(), centers[..., level]))
        candidate = torch.stack(proposed, -1).sort(-1).values.contiguous()
        q, error = evaluate(candidate)
        take = error < best_error
        centers = torch.where(take[..., None], candidate, centers)
        best_q = torch.where(take[..., None], q, best_q)
        best_error = torch.where(take, error, best_error)
    shifts = torch.arange(8, device=x.device, dtype=torch.long) * 2
    words = (best_q.reshape(-1, 8) << shifts).sum(-1)
    codes = torch.stack(((words & 255).byte(), ((words >> 8) & 255).byte()), -1).flatten()
    empty = torch.empty(0, dtype=torch.float32, device=x.device)
    body = replace(baseline, codes=codes, minimum=empty, scale=empty, codebook=centers.contiguous())
    if sum(t.numel() * t.element_size() for t in body.tensors()) != sum(
        t.numel() * t.element_size() for t in baseline.tensors()
    ):
        raise RuntimeError("nonuniform codebook changed the encoded payload/header size")
    return body
