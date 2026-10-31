"""Cache-local affine range fitting, not model training or a new attention path.

Uses the existing two-bit packet and decoder. Min/max is always a candidate;
an update is retained only if its actual BF16 reconstruction MSE is lower.
No benchmark output, draft answer, reference, or learned weight is used.
"""

from dataclasses import replace
from typing import Any

import torch

from mosaickv.evaluation.precision_tier import QuantizedBody, pack


def pack_mse_values(values: Any) -> QuantizedBody:
    if values.ndim != 4 or values.shape[-1] != 128:
        raise ValueError("BF16 value rows of feature width 128 required")
    baseline = pack(values, bits=2, axis=3)
    x = values.float().reshape(*values.shape[:-1], 1, 128)
    low, high = x.amin(-1), x.amax(-1)
    mean = x.mean(-1)
    best_min, best_scale = baseline.minimum, baseline.scale

    def evaluate(minimum, scale):
        q = ((x - minimum[..., None]) / scale[..., None]).round().clamp(0, 3)
        y = (q * scale[..., None] + minimum[..., None]).bfloat16().float()
        return q, (x - y).square().sum(-1)

    best_q, best_error = evaluate(best_min, best_scale)

    def consider(minimum, scale):
        nonlocal best_q, best_min, best_scale, best_error
        q, error = evaluate(minimum, scale)
        valid = torch.isfinite(error) & torch.isfinite(scale) & (scale > 0)
        take = valid & (error < best_error)
        best_q = torch.where(take[..., None], q, best_q)
        best_min = torch.where(take, minimum, best_min)
        best_scale = torch.where(take, scale, best_scale)
        best_error = torch.where(take, error, best_error)

    for radius in (0.5, 0.625, 0.75, 0.875):
        minimum = mean + radius * (low - mean)
        maximum = mean + radius * (high - mean)
        scale = (maximum - minimum) / 3
        consider(minimum, torch.where(scale == 0, torch.ones_like(scale), scale))
    # Fixed two coordinate updates, not an open-ended optimization loop.
    for _ in range(2):
        qmean = best_q.mean(-1)
        centered = best_q - qmean[..., None]
        variance = centered.square().sum(-1)
        scale = (centered * (x - mean[..., None])).sum(-1) / variance.clamp_min(1e-20)
        consider(mean - scale * qmean, scale)
    shifts = torch.arange(8, device=values.device, dtype=torch.long) * 2
    words = (best_q.long().reshape(-1, 8) << shifts).sum(-1)
    codes = torch.stack(((words & 255).byte(), ((words >> 8) & 255).byte()), -1).flatten()
    return replace(
        baseline, codes=codes, minimum=best_min.contiguous(), scale=best_scale.contiguous()
    )
