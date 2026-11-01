"""Prompt-only diagonal sensitivity bound, not a full output Jacobian.

Each query contributes before reduction. RoPE operator norms use the original
phases, and exact K/V rows have zero source-quantization sensitivity. No teacher
targets, reference answers or model parameter updates are accepted.
"""

import math

import torch


@torch.inference_mode()
def importance(query, key, value, wk, wv, positions, query_positions, phases, exact_rows):
    h, n, d = key.shape[1:]
    width = h * d
    if (
        query.device.type != "cpu"
        or key.shape != value.shape
        or d != 128
        or query.shape[:2] != (1, h)
        or query.shape[-1] != d
        or wk.shape != (width, width)
        or wv.shape != wk.shape
        or positions.shape != (h, n)
        or query_positions.shape != (query.shape[-2],)
        or exact_rows.shape != (h, n)
        or exact_rows.dtype != torch.bool
        or phases[0].shape != (1, n, d)
        or phases[1].shape != phases[0].shape
        or not torch.equal(phases[0][..., :64], phases[0][..., 64:])
        or not torch.equal(phases[1][..., :64], phases[1][..., 64:])
        or not torch.equal(positions, torch.arange(n).repeat(h, 1))
        or any(
            t.device.type != "cpu" or not torch.isfinite(t).all()
            for t in (query, key, value, wk, wv, *phases)
        )
    ):
        raise ValueError(
            "finite CPU full-source MHA geometry, original positions/phases and exact rows required"
        )
    scores = (query.float() @ key.float().transpose(-1, -2)).bfloat16() * (1 / math.sqrt(d))
    causal = positions[None, :, None, :] > query_positions[None, None, :, None]
    probabilities = scores.masked_fill(causal, float("-inf")).float().softmax(-1).bfloat16().float()
    output = (probabilities @ value.float()).bfloat16().float()[0]
    values = value.float()[0]
    distances = (
        output.square().sum(-1)[..., None]
        + values.square().sum(-1)[:, None, :]
        - 2 * (output @ values.transpose(-1, -2))
    ).clamp_min(0)
    a2 = probabilities[0].square() * (~exact_rows)[:, None, :]
    qnorm = query.float()[0].square().sum(-1)
    # ||R_i^T q||^2 <= max_j(cos_ij^2+sin_ij^2) ||q||^2.
    rope_bound = (phases[0].float().square() + phases[1].float().square()).amax(-1)[0]
    key_coefficient = (a2 * distances * qnorm[..., None] / d).sum(1) * rope_bound[None]
    value_coefficient = a2.sum(1)
    key_norm = wk.float().reshape(h, d, width).square().sum(1)
    value_norm = wv.float().reshape(h, d, width).square().sum(1)
    result = key_coefficient.T @ key_norm + value_coefficient.T @ value_norm
    if not torch.isfinite(result).all() or (result < 0).any():
        raise RuntimeError("invalid source sensitivity")
    return result[None, None], dict(
        surrogate="diagonal_RoPE_safe_attention_output_sensitivity_upper_bound",
        not_full_Jacobian=True,
        separate_queries_before_reduction=True,
        original_RoPE_phases_used=True,
        protected_KV_contributions_zero=True,
        no_persistent_extra_metadata=True,
    )
