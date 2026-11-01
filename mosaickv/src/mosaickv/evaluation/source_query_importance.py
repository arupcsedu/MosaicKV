"""Prompt-only diagonal sensitivity bound, not a full output Jacobian.

Each query contributes before reduction. RoPE operator norms use the original
phases, and exact K/V rows have zero source-quantization sensitivity. No teacher
targets, reference answers or model parameter updates are accepted.
"""

import math

import torch


@torch.inference_mode()
def importance(query, key, value, wk, wv, positions, query_positions, phases, exact_rows):
    return _importance(
        query, key, value, wk, wv, positions, query_positions, phases, exact_rows, device_type="cpu"
    )


@torch.inference_mode()
def importance_cuda(query, key, value, wk, wv, positions, query_positions, phases, exact_rows):
    return _importance(
        query,
        key,
        value,
        wk,
        wv,
        positions,
        query_positions,
        phases,
        exact_rows,
        device_type="cuda",
    )


def _probabilities(query, key, value, positions, query_positions):
    d = key.shape[-1]
    scores = (query.float() @ key.float().transpose(-1, -2)).bfloat16() * (1 / math.sqrt(d))
    causal = positions[None, :, None, :] > query_positions[None, None, :, None]
    valid = (~causal).any(-1, keepdim=True)
    scores = scores.masked_fill(causal, float("-inf"))
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    probabilities = scores.float().softmax(-1).bfloat16().float() * valid
    output = (probabilities @ value.float()).bfloat16().float()[0]
    values = value.float()[0]
    distances = (
        output.square().sum(-1)[..., None]
        + values.square().sum(-1)[:, None, :]
        - 2 * (output @ values.transpose(-1, -2))
    ).clamp_min(0)
    return probabilities, distances, int((~valid).sum())


@torch.inference_mode()
def kv_importance_cuda(query, key, value, positions, query_positions, exact_rows):
    if (
        key.device.type != "cuda"
        or key.shape != value.shape
        or query.shape[:2] != key.shape[:2]
        or query.shape[-1] != key.shape[-1]
        or positions.shape != key.shape[1:3]
        or exact_rows.shape != key.shape[1:3]
        or exact_rows.dtype != torch.bool
        or any(
            t.device != key.device or not torch.isfinite(t).all()
            for t in (query, key, value, positions, query_positions, exact_rows)
        )
    ):
        raise ValueError(
            "actual CUDA query, merged/selected KV and original logical positions required"
        )
    a, dist, ignored = _probabilities(query, key, value, positions, query_positions)
    squared = a[0].square() * (~exact_rows)[:, None, :]
    key_weights = torch.einsum(
        "hqn,hqd->hnd", squared * dist / key.shape[-1], query.float()[0].square()
    )
    value_weights = squared.sum(1)[..., None].expand_as(key_weights)
    return (key_weights[None].contiguous(), value_weights[None].contiguous()), dict(
        surrogate="diagonal_post_RoPE_KV_attention_output_sensitivity",
        original_logical_positions=True,
        ignored_no_causal_key_queries=ignored,
        actual_selected_or_merged_values_not_regenerated=True,
    )


def _importance(
    query, key, value, wk, wv, positions, query_positions, phases, exact_rows, *, device_type
):
    h, n, d = key.shape[1:]
    width = h * d
    if (
        query.device.type != device_type
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
        or not torch.equal(positions, torch.arange(n, device=key.device).repeat(h, 1))
        or any(
            t.device != key.device or not torch.isfinite(t).all()
            for t in (query, key, value, wk, wv, positions, query_positions, exact_rows, *phases)
        )
    ):
        raise ValueError(
            "finite CPU full-source MHA geometry, original positions/phases and exact rows required"
        )
    probabilities, distances, ignored = _probabilities(
        query, key, value, positions, query_positions
    )
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
        ignored_no_causal_key_queries=ignored,
    )


@torch.inference_mode()
def future_importance_cuda(query, value, wk, wv, phases, exact_rows, probability, output):
    return _future_importance(
        query, value, wk, wv, phases, exact_rows, probability, output, device_type="cuda"
    )


@torch.inference_mode()
def future_importance_cpu(query, value, wk, wv, phases, exact_rows, probability, output):
    """CPU fixture oracle only, never a substitute for actual CUDA execution."""
    return _future_importance(
        query, value, wk, wv, phases, exact_rows, probability, output, device_type="cpu"
    )


def blend_families(prompt, future, prompt_queries, future_queries):
    if (
        type(prompt_queries) is not int
        or type(future_queries) is not int
        or min(prompt_queries, future_queries) < 1
    ):
        raise ValueError("nonempty independent prompt and draft query families required")
    if (
        prompt.shape != future.shape
        or prompt.device != future.device
        or any(not torch.isfinite(t).all() or (t < 0).any() for t in (prompt, future))
    ):
        raise ValueError("matching finite nonnegative family sensitivities required")
    return 0.5 * prompt / prompt_queries + 0.5 * future / future_queries


def _future_importance(
    query, value, wk, wv, phases, exact_rows, probability, output, *, device_type
):
    h, n, d = value.shape[1:]
    width = h * d
    if (
        value.device.type != device_type
        or d != 128
        or query.shape[:2] != (1, h)
        or query.shape[-1] != d
        or probability.shape != (1, h, query.shape[-2], n)
        or output.shape != query.shape
        or exact_rows.shape != (h, n)
        or exact_rows.dtype != torch.bool
        or phases[0].shape != (1, n, d)
        or phases[1].shape != phases[0].shape
        or wk.shape != (width, width)
        or wv.shape != wk.shape
        or any(
            t.device != value.device or not torch.isfinite(t).all()
            for t in (query, value, wk, wv, *phases, probability, output, exact_rows)
        )
        or (probability < 0).any()
        or (probability.sum(-1) > 1 + 2**-7).any()
    ):
        raise ValueError(
            "actual device-matched forecast probabilities/output and full prompt values required"
        )
    q = query.float()[0]
    v = value.float()[0]
    o = output.float()[0]
    distances = (
        o.square().sum(-1)[..., None]
        + v.square().sum(-1)[:, None, :]
        - 2 * (o @ v.transpose(-1, -2))
    ).clamp_min(0)
    a2 = probability[0].float().square() * (~exact_rows)[:, None, :]
    rho = (phases[0].float().square() + phases[1].float().square()).amax(-1)[0]
    key_coefficient = (a2 * distances * q.square().sum(-1)[..., None] / d).sum(1) * rho[None]
    value_coefficient = a2.sum(1)
    result = key_coefficient.T @ wk.float().reshape(h, d, width).square().sum(
        1
    ) + value_coefficient.T @ wv.float().reshape(h, d, width).square().sum(1)
    if not torch.isfinite(result).all() or (result < 0).any():
        raise RuntimeError("invalid future sensitivity")
    return result[None, None]
