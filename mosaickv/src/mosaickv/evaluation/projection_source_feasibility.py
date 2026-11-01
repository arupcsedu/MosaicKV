"""CPU-only projection-source numerical feasibility, not a GPU/cache runtime."""

import math

import torch


def project(source, weight, *, heads, phases=None):
    if (
        source.device.type != "cpu"
        or source.ndim != 3
        or source.shape[0] != 1
        or source.dtype != torch.bfloat16
        or weight.device.type != "cpu"
        or weight.shape != (heads * 128, source.shape[-1])
        or not torch.isfinite(source).all()
        or not torch.isfinite(weight).all()
    ):
        raise ValueError("finite CPU BF16 projection input and compatible frozen weight required")
    # Consistent CPU numerical oracle, NOT native GPU CUBLAS parity evidence.
    result = torch.nn.functional.linear(source.float(), weight.float()).bfloat16()
    result = result.reshape(1, source.shape[1], heads, 128).transpose(1, 2)
    if phases is not None:
        cos, sin = phases
        if cos.shape != (1, source.shape[1], 128) or sin.shape != cos.shape:
            raise ValueError("original full logical phases required")
        rotated = torch.cat((-result[..., 64:], result[..., :64]), -1)
        result = result * cos[:, None] + rotated * sin[:, None]
    return result.contiguous()


def attention_output(query, key, value, positions, query_positions):
    if (
        query.ndim != 4
        or key.shape != value.shape
        or query.shape[:2] != key.shape[:2]
        or query.shape[-1] != key.shape[-1]
        or positions.shape != key.shape[1:3]
        or query_positions.shape != (query.shape[2],)
        or any(t.device.type != "cpu" for t in (query, key, value, positions, query_positions))
        or (positions[:, 1:] <= positions[:, :-1]).any()
    ):
        raise ValueError("consistent CPU heads, original positions and causal queries required")
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)).bfloat16()
    scores = scores * (1 / math.sqrt(key.shape[-1]))
    mask = positions[None, :, None, :] > query_positions[None, None, :, None]
    scores = scores.masked_fill(mask, float("-inf"))
    if torch.isneginf(scores).all(-1).any():
        raise ValueError("every prompt query needs at least one causal source position")
    probabilities = scores.float().softmax(-1).bfloat16()
    return torch.matmul(probabilities.float(), value.float()).bfloat16().float()


def errors(reference, candidate):
    if reference.shape != candidate.shape or not torch.isfinite(candidate).all():
        raise ValueError("finite same-shape outputs required")
    diff = (reference.double() - candidate.double()).square().sum(-1)
    norms = reference.double().square().sum(-1)
    relative = diff / norms.clamp_min(1e-12)
    return dict(
        error_sse=float(diff.sum()),
        reference_sse=float(norms.sum()),
        relative_per_query_head=relative.flatten().tolist(),
    )


def aggregate(records):
    if not records:
        raise ValueError("nonempty layer evidence required")
    values = torch.tensor(
        [v for r in records for v in r["relative_per_query_head"]], dtype=torch.float64
    )
    return dict(
        relative_mse=sum(r["error_sse"] for r in records)
        / max(sum(r["reference_sse"] for r in records), 1e-12),
        p95_per_query_head_relative_mse=float(torch.quantile(values, 0.95)),
        query_head_count=len(values),
        layer_count=len(records),
    )
