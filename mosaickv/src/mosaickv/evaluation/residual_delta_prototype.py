"""Explicit parent-residual prototype variant, not absolute K/V compaction.

Average only the parent's reconstruction error, in phase-neutral key space.
This replaces an existing prototype overlay payload; it does NOT retire the
shared source representation. No gradients, model updates or labels are used.
"""

from __future__ import annotations

import torch


def construct(original, parent, phases, members, anchor, weights):
    if (
        len(original) != 2
        or len(parent) != 2
        or original[0].ndim != 2
        or original[0].shape[-1] % 2
        or any(
            t.shape != original[0].shape
            or t.dtype != torch.bfloat16
            or t.device != original[0].device
            for t in (*original, *parent)
        )
        or tuple(sorted(set(members))) != tuple(members)
        or not 2 <= len(members) <= 8
        or min(members) < 0
        or max(members) >= len(original[0])
        or type(anchor) is not int
        or not 0 <= anchor < len(original[0])
        or len(weights) != len(members)
    ):
        raise ValueError("aligned BF16 parent/original and bounded original positions required")
    cos, sin = phases
    if cos.shape != original[0].shape or sin.shape != cos.shape:
        raise ValueError("actual complete original rotary phases required")
    weight = torch.as_tensor(weights, device=original[0].device, dtype=torch.float32)
    if not bool(torch.isfinite(weight).all()) or not bool((weight > 0).all()):
        raise ValueError("finite strictly positive query-demand weights required")
    weight = (weight / weight.sum()).reshape(-1, 1)
    index = torch.tensor(members, device=original[0].device)
    delta = tuple(
        a.index_select(0, index).float() - b.index_select(0, index).float()
        for a, b in zip(original, parent, strict=True)
    )
    c, s = cos.index_select(0, index).float(), sin.index_select(0, index).float()
    half = c.shape[-1] // 2

    def rotate(x):
        return torch.cat((-x[..., half:], x[..., :half]), -1)

    neutral = (delta[0] * c - rotate(delta[0]) * s) / (c.square() + s.square()).clamp_min(1e-12)
    key = (neutral * weight).sum(0, keepdim=True)
    value = (delta[1] * weight).sum(0, keepdim=True)
    at_anchor = key * cos[anchor].float() + rotate(key) * sin[anchor].float()
    pair = at_anchor.to(original[0].dtype), value.to(original[1].dtype)
    if any(not bool(torch.isfinite(t).all()) for t in pair):
        raise ValueError("nonfinite residual prototype")
    dispersion = float(
        ((neutral - key).square().sum(-1) + (delta[1] - value).square().sum(-1)).mean()
    )
    return pair, dispersion


def study(payload):
    """CPU-only reuse of consumed evaluation captures; no model or labels."""
    from mosaickv.evaluation.storage_component_gate import precision_evidence
    from mosaickv.evaluation.storage_six_overlay import LayerOverlay

    if not payload.get("evaluation_only") or payload.get("labels_accessed") is not False:
        raise ValueError("label-free evaluation-only capture required")
    ids = tuple(payload["evidence"]["prototype_members"])
    # Uniform, preregistered weights: do not fit to observed error outcomes.
    pair, dispersion = construct(
        payload["original_pair"],
        payload["parent_pair"],
        payload["phases"],
        ids,
        ids[-1],
        (1.0,) * len(ids),
    )
    subset = tuple(t[(*ids, ids[-1]), :].clone() for t in payload["phases"])
    overlay = LayerOverlay(
        head=0,
        positions=ids,
        exact_ids=tuple(payload["evidence"]["selected_extra"]),
        exact_pair=payload["overlay"]["exact_pair"],
        prototype_pair=pair,
        phases=subset,
        residual_pair=tuple(t[list(ids)].clone() for t in payload["original_pair"]),
        anchor=ids[-1],
        dispersion=dispersion,
        evidence={},
    )
    parent = tuple(t.clone() for t in payload["parent_pair"])
    # Actual BF16 addition, matching the proposed live reader, not FP64 fitting.
    correction = overlay.member_keys, pair[1].expand(len(ids), -1)
    effective = tuple(
        (a[list(ids)] + b).contiguous() for a, b in zip(parent, correction, strict=True)
    )
    candidate = dict(
        payload,
        overlay=dict(payload["overlay"], member_keys=effective[0], prototype_value=effective[1]),
    )
    result = precision_evidence(candidate)
    result.update(
        representation="parent_residual_delta_prototype_uniform_weights_v1",
        parent_source_retired=False,
        no_source_memory_savings_claim=True,
        construction_uses_all_prompt_capture_no_benchmark_labels=True,
        dispersion=dispersion,
    )
    return result
