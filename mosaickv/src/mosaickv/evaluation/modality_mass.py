"""Label-blind draft-conditioned visual/text normalization diagnostic.

No new/replacement K/V, RoPE averaging, donor eviction, or trained parameters.
The log-odds correction is only a query forecast, not a guarantee of fidelity.
Every text token (including mandatory delimiter) and generated token has zero
bias. Persistent FP32 bias and validity tensors are charged to cache storage.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def estimate(attention_steps, memberships, text_positions, *, strength):
    if type(strength) not in (float, int) or strength not in (0.0, 0.5, 1.0):
        raise ValueError("registered zero/half/full correction strength required")
    if len(attention_steps) != 4 or not memberships or not text_positions:
        raise ValueError("four actual draft steps, source memberships, and text map required")
    layers = len(memberships)
    if any(len(s) != layers for s in attention_steps):
        raise ValueError("complete draft attention for every source layer required")
    n = attention_steps[0][0].shape[-1] - 1
    text_positions = tuple(text_positions)
    if (
        tuple(sorted(set(text_positions))) != text_positions
        or text_positions[-1] != n - 1
        or text_positions[0] < 0
        or len(text_positions) >= n
    ):
        raise ValueError(
            "authoritative text map including mandatory delimiter and visual grid required"
        )
    biases, records = [], []
    for layer, heads in enumerate(memberships):
        if not heads or len({len(h) for h in heads}) != 1:
            raise ValueError("uniform physical head capacity required")
        steps, identified = [], []
        device = attention_steps[0][layer].device
        text = torch.zeros(n, device=device, dtype=torch.bool)
        text[list(text_positions)] = True
        selected = torch.zeros(len(heads), n, device=device, dtype=torch.bool)
        visual_slots = []
        counts = []
        for h, row in enumerate(heads):
            valid = tuple(p for p in row if p is not None)
            if (
                not valid
                or tuple(sorted(set(valid))) != valid
                or valid[0] < 0
                or valid[-1] != n - 1
                or any(type(p) is not int for p in valid)
                or any(p is not None for p in row[len(valid) :])
            ):
                raise ValueError("sorted original memberships and tail-only padding required")
            selected[h, list(valid)] = True
            counts.append(len(valid))
            visual_slots.append(tuple(p is not None and p not in text_positions for p in row))
        for step, attention in enumerate(attention_steps):
            p = attention[layer]
            if (
                p.shape != (1, len(heads), 1, n + step + 1)
                or p.device != device
                or not bool(torch.isfinite(p).all())
                or bool((p < 0).any())
            ):
                raise ValueError(
                    "finite, batch-one MHA draft attention with original logical positions required"
                )
            p = p[0, :, 0, :n].detach().float()
            full_visual, full_text = p[:, ~text].sum(-1), p[:, text].sum(-1)
            kept_visual = (p * (selected & ~text)).sum(-1)
            kept_text = (p * (selected & text)).sum(-1)
            masses = torch.stack((full_visual, full_text, kept_visual, kept_text))
            valid_mass = (masses > 0).all(0)
            # Zero mass is not evidence for an infinite multiplier. Such heads
            # receive the unchanged zero correction, explicitly marked unknown.
            logs = masses.clamp_min(torch.finfo(torch.float32).tiny).log()
            correction = logs[0] - logs[1] - logs[2] + logs[3]
            steps.append(torch.where(valid_mass, correction, 0.0))
            identified.append(valid_mass)
        individual = torch.stack(steps)
        valid_queries = torch.stack(identified)
        # Average log odds, NOT average distributions: each query/head is kept.
        correction = individual.mean(0).clamp(-math.log(2), math.log(2)) * strength
        correction = torch.where(valid_queries.all(0), correction, 0.0)
        mask = torch.tensor(visual_slots, device=device, dtype=torch.bool)
        bias = (mask * correction[:, None]).float().contiguous()
        if not torch.isfinite(bias).all():
            raise RuntimeError("nonfinite calibrated mask")
        biases.append(bias)
        records.append(
            dict(
                layer=layer,
                per_draft_query_log_odds_error=[
                    [float(v) if ok else None for v, ok in zip(row, valid, strict=True)]
                    for row, valid in zip(
                        individual.cpu().tolist(), valid_queries.cpu().tolist(), strict=True
                    )
                ],
                chosen_visual_bias=correction.cpu().tolist(),
                predicted_log_odds_error_after=[
                    [float(v) if ok else None for v, ok in zip(row, valid, strict=True)]
                    for row, valid in zip(
                        (individual - correction[None]).cpu().tolist(),
                        valid_queries.cpu().tolist(),
                        strict=True,
                    )
                ],
                unidentifiable_heads_bypassed=int((~valid_queries.all(0)).sum()),
                valid_prompt_counts=counts,
                nonzero_entries=int(torch.count_nonzero(bias)),
                text_and_mandatory_bias_zero=True,
                generated_history_bias_zero=True,
                query_forecast_not_actual_future_guarantee=True,
            )
        )
    return tuple(biases), records


@dataclass(frozen=True)
class CalibratedFormat:
    base: Any
    log_mass_bias: Any
    valid_prompt_counts: Any
    kind: str = "common_draft_modality_log_odds_diagnostic"

    def __post_init__(self):
        if (
            self.log_mass_bias.shape != self.base.shape[1:3]
            or self.log_mass_bias.dtype != torch.float32
            or not self.log_mass_bias.is_contiguous()
            or self.valid_prompt_counts.dtype != torch.int16
            or self.valid_prompt_counts.shape != (self.base.shape[1],)
            or self.log_mass_bias.device != self.base.private()[0].device
            or self.valid_prompt_counts.device != self.log_mass_bias.device
            or not bool(torch.isfinite(self.log_mass_bias).all())
            or bool((self.log_mass_bias.abs() > math.log(2) + 1e-7).any())
            or bool(
                (
                    (self.valid_prompt_counts < 1)
                    | (self.valid_prompt_counts > self.base.shape[-2])
                ).any()
            )
        ):
            raise ValueError("finite bounded FP32 bias and original validity counts required")

    @property
    def shape(self):
        return self.base.shape

    def private(self):
        return (*self.base.private(), self.log_mass_bias, self.valid_prompt_counts)

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill(self, attention, targets, *, direct):
        self.base.fill(attention, targets, direct=direct)

    def materialize(self, attention):
        return self.base.materialize(attention)


def wrap(formats, biases, memberships):
    result = []
    for base, bias, heads in zip(formats, biases, memberships, strict=True):
        counts = torch.tensor(
            [sum(p is not None for p in h) for h in heads], device=bias.device, dtype=torch.int16
        )
        if hasattr(base, "valid_prompt_counts"):
            if not torch.equal(counts, base.valid_prompt_counts):
                raise ValueError("calibration cannot change padding membership")
            counts = base.valid_prompt_counts
        result.append(CalibratedFormat(base, bias, counts))
    return tuple(result)


def gpu_controls():
    from mosaickv.evaluation.completion_masks import build_mask

    records = []
    for dtype in (torch.bfloat16, torch.float32):
        counts = torch.tensor([3, 5], dtype=torch.int16)
        bias = torch.tensor([[0.25, 0.0, -0.25, 0.0, 0.0], [-0.5, 0.0, 0.5, 0.0, 0.0]])
        old = torch.zeros(1, 1, 1, 11, dtype=dtype)
        old[..., 0] = -torch.inf
        expected = build_mask(counts, 9, 5, dtype, old, log_mass_bias=bias)
        actual = build_mask(counts.cuda(), 9, 5, dtype, old.cuda(), log_mass_bias=bias.cuda())
        if not torch.equal(actual.cpu(), expected):
            raise RuntimeError("calibrated CUDA mask differs from CPU oracle")
        records.append(
            dict(dtype=str(dtype), CPU_CUDA_equal=True, history_zero=True, padding_excluded=True)
        )
    return records
