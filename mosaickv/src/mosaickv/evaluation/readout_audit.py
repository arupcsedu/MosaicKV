"""CPU-only readout diagnostics, not new model execution or acceptance rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReadoutAuditPolicy:
    minimum_rank_correlation: float = 0.3
    minimum_rank_gain_over_reconstruction: float = 0.1
    minimum_positive_precision: float = 0.75
    minimum_positive_predictions: int = 4


SCORE_PROVENANCE = {
    "head_reconstruction": "pre_probe_FullKV_queries_original_values_and_memberships",
    "projected_residual": "pre_probe_FullKV_and_anchor_states_and_predicted_local_projection",
    "identity_tail_readout_linear": "pre_probe_identity_downstream_Jacobian_NOT_measured_transfer",
    "identity_tail_readout_quadratic": "pre_probe_identity_tail_NOT_measured_downstream_transfer",
    "actual_final_hidden": "post_probe_requires_candidate_final_hidden_state",
    "actual_normalized_hidden": "post_probe_requires_candidate_final_hidden_state",
    "actual_readout_linear": "post_probe_requires_candidate_final_hidden_state",
    "actual_readout_quadratic": "post_probe_requires_candidate_final_hidden_state",
    "actual_cpu_readout_kl": "post_probe_requires_candidate_final_hidden_state",
}
PROSPECTIVE_SCORES = tuple(k for k, v in SCORE_PROVENANCE.items() if v.startswith("pre_probe"))


class FrozenReadout:
    """Frozen RMSNorm + vocabulary projection; analytic derivatives, no training.

    FP64 sideband arithmetic is explicitly distinct from the original FP16 GPU
    forward and registered FP32 guard. It cannot change a rejection to a pass.
    """

    def __init__(self, norm_weight: Any, lm_weight: Any, epsilon: float):
        import torch

        if (
            norm_weight.device.type != "cpu"
            or lm_weight.device.type != "cpu"
            or norm_weight.ndim != 1
            or lm_weight.ndim != 2
            or lm_weight.shape[1] != norm_weight.numel()
            or not math.isfinite(epsilon)
            or epsilon <= 0
            or not torch.isfinite(norm_weight).all()
            or not torch.isfinite(lm_weight).all()
        ):
            raise ValueError("readout audit requires compatible finite CPU weights and epsilon")
        self.norm = norm_weight.detach().double().clone()
        self.weight = lm_weight.detach().double().clone()
        self.epsilon = epsilon

    def normalize(self, hidden):
        import torch

        if (
            hidden.device.type != "cpu"
            or hidden.shape[-1] != self.norm.numel()
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError("invalid readout hidden states")
        x = hidden.detach().double()
        return x * (x.square().mean(-1, keepdim=True) + self.epsilon).rsqrt() * self.norm

    def logits(self, hidden):
        return self.normalize(hidden) @ self.weight.T

    def norm_jvp(self, anchor, delta):
        if anchor.shape != delta.shape:
            raise ValueError("readout directional derivative needs matching shapes")
        x, d = anchor.detach().double(), delta.detach().double()
        scale = (x.square().mean(-1, keepdim=True) + self.epsilon).rsqrt()
        return self.norm * (scale * d - x * scale.pow(3) * (x * d).mean(-1, keepdim=True))

    def kl_gradient(self, full_logits, anchor_hidden):
        """Exact analytic gradient of the FP64 sideband KL wrt anchor hidden."""
        import torch

        p = torch.softmax(full_logits.double(), -1)
        q = torch.softmax(self.logits(anchor_hidden), -1)
        x = anchor_hidden.detach().double()
        scale = (x.square().mean(-1, keepdim=True) + self.epsilon).rsqrt()
        grad_normed = ((q - p) @ self.weight) * self.norm
        return scale * grad_normed - x * scale.pow(3) * (x * grad_normed).mean(-1, keepdim=True)

    def directional_gains(self, full_logits, anchor_hidden, delta):
        import torch

        p = torch.softmax(full_logits.double(), -1)
        q = torch.softmax(self.logits(anchor_hidden), -1)
        dz = self.norm_jvp(anchor_hidden, delta) @ self.weight.T
        linear = -((q - p) * dz).sum(-1)
        # Fisher/logit-space quadratic term. Does NOT include RMSNorm's
        # second derivative; it is an approximation, not a KL upper bound.
        variance = (q * dz.square()).sum(-1) - (q * dz).sum(-1).square()
        return linear, linear - 0.5 * variance


def kl_divergence(full_logits, candidate_logits):
    import torch

    lp, lq = (
        torch.log_softmax(full_logits.double(), -1),
        torch.log_softmax(candidate_logits.double(), -1),
    )
    return (lp.exp() * (lp - lq)).sum(-1)


def squared_error_gain(reference, anchor, candidate):
    return (
        (anchor.double() - reference.double()).square()
        - (candidate.double() - reference.double()).square()
    ).mean(-1)


def projected_residual_gain(full_residual, anchor_residual, predicted_delta):
    """Uses no candidate-forward tensors. Positive means predicted improvement."""
    error = anchor_residual.double() - full_residual.double()
    delta = predicted_delta.double()
    return -(2 * error * delta + delta.square()).mean(-1)


def rank_correlation(left, right):
    """Spearman with average ranks for ties; None for undefined constant ranks."""
    if (
        len(left) != len(right)
        or len(left) < 2
        or not all(math.isfinite(x) for x in [*left, *right])
    ):
        raise ValueError("rank audit requires matching finite nonempty observations")

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            for index in order[i:j]:
                out[index] = (i + j - 1) / 2
            i = j
        return out

    a, b = ranks(left), ranks(right)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    denominator = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return None if denominator == 0 else numerator / denominator


def audit_ranking(rows, policy: ReadoutAuditPolicy):
    """Fixed-prefix descriptive screening, never statistical generalization.

    Positive outcomes use the original registered KL tolerance. No candidate
    failure-prefix selection is used for these prospective comparisons.
    """
    from mosaickv.config import DraftAllocationGuardConfig

    guard = DraftAllocationGuardConfig()
    if not rows or any(row["prefix"] != 1 for row in rows):
        raise ValueError("prospective ranking audit requires prefix one for every proposal")
    actual = [row["registered_kl_gain"] for row in rows]
    baseline = rank_correlation([row["head_reconstruction"] for row in rows], actual)
    result = {}
    for name in PROSPECTIVE_SCORES:
        predicted = [row[name] for row in rows]
        rho = rank_correlation(predicted, actual)
        positive = [r for r in rows if r[name] > 0]
        correct = sum(
            r["registered_kl_gain"]
            > guard.absolute_tolerance + guard.relative_tolerance * abs(r["registered_kl_before"])
            for r in positive
        )
        precision = None if not positive else correct / len(positive)
        useful = (
            name != "head_reconstruction"
            and rho is not None
            and baseline is not None
            and rho >= policy.minimum_rank_correlation
            and rho >= baseline + policy.minimum_rank_gain_over_reconstruction
            and len(positive) >= policy.minimum_positive_predictions
            and precision is not None
            and precision >= policy.minimum_positive_precision
        )
        result[name] = dict(
            rank_correlation=rho,
            reconstruction_rank_correlation=baseline,
            positive_predictions=len(positive),
            correctly_predicted_KL_improvements=correct,
            positive_precision=precision,
            descriptive_signal=bool(useful),
        )
    return result
