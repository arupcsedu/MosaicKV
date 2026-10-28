"""Bounded per-request cache least squares; no model training or labels.

Experimental protected attention-matching variant. Old SVD fits and their
negative artifacts are unchanged. Value balls are enforced AT EVERY ITERATION,
not by accepting unbounded fits and subsequently increasing the safety limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mosaickv.prototypes.attention_compaction import (
    AttentionCompactionPolicy,
    fit_batched_mass_bias,
)


@dataclass(frozen=True)
class ConstrainedValuePolicy:
    iterations: int = 256
    ridge_multiplier: float = 1.0
    maximum_delta_norm_ratio: float | None = None

    def __post_init__(self) -> None:
        if type(self.iterations) is not int or not 1 <= self.iterations <= 1024:
            raise ValueError("constrained value iterations must be an integer in [1,1024]")
        if (
            type(self.ridge_multiplier) not in (int, float)
            or not math.isfinite(self.ridge_multiplier)
            or self.ridge_multiplier < 1
        ):
            raise ValueError("ridge multiplier must be finite and >=1; it cannot weaken ridge")
        if self.maximum_delta_norm_ratio is not None and (
            type(self.maximum_delta_norm_ratio) not in (int, float)
            or not math.isfinite(self.maximum_delta_norm_ratio)
            or not 0 <= self.maximum_delta_norm_ratio <= 1
        ):
            raise ValueError("delta norm ratio must be None or finite in [0,1]")


@dataclass(frozen=True)
class ComponentFit:
    values: Any
    log_bias: Any
    admitted: Any
    attempted_values: Any
    attempted_log_bias: Any
    diagnostics: dict[str, Any]


def attention_output(
    attention: Any, values: Any, log_bias: Any, history_mass: Any, history_numerator: Any
) -> Any:
    import torch

    a, v, w = attention.double(), values.double(), log_bias.double().exp()
    mass = torch.bmm(a, w[:, :, None]).squeeze(-1) + history_mass.double()
    if not torch.isfinite(mass).all() or (mass <= 1e-12).any():
        raise ValueError("compaction evidence has insufficient/nonfinite query mass")
    numerator = torch.bmm(a, v * w[:, :, None]) + history_numerator.double()
    return numerator / mass[:, :, None]


def solve_bounded_values(
    matrix: Any,
    residual: Any,
    original: Any,
    protected: Any,
    core: AttentionCompactionPolicy,
    policy: ConstrainedValuePolicy,
) -> tuple[Any, dict[str, Any]]:
    """min .5||X*D-R||² + .5*lambda||D||², ||V0[j]+D[j]||<=4||V0[j]||.

    `matrix` is normalized attention, `residual` excludes original V/history.
    Protected corrections are exactly zero. Fixed projected FISTA retains the
    best feasible regularized objective, with the zero correction as a baseline.
    All work is FP64; actual stored-dtype constraints are checked independently.
    """
    import torch

    if (
        not isinstance(policy, ConstrainedValuePolicy)
        or not isinstance(core, AttentionCompactionPolicy)
        or not core.enabled
        or matrix.ndim != 3
        or original.ndim != 3
        or residual.ndim != 3
        or min(matrix.shape) < 1
        or min(original.shape) < 1
        or matrix.shape[0] != original.shape[0]
        or matrix.shape[2] != original.shape[1]
        or residual.shape != (matrix.shape[0], matrix.shape[1], original.shape[2])
        or protected.shape != original.shape[:2]
        or protected.dtype != torch.bool
        or any(t.device != matrix.device for t in (residual, original, protected))
        or any(
            not t.is_floating_point() or not torch.isfinite(t).all()
            for t in (matrix, residual, original)
        )
    ):
        raise ValueError(
            "constrained solve requires enabled, finite compatible head/query/value tensors"
        )
    with torch.no_grad():
        if (
            policy.maximum_delta_norm_ratio is not None
            and 1 + policy.maximum_delta_norm_ratio > core.maximum_value_norm_ratio
        ):
            raise ValueError("delta ball must lie inside the original value-norm ball")
        v0, r = original.detach().double(), residual.detach().double()
        mutable = (~protected)[:, :, None]
        x = matrix.detach().double() * mutable.transpose(1, 2)
        spectral = torch.linalg.svdvals(x)[:, 0].square()
        ridge = core.value_ridge * policy.ridge_multiplier * spectral.clamp_min(1e-30)
        lipschitz = (spectral + ridge).clamp_min(1e-30)
        norms = v0.norm(dim=-1, keepdim=True)
        # A conservative storage-cast margin, never a larger norm allowance.
        margin = 1 - 2 * torch.finfo(original.dtype).eps
        work_ratio = max(1.0, core.maximum_value_norm_ratio * margin)
        radius = work_ratio * norms
        delta_radius = None
        if policy.maximum_delta_norm_ratio is not None:
            # Reserve cast error relative to V0, not just the smaller delta.
            work_delta_ratio = max(
                0.0, policy.maximum_delta_norm_ratio * margin - 2 * torch.finfo(original.dtype).eps
            )
            delta_radius = work_delta_ratio * norms
        delta = torch.zeros_like(v0)
        y, momentum, best = delta.clone(), 1.0, delta.clone()
        best_loss = r.square().sum((1, 2))
        transpose = x.transpose(1, 2)
        for _ in range(policy.iterations):
            error = torch.bmm(x, y) - r
            gradient = torch.bmm(transpose, error) + ridge[:, None, None] * y
            tentative = v0 + y - gradient / lipschitz[:, None, None]
            if delta_radius is not None:
                offset = tentative - v0
                offset *= (
                    delta_radius / offset.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                ).clamp_max(1)
                tentative = v0 + offset
            scale = (radius / tentative.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp_max(1)
            projected = tentative * scale
            updated = torch.where(mutable, projected - v0, 0)
            next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
            y = updated + (momentum - 1) / next_momentum * (updated - delta)
            delta, momentum = updated, next_momentum
            error = torch.bmm(x, delta) - r
            loss = error.square().sum((1, 2)) + ridge * delta.square().sum((1, 2))
            improve = loss < best_loss
            best = torch.where(improve[:, None, None], delta, best)
            best_loss = torch.minimum(best_loss, loss)
        stored = (v0 + best).to(original.dtype)
        stored[protected] = original[protected]
        # Rare rounding at a boundary is fixed row-locally, not by widening 4x.
        for _ in range(3):
            if policy.maximum_delta_norm_ratio is not None:
                actual_delta = stored.double() - v0
                delta_norm = actual_delta.norm(dim=-1, keepdim=True)
                excessive_delta = delta_norm > policy.maximum_delta_norm_ratio * norms
                factor = (
                    policy.maximum_delta_norm_ratio * norms / delta_norm.clamp_min(1e-30) * margin
                ).clamp_max(1)
                corrected = (v0 + actual_delta * factor).to(original.dtype)
                stored = torch.where(excessive_delta & mutable, corrected, stored)
            actual_norm = stored.double().norm(dim=-1, keepdim=True)
            excessive = actual_norm > core.maximum_value_norm_ratio * norms
            factor = (
                (core.maximum_value_norm_ratio * norms) / actual_norm.clamp_min(1e-30) * margin
            ).clamp_max(1)
            stored = torch.where(
                excessive & mutable, (stored.double() * factor).to(original.dtype), stored
            )
            stored[protected] = original[protected]
        excessive = (
            stored.double().norm(dim=-1, keepdim=True) > core.maximum_value_norm_ratio * norms
        ) | ~torch.isfinite(stored).all(-1, keepdim=True)
        if policy.maximum_delta_norm_ratio is not None:
            excessive |= (stored.double() - v0).norm(
                dim=-1, keepdim=True
            ) > policy.maximum_delta_norm_ratio * norms
        stored = torch.where(excessive, original, stored)
        maximum_ratio = (stored.double().norm(dim=-1) / norms.squeeze(-1).clamp_min(1e-12)).amax(-1)
        actual_delta = stored.double() - v0
        actual_loss = (torch.bmm(x, actual_delta) - r).square().sum((1, 2))
        actual_loss += ridge * actual_delta.square().sum((1, 2))
        return stored.detach().clone().contiguous(), dict(
            constrained_maximum_value_norm_ratio=maximum_ratio,
            storage_rounding_rollback_rows=excessive.squeeze(-1).sum(-1),
            regularized_objective_after_cast=actual_loss,
            regularized_objective_initial=r.square().sum((1, 2)),
            relative_ridge=core.value_ridge * policy.ridge_multiplier,
            maximum_value_delta_norm_ratio=(
                actual_delta.norm(dim=-1) / norms.squeeze(-1).clamp_min(1e-12)
            ).amax(-1),
            delta_norm_limit=policy.maximum_delta_norm_ratio,
            working_value_norm_ratio=work_ratio,
            iterations=policy.iterations,
            constraints_enforced_during_optimization=True,
        )


def fit_component(
    attention: Any,
    original: Any,
    target: Any,
    history_mass: Any,
    history_numerator: Any,
    protected: Any,
    mode: str,
    core: AttentionCompactionPolicy,
    policy: ConstrainedValuePolicy,
    *,
    mass_bias: Any = None,
) -> ComponentFit:
    """Fit ONLY passed-in training queries; no holdout/annotations are arguments."""
    import torch

    if mode not in {"anchor", "mass_only", "value_only", "joint"}:
        raise ValueError("unknown compaction component")
    if (
        attention.ndim != 3
        or original.ndim != 3
        or min(attention.shape) < 1
        or original.shape[:2] != (attention.shape[0], attention.shape[2])
        or target.shape != (attention.shape[0], attention.shape[1], original.shape[2])
        or history_numerator.shape != target.shape
        or history_mass.shape != attention.shape[:2]
        or protected.shape != original.shape[:2]
        or protected.dtype != torch.bool
        or any(
            t.device != attention.device
            for t in (original, target, history_mass, history_numerator, protected)
        )
        or any(
            not t.is_floating_point() or not torch.isfinite(t).all()
            for t in (attention, original, target, history_mass, history_numerator)
        )
        or (attention < 0).any()
        or (history_mass < 0).any()
        or (attention.sum(-1) + history_mass > 1 + 1e-5).any()
        or not core.enabled
    ):
        raise ValueError("component fit requires enabled, compatible finite training evidence")
    with torch.no_grad():
        zero = torch.zeros_like(original[:, :, 0], dtype=torch.float32)
        before = (
            (
                attention_output(attention, original, zero, history_mass, history_numerator)
                - target.double()
            )
            .square()
            .mean((1, 2))
        )
        bias = zero.clone()
        values = original.detach().clone()
        extra: dict[str, Any] = {}
        if mode in {"mass_only", "joint"}:
            bias = (
                fit_batched_mass_bias(attention, history_mass, protected, core)
                if mass_bias is None
                else mass_bias.clone()
            )
            if (
                bias.shape != zero.shape
                or bias.dtype != torch.float32
                or not torch.isfinite(bias).all()
                or (bias < 0).any()
                or (bias.double().exp() > core.maximum_mass_multiplier).any()
                or (bias[protected] != 0).any()
            ):
                raise ValueError("mass biases violate the fixed multiplier/protection contract")
        if mode in {"value_only", "joint"}:
            a, w = attention.double(), bias.double().exp()
            mass = torch.bmm(a, w[:, :, None]).squeeze(-1) + history_mass.double()
            if (mass <= 1e-12).any():
                raise ValueError("insufficient mass for constrained value solve")
            normalized = a * w[:, None, :] / mass[:, :, None]
            residual = target.double() - history_numerator.double() / mass[:, :, None]
            residual -= torch.bmm(normalized, original.double())
            values, extra = solve_bounded_values(
                normalized, residual, original, protected, core, policy
            )
        attempted_output = attention_output(
            attention, values, bias, history_mass, history_numerator
        )
        after = (attempted_output - target.double()).square().mean((1, 2))
        ratios = (
            values.double().norm(dim=-1) / original.double().norm(dim=-1).clamp_min(1e-12)
        ).amax(-1)
        admitted = (after < before - core.minimum_improvement) & (
            ratios <= core.maximum_value_norm_ratio
        )
        if mode == "anchor":
            admitted[:] = False
        returned_v = torch.where(admitted[:, None, None], values, original)
        returned_b = torch.where(admitted[:, None], bias, zero)
        return ComponentFit(
            returned_v.detach().clone().contiguous(),
            returned_b.detach().clone().contiguous(),
            admitted.detach().clone(),
            values.detach().clone(),
            bias.detach().clone(),
            dict(
                training_mse_before=before,
                attempted_training_mse_after=after,
                returned_training_mse=torch.where(admitted, after, before),
                maximum_value_norm_ratio=ratios,
                protected_values_unchanged=bool(
                    torch.equal(values[protected], original[protected])
                ),
                protected_bias_zero=bool((bias[protected] == 0).all()),
                **extra,
            ),
        )
