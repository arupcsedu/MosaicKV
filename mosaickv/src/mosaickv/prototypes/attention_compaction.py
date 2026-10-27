"""Experimental mass/value compaction; NOT exact retention or an HF integration.

Attention matching is prior work: https://arxiv.org/abs/2602.16284v2.
This local variant protects mandatory entries, keeps original post-RoPE keys,
and explicitly includes retained generated history. No model parameters or
benchmark labels are inputs. The output contains synthetic value summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any


@dataclass(frozen=True)
class AttentionCompactionPolicy:
    enabled: bool = False
    mass_iterations: int = 256
    mass_ridge: float = 1e-6
    value_ridge: float = 1e-6
    maximum_mass_multiplier: float = 64.0
    maximum_value_norm_ratio: float = 4.0
    minimum_improvement: float = 1e-8
    svd_relative_tolerance: float = 1e-6

    def __post_init__(self):
        if (
            type(self.enabled) is not bool
            or type(self.mass_iterations) is not int
            or not 1 <= self.mass_iterations <= 1024
        ):
            raise ValueError(
                "compaction enabled must be boolean; mass_iterations must be in [1,1024]"
            )
        for name in ("mass_ridge", "value_ridge", "minimum_improvement", "svd_relative_tolerance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"compaction {name} must be finite and nonnegative")
        if not 0 < self.svd_relative_tolerance < 1:
            raise ValueError("compaction svd_relative_tolerance must be in (0,1)")
        for name in ("maximum_mass_multiplier", "maximum_value_norm_ratio"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 1
            ):
                raise ValueError(f"compaction {name} must be finite and at least one")


@dataclass(frozen=True)
class CompactionCapacity:
    prompt_positions: int
    history_positions: int
    lifetime_kv_bytes: int
    log_bias_bytes: int
    additive_mask_bytes: int
    total_persistent_bytes: int
    unused_bytes: int


def plan_compaction_capacity(
    *,
    cap_bytes: int,
    layers: int,
    kv_heads: int,
    query_heads: int,
    key_dimension: int,
    value_dimension: int,
    cache_element_bytes: int,
    history_positions: int,
    mandatory_positions: int = 2,
) -> CompactionCapacity:
    """Count actual proposed allocations, not just tensor-view payloads.

    Dense uniform per-layer/head slots. FP32 prefix biases and a preallocated
    FP32 additive mask covering prompt+history are charged in addition to KV.
    This is capacity planning, never a measured GPU-memory result.
    """
    args = locals().copy()
    for name, value in args.items():
        minimum = 0 if name in {"history_positions", "mandatory_positions"} else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    kv = layers * kv_heads * (key_dimension + value_dimension) * cache_element_bytes
    bias = layers * kv_heads * 4
    mask = layers * query_heads * 4
    fixed = history_positions * (kv + mask)
    positions = (cap_bytes - fixed) // (kv + bias + mask)
    if positions < max(1, mandatory_positions):
        raise ValueError(
            "hard cap cannot fit mandatory prompt entries plus history and auxiliary storage"
        )
    kv_bytes = kv * (positions + history_positions)
    bias_bytes = bias * positions
    mask_bytes = mask * (positions + history_positions)
    total = kv_bytes + bias_bytes + mask_bytes
    return CompactionCapacity(
        positions, history_positions, kv_bytes, bias_bytes, mask_bytes, total, cap_bytes - total
    )


def _validate_mass(attention, history_mass, protected):
    import torch

    if (
        attention.ndim != 2
        or min(attention.shape) < 1
        or history_mass.shape != attention.shape[:1]
        or protected.shape != attention.shape[1:]
        or protected.dtype != torch.bool
        or attention.device != history_mass.device
        or attention.device != protected.device
        or not attention.is_floating_point()
        or not history_mass.is_floating_point()
        or not torch.isfinite(attention).all()
        or not torch.isfinite(history_mass).all()
        or (attention < 0).any()
        or (history_mass < 0).any()
        or (attention.sum(-1) + history_mass > 1 + 1e-5).any()
    ):
        raise ValueError(
            "mass evidence requires finite compatible full-normalized probabilities "
            "and a protected mask"
        )


def fit_batched_mass_bias(attention, history_mass, protected, policy: AttentionCompactionPolicy):
    """Fit independent [route, query, slot] systems without model forwards.

    The fixed iteration bound and shared tensor operations avoid a Python
    optimization loop per head. Protected columns remain exactly unbiased.
    """
    import torch

    if (
        attention.ndim != 3
        or min(attention.shape) < 1
        or history_mass.shape != attention.shape[:2]
        or protected.shape != (attention.shape[0], attention.shape[2])
    ):
        raise ValueError("batched mass evidence requires [route,query,slot] tensors")
    _validate_mass(attention.flatten(0, 1), history_mass.flatten(), protected[0])
    if protected.dtype != torch.bool or protected.device != attention.device:
        raise ValueError("batched protected masks must be boolean on the evidence device")
    if not policy.enabled:
        raise ValueError("experimental attention compaction is disabled")
    with torch.no_grad():
        a, h = attention.detach().double(), history_mass.detach().double()
        mutable = (~protected).double()
        matrix = a * mutable[:, None, :]
        missing = (1 - h - a.sum(-1)).clamp_min(0)
        x = torch.zeros_like(mutable)
        squared = matrix.square().sum((1, 2))
        ridge = policy.mass_ridge * squared.clamp_min(1e-30)
        lipschitz = (squared + ridge).clamp_min(1e-30)
        y, momentum = x.clone(), 1.0
        best, best_loss = x.clone(), missing.square().sum(-1)
        for _ in range(policy.mass_iterations):
            error = torch.bmm(matrix, y[:, :, None]).squeeze(-1) - missing
            gradient = torch.bmm(matrix.transpose(1, 2), error[:, :, None]).squeeze(-1)
            gradient += ridge[:, None] * y
            updated = (y - gradient / lipschitz[:, None]).clamp(
                0, policy.maximum_mass_multiplier - 1
            ) * mutable
            next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
            y = updated + (momentum - 1) / next_momentum * (updated - x)
            x, momentum = updated, next_momentum
            error = torch.bmm(matrix, x[:, :, None]).squeeze(-1) - missing
            loss = error.square().sum(-1)
            improve = loss < best_loss
            best = torch.where(improve[:, None], x, best)
            best_loss = torch.minimum(loss, best_loss)
        # Rounding log(Wmax) upward in FP32 could violate the multiplier cap.
        # Encode the greatest representable bias whose decoded weight fits.
        bound = torch.tensor(
            math.log(policy.maximum_mass_multiplier), device=a.device, dtype=torch.float32
        )
        if float(bound.double().exp()) > policy.maximum_mass_multiplier:
            bound = torch.nextafter(bound, torch.full_like(bound, -float("inf")))
        return torch.minimum((1 + best).log().float(), bound)


def fit_mass_bias(attention, history_mass, protected, policy: AttentionCompactionPolicy):
    """Fixed-iteration projected least squares; no labels or model forwards.

    In full-distribution probability units, fit A*w + h ~= 1 with w>=1.
    Protected entries have w=1 exactly. Bounded multipliers prevent a token
    with near-zero probability from receiving an arbitrarily large bias.
    """
    import torch

    _validate_mass(attention, history_mass, protected)
    if not policy.enabled:
        raise ValueError("experimental attention compaction is disabled")
    with torch.no_grad():
        a, h = attention.detach().double(), history_mass.detach().double()
        mutable = ~protected
        matrix = a[:, mutable]
        log_bias = fit_batched_mass_bias(
            attention[None], history_mass[None], protected[None], policy
        )[0]
        weights = log_bias.double().exp()  # Audit AFTER the storage cast.
        if matrix.numel():
            singular = torch.linalg.svdvals(matrix)
            rank = (
                int((singular > singular[0] * policy.svd_relative_tolerance).sum())
                if float(singular[0]) > 0
                else 0
            )
            condition = float(singular[0] / singular[-1]) if float(singular[-1]) > 0 else None
            if condition is not None and not math.isfinite(condition):
                condition = None
        else:
            rank, condition = 0, None
        return log_bias, dict(
            baseline_mass_rmse=float((a.sum(-1) + h - 1).square().mean().sqrt()),
            fitted_mass_rmse=float((a @ weights + h - 1).square().mean().sqrt()),
            mutable_columns=int(mutable.sum()),
            reference_queries=a.shape[0],
            numerical_rank=rank,
            condition_number=condition,
            maximum_multiplier=float(weights.max()),
            iterations=policy.mass_iterations,
            protected_bias_bitwise_zero=bool((log_bias[protected] == 0).all()),
        )


@dataclass(frozen=True)
class CompactedHead:
    keys: Any
    values: Any
    log_bias: Any
    logical_positions: tuple[int, ...]
    construction_accepted: bool
    diagnostics: dict[str, Any]
    requires_model_validation: bool = True

    @property
    def storage_bytes(self):
        # Constructors create independent contiguous allocations, not aliases
        # to a full-size source cache, so these are actual backing-storage bytes.
        return sum(t.untyped_storage().nbytes() for t in (self.keys, self.values, self.log_bias))


def compact_head(
    keys,
    values,
    selected_attention,
    full_outputs,
    history_mass,
    history_numerator,
    protected,
    logical_positions: tuple[int, ...],
    *,
    source_positions: int,
    policy: AttentionCompactionPolicy,
    evidence_kind: str,
    retention_one: bool = False,
) -> CompactedHead:
    """Keep keys/positions; refit only unprotected value summaries.

    Full outputs and attention are from prompt or temporary deterministic draft
    forwards. They are never annotation targets. A successful algebraic fit
    DOES NOT authorize benchmark claims or bypass the registered logit guard.
    """
    import torch

    _validate_mass(selected_attention, history_mass, protected)
    if (
        evidence_kind not in {"prompt_window", "draft_rollout", "hybrid"}
        or keys.ndim != 2
        or values.ndim != 2
        or min(keys.shape) < 1
        or min(values.shape) < 1
        or keys.shape[0] != values.shape[0]
        or values.shape[0] != selected_attention.shape[1]
        or full_outputs.shape != (selected_attention.shape[0], values.shape[1])
        or history_numerator.shape != full_outputs.shape
        or keys.dtype != values.dtype
        or not keys.is_floating_point()
        or not full_outputs.is_floating_point()
        or not history_numerator.is_floating_point()
        or any(
            t.device != keys.device
            for t in (values, selected_attention, full_outputs, history_numerator)
        )
        or any(not torch.isfinite(t).all() for t in (keys, values, full_outputs, history_numerator))
        or type(source_positions) is not int
        or source_positions < keys.shape[0]
        or type(retention_one) is not bool
        or type(logical_positions) is not tuple
        or len(logical_positions) != keys.shape[0]
        or any(type(p) is not int or p < 0 for p in logical_positions)
        or any(a >= b for a, b in pairwise(logical_positions))
    ):
        raise ValueError(
            "compaction requires compatible finite single-head evidence "
            "and monotonic original positions"
        )
    if retention_one and source_positions != keys.shape[0]:
        raise ValueError("retention one requires every source position")
    with torch.no_grad():

        def result(new_values, bias, accepted, diagnostics):
            return CompactedHead(
                keys.detach().clone().contiguous(),
                new_values.detach().clone().contiguous(),
                bias.detach().clone().contiguous(),
                logical_positions,
                accepted,
                dict(
                    diagnostics,
                    representation="original_keys_plus_synthetic_values_and_log_bias",
                    exact_tier=False,
                    model_parameters_updated=False,
                    evidence_kind=evidence_kind,
                    benchmark_quality_validated=False,
                ),
            )

        zero = torch.zeros(keys.shape[0], dtype=torch.float32, device=keys.device)
        if retention_one:
            return result(values, zero, False, dict(reason="retention_one_identity_no_fit"))
        if not policy.enabled:
            return result(values, zero, False, dict(reason="disabled_identity_no_fit"))
        a, v = selected_attention.detach().double(), values.detach().double()
        y = full_outputs.detach().double()
        hm, hn = history_mass.detach().double(), history_numerator.detach().double()
        initial_output = (a @ v + hn) / (a.sum(-1) + hm).clamp_min(1e-30)[:, None]
        before = float((initial_output - y).square().mean())
        if not math.isfinite(before):
            return result(values, zero, False, dict(reason="nonfinite_reference_output_error"))
        bias, mass_record = fit_mass_bias(a, hm, protected, policy)
        weights = bias.double().exp()
        mass = a @ weights + hm
        if (mass <= 1e-12).any():
            return result(
                values, zero, False, dict(reason="insufficient_query_mass", mass=mass_record)
            )
        normalized = a * weights[None] / mass[:, None]
        matrix = normalized[:, ~protected]
        residual = y - (normalized @ v + hn / mass[:, None])
        if not matrix.numel():
            return result(
                values, zero, False, dict(reason="all_values_protected", mass=mass_record)
            )
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        ridge = policy.value_ridge * singular.square().max().clamp_min(1e-30)
        keep = singular > singular[0] * policy.svd_relative_tolerance
        factors = torch.where(keep, singular / (singular.square() + ridge).clamp_min(1e-30), 0)
        correction = (vh.T * factors) @ (u.T @ residual)
        attempted = v.clone()
        attempted[~protected] += correction
        stored = attempted.to(values.dtype)
        # Protected tensors are copied from the original dtype, never round-tripped.
        stored[protected] = values[protected]
        # Enforce the bound on the actual stored representation, not FP64 work.
        ratio = float(
            (
                stored[~protected].double().norm(dim=-1)
                / v[~protected].norm(dim=-1).clamp_min(1e-12)
            ).max()
        )
        output = (a @ (stored.double() * weights[:, None]) + hn) / mass[:, None]
        after = float((output - y).square().mean())
        finite = bool(torch.isfinite(stored).all()) and math.isfinite(after)
        accepted = (
            finite
            and math.isfinite(ratio)
            and ratio <= policy.maximum_value_norm_ratio
            and after < before - policy.minimum_improvement
        )
        diagnostics = dict(
            reason="constructed_requires_model_validation" if accepted else "bounded_fit_rejected",
            reference_output_mse_before=before,
            attempted_reference_output_mse_after=after if math.isfinite(after) else None,
            returned_reference_output_mse=after if accepted else before,
            maximum_value_norm_ratio=ratio if math.isfinite(ratio) else None,
            mutable_values=int((~protected).sum()),
            mass=mass_record,
            reference_queries=selected_attention.shape[0],
            measured_head_storage_bytes=keys.numel() * keys.element_size()
            + values.numel() * values.element_size()
            + zero.numel() * zero.element_size(),
            protected_values_bitwise_unchanged=bool(
                torch.equal(stored[protected], values[protected])
            ),
        )
        return result(
            stored if accepted else values, bias if accepted else zero, accepted, diagnostics
        )
