"""Prompt-output reconstruction by exact subsets; no replacement K/V or model probes.

Attention matching is related work (https://arxiv.org/abs/2602.16284), not a
novel general principle claimed here. This development variant only swaps
equal-cost source positions, preserving their original keys and values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mosaickv.config import OutputPreservingConfig


@dataclass(frozen=True)
class Reconstruction:
    numerator: Any
    mass: Any
    output: Any
    error: float


@dataclass(frozen=True)
class SwapResult:
    positions: tuple[int, ...]
    initial_error: float
    final_error: float
    swaps: tuple[dict[str, int | float], ...]
    evaluated_pairs: int
    rejected_zero_mass_pairs: int
    validation_initial_error: float | None = None
    validation_final_error: float | None = None
    rejected_validation_pairs: int = 0


def prepare_evidence(attention: Any, values: Any, *, pooled: bool = False) -> tuple[Any, Any]:
    """Retain each query row, normalize capture rounding, and accumulate in FP32.

    Float64 is retained for numerical fixtures. Causal zeros remain exact zeros.
    Neither input is mutated. The pooled ablation intentionally discards query identity.
    """
    import torch

    if attention.ndim != 2 or values.ndim != 2 or attention.shape[1] != values.shape[0]:
        raise ValueError("evidence must be [queries, positions] and [positions, value_dim]")
    if min(*attention.shape, *values.shape) < 1 or attention.device != values.device:
        raise ValueError("evidence must be nonempty and colocated")
    if not attention.is_floating_point() or not values.is_floating_point():
        raise ValueError("attention and values must be floating-point tensors")
    if not bool(torch.isfinite(attention).all() & torch.isfinite(values).all()):
        raise ValueError("evidence must be finite")
    if bool((attention < 0).any()) or bool((attention.sum(-1) <= 0).any()):
        raise ValueError("attention must be nonnegative with positive row sums")
    dtype = torch.float64 if attention.dtype == values.dtype == torch.float64 else torch.float32
    a, v = attention.detach().to(dtype), values.detach().to(dtype)
    a = a / a.sum(-1, keepdim=True)
    if pooled:
        a = a.mean(0, keepdim=True)
    return a, v


def reconstruct(
    attention: Any,
    values: Any,
    positions: Sequence[int],
    *,
    minimum_mass: float = 1e-8,
) -> Reconstruction:
    """Evaluate renormalized output, not unnormalized retained attention mass."""
    import torch

    selected = tuple(positions)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(type(p) is not int or p < 0 or p >= values.shape[0] for p in selected)
    ):
        raise ValueError("selected positions must be nonempty, unique, in-range integers")
    index = torch.tensor(selected, device=attention.device, dtype=torch.long)
    retained = attention.index_select(1, index)
    mass = retained.sum(-1)
    if bool((mass <= minimum_mass).any()):
        raise ValueError("selected cache leaves a query with zero/insufficient attention mass")
    numerator = retained @ values.index_select(0, index)
    output = numerator / mass[:, None]
    error = float(((attention @ values - output) ** 2).sum().item())
    if not bool(torch.isfinite(output).all()) or not torch.isfinite(torch.tensor(error)):
        raise ValueError("nonfinite reconstruction")
    return Reconstruction(numerator, mass, output, error)


def corrective_swaps(
    attention: Any,
    values: Any,
    positions: Sequence[int],
    mandatory: Sequence[int],
    config: OutputPreservingConfig,
    *,
    validation_attention: Any | None = None,
) -> SwapResult:
    """Bounded sensitivity shortlist followed by exact, deterministic pair tests.

    Inputs are normalized evidence from prepare_evidence. Fixed capacity and
    same-route one-token blocks make each exchange exactly byte-neutral. The
    gradient proposes candidates only; acceptance uses the actual normalized
    objective, recomputed from the final subset to guard against cancellation.
    """
    import torch

    selected = tuple(sorted(positions))
    protected = frozenset(mandatory)
    if not protected.issubset(selected):
        raise ValueError("mandatory positions are absent from the initial exact subset")
    initial = reconstruct(attention, values, selected, minimum_mass=config.minimum_query_mass)
    current = initial
    target = attention @ values
    validation_current = (
        None
        if validation_attention is None
        else reconstruct(
            validation_attention,
            values,
            selected,
            minimum_mass=config.minimum_query_mass,
        )
    )
    validation_initial = validation_current
    validation_target = (
        None if validation_attention is None else validation_attention @ values
    )
    swaps: list[dict[str, int | float]] = []
    evaluated = rejected = rejected_validation = 0
    all_positions = torch.arange(values.shape[0], device=values.device)
    for _ in range(config.max_swaps_per_head):
        mask = torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)
        mask[list(selected)] = True
        removable = mask.clone()
        if protected:
            removable[list(sorted(protected))] = False
        donors, recipients = all_positions[removable], all_positions[~mask]
        if donors.numel() == 0 or recipients.numel() == 0:
            break
        error = current.output - target
        # Derivative of the normalized reconstruction loss for membership weight i.
        sensitivity = 2 * (
            ((attention.T @ (error / current.mass[:, None])) * values).sum(-1)
            - attention.T @ ((error * current.output).sum(-1) / current.mass)
        )
        donors = (
            donors[
                torch.argsort(sensitivity[donors], descending=True, stable=True)[
                    : config.shortlist_size
                ]
            ]
            .sort()
            .values
        )
        recipients = (
            recipients[torch.argsort(sensitivity[recipients], stable=True)[: config.shortlist_size]]
            .sort()
            .values
        )
        pairs = torch.cartesian_prod(donors, recipients).reshape(-1, 2)
        best_error = current.error
        best_pair: tuple[int, int] | None = None
        for start in range(0, int(pairs.shape[0]), config.pair_chunk_size):
            chunk = pairs[start : start + config.pair_chunk_size]
            remove, add = chunk[:, 0], chunk[:, 1]
            a_remove, a_add = attention[:, remove].T, attention[:, add].T
            mass = current.mass[None] - a_remove + a_add
            valid = (mass > config.minimum_query_mass).all(-1)
            rejected += int((~valid).sum().item())
            numerator = (
                current.numerator[None]
                - a_remove[:, :, None] * values[remove, None, :]
                + a_add[:, :, None] * values[add, None, :]
            )
            # Invalid proposals are rejected, not repaired by fictitious attention.
            safe_mass = torch.where(valid[:, None], mass, torch.ones_like(mass))
            losses = ((numerator / safe_mass[:, :, None] - target[None]) ** 2).sum((1, 2))
            losses = torch.where(valid & torch.isfinite(losses), losses, torch.inf)
            if validation_current is not None:
                assert validation_attention is not None
                assert validation_target is not None
                validation_remove = validation_attention[:, remove].T
                validation_add = validation_attention[:, add].T
                validation_mass = (
                    validation_current.mass[None] - validation_remove + validation_add
                )
                validation_valid = (
                    validation_mass > config.minimum_query_mass
                ).all(-1)
                validation_numerator = (
                    validation_current.numerator[None]
                    - validation_remove[:, :, None] * values[remove, None, :]
                    + validation_add[:, :, None] * values[add, None, :]
                )
                safe_validation_mass = torch.where(
                    validation_valid[:, None],
                    validation_mass,
                    torch.ones_like(validation_mass),
                )
                validation_losses = (
                    (
                        validation_numerator / safe_validation_mass[:, :, None]
                        - validation_target[None]
                    )
                    ** 2
                ).sum((1, 2))
                validation_improves = (
                    validation_valid
                    & torch.isfinite(validation_losses)
                    & (
                        validation_losses
                        < validation_current.error - config.minimum_improvement
                    )
                )
                prompt_improves = losses < current.error - config.minimum_improvement
                rejected_validation += int((prompt_improves & ~validation_improves).sum().item())
                losses = torch.where(validation_improves, losses, torch.inf)
            evaluated += int(chunk.shape[0])
            index = int(losses.argmin().item())
            loss = float(losses[index].item())
            if loss < best_error - config.minimum_improvement:
                best_error = loss
                donor, recipient = chunk[index].tolist()
                best_pair = (int(donor), int(recipient))
        if best_pair is None:
            break
        donor, recipient = best_pair
        proposal = tuple(sorted((set(selected) - {donor}) | {recipient}))
        verified = reconstruct(attention, values, proposal, minimum_mass=config.minimum_query_mass)
        if verified.error >= current.error - config.minimum_improvement:
            break
        validation_verified = (
            None
            if validation_attention is None
            else reconstruct(
                validation_attention,
                values,
                proposal,
                minimum_mass=config.minimum_query_mass,
            )
        )
        if validation_current is not None and (
            validation_verified is None
            or validation_verified.error
            >= validation_current.error - config.minimum_improvement
        ):
            raise RuntimeError("joint draft validation changed between proposal and verification")
        swaps.append(
            {
                "removed_position": donor,
                "added_position": recipient,
                "error_before": current.error,
                "error_after": verified.error,
                **(
                    {}
                    if validation_current is None or validation_verified is None
                    else {
                        "draft_error_before": validation_current.error,
                        "draft_error_after": validation_verified.error,
                    }
                ),
            }
        )
        selected, current = proposal, verified
        validation_current = validation_verified
    return SwapResult(
        selected,
        initial.error,
        current.error,
        tuple(swaps),
        evaluated,
        rejected,
        (
            None
            if validation_initial is None
            else validation_initial.error
        ),
        (
            None
            if validation_current is None
            else validation_current.error
        ),
        rejected_validation,
    )
