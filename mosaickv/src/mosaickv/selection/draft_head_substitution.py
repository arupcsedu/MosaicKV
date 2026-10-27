"""Bounded exact-capacity proposals from individual greedy-draft attentions.

Renormalized value-output reconstruction generates the pool only. A proposal
is never accepted without the independent, unchanged actual-logit guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mosaickv.selection.head_substitution import HeadSubstitution


@dataclass(frozen=True)
class DraftHeadPoolPolicy:
    choices_per_side: int = 4
    position_chunk_size: int = 64

    def __post_init__(self) -> None:
        if (
            type(self.choices_per_side) is not int
            or not 1 <= self.choices_per_side <= 8
            or type(self.position_chunk_size) is not int
            or not 1 <= self.position_chunk_size <= 64
        ):
            raise ValueError("draft-head proposal shortlists/chunks must remain bounded")


def _single_scores(p, v, numerator, denominator, target, *, sign, chunk, gradient=None):
    import torch

    scores = []
    for start in range(0, p.shape[-1], chunk):
        end = min(start + chunk, p.shape[-1])
        weight = p[:, :, start:end]
        num = numerator.unsqueeze(2) + sign * weight.unsqueeze(-1) * v[:, None, start:end]
        den = denominator.unsqueeze(-1) + sign * weight
        output = num / den.clamp_min(1e-30).unsqueeze(-1)
        loss = (
            (output - target.unsqueeze(2)).square()
            if gradient is None
            else output * gradient.unsqueeze(2)
        )
        score = loss.mean((1, 3))
        scores.append(score.masked_fill(~(den > 0).all(1), float("inf")))
    return torch.cat(scores, dim=-1)


def draft_substitution_pool(
    full_state: Any,
    anchor: Any,
    evidence: tuple[dict[str, Any], ...],
    *,
    limit: int,
    policy: DraftHeadPoolPolicy | None = None,
    output_gradients: tuple[Any, ...] | None = None,
) -> tuple[tuple[HeadSubstitution, ...], list[dict[str, Any]]]:
    import torch

    from mosaickv.measurements.cache_lifecycle import CacheLifecycle

    policy = DraftHeadPoolPolicy() if policy is None else policy
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("draft-head pool permits at most sixteen proposals")
    if not anchor.all_slots_valid or anchor.prototype_slots or anchor.layerwise_logical_positions:
        raise ValueError("draft-head proposals require an unpadded, exact-only anchor")
    if len(evidence) != len(full_state.layers):
        raise ValueError("draft evidence must cover every source layer")
    if output_gradients is not None and len(output_gradients) != len(full_state.layers):
        raise ValueError("downstream gradients must cover every source layer")
    logical = full_state.logical_positions.physical_to_logical
    physical = {position: i for i, position in enumerate(logical)}
    positions: dict[tuple[int, int], set[int]] = {
        (i, h): set() for i, layer in enumerate(full_state.layers) for h in range(layer.kv_heads)
    }
    for row in anchor.slot_records:
        route = row["layer"], row["kv_head"]
        position = physical[row["logical_position"]]
        if row["tier"] != "exact" or position in positions[route]:
            raise ValueError("draft anchor has duplicate/non-exact source membership")
        positions[route].add(position)
    protected: dict[tuple[int, int], set[int]] = {route: set() for route in positions}
    for block in full_state.blocks:
        if block.mandatory:
            protected[block.layer, block.kv_head].update(block.physical_cache_indices)
    candidates = []
    for i, (source, old, data) in enumerate(
        zip(full_state.layers, anchor.snapshot.layers, evidence, strict=True)
    ):
        if (
            (
                source.key_head_dimension,
                source.value_head_dimension,
                source.key_sequence_dimension,
                source.value_sequence_dimension,
            )
            != (1, 1, 2, 2)
            or source.key.shape[0] != 1
            or source.value.shape[0] != 1
        ):
            raise ValueError("draft-head proposals support audited BHSD cache tensors only")
        heads, n = source.kv_heads, source.sequence_length
        v = source.value.detach()[0].float()
        p, target, tail_num, tail_den = [
            data[key].to(v.device, torch.float32)
            for key in ("prompt_probabilities", "target", "generated_numerator", "generated_mass")
        ]
        if (
            p.ndim != 3
            or p.shape[0] != heads
            or p.shape[-1] != n
            or p.shape[1] < 1
            or target.shape != (heads, p.shape[1], v.shape[-1])
            or tail_num.shape != target.shape
            or tail_den.shape != p.shape[:2]
            or not all(bool(torch.isfinite(t).all()) for t in (p, target, tail_num, tail_den))
            or bool((p < 0).any())
            or bool((tail_den < 0).any())
            or not bool(torch.isfinite(v).all())
            or bool((p.sum(-1) + tail_den > 1 + 1e-5).any())
        ):
            raise ValueError("draft-head evidence has invalid shape or nonfinite values")
        mask = torch.zeros((heads, n), dtype=torch.bool, device=v.device)
        mandatory = torch.zeros_like(mask)
        for h in range(heads):
            if len(positions[i, h]) != old.key.shape[2]:
                raise ValueError("draft anchor per-head capacity must match physical slots")
            mask[h, list(sorted(positions[i, h]))] = True
        for h in range(heads):
            mandatory[h, list(sorted(protected[i, h]))] = True
        if bool((mandatory & ~mask).any()):
            raise ValueError("draft anchor omits mandatory evidence")
        num = torch.bmm(p * mask.unsqueeze(1), v) + tail_num
        den = (p * mask.unsqueeze(1)).sum(-1) + tail_den
        if not bool((den > 0).all()):
            raise ValueError("draft anchor has zero retained probability")
        gradient = None if output_gradients is None else output_gradients[i].to(v.device, v.dtype)
        if gradient is not None and (
            gradient.shape != target.shape or not torch.isfinite(gradient).all()
        ):
            raise ValueError("downstream gradient shape/nonfinite mismatch")
        reconstruction = (num / den.unsqueeze(-1) - target).square().mean((1, 2))
        baseline = (
            reconstruction
            if gradient is None
            else (num / den.unsqueeze(-1) * gradient).mean((1, 2))
        )
        add = _single_scores(
            p, v, num, den, target, sign=1, chunk=policy.position_chunk_size, gradient=gradient
        )
        drop = _single_scores(
            p, v, num, den, target, sign=-1, chunk=policy.position_chunk_size, gradient=gradient
        )
        add.masked_fill_(mask, float("inf"))
        drop.masked_fill_(~mask | mandatory, float("inf"))
        k = min(policy.choices_per_side, n)
        # Stable sorting preserves physical-position order for exact score ties.
        additions = torch.argsort(add, dim=-1, stable=True)[:, :k]
        removals = torch.argsort(drop, dim=-1, stable=True)[:, :k]
        best = torch.full_like(baseline, float("inf"))
        best_add = torch.full((heads,), -1, device=v.device, dtype=torch.long)
        best_drop = best_add.clone()
        routes = torch.arange(heads, device=v.device)
        for d in range(k):
            for a in range(k):
                receiver, donor = additions[:, a], removals[:, d]
                pa, pd = p[routes, :, receiver], p[routes, :, donor]
                candidate_den = den + pa - pd
                candidate_num = (
                    num
                    + pa.unsqueeze(-1) * v[routes, receiver, None]
                    - pd.unsqueeze(-1) * v[routes, donor, None]
                )
                output = candidate_num / candidate_den.clamp_min(1e-30).unsqueeze(-1)
                error = (
                    (output - target).square() if gradient is None else output * gradient
                ).mean((1, 2))
                valid = ~mask[routes, receiver] & mask[routes, donor] & ~mandatory[routes, donor]
                valid = valid & (candidate_den > 0).all(-1) & torch.isfinite(error)
                improve = valid & (error < best)
                best = torch.where(improve, error, best)
                best_add = torch.where(improve, receiver, best_add)
                best_drop = torch.where(improve, donor, best_drop)
        cost = (
            source.key.shape[-1] * source.key.element_size()
            + source.value.shape[-1] * source.value.element_size()
        )
        for h, (before, after, r, d) in enumerate(
            zip(
                baseline.tolist(), best.tolist(), best_add.tolist(), best_drop.tolist(), strict=True
            )
        ):
            if after < before:
                proposal = HeadSubstitution(i, h, d, r, cost)
                pa, pd = p[h, :, r], p[h, :, d]
                actual = (num[h] + pa[:, None] * v[h, r] - pd[:, None] * v[h, d]) / (
                    den[h] + pa - pd
                )[:, None]
                reconstructed_after = float((actual - target[h]).square().mean())
                candidates.append(
                    (
                        before - after,
                        i,
                        h,
                        d,
                        r,
                        proposal,
                        before,
                        after,
                        float(reconstruction[h]),
                        reconstructed_after,
                    )
                )
    # At most one analytically best exchange per route. Cross-route ranking is
    # absolute draft-output error reduction, not prompt mass or reference labels.
    ranked = sorted(candidates, key=lambda row: (-row[0], *row[1:5]))[:limit]
    storage = CacheLifecycle.count([(layer.key, layer.value) for layer in anchor.snapshot.layers])[
        1
    ]
    if storage != sum(
        len(positions[i, 0]) * source.byte_size // source.sequence_length
        for i, source in enumerate(full_state.layers)
    ):
        raise ValueError("draft anchor storage boundary differs from source-membership bytes")
    return tuple(row[5] for row in ranked), [
        dict(
            layer=row[1],
            kv_head=row[2],
            removed_position=row[3],
            added_position=row[4],
            draft_output_error_before=row[8],
            draft_output_error_after=row[9],
            predicted_gain=row[0],
            ranking_objective="draft_output_reconstruction"
            if output_gradients is None
            else "frozen_downstream_directional_KL",
            ranking_objective_before=row[6],
            ranking_objective_after=row[7],
            replaced_bytes=row[5].replaced_bytes,
        )
        for row in ranked
    ]
