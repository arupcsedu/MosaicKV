"""Per-query, whole-value-row exact upgrades within an existing byte allowance.

This is a precision ablation, not prototype merging or residual repair. It
changes no source memberships, K/V positions, model parameters or history.
"""

from dataclasses import dataclass, replace
from typing import Any

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch, decoded, encode
from mosaickv.evaluation.head_local_precision import HeadLocalFormat, cpu_packet
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@dataclass(frozen=True)
class ValueRowFormat:
    base: HeadLocalFormat
    positions: Any
    values: Any = None
    kind: str = "evicastkv_per_query_value_row_precision_variant"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.base.shape

    def private(self) -> tuple[Any, ...]:
        return (
            *self.base.private(),
            self.positions,
            *(self.values.buffers if self.values is not None else ()),
        )

    def shared(self) -> tuple[Any, ...]:
        return self.base.shared()

    def tensors(self) -> tuple[Any, ...]:
        return self.private()

    @property
    def component_bytes(self) -> int:
        return backing_bytes(self.private())

    def fill(self, attention: Any, destinations: Any, *, direct: bool) -> None:
        self.base.fill(attention, destinations, direct=direct)
        if self.values is None:
            return
        if destinations[1].device.type == "cuda":
            from mosaickv.evaluation.head_local_precision_kernel import scatter

            scatter(self.values, self.positions, destinations[1], prompt_length=self.shape[-2])
        else:
            n = self.shape[-2]
            destinations[1][0, self.positions.long() // n, self.positions.long() % n] = decoded(
                self.values
            )[0, 0]

    def materialize(self, attention: Any = None) -> tuple[Any, ...]:
        output = tuple(
            torch.empty(self.shape, device=self.positions.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, output, direct=True)
        return output


def upgrade_rows(
    packet: HeadLocalFormat,
    pair: Any,
    probabilities: Any,
    memberships: Any,
) -> tuple[ValueRowFormat, dict[str, Any]]:
    """Spend the old scalar-V patch bytes on prompt-demanded original V rows.

    The max across individual queries avoids pooled-query cancellation. It is
    a prospective reconstruction score, not a guarantee of task improvement.
    """
    if len(packet.base.corrections) != 2:
        raise ValueError("an existing funded value-exception allowance is required")
    old = packet.base.corrections[1]
    budget = backing_bytes(old.tensors)
    empty = SparseBitPatch(old.indices[:0].clone(), old.values[:0].clone())
    base = replace(
        packet, base=replace(packet.base, corrections=(packet.base.corrections[0], empty))
    )
    h, n, d = packet.shape[1:]
    reference = pair[1]
    error = (reference.float() - base.materialize()[1].float()).square().sum(-1)
    if (
        probabilities.ndim != 4
        or probabilities.shape[0] != 1
        or probabilities.shape[1] != h
        or not bool(torch.isfinite(probabilities).all())
        or bool((probabilities < 0).any())
    ):
        raise ValueError("finite nonnegative per-query eager attention required")
    index = torch.tensor(memberships, device=reference.device, dtype=torch.long)[None, :, None, :]
    demand = (
        probabilities[..., -16:, :]
        .float()
        .gather(-1, index.expand(1, h, min(16, probabilities.shape[-2]), n))
    )
    demand = demand / demand.sum(-1, keepdim=True).clamp_min(1e-20)
    score = demand.square().amax(-2) * error
    ranking = torch.argsort(score.flatten(), descending=True, stable=True)
    ranking = ranking[score.flatten()[ranking] > 0]
    raw_rows = reference.reshape(1, 1, h * n, d)

    def candidate(count: int) -> ValueRowFormat:
        positions = ranking[:count].to(torch.int32).clone()
        values = encode(raw_rows.index_select(2, positions.long()).contiguous()) if count else None
        return ValueRowFormat(base, positions, values)

    # For a fixed ranked prefix, the lossless codec's SM/header/literal count
    # and raw fallback sizes are nondecreasing. Binary search uses actual bytes,
    # not a guessed per-row compression factor. Retain rank order when encoding.
    lower, upper = 0, min(int(ranking.numel()), budget // (d + 4))
    best = candidate(0)
    base_bytes = base.component_bytes
    while lower <= upper:
        middle = (lower + upper) // 2
        trial = candidate(middle)
        if trial.component_bytes - base_bytes <= budget:
            best, lower = trial, middle + 1
        else:
            upper = middle - 1
    if best.component_bytes > packet.component_bytes:
        raise RuntimeError("whole-row upgrades exceeded the old scalar exception bytes")
    following = (
        candidate(int(best.positions.numel()) + 1).component_bytes - base_bytes
        if best.positions.numel() < ranking.numel()
        else None
    )
    if following is not None and following <= budget:
        raise RuntimeError("row-capacity search left a feasible next row unused")
    return best, dict(
        old_scalar_exception_bytes=budget,
        actual_row_upgrade_bytes=best.component_bytes - base_bytes,
        exact_value_rows=int(best.positions.numel()),
        scalar_values_covered=int(best.positions.numel()) * d,
        previous_scalar_values=int(old.indices.numel()),
        next_prefix_bytes=following,
        no_extra_bytes=True,
        no_key_or_source_position_changes=True,
        score="max_individual_prompt_query_demand_squared_times_value_error_norm_squared",
        target_or_reference_answers_used=False,
    )


def cpu_row_packet(packet: ValueRowFormat) -> ValueRowFormat:
    return replace(
        packet,
        base=cpu_packet(packet.base),
        positions=packet.positions.cpu().clone(),
        values=None
        if packet.values is None
        else replace(packet.values, buffers=tuple(t.cpu().clone() for t in packet.values.buffers)),
    )
