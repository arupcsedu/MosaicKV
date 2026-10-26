"""Bounded one-head equal-capacity substitution POOL, not a quality shortlist.

Prompt tensors provide broad proposals only. Actual model draft logits must
construct the shortlist and the full horizon must authorize any acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mosaickv.selection.direct_error_exchange import _exchange_contexts


@dataclass(frozen=True)
class HeadSubstitution:
    layer: int
    head: int
    removed_position: int
    added_position: int
    replaced_bytes: int


def substitution_pool(
    full_state: Any, attention: dict, anchor: Any, *, prompt_bytes: int, limit: int
) -> tuple[HeadSubstitution, ...]:
    """Cover layers before revisiting them; no positive-prompt-gain filter.

    Within each layer, larger reconstruction residuals get first coverage.
    Each head proposes restoring its largest omitted attention-mass token and
    replacing its lowest-mass nonmandatory token. This is deliberately a pool,
    not a claim that the proposal improves prompt reconstruction or answers.
    """
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("head substitution pool must contain at most 16 proposals")
    contexts, _, _, _ = _exchange_contexts(
        full_state,
        attention,
        anchor.slot_records,
        prompt_budget_bytes=prompt_bytes,
        uniform_capacity=None,
    )
    queues = []
    for layer_index, (context, source) in enumerate(zip(contexts, full_state.layers, strict=True)):
        candidates = []
        mass = context.mass.detach().cpu().numpy()
        present = context.mask.detach().cpu().numpy()
        mandatory = context.mandatory.detach().cpu().numpy()
        errors = context.errors.detach().cpu().tolist()
        for head in range(source.kv_heads):
            removed = [
                p
                for p in range(source.sequence_length)
                if present[head, p] and not mandatory[head, p]
            ]
            omitted = [p for p in range(source.sequence_length) if not present[head, p]]
            if not removed or not omitted:
                continue
            d = min(removed, key=lambda p: (float(mass[head, p]), p))
            r = min(omitted, key=lambda p: (-float(mass[head, p]), p))
            cost = (
                source.key.shape[-1] * source.key.element_size()
                + source.value.shape[-1] * source.value.element_size()
            )
            candidates.append(
                (-errors[head], head, HeadSubstitution(layer_index, head, d, r, cost))
            )
        queues.append([row[2] for row in sorted(candidates, key=lambda row: row[:2])])
    # Evenly cover the model's depth if the pool is smaller than its layer count.
    layers = [i for i, queue in enumerate(queues) if queue]
    if len(layers) > limit:
        layers = [layers[round(i * (len(layers) - 1) / max(1, limit - 1))] for i in range(limit)]
    result = []
    for level in range(max((len(q) for q in queues), default=0)):
        for layer in layers:
            if level < len(queues[layer]):
                result.append(queues[layer][level])
                if len(result) == limit:
                    return tuple(result)
    return tuple(result)


def pack_substitution(full_state: Any, anchor: Any, proposal: HeadSubstitution) -> Any:
    """Alias untouched layers; copy one layer and replace one head verbatim."""
    from dataclasses import replace

    import torch

    from mosaickv.measurements.cache_lifecycle import CacheLifecycle

    if not anchor.all_slots_valid or anchor.prototype_slots or anchor.layerwise_logical_positions:
        raise ValueError("head substitution requires an unpadded exact MosaicKV anchor")
    source = full_state.layers[proposal.layer]
    if not 0 <= proposal.head < source.kv_heads:
        raise ValueError("invalid substitution head")
    mandatory = {
        p
        for b in full_state.blocks
        if b.mandatory and b.layer == proposal.layer and b.kv_head == proposal.head
        for p in b.physical_cache_indices
    }
    if proposal.removed_position in mandatory:
        raise ValueError("head substitution cannot remove mandatory evidence")
    logical = full_state.logical_positions.physical_to_logical
    physical = {position: i for i, position in enumerate(logical)}
    rows = [
        row
        for row in anchor.slot_records
        if (row["layer"], row["kv_head"]) == (proposal.layer, proposal.head)
    ]
    chosen = {physical[row["logical_position"]] for row in rows}
    if (
        proposal.removed_position not in chosen
        or proposal.added_position in chosen
        or not 0 <= proposal.added_position < source.sequence_length
    ):
        raise ValueError(
            "substitution must replace one selected position with one omitted position"
        )
    cost = (
        source.key.shape[-1] * source.key.element_size()
        + source.value.shape[-1] * source.value.element_size()
    )
    if proposal.replaced_bytes != cost:
        raise ValueError("head substitution physical-byte metadata mismatch")
    chosen.remove(proposal.removed_position)
    chosen.add(proposal.added_position)
    positions = sorted(chosen)
    old = anchor.snapshot.layers[proposal.layer]
    key, value = old.key.clone(), old.value.clone()
    key[0, proposal.head] = source.key[0, proposal.head].index_select(
        0, torch.tensor(positions, device=key.device)
    )
    value[0, proposal.head] = source.value[0, proposal.head].index_select(
        0, torch.tensor(positions, device=value.device)
    )
    layers = list(anchor.snapshot.layers)
    layers[proposal.layer] = replace(old, key=key, value=value)
    records = tuple(
        row
        for row in anchor.slot_records
        if (row["layer"], row["kv_head"]) != (proposal.layer, proposal.head)
    ) + tuple(
        dict(
            layer=proposal.layer,
            kv_head=proposal.head,
            slot=slot,
            logical_position=logical[p],
            physical_position=p,
            tier="exact",
        )
        for slot, p in enumerate(positions)
    )
    result = replace(
        anchor, snapshot=replace(anchor.snapshot, layers=tuple(layers)), slot_records=records
    )

    def size(choice):
        return CacheLifecycle.count(
            tuple((layer.key, layer.value) for layer in choice.snapshot.layers)
        )[1]

    if size(result) != size(anchor):
        raise ValueError("head substitution changed physical storage")
    return result
