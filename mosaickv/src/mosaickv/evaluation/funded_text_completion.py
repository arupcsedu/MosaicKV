"""Restore a complete generation delimiter from already archived text, no donors.

Explicit padded-layout development variant. All physical padding and compact
validity metadata count toward measured cache storage; it is NOT a zero-padding
gate pass. Original decoded K/V are preserved; added text is original BF16.
"""

from dataclasses import dataclass, replace
from typing import Any

import torch

from mosaickv.evaluation.cache_cycles import compile_cycles
from mosaickv.evaluation.exact_storage import SparseBitPatch, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.text_source_storage import (
    TextProjection,
    TextSourceFormat,
    optimize,
    strip_text_overlays,
    text_slots,
)


def generation_delimiter(prepared, tokenizer):
    ids = tuple(tokenizer.encode("ASSISTANT:", add_special_tokens=False))
    tokens = prepared.model_inputs["input_ids"][0].tolist()
    if not ids or tuple(tokens[-len(ids) :]) != ids:
        raise NotImplementedError("exact pinned ASSISTANT: generation suffix required")
    positions = tuple(range(len(tokens) - len(ids), len(tokens)))
    text = {p for s in prepared.modality_map if s.modality == "text" for p in range(s.start, s.end)}
    if not set(positions).issubset(text):
        raise ValueError("generation suffix must be authoritative text")
    return positions


def additions(positions, delimiter, source_length):
    delimiter = tuple(delimiter)
    if (
        not positions
        or len({len(h) for h in positions}) != 1
        or not delimiter
        or tuple(sorted(set(delimiter))) != delimiter
        or delimiter[-1] != source_length - 1
        or delimiter[0] < 0
        or any(
            not h
            or tuple(sorted(set(h))) != tuple(h)
            or h[0] < 0
            or h[-1] >= source_length
            or source_length - 1 not in h
            for h in positions
        )
    ):
        raise ValueError("sorted heads and exact original generation suffix required")
    return tuple(tuple(p for p in delimiter if p not in set(h)) for h in positions)


@dataclass(frozen=True)
class ExtendZeros:
    anchor: Any
    extra: int

    @property
    def shape(self):
        shape = list(self.anchor.shape)
        shape[-2] += self.extra
        return tuple(shape)

    def private(self):
        return self.anchor.private()

    def shared(self):
        return self.anchor.shared()

    def fill(self, attention, targets, *, direct):
        self.anchor.fill(attention, targets, direct=direct)
        start, end = self.anchor.shape[-2], self.shape[-2]
        for target in targets:
            target[..., start:end, :].zero_()


@dataclass(frozen=True)
class SortedCompletion:
    base: Any
    cycles: Any
    valid_prompt_counts: Any
    kind: str = "common_source_funded_delimiter_completion_ablation"

    def __post_init__(self):
        counts = self.valid_prompt_counts
        if (
            counts.shape != (self.base.shape[1],)
            or counts.dtype != torch.int16
            or counts.device != self.base.private()[0].device
            or not bool(((counts > 0) & (counts <= self.base.shape[-2])).all())
            or self.cycles.length != self.base.shape[-2]
        ):
            raise ValueError("audited positive per-head validity and cache-cycle shape required")

    @property
    def shape(self):
        return self.base.shape

    def private(self):
        return (*self.base.private(), *self.cycles.tensors(), self.valid_prompt_counts)

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill(self, attention, targets, *, direct):
        self.base.fill(attention, targets, direct=direct)
        self.cycles.apply(targets)

    def materialize(self, attention):
        output = tuple(
            torch.empty(self.shape, dtype=torch.bfloat16, device=self.valid_prompt_counts.device)
            for _ in range(2)
        )
        self.fill(attention, output, direct=True)
        return output


@torch.inference_mode()
def build_layer(packet, record, positions, text, delimiter, attention, source):
    extra = additions(positions, delimiter, source[0].shape[-2])
    count = max(map(len, extra))
    if not count:
        retained, proof = optimize(packet, record, positions, text, attention)
        return (
            retained,
            positions,
            dict(proof, added_tokens=0, original_selected_KV_bitwise_preserved=True),
        )
    original = packet.materialize(attention)
    old_n = packet.shape[-2]
    reference = tuple(
        torch.cat(
            (a, torch.zeros(*a.shape[:2], count, a.shape[-1], device=a.device, dtype=a.dtype)), 2
        )
        for a in original
    )
    for head, new in enumerate(extra):
        for target, raw in zip(reference, source, strict=True):
            if new:
                target[0, head, old_n : old_n + len(new)] = raw[0, head, list(new)]
    # Dummy SORT KEYS are never model logical positions. Invalid rows remain
    # zero, sort after every original source token, and are masked in attention.
    source_length = source[0].shape[-2]
    physical = tuple(
        tuple((*old, *new, *range(source_length, source_length + count - len(new))))
        for old, new in zip(positions, extra, strict=True)
    )
    base = ExtendZeros(strip_text_overlays(packet, original, positions, set(text)), count)
    hidden = record["normalized"]
    encoded = encode(hidden[:, None].contiguous())
    check = torch.empty(0, device=hidden.device, dtype=torch.int16)
    if hidden.device.type == "cuda":
        from mosaickv.evaluation.encoded_projection_kernel import checkpoints

        check = checkpoints(encoded)
    projection = TextProjection(
        encoded,
        torch.tensor(text, device=hidden.device, dtype=torch.int32),
        torch.arange(len(text), device=hidden.device, dtype=torch.int32),
        torch.tensor(text_slots(physical, text), device=hidden.device, dtype=torch.int16),
        check,
        record["cos"],
        record["sin"],
        base.shape,
    )
    candidate = TextSourceFormat(base, projection)
    actual = candidate.materialize(attention)
    patches = tuple(SparseBitPatch.build(a, b) for a, b in zip(reference, actual, strict=True))
    n, d = base.shape[-2:]
    mask = torch.tensor([[p in set(text) for p in h] for h in physical], device=hidden.device)
    if any(
        not bool(mask[p.indices.long() // (n * d), p.indices.long() // d % n].all())
        for p in patches
    ):
        raise RuntimeError("completion changed nontext/padding source values")
    candidate = replace(candidate, projection=replace(projection, patches=patches))
    cycles = compile_cycles(physical, device=hidden.device)
    counts = torch.tensor(
        [old_n + len(new) for new in extra], device=hidden.device, dtype=torch.int16
    )
    sorted_keys = tuple(tuple(sorted(p)) for p in physical)
    order = torch.tensor(
        [sorted(range(n), key=lambda j: p[j]) for p in physical], device=hidden.device
    )[None, :, :, None].expand(1, len(physical), n, d)
    expected = tuple(t.gather(2, order) for t in reference)
    result = SortedCompletion(candidate, cycles, counts)
    for _ in range(3):
        actual = result.materialize(attention)
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(expected, actual, strict=True)
        ):
            raise RuntimeError("sorted completion failed bitwise K/V reconstruction")
    retained_index = torch.tensor(
        [[row.index(p) for p in old] for row, old in zip(sorted_keys, positions, strict=True)],
        device=hidden.device,
    )[None, :, :, None].expand_as(original[0])
    if not all(
        torch.equal(a.view(torch.int16), b.gather(2, retained_index).view(torch.int16))
        for a, b in zip(original, actual, strict=True)
    ):
        raise RuntimeError("completion changed a previously selected cache value")
    memberships = tuple(tuple(p if p < source_length else None for p in row) for row in sorted_keys)
    return (
        result,
        memberships,
        dict(
            added_tokens=sum(map(len, extra)),
            extra_physical_capacity_per_head=count,
            physical_padding_slots=sum(count - len(new) for new in extra),
            no_donor_removal=True,
            original_selected_KV_bitwise_preserved=True,
            complete_generation_delimiter_every_head=True,
            original_logical_positions_not_renumbered=True,
            physical_source_memberships=memberships,
            actual_added_positions=extra,
            cycle_metadata_bytes=backing_bytes(cycles.tensors()),
            validity_metadata_bytes=backing_bytes((counts,)),
            patch_bytes=sum(backing_bytes(p.tensors) for p in patches),
            original_component_bytes=packet.component_bytes,
            candidate_component_bytes=result.component_bytes,
            virtual_zero_padding_materialized_and_counted_in_decode_peak=True,
        ),
    )


def transaction(packets, records, memberships, text, delimiter, attentions, source):
    candidates, final_positions, proofs = [], [], []
    for i, packet in enumerate(packets):
        candidate, positions, proof = build_layer(
            packet, records[i], memberships[i], text, delimiter, attentions[i], source[i]
        )
        candidates.append(candidate)
        final_positions.append(positions)
        proofs.append(dict(layer=i, **proof))
    original_bytes = backing_bytes(tuple(t for f in packets for t in f.tensors()))
    proposed_bytes = backing_bytes(tuple(t for f in candidates for t in f.tensors()))
    accepted = proposed_bytes <= original_bytes
    if not accepted:
        candidates = [
            optimize(p, records[i], memberships[i], text, attentions[i])[0]
            for i, p in enumerate(packets)
        ]
        final_positions = list(memberships)
    final_bytes = backing_bytes(tuple(t for f in candidates for t in f.tensors()))
    if final_bytes > original_bytes:
        raise RuntimeError("completion exceeded its original measured physical byte cap")
    return tuple(candidates), dict(
        layers=proofs,
        proposed_completion_accepted=accepted,
        original_physical_cap_bytes=original_bytes,
        proposed_physical_bytes=proposed_bytes,
        final_physical_bytes=final_bytes,
        no_fullkv_fallback=True,
        no_donor_removal=True,
        original_selected_KV_bitwise_preserved=True,
        no_benchmark_labels_used=True,
        final_source_memberships=final_positions,
        fallback=None if accepted else "source_only_no_capacity_increase",
        physical_padding_charged_not_zero_padding_claim=True,
        peak_gate_requires_actual_decode_not_initial_projection=True,
    )
