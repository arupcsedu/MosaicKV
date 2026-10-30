"""Construct only useful frozen-weight replay formats, preserving exact choices."""

from typing import Any

import torch

from mosaickv.evaluation.compact_storage import build_compact
from mosaickv.evaluation.dual_projection import build_dual
from mosaickv.evaluation.exact_storage import ExactFormat, decoded, encode, token_partition
from mosaickv.evaluation.projection_archive import ProjectionArchive


def base_formats(
    record: Any, key: Any, value: Any, positions: Any
) -> tuple[list[ExactFormat], Any]:
    """Same source/KV/hybrid payloads as formats_for_layer; no redundant GEMMs."""
    high, direct = token_partition(positions)
    selected = torch.tensor(positions, device=key.device, dtype=torch.int64)
    if key.shape != value.shape or key.dtype != torch.bfloat16 or selected.shape[0] != key.shape[1]:
        raise ValueError("matching BF16 MHA K/V and complete head membership required")
    if (
        max(h[-1] for h in positions) >= key.shape[-2]
        or record["normalized"].shape[1] != key.shape[-2]
    ):
        raise ValueError("selected source positions exceed original cache")
    index = selected[None, :, :, None].expand(1, key.shape[1], selected.shape[1], key.shape[-1])
    reference = tuple(t.gather(2, index) for t in (key, value))
    shape = tuple(reference[0].shape)
    kv = ExactFormat("kv", shape, pair=tuple(encode(t) for t in reference))
    for body, original in zip(kv.pair, reference, strict=True):
        if not torch.equal(decoded(body).view(torch.int16), original.view(torch.int16)):
            raise RuntimeError("direct constructor codec changed original KV")
    union_ids = tuple(sorted({p for head in positions for p in head}))
    union = torch.tensor(union_ids, device=key.device, dtype=torch.int64)
    if all(list(h) == list(positions[0]) for h in positions):
        selected = union[None].expand(len(positions), -1)
    source = encode(record["normalized"].index_select(1, union).unsqueeze(1).contiguous())
    archive = ProjectionArchive(
        source, union, selected, key.shape[-2], record["cos"], record["sin"]
    )
    choices = [kv, ExactFormat("source", shape, archive=archive)]
    if high:
        high_union = torch.tensor(high, device=key.device, dtype=torch.int64)
        high_source = (
            source
            if high == union_ids
            else encode(record["normalized"].index_select(1, high_union).unsqueeze(1).contiguous())
        )
        partial = ProjectionArchive(
            high_source, high_union, selected, key.shape[-2], record["cos"], record["sin"]
        )
        slots = torch.tensor(direct, device=key.device, dtype=torch.int64)
        pair = (
            tuple(
                encode(
                    t.reshape(-1, shape[-1])
                    .index_select(0, slots)
                    .reshape(1, 1, len(direct), shape[-1])
                )
                for t in reference
            )
            if direct
            else ()
        )
        choices.append(ExactFormat("token_hybrid", shape, pair=pair, archive=partial, slots=slots))
    return choices, reference


def construct(adapter: Any, prefill: Any, capture: Any, positions: Any) -> tuple[Any, list[Any]]:
    original, _ = adapter._legacy_layers(prefill.state.past_key_values)
    if len(original) != len(positions) or len(capture.records) != len(original):
        raise ValueError("complete original source and membership required")
    result, proof = [], []
    for layer, ((key, value), heads) in enumerate(zip(original, positions, strict=True)):
        attention = adapter._language_layers()[layer].self_attn
        candidates, reference = base_formats(capture.records[layer], key, value, heads)
        options = [candidates[0]]
        records = []
        for candidate in candidates[1:]:
            compact, compact_proof = build_compact(
                candidate, attention, reference, reduced_precision=False
            )
            options.append(compact)
            records.append(compact_proof)
            # Both policies have identical non-correction storage. A zero-byte
            # correction is the proven lower bound; the other policy cannot win.
            if compact_proof["correction_storage_bytes"]:
                alternative, alternative_proof = build_compact(
                    candidate, attention, reference, reduced_precision=True
                )
                options.append(alternative)
                records.append(alternative_proof)
            dual, dual_proof = build_dual(candidate, attention, reference)
            options.append(dual)
            records.append(dual_proof)
        chosen = min(options, key=lambda f: (f.component_bytes, f.kind))
        result.append(chosen)
        proof.append(
            dict(
                layer=layer,
                kind=chosen.kind,
                component_bytes=chosen.component_bytes,
                candidates=records,
                three_independent_original_KV_checks=True,
                unchanged_memberships=True,
                zero_correction_lower_bound_pruning=True,
            )
        )
    return tuple(result), proof
