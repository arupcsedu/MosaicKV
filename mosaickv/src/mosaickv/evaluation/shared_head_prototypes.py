"""Experimental all-consumer SOURCE prototypes, not additive K/V overlays.

One map owns every input channel, hence every K/V head. Prototypes combine
integer source codes only WITHIN one affine-quantizer group. The original
headers and per-position RoPE remain unchanged. This changes source values;
it is not a lossless codec or the validated storage-only method.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.source_replacement import validate


def integers(codes, rows, width):
    """Device-independent oracle; bounded construction, never decode hot path."""
    raw = codes.reshape(-1, 3).to(torch.int32)
    word = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    shifts = torch.arange(8, device=codes.device, dtype=torch.int32) * 3
    return ((word[:, None] >> shifts) & 7).reshape(width, rows).transpose(0, 1)


def serialize(values):
    x = values.transpose(0, 1).contiguous().reshape(-1, 8).to(torch.int32)
    shifts = torch.arange(8, device=x.device, dtype=torch.int32) * 3
    word = (x << shifts).sum(-1, dtype=torch.int32)
    return torch.stack(
        tuple(((word >> (8 * i)) & 255).to(torch.uint8) for i in range(3)), -1
    ).flatten()


@dataclass
class SharedHeadReplacement:
    body: object
    residual_codes: torch.Tensor
    residual_positions: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    original_owner: object
    heads: int
    original_storage_bytes: int
    changed_code_symbols: int = 0
    repair_events: int = 0

    def record(self):
        return dict(
            variant="shared_all_head_source_prototype_v1",
            affected_KV_heads=list(range(self.heads)),
            affected_components=["key", "value"],
            source_channels=self.body.shape[-1],
            original_positions=self.body.shape[-2],
            physical_source_rows=self.body.stored_rows or self.body.shape[-2],
            prototype_groups=[list(g) for g in self.groups],
            old_source_tensor_alive=self.original_owner() is not None,
            old_source_backing_retirement_requires_allocator_ledger=True,
            original_body_bytes=self.original_storage_bytes,
            replacement_body_bytes=backing_bytes(self.body.tensors()),
            CPU_residual_bytes=backing_bytes((self.residual_codes,)),
            residual_pinned=bool(self.residual_codes.is_pinned()),
            repair_events=self.repair_events,
            changed_code_symbols=self.changed_code_symbols,
            prototype_domain="affine_source_codes_before_all_KV_projections_and_RoPE",
            no_post_RoPE_key_averaging=True,
            logical_positions_preserved=True,
            benchmark_labels_accessed=False,
            model_quality_established=False,
        )

    @torch.inference_mode()
    def promote(self, *, captured=False):
        """All-head repair commits a NEW owner; never resize a captured replay.

        The caller must rebind its private codec/reader at a decode transaction
        boundary. This primitive alone does not certify online model repair.
        """
        if captured or self.repair_events:
            raise RuntimeError("one uncaptured, request-private source repair required")
        _n, width = self.body.shape[2:]
        code = integers(self.body.codes, self.body.stored_rows, width)
        full = code.index_select(0, self.body.row_map.long())
        residual = integers(
            self.residual_codes.to(full.device, non_blocking=True),
            len(self.residual_positions),
            width,
        )
        ids = torch.tensor(self.residual_positions, device=full.device, dtype=torch.long)
        full.index_copy_(0, ids, residual)
        restored = replace(
            self.body, codes=serialize(full), row_map=None, stored_rows=0, row_map_shared=False
        )
        validate(restored)
        self.body, self.repair_events = restored, 1
        return restored


@torch.inference_mode()
def construct(body, groups, *, heads, mandatory=(), retention_one=False):
    """Retire the original all-head code owner, not merely one head's view.

    Only body headers are aliased; neither original code tensor nor a view of
    its storage survives in this object. Residual codes use separate CPU
    storage. Actual complete-codec savings and model fidelity are external gates.
    """
    validate(body)
    if type(heads) is not int or heads < 1 or body.shape[-1] % heads:
        raise ValueError("complete divisible K/V head coverage required")
    if type(retention_one) is not bool:
        raise ValueError("explicit retention-one control required")
    if retention_one:
        if groups:
            raise ValueError("retention one forbids prototype transformations")
        return body
    if body.row_map is not None:
        raise ValueError("one original source owner required; no nested replacements")
    n, width = body.shape[2:]
    groups = tuple(tuple(g) for g in groups)
    mandatory = tuple(mandatory)
    if any(type(i) is not int or not 0 <= i < n for i in mandatory):
        raise ValueError("valid original mandatory positions required")
    members = [i for g in groups for i in g]
    if (
        not groups
        or any(len(g) < 2 or tuple(sorted(set(g))) != g for g in groups)
        or any(type(i) is not int or not 0 <= i < n for i in members)
        or len(set(members)) != len(members)
        or set(members) & set(mandatory)
        or any(g[0] // body.group != g[-1] // body.group for g in groups)
    ):
        raise ValueError("disjoint sorted nonmandatory groups in the same quantizer group required")
    codes = integers(body.codes, n, width)
    residual = serialize(codes[members]).to(device="cpu", copy=True).contiguous()
    if body.codes.is_cuda:
        residual = residual.pin_memory()
    mapping = torch.arange(n, device=codes.device, dtype=torch.int32)
    removed = {i for g in groups for i in g[1:]}
    retained = [i for i in range(n) if i not in removed]
    compact = codes[retained].clone()
    location = {p: i for i, p in enumerate(retained)}
    changed = 0
    for g in groups:
        # Equal weighted FP32 source-code barycenter. Shared headers make this
        # an affine-source prototype; rounding is an EXPLICIT approximation.
        compact[location[g[0]]] = codes[list(g)].float().mean(0).round().to(torch.int32)
        changed += int((codes[list(g)] != compact[location[g[0]]]).sum())
        mapping[list(g)] = location[g[0]]
    for p in retained:
        if p not in members:
            mapping[p] = location[p]
    out = replace(
        body,
        codes=serialize(compact),
        row_map=mapping,
        stored_rows=len(retained),
        row_map_shared=False,
    )
    validate(out)
    if out.codes.untyped_storage().data_ptr() == body.codes.untyped_storage().data_ptr():
        raise RuntimeError("replacement illegally aliases original all-head source")
    return SharedHeadReplacement(
        out,
        residual,
        tuple(members),
        groups,
        weakref.ref(body.codes),
        heads,
        backing_bytes(body.tensors()),
        changed_code_symbols=changed,
    )
