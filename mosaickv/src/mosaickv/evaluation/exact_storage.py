"""Experimental exact representations, independent of selector and task labels.

This is an isolated storage diagnostic, not an enabled paper method. Original
selected memberships never change. Actual buffers, indices and patches count.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

import torch

from mosaickv.evaluation.lossless_bf16 import PackedBF16, pack_cpu, unpack_into
from mosaickv.evaluation.lossless_prompt_cache import PromptStorageLayer, backing_bytes
from mosaickv.evaluation.projection_archive import (
    ProjectionArchive,
    build_archive,
    reconstruct_archive,
    reconstruct_compact_archive,
)
from mosaickv.evaluation.projection_cache import ProjectionCache, ScratchAudit, identity


def token_partition(positions: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Raw-payload proposal only; actual serialized cost selects the format."""
    if (
        not positions
        or len({len(h) for h in positions}) != 1
        or any(
            not h
            or tuple(sorted(set(h))) != tuple(h)
            or any(type(p) is not int or p < 0 for p in h)
            for h in positions
        )
    ):
        raise ValueError("uniform sorted unique nonnegative head positions required")
    degree = Counter(p for h in positions for p in h)
    source = tuple(sorted(p for p, n in degree.items() if 2 * n > len(positions)))
    shared = set(source)
    capacity = len(positions[0])
    direct = tuple(
        h * capacity + j
        for h, row in enumerate(positions)
        for j, p in enumerate(row)
        if p not in shared
    )
    return source, direct


def encode(tensor: Any) -> PackedBF16:
    if tensor.device.type == "cuda":
        from mosaickv.evaluation.lossless_bf16_kernel import encode as pack_gpu

        gpu_packer: Any = pack_gpu
        return cast(PackedBF16, gpu_packer(tensor.contiguous()))
    return pack_cpu(tensor)


def decoded(body: PackedBF16) -> Any:
    output = torch.empty(body.shape, device=body.buffers[0].device, dtype=torch.bfloat16)
    unpack_into(body, output)
    return output


def gpu_controls() -> list[dict[str, Any]]:
    """Validate encoded/raw CUDA writes into interior strided history windows."""
    from mosaickv.evaluation.lossless_bf16_kernel import encode as pack_gpu
    from mosaickv.evaluation.projection_rope_kernel import gpu_control

    results = gpu_control()
    gpu_packer: Any = pack_gpu
    generator = torch.Generator(device="cuda").manual_seed(20261030)
    tensor = torch.randn((1, 3, 7, 16), device="cuda", generator=generator).bfloat16()
    for raw in (False, True):
        packed = (
            PackedBF16(tuple(tensor.shape), True, (tensor.clone(),))
            if raw
            else gpu_packer(tensor, raw_fallback=False)
        )
        backing = torch.full((1, 3, 19, 16), -71, device="cuda", dtype=torch.bfloat16)
        target = backing[..., 4:11, :]
        unpack_into(packed, target)
        equal = torch.equal(target.view(torch.int16), tensor.view(torch.int16))
        unchanged = bool((backing[..., :4, :] == -71).all()) and bool(
            (backing[..., 11:, :] == -71).all()
        )
        if not equal or not unchanged:
            raise RuntimeError("strided CUDA decode corrupted a head or surrounding history")
        results.append(
            dict(codec_raw=raw, bitwise_equal=equal, surrounding_sentinels_unchanged=unchanged)
        )
    return results


@dataclass(frozen=True)
class SparseBitPatch:
    indices: Any
    values: Any

    @classmethod
    def build(cls, reference: Any, candidate: Any) -> SparseBitPatch:
        if (
            reference.dtype != torch.bfloat16
            or candidate.dtype != torch.bfloat16
            or reference.shape != candidate.shape
            or reference.device != candidate.device
            or reference.numel() >= 2**31
        ):
            raise ValueError("matching BF16 tensors with int32-addressable elements required")
        original = reference.contiguous().view(torch.int16).flatten()
        other = candidate.contiguous().view(torch.int16).flatten()
        indices = (original != other).nonzero().flatten()
        return cls(indices.to(torch.int32), original.index_select(0, indices).view(torch.bfloat16))

    @property
    def tensors(self) -> tuple[Any, ...]:
        return self.indices, self.values

    def apply(self, candidate: Any) -> None:
        if candidate.dtype != torch.bfloat16 or not candidate.is_contiguous():
            raise ValueError("patching requires contiguous BF16 candidate")
        with torch.inference_mode():
            candidate.view(torch.int16).flatten().index_copy_(
                0, self.indices.long(), self.values.view(torch.int16)
            )


@dataclass(frozen=True)
class ExactFormat:
    kind: str
    shape: tuple[int, ...]
    pair: tuple[PackedBF16, ...] = ()
    archive: ProjectionArchive | None = None
    slots: Any = None
    patches: tuple[SparseBitPatch, ...] = ()

    def private(self) -> tuple[Any, ...]:
        values = tuple(t for b in self.pair for t in b.buffers)
        if self.archive is not None:
            a = self.archive
            values += (*a.source.buffers, a.union, a.selected)
        if self.slots is not None:
            values += (self.slots,)
        return (*values, *(t for p in self.patches for t in p.tensors))

    def shared(self) -> tuple[Any, ...]:
        return (self.archive.cos, self.archive.sin) if self.archive is not None else ()

    def tensors(self) -> tuple[Any, ...]:
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self) -> int:
        return backing_bytes(self.tensors())

    def materialize(self, attention: Any) -> tuple[Any, ...]:
        if self.kind == "kv":
            return tuple(decoded(b) for b in self.pair)
        assert self.archive is not None
        if self.kind == "compact_patched":
            output = reconstruct_compact_archive(self.archive, attention)
            for patch, value in zip(self.patches, output, strict=True):
                patch.apply(value)
        else:
            output = reconstruct_archive(self.archive, attention)
            self.overlay(output)
        return output

    def overlay(self, outputs: tuple[Any, ...]) -> None:
        if self.slots is None or self.slots.numel() == 0:
            return
        for body, output in zip(self.pair, outputs, strict=True):
            value = decoded(body).reshape(-1, self.shape[-1])
            # Output may be a strided prefix view inside a longer layer lease.
            heads = self.slots // self.shape[-2]
            positions = self.slots % self.shape[-2]
            with torch.inference_mode():
                output[0, heads, positions, :] = value

    def fill(self, attention: Any, destinations: tuple[Any, ...], *, direct: bool) -> None:
        prefix = tuple(t[..., : self.shape[-2], :] for t in destinations)
        if not direct or self.kind == "compact_patched":
            outputs = self.materialize(attention)
            for source, target in zip(outputs, prefix, strict=True):
                target.copy_(source)
            return
        if self.kind == "kv":
            for body, target in zip(self.pair, prefix, strict=True):
                unpack_into(body, target)
        else:
            from mosaickv.evaluation.projection_into import reconstruct_into

            assert self.archive is not None
            reconstruct_into(self.archive, attention, prefix)
            self.overlay(prefix)


def formats_for_layer(
    record: Any,
    attention: Any,
    key: Any,
    value: Any,
    positions: Any,
    *,
    patch_repeats: int = 0,
    only_kv: bool = False,
    include_token: bool = True,
) -> tuple[list[ExactFormat], dict[str, Any]]:
    token_partition(positions)
    selected = torch.tensor(positions, device=key.device, dtype=torch.int64)
    index = selected[None, :, :, None].expand(1, key.shape[1], len(positions[0]), key.shape[-1])
    reference = key.gather(2, index), value.gather(2, index)
    shape = tuple(reference[0].shape)
    kv = ExactFormat("kv", shape, pair=tuple(encode(t) for t in reference))
    if only_kv:
        return [kv], dict(
            candidate_component_bytes={"kv": kv.component_bytes},
            memberships_unchanged=True,
            unused_source_formats_constructed=False,
        )
    source = build_archive(record, positions, encoded=True)
    all_source = ExactFormat("source", shape, archive=source)
    choices = [kv, all_source]
    high, direct = token_partition(positions)
    if high and include_token:
        union = torch.tensor(high, device=key.device, dtype=torch.int64)
        hidden = record["normalized"].index_select(1, union).unsqueeze(1).contiguous()
        archive = ProjectionArchive(
            encode(hidden), union, source.selected, source.original_length, source.cos, source.sin
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
        choices.append(ExactFormat("token_hybrid", shape, pair=pair, archive=archive, slots=slots))
    result: dict[str, Any] = dict(
        candidate_component_bytes={f.kind: f.component_bytes for f in choices},
        source_rows=len(high),
        direct_head_token_pairs=len(direct),
        memberships_unchanged=True,
        raw_threshold_is_only_a_proposal=True,
        actual_encoded_cost_controls_choice=True,
    )
    if patch_repeats:
        if type(patch_repeats) is not int or patch_repeats < 3:
            raise ValueError("at least three deterministic patch checks required")
        compact = reconstruct_compact_archive(source, attention)
        patches = tuple(SparseBitPatch.build(a, b) for a, b in zip(reference, compact, strict=True))
        candidate = ExactFormat("compact_patched", shape, archive=source, patches=patches)
        for _ in range(patch_repeats):
            actual = candidate.materialize(attention)
            if not all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(reference, actual, strict=True)
            ):
                raise RuntimeError("fixed compact patches do not reproduce original K/V bits")
        choices.append(candidate)
        result.update(
            patch_entries=[int(p.indices.numel()) for p in patches],
            patch_storage_bytes=backing_bytes(tuple(t for p in patches for t in p.tensors)),
            patch_checked_repetitions=patch_repeats,
            patch_component_bytes=candidate.component_bytes,
            patch_repeat_bitwise=True,
        )
    # Every candidate, including mixed low-degree direct rows, must match exactly.
    for candidate in choices:
        actual = candidate.materialize(attention)
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(reference, actual, strict=True)
        ):
            raise RuntimeError(f"{candidate.kind} changed an original K/V bit")
    return choices, result


class ExactLayer(PromptStorageLayer):
    def __init__(
        self,
        representation: ExactFormat,
        attention: Any,
        *,
        observe: Any,
        direct: bool,
        updates: int,
        audit: bool,
    ) -> None:
        from transformers.cache_utils import CacheLayerMixin

        cast(Any, CacheLayerMixin).__init__(self)
        self.representation, self.attention = representation, attention
        self.direct, self.audit = direct, audit
        self.workspace_operations = 0
        self.weights = {identity(t) for t in attention.parameters()}
        device = representation.private()[0].device
        self._initialize_storage(
            representation.shape,
            device,
            backing_bytes(representation.private()),
            observe=observe,
            fused_unpack=False,
            maximum_decode_updates=updates,
        )

    def owned_tensors(self) -> tuple[Any, ...]:
        return (*self.representation.tensors(), *self.history, *self.leased)

    def materialize(self) -> None:
        from contextlib import nullcontext

        mode = (
            ScratchAudit(
                excluded=self.weights | {identity(t) for t in self.owned_tensors()},
                observe=self.observe,
            )
            if self.audit
            else None
        )
        with mode if mode is not None else nullcontext():
            from mosaickv.evaluation.native_source_chunks import source_packet

            if source_packet(self.representation) is not None:
                self.representation.fill(
                    self.attention, self.leased, direct=self.direct, workspace_observer=self.observe
                )
            else:
                self.representation.fill(self.attention, self.leased, direct=self.direct)
            self.fill_history()
        if mode is not None:
            self.workspace_operations += mode.operations

    def fill_history(self) -> None:
        for history, dest in zip(self.history, self.leased, strict=True):
            dest[..., self.prompt_length :, :].copy_(history[..., : self.history_length, :])


class ExactCache(ProjectionCache):
    def storage_record(self) -> dict[str, Any]:
        row = super().storage_record()
        row.update(
            boundary="post_construction_exact_format_history_and_observed_tensor_workspace",
            includes="exact_encoded_payload,source_and_direct_slots,patches,shared_rotary,history,lease,observed_tensor_workspace",
            generated_history_storage=[
                dict(
                    sealed_chunks=len(getattr(layer, "sealed", ())),
                    raw_tail_used=getattr(
                        layer, "tail_used", cast(ExactLayer, layer).history_length
                    ),
                )
                for layer in self.layers
            ],
            unchanged_original_selected_memberships=True,
        )
        return row


def construct(
    adapter: Any,
    prefill: Any,
    capture: Any,
    positions: Any,
    *,
    mode: str = "token",
    direct: bool = False,
    sealed_history: bool = False,
    updates: int = 32,
    audit: bool = True,
) -> ExactCache:
    if mode not in ("kv", "token", "patched"):
        raise ValueError("explicit kv, token or patched storage mode required")
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    if len(source) != len(positions):
        raise ValueError("complete layer membership required")
    cache = ExactCache(audit_workspace=audit)
    shared: list[Any] = []
    for layer_id, ((key, value), selected) in enumerate(zip(source, positions, strict=True)):
        attention = adapter._language_layers()[layer_id].self_attn
        choices, proof = formats_for_layer(
            capture.records[layer_id],
            attention,
            key,
            value,
            selected,
            patch_repeats=3 if mode == "patched" else 0,
            only_kv=mode == "kv",
            include_token=mode == "token",
        )
        allowed = [
            f
            for f in choices
            if (
                f.kind == "kv"
                if mode == "kv"
                else f.kind in ("kv", "compact_patched")
                if mode == "patched"
                else f.kind in ("kv", "source", "token_hybrid")
            )
        ]
        chosen = min(allowed, key=lambda f: (f.component_bytes, f.kind != "kv", f.kind))
        shared.extend(chosen.shared())
        cls = ExactLayer
        if sealed_history:
            from mosaickv.evaluation.sealed_history import SealedHistoryLayer

            cls = SealedHistoryLayer
        layer = cls(
            chosen, attention, observe=cache.observe, direct=direct, updates=updates, audit=audit
        )
        cache.layers.append(layer)
        cache.build_records.append(
            dict(
                layer=layer_id, format=chosen.kind, component_bytes=chosen.component_bytes, **proof
            )
        )
    cache.shared_tensors = tuple(shared)
    cache.shared_bytes = backing_bytes(cache.shared_tensors)
    cache.initial_bytes = backing_bytes(
        tuple(t for layer in cache.layers for t in cast(ExactLayer, layer).owned_tensors())
    )
    cache.observe()
    return cache
