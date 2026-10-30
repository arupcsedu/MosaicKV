"""Exact-corrected fused projection feasibility, not an enabled paper method."""

from dataclasses import dataclass, replace
from typing import Any

import torch

from mosaickv.evaluation.compact_storage import CompactFormat
from mosaickv.evaluation.exact_storage import ExactFormat, SparseBitPatch, decoded


def output_slots(union: Any, selected: Any) -> Any:
    """Map original source rows to unique head-local physical output slots."""
    if (
        selected.ndim != 2
        or selected.shape[-1] >= 32768
        or not selected.shape[-1]
        or union.ndim != 1
        or union.dtype not in (torch.int32, torch.int64)
        or selected.dtype not in (torch.int32, torch.int64)
        or bool((union < 0).any())
        or bool((selected < 0).any())
        or bool((union >= 2**31).any())
        or bool((selected >= 2**31).any())
    ):
        raise ValueError("uniform int16-addressable per-head capacity required")
    if not union.numel() or not bool((union[1:] > union[:-1]).all()):
        raise ValueError("nonempty sorted unique source union required")
    if not bool((selected[:, 1:] > selected[:, :-1]).all()):
        raise ValueError("sorted unique logical positions required per head")
    if selected.stride(0) == 0 and torch.equal(selected[0], union):
        return torch.arange(union.numel(), device=union.device, dtype=torch.int16)[None].expand(
            selected.shape[0], -1
        )
    index = torch.searchsorted(
        selected.contiguous(), union[None].expand(selected.shape[0], -1).contiguous()
    )
    index.clamp_(max=selected.shape[-1] - 1)
    return torch.where(selected.gather(1, index) == union[None], index, -1).to(torch.int16)


@dataclass(frozen=True)
class DualFormat(ExactFormat):
    output_slots: Any = None

    def private(self) -> tuple[Any, ...]:
        return (*super().private(), self.output_slots)

    def materialize(self, attention: Any) -> tuple[Any, ...]:
        outputs = tuple(
            torch.empty(self.shape, device=self.archive.union.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, outputs, direct=True)
        return outputs

    def fill(self, attention: Any, destinations: Any, *, direct: bool) -> None:
        from mosaickv.evaluation.dual_projection_kernel import apply_patch, project

        if self.archive is None or self.output_slots is None:
            raise ValueError("complete original-source archive required")
        targets = tuple(t[..., : self.shape[-2], :] for t in destinations)
        with torch.inference_mode():
            source = decoded(self.archive.source).squeeze(1)
            project(source, attention, self.archive, self.output_slots, targets)
            del source
            CompactFormat.overlay(self, targets)
            for patch, target in zip(self.patches, targets, strict=True):
                apply_patch(patch, target)


def build_dual(
    base: ExactFormat, attention: Any, references: Any
) -> tuple[DualFormat, dict[str, Any]]:
    if base.archive is None:
        raise ValueError("original source required")
    archive = base.archive
    union = archive.union.to(torch.int32)
    selected = (
        union[None].expand(archive.selected.shape[0], -1)
        if archive.selected.stride(0) == 0
        else archive.selected.to(torch.int32)
    )
    narrow = replace(archive, union=union, selected=selected)
    coordinates = (
        None
        if base.slots is None
        else torch.stack((base.slots // base.shape[-2], base.slots % base.shape[-2])).to(
            torch.int32
        )
    )
    result = DualFormat(
        "dual_projection",
        base.shape,
        base.pair,
        narrow,
        coordinates,
        (),
        output_slots(union, selected),
    )
    actual = result.materialize(attention)
    patches = tuple(SparseBitPatch.build(a, b) for a, b in zip(references, actual, strict=True))
    result = DualFormat(
        result.kind, result.shape, result.pair, narrow, coordinates, patches, result.output_slots
    )
    for _ in range(3):
        actual = result.materialize(attention)
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(references, actual, strict=True)
        ):
            raise RuntimeError("dual projection exact corrections did not reproduce original KV")
    return result, dict(
        kind=result.kind,
        source_tokens=union.numel(),
        changed_elements=[p.indices.numel() for p in patches],
        correction_storage_bytes=sum(
            t.numel() * t.element_size() for p in patches for t in p.tensors
        ),
        total_component_bytes=result.component_bytes,
        repeated_original_KV_bitwise=True,
        repetitions=3,
    )


def gpu_controls() -> dict[str, Any]:
    from types import SimpleNamespace

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.projection_archive import ProjectionArchive

    torch.manual_seed(20261030)
    source = torch.randn(1, 5, 256, device="cuda").bfloat16()
    k_proj = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    v_proj = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    attention = SimpleNamespace(k_proj=k_proj, v_proj=v_proj, head_dim=128)
    phase = torch.randn(1, 9, 128, device="cuda")
    union = torch.tensor([0, 2, 4, 6, 8], device="cuda")
    selected = torch.tensor([[0, 4, 8], [2, 4, 6]], device="cuda")
    archive = ProjectionArchive(
        PackedBF16((1, 1, 5, 256), True, (source[:, None],)),
        union,
        selected,
        9,
        phase.cos().bfloat16(),
        phase.sin().bfloat16(),
    )
    with torch.inference_mode():
        key = k_proj(source).view(1, 5, 2, 128).transpose(1, 2)
        value = v_proj(source).view(1, 5, 2, 128).transpose(1, 2)
        _, key = apply_rotary_pos_emb(key[:, :1], key, archive.cos[:, union], archive.sin[:, union])
        index = torch.searchsorted(union, selected)[None, :, :, None].expand(1, 2, 3, 128)
        reference = tuple(t.gather(2, index) for t in (key, value))
        base = ExactFormat("source", (1, 2, 3, 128), archive=archive)
        format, proof = build_dual(base, attention, reference)
        backing = tuple(
            torch.full((1, 2, 11, 128), -71, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        )
        targets = tuple(t[..., 3:6, :] for t in backing)
        format.fill(attention, targets, direct=True)
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(targets, reference, strict=True)
        ):
            raise RuntimeError("dual projection strided bit equality failed")
        if not all(
            bool((b[..., :3, :] == -71).all() and (b[..., 6:, :] == -71).all()) for b in backing
        ):
            raise RuntimeError("dual projection overwrote strided lease sentinels")
    return dict(strided_original_bits=True, sentinels_unchanged=True, **proof)
