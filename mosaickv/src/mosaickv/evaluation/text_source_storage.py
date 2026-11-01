"""Lossless shared-source storage for already protected text K/V.

All methods receive the same option. Merged baseline text is corrected to its
ACTUAL merged values; expensive patches cause a deterministic raw fallback.
No selections, model weights, attention, logical positions or K/V bits change.
This is a representation ablation, not a claim of benchmark superiority.
"""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch, decoded, encode
from mosaickv.evaluation.head_local_precision import HeadLocalFormat
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class TextSourceCapture:
    """Capture only text projection inputs and their ACTUAL original RoPE phase."""

    def __init__(self, adapter, prepared):
        self.adapter = adapter
        self.positions = tuple(
            p
            for span in prepared.modality_map
            if span.modality == "text"
            for p in range(span.start, span.end)
        )
        if not self.positions or prepared.logical_sequence_length - 1 not in self.positions:
            raise ValueError("authoritative text spans including final delimiter required")
        self.stack, self.records = ExitStack(), {}
        self.phases, self.phase_identity = None, None

    def __enter__(self):
        try:
            for i, block in enumerate(self.adapter._language_layers()):

                def before(_module, args, kwargs, *, index=i):
                    hidden = kwargs.get("hidden_states", args[0] if args else None)
                    rotary = kwargs.get("position_embeddings")
                    if hidden is None or rotary is None or index in self.records:
                        raise ValueError("one original text-prefill projection invocation required")
                    if hidden.dtype != torch.bfloat16 or hidden.shape[0] != 1:
                        raise ValueError("batch-one BF16 normalized input required")
                    ids = torch.tensor(self.positions, device=hidden.device, dtype=torch.long)
                    identity = tuple(
                        (x.data_ptr(), tuple(x.shape), tuple(x.stride())) for x in rotary
                    )
                    if self.phases is None:
                        self.phases = tuple(x.index_select(1, ids).contiguous() for x in rotary)
                        self.phase_identity = identity
                    elif identity != self.phase_identity:
                        raise ValueError("all LLaVA layers must share the original phase tensors")
                    self.records[index] = dict(
                        normalized=hidden.index_select(1, ids).detach().contiguous(),
                        cos=self.phases[0],
                        sin=self.phases[1],
                    )

                handle = block.self_attn.register_forward_pre_hook(before, with_kwargs=True)
                self.stack.callback(handle.remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def text_slots(positions, text):
    if (
        not positions
        or len({len(h) for h in positions}) != 1
        or tuple(sorted(set(text))) != tuple(text)
        or any(
            len(set(h)) != len(h) or any(type(p) is not int or p < 0 for p in h) for h in positions
        )
    ):
        raise ValueError("sorted text IDs and uniform unique head positions required")
    return tuple(tuple({p: j for j, p in enumerate(h)}.get(t, -1) for t in text) for h in positions)


def strip_text_overlays(packet, references, positions, text):
    """Drop only redundant exact text bodies; visual exact protection remains."""
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    if isinstance(packet, SharedSourceFormat):
        h, n, d = packet.shape[1:]
        if tuple(map(tuple, positions)) != (tuple(range(n)),) * h:
            raise ValueError("full original identity source positions required")
        ids = packet.row_positions.long()
        keep = torch.tensor(
            [int(p) % n not in text for p in ids.cpu().tolist()],
            device=ids.device,
            dtype=torch.bool,
        )
        if not keep.any():
            keep[0] = True  # retain the existing nonempty scatter contract
        retained = ids[keep]
        exact = tuple(
            encode(t[0, retained // n, retained % n].reshape(1, 1, -1, d).contiguous())
            for t in references
        )
        return replace(packet, row_positions=retained.to(torch.int32), exact=exact)
    if not isinstance(packet, HeadLocalFormat):
        raise NotImplementedError("explicit head-local common format required")
    protected = packet.base.exact_indices.cpu().tolist()
    keep = [j for j in protected if any(h[j] not in text for h in positions)]
    # Existing exact scatter requires a nonempty body. One redundant row costs
    # little and is safer than changing the shared decoder's empty-body API.
    if not keep:
        keep = protected[:1]
    if not keep:
        raise ValueError("original nonempty common exact overlay required")
    ids = torch.tensor(keep, dtype=torch.int32, device=references[0].device)
    bodies = tuple(encode(x.index_select(2, ids.long()).contiguous()) for x in references)
    return replace(
        packet,
        base=replace(packet.base, exact_indices=ids, exact_values=bodies),
        flat_tokens=torch.empty(0, device=ids.device, dtype=torch.int32),
        protected=(),
    )


@dataclass(frozen=True)
class TextProjection:
    source: Any
    original_text: Any
    union: Any
    slots: Any
    checkpoints: Any
    cos: Any
    sin: Any
    shape: tuple[int, ...]
    patches: tuple[Any, ...] = ()

    def private(self):
        return (
            *self.source.buffers,
            self.original_text,
            self.union,
            self.slots,
            self.checkpoints,
            *(t for patch in self.patches for t in patch.tensors),
        )

    def shared(self):
        return self.cos, self.sin

    @torch.inference_mode()
    def fill(self, attention, targets):
        targets = tuple(t[..., : self.shape[-2], :] for t in targets)
        if targets[0].device.type == "cuda":
            from mosaickv.evaluation.encoded_projection_kernel import project

            # Only the selected SHAPE is used by the audited projection kernel.
            # Physical destinations are the explicitly charged int16 slot map.
            # Local source-row indices address the captured ORIGINAL RoPE phase;
            # they never renumber a model token's logical/decode position.
            archive = SimpleNamespace(
                selected=SimpleNamespace(shape=self.shape[1:3]),
                union=self.union,
                cos=self.cos,
                sin=self.sin,
            )
            project(self.source, self.checkpoints, attention, archive, self.slots, targets)
            if self.patches:
                from mosaickv.evaluation.dual_projection_kernel import apply_patch

                for patch, target in zip(self.patches, targets, strict=True):
                    apply_patch(patch, target)
        else:
            # Reference only: production projects directly from encoded bytes
            # into the final strided lease, without a decoded source/KV buffer.
            x = decoded(self.source).squeeze(1)
            h, d = self.shape[1], self.shape[-1]
            key = attention.k_proj(x).reshape(1, -1, h, d).transpose(1, 2)
            value = attention.v_proj(x).reshape(1, -1, h, d).transpose(1, 2)
            rotated = torch.cat((-key[..., d // 2 :], key[..., : d // 2]), -1)
            key = key * self.cos[:, None] + rotated * self.sin[:, None]
            for head in range(h):
                valid = self.slots[head] >= 0
                for tensor, target in zip((key, value), targets, strict=True):
                    target[0, head, self.slots[head, valid].long()] = tensor[0, head, valid]
            for patch, target in zip(self.patches, targets if self.patches else (), strict=True):
                index = patch.indices.long()
                n = self.shape[-2]
                target[0, index // (n * d), index // d % n, index % d] = patch.values


@dataclass(frozen=True)
class TextSourceFormat:
    base: Any
    projection: TextProjection
    kind: str = "common_bit_exact_text_source_storage_ablation"

    @property
    def shape(self):
        return self.base.shape

    def private(self):
        return (*self.base.private(), *self.projection.private())

    def shared(self):
        return (*self.base.shared(), *self.projection.shared())

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill(self, attention, destinations, *, direct):
        self.base.fill(attention, destinations, direct=direct)
        self.projection.fill(attention, destinations)

    def materialize(self, attention):
        output = tuple(
            torch.empty(self.shape, device=self.projection.union.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        self.fill(attention, output, direct=True)
        return output


@torch.inference_mode()
def optimize(packet, record, positions, text, attention):
    """Use source+exact residuals only when smaller; otherwise keep ACTUAL KV."""
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    if not isinstance(packet, (HeadLocalFormat, SharedSourceFormat)):
        raise NotImplementedError("head-local exact-text format required")
    text = tuple(text)
    slots = text_slots(positions, text)
    hidden = record["normalized"]
    h, n, d = packet.shape[1:]
    if (
        d != 128
        or hidden.shape != (1, len(text), h * d)
        or hidden.dtype != torch.bfloat16
        or record["cos"].shape != (1, len(text), d)
        or record["sin"].shape != record["cos"].shape
        or n >= 32768
        or not text
    ):
        raise ValueError("complete BF16 full-MHA text source and original phase required")
    reference = packet.materialize(attention)
    base = strip_text_overlays(packet, reference, positions, set(text))
    device = hidden.device
    source = encode(hidden[:, None].contiguous())
    checkpoints = torch.empty(0, dtype=torch.int16, device=device)
    if device.type == "cuda":
        from mosaickv.evaluation.encoded_projection_kernel import checkpoints as make_checkpoints

        checkpoints = make_checkpoints(source)
    projection = TextProjection(
        source,
        torch.tensor(text, dtype=torch.int32, device=device),
        torch.arange(len(text), dtype=torch.int32, device=device),
        torch.tensor(slots, dtype=torch.int16, device=device),
        checkpoints,
        record["cos"],
        record["sin"],
        packet.shape,
    )
    proposed = TextSourceFormat(base, projection)
    actual = proposed.materialize(attention)
    patches = tuple(SparseBitPatch.build(a, b) for a, b in zip(reference, actual, strict=True))
    # Verify every correction concerns a protected text slot. A visual change is
    # an implementation defect, not a reason to hide a larger patch/fallback.
    text_mask = torch.tensor(
        [[p in text for p in head] for head in positions], device=device, dtype=torch.bool
    )
    for patch in patches:
        idx = patch.indices.long()
        if not bool(text_mask[idx // (n * d), idx // d % n].all()):
            raise RuntimeError("source overlay changed a nontext cache scalar")
    proposed = replace(proposed, projection=replace(projection, patches=patches))
    for _ in range(3):
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(reference, proposed.materialize(attention), strict=True)
        ):
            raise RuntimeError("text source changed an existing decoded K/V bit")
    use_source = proposed.component_bytes < packet.component_bytes
    return (proposed if use_source else packet), dict(
        original_component_bytes=packet.component_bytes,
        candidate_component_bytes=proposed.component_bytes,
        retained_component_bytes=(proposed if use_source else packet).component_bytes,
        source_used=use_source,
        fallback_reason=None if use_source else "exact_residuals_and_headers_not_smaller",
        changed_selected_positions=0,
        changed_decoded_KV_scalars=0,
        patch_scalars=[p.indices.numel() for p in patches],
        patch_bytes=sum(backing_bytes(p.tensors) for p in patches),
        source_bytes=backing_bytes(source.buffers),
        selected_text_slots=sum(s >= 0 for head in slots for s in head),
        raw_text_source_rows=len(text),
        phase_values_are_original_not_renumbered=True,
        merged_baseline_values_preserved=True,
        no_labels_or_generated_answers_used=True,
        three_independent_materializations_bitwise=True,
        same_option_offered_to_every_baseline=True,
    )
