"""Lossless fusion of an existing source+text packet, not a quality intervention."""

from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch, decoded
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@dataclass(frozen=True)
class ExactInput:
    source: object
    positions: object

    def tensors(self):
        return (*self.source.buffers, self.positions)


def overlay_input(overlay, target, first):
    if target.device.type == "cuda":
        from mosaickv.evaluation.integrated_source_input_kernel import scatter

        scatter(overlay.source, overlay.positions, target, first)
    else:
        positions = overlay.positions.long()
        valid = (positions >= first) & (positions < first + target.shape[1])
        target[0, positions[valid] - first] = decoded(overlay.source)[0, 0, valid]


@torch.inference_mode()
def fuse(packet, attention):
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    if not isinstance(packet, TextSourceFormat) or not isinstance(packet.base, SharedSourceFormat):
        return packet, dict(used=False, reason="no_shared_source_text_packet")
    base = packet.base
    if base.projection_backend != "native_bmm_256" or base.input_overlay is not None:
        raise ValueError("one native source/text fusion required")
    overlay = ExactInput(packet.projection.source, packet.projection.original_text)
    reference = packet.materialize(attention)
    candidate = replace(
        base,
        input_overlay=overlay,
        transport_patches=(),
        kind="common_bit_exact_integrated_source_inputs_ablation",
    )
    actual = candidate.materialize(attention)
    corrections = tuple(SparseBitPatch.build(a, b) for a, b in zip(reference, actual, strict=True))
    n, d = base.shape[-2:]
    for component, patch in enumerate(corrections):
        ids = patch.indices.long()
        allowed = torch.isin(ids // d % n, overlay.positions.long())
        if base.transport_patches:
            allowed |= torch.isin(ids, base.transport_patches[component].indices.long())
        if not allowed.all():
            raise RuntimeError("exact source-input fusion changed a non-overlay scalar")
    candidate = replace(candidate, transport_patches=corrections)
    for _ in range(3):
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(reference, candidate.materialize(attention), strict=True)
        ):
            raise RuntimeError("integrated source input did not preserve K/V bits")
    use = candidate.component_bytes <= packet.component_bytes
    return (candidate if use else packet), dict(
        used=use,
        reason=None if use else "exact_corrections_not_smaller",
        old_bytes=packet.component_bytes,
        candidate_bytes=candidate.component_bytes,
        three_materializations_bitwise=True,
        exact_input_bytes=backing_bytes(overlay.tensors()),
        correction_scalars=[p.indices.numel() for p in corrections],
        no_task_logits_or_answers_used=True,
    )
