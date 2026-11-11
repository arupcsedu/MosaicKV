"""Consumed-input, native-wide exact-input replacement diagnostic.

Replace duplicated exact TEXT K/V rows with one RAW original BF16 input row,
keeping the original full-length projection geometry. First preserve every
word of the native-quantized parent. Only its measured net savings can fund
additional original input rows. This is neither the six-mechanism algorithm
nor a claim that a partly quantized prompt reproduces FullKV answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch, decoded, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import (
    native_component,
    owner,
    projection_signature,
)
from mosaickv.evaluation.native_quantized_source_control import (
    _packet_signature,
    _targets,
    _tensor_signature,
    _validate,
    native_quantized_fill,
)
from mosaickv.evaluation.precision_tier import unpack_cpu
from mosaickv.evaluation.shared_source_format import apply_transport_patch

KIND = "native_wide_exact_input_replacement_diagnostic"


class FeasibilityDeclined(RuntimeError):
    """An explicit failed feasibility check, never an implicit raw/full fallback."""

    def __init__(self, reason, **details):
        self.report = dict(feasible=False, reason=reason, **details)
        super().__init__(reason)


def _bitwise(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _decode(packet):
    if packet.source.codes.device.type == "cuda":
        from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction

        return decode_source_for_construction(packet)
    return unpack_cpu(packet.source).squeeze(1)


@dataclass(frozen=True)
class ExactInputReplacementFormat:
    """Only the retained source, reduced exact rows and new raw-input owners.

    No reference to a full original X/K/V, model, or retired parent exact body
    is stored. Original fused-reader patches remain charged but intentionally
    unused, identically to the V20 native-quantized parent control.
    """

    base: object
    input_rows: object
    input_positions: object
    text_positions: tuple
    frozen_projections: tuple
    text_patches: tuple = ()
    kind: str = KIND
    _fingerprint: tuple = field(init=False, repr=False)

    def __post_init__(self):
        _, heads, length, dim = self.base.shape
        positions = self.input_positions
        if (
            self.kind != KIND
            or positions.ndim != 1
            or positions.dtype != torch.int32
            or positions.device != self.base.source.codes.device
            or not positions.is_contiguous()
            or not positions.numel()
            or self.input_rows.shape != (1, positions.numel(), heads * dim)
            or self.input_rows.dtype != torch.bfloat16
            or self.input_rows.device != positions.device
            or not self.input_rows.is_contiguous()
            or not torch.isfinite(self.input_rows).all().item()
            or not self.text_positions
            or tuple(sorted(set(self.text_positions))) != self.text_positions
            or any(type(p) is not int or not 0 <= p < length for p in self.text_positions)
            or (positions < 0).any().item()
            or (positions >= length).any().item()
            or (positions[1:] <= positions[:-1]).any().item()
            or not set(self.text_positions).issubset(positions.cpu().tolist())
            or (self.text_patches and len(self.text_patches) != 2)
        ):
            raise ValueError(
                "Finite private raw BF16 rows and sorted original input positions required"
            )
        for patch in self.text_patches:
            if not isinstance(patch, SparseBitPatch) or patch.indices.dtype != torch.int32:
                raise ValueError("Explicit text-only original-word corrections required")
            if patch.indices.numel():
                rows = (patch.indices.long() // dim % length).cpu().tolist()
                if not set(rows).issubset(self.text_positions):
                    raise ValueError("Numerical corrections may touch only replaced text rows")
        object.__setattr__(self, "_fingerprint", self._signature())

    @property
    def shape(self):
        return self.base.shape

    def private(self):
        return (
            *self.base.private(),
            self.input_rows,
            self.input_positions,
            *(t for patch in self.text_patches for t in patch.tensors),
        )

    def shared(self):
        return self.base.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def _signature(self):
        return (
            _packet_signature(self.base),
            tuple(
                _tensor_signature(t)
                for t in (
                    self.input_rows,
                    self.input_positions,
                    *(t for patch in self.text_patches for t in patch.tensors),
                )
            ),
        )

    @torch.inference_mode()
    def fill(self, attention, targets, *, direct, workspace_observer=None):
        if type(direct) is not bool or torch.is_autocast_enabled(self.input_rows.device.type):
            raise ValueError("Explicit direct policy and disabled autocast required")
        if (
            projection_signature(attention) != self.frozen_projections
            or self._signature() != self._fingerprint
        ):
            raise RuntimeError("Frozen original projections or private source packet changed")
        _targets(self.base, attention, targets)
        private = {owner(t) for t in self.tensors()}
        if any(owner(t) in private for t in targets):
            raise ValueError("Destinations cannot alias replacement storage")
        normalized = _decode(self.base)
        overlay_indices = self.input_positions.long()
        normalized.index_copy_(1, overlay_indices, self.input_rows)
        length = self.shape[2]
        indices = self.base.row_positions.long()
        heads, positions = indices // length, indices % length

        def observe(tensors):
            if workspace_observer is not None:
                workspace_observer(
                    (normalized, overlay_indices, indices, heads, positions, *tensors)
                )

        observe(())
        for component, (target, body) in enumerate(zip(targets, self.base.exact, strict=True)):
            value = native_component(
                normalized, attention, component, self.base.cos, self.base.sin, observer=observe
            )
            target[..., :length, :].copy_(value)
            exact = decoded(body)
            observe((value, exact))
            target[0, heads, positions] = exact[0, 0]
            del value, exact
        for patch, target in zip(
            self.text_patches, targets if self.text_patches else (), strict=True
        ):
            apply_transport_patch(patch, target, prompt_length=length)

    def materialize(self, attention):
        targets = tuple(
            torch.empty(self.shape, dtype=torch.bfloat16, device=self.input_rows.device)
            for _ in range(2)
        )
        self.fill(attention, targets, direct=True)
        return targets


def _raw_rows(normalized, positions):
    ids = torch.tensor(positions, dtype=torch.int32, device=normalized.device)
    rows = normalized.detach().index_select(1, ids.long()).contiguous()
    if owner(rows) == owner(normalized):
        raise RuntimeError("Exact input rows alias the full original source")
    return rows, ids


def _verify_protected(format, attention, packet, original_pair):
    actual = format.materialize(attention)
    length = packet.shape[2]
    ids = packet.row_positions.long()
    for out, original in zip(actual, original_pair, strict=True):
        if not _bitwise(
            out[0, ids // length, ids % length], original[0, ids // length, ids % length]
        ):
            raise FeasibilityDeclined("Old protected original K/V changed")
    return actual


def _choose_rows(ranking, funding, layer_count):
    remaining, chosen, selected = funding, [[] for _ in range(layer_count)], []
    for score, layer, position, cost in sorted(ranking, key=lambda r: (-r[0], r[1], r[2])):
        if cost <= remaining:
            chosen[layer].append(position)
            remaining -= cost
            selected.append(dict(layer=layer, position=position, score=score, raw_row_bytes=cost))
    return chosen, selected, remaining


@torch.inference_mode()
def build_replacement_formats(
    adapter,
    packets,
    normalized_records,
    original_pairs,
    prompt_windows,
    text_positions,
    *,
    cap_bytes,
    prospective_row_scores=None,
    ranking_mode="prompt_mse",
):
    """One fixed, label-blind allocation; its only funding is exact-text savings.

    Prompt demand is ``max_head(mean_query(attention))``. Multiply by decoded
    input-row MSE, sort by decreasing score then layer/original position, and
    spend the actual residual allowance. The optional, explicitly labeled
    ``draft_kl_margin_gradient`` mode accepts only precomputed signed CPU score
    vectors derived from model drafts; it never falls back to prompt MSE. No
    raw logits, draft tokens, labels or benchmark answers enter this constructor.
    """
    packets, pairs = tuple(packets), tuple(original_pairs)
    blocks = tuple(adapter._language_layers())
    text = tuple(text_positions)
    if (
        type(cap_bytes) is not int
        or cap_bytes < 1
        or not packets
        or len(packets) != len(pairs)
        or len(packets) != len(blocks)
        or not text
        or any(type(p) is not int for p in text)
        or tuple(sorted(set(text))) != text
        or len({tuple(packet.shape) for packet in packets}) != 1
    ):
        raise ValueError(
            "Complete common-geometry layers, original text IDs and physical cap required"
        )
    if ranking_mode not in ("prompt_mse", "draft_kl_margin_gradient"):
        raise ValueError("Explicit prompt_mse or draft_kl_margin_gradient ranking required")
    if ranking_mode == "prompt_mse":
        if prospective_row_scores is not None:
            raise ValueError("Prospective score vectors require the explicit draft ranking mode")
    elif (
        type(prospective_row_scores) is not dict
        or any(type(layer) is not int for layer in prospective_row_scores)
        or set(prospective_row_scores) != set(range(len(packets)))
        or any(
            not isinstance(scores, torch.Tensor)
            or scores.device.type != "cpu"
            or not scores.is_floating_point()
            or scores.shape != (packets[layer].shape[2],)
            or not torch.isfinite(scores).all().item()
            for layer, scores in prospective_row_scores.items()
        )
    ):
        raise ValueError("Complete CPU finite floating [N] prospective score vectors required")
    parent_bytes = backing_bytes(tuple(t for packet in packets for t in packet.tensors()))
    if parent_bytes > cap_bytes:
        raise FeasibilityDeclined(
            "Native parent already exceeds hard cap", parent_bytes=parent_bytes
        )
    text_formats, ranking, layer_reports = [], [], []
    for layer, (packet, pair, block) in enumerate(zip(packets, pairs, blocks, strict=True)):
        attention = block.self_attn
        signature = _validate(packet, attention)
        _, heads, length, dim = packet.shape
        original, window = normalized_records[layer], prompt_windows[layer]
        if (
            original.shape != (1, length, heads * dim)
            or original.dtype != torch.bfloat16
            or original.device != packet.source.codes.device
            or not torch.isfinite(original).all().item()
            or any(
                tensor.shape != packet.shape
                or tensor.dtype != torch.bfloat16
                or tensor.device != original.device
                for tensor in pair
            )
            or len(pair) != 2
            or window.ndim != 4
            or window.shape[:2] != (1, heads)
            or not 1 <= window.shape[-2] <= length
            or window.shape[-1] != length
            or window.device != original.device
            or not torch.isfinite(window).all().item()
            or (window < 0).any().item()
            or not 0 <= text[0] <= text[-1] < length
        ):
            raise ValueError(
                "Original BF16 normalized inputs/KV and prompt-only attention required"
            )
        old_rows = tuple(packet.row_positions.cpu().tolist())
        old_row_set = frozenset(old_rows)
        if not {head * length + p for head in range(heads) for p in text}.issubset(old_rows):
            raise ValueError("All text positions must already be exact in every head")
        reference = tuple(
            torch.empty(packet.shape, device=original.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        native_quantized_fill(packet, attention, reference, direct=True)
        protected_ids = packet.row_positions.long()
        for value, expected in zip(reference, pair, strict=True):
            if not _bitwise(
                value[0, protected_ids // length, protected_ids % length],
                expected[0, protected_ids // length, protected_ids % length],
            ):
                raise FeasibilityDeclined("Parent protected rows are not original K/V")
        retained = tuple(p for p in old_rows if p % length not in set(text)) or old_rows[:1]
        ids = torch.tensor(retained, device=original.device, dtype=torch.int32)
        parts = tuple(
            encode(
                t[0, ids.long() // length, ids.long() % length].reshape(1, 1, -1, dim).contiguous()
            )
            for t in pair
        )
        base = replace(
            packet,
            row_positions=ids,
            exact=parts,
            row_positions_cpu=None,
        )
        rows, positions = _raw_rows(original, text)
        candidate = ExactInputReplacementFormat(base, rows, positions, text, signature)
        actual = candidate.materialize(attention)
        patches = tuple(SparseBitPatch.build(a, b) for a, b in zip(reference, actual, strict=True))
        candidate = replace(candidate, text_patches=patches)
        verified = candidate.materialize(attention)
        if not all(_bitwise(a, b) for a, b in zip(reference, verified, strict=True)):
            raise FeasibilityDeclined("Text replacement does not preserve native parent K/V")
        _verify_protected(candidate, attention, packet, pair)
        text_formats.append(candidate)
        if ranking_mode == "prompt_mse":
            decoded_source = _decode(packet)
            error = (original.double() - decoded_source.double()).square().mean(-1)[0]
            demand = window.double().mean(-2)[0].amax(0)
            scores = (error * demand).cpu().tolist()
            del decoded_source, error, demand
        else:
            scores = prospective_row_scores[layer].detach().tolist()
        row_cost = heads * dim * 2 + 4
        ranking.extend(
            (float(score), layer, position, row_cost)
            for position, score in enumerate(scores)
            if position not in set(text)
            and score > 0
            and not all(head * length + position in old_row_set for head in range(heads))
        )
        layer_reports.append(
            dict(
                layer=layer,
                parent_component_bytes=packet.component_bytes,
                text_component_bytes=candidate.component_bytes,
                text_input_rows=len(text),
                remaining_exact_head_rows=len(retained),
                old_protected_head_rows=len(old_rows),
                text_numeric_patch_scalars=[p.indices.numel() for p in patches],
                native_parent_all_KV_bitwise_after_text_patches_before_optional_rows=True,
                old_protected_KV_bitwise=True,
            )
        )
        del reference, actual, verified, scores
    text_formats = tuple(text_formats)
    text_bytes = backing_bytes(tuple(t for f in text_formats for t in f.tensors()))
    if text_bytes > cap_bytes or text_bytes >= parent_bytes:
        raise FeasibilityDeclined(
            "Exact-text replacement has no usable net physical savings",
            parent_bytes=parent_bytes,
            text_bytes=text_bytes,
            cap_bytes=cap_bytes,
            layers=layer_reports,
        )
    funding = min(parent_bytes - text_bytes, cap_bytes - text_bytes)
    chosen, selected, remaining = _choose_rows(ranking, funding, len(packets))
    corrected = []
    for layer, (candidate, block, packet, pair) in enumerate(
        zip(text_formats, blocks, packets, pairs, strict=True)
    ):
        if chosen[layer]:
            positions = tuple(sorted((*text, *chosen[layer])))
            rows, indices = _raw_rows(normalized_records[layer], positions)
            candidate = replace(candidate, input_rows=rows, input_positions=indices)
        _verify_protected(candidate, block.self_attn, packet, pair)
        forbidden = {owner(normalized_records[layer]), *(owner(t) for t in pair)}
        if any(owner(t) in forbidden for t in candidate.private() if t.numel()):
            raise RuntimeError("Replacement aliases original source or cache values")
        corrected.append(candidate)
    corrected = tuple(corrected)
    actual_bytes = backing_bytes(tuple(t for f in corrected for t in f.tensors()))
    if actual_bytes > cap_bytes or actual_bytes - text_bytes > funding:
        raise RuntimeError(
            "Constructed exact-input rows exceeded measured source-retirement funding"
        )
    if actual_bytes - text_bytes != funding - remaining:
        raise RuntimeError("Actual unique-owner row growth differs from charged raw-row costs")
    report = dict(
        kind=KIND,
        feasible=True,
        reference="native_wide_same_quantized_source_parent_NOT_fused_reader_or_FullKV",
        parent_initial_bytes=parent_bytes,
        text_initial_bytes=text_bytes,
        corrected_initial_bytes=actual_bytes,
        cap_bytes=cap_bytes,
        exact_text_retirement_funding_bytes=funding,
        restored_input_row_bytes=actual_bytes - text_bytes,
        cap_shortfall_bytes=cap_bytes - actual_bytes,
        unused_retirement_funding_bytes=remaining,
        selected_original_nontext_rows=selected,
        selected_original_nontext_row_count=len(selected),
        additional_input_rows=len(selected),
        fully_protected_all_head_rows_excluded=True,
        layers=layer_reports,
        raw_original_BF16_input_rows=True,
        original_full_prompt_projection_geometry=True,
        original_three_bit_code_and_header_owners_retained=True,
        original_fused_transport_patch_owners_retained_but_unused=True,
        native_parent_KV_bitwise_preserved_before_optional_rows=True,
        all_old_protected_original_KV_bitwise=True,
        CPU_tensor_metadata_bytes=0,
        Python_identity_metadata_not_counted_as_GPU_storage=True,
        original_source_alias=False,
        old_text_owner_retirement_certified=False,
        model_weights_unchanged=True,
        ranking_mode=ranking_mode,
        model_draft_derived_scoring=ranking_mode == "draft_kl_margin_gradient",
        constructor_raw_logits_or_labels_accessed=False,
        benchmark_answers_accessed=False,
        quality_or_latency_improvement_guaranteed=False,
        whole_request_KV_peak_certified=False,
    )
    if ranking_mode == "prompt_mse":
        report["benchmark_answers_or_logits_accessed"] = False
    return text_formats, corrected, report
