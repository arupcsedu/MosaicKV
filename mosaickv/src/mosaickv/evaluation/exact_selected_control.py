"""Planner-independent, original-BF16 exact-selection transport controls.

These controls compare two transports of the SAME selected original K/V. They
do not turn a failed FullKV quality guard into a pass, implement the six-stage
research method, or certify whole-request peak memory. Raw append-only history
has value-independent storage geometry; the prompt cap is initial-only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from mosaickv.adapters.huggingface.types import DecodeState
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport, backing_bytes
from mosaickv.evaluation.request_precision import ranked_positions


@dataclass(frozen=True)
class RawSelectedFormat(ExactFormat):
    """Raw selected payload plus explicitly CPU-owned logical-position metadata."""

    original_positions: Any = None
    original_length: int = 0
    original_KV_bitwise_verified: bool = False


def mandatory_positions(prepared: Any, n: int) -> tuple[int, ...]:
    """Pinned old control policy: all text, first four, and last eight tokens."""
    if type(n) is not int or n < 1:
        raise ValueError("positive original cache length required")
    selected = set(range(min(4, n))) | set(range(max(0, n - 8), n))
    for span in prepared.modality_map:
        modality = getattr(span.modality, "value", span.modality)
        if not 0 <= span.start <= span.end <= n:
            raise ValueError("modality spans must lie within the original cache")
        if modality == "text":
            selected.update(range(span.start, span.end))
    return tuple(sorted(selected))


def select_uniform_positions(
    scores: Any, count: int, mandatory: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Stable source-index tie breaks; sorted original positions in each head."""
    if (
        type(count) is not int
        or not isinstance(mandatory, tuple)
        or any(type(p) is not int for p in mandatory)
        or tuple(sorted(set(mandatory))) != mandatory
    ):
        raise ValueError("integer capacity and sorted unique mandatory positions required")
    return ranked_positions(scores, count, mandatory)


@torch.inference_mode()
def build_raw_formats(
    adapter: Any,
    originalpairs: Any,
    positions: Any,
    *,
    mandatory: tuple[int, ...] = (),
) -> tuple[RawSelectedFormat, ...]:
    """Gather ORIGINAL BF16 bits, never reconstructed/quantized source values.

    Independent per-head checks bound the temporary verification reference to
    one selected head. No raw source or verification K/V is kept on CPU or in
    the returned representation. Only the position map is CPU-owned metadata.
    """
    pairs, memberships = tuple(originalpairs), tuple(positions)
    if not pairs or len(pairs) != len(adapter._language_layers()) or len(memberships) != len(pairs):
        raise ValueError("one original K/V pair and membership map per model layer required")
    formats: list[RawSelectedFormat] = []
    capacity = None
    for layer, (pair, rows) in enumerate(zip(pairs, memberships, strict=True)):
        if len(pair) != 2:
            raise ValueError("each original layer must contain exactly K and V")
        key, value = pair
        if (
            key.ndim != 4
            or key.shape[0] != 1
            or key.shape != value.shape
            or key.dtype != torch.bfloat16
            or value.dtype != torch.bfloat16
            or key.device != value.device
            or min(key.shape) < 1
        ):
            raise ValueError("matching nonempty batch-one original BF16 BHND K/V required")
        rows = tuple(tuple(row) for row in rows)
        heads, length, width = key.shape[1:]
        count = len(rows[0]) if rows else 0
        if (
            len(rows) != heads
            or not 1 <= count <= length
            or any(
                len(row) != count
                or any(type(p) is not int or not 0 <= p < length for p in row)
                or tuple(sorted(set(row))) != row
                or not set(mandatory).issubset(row)
                for row in rows
            )
        ):
            raise ValueError(
                f"layer {layer}: uniform sorted unique bounded mandatory memberships required"
            )
        if capacity is not None and capacity != count:
            raise ValueError("common uniform physical capacity is required across layers")
        capacity = count
        cpu_positions = torch.tensor(rows, dtype=torch.int64, device="cpu")
        index = cpu_positions.to(key.device)[None, :, :, None].expand(1, heads, count, width)
        selected = tuple(t.detach().gather(2, index).contiguous() for t in pair)
        for source, retained in zip(pair, selected, strict=True):
            for head in range(heads):
                original = source[:, head].index_select(1, index[0, head, :, 0])
                if not torch.equal(original.view(torch.int16), retained[:, head].view(torch.int16)):
                    raise RuntimeError("retained K/V differs from original BF16 source bits")
                del original
        formats.append(
            RawSelectedFormat(
                "kv",
                tuple(selected[0].shape),
                pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in selected),
                original_positions=cpu_positions,
                original_length=length,
                original_KV_bitwise_verified=True,
            )
        )
    return tuple(formats)


def raw_format_report(formats: Any) -> dict[str, Any]:
    """Full backing sizes (including views), with CPU metadata separate."""
    formats = tuple(formats)
    if not formats or any(
        not isinstance(f, RawSelectedFormat)
        or f.kind != "kv"
        or not f.original_KV_bitwise_verified
        or len(f.pair) != 2
        or any(not b.raw or len(b.buffers) != 1 for b in f.pair)
        or f.original_positions.device.type != "cpu"
        for f in formats
    ):
        raise ValueError("verified original-BF16 raw exact formats required")
    tensors = tuple(t for f in formats for t in f.tensors())
    maps = tuple(f.original_positions for f in formats)
    return dict(
        representation="original_BF16_exact_selection_transport_control",
        original_KV_bitwise_verified=True,
        raw_KV_backing_bytes=backing_bytes(tensors),
        raw_KV_view_payload_bytes=sum(t.numel() * t.element_size() for t in tensors),
        CPU_position_map_backing_bytes=backing_bytes(maps),
        original_position_maps=[p.tolist() for p in maps],
        position_map_sha256=hashlib.sha256(b"".join(p.numpy().tobytes() for p in maps)).hexdigest(),
        original_lengths=[f.original_length for f in formats],
        physical_lengths=[f.shape[-2] for f in formats],
        prompt_cap_boundary="initial_only_not_lifetime",
        generated_history="raw_BF16_append_only_common_16_position_growth",
        history_quantized=False,
        source_encoding_or_norm_quantization=False,
        benchmark_labels_or_reference_answers_used=False,
        same_selected_original_KV_transport_only=True,
        FullKV_quality_guard_evaluated=False,
        whole_request_KV_peak_certified=False,
    )


def _state(prefill: Any, cache: Any, formats: Any) -> DecodeState:
    count = int(formats[0].shape[-2])
    if len({f.shape[-2] for f in formats}) != 1:
        raise ValueError("common physical capacity required")
    if prefill.state.logical_sequence_length < max(f.original_length for f in formats):
        raise ValueError("logical length cannot precede original cache positions")
    return DecodeState(
        cache,
        torch.ones(1, count, dtype=torch.long, device=formats[0].pair[0].buffers[0].device),
        count,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
        prefill.state.modality_map,
        dict(
            prefill.state.model_state, mosaickv_packed_all_valid=True, mosaickv_cache_all_valid=True
        ),
    )


@torch.inference_mode()
def native_same_selected_state(adapter: Any, prefill: Any, formats: Any) -> DecodeState:
    """Independent native HF cache of the exact, already-verified original K/V."""
    from transformers import DynamicCache

    formats = tuple(formats)
    raw_format_report(formats)
    if len(formats) != len(adapter._language_layers()):
        raise ValueError("native reference must cover every model layer")
    cache = DynamicCache()
    for index, f in enumerate(formats):
        key, value = (b.buffers[0].clone() for b in f.pair)
        cache.update(key, value, index)
    return _state(prefill, cache, formats)


def _decode(adapter: Any, prefill: Any, state: DecodeState, tokens: int, forced: Any) -> Any:
    if type(tokens) is not int or not 2 <= tokens <= 128:
        raise ValueError("two through 128 prediction diagnostic required")
    if forced is not None and (
        len(forced) != tokens or any(type(token) is not int or token < 0 for token in forced)
    ):
        raise ValueError("complete nonnegative integer diagnostic prefix required")
    values = [prefill.logits.detach().reshape(-1).float().cpu()]
    token, initial_next = prefill.next_token_id, state.next_decode_position
    for step in range(tokens - 1):
        if forced is not None:
            token = torch.tensor([[forced[step]]], device=state.attention_mask.device)
        output = adapter.decode_one_token(token, state, capture_queries=False)
        if output.state is not state:
            raise RuntimeError("adapter must preserve the isolated mutable decode-state handle")
        values.append(output.logits.detach().reshape(-1).float().cpu())
        token = output.next_token_id
    logits = torch.stack(values)
    if (
        not bool(torch.isfinite(logits).all())
        or state.next_decode_position != initial_next + tokens - 1
    ):
        raise RuntimeError("nonfinite logits or incorrect original logical decode position")
    return logits


@torch.inference_mode()
def run_native_same_selected(
    adapter: Any, prefill: Any, formats: Any, *, tokens: int = 128, forced: Any = None
) -> tuple[Any, dict[str, Any]]:
    """CPU/GPU native transport reference, not a performance measurement."""
    formats = tuple(formats)
    state = native_same_selected_state(adapter, prefill, formats)
    logits = _decode(adapter, prefill, state, tokens, forced)
    pairs, _ = adapter._legacy_layers(state.past_key_values)
    return logits, dict(
        **raw_format_report(formats),
        reference="native_DynamicCache_of_SAME_selected_original_KV_not_FullKV",
        final_native_KV_backing_bytes=backing_bytes(t for pair in pairs for t in pair),
        original_next_decode_position=state.next_decode_position,
        output_predictions=tokens,
        generated_KV_updates=tokens - 1,
        timing_measured=False,
        no_latency_or_benchmark_quality_claim=True,
    )


@torch.inference_mode()
def run_exact_selected_cpu(
    adapter: Any, prefill: Any, formats: Any, *, tokens: int = 128, forced: Any = None
) -> tuple[Any, dict[str, Any]]:
    """CPU regression path using the identical raw append/common-lease classes."""
    from mosaickv.evaluation.storage_runs import cache_from_formats

    formats = tuple(formats)
    report = raw_format_report(formats)
    if any(t.device.type != "cpu" for f in formats for t in f.tensors()):
        raise ValueError("CPU-only regression path; use storage_runs.run for CUDA timing")
    cache = cache_from_formats(
        adapter,
        formats,
        updates=tokens,
        audit=False,
        sealed=False,
        source_history=False,
        shared_workspace=False,
        sequential_lease=False,
    )
    state = _state(prefill, cache, formats)
    with IsolatedStorageTransport(adapter, cache):
        logits = _decode(adapter, prefill, state, tokens, forced)
    storage = cache.storage_record()
    if any(layer.leased for layer in cache.layers):
        raise RuntimeError("raw exact cache lease leaked after decode")
    return logits, dict(
        **report,
        storage=storage,
        output_predictions=tokens,
        generated_KV_updates=tokens - 1,
        timing_measured=False,
        no_latency_or_benchmark_quality_claim=True,
    )
