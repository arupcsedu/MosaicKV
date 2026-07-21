"""Paper-faithful ICLR VL-Cache reimplementation.

This is the sparsity/modality-aware method from Tu et al. (ICLR 2025),
arXiv:2410.23317.  It is unrelated to the later recurring-image cache-reuse
system with a similar name and is never presented as official author code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

import numpy as np

from mosaickv.cache_state import FullKVState, Modality
from mosaickv.config import CacheConfig, VLCacheConfig
from mosaickv.types import BudgetUnit, JsonObject


class VLCacheReimplementationError(RuntimeError):
    """Raised when the paper method cannot be represented without guessing."""


def assert_vl_cache_calibration_disjoint(
    calibration_sample_ids: tuple[str, ...],
    evaluation_sample_ids: tuple[str, ...],
) -> None:
    """Reject duplicate identities and calibration/evaluation leakage."""

    for label, values in (
        ("calibration", calibration_sample_ids),
        ("evaluation", evaluation_sample_ids),
    ):
        if any(not value.strip() for value in values):
            raise VLCacheReimplementationError(f"VL-Cache {label} sample IDs cannot be empty")
        if len(values) != len(set(values)):
            raise VLCacheReimplementationError(f"VL-Cache {label} sample IDs must be unique")
    overlap = sorted(set(calibration_sample_ids).intersection(evaluation_sample_ids))
    if overlap:
        raise VLCacheReimplementationError(
            "VL-Cache calibration and evaluation samples overlap: " + ", ".join(overlap[:5])
        )


def infer_post_vision_start(full_state: FullKVState) -> int:
    """Return the first prompt position after the final image/video span."""

    media = tuple(
        span
        for span in full_state.modality_spans
        if span.modality in {Modality.IMAGE, Modality.VIDEO}
    )
    if not media:
        raise VLCacheReimplementationError(
            "vl_cache_reimpl requires an image/video span to define post-vision attention"
        )
    start = max(span.end for span in media)
    if start >= full_state.active_sequence_length:
        raise VLCacheReimplementationError(
            "vl_cache_reimpl requires at least one language token after the final visual token"
        )
    return start


def _numpy_attention(value: Any, layer: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    if hasattr(value, "detach"):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] != 1:
        raise VLCacheReimplementationError(
            f"VL-Cache attention layer {layer} must have shape [1,H,Q,K]"
        )
    if not bool(np.all(np.isfinite(array))) or bool(np.any(array < 0)):
        raise VLCacheReimplementationError(
            f"VL-Cache attention layer {layer} is non-finite or negative"
        )
    return array


def threshold_filter(
    attention: np.ndarray[Any, np.dtype[np.float32]], threshold: float
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply paper Equation (1), thresholding relative to each row maximum."""

    values = np.asarray(attention, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] < 1:
        raise VLCacheReimplementationError("ThresholdFilter requires non-empty attention rows")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise VLCacheReimplementationError("ThresholdFilter p must be in (0, 1)")
    row_maximum = np.max(values, axis=-1, keepdims=True)
    return np.where(values >= threshold * row_maximum, values, 0).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class VLCacheAttentionStatistics:
    """Post-vision sparsity and accumulated token scores from one prefill."""

    post_vision_start: int
    query_start: int
    query_end: int
    layer_query_head_sparsity: tuple[tuple[float, ...], ...]
    layer_sparsity: tuple[float, ...]
    layer_density: tuple[float, ...]
    kv_head_token_scores: tuple[tuple[tuple[float, ...], ...], ...]

    def __post_init__(self) -> None:
        if not 0 <= self.post_vision_start <= self.query_start < self.query_end:
            raise VLCacheReimplementationError("invalid VL-Cache post-vision query range")
        layer_count = len(self.layer_sparsity)
        if layer_count < 1 or not (
            len(self.layer_query_head_sparsity)
            == len(self.layer_density)
            == len(self.kv_head_token_scores)
            == layer_count
        ):
            raise VLCacheReimplementationError("VL-Cache statistics do not cover every layer")
        for sparse, density in zip(self.layer_sparsity, self.layer_density, strict=True):
            if not (0 <= sparse <= 1 and 0 <= density <= 1):
                raise VLCacheReimplementationError("VL-Cache sparsity lies outside [0, 1]")
            if not math.isclose(sparse + density, 1.0, rel_tol=0, abs_tol=1e-7):
                raise VLCacheReimplementationError("VL-Cache density must equal 1 - sparsity")


def post_vision_attention_statistics(
    attention_weights: tuple[Any, ...],
    *,
    kv_heads_by_layer: tuple[int, ...],
    post_vision_start: int,
    threshold: float = 0.01,
    max_post_vision_queries: int | None = None,
) -> VLCacheAttentionStatistics:
    """Compute Algorithm 1 sparsity and Section 4.2 token scores.

    Query heads are averaged in contiguous GQA groups only for token scoring;
    Algorithm 1 sparsity is measured per query head and then averaged by layer.
    """

    if not attention_weights:
        raise VLCacheReimplementationError("vl_cache_reimpl requires eager prefill attentions")
    if len(attention_weights) != len(kv_heads_by_layer):
        raise VLCacheReimplementationError("attention/cache layer counts differ")
    if max_post_vision_queries is not None and max_post_vision_queries < 1:
        raise VLCacheReimplementationError("max_post_vision_queries must be positive or null")

    query_head_sparsity: list[tuple[float, ...]] = []
    layer_sparsity: list[float] = []
    scores_by_layer: list[tuple[tuple[float, ...], ...]] = []
    common_query_start: int | None = None
    common_query_end: int | None = None
    for layer_index, (raw, kv_heads) in enumerate(
        zip(attention_weights, kv_heads_by_layer, strict=True)
    ):
        attention = _numpy_attention(raw, layer_index)[0]
        query_heads, query_length, key_length = attention.shape
        query_offset = key_length - query_length
        if query_offset < 0 or not 0 <= post_vision_start < key_length:
            raise VLCacheReimplementationError(
                f"VL-Cache layer {layer_index} has incompatible Q/K lengths"
            )
        query_start = max(post_vision_start, query_offset)
        if max_post_vision_queries is not None:
            query_start = max(query_start, key_length - max_post_vision_queries)
        query_end = key_length
        local_start = query_start - query_offset
        if local_start >= query_length:
            raise VLCacheReimplementationError(
                f"VL-Cache layer {layer_index} exposes no post-vision query rows"
            )
        if common_query_start is None:
            common_query_start, common_query_end = query_start, query_end
        elif (query_start, query_end) != (common_query_start, common_query_end):
            raise VLCacheReimplementationError("VL-Cache layers use different query ranges")
        sliced = attention[:, local_start:, :]
        global_queries = np.arange(query_start, query_end, dtype=np.int64)
        keys = np.arange(key_length, dtype=np.int64)
        causal = keys[None, :] <= global_queries[:, None]
        filtered = threshold_filter(sliced, threshold)
        nonzero = (filtered > 0) & causal[None, :, :]
        valid_count = int(causal.sum())
        if valid_count < 1:
            raise VLCacheReimplementationError("VL-Cache causal post-vision region is empty")
        head_sparsity = tuple(
            float(1.0 - nonzero[head].sum(dtype=np.int64) / valid_count)
            for head in range(query_heads)
        )
        query_head_sparsity.append(head_sparsity)
        layer_sparsity.append(float(sum(head_sparsity) / len(head_sparsity)))

        if kv_heads < 1 or query_heads % kv_heads:
            raise VLCacheReimplementationError(
                f"VL-Cache layer {layer_index} cannot map {query_heads} query heads "
                f"to {kv_heads} KV heads"
            )
        group_size = query_heads // kv_heads
        grouped = sliced.reshape(kv_heads, group_size, sliced.shape[-2], key_length)
        # Paper Section 3.2 defines psi as a sum over post-vision queries.
        # Averaging GQA query heads is an explicitly documented implementation choice.
        accumulated = grouped.mean(axis=1, dtype=np.float32).sum(axis=1, dtype=np.float32)
        scores_by_layer.append(
            tuple(tuple(float(value) for value in row.tolist()) for row in accumulated)
        )

    assert common_query_start is not None and common_query_end is not None
    sparsity_tuple = tuple(layer_sparsity)
    return VLCacheAttentionStatistics(
        post_vision_start,
        common_query_start,
        common_query_end,
        tuple(query_head_sparsity),
        sparsity_tuple,
        tuple(1.0 - value for value in sparsity_tuple),
        tuple(scores_by_layer),
    )


def paper_layer_retention_ratios(
    layer_sparsity: tuple[float, ...],
    *,
    retention_ratio: float,
    minimum: float = 0.01,
    maximum: float = 1.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return Algorithm 1 raw and clipped layer budgets beta."""

    if not layer_sparsity:
        raise VLCacheReimplementationError("VL-Cache allocation requires at least one layer")
    if not 0 < retention_ratio <= 1:
        raise VLCacheReimplementationError("VL-Cache alpha must be in (0, 1]")
    if not 0 < minimum <= maximum <= 1:
        raise VLCacheReimplementationError("VL-Cache clipping bounds are invalid")
    density = tuple(1.0 - value for value in layer_sparsity)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in density):
        raise VLCacheReimplementationError("VL-Cache layer sparsity must be in [0, 1]")
    normalizer = sum(density)
    if normalizer <= 0:
        raise VLCacheReimplementationError("VL-Cache cannot allocate an all-sparse prompt")
    layer_count = len(density)
    raw = tuple(value / normalizer * retention_ratio * layer_count for value in density)
    clipped = tuple(min(maximum, max(minimum, value)) for value in raw)
    return raw, clipped


def _bounded_apportion(
    desired: tuple[float, ...],
    lower: tuple[int, ...],
    upper: tuple[int, ...],
    target: int,
) -> tuple[int, ...]:
    """Deterministically realize fractional paper budgets under one hard total."""

    if not (len(desired) == len(lower) == len(upper)):
        raise VLCacheReimplementationError("VL-Cache allocation vectors differ in length")
    if not sum(lower) <= target <= sum(upper):
        raise VLCacheReimplementationError(
            f"VL-Cache target {target} is outside clipped/mandatory bounds "
            f"[{sum(lower)}, {sum(upper)}]"
        )
    clipped = tuple(
        min(float(high), max(float(low), value))
        for value, low, high in zip(desired, lower, upper, strict=True)
    )
    base = [math.floor(value + 1e-12) for value in clipped]
    if sum(base) > target:
        order = sorted(
            range(len(base)),
            key=lambda index: (clipped[index] - base[index], index),
        )
        remaining = sum(base) - target
        for index in order:
            removable = min(remaining, base[index] - lower[index])
            base[index] -= removable
            remaining -= removable
            if remaining == 0:
                break
    else:
        remaining = target - sum(base)
        order = sorted(
            range(len(base)),
            key=lambda index: (-(clipped[index] - base[index]), index),
        )
        for index in order:
            if remaining == 0:
                break
            if base[index] < upper[index]:
                base[index] += 1
                remaining -= 1
        if remaining:
            for index in range(len(base)):
                addition = min(remaining, upper[index] - base[index])
                base[index] += addition
                remaining -= addition
                if remaining == 0:
                    break
    if remaining or sum(base) != target:
        raise VLCacheReimplementationError("VL-Cache could not realize the hard global budget")
    return tuple(base)


def _sha_floats(values: tuple[float, ...]) -> str:
    array = np.asarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class VLCacheHeadState:
    """Per-KV-head Top-K result using accumulated post-vision attention."""

    layer: int
    kv_head: int
    source_positions: int
    selected_physical_positions: tuple[int, ...]
    selected_logical_positions: tuple[int, ...]
    mandatory_positions: tuple[int, ...]
    recent_positions: tuple[int, ...]
    token_score_sha256: str
    token_score_sum: float
    retained_bytes: int

    @property
    def retained_positions(self) -> int:
        return len(self.selected_physical_positions)

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kv_head < 0 or self.source_positions < 1:
            raise VLCacheReimplementationError("VL-Cache head identity/length is invalid")
        if self.selected_physical_positions != tuple(sorted(set(self.selected_physical_positions))):
            raise VLCacheReimplementationError("VL-Cache selected positions must be sorted/unique")
        if not self.selected_physical_positions or self.selected_physical_positions[-1] >= (
            self.source_positions
        ):
            raise VLCacheReimplementationError("VL-Cache selected position is outside the source")
        selected = set(self.selected_physical_positions)
        if (
            not set(self.mandatory_positions) <= selected
            or not set(self.recent_positions) <= selected
        ):
            raise VLCacheReimplementationError("VL-Cache removed a protected token")
        if len(self.selected_logical_positions) != self.retained_positions:
            raise VLCacheReimplementationError("VL-Cache physical/logical selections differ")
        if (
            len(self.token_score_sha256) != 64
            or not math.isfinite(self.token_score_sum)
            or self.token_score_sum < 0
            or self.retained_bytes < 1
        ):
            raise VLCacheReimplementationError("VL-Cache head provenance/accounting is invalid")


@dataclass(frozen=True, slots=True)
class VLCacheLayerState:
    """Paper fractional allocation and realized integer budget for one layer."""

    layer: int
    query_head_sparsity: tuple[float, ...]
    sparsity: float
    density: float
    raw_retention_ratio: float
    clipped_retention_ratio: float
    retained_positions_per_head: int
    heads: tuple[VLCacheHeadState, ...]

    def __post_init__(self) -> None:
        if self.layer < 0 or not self.heads or self.retained_positions_per_head < 1:
            raise VLCacheReimplementationError("VL-Cache layer identity/budget is invalid")
        if any(head.layer != self.layer for head in self.heads):
            raise VLCacheReimplementationError("VL-Cache layer contains a mismatched head")
        if tuple(head.kv_head for head in self.heads) != tuple(range(len(self.heads))):
            raise VLCacheReimplementationError("VL-Cache KV heads are not in canonical order")
        if any(head.retained_positions != self.retained_positions_per_head for head in self.heads):
            raise VLCacheReimplementationError("VL-Cache layer heads have different budgets")
        if not math.isclose(self.density, 1.0 - self.sparsity, rel_tol=0, abs_tol=1e-7):
            raise VLCacheReimplementationError("VL-Cache layer density is inconsistent")
        if any(
            not math.isfinite(value)
            for value in (
                self.sparsity,
                self.density,
                self.raw_retention_ratio,
                self.clipped_retention_ratio,
            )
        ):
            raise VLCacheReimplementationError("VL-Cache layer statistics must be finite")


@dataclass(frozen=True, slots=True)
class VLCacheCompressionPlan:
    """Exact-only paper selection plan consumed by the common cache packer."""

    full_state: FullKVState
    config: VLCacheConfig
    statistics: VLCacheAttentionStatistics
    layers: tuple[VLCacheLayerState, ...]
    source_slots: int
    active_slots: int
    source_bytes: int
    retained_bytes: int
    target_slots: int

    def __post_init__(self) -> None:
        if len(self.layers) != len(self.full_state.layers):
            raise VLCacheReimplementationError("VL-Cache plan does not cover every layer")
        heads = tuple(head for layer in self.layers for head in layer.heads)
        if self.active_slots != sum(head.retained_positions for head in heads):
            raise VLCacheReimplementationError("VL-Cache active slot accounting is inconsistent")
        if self.retained_bytes != sum(head.retained_bytes for head in heads):
            raise VLCacheReimplementationError("VL-Cache byte accounting is inconsistent")
        if self.source_slots != len(self.full_state.blocks):
            raise VLCacheReimplementationError("VL-Cache source slot accounting is inconsistent")
        if self.active_slots != self.target_slots or self.retained_bytes > self.source_bytes:
            raise VLCacheReimplementationError("VL-Cache exceeds its hard budget/source bytes")
        if self.full_state.block_size != 1:
            raise VLCacheReimplementationError("VL-Cache requires token-sized source blocks")

    @property
    def is_retention_one(self) -> bool:
        return self.active_slots == self.source_slots

    def trace(self) -> JsonObject:
        calibration_ids = tuple(sorted(self.config.calibration_sample_ids))
        calibration_sha = (
            hashlib.sha256("\0".join(calibration_ids).encode()).hexdigest()
            if calibration_ids
            else None
        )
        return {
            "implementation": "vl_cache_reimpl",
            "official_code": False,
            "paper": "Tu et al., ICLR 2025, arXiv:2410.23317v1",
            "distinct_from_recurring_image_vlcache": True,
            "paper_equations": {
                "threshold_filter": "Equation 1",
                "attention_sparsity": "Equation 2 and Algorithm 1 lines 3-7",
                "layer_budget": "Algorithm 1 lines 10-21",
                "token_score": "Section 3.2 accumulated post-vision attention",
            },
            "implementation_decisions": {
                "integer_budget": "bounded_largest_remainder_with_exact_hard_total",
                "gqa": "contiguous_query_head_groups_mean_to_each_kv_head",
                "topk_ties": "lower_physical_position_first",
                "recent_window_rounding": "floor_fraction_of_realized_layer_budget",
                "post_vision_keys": "all_causally_visible_prompt_keys",
            },
            "parameters": {
                "sparsity_threshold": self.config.sparsity_threshold,
                "min_layer_retention": self.config.min_layer_retention,
                "max_layer_retention": self.config.max_layer_retention,
                "recent_window_fraction": self.config.recent_window_fraction,
                "max_post_vision_queries": self.config.max_post_vision_queries,
            },
            "post_vision_start": self.statistics.post_vision_start,
            "query_start": self.statistics.query_start,
            "query_end": self.statistics.query_end,
            "source_slots": self.source_slots,
            "target_slots": self.target_slots,
            "active_slots": self.active_slots,
            "source_bytes": self.source_bytes,
            "retained_bytes": self.retained_bytes,
            "realized_slot_retention_ratio": self.active_slots / self.source_slots,
            "calibration": {
                "required_by_paper_method": False,
                "dataset_id": self.config.calibration_dataset_id,
                "dataset_revision": self.config.calibration_dataset_revision,
                "split": self.config.calibration_split,
                "sample_count": len(calibration_ids),
                "sample_ids_sha256": calibration_sha,
            },
            "layers": [
                {
                    "layer": layer.layer,
                    "query_head_sparsity": list(layer.query_head_sparsity),
                    "sparsity": layer.sparsity,
                    "density": layer.density,
                    "raw_retention_ratio": layer.raw_retention_ratio,
                    "clipped_retention_ratio": layer.clipped_retention_ratio,
                    "retained_positions_per_head": layer.retained_positions_per_head,
                    "heads": [
                        {
                            "kv_head": head.kv_head,
                            "selected_physical_positions": list(head.selected_physical_positions),
                            "selected_logical_positions": list(head.selected_logical_positions),
                            "mandatory_positions": list(head.mandatory_positions),
                            "recent_positions": list(head.recent_positions),
                            "token_score_sha256": head.token_score_sha256,
                            "token_score_sum": head.token_score_sum,
                            "retained_bytes": head.retained_bytes,
                        }
                        for head in layer.heads
                    ],
                }
                for layer in self.layers
            ],
        }


def _select_positions(
    scores: tuple[float, ...],
    retained: int,
    mandatory: tuple[int, ...],
    recent_fraction: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    length = len(scores)
    recent_count = math.floor(retained * recent_fraction + 1e-12)
    recent = tuple(range(max(0, length - recent_count), length))
    protected = set(mandatory).union(recent)
    if len(protected) > retained:
        raise VLCacheReimplementationError(
            "VL-Cache layer budget cannot retain mandatory/recent tokens"
        )
    candidates = tuple(position for position in range(length) if position not in protected)
    ranked = sorted(candidates, key=lambda position: (-scores[position], position))
    selected = tuple(sorted((*protected, *ranked[: retained - len(protected)])))
    return selected, recent


def build_vl_cache_reimpl_plan(
    full_state: FullKVState,
    attention_weights: tuple[Any, ...],
    vl_cache_config: VLCacheConfig,
    cache_config: CacheConfig,
) -> VLCacheCompressionPlan:
    """Apply the paper equations and exact-only Top-K cache selection."""

    if not vl_cache_config.enabled:
        raise VLCacheReimplementationError("vl_cache_reimpl requires vl_cache.enabled=true")
    if full_state.block_size != 1 or cache_config.block_size != 1:
        raise VLCacheReimplementationError("vl_cache_reimpl requires cache.block_size=1")
    if cache_config.budget_unit is not BudgetUnit.BLOCKS:
        raise VLCacheReimplementationError("vl_cache_reimpl supports block budgets only")
    lengths = tuple(layer.sequence_length for layer in full_state.layers)
    kv_heads = tuple(layer.kv_heads for layer in full_state.layers)
    if len(set(lengths)) != 1 or len(set(kv_heads)) != 1:
        raise VLCacheReimplementationError(
            "paper-faithful VL-Cache requires uniform decoder sequence lengths and KV-head counts"
        )
    post_vision_start = infer_post_vision_start(full_state)
    statistics = post_vision_attention_statistics(
        attention_weights,
        kv_heads_by_layer=kv_heads,
        post_vision_start=post_vision_start,
        threshold=vl_cache_config.sparsity_threshold,
        max_post_vision_queries=vl_cache_config.max_post_vision_queries,
    )
    raw_ratios, clipped_ratios = paper_layer_retention_ratios(
        statistics.layer_sparsity,
        retention_ratio=cache_config.retention_ratio,
        minimum=vl_cache_config.min_layer_retention,
        maximum=vl_cache_config.max_layer_retention,
    )
    layer_count = len(lengths)
    sequence_length = lengths[0]
    heads_per_layer = kv_heads[0]
    source_slots = len(full_state.blocks)
    ratio_target_slots = math.floor(source_slots * cache_config.retention_ratio + 1e-12)
    target_slots = min(source_slots, ratio_target_slots, cache_config.budget_value)
    if cache_config.retention_ratio == 1.0:
        if cache_config.budget_value < source_slots:
            raise VLCacheReimplementationError(
                "VL-Cache retention 1.0 requires cache.budget_value to cover the full cache"
            )
        retained_by_layer = lengths
        target_slots = source_slots
    else:
        # The paper gives fractional layer ratios but not integer rounding.  We
        # apportion positions, preserving one shared count for every KV head.
        target_positions = target_slots // heads_per_layer
        target_slots = target_positions * heads_per_layer
        mandatory_counts = tuple(
            sum(
                block.mandatory
                for block in full_state.blocks
                if block.layer == layer_index and block.kv_head == 0
            )
            for layer_index in range(layer_count)
        )
        lower = tuple(
            max(
                1,
                mandatory_count,
                math.floor(vl_cache_config.min_layer_retention * length + 1e-12),
            )
            for length, mandatory_count in zip(lengths, mandatory_counts, strict=True)
        )
        upper = tuple(
            min(length, max(low, math.floor(vl_cache_config.max_layer_retention * length + 1e-12)))
            for length, low in zip(lengths, lower, strict=True)
        )
        retained_by_layer = _bounded_apportion(
            tuple(ratio * length for ratio, length in zip(clipped_ratios, lengths, strict=True)),
            lower,
            upper,
            target_positions,
        )

    blocks = {
        (block.layer, block.kv_head, block.physical_cache_indices[0]): block
        for block in full_state.blocks
    }
    layers: list[VLCacheLayerState] = []
    retained_bytes = 0
    for layer_index in range(layer_count):
        layer_heads: list[VLCacheHeadState] = []
        retained = retained_by_layer[layer_index]
        for kv_head in range(heads_per_layer):
            scores = statistics.kv_head_token_scores[layer_index][kv_head]
            mandatory = tuple(
                position
                for position in range(sequence_length)
                if blocks[(layer_index, kv_head, position)].mandatory
            )
            selected: tuple[int, ...]
            recent: tuple[int, ...]
            if cache_config.retention_ratio == 1.0:
                selected = tuple(range(sequence_length))
                recent = ()
            else:
                selected, recent = _select_positions(
                    scores,
                    retained,
                    mandatory,
                    vl_cache_config.recent_window_fraction,
                )
            selected_bytes = sum(
                blocks[(layer_index, kv_head, position)].byte_size for position in selected
            )
            retained_bytes += selected_bytes
            layer_heads.append(
                VLCacheHeadState(
                    layer_index,
                    kv_head,
                    sequence_length,
                    selected,
                    full_state.logical_positions.gather(selected),
                    mandatory,
                    recent,
                    _sha_floats(scores),
                    float(sum(scores)),
                    selected_bytes,
                )
            )
        layers.append(
            VLCacheLayerState(
                layer_index,
                statistics.layer_query_head_sparsity[layer_index],
                statistics.layer_sparsity[layer_index],
                statistics.layer_density[layer_index],
                raw_ratios[layer_index],
                clipped_ratios[layer_index],
                retained,
                tuple(layer_heads),
            )
        )
    active_slots = sum(head.retained_positions for layer in layers for head in layer.heads)
    return VLCacheCompressionPlan(
        full_state,
        vl_cache_config,
        statistics,
        tuple(layers),
        source_slots,
        active_slots,
        full_state.active_bytes,
        retained_bytes,
        target_slots,
    )


def _position_payload(
    tensor: Any,
    *,
    head_axis: int,
    head: int,
    sequence_axis: int,
    position: int,
) -> Any:
    index: list[Any] = [slice(None)] * int(tensor.ndim)
    index[head_axis] = slice(head, head + 1)
    index[sequence_axis] = slice(position, position + 1)
    result = tensor[tuple(index)]
    clone = getattr(result, "clone", None)
    return clone() if clone is not None else result.copy()


def vl_cache_runtime_payloads(
    plan: VLCacheCompressionPlan,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]],
    tuple[JsonObject, ...],
]:
    """Expose selected exact K/V positions through the shared packer interface."""

    payloads: dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]] = {}
    records: list[JsonObject] = []
    for layer in plan.layers:
        storage = plan.full_state.layers[layer.layer]
        for head in layer.heads:
            entries: list[tuple[int, str, int, Any, Any]] = []
            recent = set(head.recent_positions)
            mandatory = set(head.mandatory_positions)
            for slot, physical in enumerate(head.selected_physical_positions):
                logical = plan.full_state.logical_positions.logical_for_physical(physical)
                entries.append(
                    (
                        logical,
                        "vl_cache_reimpl",
                        physical,
                        _position_payload(
                            storage.key,
                            head_axis=storage.key_head_dimension,
                            head=head.kv_head,
                            sequence_axis=storage.key_sequence_dimension,
                            position=physical,
                        ),
                        _position_payload(
                            storage.value,
                            head_axis=storage.value_head_dimension,
                            head=head.kv_head,
                            sequence_axis=storage.value_sequence_dimension,
                            position=physical,
                        ),
                    )
                )
                reason = (
                    "mandatory"
                    if physical in mandatory
                    else "recent_window"
                    if physical in recent
                    else "post_vision_attention_topk"
                )
                records.append(
                    {
                        "layer": layer.layer,
                        "kv_head": head.kv_head,
                        "slot": slot,
                        "physical_position": physical,
                        "logical_position": logical,
                        "tier": "vl_cache_reimpl_exact",
                        "selection_reason": reason,
                    }
                )
            payloads[(layer.layer, head.kv_head)] = entries
    return payloads, tuple(records)


@dataclass(frozen=True, slots=True)
class VLCacheSensitivityPoint:
    """Structural output for one ambiguity-grid setting (not a measured result)."""

    sparsity_threshold: float
    min_layer_retention: float
    max_layer_retention: float
    recent_window_fraction: float
    max_post_vision_queries: int | None
    layer_sparsity: tuple[float, ...]
    layer_retained_positions: tuple[int, ...]
    active_slots: int
    retained_bytes: int
    selected_positions_sha256: str

    def to_json_object(self) -> JsonObject:
        return {
            "measurement_type": "synthetic_or_calibration_structural_diagnostic",
            "method": "vl_cache_reimpl",
            "sparsity_threshold": self.sparsity_threshold,
            "min_layer_retention": self.min_layer_retention,
            "max_layer_retention": self.max_layer_retention,
            "recent_window_fraction": self.recent_window_fraction,
            "max_post_vision_queries": self.max_post_vision_queries,
            "layer_sparsity": list(self.layer_sparsity),
            "layer_retained_positions": list(self.layer_retained_positions),
            "active_slots": self.active_slots,
            "retained_bytes": self.retained_bytes,
            "selected_positions_sha256": self.selected_positions_sha256,
        }


def analyze_vl_cache_sensitivity(
    full_state: FullKVState,
    attention_weights: tuple[Any, ...],
    base_config: VLCacheConfig,
    cache_config: CacheConfig,
    *,
    calibration_sample_id: str,
    calibration_sample_ids: tuple[str, ...],
    evaluation_sample_ids: tuple[str, ...],
    sparsity_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02),
    min_layer_retentions: tuple[float, ...] = (0.01,),
    max_layer_retentions: tuple[float, ...] = (1.0,),
    recent_window_fractions: tuple[float, ...] = (0.0, 0.1, 0.2),
    max_post_vision_queries: tuple[int | None, ...] = (None, 50),
) -> tuple[VLCacheSensitivityPoint, ...]:
    """Evaluate ambiguous choices on calibration attention only.

    This returns structural cache diagnostics, never task scores.  Benchmark
    scoring remains in the shared evaluation harness on a disjoint split.
    """

    assert_vl_cache_calibration_disjoint(calibration_sample_ids, evaluation_sample_ids)
    if calibration_sample_id not in set(calibration_sample_ids):
        raise VLCacheReimplementationError(
            "VL-Cache sensitivity input is not registered as a calibration sample"
        )
    points: list[VLCacheSensitivityPoint] = []
    for threshold, minimum, maximum, recent, query_limit in product(
        sparsity_thresholds,
        min_layer_retentions,
        max_layer_retentions,
        recent_window_fractions,
        max_post_vision_queries,
    ):
        plan = build_vl_cache_reimpl_plan(
            full_state,
            attention_weights,
            replace(
                base_config,
                sparsity_threshold=threshold,
                min_layer_retention=minimum,
                max_layer_retention=maximum,
                recent_window_fraction=recent,
                max_post_vision_queries=query_limit,
            ),
            cache_config,
        )
        canonical = "|".join(
            f"{head.layer}:{head.kv_head}:"
            + ",".join(str(value) for value in head.selected_physical_positions)
            for layer in plan.layers
            for head in layer.heads
        )
        points.append(
            VLCacheSensitivityPoint(
                threshold,
                minimum,
                maximum,
                recent,
                query_limit,
                plan.statistics.layer_sparsity,
                tuple(layer.retained_positions_per_head for layer in plan.layers),
                plan.active_slots,
                plan.retained_bytes,
                hashlib.sha256(canonical.encode()).hexdigest(),
            )
        )
    return tuple(points)


__all__ = [
    "VLCacheAttentionStatistics",
    "VLCacheCompressionPlan",
    "VLCacheHeadState",
    "VLCacheLayerState",
    "VLCacheReimplementationError",
    "VLCacheSensitivityPoint",
    "analyze_vl_cache_sensitivity",
    "assert_vl_cache_calibration_disjoint",
    "build_vl_cache_reimpl_plan",
    "infer_post_vision_start",
    "paper_layer_retention_ratios",
    "post_vision_attention_statistics",
    "threshold_filter",
    "vl_cache_runtime_payloads",
]
