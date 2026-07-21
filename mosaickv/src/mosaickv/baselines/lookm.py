"""Paper-faithful local LOOK-M reimplementation.

This module is not official LOOK-M code.  It implements equations (4)--(12)
from Wan et al. (Findings of EMNLP 2024) over MosaicKV's inspected FullKV
state.  The official source is pinned separately under ``third_party/LOOK-M``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mosaickv.cache_state import FullKVState, Modality, tensor_storage_bytes
from mosaickv.config import CacheConfig, LookMConfig
from mosaickv.types import JsonObject, LookMMergeStrategy


class LookMReimplementationError(RuntimeError):
    """Raised when the paper algorithm cannot be represented faithfully."""


@dataclass(frozen=True, slots=True)
class LookMAssignment:
    """Nearest conserved pivot selected for one evicted source position."""

    source_physical_position: int
    source_logical_position: int
    pivot_physical_position: int
    pivot_logical_position: int
    cosine_similarity: float

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.source_physical_position,
                self.source_logical_position,
                self.pivot_physical_position,
                self.pivot_logical_position,
            )
        ):
            raise ValueError("LOOK-M source and pivot positions must be nonnegative")
        if not math.isfinite(self.cosine_similarity) or not -1 <= self.cosine_similarity <= 1:
            raise ValueError("LOOK-M cosine similarity must be finite and in [-1, 1]")


@dataclass(frozen=True, slots=True)
class LookMHeadState:
    """One layer/head's paper scores, selected positions, and merged K/V."""

    layer: int
    kv_head: int
    cumulative_attention_scores: tuple[float, ...]
    text_prior_value: float
    important_physical_positions: tuple[int, ...]
    recent_physical_positions: tuple[int, ...]
    selected_physical_positions: tuple[int, ...]
    selected_logical_positions: tuple[int, ...]
    evicted_physical_positions: tuple[int, ...]
    assignments: tuple[LookMAssignment, ...]
    key: Any
    value: Any
    merge_strategy: LookMMergeStrategy

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kv_head < 0:
            raise ValueError("LOOK-M layer and KV head must be nonnegative")
        length = len(self.cumulative_attention_scores)
        if length < 1 or any(
            not math.isfinite(score) or score < 0 for score in self.cumulative_attention_scores
        ):
            raise ValueError("LOOK-M cumulative attention scores must be finite and nonnegative")
        if not math.isfinite(self.text_prior_value) or self.text_prior_value < 0:
            raise ValueError("LOOK-M text-prior value must be finite and nonnegative")
        selected = self.selected_physical_positions
        evicted = self.evicted_physical_positions
        if selected != tuple(sorted(set(selected))):
            raise ValueError("LOOK-M selected positions must be sorted and unique")
        if evicted != tuple(sorted(set(evicted))):
            raise ValueError("LOOK-M evicted positions must be sorted and unique")
        if set(selected).intersection(evicted):
            raise ValueError("LOOK-M selected and evicted positions cannot overlap")
        if tuple(sorted((*selected, *evicted))) != tuple(range(length)):
            raise ValueError("LOOK-M selected and evicted positions must partition the cache")
        if set(self.important_physical_positions).intersection(self.recent_physical_positions):
            raise ValueError("LOOK-M important and recent windows must be disjoint")
        if tuple(sorted((*self.important_physical_positions, *self.recent_physical_positions))) != (
            selected
        ):
            raise ValueError("LOOK-M important + recent positions must equal selected positions")
        if len(self.selected_logical_positions) != len(selected):
            raise ValueError("LOOK-M physical and logical selected positions must align")
        if tuple(item.source_physical_position for item in self.assignments) != evicted:
            raise ValueError("LOOK-M pivot assignments must cover every evicted position")
        if any(item.pivot_physical_position not in selected for item in self.assignments):
            raise ValueError("LOOK-M assignment pivot must be a conserved position")
        key_shape = tuple(int(value) for value in self.key.shape)
        value_shape = tuple(int(value) for value in self.value.shape)
        if len(key_shape) != 4 or len(value_shape) != 4:
            raise ValueError("LOOK-M merged K/V tensors must have rank four")
        if key_shape[1] != 1 or value_shape[1] != 1:
            raise ValueError("LOOK-M head payloads must contain exactly one KV head")
        if key_shape[2] != len(selected) or value_shape[2] != len(selected):
            raise ValueError("LOOK-M merged K/V sequence length must equal selected positions")

    @property
    def active_bytes(self) -> int:
        """Exact merged K/V payload bytes for this layer/head."""

        return tensor_storage_bytes(self.key) + tensor_storage_bytes(self.value)


@dataclass(frozen=True, slots=True)
class LookMCompressionPlan:
    """All paper-algorithm outputs consumed by the unified HF cache packer."""

    full_state: FullKVState
    config: LookMConfig
    heads: tuple[LookMHeadState, ...]
    source_blocks: int
    active_slots: int
    source_bytes: int
    active_bytes: int
    nominal_retention_ratio: float
    realized_slot_retention_ratio: float

    def __post_init__(self) -> None:
        expected_heads = sum(layer.kv_heads for layer in self.full_state.layers)
        if len(self.heads) != expected_heads:
            raise ValueError("LOOK-M plan must cover every layer and KV head")
        identities = tuple((head.layer, head.kv_head) for head in self.heads)
        expected_identities = tuple(
            (layer, head)
            for layer, storage in enumerate(self.full_state.layers)
            for head in range(storage.kv_heads)
        )
        if identities != expected_identities:
            raise ValueError("LOOK-M head states must use canonical layer/head order")
        if self.source_blocks != len(self.full_state.blocks):
            raise ValueError("LOOK-M source block count is inconsistent")
        if self.active_slots != sum(len(head.selected_physical_positions) for head in self.heads):
            raise ValueError("LOOK-M active slot count is inconsistent")
        if self.source_bytes != self.full_state.active_bytes:
            raise ValueError("LOOK-M source byte accounting is inconsistent")
        if self.active_bytes != sum(head.active_bytes for head in self.heads):
            raise ValueError("LOOK-M active byte accounting is inconsistent")
        expected_ratio = self.active_slots / self.source_blocks
        if not math.isclose(
            self.realized_slot_retention_ratio,
            expected_ratio,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError("LOOK-M realized retention ratio is inconsistent")
        if not math.isclose(
            self.nominal_retention_ratio,
            self.config.recent_ratio + self.config.important_ratio,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError("LOOK-M nominal retention ratio is inconsistent")

    @property
    def selected_positions(self) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        """Canonical layer/head selected physical positions for parity reports."""

        return tuple(
            (head.layer, head.kv_head, head.selected_physical_positions) for head in self.heads
        )

    def trace(self) -> JsonObject:
        """Return machine-readable paper scoring and pivot-selection provenance."""

        return {
            "implementation": "lookm_reimpl",
            "official_code": False,
            "paper": "Wan et al., Findings of EMNLP 2024, equations 4-12",
            "official_repository_sha": self.config.official_repository_sha,
            "text_prior": self.config.text_prior,
            "merge_strategy": self.config.merge_strategy.value,
            "recent_ratio": self.config.recent_ratio,
            "important_ratio": self.config.important_ratio,
            "nominal_retention_ratio": self.nominal_retention_ratio,
            "realized_slot_retention_ratio": self.realized_slot_retention_ratio,
            "source_blocks": self.source_blocks,
            "active_slots": self.active_slots,
            "source_bytes": self.source_bytes,
            "selected_source_bytes": self.active_bytes,
            "active_merged_bytes": self.active_bytes,
            "heads": [
                {
                    "layer": head.layer,
                    "kv_head": head.kv_head,
                    "text_prior_value": head.text_prior_value,
                    "source_position_count": len(head.cumulative_attention_scores),
                    "cumulative_attention_score_sum": sum(head.cumulative_attention_scores),
                    "cumulative_attention_score_min": min(head.cumulative_attention_scores),
                    "cumulative_attention_score_max": max(head.cumulative_attention_scores),
                    "cumulative_attention_scores_sha256": _float_tuple_sha256(
                        head.cumulative_attention_scores
                    ),
                    "important_physical_positions": list(head.important_physical_positions),
                    "recent_physical_positions": list(head.recent_physical_positions),
                    "selected_physical_positions": list(head.selected_physical_positions),
                    "selected_logical_positions": list(head.selected_logical_positions),
                    "evicted_position_count": len(head.evicted_physical_positions),
                    "pivot_assignment_count": len(head.assignments),
                    "pivot_assignments_sha256": _assignment_sha256(head.assignments),
                    "pivot_cosine_min": (
                        min(item.cosine_similarity for item in head.assignments)
                        if head.assignments
                        else None
                    ),
                    "pivot_cosine_max": (
                        max(item.cosine_similarity for item in head.assignments)
                        if head.assignments
                        else None
                    ),
                }
                for head in self.heads
            ],
        }


def _float_tuple_sha256(values: tuple[float, ...]) -> str:
    canonical = np.asarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _assignment_sha256(assignments: tuple[LookMAssignment, ...]) -> str:
    digest = hashlib.sha256()
    for assignment in assignments:
        digest.update(
            np.asarray(
                (
                    assignment.source_physical_position,
                    assignment.source_logical_position,
                    assignment.pivot_physical_position,
                    assignment.pivot_logical_position,
                ),
                dtype="<i8",
            ).tobytes(order="C")
        )
        digest.update(np.asarray((assignment.cosine_similarity,), dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        raise LookMReimplementationError("LOOK-M attention or similarity contains NaN/Inf")
    return result


def _take(tensor: Any, axis: int, indices: tuple[int, ...]) -> Any:
    if hasattr(tensor, "index_select"):
        import torch

        index = torch.tensor(indices, dtype=torch.long, device=tensor.device)
        return torch.index_select(tensor, axis, index)
    return np.take(tensor, indices, axis=axis).copy()


def _clone(tensor: Any) -> Any:
    if hasattr(tensor, "detach"):
        return tensor.detach().clone()
    return np.asarray(tensor).copy()


def _head_positions(tensor: Any, head: int, positions: tuple[int, ...]) -> Any:
    selected_head = _take(tensor, 1, (head,))
    return _clone(_take(selected_head, 2, positions))


def _cosine_matrix(evicted: Any, conserved: Any) -> np.ndarray[Any, Any]:
    is_cuda_tensor = (
        hasattr(evicted, "detach")
        and getattr(getattr(evicted, "device", None), "type", None) == "cuda"
    )
    if is_cuda_tensor:
        import torch

        epsilon = torch.finfo(evicted.dtype).eps
        evicted_norm = torch.linalg.vector_norm(evicted, dim=-1, keepdim=True).clamp_min(epsilon)
        conserved_norm = torch.linalg.vector_norm(conserved, dim=-1, keepdim=True).clamp_min(
            epsilon
        )
        similarity = (evicted / evicted_norm) @ (conserved / conserved_norm).transpose(-1, -2)
    else:
        evicted_array = _to_numpy(evicted)
        conserved_array = _to_numpy(conserved)
        epsilon = np.finfo(evicted_array.dtype).eps
        evicted_norm = np.maximum(np.linalg.norm(evicted_array, axis=-1, keepdims=True), epsilon)
        conserved_norm = np.maximum(
            np.linalg.norm(conserved_array, axis=-1, keepdims=True), epsilon
        )
        similarity = (evicted_array / evicted_norm) @ np.swapaxes(
            conserved_array / conserved_norm, -1, -2
        )
    matrix = _to_numpy(similarity)
    if matrix.shape[0] != 1 or matrix.shape[1] != 1:
        raise LookMReimplementationError("lookm_reimpl currently requires batch size one")
    return np.asarray(np.clip(matrix[0, 0], -1.0, 1.0), dtype=np.float64)


def _sum_sequence(tensor: Any) -> Any:
    if hasattr(tensor, "detach"):
        return tensor.sum(dim=2, keepdim=True)
    return np.asarray(tensor).sum(axis=2, keepdims=True)


def _merge_payload(
    conserved: Any,
    evicted: Any,
    pivot_offsets: tuple[int, ...],
    similarities: tuple[float, ...],
    strategy: LookMMergeStrategy,
) -> Any:
    result = _clone(conserved)
    original = _clone(conserved)
    for pivot_offset in range(int(conserved.shape[2])):
        source_offsets = tuple(
            index for index, assigned in enumerate(pivot_offsets) if assigned == pivot_offset
        )
        if not source_offsets:
            continue
        pivot = original[:, :, pivot_offset : pivot_offset + 1, :]
        members = _take(evicted, 2, source_offsets)
        if strategy is LookMMergeStrategy.AVERAGED:
            merged = (pivot + _sum_sequence(members)) / (len(source_offsets) + 1)
        elif strategy is LookMMergeStrategy.PIVOTAL:
            pivotal = (members + pivot) / 2
            merged = (pivot + _sum_sequence(pivotal)) / (len(source_offsets) + 1)
        else:
            if hasattr(members, "detach"):
                import torch

                weights = torch.tensor(
                    [similarities[index] for index in source_offsets],
                    dtype=members.dtype,
                    device=members.device,
                ).reshape(1, 1, -1, 1)
            else:
                weights = np.asarray(
                    [similarities[index] for index in source_offsets],
                    dtype=members.dtype,
                ).reshape(1, 1, -1, 1)
            merged = (pivot + _sum_sequence(members * weights)) / (len(source_offsets) + 1)
        result[:, :, pivot_offset : pivot_offset + 1, :] = merged
    return result


def _modality_by_physical(full_state: FullKVState) -> tuple[Modality, ...]:
    result: list[Modality] = []
    for physical in range(full_state.active_sequence_length):
        logical = full_state.logical_positions.logical_for_physical(physical)
        matches = [span.modality for span in full_state.modality_spans if span.contains(logical)]
        if len(matches) != 1:
            raise LookMReimplementationError(
                f"physical position {physical} has ambiguous modality metadata"
            )
        result.append(matches[0])
    return tuple(result)


def _selection_counts(length: int, config: LookMConfig) -> tuple[int, int]:
    if math.isclose(
        config.recent_ratio + config.important_ratio,
        1.0,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        recent = int(length * config.recent_ratio)
        return recent, length - recent
    return int(length * config.recent_ratio), int(length * config.important_ratio)


def build_lookm_reimpl_plan(
    full_state: FullKVState,
    attention_weights: tuple[Any, ...],
    lookm_config: LookMConfig,
    cache_config: CacheConfig,
) -> LookMCompressionPlan:
    """Apply paper text-prior selection and the configured merge equation."""

    if not lookm_config.enabled or not lookm_config.text_prior:
        raise LookMReimplementationError(
            "lookm_reimpl requires enabled paper text-prior configuration"
        )
    if full_state.block_size != 1:
        raise LookMReimplementationError("lookm_reimpl requires token-sized source blocks")
    if len(attention_weights) != len(full_state.layers):
        raise LookMReimplementationError(
            "LOOK-M attention layer count does not match the source cache"
        )
    length = full_state.active_sequence_length
    recent_count, important_count = _selection_counts(length, lookm_config)
    if recent_count + important_count < 1:
        raise LookMReimplementationError(
            "LOOK-M ratios retain zero positions for this prompt; use a longer prompt or ratio"
        )
    prefix_length = length - recent_count
    if important_count > prefix_length:
        raise LookMReimplementationError("LOOK-M important count exceeds non-recent prefix")
    modalities = _modality_by_physical(full_state)
    recent_positions = tuple(range(prefix_length, length))
    head_states: list[LookMHeadState] = []
    for layer_index, (layer, raw_attention) in enumerate(
        zip(full_state.layers, attention_weights, strict=True)
    ):
        attention = _to_numpy(raw_attention)
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise LookMReimplementationError(
                "lookm_reimpl requires eager attention shaped [1, heads, queries, keys]"
            )
        if attention.shape[1] != layer.kv_heads:
            raise LookMReimplementationError(
                "lookm_reimpl follows the official LLaVA MHA path and requires "
                "query heads to equal KV heads"
            )
        if attention.shape[-1] != length:
            raise LookMReimplementationError("LOOK-M attention key length is inconsistent")
        cumulative = attention.sum(axis=(0, 2))
        for kv_head in range(layer.kv_heads):
            source_scores = cumulative[kv_head]
            text_prior = float(np.max(source_scores))
            ranked_scores = source_scores.copy()
            for physical, modality in enumerate(modalities):
                if modality is Modality.TEXT:
                    ranked_scores[physical] += text_prior
            important_positions = tuple(
                sorted(
                    sorted(
                        range(prefix_length),
                        key=lambda position: (-float(ranked_scores[position]), position),
                    )[:important_count]
                )
            )
            selected_positions = tuple(sorted((*important_positions, *recent_positions)))
            if not set(full_state.mandatory_logical_positions).issubset(
                {
                    full_state.logical_positions.logical_for_physical(position)
                    for position in selected_positions
                }
            ):
                raise LookMReimplementationError(
                    "LOOK-M selected positions removed a mandatory recent token"
                )
            evicted_positions = tuple(
                position for position in range(length) if position not in set(selected_positions)
            )
            conserved_key = _head_positions(layer.key, kv_head, selected_positions)
            conserved_value = _head_positions(layer.value, kv_head, selected_positions)
            assignments: list[LookMAssignment] = []
            pivot_offsets: tuple[int, ...] = ()
            similarities: tuple[float, ...] = ()
            if evicted_positions:
                evicted_key = _head_positions(layer.key, kv_head, evicted_positions)
                evicted_value = _head_positions(layer.value, kv_head, evicted_positions)
                similarity = _cosine_matrix(evicted_key, conserved_key)
                pivot_offsets = tuple(int(np.argmax(row)) for row in similarity)
                similarities = tuple(
                    float(similarity[index, pivot]) for index, pivot in enumerate(pivot_offsets)
                )
                for physical, pivot_offset, cosine in zip(
                    evicted_positions,
                    pivot_offsets,
                    similarities,
                    strict=True,
                ):
                    pivot_physical = selected_positions[pivot_offset]
                    assignments.append(
                        LookMAssignment(
                            physical,
                            full_state.logical_positions.logical_for_physical(physical),
                            pivot_physical,
                            full_state.logical_positions.logical_for_physical(pivot_physical),
                            cosine,
                        )
                    )
                conserved_key = _merge_payload(
                    conserved_key,
                    evicted_key,
                    pivot_offsets,
                    similarities,
                    lookm_config.merge_strategy,
                )
                conserved_value = _merge_payload(
                    conserved_value,
                    evicted_value,
                    pivot_offsets,
                    similarities,
                    lookm_config.merge_strategy,
                )
            selected_logical = tuple(
                full_state.logical_positions.logical_for_physical(position)
                for position in selected_positions
            )
            head_states.append(
                LookMHeadState(
                    layer_index,
                    kv_head,
                    tuple(float(score) for score in source_scores),
                    text_prior,
                    important_positions,
                    recent_positions,
                    selected_positions,
                    selected_logical,
                    evicted_positions,
                    tuple(assignments),
                    conserved_key,
                    conserved_value,
                    lookm_config.merge_strategy,
                )
            )
    active_slots = sum(len(head.selected_physical_positions) for head in head_states)
    if active_slots > cache_config.budget_value:
        raise LookMReimplementationError(
            "LOOK-M active token/head slots exceed cache.budget_value: "
            f"active={active_slots}, budget={cache_config.budget_value}"
        )
    active_bytes = sum(head.active_bytes for head in head_states)
    return LookMCompressionPlan(
        full_state,
        lookm_config,
        tuple(head_states),
        len(full_state.blocks),
        active_slots,
        full_state.active_bytes,
        active_bytes,
        lookm_config.recent_ratio + lookm_config.important_ratio,
        active_slots / len(full_state.blocks),
    )


def lookm_runtime_payloads(
    plan: LookMCompressionPlan,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]],
    tuple[JsonObject, ...],
]:
    """Convert a LOOK-M plan into the shared HF cache-packer payload protocol."""

    payloads: dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]] = {}
    records: list[JsonObject] = []
    for head in plan.heads:
        entries: list[tuple[int, str, int, Any, Any]] = []
        for slot, (physical, logical) in enumerate(
            zip(
                head.selected_physical_positions,
                head.selected_logical_positions,
                strict=True,
            )
        ):
            key = head.key[:, :, slot : slot + 1, :]
            value = head.value[:, :, slot : slot + 1, :]
            entries.append((logical, "lookm_reimpl", physical, key, value))
            records.append(
                {
                    "layer": head.layer,
                    "kv_head": head.kv_head,
                    "slot": slot,
                    "logical_position": logical,
                    "physical_position": physical,
                    "tier": "lookm_reimpl_merged",
                    "source_id": physical,
                }
            )
        payloads[(head.layer, head.kv_head)] = entries
    return payloads, tuple(records)


__all__ = [
    "LookMAssignment",
    "LookMCompressionPlan",
    "LookMHeadState",
    "LookMReimplementationError",
    "build_lookm_reimpl_plan",
    "lookm_runtime_payloads",
]
