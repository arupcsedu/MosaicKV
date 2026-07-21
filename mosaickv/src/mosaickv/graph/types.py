"""Immutable sparse-graph schemas for MosaicKV evidence construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mosaickv.cache_state import KVBlockDescriptor, Modality


class EdgeType(StrEnum):
    """Independently weighted evidence relationships."""

    SEMANTIC_SIMILARITY = "semantic_similarity"
    ATTENTION_COACTIVATION = "prompt_attention_coactivation"
    SPATIAL_ADJACENCY = "spatial_adjacency"
    OCR_LAYOUT = "ocr_layout"
    TEMPORAL_ADJACENCY = "temporal_adjacency"
    SAME_EVIDENCE_REGION = "same_evidence_region"
    CROSS_MODAL_ALIGNMENT = "cross_modal_alignment"
    FALLBACK_POSITIONAL = "fallback_positional"


@dataclass(frozen=True, slots=True)
class BlockEvidenceMetadata:
    """Optional normalized annotations for one block-level graph node.

    ``normalized_box`` is always ``(left, top, right, bottom)`` in ``[0, 1]``.
    The cache-state descriptor remains the source of truth for logical positions,
    modality, layer, and KV head.
    """

    image_index: int | None = None
    frame_index: int | None = None
    page_index: int | None = None
    clip_index: int | None = None
    normalized_box: tuple[float, float, float, float] | None = None
    ocr_text: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    page_region: str | None = None
    evidence_region: str | None = None
    alignment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "image_index",
            "frame_index",
            "page_index",
            "clip_index",
            "row_index",
            "column_index",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative when provided")
        if self.normalized_box is not None:
            if len(self.normalized_box) != 4 or any(
                not math.isfinite(value) for value in self.normalized_box
            ):
                raise ValueError("normalized_box must contain four finite coordinates")
            left, top, right, bottom = self.normalized_box
            if not (0 <= left <= right <= 1 and 0 <= top <= bottom <= 1):
                raise ValueError("normalized_box must be (left, top, right, bottom) within [0, 1]")
        for name in ("ocr_text", "page_region", "evidence_region"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must be non-empty when provided")
        if any(not value.strip() for value in self.alignment_ids):
            raise ValueError("alignment_ids must contain non-empty strings")
        if len(set(self.alignment_ids)) != len(self.alignment_ids):
            raise ValueError("alignment_ids cannot contain duplicates")

    @property
    def has_structural_metadata(self) -> bool:
        """Whether this node can participate in a non-semantic metadata edge."""

        return any(
            value is not None
            for value in (
                self.normalized_box,
                self.ocr_text,
                self.row_index,
                self.column_index,
                self.page_region,
                self.evidence_region,
                self.frame_index,
            )
        ) or bool(self.alignment_ids)

    @classmethod
    def from_block(cls, block: KVBlockDescriptor) -> BlockEvidenceMetadata:
        """Derive only metadata that is unambiguously normalized."""

        box = block.region
        normalized_box = (
            box if box is not None and all(0 <= coordinate <= 1 for coordinate in box) else None
        )
        return cls(
            image_index=block.image_index,
            frame_index=block.frame_index,
            page_index=block.page_index,
            clip_index=(
                0 if block.modality is Modality.VIDEO and block.frame_index is not None else None
            ),
            normalized_box=normalized_box,
        )

    def with_block_defaults(self, block: KVBlockDescriptor) -> BlockEvidenceMetadata:
        """Fill source identity fields from the cache descriptor without altering annotations."""

        derived = self.from_block(block)
        return BlockEvidenceMetadata(
            image_index=self.image_index if self.image_index is not None else derived.image_index,
            frame_index=self.frame_index if self.frame_index is not None else derived.frame_index,
            page_index=self.page_index if self.page_index is not None else derived.page_index,
            clip_index=self.clip_index if self.clip_index is not None else derived.clip_index,
            normalized_box=(
                self.normalized_box if self.normalized_box is not None else derived.normalized_box
            ),
            ocr_text=self.ocr_text,
            row_index=self.row_index,
            column_index=self.column_index,
            page_region=self.page_region,
            evidence_region=self.evidence_region,
            alignment_ids=self.alignment_ids,
        )


@dataclass(frozen=True, slots=True)
class PooledBlockDescriptor:
    """One graph node with pooled K/V/hidden semantic features."""

    node_id: int
    block: KVBlockDescriptor
    pooled_key: Any
    pooled_value: Any
    pooled_hidden_state: Any | None
    semantic_embedding: Any
    evidence: BlockEvidenceMetadata

    def __post_init__(self) -> None:
        import numpy as np

        if self.node_id < 0:
            raise ValueError("graph node_id must be nonnegative")
        vectors = (self.pooled_key, self.pooled_value, self.semantic_embedding)
        if any(np.asarray(vector).ndim != 1 for vector in vectors):
            raise ValueError("pooled block vectors must be one-dimensional")
        if self.pooled_hidden_state is not None and np.asarray(self.pooled_hidden_state).ndim != 1:
            raise ValueError("pooled hidden state must be one-dimensional")
        if any(not bool(np.all(np.isfinite(np.asarray(vector)))) for vector in vectors):
            raise ValueError("pooled block vectors must be finite")
        if np.asarray(self.semantic_embedding).size == 0:
            raise ValueError("semantic_embedding cannot be empty")

    @property
    def logical_center(self) -> float:
        positions = self.block.original_logical_positions
        return (positions[0] + positions[-1]) / 2.0


@dataclass(frozen=True, slots=True)
class GraphDiagnostics:
    """Auditable summary computed from the stored sparse graph only."""

    node_count: int
    edge_count: int
    connected_components: int
    modality_mixing: float
    average_degree: float
    evidence_cluster_coverage: float | None
    edge_counts: tuple[tuple[EdgeType, int], ...]
    maximum_out_degree: int
    fallback_used: bool

    def __post_init__(self) -> None:
        if self.node_count < 0 or self.edge_count < 0 or self.connected_components < 0:
            raise ValueError("graph diagnostic counts must be nonnegative")
        if not 0 <= self.modality_mixing <= 1:
            raise ValueError("modality_mixing must be in [0, 1]")
        if self.average_degree < 0 or not math.isfinite(self.average_degree):
            raise ValueError("average_degree must be finite and nonnegative")
        if self.evidence_cluster_coverage is not None and not (
            0 <= self.evidence_cluster_coverage <= 1
        ):
            raise ValueError("evidence_cluster_coverage must be in [0, 1]")
        if sum(count for _edge_type, count in self.edge_counts) != self.edge_count:
            raise ValueError("per-type edge counts do not sum to edge_count")
        if self.maximum_out_degree < 0:
            raise ValueError("maximum_out_degree must be nonnegative")


@dataclass(frozen=True, slots=True)
class SparseCSRGraph:
    """Compressed sparse row view preserving typed parallel edges."""

    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    weights: tuple[float, ...]
    edge_types: tuple[EdgeType, ...]

    def __post_init__(self) -> None:
        if not self.indptr or self.indptr[0] != 0:
            raise ValueError("CSR indptr must start at zero")
        if any(left > right for left, right in zip(self.indptr, self.indptr[1:], strict=False)):
            raise ValueError("CSR indptr must be monotonic")
        edge_count = len(self.indices)
        if self.indptr[-1] != edge_count:
            raise ValueError("CSR indptr terminal value must equal the edge count")
        if len(self.weights) != edge_count or len(self.edge_types) != edge_count:
            raise ValueError("CSR edge arrays must align")


@dataclass(frozen=True, slots=True)
class SparseEvidenceGraph:
    """Typed sparse COO evidence graph over pooled KV blocks."""

    nodes: tuple[PooledBlockDescriptor, ...]
    row_indices: tuple[int, ...]
    column_indices: tuple[int, ...]
    weights: tuple[float, ...]
    edge_types: tuple[EdgeType, ...]
    diagnostics: GraphDiagnostics

    def __post_init__(self) -> None:
        if tuple(node.node_id for node in self.nodes) != tuple(range(len(self.nodes))):
            raise ValueError("graph node IDs must be contiguous and ordered")
        edge_count = len(self.row_indices)
        if not (
            len(self.column_indices) == len(self.weights) == len(self.edge_types) == edge_count
        ):
            raise ValueError("COO edge arrays must align")
        if any(index < 0 or index >= len(self.nodes) for index in self.row_indices):
            raise ValueError("COO row index lies outside the node table")
        if any(index < 0 or index >= len(self.nodes) for index in self.column_indices):
            raise ValueError("COO column index lies outside the node table")
        if any(
            source == target
            for source, target in zip(self.row_indices, self.column_indices, strict=True)
        ):
            raise ValueError("self edges are not stored")
        if any(weight <= 0 or not math.isfinite(weight) for weight in self.weights):
            raise ValueError("stored graph weights must be finite and positive")
        identities = tuple(zip(self.row_indices, self.column_indices, self.edge_types, strict=True))
        if len(set(identities)) != edge_count:
            raise ValueError("duplicate typed COO edges are not allowed")
        if self.diagnostics.node_count != len(self.nodes):
            raise ValueError("diagnostic node count does not match the node table")
        if self.diagnostics.edge_count != edge_count:
            raise ValueError("diagnostic edge count does not match COO storage")

    def to_csr(self) -> SparseCSRGraph:
        """Return a stable row-major CSR view without densifying the graph."""

        order = sorted(
            range(len(self.row_indices)),
            key=lambda index: (
                self.row_indices[index],
                self.column_indices[index],
                self.edge_types[index].value,
            ),
        )
        counts = [0] * len(self.nodes)
        for index in order:
            counts[self.row_indices[index]] += 1
        indptr = [0]
        for count in counts:
            indptr.append(indptr[-1] + count)
        return SparseCSRGraph(
            indptr=tuple(indptr),
            indices=tuple(self.column_indices[index] for index in order),
            weights=tuple(self.weights[index] for index in order),
            edge_types=tuple(self.edge_types[index] for index in order),
        )

    def neighbors(
        self, node_id: int, edge_type: EdgeType | None = None
    ) -> tuple[tuple[int, float, EdgeType], ...]:
        """Return outgoing neighbors from the sparse edge table."""

        if node_id < 0 or node_id >= len(self.nodes):
            raise IndexError(f"graph node does not exist: {node_id}")
        return tuple(
            (target, weight, kind)
            for source, target, weight, kind in zip(
                self.row_indices,
                self.column_indices,
                self.weights,
                self.edge_types,
                strict=True,
            )
            if source == node_id and (edge_type is None or kind is edge_type)
        )


__all__ = [
    "BlockEvidenceMetadata",
    "EdgeType",
    "GraphDiagnostics",
    "PooledBlockDescriptor",
    "SparseCSRGraph",
    "SparseEvidenceGraph",
]
