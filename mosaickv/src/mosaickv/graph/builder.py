"""Sparse, independently ablatable cross-modal evidence graph construction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from heapq import nsmallest
from typing import Any

import numpy as np

from mosaickv.cache_state import FullKVState
from mosaickv.config import GraphConfig
from mosaickv.graph.pooling import pool_block_descriptors, pool_prompt_attention_coactivation
from mosaickv.graph.types import (
    BlockEvidenceMetadata,
    EdgeType,
    GraphDiagnostics,
    PooledBlockDescriptor,
    SparseEvidenceGraph,
)


@dataclass(frozen=True, slots=True)
class _CandidateEdge:
    source: int
    target: int
    edge_type: EdgeType
    weight: float


class _EdgeAccumulator:
    """Deduplicate candidates, then enforce a per-node/per-source degree cap."""

    def __init__(self, config: GraphConfig) -> None:
        self._config = config
        self._edges: dict[tuple[int, EdgeType], dict[int, float]] = defaultdict(dict)

    def add(
        self,
        source: int,
        target: int,
        edge_type: EdgeType,
        score: float,
        source_weight: float,
    ) -> None:
        if source == target or source_weight == 0:
            return
        if not math.isfinite(score):
            raise ValueError(f"non-finite {edge_type.value} edge score")
        weighted = max(0.0, min(1.0, score)) * source_weight
        if weighted <= 0 or weighted < self._config.min_edge_weight:
            return
        candidates = self._edges[(source, edge_type)]
        candidates[target] = max(weighted, candidates.get(target, 0.0))
        if len(candidates) > self._config.max_neighbors:
            worst_target = min(candidates, key=lambda item: (candidates[item], -item))
            del candidates[worst_target]

    def finalize(self) -> tuple[_CandidateEdge, ...]:
        retained: list[_CandidateEdge] = []
        for (source, edge_type), candidates in self._edges.items():
            retained.extend(
                _CandidateEdge(source, target, edge_type, weight)
                for target, weight in candidates.items()
            )
        return tuple(
            sorted(
                retained,
                key=lambda edge: (edge.source, edge.target, edge.edge_type.value),
            )
        )


def _compatible(
    source: PooledBlockDescriptor,
    target: PooledBlockDescriptor,
    config: GraphConfig,
    *,
    maximum_position_span: int | None = None,
) -> bool:
    if source.node_id == target.node_id:
        return False
    if config.require_same_layer and source.block.layer != target.block.layer:
        return False
    if config.require_same_kv_head and source.block.kv_head != target.block.kv_head:
        return False
    pair = f"{source.block.modality.value}:{target.block.modality.value}"
    if pair not in config.allowed_modality_pairs:
        return False
    return maximum_position_span is None or (
        abs(source.logical_center - target.logical_center) <= maximum_position_span
    )


def _top_indices(
    candidate_ids: Sequence[int], scores: np.ndarray[Any, Any], count: int
) -> tuple[int, ...]:
    """Deterministic top-k with target ID as the tie breaker."""

    if not candidate_ids:
        return ()
    ordered = nsmallest(
        count,
        range(len(candidate_ids)),
        key=lambda index: (-float(scores[index]), candidate_ids[index]),
    )
    return tuple(ordered)


def _add_cosine_edges(
    nodes: tuple[PooledBlockDescriptor, ...],
    features: Mapping[int, np.ndarray[Any, Any]],
    *,
    edge_type: EdgeType,
    source_weight: float,
    maximum_position_span: int | None,
    config: GraphConfig,
    accumulator: _EdgeAccumulator,
) -> None:
    if source_weight == 0 or not features:
        return
    feature_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    normalized: dict[int, np.ndarray[Any, Any]] = {}
    for node_id, feature in features.items():
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not bool(np.all(np.isfinite(vector))):
            raise ValueError(f"{edge_type.value} features must be finite")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            continue
        normalized[node_id] = vector / norm
        node = nodes[node_id]
        feature_groups[
            (
                vector.size,
                node.block.layer if config.require_same_layer else -1,
                node.block.kv_head if config.require_same_kv_head else -1,
            )
        ].append(node_id)

    # Only a chunk-by-candidate score tile is materialized.  With a fixed chunk
    # size this is O(n) auxiliary memory, never an n-by-n matrix.
    for node_ids in feature_groups.values():
        node_ids.sort()
        matrix = np.stack([normalized[node_id] for node_id in node_ids])
        for start in range(0, len(node_ids), config.similarity_chunk_size):
            source_ids = node_ids[start : start + config.similarity_chunk_size]
            score_tile = matrix[start : start + len(source_ids)] @ matrix.T
            for row, source_id in enumerate(source_ids):
                source = nodes[source_id]
                compatible_indices = [
                    index
                    for index, target_id in enumerate(node_ids)
                    if _compatible(
                        source,
                        nodes[target_id],
                        config,
                        maximum_position_span=maximum_position_span,
                    )
                    and float(score_tile[row, index]) > 0
                ]
                candidate_ids = [node_ids[index] for index in compatible_indices]
                candidate_scores = score_tile[row, compatible_indices]
                for selected_index in _top_indices(
                    candidate_ids, candidate_scores, config.max_neighbors
                ):
                    accumulator.add(
                        source_id,
                        candidate_ids[selected_index],
                        edge_type,
                        float(candidate_scores[selected_index]),
                        source_weight,
                    )


def _compatibility_prefix(node: PooledBlockDescriptor, config: GraphConfig) -> tuple[int, int]:
    return (
        node.block.layer if config.require_same_layer else -1,
        node.block.kv_head if config.require_same_kv_head else -1,
    )


def _container_key(node: PooledBlockDescriptor) -> tuple[int | None, int | None]:
    return node.evidence.image_index, node.evidence.page_index


def _rectangle_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    horizontal = max(first[0] - second[2], second[0] - first[2], 0.0)
    vertical = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(horizontal, vertical)


def _add_spatial_edges(
    nodes: tuple[PooledBlockDescriptor, ...],
    config: GraphConfig,
    accumulator: _EdgeAccumulator,
) -> None:
    if config.spatial_weight == 0:
        return
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for node in nodes:
        if node.evidence.normalized_box is not None:
            groups[(*_compatibility_prefix(node, config), *_container_key(node))].append(
                node.node_id
            )
    cell_size = config.spatial_max_distance
    for node_ids in groups.values():
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for node_id in node_ids:
            left, top, right, bottom = nodes[node_id].evidence.normalized_box or (0, 0, 0, 0)
            min_x = max(0, math.floor(min(left, 1 - 1e-12) / cell_size))
            max_x = math.floor(min(right, 1 - 1e-12) / cell_size)
            min_y = max(0, math.floor(min(top, 1 - 1e-12) / cell_size))
            max_y = math.floor(min(bottom, 1 - 1e-12) / cell_size)
            for x_index in range(min_x, max_x + 1):
                for y_index in range(min_y, max_y + 1):
                    grid[(x_index, y_index)].append(node_id)
        for source_id in node_ids:
            source_box = nodes[source_id].evidence.normalized_box
            if source_box is None:  # pragma: no cover - filtered above
                continue
            left, top, right, bottom = source_box
            min_x = max(0, math.floor(max(0.0, left - cell_size) / cell_size))
            max_x = math.floor(min(right + cell_size, 1 - 1e-12) / cell_size)
            min_y = max(0, math.floor(max(0.0, top - cell_size) / cell_size))
            max_y = math.floor(min(bottom + cell_size, 1 - 1e-12) / cell_size)
            candidate_ids: set[int] = set()
            for x_index in range(min_x, max_x + 1):
                for y_index in range(min_y, max_y + 1):
                    candidate_ids.update(grid.get((x_index, y_index), ()))
            scored: dict[int, float] = {}
            for target_id in candidate_ids:
                target = nodes[target_id]
                target_box = target.evidence.normalized_box
                if target_box is None or not _compatible(nodes[source_id], target, config):
                    continue
                distance = _rectangle_distance(source_box, target_box)
                if distance <= config.spatial_max_distance:
                    score = 1.0 - distance / config.spatial_max_distance
                    scored[target_id] = max(
                        score,
                        float(np.finfo(np.float32).eps),
                    )
            for target_id, score in sorted(scored.items(), key=lambda item: (-item[1], item[0]))[
                : config.max_neighbors
            ]:
                accumulator.add(
                    source_id,
                    target_id,
                    EdgeType.SPATIAL_ADJACENCY,
                    score,
                    config.spatial_weight,
                )


def _add_sparse_group(
    node_ids: Sequence[int],
    nodes: tuple[PooledBlockDescriptor, ...],
    config: GraphConfig,
    accumulator: _EdgeAccumulator,
    *,
    edge_type: EdgeType,
    score: float,
    source_weight: float,
    cross_modal_only: bool = False,
) -> None:
    ordered = sorted(set(node_ids), key=lambda node_id: (nodes[node_id].logical_center, node_id))
    for index, source_id in enumerate(ordered):
        first = max(0, index - config.max_neighbors)
        last = min(len(ordered), index + config.max_neighbors + 1)
        for target_id in ordered[first:last]:
            if not _compatible(nodes[source_id], nodes[target_id], config):
                continue
            if (
                cross_modal_only
                and nodes[source_id].block.modality is nodes[target_id].block.modality
            ):
                continue
            accumulator.add(source_id, target_id, edge_type, score, source_weight)


def _add_layout_edges(
    nodes: tuple[PooledBlockDescriptor, ...],
    config: GraphConfig,
    accumulator: _EdgeAccumulator,
) -> None:
    if config.layout_weight == 0:
        return
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    box_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for node in nodes:
        metadata = node.evidence
        prefix = (*_compatibility_prefix(node, config), *_container_key(node))
        if metadata.row_index is not None:
            groups[(*prefix, "row", metadata.row_index)].append(node.node_id)
        if metadata.column_index is not None:
            groups[(*prefix, "column", metadata.column_index)].append(node.node_id)
        if metadata.page_region is not None:
            groups[(*prefix, "page_region", metadata.page_region)].append(node.node_id)
        if metadata.ocr_text is not None:
            groups[(*prefix, "ocr", metadata.ocr_text.casefold().strip())].append(node.node_id)
            if metadata.normalized_box is not None:
                box_groups[prefix].append(node.node_id)
    for node_ids in groups.values():
        _add_sparse_group(
            node_ids,
            nodes,
            config,
            accumulator,
            edge_type=EdgeType.OCR_LAYOUT,
            score=1.0,
            source_weight=config.layout_weight,
        )
    # OCR boxes also provide reading-order/layout evidence when row and column
    # labels are absent.  Comparing only a bounded window in normalized
    # top-to-bottom/left-to-right order keeps candidate storage sparse.
    for node_ids in box_groups.values():
        ordered = sorted(
            node_ids,
            key=lambda node_id: (
                (nodes[node_id].evidence.normalized_box or (0, 0, 0, 0))[1],
                (nodes[node_id].evidence.normalized_box or (0, 0, 0, 0))[0],
                node_id,
            ),
        )
        for index, source_id in enumerate(ordered):
            source_box = nodes[source_id].evidence.normalized_box
            if source_box is None:  # pragma: no cover - filtered above
                continue
            first = max(0, index - config.max_neighbors)
            last = min(len(ordered), index + config.max_neighbors + 1)
            for target_id in ordered[first:last]:
                target_box = nodes[target_id].evidence.normalized_box
                if target_box is None or not _compatible(
                    nodes[source_id], nodes[target_id], config
                ):
                    continue
                horizontal_overlap = min(source_box[2], target_box[2]) >= max(
                    source_box[0], target_box[0]
                )
                vertical_overlap = min(source_box[3], target_box[3]) >= max(
                    source_box[1], target_box[1]
                )
                distance = _rectangle_distance(source_box, target_box)
                if (
                    horizontal_overlap
                    or vertical_overlap
                    or distance <= config.spatial_max_distance
                ):
                    score = 1.0 - min(distance / config.spatial_max_distance, 1.0)
                    accumulator.add(
                        source_id,
                        target_id,
                        EdgeType.OCR_LAYOUT,
                        max(score, float(np.finfo(np.float32).eps)),
                        config.layout_weight,
                    )


def _add_temporal_edges(
    nodes: tuple[PooledBlockDescriptor, ...],
    config: GraphConfig,
    accumulator: _EdgeAccumulator,
) -> None:
    if config.temporal_weight == 0:
        return
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for node in nodes:
        metadata = node.evidence
        if metadata.frame_index is None:
            continue
        layer, head = _compatibility_prefix(node, config)
        groups[(layer, head, metadata.clip_index or 0)].append(node.node_id)
    for node_ids in groups.values():
        ordered = sorted(
            node_ids,
            key=lambda node_id: (
                nodes[node_id].evidence.frame_index or 0,
                nodes[node_id].logical_center,
                node_id,
            ),
        )
        for index, source_id in enumerate(ordered):
            first = max(0, index - config.max_neighbors)
            last = min(len(ordered), index + config.max_neighbors + 1)
            source_frame = nodes[source_id].evidence.frame_index
            for target_id in ordered[first:last]:
                target_frame = nodes[target_id].evidence.frame_index
                if source_frame is None or target_frame is None:
                    continue
                delta = abs(source_frame - target_frame)
                if delta <= config.temporal_window and _compatible(
                    nodes[source_id], nodes[target_id], config
                ):
                    score = 1.0 - delta / (config.temporal_window + 1)
                    accumulator.add(
                        source_id,
                        target_id,
                        EdgeType.TEMPORAL_ADJACENCY,
                        score,
                        config.temporal_weight,
                    )


def _add_identifier_edges(
    nodes: tuple[PooledBlockDescriptor, ...],
    config: GraphConfig,
    accumulator: _EdgeAccumulator,
) -> None:
    evidence_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    alignment_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for node in nodes:
        prefix = _compatibility_prefix(node, config)
        if node.evidence.evidence_region is not None:
            evidence_groups[(*prefix, node.evidence.evidence_region)].append(node.node_id)
        for alignment_id in node.evidence.alignment_ids:
            alignment_groups[(*prefix, alignment_id)].append(node.node_id)
    if config.same_region_weight > 0:
        for node_ids in evidence_groups.values():
            _add_sparse_group(
                node_ids,
                nodes,
                config,
                accumulator,
                edge_type=EdgeType.SAME_EVIDENCE_REGION,
                score=1.0,
                source_weight=config.same_region_weight,
            )
    if config.cross_modal_weight > 0:
        for node_ids in alignment_groups.values():
            _add_sparse_group(
                node_ids,
                nodes,
                config,
                accumulator,
                edge_type=EdgeType.CROSS_MODAL_ALIGNMENT,
                score=1.0,
                source_weight=config.cross_modal_weight,
                cross_modal_only=True,
            )


def _add_fallback_edges(
    nodes: tuple[PooledBlockDescriptor, ...],
    config: GraphConfig,
    accumulator: _EdgeAccumulator,
) -> None:
    if config.fallback_weight == 0:
        return
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node in nodes:
        groups[_compatibility_prefix(node, config)].append(node.node_id)
    for node_ids in groups.values():
        ordered = sorted(node_ids, key=lambda node_id: (nodes[node_id].logical_center, node_id))
        for index, source_id in enumerate(ordered):
            first = max(0, index - config.max_neighbors)
            last = min(len(ordered), index + config.max_neighbors + 1)
            for target_id in ordered[first:last]:
                if _compatible(
                    nodes[source_id],
                    nodes[target_id],
                    config,
                    maximum_position_span=config.fallback_max_position_span,
                ):
                    distance = abs(
                        nodes[source_id].logical_center - nodes[target_id].logical_center
                    )
                    span = config.fallback_max_position_span
                    score = 1.0 if span is None else 1.0 - distance / (span + 1)
                    accumulator.add(
                        source_id,
                        target_id,
                        EdgeType.FALLBACK_POSITIONAL,
                        score,
                        config.fallback_weight,
                    )


def _attention_features(
    attention_coactivation: Any | Mapping[int, Any] | None,
    node_count: int,
) -> dict[int, np.ndarray[Any, Any]]:
    if attention_coactivation is None:
        return {}
    if isinstance(attention_coactivation, Mapping):
        unknown = sorted(set(attention_coactivation) - set(range(node_count)))
        if unknown:
            raise ValueError(f"attention features reference unknown graph node(s): {unknown}")
        return {
            int(node_id): np.asarray(value, dtype=np.float32).reshape(-1)
            for node_id, value in attention_coactivation.items()
        }
    value = attention_coactivation
    if type(value).__module__.startswith("torch"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != node_count:
        raise ValueError("attention_coactivation must have shape [graph_nodes, prompt_features]")
    return {node_id: array[node_id] for node_id in range(node_count)}


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _diagnostics(
    nodes: tuple[PooledBlockDescriptor, ...],
    edges: tuple[_CandidateEdge, ...],
    *,
    fallback_used: bool,
) -> GraphDiagnostics:
    disjoint = _DisjointSet(len(nodes))
    out_degrees = [0] * len(nodes)
    mixing = 0
    for edge in edges:
        disjoint.union(edge.source, edge.target)
        out_degrees[edge.source] += 1
        if nodes[edge.source].block.modality is not nodes[edge.target].block.modality:
            mixing += 1
    components = len({disjoint.find(node_id) for node_id in range(len(nodes))})
    evidence_groups: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for node in nodes:
        if node.evidence.evidence_region is not None:
            evidence_groups[
                (node.block.layer, node.block.kv_head, node.evidence.evidence_region)
            ].append(node.node_id)
    covered = 0
    node_clusters: dict[int, tuple[int, int, str]] = {}
    local_sets: dict[tuple[int, int, str], _DisjointSet] = {}
    local_ids: dict[tuple[int, int, str], dict[int, int]] = {}
    for cluster, node_ids in evidence_groups.items():
        local_sets[cluster] = _DisjointSet(len(node_ids))
        local_ids[cluster] = {node_id: index for index, node_id in enumerate(node_ids)}
        node_clusters.update({node_id: cluster for node_id in node_ids})
    for edge in edges:
        source_cluster = node_clusters.get(edge.source)
        if source_cluster is not None and source_cluster == node_clusters.get(edge.target):
            local_sets[source_cluster].union(
                local_ids[source_cluster][edge.source],
                local_ids[source_cluster][edge.target],
            )
    for cluster, node_ids in evidence_groups.items():
        if len(node_ids) == 1:
            covered += 1
            continue
        local = local_sets[cluster]
        if len({local.find(index) for index in range(len(node_ids))}) == 1:
            covered += 1
    edge_counts = Counter(edge.edge_type for edge in edges)
    return GraphDiagnostics(
        node_count=len(nodes),
        edge_count=len(edges),
        connected_components=components,
        modality_mixing=(mixing / len(edges) if edges else 0.0),
        average_degree=(len(edges) / len(nodes) if nodes else 0.0),
        evidence_cluster_coverage=(covered / len(evidence_groups) if evidence_groups else None),
        edge_counts=tuple((edge_type, edge_counts[edge_type]) for edge_type in EdgeType),
        maximum_out_degree=max(out_degrees, default=0),
        fallback_used=fallback_used,
    )


def build_evidence_graph(
    full_state: FullKVState,
    config: GraphConfig,
    *,
    hidden_states: Sequence[Any] | None = None,
    hidden_sequence_dimension: int = -2,
    attention_coactivation: Any | Mapping[int, Any] | None = None,
    prompt_attentions: Sequence[Any] | None = None,
    metadata_by_node: Mapping[int, BlockEvidenceMetadata] | None = None,
) -> SparseEvidenceGraph:
    """Construct all enabled evidence sources in bounded sparse form.

    Each edge type is capped independently at ``max_neighbors`` outgoing
    neighbors per node.  Removing one source therefore cannot change candidate
    competition or weights in any other source.  Similarity construction uses
    chunked score tiles; structural sources use indexed metadata groups.
    """

    if not config.enabled:
        raise ValueError("cannot build an evidence graph when graph.enabled is false")
    if attention_coactivation is not None and prompt_attentions is not None:
        raise ValueError(
            "provide either pre-pooled attention_coactivation or prompt_attentions, not both"
        )
    nodes = pool_block_descriptors(
        full_state,
        hidden_states=hidden_states,
        hidden_sequence_dimension=hidden_sequence_dimension,
        metadata_by_node=metadata_by_node,
    )
    accumulator = _EdgeAccumulator(config)
    _add_cosine_edges(
        nodes,
        {node.node_id: np.asarray(node.semantic_embedding) for node in nodes},
        edge_type=EdgeType.SEMANTIC_SIMILARITY,
        source_weight=config.semantic_weight,
        maximum_position_span=config.semantic_max_position_span,
        config=config,
        accumulator=accumulator,
    )
    _add_cosine_edges(
        nodes,
        (
            pool_prompt_attention_coactivation(full_state, prompt_attentions)
            if prompt_attentions is not None
            else _attention_features(attention_coactivation, len(nodes))
        ),
        edge_type=EdgeType.ATTENTION_COACTIVATION,
        source_weight=config.attention_weight,
        maximum_position_span=config.attention_max_position_span,
        config=config,
        accumulator=accumulator,
    )
    _add_spatial_edges(nodes, config, accumulator)
    _add_layout_edges(nodes, config, accumulator)
    _add_temporal_edges(nodes, config, accumulator)
    _add_identifier_edges(nodes, config, accumulator)
    metadata_unavailable = not any(node.evidence.has_structural_metadata for node in nodes)
    fallback_used = metadata_unavailable and config.fallback_weight > 0
    if metadata_unavailable:
        _add_fallback_edges(nodes, config, accumulator)
    edges = accumulator.finalize()
    diagnostics = _diagnostics(nodes, edges, fallback_used=fallback_used)
    return SparseEvidenceGraph(
        nodes=nodes,
        row_indices=tuple(edge.source for edge in edges),
        column_indices=tuple(edge.target for edge in edges),
        weights=tuple(edge.weight for edge in edges),
        edge_types=tuple(edge.edge_type for edge in edges),
        diagnostics=diagnostics,
    )


__all__ = ["build_evidence_graph"]
