"""Sparse, independently ablatable cross-modal evidence graph construction."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import pairwise
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

_positional_search = ContextVar("evicastkv_positional_graph_search", default="reference")
_cosine_search = ContextVar("evicastkv_cosine_topk_search", default="stable_sort")


@contextmanager
def cosine_search(mode):
    if mode not in ("stable_sort", "partition"):
        raise ValueError("registered stable-sort/partition cosine search required")
    token = _cosine_search.set(mode)
    try:
        yield
    finally:
        _cosine_search.reset(token)


def _positive_partition_topk(scores, self_columns, count):
    """Exactly the positive descending-score/lower-column-ID top-k set.

    Partition determines the cutoff, then a prefix rank resolves ALL ties in
    original sorted-node-ID order. Never trust argpartition's arbitrary ties.
    Return COO row/column order; final graph sorting is unchanged.
    """
    values = np.asarray(scores)
    diagonal = np.asarray(self_columns, dtype=np.int64)
    if (
        values.ndim != 2
        or values.shape[1] < 1
        or diagonal.shape != (values.shape[0],)
        or type(count) is not int
        or count < 1
        or not bool(np.isfinite(values).all())
        or bool(((diagonal < 0) | (diagonal >= values.shape[1])).any())
    ):
        raise ValueError("finite score tiles, valid self columns and positive top-k required")
    work = values.copy()
    work[np.arange(len(work)), diagonal] = -np.inf
    k = min(count, work.shape[1])
    cutoff = np.partition(work, work.shape[1] - k, axis=1)[:, work.shape[1] - k]
    above = work > cutoff[:, None]
    ties = work == cutoff[:, None]
    needed = k - np.count_nonzero(above, axis=1)
    retained = (above | (ties & (np.cumsum(ties, axis=1) <= needed[:, None]))) & (work > 0)
    return np.nonzero(retained)


@contextmanager
def positional_search(mode):
    """Execution-only policy: both paths must produce exactly the same edges."""
    if mode not in ("reference", "indexed"):
        raise ValueError("registered reference/indexed positional search required")
    token = _positional_search.set(mode)
    try:
        yield
    finally:
        _positional_search.reset(token)


def _nearest_center_indices(centers, source_index, count, span):
    """Exact distance/index top-k on sorted centers, including duplicate centers.

    Consume equal-center groups in ascending index order. A naive adjacent
    two-pointer walk gets duplicate-center ties WRONG on the left side.
    Only selected groups/indices are visited; do not scan every target per node.
    """
    center = centers[source_index]
    first, last = bisect_left(centers, center), bisect_right(centers, center)
    chosen = []
    for index in range(first, last):
        if index != source_index:
            chosen.append(index)
            if len(chosen) == count:
                return tuple(chosen)
    left, right = first - 1, last
    while len(chosen) < count and (left >= 0 or right < len(centers)):
        left_distance = abs(center - centers[left]) if left >= 0 else math.inf
        right_distance = abs(center - centers[right]) if right < len(centers) else math.inf
        distance = min(left_distance, right_distance)
        if span is not None and distance > span:
            break
        if left_distance <= right_distance:
            start, end = bisect_left(centers, centers[left], 0, left + 1), left + 1
            left = start - 1
        else:
            start, end = right, bisect_right(centers, centers[right], right, len(centers))
            right = end
        chosen.extend(range(start, min(end, start + count - len(chosen))))
    return tuple(chosen)


class _EdgeAccumulator:
    """Deduplicate candidates, then enforce a per-node/per-source degree cap."""

    def __init__(self, config: GraphConfig) -> None:
        self._config = config
        self._edges: dict[tuple[int, EdgeType], dict[int, float]] = defaultdict(dict)
        self._preselected_rows: list[int] = []
        self._preselected_columns: list[int] = []
        self._preselected_types: list[EdgeType] = []
        self._preselected_weights: list[float] = []
        # The routed fast path produces one already-ranked top-k segment per
        # (source tile, edge type).  Keeping those segments as contiguous
        # arrays avoids constructing one Python tuple/dictionary entry per
        # candidate before the COO sort.  Scalar structural edge producers
        # continue to use the small lists above.
        self._segmented_preselected: list[
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], EdgeType]
        ] = []

    @property
    def _uses_segmented_topk(self) -> bool:
        return self._config.edge_accumulation_mode == "segmented_tensor_topk"

    def _add_preselected_array(
        self,
        sources: np.ndarray[Any, Any],
        targets: np.ndarray[Any, Any],
        edge_type: EdgeType,
        scores: np.ndarray[Any, Any],
        source_weight: float,
    ) -> None:
        """Store a vectorized, duplicate-free source/target edge segment."""

        if source_weight == 0 or sources.size == 0:
            return
        if sources.shape != targets.shape or sources.shape != scores.shape:
            raise ValueError("preselected edge batches must align")
        source_array = np.asarray(sources, dtype=np.int64).reshape(-1)
        target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
        score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if not bool(np.all(np.isfinite(score_array))):
            raise ValueError(f"non-finite {edge_type.value} edge score")
        if bool(np.any(source_array == target_array)):
            raise ValueError("preselected edge batches cannot contain self edges")
        weighted = np.clip(score_array, 0.0, 1.0) * source_weight
        minimum_weight = max(float(self._config.min_edge_weight), float(np.finfo(np.float64).tiny))
        retained = weighted >= minimum_weight
        if not bool(np.any(retained)):
            return
        self._segmented_preselected.append(
            (
                source_array[retained],
                target_array[retained],
                weighted[retained],
                edge_type,
            )
        )

    def add_preselected(
        self,
        source: int,
        target: int,
        edge_type: EdgeType,
        score: float,
        source_weight: float,
    ) -> None:
        """Append an already top-k, duplicate-free candidate without a trim dictionary."""

        if source == target or source_weight == 0:
            return
        if not math.isfinite(score):
            raise ValueError(f"non-finite {edge_type.value} edge score")
        weighted = max(0.0, min(1.0, score)) * source_weight
        if weighted <= 0 or weighted < self._config.min_edge_weight:
            return
        self._preselected_rows.append(source)
        self._preselected_columns.append(target)
        self._preselected_types.append(edge_type)
        self._preselected_weights.append(weighted)

    def add_preselected_many(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        edge_type: EdgeType,
        scores: Sequence[float],
        source_weight: float,
    ) -> None:
        """Append one duplicate-free top-k batch with vectorized validation."""

        if source_weight == 0 or not sources:
            return
        if not (len(sources) == len(targets) == len(scores)):
            raise ValueError("preselected edge batches must align")
        source_array = np.asarray(sources, dtype=np.int64)
        target_array = np.asarray(targets, dtype=np.int64)
        score_array = np.asarray(scores, dtype=np.float64)
        if self._uses_segmented_topk:
            self._add_preselected_array(
                source_array,
                target_array,
                edge_type,
                score_array,
                source_weight,
            )
            return
        if not bool(np.all(np.isfinite(score_array))):
            raise ValueError(f"non-finite {edge_type.value} edge score")
        weighted = np.clip(score_array, 0.0, 1.0) * source_weight
        minimum_weight = max(float(self._config.min_edge_weight), float(np.finfo(np.float64).tiny))
        retained = weighted >= minimum_weight
        retained_sources = tuple(
            source for source, keep in zip(sources, retained, strict=True) if keep
        )
        retained_targets = tuple(
            target for target, keep in zip(targets, retained, strict=True) if keep
        )
        retained_weights = tuple(float(value) for value in weighted[retained])
        if any(
            source == target
            for source, target in zip(retained_sources, retained_targets, strict=True)
        ):
            raise ValueError("preselected edge batches cannot contain self edges")
        self._preselected_rows.extend(retained_sources)
        self._preselected_columns.extend(retained_targets)
        self._preselected_types.extend((edge_type,) * len(retained_sources))
        self._preselected_weights.extend(retained_weights)

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

    def finalize(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...], tuple[EdgeType, ...]]:
        rows = self._preselected_rows
        columns = self._preselected_columns
        edge_types = self._preselected_types
        weights = self._preselected_weights
        for (source, edge_type), edge_candidates in self._edges.items():
            for target, weight in edge_candidates.items():
                rows.append(source)
                columns.append(target)
                edge_types.append(edge_type)
                weights.append(weight)
        if not rows and not self._segmented_preselected:
            return (), (), (), ()
        ordered_edge_types = tuple(sorted(EdgeType, key=lambda item: item.value))
        type_codes = {edge_type: index for index, edge_type in enumerate(ordered_edge_types)}
        row_segments: list[np.ndarray[Any, Any]] = [np.asarray(rows, dtype=np.int64)]
        column_segments: list[np.ndarray[Any, Any]] = [np.asarray(columns, dtype=np.int64)]
        weight_segments: list[np.ndarray[Any, Any]] = [np.asarray(weights, dtype=np.float64)]
        type_segments: list[np.ndarray[Any, Any]] = [
            np.fromiter(
                (type_codes[edge_type] for edge_type in edge_types),
                dtype=np.int16,
                count=len(edge_types),
            )
        ]
        for (
            segment_rows,
            segment_columns,
            segment_weights,
            edge_type,
        ) in self._segmented_preselected:
            row_segments.append(segment_rows)
            column_segments.append(segment_columns)
            weight_segments.append(segment_weights)
            type_segments.append(np.full(segment_rows.shape, type_codes[edge_type], dtype=np.int16))
        row_array = np.concatenate(row_segments)
        column_array = np.concatenate(column_segments)
        weight_array = np.concatenate(weight_segments)
        type_array = np.concatenate(type_segments)
        order = np.lexsort(
            (
                type_array,
                column_array,
                row_array,
            )
        )
        ordered_types = tuple(ordered_edge_types[int(type_array[int(index)])] for index in order)
        return (
            tuple(int(value) for value in row_array[order]),
            tuple(int(value) for value in column_array[order]),
            tuple(float(value) for value in weight_array[order]),
            ordered_types,
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
    identifiers = np.asarray(candidate_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    ordered = np.lexsort((identifiers, -values))
    return tuple(int(index) for index in ordered[:count])


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
    add_preselected = config.edge_accumulation_mode in {
        "preselected",
        "segmented_tensor_topk",
    }
    batch_sources: list[int] = []
    batch_targets: list[int] = []
    batch_scores: list[float] = []
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
        unrestricted_group = (
            config.require_same_layer
            and config.require_same_kv_head
            and maximum_position_span is None
            and len(config.allowed_modality_pairs) == 9
        )
        for start in range(0, len(node_ids), config.similarity_chunk_size):
            source_ids = node_ids[start : start + config.similarity_chunk_size]
            score_tile = matrix[start : start + len(source_ids)] @ matrix.T
            if unrestricted_group:
                if _cosine_search.get() == "partition" and accumulator._uses_segmented_topk:
                    row_indices, columns = _positive_partition_topk(
                        score_tile, np.arange(start, start + len(source_ids)), config.max_neighbors
                    )
                    accumulator._add_preselected_array(
                        np.asarray(source_ids, dtype=np.int64)[row_indices],
                        np.asarray(node_ids, dtype=np.int64)[columns],
                        edge_type,
                        score_tile[row_indices, columns],
                        source_weight,
                    )
                    continue
                # Columns follow sorted node IDs. A stable descending sort is
                # therefore exactly the documented lower-node-ID tie break,
                # but performs one vectorized ordering operation per tile
                # instead of one lexsort/allocation per source node.
                positive = score_tile > 0
                for row in range(len(source_ids)):
                    positive[row, start + row] = False
                ranked = np.argsort(-score_tile, axis=1, kind="stable")
                if add_preselected:
                    ranked_positive = np.take_along_axis(positive, ranked, axis=1)
                    retained_rank = np.cumsum(ranked_positive, axis=1) <= config.max_neighbors
                    retained = ranked_positive & retained_rank
                    row_indices, rank_indices = np.nonzero(retained)
                    columns = ranked[row_indices, rank_indices]
                    if accumulator._uses_segmented_topk:
                        accumulator._add_preselected_array(
                            np.asarray(source_ids, dtype=np.int64)[row_indices],
                            np.asarray(node_ids, dtype=np.int64)[columns],
                            edge_type,
                            score_tile[row_indices, columns],
                            source_weight,
                        )
                    else:
                        batch_sources.extend(source_ids[int(row)] for row in row_indices)
                        batch_targets.extend(node_ids[int(column)] for column in columns)
                        batch_scores.extend(
                            float(score_tile[row, column])
                            for row, column in zip(row_indices, columns, strict=True)
                        )
                else:
                    for row, source_id in enumerate(source_ids):
                        retained_count = 0
                        for column in ranked[row]:
                            if not positive[row, column]:
                                continue
                            accumulator.add(
                                source_id,
                                node_ids[int(column)],
                                edge_type,
                                float(score_tile[row, column]),
                                source_weight,
                            )
                            retained_count += 1
                            if retained_count == config.max_neighbors:
                                break
                continue
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
    if batch_sources:
        accumulator.add_preselected_many(
            batch_sources,
            batch_targets,
            edge_type,
            batch_scores,
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
    *,
    eligible_node_ids: frozenset[int] | None = None,
) -> None:
    if config.spatial_weight == 0:
        return
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for node in nodes:
        if eligible_node_ids is not None and node.node_id not in eligible_node_ids:
            continue
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
    *,
    eligible_node_ids: frozenset[int] | None = None,
) -> None:
    if config.layout_weight == 0:
        return
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    box_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for node in nodes:
        if eligible_node_ids is not None and node.node_id not in eligible_node_ids:
            continue
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
    *,
    eligible_node_ids: frozenset[int] | None = None,
) -> None:
    if config.temporal_weight == 0:
        return
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for node in nodes:
        if eligible_node_ids is not None and node.node_id not in eligible_node_ids:
            continue
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
    *,
    eligible_node_ids: frozenset[int] | None = None,
) -> None:
    evidence_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    alignment_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for node in nodes:
        if eligible_node_ids is not None and node.node_id not in eligible_node_ids:
            continue
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
    *,
    eligible_node_ids: frozenset[int] | None = None,
) -> None:
    if config.fallback_weight == 0:
        return
    add_edge = (
        accumulator.add_preselected
        if config.edge_accumulation_mode in {"preselected", "segmented_tensor_topk"}
        else accumulator.add
    )
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node in nodes:
        if eligible_node_ids is not None and node.node_id not in eligible_node_ids:
            continue
        groups[_compatibility_prefix(node, config)].append(node.node_id)
    unrestricted_templates: dict[
        tuple[tuple[float, ...], int | None, int],
        tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]],
    ] = {}
    batch_sources: list[int] = []
    batch_targets: list[int] = []
    batch_scores: list[float] = []
    for node_ids in groups.values():
        # FullKV blockization emits each layer/head group in logical order.
        # Retain a defensive fallback for independently constructed test graphs
        # without paying for an unconditional sort in normal runs.
        ordered = node_ids
        if any(
            (nodes[left].logical_center, left) > (nodes[right].logical_center, right)
            for left, right in pairwise(node_ids)
        ):
            ordered = sorted(
                node_ids,
                key=lambda node_id: (nodes[node_id].logical_center, node_id),
            )
        centers = tuple(nodes[node_id].logical_center for node_id in ordered)
        unrestricted_group = (
            config.require_same_layer
            and config.require_same_kv_head
            and len(config.allowed_modality_pairs) == 9
        )
        if unrestricted_group:
            template_key = (
                centers,
                config.fallback_max_position_span,
                config.max_neighbors,
            )
            template = unrestricted_templates.get(template_key)
            if template is None:
                template_sources: list[int] = []
                template_targets: list[int] = []
                template_scores: list[float] = []
                for index, source_center in enumerate(centers):
                    neighbor_candidates = (
                        tuple(
                            (abs(source_center - centers[target]), target)
                            for target in _nearest_center_indices(
                                centers,
                                index,
                                config.max_neighbors,
                                config.fallback_max_position_span,
                            )
                        )
                        if _positional_search.get() == "indexed"
                        else tuple(
                            sorted(
                                (
                                    (abs(source_center - target_center), target_index)
                                    for target_index, target_center in enumerate(centers)
                                    if target_index != index
                                    and (
                                        config.fallback_max_position_span is None
                                        or abs(source_center - target_center)
                                        <= config.fallback_max_position_span
                                    )
                                ),
                                key=lambda item: (item[0], item[1]),
                            )[: config.max_neighbors]
                        )
                    )
                    for distance, target_index in neighbor_candidates:
                        template_sources.append(index)
                        template_targets.append(target_index)
                        span = config.fallback_max_position_span
                        template_scores.append(1.0 if span is None else 1.0 - distance / (span + 1))
                template = (
                    tuple(template_sources),
                    tuple(template_targets),
                    tuple(template_scores),
                )
                unrestricted_templates[template_key] = template
            local_source_indices, local_target_indices, local_edge_scores = template
            batch_sources.extend(ordered[index] for index in local_source_indices)
            batch_targets.extend(ordered[index] for index in local_target_indices)
            batch_scores.extend(local_edge_scores)
            continue
        for index, source_id in enumerate(ordered):
            source = nodes[source_id]
            source_center = centers[index]
            first = max(0, index - config.max_neighbors)
            last = min(len(ordered), index + config.max_neighbors + 1)
            compatible_candidates: list[tuple[int, float]] = []
            for target_index, target_id in enumerate(ordered[first:last], start=first):
                target = nodes[target_id]
                compatible = _compatible(
                    source,
                    target,
                    config,
                    maximum_position_span=config.fallback_max_position_span,
                )
                if compatible:
                    distance = abs(source_center - centers[target_index])
                    span = config.fallback_max_position_span
                    score = 1.0 if span is None else 1.0 - distance / (span + 1)
                    compatible_candidates.append((target_id, score))
            for target_id, score in sorted(
                compatible_candidates,
                key=lambda item: (-item[1], item[0]),
            )[: config.max_neighbors]:
                add_edge(
                    source_id,
                    target_id,
                    EdgeType.FALLBACK_POSITIONAL,
                    score,
                    config.fallback_weight,
                )
    if batch_sources:
        accumulator.add_preselected_many(
            batch_sources,
            batch_targets,
            EdgeType.FALLBACK_POSITIONAL,
            batch_scores,
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
    rows: Sequence[int],
    columns: Sequence[int],
    edge_types: Sequence[EdgeType],
    *,
    fallback_used: bool,
    candidate_node_ids: frozenset[int] | None = None,
    candidate_graph_mode: str = "full",
) -> GraphDiagnostics:
    edge_count = len(rows)
    row_array = np.asarray(rows, dtype=np.int64)
    column_array = np.asarray(columns, dtype=np.int64)
    out_degrees = np.bincount(row_array, minlength=len(nodes))
    modality_codes = (
        {modality: index for index, modality in enumerate(type(nodes[0].block.modality))}
        if nodes
        else {}
    )
    modalities = (
        np.fromiter(
            (modality_codes[node.block.modality] for node in nodes),
            dtype=np.int8,
            count=len(nodes),
        )
        if nodes
        else np.empty(0, dtype=np.int8)
    )
    mixing = int(np.count_nonzero(modalities[row_array] != modalities[column_array]))
    try:
        from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
        from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]

        adjacency = coo_matrix(
            (np.ones(edge_count, dtype=np.int8), (row_array, column_array)),
            shape=(len(nodes), len(nodes)),
        )
        components = int(
            connected_components(
                adjacency,
                directed=False,
                return_labels=False,
            )
        )
    except ImportError:  # pragma: no cover - minimal installs use the reference fallback
        disjoint = _DisjointSet(len(nodes))
        for source, target in zip(rows, columns, strict=True):
            disjoint.union(source, target)
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
    for source, target in zip(rows, columns, strict=True):
        source_cluster = node_clusters.get(source)
        if source_cluster is not None and source_cluster == node_clusters.get(target):
            local_sets[source_cluster].union(
                local_ids[source_cluster][source],
                local_ids[source_cluster][target],
            )
    for cluster, node_ids in evidence_groups.items():
        if len(node_ids) == 1:
            covered += 1
            continue
        local = local_sets[cluster]
        if len({local.find(index) for index in range(len(node_ids))}) == 1:
            covered += 1
    edge_counts = Counter(edge_types)
    return GraphDiagnostics(
        node_count=len(nodes),
        edge_count=edge_count,
        connected_components=components,
        modality_mixing=(mixing / edge_count if edge_count else 0.0),
        average_degree=(edge_count / len(nodes) if nodes else 0.0),
        evidence_cluster_coverage=(covered / len(evidence_groups) if evidence_groups else None),
        edge_counts=tuple((edge_type, edge_counts[edge_type]) for edge_type in EdgeType),
        maximum_out_degree=int(out_degrees.max(initial=0)),
        fallback_used=fallback_used,
        candidate_node_count=(None if candidate_node_ids is None else len(candidate_node_ids)),
        candidate_fraction=(
            None
            if candidate_node_ids is None or not nodes
            else len(candidate_node_ids) / len(nodes)
        ),
        candidate_graph_mode=candidate_graph_mode,
    )


def _expand_candidate_neighbors(
    nodes: tuple[PooledBlockDescriptor, ...],
    candidate_node_ids: frozenset[int],
    hops: int,
) -> frozenset[int]:
    """Add bounded positional and explicit-evidence neighbors to a shortlist.

    Image/page identity alone is intentionally not a grouping key because it
    would commonly expand one visual candidate to every patch in the image.
    Fine region, OCR/layout, alignment, and frame identifiers are selective
    enough to preserve local evidence without recreating the full graph.
    """

    if hops < 0:
        raise ValueError("candidate neighbor hops must be nonnegative")
    selected = set(candidate_node_ids)
    ordered_by_head: dict[tuple[int, int], list[int]] = defaultdict(list)
    evidence_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for node in nodes:
        block = node.block
        ordered_by_head[(block.layer, block.kv_head)].append(node.node_id)
        evidence = node.evidence
        for kind, value in (
            ("region", evidence.evidence_region),
            ("page_region", evidence.page_region),
            ("row", evidence.row_index),
            ("column", evidence.column_index),
            ("frame", evidence.frame_index),
        ):
            if value is not None:
                evidence_groups[(block.layer, block.kv_head, kind, value)].append(node.node_id)
        for alignment_id in evidence.alignment_ids:
            evidence_groups[(block.layer, block.kv_head, "alignment", alignment_id)].append(
                node.node_id
            )
    for group in ordered_by_head.values():
        group.sort(key=lambda node_id: (nodes[node_id].logical_center, node_id))
        offsets = {node_id: index for index, node_id in enumerate(group)}
        for node_id in tuple(selected.intersection(offsets)):
            center = offsets[node_id]
            selected.update(group[max(0, center - hops) : center + hops + 1])
    selected_snapshot = frozenset(selected)
    for members in evidence_groups.values():
        if selected_snapshot.intersection(members):
            selected.update(members)
    return frozenset(selected)


def build_evidence_graph(
    full_state: FullKVState,
    config: GraphConfig,
    *,
    hidden_states: Sequence[Any] | None = None,
    hidden_sequence_dimension: int = -2,
    attention_coactivation: Any | Mapping[int, Any] | None = None,
    prompt_attentions: Sequence[Any] | None = None,
    prompt_attention_query_window: int | None = None,
    metadata_by_node: Mapping[int, BlockEvidenceMetadata] | None = None,
    similarity_candidate_node_ids: Sequence[int] | None = None,
    source_node_ids: Sequence[int] | None = None,
) -> SparseEvidenceGraph:
    """Construct all enabled evidence sources in bounded sparse form.

    Each edge type is capped independently at ``max_neighbors`` outgoing
    neighbors per node.  Removing one source therefore cannot change candidate
    competition or weights in any other source.  Similarity construction uses
    chunked score tiles; structural sources use indexed metadata groups.
    """

    if attention_coactivation is not None and prompt_attentions is not None:
        raise ValueError(
            "provide either pre-pooled attention_coactivation or prompt_attentions, not both"
        )
    source_table = (
        tuple(range(len(full_state.blocks)))
        if source_node_ids is None
        else tuple(int(node_id) for node_id in source_node_ids)
    )
    if len(source_table) != len(set(source_table)) or tuple(sorted(source_table)) != source_table:
        raise ValueError("graph source node IDs must be unique and ordered")
    unknown_sources = set(source_table) - set(range(len(full_state.blocks)))
    if unknown_sources:
        raise ValueError(f"graph contains unknown source nodes: {sorted(unknown_sources)}")
    source_to_local = {
        source_node_id: local_node_id for local_node_id, source_node_id in enumerate(source_table)
    }
    similarity_source_candidates = (
        None
        if similarity_candidate_node_ids is None
        else frozenset(int(node_id) for node_id in similarity_candidate_node_ids)
    )
    if similarity_source_candidates is not None:
        unknown = similarity_source_candidates - set(source_table)
        if unknown:
            raise ValueError(
                f"similarity shortlist contains unknown graph nodes: {sorted(unknown)}"
            )
    similarity_candidates = (
        None
        if similarity_source_candidates is None
        else frozenset(source_to_local[node_id] for node_id in similarity_source_candidates)
    )
    if config.candidate_descriptor_pooling and similarity_candidates is None:
        raise ValueError(
            "candidate descriptor pooling requires explicit similarity candidate node IDs"
        )
    nodes = pool_block_descriptors(
        full_state,
        hidden_states=hidden_states,
        hidden_sequence_dimension=hidden_sequence_dimension,
        metadata_by_node=metadata_by_node,
        host_transfer_mode=config.attention_host_transfer_mode,
        eligible_node_ids=(
            similarity_candidate_node_ids if config.candidate_descriptor_pooling else None
        ),
        node_table_ids=source_table,
    )
    graph_candidates = similarity_candidates
    if graph_candidates is not None and config.candidate_graph_mode == "shortlist_rerank":
        graph_candidates = _expand_candidate_neighbors(
            nodes,
            graph_candidates,
            config.candidate_neighbor_hops,
        )
        # Candidate expansion occurs after the first pooled pass. Structural
        # neighbors can participate through metadata edges, but deliberately
        # retain zero learned descriptors unless they were in the original
        # prompt shortlist. This avoids a second device-to-host cache scan.
    if not config.enabled:
        rows: tuple[int, ...] = ()
        columns: tuple[int, ...] = ()
        weights: tuple[float, ...] = ()
        edge_types: tuple[EdgeType, ...] = ()
        return SparseEvidenceGraph._from_validated(
            nodes=nodes,
            row_indices=rows,
            column_indices=columns,
            weights=weights,
            edge_types=edge_types,
            diagnostics=_diagnostics(
                nodes,
                rows,
                columns,
                edge_types,
                fallback_used=False,
                candidate_node_ids=graph_candidates,
                candidate_graph_mode=config.candidate_graph_mode,
            ),
        )
    accumulator = _EdgeAccumulator(config)
    _add_cosine_edges(
        nodes,
        {
            node.node_id: np.asarray(node.semantic_embedding)
            for node in nodes
            if similarity_candidates is None or node.node_id in similarity_candidates
        },
        edge_type=EdgeType.SEMANTIC_SIMILARITY,
        source_weight=config.semantic_weight,
        maximum_position_span=config.semantic_max_position_span,
        config=config,
        accumulator=accumulator,
    )
    coactivation_features = (
        {
            source_to_local[source_node_id]: feature
            for source_node_id, feature in pool_prompt_attention_coactivation(
                full_state,
                prompt_attentions,
                host_transfer_mode=config.attention_host_transfer_mode,
                eligible_node_ids=similarity_candidate_node_ids,
                query_window=prompt_attention_query_window,
            ).items()
            if source_node_id in source_to_local
        }
        if prompt_attentions is not None
        else {
            node_id: feature
            for node_id, feature in _attention_features(
                attention_coactivation,
                len(nodes),
            ).items()
            if similarity_candidates is None or node_id in similarity_candidates
        }
    )
    _add_cosine_edges(
        nodes,
        coactivation_features,
        edge_type=EdgeType.ATTENTION_COACTIVATION,
        source_weight=config.attention_weight,
        maximum_position_span=config.attention_max_position_span,
        config=config,
        accumulator=accumulator,
    )
    structural_candidates = (
        graph_candidates if config.candidate_graph_mode == "shortlist_rerank" else None
    )
    _add_spatial_edges(
        nodes,
        config,
        accumulator,
        eligible_node_ids=structural_candidates,
    )
    _add_layout_edges(
        nodes,
        config,
        accumulator,
        eligible_node_ids=structural_candidates,
    )
    _add_temporal_edges(
        nodes,
        config,
        accumulator,
        eligible_node_ids=structural_candidates,
    )
    _add_identifier_edges(
        nodes,
        config,
        accumulator,
        eligible_node_ids=structural_candidates,
    )
    # Positional locality is an independent sparse evidence source, not a
    # replacement for semantic/structural metadata.  Keeping it for every node
    # gives downstream prototype construction a bounded local compatibility
    # path even when learned descriptors select mutually dissimilar neighbors.
    # Each source still contributes at most ``max_neighbors`` outgoing edges,
    # so the graph remains O(n*d).
    fallback_used = bool(config.candidate_graph_mode == "full" and config.fallback_weight > 0)
    if config.candidate_graph_mode == "full":
        _add_fallback_edges(nodes, config, accumulator)
    rows, columns, weights, edge_types = accumulator.finalize()
    diagnostics = _diagnostics(
        nodes,
        rows,
        columns,
        edge_types,
        fallback_used=fallback_used,
        candidate_node_ids=graph_candidates,
        candidate_graph_mode=config.candidate_graph_mode,
    )
    return SparseEvidenceGraph._from_validated(
        nodes=nodes,
        row_indices=rows,
        column_indices=columns,
        weights=weights,
        edge_types=edge_types,
        diagnostics=diagnostics,
    )


__all__ = ["build_evidence_graph"]
