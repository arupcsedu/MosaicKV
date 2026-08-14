"""Value-aware, fully decomposed local utility computation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np

from mosaickv.config import UtilityConfig
from mosaickv.graph import SparseEvidenceGraph
from mosaickv.selection.types import BlockUtility, BlockUtilityTable

HeadId = tuple[int, int]


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if type(value).__module__.startswith("torch"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        raise ValueError("forecast attention must contain only finite values")
    return cast("np.ndarray[Any, Any]", result)


def _head_groups(graph: SparseEvidenceGraph) -> dict[HeadId, list[int]]:
    result: dict[HeadId, list[int]] = defaultdict(list)
    for node in graph.nodes:
        result[(node.block.layer, node.block.kv_head)].append(node.node_id)
    return dict(result)


def _normalize_head_probabilities(
    graph: SparseEvidenceGraph, probabilities: Mapping[int, float]
) -> tuple[float, ...]:
    expected = set(range(len(graph.nodes)))
    if set(probabilities) != expected:
        missing = sorted(expected - set(probabilities))
        extra = sorted(set(probabilities) - expected)
        raise ValueError(
            "forecast attention must cover every graph node exactly once; "
            f"missing={missing}, extra={extra}"
        )
    result = [0.0] * len(graph.nodes)
    for head, node_ids in _head_groups(graph).items():
        values = [float(probabilities[node_id]) for node_id in node_ids]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"forecast attention for head {head} must be finite and nonnegative")
        total = sum(values)
        if total <= 0:
            raise ValueError(f"forecast attention for head {head} has zero total probability")
        for node_id, value in zip(node_ids, values, strict=True):
            result[node_id] = value / total
    return tuple(result)


def _block_attention_from_heads(
    graph: SparseEvidenceGraph,
    attention_by_head: Mapping[HeadId, Any],
) -> tuple[float, ...]:
    groups = _head_groups(graph)
    if set(attention_by_head) != set(groups):
        missing = sorted(set(groups) - set(attention_by_head))
        extra = sorted(set(attention_by_head) - set(groups))
        raise ValueError(
            "forecast head attention must cover every graph layer/KV-head; "
            f"missing={missing}, extra={extra}"
        )
    block_values: dict[int, float] = {}
    for head, node_ids in groups.items():
        attention = _to_numpy(attention_by_head[head]).reshape(-1)
        if bool(np.any(attention < 0)):
            raise ValueError(f"forecast attention for head {head} must be nonnegative")
        required_length = (
            max(
                position
                for node_id in node_ids
                for position in graph.nodes[node_id].block.physical_cache_indices
            )
            + 1
        )
        if attention.size < required_length:
            raise ValueError(f"forecast attention for head {head} is shorter than its cache")
        for node_id in node_ids:
            positions = graph.nodes[node_id].block.physical_cache_indices
            block_values[node_id] = float(attention[list(positions)].sum())
    return _normalize_head_probabilities(graph, block_values)


def _node_attention(
    graph: SparseEvidenceGraph,
    *,
    forecast_attention_by_node: Mapping[int, float] | Any | None,
    forecast_attention_by_head: Mapping[HeadId, Any] | None,
) -> tuple[float, ...]:
    if (forecast_attention_by_node is None) == (forecast_attention_by_head is None):
        raise ValueError(
            "provide exactly one of forecast_attention_by_node or forecast_attention_by_head"
        )
    if forecast_attention_by_head is not None:
        return _block_attention_from_heads(graph, forecast_attention_by_head)
    if isinstance(forecast_attention_by_node, Mapping):
        mapped = {
            int(node_id): float(value) for node_id, value in forecast_attention_by_node.items()
        }
    else:
        values = _to_numpy(forecast_attention_by_node).reshape(-1)
        if len(values) != len(graph.nodes):
            raise ValueError("per-node forecast attention length does not match graph nodes")
        mapped = {node_id: float(value) for node_id, value in enumerate(values)}
    return _normalize_head_probabilities(graph, mapped)


def _neighbor_sets(graph: SparseEvidenceGraph) -> tuple[frozenset[int], ...]:
    neighbors: list[set[int]] = [set() for _node in graph.nodes]
    for source, target in zip(graph.row_indices, graph.column_indices, strict=True):
        neighbors[source].add(target)
        neighbors[target].add(source)
    return tuple(frozenset(values) for values in neighbors)


def _value_novelty_reference(
    graph: SparseEvidenceGraph,
    *,
    eligible_node_ids: Iterable[int] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Compute the reference novelty while caching each vector norm exactly once."""

    eligible = _validated_eligible_nodes(graph, eligible_node_ids)
    values = {
        node_id: np.asarray(graph.nodes[node_id].pooled_value, dtype=np.float64).reshape(-1)
        for node_id in eligible
    }
    norms_list = [0.0] * len(graph.nodes)
    for node_id, vector in values.items():
        norms_list[node_id] = float(np.linalg.norm(vector))
    norms = tuple(norms_list)
    neighbors = _neighbor_sets(graph)
    similarities: dict[tuple[int, int], float] = {}
    novelty: list[float] = []
    redundancy: list[float] = []
    for node_id in range(len(graph.nodes)):
        if node_id not in eligible:
            novelty.append(0.0)
            redundancy.append(1.0)
            continue
        vector = values[node_id]
        maximum_similarity = 0.0
        for neighbor_id in neighbors[node_id]:
            if neighbor_id not in eligible:
                continue
            neighbor = values[neighbor_id]
            if neighbor.size != vector.size:
                continue
            pair = (node_id, neighbor_id) if node_id < neighbor_id else (neighbor_id, node_id)
            similarity = similarities.get(pair)
            if similarity is None:
                first, second = pair
                denominator = norms[first] * norms[second]
                similarity = (
                    float(np.dot(values[first], values[second]) / denominator)
                    if denominator
                    else 0.0
                )
                similarities[pair] = similarity
            maximum_similarity = max(maximum_similarity, min(1.0, similarity))
        # Candidate-first graph construction represents non-candidate nodes
        # with explicit zero descriptors. They remain eligible through prompt
        # attention, but must not receive an artificial maximum novelty reward
        # merely because their expensive K/V descriptor was not pooled.
        if norms[node_id] == 0:
            novelty.append(0.0)
            redundancy.append(1.0)
        else:
            redundancy.append(maximum_similarity)
            novelty.append(1.0 - maximum_similarity)
    return tuple(novelty), tuple(redundancy), norms


def _value_novelty_vectorized(
    graph: SparseEvidenceGraph,
    *,
    edge_chunk_size: int = 16_384,
    eligible_node_ids: Iterable[int] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Compute the same neighbor novelty with bounded vectorized edge tiles.

    A normal VLM graph contains one common value width but hundreds of
    thousands of typed edges. The scalar reference dispatches a small
    ``np.dot`` for every unique neighbor pair. This path evaluates bounded
    edge tiles and uses ``np.maximum.at`` to preserve the undirected maximum
    similarity reduction without materializing an ``n x n`` matrix.
    """

    if edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be positive")
    eligible = _validated_eligible_nodes(graph, eligible_node_ids)
    if not eligible:
        zeros = (0.0,) * len(graph.nodes)
        return zeros, (1.0,) * len(graph.nodes), zeros
    ordered_eligible = tuple(sorted(eligible))
    values = [
        np.asarray(graph.nodes[node_id].pooled_value, dtype=np.float64).reshape(-1)
        for node_id in ordered_eligible
    ]
    feature_sizes = {int(value.size) for value in values}
    if len(feature_sizes) != 1:
        return _value_novelty_reference(graph, eligible_node_ids=eligible)
    matrix = np.stack(values, axis=0)
    eligible_norms = np.linalg.norm(matrix, axis=1)
    maximum_similarity: np.ndarray[Any, Any] = np.zeros(
        len(ordered_eligible), dtype=np.float64
    )
    rows = np.asarray(graph.row_indices, dtype=np.int64)
    columns = np.asarray(graph.column_indices, dtype=np.int64)
    local_by_global: np.ndarray[Any, Any] = np.full(
        len(graph.nodes), -1, dtype=np.int64
    )
    local_by_global[np.asarray(ordered_eligible, dtype=np.int64)] = np.arange(
        len(ordered_eligible), dtype=np.int64
    )
    local_rows = local_by_global[rows]
    local_columns = local_by_global[columns]
    admitted_edges = (local_rows >= 0) & (local_columns >= 0)
    rows = local_rows[admitted_edges]
    columns = local_columns[admitted_edges]
    for start in range(0, len(rows), edge_chunk_size):
        source = rows[start : start + edge_chunk_size]
        target = columns[start : start + edge_chunk_size]
        denominator = eligible_norms[source] * eligible_norms[target]
        dot_products = np.einsum(
            "ij,ij->i",
            matrix[source],
            matrix[target],
            optimize=False,
        )
        similarities = np.divide(
            dot_products,
            denominator,
            out=np.zeros_like(dot_products),
            where=denominator != 0,
        )
        similarities = np.minimum(1.0, similarities)
        np.maximum.at(maximum_similarity, source, similarities)
        np.maximum.at(maximum_similarity, target, similarities)
    zero_norm = eligible_norms == 0
    eligible_novelty = np.where(zero_norm, 0.0, 1.0 - maximum_similarity)
    maximum_similarity[zero_norm] = 1.0
    novelty: np.ndarray[Any, Any] = np.zeros(len(graph.nodes), dtype=np.float64)
    redundancy: np.ndarray[Any, Any] = np.ones(len(graph.nodes), dtype=np.float64)
    norms_array: np.ndarray[Any, Any] = np.zeros(len(graph.nodes), dtype=np.float64)
    eligible_array = np.asarray(ordered_eligible, dtype=np.int64)
    novelty[eligible_array] = eligible_novelty
    redundancy[eligible_array] = maximum_similarity
    norms_array[eligible_array] = eligible_norms
    return (
        tuple(float(value) for value in novelty),
        tuple(float(value) for value in redundancy),
        tuple(float(value) for value in norms_array),
    )


def _validated_eligible_nodes(
    graph: SparseEvidenceGraph,
    eligible_node_ids: Iterable[int] | None,
) -> frozenset[int]:
    if eligible_node_ids is None:
        return frozenset(range(len(graph.nodes)))
    eligible = frozenset(int(node_id) for node_id in eligible_node_ids)
    unknown = eligible - set(range(len(graph.nodes)))
    if unknown:
        raise ValueError(f"value-aware utility contains unknown node IDs: {sorted(unknown)}")
    return eligible


def _centrality_and_singleton_coverage(
    graph: SparseEvidenceGraph,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    node_count = len(graph.nodes)
    rows = np.asarray(graph.row_indices, dtype=np.int64)
    columns = np.asarray(graph.column_indices, dtype=np.int64)
    weights = np.asarray(graph.weights, dtype=np.float64)
    degree: np.ndarray[Any, Any] = np.zeros(node_count, dtype=np.float64)
    np.add.at(degree, rows, weights)
    np.add.at(degree, columns, weights)
    maximum_degree = float(degree.max(initial=0.0))
    centrality = degree / maximum_degree if maximum_degree else degree

    # Parallel typed edges share a facility pair; only their maximum weight
    # contributes, exactly matching the sparse dictionary reference path.
    encoded_pairs = rows * node_count + columns
    unique_pairs, inverse = np.unique(encoded_pairs, return_inverse=True)
    pair_weights: np.ndarray[Any, Any] = np.zeros(
        len(unique_pairs), dtype=np.float64
    )
    np.maximum.at(pair_weights, inverse, weights)
    pair_sources = unique_pairs // node_count
    pair_targets = unique_pairs % node_count
    nonself = pair_sources != pair_targets
    facility_sums: np.ndarray[Any, Any] = np.zeros(node_count, dtype=np.float64)
    np.add.at(facility_sums, pair_sources[nonself], pair_weights[nonself])
    coverage = (1.0 + facility_sums) / node_count
    return tuple(float(value) for value in centrality), tuple(float(value) for value in coverage)


def _modality_rarity(graph: SparseEvidenceGraph) -> tuple[float, ...]:
    counts: dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        counts[node.block.modality.value] += 1
    node_count = len(graph.nodes)
    if node_count == 1:
        return (0.0,)
    return tuple(
        (node_count - counts[node.block.modality.value]) / (node_count - 1) for node in graph.nodes
    )


def _normalize_head_signal(
    graph: SparseEvidenceGraph,
    values: tuple[float, ...],
) -> tuple[float, ...]:
    """Put one nonnegative signal on the same per-head L1 scale as attention.

    The utility equation combines forecast probabilities, value contribution,
    and uniqueness with dimensionless coefficients.  Comparing a probability
    distribution with an unnormalized value norm or cosine distance makes the
    coefficients sequence-length dependent.  Normalize each layer/KV-head
    independently so every nonzero signal sums to one in the same stratum.
    """

    if len(values) != len(graph.nodes):
        raise ValueError("utility signal length does not match graph nodes")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("utility signals must be finite and nonnegative")
    normalized = [0.0] * len(values)
    for node_ids in _head_groups(graph).values():
        total = sum(values[node_id] for node_id in node_ids)
        if total <= 0:
            continue
        for node_id in node_ids:
            normalized[node_id] = values[node_id] / total
    return tuple(normalized)


def compute_block_utilities(
    graph: SparseEvidenceGraph,
    weights: UtilityConfig,
    *,
    forecast_attention_by_node: Mapping[int, float] | Any | None = None,
    forecast_attention_by_head: Mapping[HeadId, Any] | None = None,
    attention_provenance: str,
    rope_aware: bool,
    value_aware_node_ids: Iterable[int] | None = None,
) -> BlockUtilityTable:
    """Compute every requested block signal and the exact signed local utility.

    The caller must explicitly attest that supplied attention is RoPE-aware.
    MosaicKV's current forecast centroids are captured before RoPE while cached
    keys are after RoPE, so this function deliberately does not dot those
    incompatible representations together.
    """

    if not rope_aware:
        raise ValueError("block utility requires explicitly RoPE-aware forecast attention")
    if not attention_provenance.strip():
        raise ValueError("attention_provenance must be non-empty")
    probabilities = _node_attention(
        graph,
        forecast_attention_by_node=forecast_attention_by_node,
        forecast_attention_by_head=forecast_attention_by_head,
    )
    if weights.execution_mode == "reference_sparse":
        raw_novelty, _raw_redundancy, value_norms = _value_novelty_reference(
            graph,
            eligible_node_ids=value_aware_node_ids,
        )
    elif weights.execution_mode == "vectorized_edges":
        raw_novelty, _raw_redundancy, value_norms = _value_novelty_vectorized(
            graph,
            eligible_node_ids=value_aware_node_ids,
        )
    else:  # pragma: no cover - strict config validation guards this
        raise ValueError(f"unsupported utility execution mode: {weights.execution_mode}")
    centrality, coverage = _centrality_and_singleton_coverage(graph)
    rarity = _modality_rarity(graph)
    raw_contribution = tuple(
        probabilities[node.node_id] * value_norms[node.node_id]
        for node in graph.nodes
    )
    if weights.normalization != "layer_head_l1":  # pragma: no cover - strict config guards this
        raise ValueError(f"unsupported utility normalization: {weights.normalization}")
    contribution = _normalize_head_signal(graph, raw_contribution)
    novelty = _normalize_head_signal(graph, raw_novelty)
    redundancy = tuple(1.0 - value for value in novelty)
    utilities: list[BlockUtility] = []
    for node in graph.nodes:
        node_id = node.node_id
        local = (
            weights.lambda_q * probabilities[node_id]
            - weights.lambda_v * contribution[node_id]
            - weights.lambda_o * novelty[node_id]
        )
        utilities.append(
            BlockUtility._from_validated(
                node_id=node_id,
                forecast_attention_probability=probabilities[node_id],
                value_novelty=novelty[node_id],
                expected_attention_output_contribution=contribution[node_id],
                graph_centrality=centrality[node_id],
                singleton_coverage_gain=coverage[node_id],
                modality_rarity=rarity[node_id],
                redundancy_penalty=redundancy[node_id],
                mandatory_priority=1.0 if node.block.mandatory else 0.0,
                local_utility=local,
            )
        )
    return BlockUtilityTable._from_validated(
        blocks=tuple(utilities),
        weights=weights,
        attention_provenance=attention_provenance,
    )


__all__ = ["HeadId", "compute_block_utilities"]
