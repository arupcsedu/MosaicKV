"""Value-aware, fully decomposed local utility computation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

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
    return result


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


def _value_novelty(graph: SparseEvidenceGraph) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = [np.asarray(node.pooled_value, dtype=np.float64).reshape(-1) for node in graph.nodes]
    neighbors = _neighbor_sets(graph)
    novelty: list[float] = []
    redundancy: list[float] = []
    for node_id, vector in enumerate(values):
        vector_norm = float(np.linalg.norm(vector))
        maximum_similarity = 0.0
        for neighbor_id in neighbors[node_id]:
            neighbor = values[neighbor_id]
            if neighbor.size != vector.size:
                continue
            denominator = vector_norm * float(np.linalg.norm(neighbor))
            similarity = float(np.dot(vector, neighbor) / denominator) if denominator else 0.0
            maximum_similarity = max(maximum_similarity, min(1.0, similarity))
        redundancy.append(maximum_similarity)
        novelty.append(1.0 - maximum_similarity)
    return tuple(novelty), tuple(redundancy)


def _centrality_and_singleton_coverage(
    graph: SparseEvidenceGraph,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    degree = [0.0] * len(graph.nodes)
    facility: list[dict[int, float]] = [dict() for _node in graph.nodes]
    for source, target, weight in zip(
        graph.row_indices, graph.column_indices, graph.weights, strict=True
    ):
        degree[source] += weight
        degree[target] += weight
        facility[source][target] = max(weight, facility[source].get(target, 0.0))
    maximum_degree = max(degree, default=0.0)
    centrality = tuple(value / maximum_degree if maximum_degree else 0.0 for value in degree)
    node_count = len(graph.nodes)
    coverage = tuple(
        (1.0 + sum(weight for target, weight in targets.items() if target != node_id)) / node_count
        for node_id, targets in enumerate(facility)
    )
    return centrality, coverage


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


def compute_block_utilities(
    graph: SparseEvidenceGraph,
    weights: UtilityConfig,
    *,
    forecast_attention_by_node: Mapping[int, float] | Any | None = None,
    forecast_attention_by_head: Mapping[HeadId, Any] | None = None,
    attention_provenance: str,
    rope_aware: bool,
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
    novelty, redundancy = _value_novelty(graph)
    centrality, coverage = _centrality_and_singleton_coverage(graph)
    rarity = _modality_rarity(graph)
    utilities: list[BlockUtility] = []
    for node in graph.nodes:
        node_id = node.node_id
        value_norm = float(np.linalg.norm(np.asarray(node.pooled_value, dtype=np.float64)))
        contribution = probabilities[node_id] * value_norm
        local = (
            weights.lambda_q * probabilities[node_id]
            - weights.lambda_v * contribution
            - weights.lambda_o * novelty[node_id]
        )
        utilities.append(
            BlockUtility(
                node_id=node_id,
                forecast_attention_probability=probabilities[node_id],
                value_novelty=novelty[node_id],
                expected_attention_output_contribution=contribution,
                graph_centrality=centrality[node_id],
                singleton_coverage_gain=coverage[node_id],
                modality_rarity=rarity[node_id],
                redundancy_penalty=redundancy[node_id],
                mandatory_priority=1.0 if node.block.mandatory else 0.0,
                local_utility=local,
            )
        )
    return BlockUtilityTable(tuple(utilities), weights, attention_provenance)


__all__ = ["HeadId", "compute_block_utilities"]
