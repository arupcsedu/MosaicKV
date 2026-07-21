from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import ConfigurationError, GraphConfig
from mosaickv.graph import (
    BlockEvidenceMetadata,
    EdgeType,
    SparseEvidenceGraph,
    build_evidence_graph,
    pool_block_descriptors,
    pool_prompt_attention_coactivation,
)


def _full_state(
    modalities: tuple[Modality, ...],
    *,
    heads: int = 1,
    block_size: int = 1,
) -> FullKVState:
    sequence_length = len(modalities)
    key = np.arange(heads * sequence_length * 3, dtype=np.float32).reshape(
        1, heads, sequence_length, 3
    )
    value = (key + 1).copy()
    spans = tuple(
        ModalitySpan(
            position,
            position + 1,
            modality,
            image_index=0 if modality is Modality.IMAGE else None,
            frame_index=position if modality is Modality.VIDEO else None,
            page_index=0 if modality is Modality.IMAGE else None,
        )
        for position, modality in enumerate(modalities)
    )
    return FullKVState.from_tensors(
        ((key, value),),
        modality_spans=spans,
        block_size=block_size,
    )


def _isolated_config(**changes: object) -> GraphConfig:
    values: dict[str, object] = {
        "semantic_weight": 0.0,
        "attention_weight": 0.0,
        "spatial_weight": 0.0,
        "layout_weight": 0.0,
        "temporal_weight": 0.0,
        "same_region_weight": 0.0,
        "cross_modal_weight": 0.0,
        "fallback_weight": 0.0,
    }
    values.update(changes)
    return GraphConfig(**values)  # type: ignore[arg-type]


def _edge_pairs(graph: SparseEvidenceGraph, edge_type: EdgeType) -> set[tuple[int, int]]:
    rows = graph.row_indices
    columns = graph.column_indices
    kinds = graph.edge_types
    return {
        (source, target)
        for source, target, kind in zip(rows, columns, kinds, strict=True)
        if kind is edge_type
    }


def test_pooling_uses_recorded_cache_axes_and_optional_hidden_states() -> None:
    key = np.arange(1 * 1 * 2 * 4, dtype=np.float32).reshape(1, 1, 2, 4)
    value = (key + 10).copy()
    full = FullKVState.from_tensors(((key, value),), block_size=2)
    hidden = np.arange(1 * 4 * 3, dtype=np.float32).reshape(1, 4, 3)

    nodes = pool_block_descriptors(full, hidden_states=(hidden,))

    assert len(nodes) == 1
    assert np.array_equal(nodes[0].pooled_key, key[0, 0].mean(axis=0))
    assert np.array_equal(nodes[0].pooled_value, value[0, 0].mean(axis=0))
    assert nodes[0].pooled_hidden_state is not None
    assert np.array_equal(nodes[0].pooled_hidden_state, hidden[0, :2].mean(axis=0))
    assert np.isclose(np.linalg.norm(nodes[0].semantic_embedding), 1.0)


def test_pooling_supports_feature_before_sequence_cache_layout() -> None:
    key = np.arange(1 * 1 * 3 * 4, dtype=np.float32).reshape(1, 1, 3, 4)
    value = (key + 10).copy()
    full = FullKVState.from_tensors(
        ((key, value),),
        sequence_dimension=-1,
        head_dimension=1,
        block_size=2,
    )

    nodes = pool_block_descriptors(full)

    assert len(nodes) == 2
    assert np.array_equal(nodes[0].pooled_key, key[0, 0, :, :2].mean(axis=1))
    assert np.array_equal(nodes[1].pooled_value, value[0, 0, :, 2:].mean(axis=1))


def test_synthetic_chart_has_spatial_and_cross_modal_connections() -> None:
    full = _full_state((Modality.TEXT, Modality.IMAGE, Modality.IMAGE, Modality.TEXT))
    metadata = {
        0: BlockEvidenceMetadata(alignment_ids=("chart-a",)),
        1: BlockEvidenceMetadata(
            normalized_box=(0.0, 0.0, 0.4, 0.4),
            evidence_region="series-a",
            alignment_ids=("chart-a",),
        ),
        2: BlockEvidenceMetadata(
            normalized_box=(0.4, 0.0, 0.8, 0.4),
            evidence_region="series-b",
            alignment_ids=("chart-b",),
        ),
        3: BlockEvidenceMetadata(alignment_ids=("chart-b",)),
    }
    config = _isolated_config(
        max_neighbors=2,
        spatial_weight=0.7,
        cross_modal_weight=0.9,
    )

    graph = build_evidence_graph(full, config, metadata_by_node=metadata)

    assert (1, 2) in _edge_pairs(graph, EdgeType.SPATIAL_ADJACENCY)
    assert (2, 1) in _edge_pairs(graph, EdgeType.SPATIAL_ADJACENCY)
    assert (0, 1) in _edge_pairs(graph, EdgeType.CROSS_MODAL_ALIGNMENT)
    assert (3, 2) in _edge_pairs(graph, EdgeType.CROSS_MODAL_ALIGNMENT)
    assert graph.diagnostics.modality_mixing > 0
    assert not graph.diagnostics.fallback_used


def test_synthetic_document_layout_connects_rows_columns_regions_and_ocr() -> None:
    full = _full_state((Modality.IMAGE,) * 4)
    metadata = {
        0: BlockEvidenceMetadata(
            normalized_box=(0.0, 0.0, 0.4, 0.2),
            row_index=0,
            column_index=0,
            page_region="table",
            ocr_text="Revenue",
        ),
        1: BlockEvidenceMetadata(
            normalized_box=(0.5, 0.0, 0.9, 0.2),
            row_index=0,
            column_index=1,
            page_region="table",
            ocr_text="2026",
        ),
        2: BlockEvidenceMetadata(
            normalized_box=(0.0, 0.3, 0.4, 0.5),
            row_index=1,
            column_index=0,
            page_region="table",
            ocr_text="Revenue",
        ),
        3: BlockEvidenceMetadata(
            normalized_box=(0.5, 0.3, 0.9, 0.5),
            row_index=1,
            column_index=1,
            page_region="table",
            ocr_text="2027",
        ),
    }

    graph = build_evidence_graph(
        full,
        _isolated_config(max_neighbors=4, layout_weight=1.0),
        metadata_by_node=metadata,
    )
    layout = _edge_pairs(graph, EdgeType.OCR_LAYOUT)

    assert {(0, 1), (0, 2), (0, 3)} <= layout
    assert graph.diagnostics.connected_components == 1


def test_ocr_boxes_create_layout_edges_without_explicit_row_labels() -> None:
    full = _full_state((Modality.IMAGE, Modality.IMAGE))
    metadata = {
        0: BlockEvidenceMetadata(
            normalized_box=(0.0, 0.0, 0.4, 0.2),
            ocr_text="label",
        ),
        1: BlockEvidenceMetadata(
            normalized_box=(0.5, 0.0, 0.9, 0.2),
            ocr_text="value",
        ),
    }

    graph = build_evidence_graph(
        full,
        _isolated_config(layout_weight=1.0),
        metadata_by_node=metadata,
    )

    assert _edge_pairs(graph, EdgeType.OCR_LAYOUT) == {(0, 1), (1, 0)}


def test_temporal_edges_respect_clip_frame_window_and_head_compatibility() -> None:
    full = _full_state((Modality.VIDEO,) * 3, heads=2)
    metadata = {
        node_id: BlockEvidenceMetadata(
            frame_index=(node_id % 3) * (1 if node_id % 3 < 2 else 2),
            clip_index=0,
        )
        for node_id in range(6)
    }
    graph = build_evidence_graph(
        full,
        _isolated_config(temporal_weight=1.0, temporal_window=1, max_neighbors=3),
        metadata_by_node=metadata,
    )
    temporal = _edge_pairs(graph, EdgeType.TEMPORAL_ADJACENCY)

    assert (0, 1) in temporal
    assert (1, 2) not in temporal
    assert all(
        full.blocks[source].kv_head == full.blocks[target].kv_head for source, target in temporal
    )


def test_attention_top_d_and_position_compatibility_are_sparse() -> None:
    full = _full_state((Modality.TEXT,) * 8)
    attention = np.asarray([[1.0, float(index)] for index in range(8)], dtype=np.float32)
    config = _isolated_config(
        attention_weight=1.0,
        attention_max_position_span=2,
        max_neighbors=2,
        similarity_chunk_size=3,
    )

    graph = build_evidence_graph(full, config, attention_coactivation=attention)
    attention_pairs = _edge_pairs(graph, EdgeType.ATTENTION_COACTIVATION)

    assert len(attention_pairs) <= len(graph.nodes) * config.max_neighbors
    assert all(abs(source - target) <= 2 for source, target in attention_pairs)


def test_prompt_attention_maps_pool_query_heads_by_kv_head() -> None:
    full = _full_state((Modality.TEXT,) * 3, heads=2)
    attention = np.zeros((1, 4, 2, 3), dtype=np.float32)
    attention[0, 0:2, :, 0] = (1.0, 0.0)
    attention[0, 0:2, :, 1] = (0.9, 0.1)
    attention[0, 0:2, :, 2] = (0.0, 1.0)
    attention[0, 2:4, :, :] = 0.5

    pooled = pool_prompt_attention_coactivation(full, (attention,))
    graph = build_evidence_graph(
        full,
        _isolated_config(attention_weight=1.0, max_neighbors=1),
        prompt_attentions=(attention,),
    )

    assert np.array_equal(pooled[0], np.asarray([1.0, 0.0], dtype=np.float32))
    assert (0, 1) in _edge_pairs(graph, EdgeType.ATTENTION_COACTIVATION)
    assert all(
        full.blocks[source].kv_head == full.blocks[target].kv_head
        for source, target in _edge_pairs(graph, EdgeType.ATTENTION_COACTIVATION)
    )


def test_missing_metadata_uses_local_fallback_and_csr_is_valid() -> None:
    full = _full_state((Modality.TEXT,) * 6)
    config = _isolated_config(fallback_weight=0.5, max_neighbors=2, fallback_max_position_span=2)

    graph = build_evidence_graph(full, config)
    csr = graph.to_csr()

    assert graph.diagnostics.fallback_used
    assert _edge_pairs(graph, EdgeType.FALLBACK_POSITIONAL)
    assert len(csr.indptr) == len(graph.nodes) + 1
    assert csr.indptr[-1] == graph.diagnostics.edge_count
    assert all(
        len(graph.neighbors(node.node_id, EdgeType.FALLBACK_POSITIONAL)) <= config.max_neighbors
        for node in graph.nodes
    )


def test_evidence_cluster_diagnostics_and_edge_source_ablation_are_independent() -> None:
    full = _full_state((Modality.IMAGE,) * 3)
    metadata = {
        node_id: BlockEvidenceMetadata(
            normalized_box=(node_id * 0.2, 0.0, node_id * 0.2 + 0.2, 0.2),
            row_index=0,
            evidence_region="cluster-a",
        )
        for node_id in range(3)
    }
    config = _isolated_config(max_neighbors=2, spatial_weight=0.8, layout_weight=0.6)
    complete = build_evidence_graph(full, config, metadata_by_node=metadata)
    ablated = build_evidence_graph(
        full,
        replace(config, spatial_weight=0.0),
        metadata_by_node=metadata,
    )

    assert complete.diagnostics.evidence_cluster_coverage == 1.0
    assert _edge_pairs(complete, EdgeType.SPATIAL_ADJACENCY)
    assert not _edge_pairs(ablated, EdgeType.SPATIAL_ADJACENCY)
    assert _edge_pairs(complete, EdgeType.OCR_LAYOUT) == _edge_pairs(ablated, EdgeType.OCR_LAYOUT)


def test_semantic_graph_storage_scales_with_nodes_times_degree() -> None:
    node_count = 257
    full = _full_state((Modality.TEXT,) * node_count)
    config = _isolated_config(
        semantic_weight=1.0,
        max_neighbors=3,
        similarity_chunk_size=17,
    )

    graph = build_evidence_graph(full, config)

    assert graph.diagnostics.edge_count <= node_count * config.max_neighbors
    assert graph.diagnostics.maximum_out_degree <= config.max_neighbors


def test_graph_metadata_and_configuration_fail_closed() -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        BlockEvidenceMetadata(normalized_box=(0.0, 0.0, 2.0, 1.0))
    with pytest.raises(ConfigurationError, match="allowed_modality_pairs"):
        GraphConfig(allowed_modality_pairs=("image:audio",))
    with pytest.raises(ConfigurationError, match="similarity_chunk_size"):
        GraphConfig(similarity_chunk_size=0)
