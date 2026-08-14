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


def test_disabled_graph_preserves_nodes_and_has_no_edges() -> None:
    full = _full_state((Modality.TEXT, Modality.IMAGE, Modality.TEXT))

    graph = build_evidence_graph(full, GraphConfig(enabled=False))

    assert len(graph.nodes) == len(full.blocks)
    assert graph.row_indices == ()
    assert graph.diagnostics.edge_count == 0
    assert graph.diagnostics.connected_components == len(graph.nodes)


def test_compact_graph_uses_dense_local_ids_and_preserves_source_ids() -> None:
    full = _full_state((Modality.TEXT,) * 6)

    graph = build_evidence_graph(
        full,
        _isolated_config(fallback_weight=0.5, max_neighbors=2),
        source_node_ids=(1, 3, 5),
        similarity_candidate_node_ids=(1, 5),
    )

    assert tuple(node.node_id for node in graph.nodes) == (0, 1, 2)
    assert tuple(node.source_id for node in graph.nodes) == (1, 3, 5)
    assert tuple(node.block for node in graph.nodes) == (
        full.blocks[1],
        full.blocks[3],
        full.blocks[5],
    )
    assert all(0 <= node_id < 3 for node_id in (*graph.row_indices, *graph.column_indices))


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


def test_vectorized_pooling_preserves_batch_head_and_partial_block_means() -> None:
    key = np.arange(2 * 3 * 5 * 4, dtype=np.float32).reshape(2, 3, 5, 4)
    value = (key + 100).copy()
    full = FullKVState.from_tensors(((key, value),), block_size=3)

    nodes = pool_block_descriptors(full)

    assert len(nodes) == 6
    for node in nodes:
        positions = node.block.physical_cache_indices
        head = node.block.kv_head
        assert np.array_equal(
            node.pooled_key,
            key[:, head, positions, :].reshape(-1, key.shape[-1]).mean(axis=0),
        )
        assert np.array_equal(
            node.pooled_value,
            value[:, head, positions, :].reshape(-1, value.shape[-1]).mean(axis=0),
        )


def test_fused_torch_kv_descriptor_transfer_is_bitwise_reference_equivalent() -> None:
    torch = pytest.importorskip("torch")
    key = torch.arange(2 * 3 * 5 * 4, dtype=torch.float32).reshape(2, 3, 5, 4)
    value = torch.arange(2 * 3 * 5 * 6, dtype=torch.float32).reshape(2, 3, 5, 6)
    full = FullKVState.from_tensors(((key, value),), block_size=3)

    fused = pool_block_descriptors(full, host_transfer_mode="deferred_layer")
    reference = pool_block_descriptors(full, host_transfer_mode="per_group_reference")

    assert len(fused) == len(reference)
    for observed, expected in zip(fused, reference, strict=True):
        assert np.array_equal(observed.pooled_key, expected.pooled_key)
        assert np.array_equal(observed.pooled_value, expected.pooled_value)
        assert np.array_equal(observed.semantic_embedding, expected.semantic_embedding)


def test_vectorized_torch_descriptor_normalization_matches_deferred_values() -> None:
    torch = pytest.importorskip("torch")
    key = torch.arange(2 * 3 * 5 * 4, dtype=torch.float32).reshape(2, 3, 5, 4)
    value = torch.arange(2 * 3 * 5 * 6, dtype=torch.float32).reshape(2, 3, 5, 6)
    full = FullKVState.from_tensors(((key, value),), block_size=3)

    candidate = pool_block_descriptors(
        full,
        host_transfer_mode="deferred_layer_vectorized",
    )
    reference = pool_block_descriptors(full, host_transfer_mode="deferred_layer")

    assert len(candidate) == len(reference)
    for observed, expected in zip(candidate, reference, strict=True):
        assert np.array_equal(observed.pooled_key, expected.pooled_key)
        assert np.array_equal(observed.pooled_value, expected.pooled_value)
        assert np.array_equal(observed.semantic_embedding, expected.semantic_embedding)


def test_optimized_graph_pipeline_is_bitwise_reference_equivalent() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(541)
    key = torch.randn((1, 2, 9, 7), generator=generator, dtype=torch.float32)
    value = torch.randn((1, 2, 9, 5), generator=generator, dtype=torch.float32)
    attention = torch.rand((1, 4, 9, 9), generator=generator, dtype=torch.float32)
    full = FullKVState.from_tensors(((key, value),), block_size=2)
    reference_config = GraphConfig(
        attention_host_transfer_mode="deferred_layer",
        edge_accumulation_mode="dictionary",
        similarity_chunk_size=3,
    )
    optimized_config = replace(
        reference_config,
        attention_host_transfer_mode="deferred_layer_vectorized",
        edge_accumulation_mode="preselected",
    )

    reference = build_evidence_graph(
        full,
        reference_config,
        prompt_attentions=(attention,),
    )
    optimized = build_evidence_graph(
        full,
        optimized_config,
        prompt_attentions=(attention,),
    )

    assert optimized.row_indices == reference.row_indices
    assert optimized.column_indices == reference.column_indices
    assert optimized.edge_types == reference.edge_types
    assert optimized.weights == reference.weights
    assert optimized.diagnostics == reference.diagnostics
    for observed, expected in zip(optimized.nodes, reference.nodes, strict=True):
        assert np.array_equal(observed.pooled_key, expected.pooled_key)
        assert np.array_equal(observed.pooled_value, expected.pooled_value)
        assert np.array_equal(observed.semantic_embedding, expected.semantic_embedding)


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


def test_processor_grid_metadata_reaches_spatial_graph_without_override() -> None:
    key = np.arange(1 * 1 * 8 * 3, dtype=np.float32).reshape(1, 1, 8, 3)
    full = FullKVState.from_tensors(
        ((key, key + 1),),
        modality_spans=(ModalitySpan(0, 8, Modality.IMAGE, image_index=0, grid_shape=(2, 4)),),
        block_size=2,
    )

    graph = build_evidence_graph(
        full,
        _isolated_config(spatial_weight=1.0, spatial_max_distance=0.3, max_neighbors=2),
    )

    assert _edge_pairs(graph, EdgeType.SPATIAL_ADJACENCY)
    assert not graph.diagnostics.fallback_used
    assert all(node.evidence.normalized_box is not None for node in graph.nodes)


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


def test_similarity_shortlist_preserves_all_nodes_and_fallback_edges() -> None:
    full = _full_state((Modality.TEXT,) * 6)
    attention = np.arange(1 * 1 * 2 * 6, dtype=np.float32).reshape(1, 1, 2, 6) + 1
    candidates = (1, 2, 4)

    pooled = pool_prompt_attention_coactivation(
        full,
        (attention,),
        eligible_node_ids=candidates,
    )
    graph = build_evidence_graph(
        full,
        _isolated_config(
            semantic_weight=1.0,
            attention_weight=1.0,
            fallback_weight=0.25,
            max_neighbors=2,
        ),
        prompt_attentions=(attention,),
        similarity_candidate_node_ids=candidates,
    )

    assert set(pooled) == set(candidates)
    assert len(graph.nodes) == len(full.blocks)
    similarity_edges = _edge_pairs(graph, EdgeType.SEMANTIC_SIMILARITY) | _edge_pairs(
        graph,
        EdgeType.ATTENTION_COACTIVATION,
    )
    assert similarity_edges
    assert all(source in candidates and target in candidates for source, target in similarity_edges)
    assert {source for source, _target in _edge_pairs(graph, EdgeType.FALLBACK_POSITIONAL)} == set(
        range(len(full.blocks))
    )


def test_candidate_descriptor_pooling_keeps_lightweight_structural_nodes() -> None:
    full = _full_state((Modality.TEXT,) * 6)
    candidates = (1, 2, 4)
    graph = build_evidence_graph(
        full,
        _isolated_config(
            semantic_weight=1.0,
            fallback_weight=0.25,
            max_neighbors=2,
            similarity_shortlist_multiplier=2.0,
            candidate_descriptor_pooling=True,
        ),
        similarity_candidate_node_ids=candidates,
    )

    assert len(graph.nodes) == len(full.blocks)
    assert all(
        np.any(graph.nodes[node_id].semantic_embedding)
        for node_id in candidates
    )
    assert all(
        not np.any(graph.nodes[node_id].semantic_embedding)
        for node_id in set(range(len(full.blocks))) - set(candidates)
    )
    assert {source for source, _target in _edge_pairs(graph, EdgeType.FALLBACK_POSITIONAL)} == set(
        range(len(full.blocks))
    )


def test_candidate_graph_mode_materializes_only_shortlist_edges() -> None:
    full = _full_state((Modality.TEXT,) * 8)
    candidates = (2, 3, 5)
    graph = build_evidence_graph(
        full,
        _isolated_config(
            semantic_weight=1.0,
            fallback_weight=0.25,
            max_neighbors=2,
            similarity_shortlist_multiplier=2.0,
            candidate_descriptor_pooling=True,
            candidate_graph_mode="shortlist_rerank",
        ),
        similarity_candidate_node_ids=candidates,
    )

    assert len(graph.nodes) == len(full.blocks)
    assert graph.diagnostics.candidate_graph_mode == "shortlist_rerank"
    assert graph.diagnostics.candidate_node_count == len(candidates)
    assert graph.diagnostics.candidate_fraction == pytest.approx(len(candidates) / 8)
    assert graph.diagnostics.fallback_used is False
    assert not _edge_pairs(graph, EdgeType.FALLBACK_POSITIONAL)
    assert graph.row_indices
    assert all(
        source in candidates and target in candidates
        for source, target in zip(graph.row_indices, graph.column_indices, strict=True)
    )


def test_similarity_shortlist_rejects_unknown_nodes() -> None:
    full = _full_state((Modality.TEXT,) * 2)
    with pytest.raises(ValueError, match="unknown graph nodes"):
        build_evidence_graph(
            full,
            GraphConfig(),
            similarity_candidate_node_ids=(99,),
        )


def test_batched_torch_prompt_attention_pool_matches_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    full = _full_state((Modality.TEXT,) * 5, heads=2, block_size=2)
    torch_full = FullKVState.from_tensors(
        tuple(
            (
                torch.as_tensor(layer.key),
                torch.as_tensor(layer.value),
            )
            for layer in full.layers
        ),
        modality_spans=full.modality_spans,
        block_size=2,
    )
    attention = np.arange(1 * 4 * 3 * 5, dtype=np.float32).reshape(1, 4, 3, 5)

    expected = pool_prompt_attention_coactivation(full, (attention,))
    tensor_attention = torch.as_tensor(attention)
    observed = pool_prompt_attention_coactivation(torch_full, (tensor_attention,))
    per_group = pool_prompt_attention_coactivation(
        torch_full,
        (tensor_attention,),
        host_transfer_mode="per_group_reference",
    )

    assert set(observed) == set(expected)
    for node_id in expected:
        assert np.allclose(observed[node_id], expected[node_id], atol=1e-6, rtol=0)
        assert np.array_equal(observed[node_id], per_group[node_id])


def test_prompt_attention_pool_rejects_unknown_host_transfer_mode() -> None:
    full = _full_state((Modality.TEXT,) * 2)
    attention = np.ones((1, 1, 1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="host_transfer_mode"):
        pool_prompt_attention_coactivation(
            full,
            (attention,),
            host_transfer_mode="unknown",
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


def test_positional_fallback_remains_available_with_structural_metadata() -> None:
    full = _full_state((Modality.IMAGE,) * 4)
    metadata = {
        node_id: BlockEvidenceMetadata(
            normalized_box=(node_id * 0.2, 0.0, node_id * 0.2 + 0.1, 0.1)
        )
        for node_id in range(4)
    }

    graph = build_evidence_graph(
        full,
        _isolated_config(
            fallback_weight=0.25,
            fallback_max_position_span=2,
            max_neighbors=2,
        ),
        metadata_by_node=metadata,
    )

    assert graph.diagnostics.fallback_used
    assert _edge_pairs(graph, EdgeType.FALLBACK_POSITIONAL)


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
