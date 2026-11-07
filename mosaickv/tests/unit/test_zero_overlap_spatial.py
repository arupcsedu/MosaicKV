from unittest.mock import patch

import numpy as np
import pytest
import torch

from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import GraphConfig
from mosaickv.graph import build_evidence_graph, builder
from mosaickv.graph.pooling import pool_block_descriptors
from mosaickv.graph.types import BlockEvidenceMetadata


@pytest.mark.parametrize("seed", range(12))
def test_exact_spatial_edges_with_irregular_boxes_filters_and_ties(seed):
    rng = np.random.default_rng(seed)
    pair = tuple(torch.randn(1, 2, 32, 8) for _ in range(2))
    full = FullKVState.from_tensors(
        (pair, pair),
        block_size=1,
        modality_spans=(
            ModalitySpan(0, 16, Modality.TEXT),
            ModalitySpan(16, 32, Modality.IMAGE, image_index=0),
        ),
    )
    metadata = {}
    for i in range(len(full.blocks)):
        x, y = sorted(rng.integers(0, 5, 2) / 4), sorted(rng.integers(0, 5, 2) / 4)
        metadata[i] = BlockEvidenceMetadata(
            normalized_box=(x[0], y[0], x[1], y[1]), image_index=i % 2, page_index=(i // 2) % 2
        )
    nodes = pool_block_descriptors(full, metadata_by_node=metadata)
    cfg = GraphConfig(
        max_neighbors=1 + seed % 7,
        require_same_layer=seed % 2 == 0,
        require_same_kv_head=seed % 3 == 0,
        allowed_modality_pairs=("text:text", "image:image", "text:image"),
    )
    results = []
    for mode in ("reference", "zero_overlap"):
        a = builder._EdgeAccumulator(cfg)
        with builder.spatial_search(mode):
            builder._add_spatial_edges(
                nodes,
                cfg,
                a,
                eligible_node_ids=frozenset(range(0, len(nodes), 2)) if seed % 2 else None,
            )
        results.append(a.finalize())
    assert results[0] == results[1]


def test_full_603_graph_and_scalar_distance_reduction_are_exact():
    generator = torch.Generator().manual_seed(603)
    pair = tuple(torch.randn(1, 1, 603, 128, generator=generator).bfloat16() for _ in range(2))
    state = FullKVState.from_tensors(
        (pair,),
        block_size=1,
        modality_spans=(
            ModalitySpan(0, 5, Modality.TEXT),
            ModalitySpan(5, 581, Modality.IMAGE, image_index=0),
            ModalitySpan(581, 603, Modality.TEXT),
        ),
    )
    cfg = GraphConfig(
        max_neighbors=4, similarity_chunk_size=128, edge_accumulation_mode="segmented_tensor_topk"
    )
    graphs, counts = [], []
    metadata = {
        5 + r * 24 + c: BlockEvidenceMetadata(
            image_index=0, normalized_box=(c / 24, r / 24, (c + 1) / 24, (r + 1) / 24)
        )
        for r in range(24)
        for c in range(24)
    }
    for mode in ("reference", "zero_overlap"):
        with (
            builder.spatial_search(mode),
            patch.object(
                builder, "_rectangle_distance", wraps=builder._rectangle_distance
            ) as distance,
        ):
            graphs.append(build_evidence_graph(state, cfg, metadata_by_node=metadata))
            counts.append(distance.call_count)
    for field in ("row_indices", "column_indices", "weights", "edge_types", "diagnostics"):
        assert getattr(graphs[0], field) == getattr(graphs[1], field)
    assert counts[0] > 50000 and counts[1] < counts[0] // 10
    assert builder._spatial_search.get() == "reference"
    with pytest.raises(ValueError), builder.spatial_search("approximate"):
        pass
