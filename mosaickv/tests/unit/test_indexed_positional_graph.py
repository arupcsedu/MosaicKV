import numpy as np
import pytest

from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import GraphConfig
from mosaickv.graph import build_evidence_graph
from mosaickv.graph.builder import _nearest_center_indices, _positional_search, positional_search


@pytest.mark.parametrize("seed", range(20))
def test_indexed_neighbors_exactly_match_all_pair_reference_with_duplicate_center_ties(seed):
    rng = np.random.default_rng(seed)
    for case in range(10):
        centers = tuple(sorted(v / 2 for v in rng.integers(-100, 101, size=rng.integers(1, 91))))
        count = int(rng.integers(1, 17))
        span = None if case % 3 == 0 else int(rng.integers(0, 251))
        for i, center in enumerate(centers):
            expected = tuple(
                j
                for _, j in sorted(
                    (abs(center - c), j)
                    for j, c in enumerate(centers)
                    if j != i and (span is None or abs(center - c) <= span)
                )[:count]
            )
            assert _nearest_center_indices(centers, i, count, span) == expected


def test_dense_603_node_graph_preserves_every_edge_type_score_order_and_descriptor():
    import torch

    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(603)
    n = 603
    pair = tuple(torch.randn(1, 1, n, 128, generator=generator).bfloat16() for _ in range(2))
    state = FullKVState.from_tensors(
        (pair,),
        block_size=1,
        modality_spans=(
            ModalitySpan(0, 5, Modality.TEXT),
            ModalitySpan(5, 581, Modality.IMAGE, image_index=0),
            ModalitySpan(581, n, Modality.TEXT),
        ),
    )
    attention = torch.rand(1, 1, 16, n, generator=generator).bfloat16()
    cfg = GraphConfig(
        max_neighbors=4, similarity_chunk_size=128, edge_accumulation_mode="segmented_tensor_topk"
    )
    with positional_search("reference"):
        a = build_evidence_graph(
            state, cfg, prompt_attentions=(attention,), prompt_attention_query_window=16
        )
    with positional_search("indexed"):
        b = build_evidence_graph(
            state, cfg, prompt_attentions=(attention,), prompt_attention_query_window=16
        )
    for field in ("row_indices", "column_indices", "weights", "edge_types", "diagnostics"):
        assert getattr(a, field) == getattr(b, field)
    for left, right in zip(a.nodes, b.nodes, strict=True):
        assert np.array_equal(left.semantic_embedding, right.semantic_embedding)


def test_positional_execution_context_restores_on_exception():
    assert _positional_search.get() == "reference"
    with pytest.raises(RuntimeError), positional_search("indexed"):
        assert _positional_search.get() == "indexed"
        raise RuntimeError("stop")
    assert _positional_search.get() == "reference"
    with pytest.raises(ValueError), positional_search("unregistered"):
        pass
