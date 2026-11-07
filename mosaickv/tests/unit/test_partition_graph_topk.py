import numpy as np
import pytest

from mosaickv.graph.builder import _cosine_search, _positive_partition_topk, cosine_search


@pytest.mark.parametrize("seed", range(20))
def test_partition_positive_sets_match_stable_full_sort_including_cutoff_ties(seed):
    rng = np.random.default_rng(seed)
    for _ in range(12):
        n = int(rng.integers(1, 56))
        rows = int(rng.integers(1, n + 1))
        count = int(rng.integers(1, 20))
        scores = rng.integers(-2, 3, size=(rows, n)).astype(np.float32)
        diagonal = np.arange(rows)
        actual = tuple(zip(*_positive_partition_topk(scores, diagonal, count), strict=True))
        expected = []
        for r in range(rows):
            selected = sorted(
                (i for i in range(n) if i != diagonal[r] and scores[r, i] > 0),
                key=lambda i: (-scores[r, i], i),
            )[:count]
            expected += [(r, i) for i in sorted(selected)]
        assert actual == tuple(expected)


def test_tied_scores_skip_self_and_choose_lowest_other_node_ids():
    rows, cols = _positive_partition_topk(np.ones((3, 7), dtype=np.float32), np.array([0, 3, 6]), 2)
    assert list(zip(rows, cols, strict=True)) == [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)]
    with pytest.raises(ValueError):
        _positive_partition_topk(np.array([[np.nan]]), [0], 1)
    with pytest.raises(RuntimeError), cosine_search("partition"):
        assert _cosine_search.get() == "partition"
        raise RuntimeError("stop")
    assert _cosine_search.get() == "stable_sort"


def test_complete_graph_remains_bitwise_identical_with_partition_and_indexed_positionals():
    import torch

    from mosaickv.cache_state import FullKVState
    from mosaickv.config import GraphConfig
    from mosaickv.graph import build_evidence_graph
    from mosaickv.graph.builder import positional_search

    generator = torch.Generator().manual_seed(118)
    pair = tuple(torch.randn(1, 3, 37, 9, generator=generator).bfloat16() for _ in range(2))
    attention = torch.rand(1, 6, 16, 37, generator=generator).bfloat16()
    full = FullKVState.from_tensors((pair,), block_size=2)
    cfg = GraphConfig(
        max_neighbors=4, edge_accumulation_mode="segmented_tensor_topk", similarity_chunk_size=7
    )
    reference = build_evidence_graph(full, cfg, prompt_attentions=(attention,))
    with cosine_search("partition"), positional_search("indexed"):
        actual = build_evidence_graph(full, cfg, prompt_attentions=(attention,))
    for field in ("row_indices", "column_indices", "weights", "edge_types", "diagnostics"):
        assert getattr(actual, field) == getattr(reference, field)
