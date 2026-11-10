from __future__ import annotations

import numpy as np
import pytest

from mosaickv.cache_state import FullKVState
from mosaickv.evaluation.fox_pooling_index import indexed_pooler, pooling_index
from mosaickv.graph.pooling import pool_prompt_attention_coactivation


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize(
    "mode", ["deferred_layer", "deferred_layer_vectorized", "per_group_reference"]
)
@pytest.mark.parametrize("window", [None, 2])
@pytest.mark.parametrize("eligible", [None, (), (0, 3, 5, 9)])
def test_index_preserves_values_order_readonly_and_errors(tensor, mode, window, eligible):
    rng = np.random.default_rng(42)
    arrays = tuple(rng.random((1, 2, 7, 3), dtype=np.float32) for _ in range(3))
    attentions = tuple(rng.random((1, 4, 5, 7), dtype=np.float32) for _ in range(3))
    if tensor:
        import torch

        arrays = tuple(torch.from_numpy(x) for x in arrays)
        attentions = tuple(torch.from_numpy(x) for x in attentions)
    state = FullKVState.from_tensors(
        tuple((x, x.clone() if tensor else x.copy()) for x in arrays), block_size=2
    )
    kwargs = dict(host_transfer_mode=mode, query_window=window, eligible_node_ids=eligible)
    reference = pool_prompt_attention_coactivation(state, attentions, **kwargs)
    observed = indexed_pooler()(state, attentions, **kwargs)
    assert list(observed) == list(reference)
    for key in reference:
        assert np.array_equal(reference[key], observed[key])
        assert not observed[key].flags.writeable


@pytest.mark.parametrize(
    "kwargs", [{"query_window": 0}, {"eligible_node_ids": [10000]}, {"host_transfer_mode": "bad"}]
)
def test_index_preserves_validation(kwargs):
    array = np.ones((1, 1, 3, 2), dtype=np.float32)
    state = FullKVState.from_tensors(((array, array.copy()),), block_size=1)
    attention = np.ones((1, 1, 2, 3), dtype=np.float32)
    for function in (pool_prompt_attention_coactivation, indexed_pooler()):
        with pytest.raises(ValueError):
            function(state, (attention,), **kwargs)


def test_patch_restores_original_builder_symbol():
    import mosaickv.graph.builder as builder

    before = builder.pool_prompt_attention_coactivation
    with pooling_index():
        assert builder.pool_prompt_attention_coactivation is indexed_pooler()
    assert builder.pool_prompt_attention_coactivation is before
