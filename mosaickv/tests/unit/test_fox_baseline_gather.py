from dataclasses import replace

import numpy as np
import pytest

from mosaickv.baselines import build_exact_baseline_plan
from mosaickv.cache_state import FullKVState
from mosaickv.config import RunConfig
from mosaickv.evaluation.fox_campaign import source_config
from mosaickv.types import MosaicKVMethod


@pytest.mark.parametrize("mode", ["source_views", "batched_layer", "per_head_reference"])
def test_baseline_honors_configured_gather_and_preserves_selected_values(mode):
    key = np.arange(2 * 16 * 4, dtype=np.float32).reshape(1, 2, 16, 4)
    value = key + 100
    source = FullKVState.from_tensors(
        ((key, value),), block_size=1, mandatory_logical_positions=(15,)
    )
    config = RunConfig.from_mapping(source_config("coco"))
    cache = replace(config.cache, budget_value=source.active_bytes // 2, exact_gather_mode=mode)
    scores = {i: float(i) for i in range(len(source.blocks))}
    plan = build_exact_baseline_plan(
        source,
        MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        cache,
        seed=13,
        prompt_attention_by_node=scores,
        position_limits={(0, 0): 8, (0, 1): 8},
    )
    assert len(plan.state.exact.blocks) == 16
    for block, k, v in zip(
        plan.state.exact.blocks,
        plan.state.exact.key_blocks,
        plan.state.exact.value_blocks,
        strict=True,
    ):
        assert np.shares_memory(k, key) is (mode == "source_views")
        assert np.shares_memory(v, value) is (mode == "source_views")
        np.testing.assert_array_equal(
            k, key[:, block.kv_head : block.kv_head + 1, block.physical_cache_indices, :]
        )
        np.testing.assert_array_equal(
            v, value[:, block.kv_head : block.kv_head + 1, block.physical_cache_indices, :]
        )
