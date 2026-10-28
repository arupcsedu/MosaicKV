import numpy as np

from mosaickv.baselines.selectors import (
    select_prompt_attention_topk_ids,
    select_prompt_attention_topk_ids_by_head,
)
from mosaickv.cache_state import FullKVState
from mosaickv.evaluation.fox_mechanisms import quota_matched_random_routes
from mosaickv.selection import SelectionBudget
from mosaickv.types import BudgetUnit


def test_random_head_control_preserves_entropy_layer_quota_and_seed():
    routes = [(layer, head) for layer in range(4) for head in range(8)]
    entropy = ((0, 0), (0, 1), (2, 1))
    first = quota_matched_random_routes(routes, entropy, 13, "input")
    assert first == quota_matched_random_routes(routes, entropy, 13, "input")
    assert [sum(layer == k for layer, _ in first) for k in range(4)] == [2, 0, 1, 0]
    assert first != quota_matched_random_routes(routes, entropy, 29, "input")


def test_stable_route_zero_topk_equivalence_is_checked_not_assumed():
    rng = np.random.default_rng(17)
    for _ in range(24):
        heads = int(rng.integers(1, 5))
        length = int(rng.integers(4, 24))
        capacity = int(rng.integers(2, length))
        k = np.zeros((1, heads, length, 4), np.float32)
        state = FullKVState.from_tensors(
            ((k, k.copy()),), block_size=1, mandatory_logical_positions=(0, length - 1)
        )
        scores = {i: float(x) for i, x in enumerate(rng.integers(0, 8, size=heads * length))}
        limits = {(0, h): capacity for h in range(heads)}
        budget = SelectionBudget(heads * capacity * 4 * 4 * 2, BudgetUnit.BYTES)
        topk = select_prompt_attention_topk_ids(
            state, budget, prompt_attention_by_node=scores, position_limits=limits
        )
        stable = select_prompt_attention_topk_ids_by_head(
            state,
            budget,
            prompt_attention_by_node=scores,
            head_routes=tuple(limits),
            position_limits=limits,
        )
        assert topk == stable.selected_node_ids
