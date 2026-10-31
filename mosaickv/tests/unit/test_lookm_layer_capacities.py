"""A single heterogeneous pass must equal the former two-full-plan composition."""

from dataclasses import replace

import pytest
import torch

from mosaickv.baselines.lookm import LookMReimplementationError, build_lookm_reimpl_plan
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import CacheConfig, LookMConfig
from mosaickv.types import BudgetUnit, LookMMergeStrategy


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strategy", list(LookMMergeStrategy))
@pytest.mark.parametrize("extra", [0, 1, 2])
def test_single_pass_is_bitwise_equal_to_two_plan_composition(dtype, strategy, extra):
    generator = torch.Generator().manual_seed(17)
    pairs = tuple(
        tuple(torch.randn(1, 2, 16, 4, generator=generator).to(dtype) for _ in range(2))
        for _ in range(3)
    )
    copies = tuple(tuple(t.clone() for t in pair) for pair in pairs)
    state = FullKVState.from_tensors(
        pairs,
        modality_spans=(ModalitySpan(0, 8, Modality.IMAGE), ModalitySpan(8, 16, Modality.TEXT)),
        block_size=1,
        mandatory_logical_positions=(15,),
    )
    attention = tuple(torch.rand(1, 2, 16, 16, generator=generator) for _ in pairs)
    look = LookMConfig(enabled=True, merge_strategy=strategy)
    cache = CacheConfig(state.active_bytes, BudgetUnit.BYTES, 0.5, 1)
    low = build_lookm_reimpl_plan(state, attention, look, cache, physical_prompt_capacity=8)
    high = build_lookm_reimpl_plan(state, attention, look, cache, physical_prompt_capacity=9)
    expected = tuple(
        h for i in range(3) for h in (high if i < extra else low).heads if h.layer == i
    )
    target_bytes = (8 * 3 + extra) * 2 * 4 * 2 * pairs[0][0].element_size()
    actual = build_lookm_reimpl_plan(
        state,
        attention,
        look,
        replace(cache, budget_value=target_bytes),
        physical_layer_capacities=tuple(8 + int(i < extra) for i in range(3)),
    )
    assert actual.active_bytes == target_bytes
    for old, new in zip(expected, actual.heads, strict=True):
        assert old.selected_physical_positions == new.selected_physical_positions
        assert old.assignments == new.assignments
        assert torch.equal(old.key, new.key)
        assert torch.equal(old.value, new.value)
        assert 15 in new.selected_physical_positions
    assert all(
        torch.equal(a, b)
        for old, new in zip(copies, pairs, strict=True)
        for a, b in zip(old, new, strict=True)
    )


@pytest.mark.parametrize("capacities", [(2,), (0, 2, 2), (2, 2, True), (17, 2, 2)])
def test_invalid_layer_capacities_fail_closed(capacities):
    pairs = tuple((torch.ones(1, 1, 16, 2), torch.ones(1, 1, 16, 2)) for _ in range(3))
    state = FullKVState.from_tensors(pairs, block_size=1)
    with pytest.raises((LookMReimplementationError, ValueError)):
        build_lookm_reimpl_plan(
            state,
            tuple(torch.ones(1, 1, 16, 16) for _ in pairs),
            LookMConfig(enabled=True),
            CacheConfig(state.active_bytes, BudgetUnit.BYTES, 0.5, 1),
            physical_layer_capacities=capacities,
        )
