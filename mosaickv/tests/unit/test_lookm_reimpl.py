from __future__ import annotations

import numpy as np
import pytest

from mosaickv.baselines import LookMReimplementationError, build_lookm_reimpl_plan
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import CacheConfig, LookMConfig
from mosaickv.types import BudgetUnit, LookMMergeStrategy


def _state() -> FullKVState:
    key = np.asarray(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )[:, None, :, :]
    value = np.asarray([10.0, 2.0, 4.0, 20.0], dtype=np.float32).reshape(1, 1, 4, 1)
    return FullKVState.from_tensors(
        ((key, value),),
        modality_spans=(
            ModalitySpan(0, 1, Modality.TEXT),
            ModalitySpan(1, 3, Modality.IMAGE, image_index=0),
            ModalitySpan(3, 4, Modality.TEXT),
        ),
        block_size=1,
        mandatory_logical_positions=(3,),
    )


def _attention() -> tuple[np.ndarray, ...]:
    # Cumulative scores are [1, 10, 2, 1]. Text prior adds max=10 to
    # positions 0 and 3, so position 0 wins the non-recent top-1 selection.
    return (np.asarray([[[[1.0, 10.0, 2.0, 1.0]]]], dtype=np.float32),)


def _config(strategy: LookMMergeStrategy) -> LookMConfig:
    return LookMConfig(
        enabled=True,
        recent_ratio=0.25,
        important_ratio=0.25,
        merge_strategy=strategy,
    )


@pytest.mark.parametrize(
    ("strategy", "expected_anchor_value"),
    (
        (LookMMergeStrategy.AVERAGED, 16.0 / 3.0),
        (LookMMergeStrategy.PIVOTAL, 23.0 / 3.0),
        (LookMMergeStrategy.WEIGHTED, 16.0 / 3.0),
    ),
)
def test_lookm_paper_text_prior_and_merge_equations(
    strategy: LookMMergeStrategy,
    expected_anchor_value: float,
) -> None:
    state = _state()
    plan = build_lookm_reimpl_plan(
        state,
        _attention(),
        _config(strategy),
        CacheConfig(4, BudgetUnit.BLOCKS, 0.5, 1),
    )
    head = plan.heads[0]

    assert head.text_prior_value == 10.0
    assert head.important_physical_positions == (0,)
    assert head.recent_physical_positions == (3,)
    assert head.selected_physical_positions == (0, 3)
    assert head.evicted_physical_positions == (1, 2)
    assert {assignment.pivot_physical_position for assignment in head.assignments} == {0}
    assert float(head.value[0, 0, 0, 0]) == pytest.approx(expected_anchor_value)
    assert plan.active_slots == 2
    assert plan.active_bytes == state.active_bytes // 2
    trace = plan.trace()
    assert trace["official_code"] is False
    trace_head = trace["heads"][0]
    assert "cumulative_attention_scores" not in trace_head
    assert len(trace_head["cumulative_attention_scores_sha256"]) == 64
    assert len(trace_head["pivot_assignments_sha256"]) == 64


def test_lookm_retention_one_is_an_exact_no_merge_path() -> None:
    state = _state()
    config = LookMConfig(
        enabled=True,
        recent_ratio=0.5,
        important_ratio=0.5,
        merge_strategy=LookMMergeStrategy.PIVOTAL,
    )
    plan = build_lookm_reimpl_plan(
        state,
        _attention(),
        config,
        CacheConfig(4, BudgetUnit.BLOCKS, 1.0, 1),
    )

    head = plan.heads[0]
    assert head.selected_physical_positions == (0, 1, 2, 3)
    assert not head.evicted_physical_positions
    assert not head.assignments
    np.testing.assert_array_equal(head.key, state.layers[0].key)
    np.testing.assert_array_equal(head.value, state.layers[0].value)
    assert plan.active_bytes == state.active_bytes


def test_lookm_enforces_common_hard_budget() -> None:
    with pytest.raises(LookMReimplementationError, match=r"exceed cache\.budget_value"):
        build_lookm_reimpl_plan(
            _state(),
            _attention(),
            _config(LookMMergeStrategy.PIVOTAL),
            CacheConfig(1, BudgetUnit.BLOCKS, 0.5, 1),
        )


def test_lookm_requires_token_sized_blocks() -> None:
    state = FullKVState.from_tensors(
        ((np.ones((1, 1, 4, 2), dtype=np.float32),) * 2,),
        block_size=2,
        mandatory_logical_positions=(3,),
    )
    with pytest.raises(LookMReimplementationError, match="token-sized"):
        build_lookm_reimpl_plan(
            state,
            _attention(),
            _config(LookMMergeStrategy.PIVOTAL),
            CacheConfig(4, BudgetUnit.BLOCKS, 0.5, 2),
        )
