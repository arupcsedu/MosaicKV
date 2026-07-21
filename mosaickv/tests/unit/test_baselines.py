from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mosaickv.baselines import (
    build_exact_baseline_plan,
    resolve_baseline_budget,
    select_exact_baseline,
    value_novelty_scores,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import CacheConfig
from mosaickv.selection import SelectionBudget
from mosaickv.types import BudgetUnit, MosaicKVMethod

COMPRESSED_BASELINES = (
    MosaicKVMethod.RANDOM_KV,
    MosaicKVMethod.UNIFORM_KV,
    MosaicKVMethod.PROMPT_ATTENTION_TOPK,
    MosaicKVMethod.VALUE_TOPK,
)


def _full() -> FullKVState:
    positions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    layers = []
    for layer in range(2):
        value = np.stack((positions, positions + layer * 0.1), axis=0)[None, ...]
        key = value + 0.25
        layers.append((key, value))
    return FullKVState.from_tensors(
        tuple(layers),
        modality_spans=(
            ModalitySpan(0, 4, Modality.TEXT),
            ModalitySpan(4, 8, Modality.IMAGE, image_index=0),
        ),
        block_size=2,
        mandatory_logical_positions=(7,),
    )


def _cache(ratio: float, unit: BudgetUnit = BudgetUnit.BLOCKS) -> CacheConfig:
    return CacheConfig(2_147_483_647, unit, ratio, 2)


def _prompt_scores(full: FullKVState) -> dict[int, float]:
    return {node_id: float(len(full.blocks) - node_id) for node_id in range(len(full.blocks))}


def _plan(full: FullKVState, method: MosaicKVMethod, ratio: float, seed: int = 11):
    return build_exact_baseline_plan(
        full,
        method,
        _cache(ratio),
        seed=seed,
        prompt_attention_by_node=(
            _prompt_scores(full) if method is MosaicKVMethod.PROMPT_ATTENTION_TOPK else None
        ),
    )


@pytest.mark.parametrize("method", COMPRESSED_BASELINES)
def test_baselines_share_budget_mandatory_policy_and_exact_byte_accounting(
    method: MosaicKVMethod,
) -> None:
    full = _full()
    plan = _plan(full, method, 0.5)
    mandatory = {node_id for node_id, block in enumerate(full.blocks) if block.mandatory}

    assert plan.source_budget_value == 16
    assert plan.active_budget_value == 8
    assert plan.selection.budget_spent == 8
    assert mandatory <= set(plan.selection.selected_node_ids)
    assert plan.selection.active_bytes == sum(
        block.byte_size for block in plan.selection.selected_blocks
    )
    assert plan.state.statistics.active_kv_bytes == plan.selection.active_bytes
    assert not plan.state.prototypes.source_blocks
    assert not plan.state.residuals.source_blocks


def test_uniform_baseline_allocates_equal_fraction_to_every_stratum() -> None:
    full = _full()
    plan = _plan(full, MosaicKVMethod.UNIFORM_KV, 0.5)
    source = Counter((block.layer, block.kv_head, block.modality) for block in full.blocks)
    selected = Counter(
        (block.layer, block.kv_head, block.modality) for block in plan.selection.selected_blocks
    )

    assert set(source) == set(selected)
    assert {selected[group] / source[group] for group in source} == {0.5}


def test_random_baseline_is_seeded_and_deterministic() -> None:
    full = _full()
    _source, budget = resolve_baseline_budget(full, _cache(0.5))

    first = select_exact_baseline(full, MosaicKVMethod.RANDOM_KV, budget, seed=7)
    second = select_exact_baseline(full, MosaicKVMethod.RANDOM_KV, budget, seed=7)
    different = select_exact_baseline(full, MosaicKVMethod.RANDOM_KV, budget, seed=8)

    assert first == second
    assert first.selection_order != different.selection_order
    assert first.seed == 7
    assert first.score_provenance.endswith("seed=7")


def test_prompt_attention_topk_uses_only_supplied_prompt_mass() -> None:
    full = _full()
    _source, budget = resolve_baseline_budget(full, _cache(0.5))
    scores = _prompt_scores(full)

    result = select_exact_baseline(
        full,
        MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        budget,
        seed=0,
        prompt_attention_by_node=scores,
    )

    mandatory = {node_id for node_id, block in enumerate(full.blocks) if block.mandatory}
    optional_selected = [node_id for node_id in result.selection_order if node_id not in mandatory]
    expected = sorted(
        (node_id for node_id in range(len(full.blocks)) if node_id not in mandatory),
        key=lambda node_id: (-scores[node_id], node_id),
    )[: len(optional_selected)]
    assert optional_selected == expected
    assert result.score_provenance == "eager_prompt_window_attention_mass_v1"


def test_prompt_attention_topk_requires_complete_scores() -> None:
    full = _full()
    _source, budget = resolve_baseline_budget(full, _cache(0.5))

    with pytest.raises(ValueError, match="cover every source block exactly once"):
        select_exact_baseline(
            full,
            MosaicKVMethod.PROMPT_ATTENTION_TOPK,
            budget,
            seed=0,
            prompt_attention_by_node={0: 1.0},
        )


def test_value_topk_is_deterministic_and_ranks_value_novelty_only() -> None:
    full = _full()
    scores = value_novelty_scores(full, chunk_size=2)
    _source, budget = resolve_baseline_budget(full, _cache(0.5))

    first = select_exact_baseline(full, MosaicKVMethod.VALUE_TOPK, budget, seed=0)
    second = select_exact_baseline(full, MosaicKVMethod.VALUE_TOPK, budget, seed=999)

    assert first.selected_node_ids == second.selected_node_ids
    assert tuple(decision.score for decision in first.decisions) == pytest.approx(
        tuple(scores[node_id] for node_id in range(len(full.blocks)))
    )
    assert first.score_provenance.startswith("nearest_value_cosine_novelty")


def test_byte_budget_never_exceeds_the_shared_exact_limit() -> None:
    full = _full()
    block_bytes = full.blocks[0].byte_size
    cache = CacheConfig(block_bytes * 8, BudgetUnit.BYTES, 0.5, 2)
    source, budget = resolve_baseline_budget(full, cache)
    # The explicit upper bound converts this into a partial byte-budget run.
    assert budget.value < source

    result = select_exact_baseline(full, MosaicKVMethod.RANDOM_KV, budget, seed=3)

    assert result.active_bytes == result.budget_spent
    assert result.active_bytes <= block_bytes * 8


@pytest.mark.parametrize("method", COMPRESSED_BASELINES)
def test_every_baseline_retention_one_reconstructs_full_cache(method: MosaicKVMethod) -> None:
    full = _full()
    plan = _plan(full, method, 1.0)

    reconstructed = plan.state.reconstruct_full_state(full)
    assert plan.selection.selected_node_ids == tuple(range(len(full.blocks)))
    assert plan.selection.active_bytes == full.active_bytes
    assert reconstructed.active_bytes == full.active_bytes
    assert not plan.state.prototypes.source_blocks
    assert not plan.state.residuals.source_blocks


def test_baseline_budget_matches_selection_budget_object() -> None:
    full = _full()
    source, resolved = resolve_baseline_budget(full, _cache(0.5))

    assert source == len(full.blocks)
    assert resolved == SelectionBudget(8, BudgetUnit.BLOCKS)
