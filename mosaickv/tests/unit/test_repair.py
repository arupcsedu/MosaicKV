from __future__ import annotations

import numpy as np
import pytest

from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    QueryVectorState,
)
from mosaickv.cache_state import FullKVState
from mosaickv.config import (
    ConfigurationError,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    SelectionConfig,
    UtilityConfig,
)
from mosaickv.graph import EdgeType, GraphDiagnostics, SparseEvidenceGraph
from mosaickv.graph.pooling import pool_block_descriptors
from mosaickv.prototypes import construct_three_tier_cache
from mosaickv.repair import (
    RepairCacheState,
    RepairTriggerReason,
    calculate_repair_signals,
    normalized_next_token_entropy,
    repair_decode_step,
)
from mosaickv.repair.oracle import evaluate_repair_decode_step
from mosaickv.selection import (
    BudgetedObjective,
    SelectionBudget,
    compute_block_utilities,
    lazy_greedy_select,
)
from mosaickv.types import BudgetUnit, RepairPolicy


def _capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        model_family="synthetic_rope_free",
        architectures=("SyntheticVLM",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=True,
        cache_classes=("tuple",),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=True,
        supports_residual_repair=True,
    )


def _graph(
    full: FullKVState,
    edges: tuple[tuple[int, int, float], ...],
) -> SparseEvidenceGraph:
    nodes = pool_block_descriptors(full)
    return SparseEvidenceGraph(
        nodes=nodes,
        row_indices=tuple(source for source, _target, _weight in edges),
        column_indices=tuple(target for _source, target, _weight in edges),
        weights=tuple(weight for _source, _target, weight in edges),
        edge_types=(EdgeType.SEMANTIC_SIMILARITY,) * len(edges),
        diagnostics=GraphDiagnostics(
            node_count=len(nodes),
            edge_count=len(edges),
            connected_components=1,
            modality_mixing=0.0,
            average_degree=len(edges) / len(nodes),
            evidence_cluster_coverage=None,
            edge_counts=((EdgeType.SEMANTIC_SIMILARITY, len(edges)),),
            maximum_out_degree=1,
            fallback_used=False,
        ),
    )


def _fixture(
    *,
    multiple_prototypes: bool = False,
    eviction_choice: bool = False,
    budget_unit: BudgetUnit = BudgetUnit.BLOCKS,
) -> RepairCacheState:
    values: tuple[float, ...]
    edges: tuple[tuple[int, int, float], ...]
    probabilities: tuple[float, ...]
    if eviction_choice:
        values = (1.0, 2.0, 3.0, 10.0, 30.0, 40.0, 50.0)
        edges = ((3, 0, 0.9), (4, 0, 0.8), (5, 1, 0.7), (6, 2, 0.6))
        probabilities = (0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0)
        budget = 48 if budget_unit is BudgetUnit.BYTES else 6
    elif multiple_prototypes:
        values = (1.0, 2.0, 10.0, 30.0, 40.0)
        edges = ((2, 0, 0.9), (3, 0, 0.8), (4, 1, 0.7))
        probabilities = (0.5, 0.5, 0.0, 0.0, 0.0)
        budget = 32 if budget_unit is BudgetUnit.BYTES else 4
    else:
        values = (1.0, 10.0, 30.0)
        edges = ((1, 0, 0.25), (2, 0, 0.75))
        probabilities = (1.0, 0.0, 0.0)
        budget = 16 if budget_unit is BudgetUnit.BYTES else 2
    key = np.asarray(values, dtype=np.float32).reshape(1, 1, -1, 1)
    value = (key * 2).copy()
    full = FullKVState.from_tensors(
        ((key, value),),
        block_size=1,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    graph = _graph(full, edges)
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node=probabilities,
        attention_provenance="synthetic_repair_fixture",
        rope_aware=True,
    )
    objective = BudgetedObjective(
        graph,
        utilities,
        SelectionConfig(lambda_g=0.0, lambda_m=0.0),
    )
    selection = lazy_greedy_select(objective, SelectionBudget(budget, budget_unit))
    construction = construct_three_tier_cache(
        full,
        graph,
        selection,
        _capabilities(),
        prototype_config=PrototypeConfig(group_size=2),
        residual_config=ResidualConfig(require_pinned_memory=False),
    )
    return RepairCacheState.from_construction(full, construction)


def _must_not_redecode(_state: RepairCacheState) -> np.ndarray:
    raise AssertionError("re-decode callback must not run")


def test_policy_none_never_repairs_or_invokes_redecode() -> None:
    state = _fixture()
    provisional = np.asarray([3.0, 1.0], dtype=np.float32)

    result = repair_decode_step(
        state,
        RepairConfig(policy=RepairPolicy.NONE),
        step_index=0,
        provisional_logits=provisional,
        prototype_attention_mass={0: 0.9},
        re_decode=_must_not_redecode,
    )

    assert not result.event.triggered
    assert result.event.trigger_reason is RepairTriggerReason.POLICY_NONE
    assert result.event.re_decode_count == 0
    assert result.state.exact_node_ids == state.exact_node_ids
    assert result.state.promoted_node_ids == ()


def test_removed_critical_block_is_restored_and_recovers_current_token() -> None:
    state = _fixture()
    calls = 0

    def re_decode(repaired: RepairCacheState) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert repaired.promoted_node_ids == (2,)
        assert repaired.exact_node_ids == (0, 2)
        return np.asarray([1.0, 4.0], dtype=np.float32)

    result = evaluate_repair_decode_step(
        state,
        RepairConfig(
            policy=RepairPolicy.PROTOTYPE_RISK,
            prototype_risk_threshold=0.1,
            max_blocks_per_step=1,
            evaluation_only=True,
        ),
        step_index=0,
        provisional_logits=np.asarray([4.0, 1.0], dtype=np.float32),
        prototype_attention_mass={0: 0.9},
        re_decode=re_decode,
        reference_token_id=1,
    )

    assert calls == 1
    assert result.event.triggered
    assert result.event.trigger_reason is RepairTriggerReason.PROTOTYPE_RISK
    assert result.event.restored_block_ids == (2,)
    assert result.event.restored_bytes == state.full_state.blocks[2].byte_size
    assert result.event.re_decode_count == 1
    assert result.event.maximum_logit_change == pytest.approx(3.0)
    assert result.event.token_changed
    assert result.event.quality_recovered is True
    assert result.event.superseded_prototype_ids == (0,)
    assert result.state.active_cost <= result.state.active_budget.value
    assert 2 not in {payload.source_node_id for payload in result.state.residual_storage.payloads}
    record = result.event.to_record()
    assert record["trigger_reason"] == "prototype_risk_threshold"
    assert record["restored_block_ids"] == [2]

    with pytest.raises(ValueError, match="at most once"):
        repair_decode_step(
            result.state,
            RepairConfig(policy=RepairPolicy.NONE),
            step_index=0,
            provisional_logits=result.final_logits,
            prototype_attention_mass={},
            re_decode=_must_not_redecode,
        )


def test_entropy_signals_and_optional_draft_kl_are_deterministic() -> None:
    state = _fixture()
    logits = np.asarray([0.0, 0.0], dtype=np.float32)

    signals = calculate_repair_signals(
        state,
        logits,
        {0: 0.4},
        draft_distribution=np.asarray([0.9, 0.1]),
    )

    assert normalized_next_token_entropy(logits) == pytest.approx(1.0)
    assert signals.normalized_entropy == pytest.approx(1.0)
    assert signals.total_prototype_attention_mass == pytest.approx(0.4)
    assert signals.prototype_attention_masses == ((0, 0.4),)
    assert signals.prototype_risks[0][1] == pytest.approx(
        0.4 * state.prototype_record(0).diagnostics.dispersion
    )
    assert signals.draft_kl_divergence is not None
    assert signals.draft_kl_divergence > 0


@pytest.mark.parametrize(
    ("policy", "entropy_threshold", "risk_threshold", "expected"),
    (
        (RepairPolicy.ENTROPY, 0.9, 1e9, RepairTriggerReason.ENTROPY),
        (
            RepairPolicy.PROTOTYPE_RISK,
            1.0,
            0.1,
            RepairTriggerReason.PROTOTYPE_RISK,
        ),
        (
            RepairPolicy.ENTROPY_OR_PROTOTYPE_RISK,
            0.9,
            0.1,
            RepairTriggerReason.ENTROPY_AND_PROTOTYPE_RISK,
        ),
    ),
)
def test_online_trigger_policies(
    policy: RepairPolicy,
    entropy_threshold: float,
    risk_threshold: float,
    expected: RepairTriggerReason,
) -> None:
    state = _fixture()
    calls = 0

    def re_decode(_state: RepairCacheState) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray([1.0, 2.0], dtype=np.float32)

    result = repair_decode_step(
        state,
        RepairConfig(
            policy=policy,
            entropy_threshold=entropy_threshold,
            prototype_risk_threshold=risk_threshold,
            max_blocks_per_step=1,
        ),
        step_index=0,
        provisional_logits=np.asarray([0.0, 0.0], dtype=np.float32),
        prototype_attention_mass={0: 0.9},
        re_decode=re_decode,
    )

    assert result.event.trigger_reason is expected
    assert result.event.re_decode_count == 1
    assert calls == 1


@pytest.mark.parametrize(
    "budget_unit",
    (BudgetUnit.BLOCKS, BudgetUnit.RETAINED_SLOTS, BudgetUnit.BYTES),
)
def test_promotion_evicts_lowest_utility_remaining_prototype_to_hold_budget(
    budget_unit: BudgetUnit,
) -> None:
    state = _fixture(eviction_choice=True, budget_unit=budget_unit)
    assert state.active_prototype_ids == (0, 1, 2)
    assert state.prototype_record(2).eviction_utility < state.prototype_record(1).eviction_utility
    assert state.active_cost == state.active_budget.value
    calls = 0

    def re_decode(repaired: RepairCacheState) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert repaired.promoted_node_ids == (3, 4)
        assert repaired.active_prototype_ids == (1,)
        return np.asarray([0.0, 1.0], dtype=np.float32)

    result = repair_decode_step(
        state,
        RepairConfig(
            policy=RepairPolicy.PROTOTYPE_RISK,
            prototype_risk_threshold=0.1,
            max_blocks_per_step=2,
        ),
        step_index=0,
        provisional_logits=np.asarray([1.0, 0.0], dtype=np.float32),
        prototype_attention_mass={0: 0.9, 1: 0.05, 2: 0.05},
        re_decode=re_decode,
    )

    assert calls == 1
    assert result.event.restored_block_ids == (3, 4)
    assert result.event.superseded_prototype_ids == (0,)
    assert result.event.evicted_prototype_ids == (2,)
    assert result.event.active_cost_before == state.active_budget.value
    assert result.event.active_cost_after == state.active_budget.value
    assert result.state.active_cost <= result.state.active_budget.value
    assert result.state.budget_evicted_prototype_ids == (2,)


def test_promoted_blocks_persist_on_later_decode_steps() -> None:
    state = _fixture()
    first = repair_decode_step(
        state,
        RepairConfig(
            policy=RepairPolicy.PROTOTYPE_RISK,
            prototype_risk_threshold=0.1,
            max_blocks_per_step=1,
        ),
        step_index=0,
        provisional_logits=np.asarray([1.0, 0.0]),
        prototype_attention_mass={0: 0.9},
        re_decode=lambda _state: np.asarray([0.0, 1.0]),
    )

    second = repair_decode_step(
        first.state,
        RepairConfig(policy=RepairPolicy.NONE),
        step_index=1,
        provisional_logits=np.asarray([0.0, 1.0]),
        prototype_attention_mass={},
        re_decode=_must_not_redecode,
    )

    assert second.state.promoted_node_ids == (2,)
    assert second.state.exact_node_ids == (0, 2)
    assert len(second.state.events) == 2


def test_oracle_policy_is_evaluation_only() -> None:
    with pytest.raises(ConfigurationError, match="evaluation_only=true"):
        RepairConfig(policy=RepairPolicy.ORACLE)

    state = _fixture()
    config = RepairConfig(
        policy=RepairPolicy.ORACLE,
        evaluation_only=True,
        max_blocks_per_step=1,
    )
    with pytest.raises(ValueError, match="isolated"):
        repair_decode_step(
            state,
            config,
            step_index=0,
            provisional_logits=np.asarray([1.0, 0.0]),
            prototype_attention_mass={0: 0.9},
            re_decode=_must_not_redecode,
        )

    calls: list[RepairCacheState] = []

    def oracle_redecode(repaired: RepairCacheState) -> np.ndarray:
        calls.append(repaired)
        return np.asarray([0.0, 1.0])

    result = evaluate_repair_decode_step(
        state,
        config,
        step_index=0,
        provisional_logits=np.asarray([1.0, 0.0]),
        prototype_attention_mass={0: 0.0},
        re_decode=oracle_redecode,
        reference_token_id=1,
        oracle_should_repair=True,
    )

    assert len(calls) == 1
    assert result.event.trigger_reason is RepairTriggerReason.ORACLE
    assert result.event.quality_recovered is True


def test_repair_config_rejects_invalid_policy_parameters() -> None:
    with pytest.raises(ConfigurationError, match="max_blocks_per_step"):
        RepairConfig(policy=RepairPolicy.ENTROPY, max_blocks_per_step=0)
    with pytest.raises(ConfigurationError, match="entropy_threshold"):
        RepairConfig(entropy_threshold=1.1)
    with pytest.raises(ConfigurationError, match="prototype_risk_threshold"):
        RepairConfig(prototype_risk_threshold=-0.1)
