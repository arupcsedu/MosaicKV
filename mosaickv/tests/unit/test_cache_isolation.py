from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import mosaickv.backends.hf_runtime as hf_runtime
from mosaickv.adapters.huggingface.types import (
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeOutput,
    DecodeState,
    QueryVectors,
)
from mosaickv.backends.cache_isolation import (
    CacheIsolationError,
    assert_decode_state_unchanged,
    capture_decode_state,
    clone_cache_snapshot,
    construct_isolated_decode_state,
)
from mosaickv.backends.hf_runtime import (
    MosaicKVCompressionPlan,
    PackedRuntimeCache,
    construct_isolated_packed_decode_state,
    probe_isolated_packed_candidate,
)
from mosaickv.baselines.selectors import HeadLocalCapacityRecord
from mosaickv.cache_state import Modality
from mosaickv.selection.hierarchical import (
    CapacityLevelKind,
    RouteCapacityLadder,
    RouteCapacityLevel,
)
from mosaickv.types import BudgetUnit, MosaicKVMethod


class MockCacheAdapter:
    def extract_past_key_values(self, value: Any, *, clone: bool = True) -> Any:
        if not clone:
            return value
        if isinstance(value, CacheSnapshot):
            source = value
        else:
            source = CacheSnapshot(
                layers=tuple(CacheLayerSnapshot(key, item_value, -2) for key, item_value in value),
                source_class=tuple,
                source_kind="tuple",
                active_sequence_length=int(value[0][0].shape[-2]),
                cached_key_state=CachedKeyState.POST_ROPE,
            )
        return clone_cache_snapshot(source)

    def inject_past_key_values(self, snapshot: CacheSnapshot) -> Any:
        return tuple((layer.key, layer.value) for layer in snapshot.layers)

    def decode_one_token(
        self,
        token: Any,
        state: DecodeState,
        *,
        capture_queries: bool,
        capture_attentions: bool,
    ) -> DecodeOutput:
        del token, capture_queries, capture_attentions
        state.past_key_values[0][0].add_(0.01)
        logits = torch.tensor([[0.0, 1.0, 2.0]])
        return DecodeOutput(
            logits=logits,
            next_token_id=torch.tensor([[2]]),
            state=state,
            query_vectors=QueryVectors(()),
        )


class AliasingMockCacheAdapter(MockCacheAdapter):
    def __init__(self, reference_cache: Any) -> None:
        self.reference_cache = reference_cache

    def inject_past_key_values(self, snapshot: CacheSnapshot) -> Any:
        del snapshot
        return self.reference_cache


def _reference_state() -> DecodeState:
    cache = tuple(
        (
            torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) + layer * 100,
            torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) + layer * 200,
        )
        for layer in range(2)
    )
    return DecodeState(
        past_key_values=cache,
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        active_cache_length=4,
        logical_sequence_length=4,
        next_decode_position=4,
        modality_map=(),
        model_state={
            "nested": {
                "mask": torch.tensor([True, False]),
                "history": [torch.tensor([1.0]), {"step": 0}],
            },
            "capacity": 4,
        },
    )


def _packed_snapshot() -> CacheSnapshot:
    return CacheSnapshot(
        layers=tuple(
            CacheLayerSnapshot(
                torch.full((1, 2, 2, 3), float(layer + 1)),
                torch.full((1, 2, 2, 3), float(layer + 11)),
                -2,
            )
            for layer in range(2)
        ),
        source_class=tuple,
        source_kind="tuple",
        active_sequence_length=2,
        cached_key_state=CachedKeyState.POST_ROPE,
    )


def _mock_ladder(route: tuple[int, int], base_node: int) -> RouteCapacityLadder:
    return RouteCapacityLadder(
        route=route,
        levels=(
            RouteCapacityLevel(
                route,
                0,
                CapacityLevelKind.FULL,
                (base_node, base_node + 1, base_node + 2),
                30,
                30.0,
                0.0,
            ),
            RouteCapacityLevel(
                route,
                1,
                CapacityLevelKind.TARGET_ADJACENT,
                (base_node,),
                10,
                10.0,
                2.0,
            ),
        ),
        protected_node_ids=(base_node,),
        target_active_bytes=10,
        target_reached=True,
    )


@dataclass(frozen=True)
class _MockHierarchicalSelectionConfig:
    stable_head_hierarchical_horizon: int = 2
    stable_head_hierarchical_maximum_candidate_evaluations: int = 6
    stable_head_hierarchical_maximum_sequential_rounds: int = 4
    stable_head_hierarchical_partition_count: int = 2
    stable_head_hierarchical_maximum_refinement_depth: int = 1
    stable_head_hierarchical_maximum_rollback_rounds: int = 1
    stable_head_hierarchical_maximum_kl: float = 0.02
    stable_head_hierarchical_maximum_logit_difference: float = 1.0
    stable_head_hierarchical_reserved_reallocation_evaluations: int = 0
    stable_head_hierarchical_reallocation_beam_width: int = 0
    stable_head_hierarchical_reallocation_maximum_combination_width: int = 0
    stable_head_hierarchical_logit_feedback_restore_donor: bool = False


@dataclass(frozen=True)
class _MockCacheConfig:
    budget_value: int = 40
    total_storage_budget_bytes: int | None = 40
    packing_mode: str = "gather"
    budget_unit: BudgetUnit = BudgetUnit.BYTES
    block_size: int = 1


@dataclass(frozen=True)
class _MockRunConfig:
    selection: _MockHierarchicalSelectionConfig = _MockHierarchicalSelectionConfig()
    cache: _MockCacheConfig = _MockCacheConfig()


def test_candidate_mutations_cannot_leak_into_reference_nested_state() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()
    checkpoint = capture_decode_state(adapter, reference)

    with construct_isolated_decode_state(
        adapter,
        reference,
        candidate_id="candidate-a",
    ) as handle:
        candidate = handle.state
        candidate.past_key_values[0][0].add_(1000)
        candidate.attention_mask.zero_()
        candidate.model_state["nested"]["mask"].logical_not_()
        candidate.model_state["nested"]["history"][0].add_(9)
        candidate.model_state["nested"]["history"].append(torch.tensor([5.0]))

        handle.assert_reference_unchanged()
        assert handle.audit.shared_cache_storage_count == 0
        assert handle.audit.shared_auxiliary_storage_count == 0

    assert handle.closed
    assert_decode_state_unchanged(adapter, reference, checkpoint)
    with pytest.raises(CacheIsolationError, match="disposed"):
        _ = handle.state


def test_packed_candidate_owns_cache_mask_and_model_metadata() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()
    packed = _packed_snapshot()
    packed_copy = clone_cache_snapshot(packed)
    validity = torch.tensor([[True, False], [True, True]])

    handle = construct_isolated_decode_state(
        adapter,
        reference,
        candidate_id="packed",
        cache_snapshot=packed,
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        active_cache_length=2,
        model_state={"mosaickv_validity_masks": (validity,)},
    )
    candidate = handle.state

    assert candidate.active_cache_length == 2
    assert candidate.logical_sequence_length == 4
    assert candidate.next_decode_position == 4
    candidate.past_key_values[0][0].zero_()
    candidate.model_state["mosaickv_validity_masks"][0].zero_()
    assert torch.equal(packed.layers[0].key, packed_copy.layers[0].key)
    assert torch.equal(validity, torch.tensor([[True, False], [True, True]]))
    handle.close()


def test_hf_packed_constructor_builds_owned_validity_metadata() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()
    validity = (torch.tensor([[True, False], [False, True]]),) * 2
    packed = PackedRuntimeCache(
        snapshot=_packed_snapshot(),
        validity_masks=validity,
        prompt_capacity=2,
        all_slots_valid=False,
        prototype_slots=(),
        prototype_attention_biases=(),
        slot_records=(),
    )

    handle = construct_isolated_packed_decode_state(
        cast(Any, adapter),
        reference,
        packed,
        candidate_id="hf-packed",
    )
    candidate = handle.state

    assert candidate.active_cache_length == 2
    assert candidate.model_state["mosaickv_prompt_capacity"] == 2
    assert candidate.model_state["mosaickv_packed_all_valid"] is False
    candidate.model_state["mosaickv_validity_masks"][0].zero_()
    assert torch.equal(validity[0], torch.tensor([[True, False], [False, True]]))
    handle.assert_reference_unchanged()
    handle.close()


def test_isolated_packed_probe_discards_mutated_candidate_state() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()
    checkpoint = capture_decode_state(adapter, reference)
    packed = PackedRuntimeCache(
        snapshot=_packed_snapshot(),
        validity_masks=(torch.ones((2, 2), dtype=torch.bool),) * 2,
        prompt_capacity=2,
        all_slots_valid=True,
        prototype_slots=(),
        prototype_attention_biases=(),
        slot_records=(),
    )

    result = probe_isolated_packed_candidate(
        cast(Any, adapter),
        reference,
        packed,
        candidate_id="probe",
        teacher_forced_inputs=(torch.tensor([[1]]), torch.tensor([[2]])),
        reference_logits=(
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.tensor([[0.0, 1.0, 2.0]]),
        ),
    )

    assert result.greedy_token_match
    assert result.maximum_forward_kl == pytest.approx(0.0)
    assert result.maximum_absolute_logit_difference == pytest.approx(0.0)
    assert result.evaluated_steps == 2
    assert_decode_state_unchanged(adapter, reference, checkpoint)


def test_hf_hierarchical_loop_materializes_every_probe_through_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MockCacheAdapter()
    adapter.capabilities = SimpleNamespace()
    reference = _reference_state()
    checkpoint = capture_decode_state(adapter, reference)
    prefill = SimpleNamespace(state=reference, next_token_id=torch.tensor([[2]]))
    full_state = SimpleNamespace(
        blocks=tuple(
            SimpleNamespace(
                byte_size=10,
                layer=0,
                kv_head=0 if index < 3 else 1,
                position_count=1,
            )
            for index in range(6)
        )
    )
    plan = MosaicKVCompressionPlan(
        method=MosaicKVMethod.MOSAICKV_EXACT,
        effective_method="mosaickv_exact",
        full_state=cast(Any, full_state),
        forecast=None,
        graph=cast(Any, None),
        utilities=cast(Any, None),
        selection=cast(Any, None),
        construction=cast(Any, None),
        source_budget_value=60,
        shared_active_budget_value=20,
        active_budget_value=20,
        planner_active_budget_value=20,
        configured_physical_capacity_fraction=1.0,
        graph_seconds=0.0,
        selection_seconds=0.0,
        utility_seconds=0.0,
        reservation_seconds=0.0,
        selector_seconds=0.0,
        tier_seconds=0.0,
        similarity_candidate_node_ids=None,
        optimization_node_ids=None,
        planner_source_node_ids=None,
        external_exact_node_ids=(),
        external_exact_capacity_records=(),
        external_exact_logit_shortlist=(),
        prompt_attention_by_node=None,
        stable_routing_seconds=0.0,
        similarity_shortlist_seconds=0.0,
        planner_mapping_seconds=0.0,
        graph_head_routes=(),
        graph_routing_provenance="mock",
    )
    ladders = (_mock_ladder((0, 0), 0), _mock_ladder((0, 1), 3))
    packed = PackedRuntimeCache(
        snapshot=_packed_snapshot(),
        validity_masks=(torch.ones((2, 2), dtype=torch.bool),) * 2,
        prompt_capacity=2,
        all_slots_valid=True,
        prototype_slots=(),
        prototype_attention_biases=(),
        slot_records=(),
    )
    materialized_bytes: list[int] = []

    def exact_construction(
        _plan: Any,
        _capabilities: Any,
        _config: Any,
        retained: Any,
        **_kwargs: Any,
    ) -> tuple[Any, Any, tuple[int, ...]]:
        retained_ids = tuple(sorted(retained))
        active_bytes = len(retained_ids) * 10
        materialized_bytes.append(active_bytes)
        construction = SimpleNamespace(
            state=SimpleNamespace(statistics=SimpleNamespace(active_kv_bytes=active_bytes)),
            active_layouts=(SimpleNamespace(active_cache_length=2),),
            exact_node_ids=retained_ids,
        )
        return construction, SimpleNamespace(), retained_ids

    monkeypatch.setattr(
        hf_runtime,
        "_hierarchical_route_ladders",
        lambda *_args, **_kwargs: ladders,
    )
    monkeypatch.setattr(
        hf_runtime,
        "_hierarchical_reference_rollout",
        lambda *_args: (
            (torch.tensor([[1]]), torch.tensor([[2]])),
            (
                torch.tensor([[0.0, 1.0, 2.0]]),
                torch.tensor([[0.0, 1.0, 2.0]]),
            ),
        ),
    )
    monkeypatch.setattr(hf_runtime, "_progressive_exact_construction", exact_construction)
    monkeypatch.setattr(hf_runtime, "pack_runtime_cache", lambda *_args, **_kwargs: packed)
    monkeypatch.setattr(
        hf_runtime,
        "_progressive_group_reservation",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(
        hf_runtime,
        "_progressive_eviction_risks",
        lambda *_args: (0.0,) * 6,
    )
    monkeypatch.setattr(
        hf_runtime,
        "_progressive_capacity_records",
        lambda *_args, **_kwargs: (),
    )

    selected_plan, decision = hf_runtime._run_hierarchical_logit_safe_eviction(
        cast(Any, adapter),
        cast(Any, prefill),
        plan,
        cast(Any, _MockRunConfig()),
        key_transform=None,
        position_limits={(0, 0): 1, (0, 1): 1},
    )

    assert decision.target_reached
    assert decision.final_joint_probe_passed
    # FullKV exceeds each route's one-position physical limit, so V107 removes
    # it from the final-candidate ladder. The feasible target becomes the
    # baseline and only the complete joint cache is isolated and probed.
    assert decision.evaluations_used == 1
    assert materialized_bytes == [20]
    assert selected_plan.construction.state.statistics.active_kv_bytes == 20
    assert_decode_state_unchanged(adapter, reference, checkpoint)


def test_hf_hierarchical_fullkv_fallback_does_not_reuse_feasible_level_zero_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtered and unrestricted ladders have distinct level-zero namespaces."""

    adapter = MockCacheAdapter()
    adapter.capabilities = SimpleNamespace()
    reference = _reference_state()
    prefill = SimpleNamespace(state=reference, next_token_id=torch.tensor([[2]]))
    full_state = SimpleNamespace(
        blocks=tuple(
            SimpleNamespace(
                byte_size=10,
                layer=0,
                kv_head=0 if index < 3 else 1,
                position_count=1,
            )
            for index in range(6)
        )
    )
    plan = MosaicKVCompressionPlan(
        method=MosaicKVMethod.MOSAICKV_EXACT,
        effective_method="mosaickv_exact",
        full_state=cast(Any, full_state),
        forecast=None,
        graph=cast(Any, None),
        utilities=cast(Any, None),
        selection=cast(Any, None),
        construction=cast(Any, None),
        source_budget_value=60,
        shared_active_budget_value=20,
        active_budget_value=20,
        planner_active_budget_value=20,
        configured_physical_capacity_fraction=1.0,
        graph_seconds=0.0,
        selection_seconds=0.0,
        utility_seconds=0.0,
        reservation_seconds=0.0,
        selector_seconds=0.0,
        tier_seconds=0.0,
        similarity_candidate_node_ids=None,
        optimization_node_ids=None,
        planner_source_node_ids=None,
        external_exact_node_ids=(),
        external_exact_capacity_records=(),
        external_exact_logit_shortlist=(),
        prompt_attention_by_node=None,
        stable_routing_seconds=0.0,
        similarity_shortlist_seconds=0.0,
        planner_mapping_seconds=0.0,
        graph_head_routes=(),
        graph_routing_provenance="mock",
    )
    ladders = (_mock_ladder((0, 0), 0), _mock_ladder((0, 1), 3))
    packed = PackedRuntimeCache(
        snapshot=_packed_snapshot(),
        validity_masks=(torch.ones((2, 2), dtype=torch.bool),) * 2,
        prompt_capacity=2,
        all_slots_valid=True,
        prototype_slots=(),
        prototype_attention_biases=(),
        slot_records=(),
    )
    materialized: list[tuple[int, ...]] = []

    def exact_construction(
        _plan: Any,
        _capabilities: Any,
        _config: Any,
        retained: Any,
        **_kwargs: Any,
    ) -> tuple[Any, Any, tuple[int, ...]]:
        retained_ids = tuple(sorted(retained))
        materialized.append(retained_ids)
        active_bytes = len(retained_ids) * 10
        construction = SimpleNamespace(
            state=SimpleNamespace(statistics=SimpleNamespace(active_kv_bytes=active_bytes)),
            active_layouts=(SimpleNamespace(active_cache_length=len(retained_ids)),),
            exact_node_ids=retained_ids,
        )
        return construction, SimpleNamespace(), retained_ids

    def force_fallback(
        filtered_ladders: Any,
        *,
        probe: Any,
        fullkv_fallback_selection: Any,
        **_kwargs: Any,
    ) -> Any:
        proposal = hf_runtime.HierarchicalCapacityProposal(
            proposal_id="filtered-level-zero",
            kind=hf_runtime.HierarchicalProposalKind.JOINT,
            level_indices=tuple((ladder.route, 0) for ladder in filtered_ladders),
            affected_routes=tuple(ladder.route for ladder in filtered_ladders),
        )
        _ = probe(proposal)
        return hf_runtime.BoundedHierarchicalProbeDecision(
            selection=fullkv_fallback_selection,
            evaluations=(SimpleNamespace(proposal=proposal, passed=False),),
            accepted_target_routes=(),
            unresolved_exact_routes=tuple(ladder.route for ladder in filtered_ladders),
            rollback_attempts=(),
            target_reached=False,
            selected_candidate_passed=False,
            final_joint_probe_passed=False,
            used_fullkv_fallback=True,
            evaluations_used=1,
            rounds_used=1,
        )

    monkeypatch.setattr(
        hf_runtime,
        "_hierarchical_route_ladders",
        lambda *_args, **_kwargs: ladders,
    )
    monkeypatch.setattr(
        hf_runtime,
        "_hierarchical_reference_rollout",
        lambda *_args: (
            (torch.tensor([[1]]), torch.tensor([[2]])),
            (
                torch.tensor([[0.0, 1.0, 2.0]]),
                torch.tensor([[0.0, 1.0, 2.0]]),
            ),
        ),
    )
    monkeypatch.setattr(hf_runtime, "_progressive_exact_construction", exact_construction)
    monkeypatch.setattr(hf_runtime, "pack_runtime_cache", lambda *_args, **_kwargs: packed)
    monkeypatch.setattr(hf_runtime, "run_bounded_hierarchical_probe_loop", force_fallback)
    monkeypatch.setattr(
        hf_runtime,
        "_progressive_group_reservation",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(hf_runtime, "_progressive_eviction_risks", lambda *_args: (0.0,) * 6)
    monkeypatch.setattr(
        hf_runtime,
        "_progressive_capacity_records",
        lambda *_args, **_kwargs: (),
    )

    selected_plan, decision = hf_runtime._run_hierarchical_logit_safe_eviction(
        cast(Any, adapter),
        cast(Any, prefill),
        plan,
        cast(Any, _MockRunConfig()),
        key_transform=None,
        position_limits={(0, 0): 1, (0, 1): 1},
    )

    assert decision.used_fullkv_fallback
    assert materialized == [(0, 3), (0, 1, 2, 3, 4, 5)]
    assert selected_plan.construction.state.statistics.active_kv_bytes == 60


def test_hierarchical_accounting_failure_trace_contains_all_nine_preflight_fields() -> None:
    accounting = hf_runtime.ExactCacheStorageAccounting(
        knapsack_bytes=48,
        unique_selected_source_bytes=40,
        planner_tier_bytes=24,
        external_exact_bytes=16,
        constructed_active_bytes=40,
        physical_padding_bytes=8,
        selected_source_block_count=5,
        planner_source_block_count=3,
        external_source_block_count=2,
    )
    payload = hf_runtime._hierarchical_accounting_failure_trace_payload(
        {
            "used_fullkv_fallback": False,
            "evaluations": [{"proposal_id": "joint"}],
        },
        accounting,
        hard_active_kv_budget_bytes=44,
        elapsed_seconds=0.25,
    )

    required = {
        "status",
        "effective_method",
        "used_fullkv_fallback",
        "attempted_active_kv_bytes",
        "hard_active_kv_budget_bytes",
        "probe_decisions",
        "timing_breakdown",
        "byte_budget_materialization",
        "hierarchical_logit_safe_eviction",
    }
    assert required <= set(payload)
    assert {
        "knapsack_bytes",
        "unique_selected_source_bytes",
        "planner_tier_bytes",
        "external_exact_bytes",
        "constructed_active_bytes",
        "physical_padding_bytes",
    } <= set(payload)
    assert payload["over_budget_cache_packed"] is False
    assert payload["over_budget_cache_decoded"] is False


def test_two_candidates_are_independent_from_each_other_and_reference() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()
    first = construct_isolated_decode_state(adapter, reference, candidate_id="first")
    second = construct_isolated_decode_state(adapter, reference, candidate_id="second")
    second_before = capture_decode_state(adapter, second.state)

    first.state.past_key_values[1][1].mul_(0)
    first.state.model_state["nested"]["mask"].zero_()

    second.assert_reference_unchanged()
    assert_decode_state_unchanged(adapter, second.state, second_before)
    first.close()
    second.close()


def test_constructor_rejects_an_adapter_that_reinjects_reference_storage() -> None:
    reference = _reference_state()
    adapter = AliasingMockCacheAdapter(reference.past_key_values)

    with pytest.raises(CacheIsolationError, match="share mutable storage"):
        construct_isolated_decode_state(adapter, reference, candidate_id="alias")


def test_adoption_rejects_a_successor_that_aliases_reference_storage() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()
    handle = construct_isolated_decode_state(adapter, reference, candidate_id="adopt")

    with pytest.raises(CacheIsolationError, match="aliases mutable reference storage"):
        handle.adopt(reference)
    handle.close()


def test_reference_mutation_is_detected_before_candidate_disposal() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()
    handle = construct_isolated_decode_state(adapter, reference, candidate_id="guard")
    reference.model_state["nested"]["mask"].zero_()

    with pytest.raises(CacheIsolationError, match="mutated the reference"):
        handle.close()
    assert handle.closed


def test_invalid_identity_and_active_length_fail_closed() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()

    with pytest.raises(CacheIsolationError, match="candidate_id"):
        construct_isolated_decode_state(adapter, reference, candidate_id="")
    with pytest.raises(CacheIsolationError, match="active length differs"):
        construct_isolated_decode_state(
            adapter,
            reference,
            candidate_id="wrong-length",
            cache_snapshot=_packed_snapshot(),
            active_cache_length=3,
        )
    with pytest.raises(CacheIsolationError, match="attention-mask length"):
        construct_isolated_decode_state(
            adapter,
            reference,
            candidate_id="wrong-mask",
            cache_snapshot=_packed_snapshot(),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )


def test_snapshot_layer_lengths_must_match_declared_active_length() -> None:
    invalid = CacheSnapshot(
        layers=(
            CacheLayerSnapshot(
                torch.ones((1, 2, 3, 4)),
                torch.ones((1, 2, 3, 4)),
                -2,
            ),
        ),
        source_class=tuple,
        source_kind="tuple",
        active_sequence_length=2,
        cached_key_state=CachedKeyState.POST_ROPE,
    )

    with pytest.raises(CacheIsolationError, match="differs from active_sequence_length"):
        clone_cache_snapshot(invalid)


def test_context_disposes_candidate_when_probe_raises() -> None:
    adapter = MockCacheAdapter()
    reference = _reference_state()
    handle = construct_isolated_decode_state(adapter, reference, candidate_id="exception")

    with pytest.raises(RuntimeError, match="probe failed"), handle:
        raise RuntimeError("probe failed")

    assert handle.closed


def test_hierarchical_fullkv_fallback_records_actual_route_capacity() -> None:
    blocks = tuple(
        SimpleNamespace(
            layer=0,
            kv_head=0,
            position_count=4,
            byte_size=8,
            modality=Modality.TEXT,
        )
        for _ in range(2)
    )
    plan = SimpleNamespace(
        full_state=SimpleNamespace(blocks=blocks),
        external_exact_capacity_records=(
            HeadLocalCapacityRecord(
                layer=0,
                kv_head=0,
                position_limit=4,
                selected_positions=4,
                shortfall_positions=0,
                selected_bytes=8,
            ),
        ),
    )
    reservation = SimpleNamespace(
        groups_by_route={},
        protected_bytes_by_route={},
        target_bytes_by_route={},
        candidate_count_by_route={},
        protected_score_by_route={},
    )

    with pytest.raises(
        hf_runtime.HuggingFaceRuntimeError,
        match="non-fallback selection exceeds",
    ):
        hf_runtime._progressive_capacity_records(
            plan,
            frozenset({0, 1}),
            reservation,
        )

    records = hf_runtime._progressive_capacity_records(
        plan,
        frozenset({0, 1}),
        reservation,
        allow_fullkv_capacity_expansion=True,
    )

    assert len(records) == 1
    assert records[0].position_limit == 8
    assert records[0].selected_positions == 8
    assert records[0].shortfall_positions == 0
    assert records[0].selected_bytes == 16


def test_hard_budget_fallback_trace_is_complete_and_rejects_before_packing() -> None:
    fallback_accounting = hf_runtime.ExactCacheStorageAccounting(
        knapsack_bytes=320,
        unique_selected_source_bytes=320,
        planner_tier_bytes=240,
        external_exact_bytes=80,
        constructed_active_bytes=320,
        physical_padding_bytes=0,
        selected_source_block_count=32,
        planner_source_block_count=24,
        external_source_block_count=8,
    )
    rejected_accounting = hf_runtime.ExactCacheStorageAccounting(
        knapsack_bytes=40,
        unique_selected_source_bytes=40,
        planner_tier_bytes=40,
        external_exact_bytes=0,
        constructed_active_bytes=40,
        physical_padding_bytes=8,
        selected_source_block_count=4,
        planner_source_block_count=4,
        external_source_block_count=0,
    )
    payload = hf_runtime._hard_budget_fallback_trace_payload(
        {
            "used_fullkv_fallback": True,
            "target_reached": False,
            "evaluations_used": 3,
            "rounds_used": 2,
            "evaluations": [
                {
                    "candidate_id": "probe-1",
                    "passed": False,
                    "forward_kl": 0.04,
                }
            ],
        },
        attempted_active_kv_bytes=320,
        hard_active_kv_budget_bytes=40,
        attempted_prompt_capacity=608,
        hard_prompt_capacity=75,
        timing_breakdown={"hierarchical_eviction": 2.5, "cache_packing": 0.0},
        fullkv_fallback_accounting=fallback_accounting,
        rejected_compressed_proposal_accounting=rejected_accounting,
        rejected_compressed_proposal_id="joint-0",
    )

    assert payload["failure_classification"] == "hard_budget_fullkv_fallback"
    assert payload["used_fullkv_fallback"] is True
    assert payload["attempted_active_kv_bytes"] == 320
    assert payload["hard_active_kv_budget_bytes"] == 40
    assert payload["over_budget_cache_packed"] is False
    assert payload["over_budget_cache_decoded"] is False
    assert payload["over_budget_cache_scope"] == "final_selected_fullkv_fallback"
    assert payload["isolated_probe_candidates_materialized"] is True
    assert payload["isolated_probe_states_discarded"] is True
    assert payload["probe_decisions"][0]["candidate_id"] == "probe-1"
    decision = payload["hierarchical_logit_safe_eviction"]
    assert decision["used_fullkv_fallback"] is True
    assert decision["evaluations"][0]["candidate_id"] == "probe-1"
    materialization = payload["byte_budget_materialization"]
    assert materialization["status"] == "rejected_before_materialization"
    assert materialization["packed_active_kv_bytes"] is None
    assert materialization["knapsack_bytes"] == 320
    assert payload["rejected_compressed_proposal_id"] == "joint-0"
    assert payload["rejected_compressed_proposal_accounting"] == asdict(
        rejected_accounting
    )
    assert payload["fullkv_fallback_accounting"] == asdict(fallback_accounting)
    assert payload["timing_breakdown"]["cache_packing"] == 0.0


def test_hard_budget_fallback_trace_rejects_nonfallback_or_in_budget_records() -> None:
    fallback_accounting = hf_runtime.ExactCacheStorageAccounting(
        knapsack_bytes=320,
        unique_selected_source_bytes=320,
        planner_tier_bytes=320,
        external_exact_bytes=0,
        constructed_active_bytes=320,
        physical_padding_bytes=0,
        selected_source_block_count=32,
        planner_source_block_count=32,
        external_source_block_count=0,
    )
    rejected_accounting = hf_runtime.ExactCacheStorageAccounting(
        knapsack_bytes=40,
        unique_selected_source_bytes=40,
        planner_tier_bytes=40,
        external_exact_bytes=0,
        constructed_active_bytes=40,
        physical_padding_bytes=0,
        selected_source_block_count=4,
        planner_source_block_count=4,
        external_source_block_count=0,
    )
    kwargs = {
        "attempted_active_kv_bytes": 320,
        "hard_active_kv_budget_bytes": 40,
        "attempted_prompt_capacity": 608,
        "hard_prompt_capacity": 75,
        "timing_breakdown": {},
        "fullkv_fallback_accounting": fallback_accounting,
        "rejected_compressed_proposal_accounting": rejected_accounting,
        "rejected_compressed_proposal_id": "joint-0",
    }
    with pytest.raises(ValueError, match="FullKV fallback"):
        hf_runtime._hard_budget_fallback_trace_payload(
            {"used_fullkv_fallback": False},
            **kwargs,
        )
    with pytest.raises(ValueError, match="active-byte budget"):
        hf_runtime._hard_budget_fallback_trace_payload(
            {"used_fullkv_fallback": True},
            **{**kwargs, "attempted_active_kv_bytes": 40},
        )


def test_feasible_route_ladder_has_multiple_levels_below_54_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocks = tuple(
        SimpleNamespace(
            byte_size=8,
            layer=0,
            kv_head=0,
            position_count=1,
            mandatory=index == 0,
        )
        for index in range(64)
    )
    plan = SimpleNamespace(
        full_state=SimpleNamespace(blocks=blocks, active_bytes=512),
    )
    reservation = SimpleNamespace(groups_by_route={})
    monkeypatch.setattr(
        hf_runtime,
        "_progressive_eviction_risks",
        lambda _plan: tuple(float(index) for index in range(64)),
    )
    monkeypatch.setattr(
        hf_runtime,
        "_progressive_group_reservation",
        lambda *_args: reservation,
    )
    monkeypatch.setattr(
        hf_runtime,
        "_progressive_fixed_source_ids",
        lambda _plan: frozenset(),
    )

    ladders = hf_runtime._hierarchical_route_ladders(
        cast(Any, plan),
        cast(Any, SimpleNamespace()),
        {(0, 0): 54},
        feasible_baseline_node_ids_by_route={(0, 0): frozenset(range(54))},
        refinement_positions=8,
    )

    assert len(ladders) == 1
    ordinary = tuple(
        level
        for level in ladders[0].levels
        if level.kind is not CapacityLevelKind.REALLOCATION_ALTERNATIVE
    )
    assert [level.active_bytes // 8 for level in ordinary] == [54, 50, 46]
    assert ordinary[0].kind is CapacityLevelKind.FEASIBLE_BASELINE
    assert all(len(level.selected_node_ids) <= 54 for level in ladders[0].levels)
    assert ladders[0].proposed_removed_bytes == 8 * 8


def test_route_capacity_failure_trace_contains_every_preflight_field() -> None:
    proposal = hf_runtime.HierarchicalCapacityProposal(
        proposal_id="joint-infeasible",
        kind=hf_runtime.HierarchicalProposalKind.JOINT,
        level_indices=(((0, 0), 0),),
        affected_routes=((0, 0),),
    )
    payload = hf_runtime._route_capacity_failure_trace_payload(
        proposal,
        (
            hf_runtime.RouteCapacityViolation(
                route=(0, 0),
                selected_positions=12,
                position_limit=8,
            ),
        ),
        attempted_active_kv_bytes=48,
        hard_active_kv_budget_bytes=40,
        attempted_prompt_capacity=12,
        hard_prompt_capacity=8,
    )

    required = {
        "effective_method",
        "used_fullkv_fallback",
        "attempted_active_kv_bytes",
        "hard_active_kv_budget_bytes",
        "probe_decisions",
        "timing_breakdown",
        "byte_budget_materialization",
        "hierarchical_logit_safe_eviction",
    }
    assert required <= set(payload)
    assert payload["used_fullkv_fallback"] is False
    assert payload["over_budget_cache_packed"] is False
    assert payload["over_budget_cache_decoded"] is False
    assert payload["probe_decisions"][0]["status"] == "rejected_before_logit_probe"


def test_hierarchical_effective_limits_intersect_external_attainable_capacity() -> None:
    plan = SimpleNamespace(
        full_state=SimpleNamespace(
            blocks=(
                SimpleNamespace(layer=0, kv_head=0),
                SimpleNamespace(layer=0, kv_head=1),
            )
        ),
        external_exact_capacity_records=(
            HeadLocalCapacityRecord(
                layer=0,
                kv_head=0,
                position_limit=64,
                selected_positions=64,
                shortfall_positions=0,
                selected_bytes=128,
            ),
        ),
    )

    effective = hf_runtime._hierarchical_effective_position_limits(
        plan,
        {(0, 0): 75, (0, 1): 75},
    )

    assert effective == {(0, 0): 64, (0, 1): 75}
