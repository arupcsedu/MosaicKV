from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mosaickv.backends.hf_runtime as hf_runtime
from mosaickv.adapters.huggingface import (
    AdapterCapabilities,
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
    HuggingFaceMultimodalAdapter,
    QueryVectorState,
)
from mosaickv.backends import build_compression_plan
from mosaickv.backends.hf_runtime import (
    _attention_output_error,
    _attention_outputs,
    _bounded_reservation,
    _compact_trace_records,
    _dense_physical_cache_budget,
    _gather_dense_heads,
    _logit_kl,
    _plan_has_live_residual_repair,
    _refined_budget_candidates,
    _route_head_attention,
    _selection_decision_evidence,
    _trace_catalog,
    _uncertainty_exact_head_routes,
    _uniform_attainable_prototype_position_limits,
    pack_runtime_payloads,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import (
    CacheConfig,
    ForecastingConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    UtilityConfig,
    synthetic_smoke_config,
)
from mosaickv.forecasting import QueryForecast, build_query_forecast
from mosaickv.selection import SelectionDecision, SelectionReason
from mosaickv.types import BudgetUnit, ForecastMode, MosaicKVMethod, RepairPolicy


def _full() -> FullKVState:
    key = np.arange(8 * 4, dtype=np.float32).reshape(1, 1, 8, 4) + 1
    value = np.flip(key, axis=-1).copy()
    return FullKVState.from_tensors(
        ((key, value),),
        block_size=1,
        mandatory_logical_positions=(0, 7),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )


def _forecast() -> QueryForecast:
    queries = np.arange(1 * 2 * 8 * 4, dtype=np.float32).reshape(1, 2, 8, 4) + 1
    return build_query_forecast(
        (queries,),
        (),
        (1,),
        ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
            centroid_count=2,
        ),
        original_logical_sequence_length=8,
        draft_cache_isolated=True,
    )


def _capabilities(*, safe: bool) -> AdapterCapabilities:
    return AdapterCapabilities(
        model_family="synthetic",
        architectures=("Synthetic",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=True,
        cache_classes=("tuple",),
        cache_sequence_dimension=-2,
        cached_key_state=(CachedKeyState.NOT_APPLICABLE if safe else CachedKeyState.POST_ROPE),
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=safe,
        supports_residual_repair=safe,
    )


def _config(method: MosaicKVMethod, ratio: float) -> RunConfig:
    return replace(
        synthetic_smoke_config(),
        method=method,
        cache=CacheConfig(8, BudgetUnit.BLOCKS, ratio, 1),
        forecasting=ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
            centroid_count=2,
        ),
        utility=UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        selection=SelectionConfig(
            lambda_g=-0.25,
            lambda_m=-0.25,
            stop_on_nonpositive_gain=False,
        ),
        residual=ResidualConfig(require_pinned_memory=False),
    )


def test_batched_mixed_tier_packing_is_byte_exact_with_reference_and_tail() -> None:
    key = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)
    value = key.add(100)
    full = FullKVState.from_tensors(
        ((key, value),),
        block_size=1,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    payloads = {
        (0, 0): (
            (0, "exact", 0, key[:, 0:1, 0:1, :], value[:, 0:1, 0:1, :]),
            (3, "prototype", 7, key[:, 0:1, 3:4, :], value[:, 0:1, 3:4, :]),
        ),
        (0, 1): ((2, "exact", 1, key[:, 1:2, 2:3, :], value[:, 1:2, 2:3, :]),),
    }
    tail_key = torch.cat((key, key[:, :, :2, :].add(1_000)), dim=2)
    tail_value = torch.cat((value, value[:, :, :2, :].add(1_000)), dim=2)
    tail = CacheSnapshot(
        (CacheLayerSnapshot(tail_key, tail_value, -2),),
        tuple,
        "tuple",
        6,
        CachedKeyState.NOT_APPLICABLE,
    )
    adapter = SimpleNamespace(
        capabilities=SimpleNamespace(cached_key_state=CachedKeyState.NOT_APPLICABLE)
    )
    kwargs = {
        "tail_snapshot": tail,
        "old_prompt_capacity": 4,
        "physical_prompt_capacity": 3,
        "prototype_log_bias_by_id": {7: 0.25},
    }

    reference = pack_runtime_payloads(
        adapter,
        full,
        payloads,
        (),
        packing_mode="per_head_reference",
        **kwargs,
    )
    batched = pack_runtime_payloads(
        adapter,
        full,
        payloads,
        (),
        packing_mode="batched_layer",
        **kwargs,
    )
    preallocated = pack_runtime_payloads(
        adapter,
        full,
        payloads,
        (),
        packing_mode="preallocated_layer",
        **kwargs,
    )

    assert torch.equal(reference.snapshot.layers[0].key, batched.snapshot.layers[0].key)
    assert torch.equal(reference.snapshot.layers[0].value, batched.snapshot.layers[0].value)
    assert torch.equal(reference.validity_masks[0], batched.validity_masks[0])
    assert torch.equal(reference.snapshot.layers[0].key, preallocated.snapshot.layers[0].key)
    assert torch.equal(reference.snapshot.layers[0].value, preallocated.snapshot.layers[0].value)
    assert torch.equal(reference.validity_masks[0], preallocated.validity_masks[0])
    assert reference.snapshot.active_sequence_length == batched.snapshot.active_sequence_length
    assert reference.prompt_capacity == batched.prompt_capacity
    assert reference.all_slots_valid == batched.all_slots_valid
    assert reference.prototype_slots == batched.prototype_slots
    assert reference.prototype_attention_biases == batched.prototype_attention_biases
    assert reference.prototype_slots == preallocated.prototype_slots
    assert reference.prototype_attention_biases == preallocated.prototype_attention_biases


def test_fused_three_tier_source_packing_matches_scalar_payload_oracle() -> None:
    key = torch.arange(8 * 4, dtype=torch.float32).reshape(1, 1, 8, 4) + 1
    full = FullKVState.from_tensors(
        ((key, key.flip(-1)),),
        block_size=1,
        mandatory_logical_positions=(0, 7),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        _config(MosaicKVMethod.MOSAICKV_PROTO, 0.5),
    )
    assert plan.construction.prototypes
    adapter = SimpleNamespace(
        capabilities=SimpleNamespace(cached_key_state=CachedKeyState.NOT_APPLICABLE)
    )
    prototype_ids = tuple(record.prototype_id for record in plan.construction.prototypes)

    reference = hf_runtime.pack_runtime_cache(
        adapter,
        full,
        plan.construction.state,
        plan.construction.prototypes,
        prototype_ids,
        packing_mode="per_head_reference",
    )
    fused = hf_runtime.pack_runtime_cache(
        adapter,
        full,
        plan.construction.state,
        plan.construction.prototypes,
        prototype_ids,
        packing_mode="preallocated_layer",
    )

    assert torch.equal(reference.snapshot.layers[0].key, fused.snapshot.layers[0].key)
    assert torch.equal(reference.snapshot.layers[0].value, fused.snapshot.layers[0].value)
    assert torch.equal(reference.validity_masks[0], fused.validity_masks[0])
    assert reference.prompt_capacity == fused.prompt_capacity
    assert reference.all_slots_valid == fused.all_slots_valid
    assert reference.prototype_slots == fused.prototype_slots
    assert reference.prototype_attention_biases == fused.prototype_attention_biases
    assert reference.slot_records == fused.slot_records


def test_packed_decode_mask_adds_prototype_multiplicity_bias_to_shared_query_heads() -> None:
    attention = torch.nn.Linear(1, 1, bias=False)
    attention.num_heads = 4  # type: ignore[attr-defined]
    attention.config = SimpleNamespace(num_attention_heads=4)  # type: ignore[attr-defined]
    layer = SimpleNamespace(self_attn=attention)
    adapter = SimpleNamespace(_language_layers=lambda: (layer,))
    state = DecodeState(
        past_key_values=(),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        active_cache_length=3,
        logical_sequence_length=8,
        next_decode_position=8,
        modality_map=(),
        model_state={
            "mosaickv_validity_masks": (torch.ones((2, 3), dtype=torch.bool),),
            "mosaickv_prototype_attention_biases": ((7, 0, 1, 2, float(np.log(4))),),
        },
    )

    mask = HuggingFaceMultimodalAdapter._packed_decode_masks(
        adapter,  # type: ignore[arg-type]
        state,
        torch.ones((1, 1), dtype=torch.long),
    )[0]

    assert tuple(mask.shape) == (1, 4, 1, 4)
    assert torch.equal(mask[0, :2, 0, 2], torch.zeros(2))
    assert torch.allclose(mask[0, 2:, 0, 2], torch.full((2,), float(np.log(4))))
    assert torch.equal(mask[..., 3], torch.zeros((1, 4, 1)))


def test_log_multiplicity_bias_exactly_preserves_identical_member_attention() -> None:
    query = torch.tensor([0.75, -0.25], dtype=torch.float64)
    exact_key = torch.tensor([-0.5, 0.5], dtype=torch.float64)
    member_key = torch.tensor([0.25, 1.0], dtype=torch.float64)
    exact_value = torch.tensor([3.0, -1.0], dtype=torch.float64)
    member_value = torch.tensor([-2.0, 4.0], dtype=torch.float64)
    member_count = 16

    full_logits = torch.stack(
        (
            query @ exact_key,
            *(query @ member_key for _ in range(member_count)),
        )
    )
    full_values = torch.stack((exact_value, *(member_value for _ in range(member_count))))
    full_output = torch.softmax(full_logits, dim=0) @ full_values

    compressed_logits = torch.stack(
        (
            query @ exact_key,
            query @ member_key + float(np.log(member_count)),
        )
    )
    compressed_values = torch.stack((exact_value, member_value))
    compressed_output = torch.softmax(compressed_logits, dim=0) @ compressed_values

    assert torch.allclose(compressed_output, full_output, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "method",
    (
        MosaicKVMethod.MOSAICKV_EXACT,
        MosaicKVMethod.MOSAICKV_PROTO,
        MosaicKVMethod.MOSAICKV_FULL,
    ),
)
def test_tiny_synthetic_cache_plans_every_runtime_method(method: MosaicKVMethod) -> None:
    full = _full()
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        _config(method, 0.5),
    )

    assert plan.method is method
    assert plan.construction.state.statistics.active_kv_bytes <= full.active_bytes
    assert plan.selection.budget_spent <= plan.active_budget_value
    assert {0, 7} <= set(plan.selection.selected_node_ids)


def test_prototype_search_reserves_anchor_slots_before_first_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_budgets: list[int] = []
    original = hf_runtime._selection

    def recording_selection(*args: object, **kwargs: object) -> object:
        requested_budgets.append(int(args[1]))
        return original(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(hf_runtime, "_selection", recording_selection)
    plan = build_compression_plan(
        _full(),
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        _config(MosaicKVMethod.MOSAICKV_PROTO, 0.5),
    )

    assert plan.construction.state.statistics.active_kv_bytes <= _full().active_bytes // 2
    # Target=4 blocks and the two mandatory blocks are also the lower-bound
    # prototype anchors, so the first exact budget is 4-2 rather than 4-1.
    assert requested_budgets == [2]


def test_prompt_anchor_reservation_uses_prompt_only_scores_inside_same_budget() -> None:
    full = _full()
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5),
        selection=replace(
            _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5).selection,
            prompt_anchor_fraction=0.5,
        ),
    )
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        config,
        prompt_attention_by_node={
            node_id: (100.0 if node_id == 6 else 0.0) for node_id in range(8)
        },
    )

    assert 6 in plan.selection.selected_node_ids
    assert plan.selection.decisions[6].reason.value == "prompt_anchor"
    assert plan.selection.budget_spent <= plan.active_budget_value


def test_prompt_ranked_similarity_shortlist_is_bounded_and_keeps_mandatory_nodes() -> None:
    full = _full()
    base = _config(MosaicKVMethod.MOSAICKV_EXACT, 0.25)
    config = replace(
        base,
        graph=replace(
            base.graph,
            similarity_shortlist_multiplier=2.0,
            similarity_shortlist_min_nodes=0,
        ),
    )
    selected = hf_runtime._similarity_shortlist(
        full,
        config,
        source_budget=8,
        target_budget=2,
        prompt_attention_by_node={node_id: float(node_id) for node_id in range(8)},
    )

    # Two retained blocks times a 2x candidate multiplier, with mandatory
    # nodes 0/7 always represented in the candidate set.
    assert selected == (0, 5, 6, 7)


def test_prototype_anchor_cover_prefers_aligned_modality_block() -> None:
    key = np.arange(11 * 4, dtype=np.float32).reshape(1, 1, 11, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(
            ModalitySpan(0, 2, Modality.TEXT),
            ModalitySpan(2, 6, Modality.IMAGE),
            ModalitySpan(6, 10, Modality.TEXT),
            ModalitySpan(10, 11, Modality.VIDEO),
        ),
        block_size=4,
        mandatory_logical_positions=(10,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    config = _config(MosaicKVMethod.MOSAICKV_PROTO, 0.5)
    graph = hf_runtime.build_evidence_graph(
        full,
        replace(config.graph, enabled=False),
    )
    utilities = hf_runtime.compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node={0: 100.0, 1: 1.0, 2: 1.0, 3: 1.0},
        attention_provenance="synthetic_anchor_alignment_fixture",
        rope_aware=True,
    )
    objective = hf_runtime.BudgetedObjective(
        graph,
        utilities,
        SelectionConfig(lambda_g=0.0, lambda_m=0.0),
    )

    anchors = hf_runtime._required_prototype_anchors(
        objective,
        config,
        position_fill_quantum=4,
    )

    assert anchors == (1, 2)
    assert full.blocks[0].position_count == 2
    assert all(full.blocks[node_id].position_count == 4 for node_id in anchors)


def test_prompt_reservation_respects_dense_per_head_capacity() -> None:
    full = _full()
    config = _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5)
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        config,
    )
    chosen = _bounded_reservation(
        hf_runtime.BudgetedObjective(plan.graph, plan.utilities, config.selection),
        8,
        BudgetUnit.BLOCKS,
        1.0,
        {node_id: float(node_id) for node_id in range(8)},
        position_limits={(0, 0): 4},
    )

    # Two mandatory source positions already consume half of the capacity.
    assert len(chosen) == 2
    assert sum(full.blocks[node_id].position_count for node_id in chosen) == 2


def test_prompt_reservation_accounts_for_prior_prototype_anchor_cost_and_capacity() -> None:
    full = _full()
    config = _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5)
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        config,
    )
    chosen = _bounded_reservation(
        hf_runtime.BudgetedObjective(plan.graph, plan.utilities, config.selection),
        4,
        BudgetUnit.BLOCKS,
        1.0,
        {node_id: float(node_id) for node_id in range(8)},
        position_limits={(0, 0): 3},
        reserved_node_ids=(1,),
    )

    # Mandatory nodes 0/7 plus prior prototype anchor 1 exhaust both the
    # three-position head limit and leave only one globally occupied slot.
    assert chosen == ()


def test_per_head_router_keeps_stable_prompt_and_routes_shifted_head_to_draft() -> None:
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5),
        forecasting=replace(
            _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5).forecasting,
            adaptive_head_routing=True,
            head_shift_threshold=0.2,
            stable_head_prompt_weight=1.0,
            shifted_head_prompt_weight=0.0,
        ),
    )
    prompt = np.asarray([0.9, 0.1])
    stable, stable_shift, stable_weight = _route_head_attention(
        prompt,
        (np.asarray([0.85, 0.15]),),
        config,
    )
    shifted, shifted_shift, shifted_weight = _route_head_attention(
        prompt,
        (np.asarray([0.1, 0.9]),),
        config,
    )

    assert stable_shift is not None and stable_shift < 0.2
    assert stable_weight == 1.0
    assert np.array_equal(stable, prompt)
    assert shifted_shift is not None and shifted_shift > 0.2
    assert shifted_weight == 0.0
    assert np.array_equal(shifted, np.asarray([0.1, 0.9]))


def test_uncertainty_exact_head_routes_are_deterministic() -> None:
    shifts = {
        (1, 0): 0.2,
        (0, 2): 0.8,
        (0, 1): 0.8,
        (2, 0): 0.1,
    }

    assert _uncertainty_exact_head_routes(shifts, 0.5) == ("0:1", "0:2")
    assert _uncertainty_exact_head_routes(shifts, 0.0) == ()
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        _uncertainty_exact_head_routes(shifts, 1.1)


def test_no_forecast_control_accepts_uniform_attention_without_query_object() -> None:
    full = _full()
    plan = build_compression_plan(
        full,
        None,
        {node_id: 1.0 for node_id in range(8)},
        _capabilities(safe=False),
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5),
        attention_provenance="declared_no_forecast_uniform_position_control_v1",
    )

    assert plan.forecast is None
    assert plan.utilities.attention_provenance.startswith("declared_no_forecast")


def test_post_rope_proto_and_full_are_explicit_exact_safety_fallbacks() -> None:
    full = _full()
    for method in (MosaicKVMethod.MOSAICKV_PROTO, MosaicKVMethod.MOSAICKV_FULL):
        plan = build_compression_plan(
            full,
            _forecast(),
            {node_id: float(8 - node_id) for node_id in range(8)},
            _capabilities(safe=False),
            _config(method, 0.5),
        )
        assert plan.effective_method.endswith("mosaickv_exact_safety_fallback")
        assert not plan.construction.prototypes
        assert not plan.construction.residual_storage.payloads
        assert not _plan_has_live_residual_repair(method, plan)


def test_policy_none_does_not_enable_online_repair_instrumentation() -> None:
    full = _full()
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        _config(MosaicKVMethod.MOSAICKV_FULL, 0.5),
    )

    assert plan.construction.prototypes
    assert plan.construction.residual_storage.payloads
    assert not _plan_has_live_residual_repair(
        MosaicKVMethod.MOSAICKV_FULL,
        plan,
        repair_policy=RepairPolicy.NONE,
    )


def test_retention_one_is_exact_and_active_bytes_are_monotonic() -> None:
    full = _full()
    active: list[int] = []
    for ratio in (0.25, 0.5, 0.75, 1.0):
        plan = build_compression_plan(
            full,
            _forecast(),
            {node_id: 1.0 for node_id in range(8)},
            _capabilities(safe=False),
            _config(MosaicKVMethod.MOSAICKV_EXACT, ratio),
        )
        active.append(plan.construction.state.statistics.active_kv_bytes)
        if ratio == 1.0:
            plan.construction.state.reconstruct_full_state(full)
            assert not plan.construction.prototypes
            assert not plan.construction.residual_storage.payloads
    assert active == sorted(active)


def test_dense_physical_byte_budget_quantizes_and_constrains_selection() -> None:
    full = _full()
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.25),
        cache=CacheConfig(full.active_bytes // 4, BudgetUnit.BYTES, 0.25, 1),
    )
    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    assert dense.prompt_capacity == 2
    assert dense.target_bytes == full.active_bytes // 4
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=False),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
    )
    assert plan.selection.selected_node_ids == (0, 7)
    assert plan.selection.active_bytes == dense.target_bytes


def test_prototype_search_handles_large_integer_byte_budget_lattice() -> None:
    key = np.arange(8 * 64, dtype=np.float32).reshape(1, 1, 8, 64) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=1,
        mandatory_logical_positions=(0, 7),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_PROTO, 0.5),
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.5, 1),
    )

    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        config,
    )

    assert plan.active_budget_value > 128
    assert plan.construction.active_device_bytes <= plan.active_budget_value


def test_full_method_materializes_prototypes_and_residuals_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    exact_gather_modes: list[object] = []
    implementation = hf_runtime.construct_three_tier_cache

    def counted_construction(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        exact_gather_modes.append(kwargs.get("exact_gather_mode"))
        return implementation(*args, **kwargs)

    monkeypatch.setattr(hf_runtime, "construct_three_tier_cache", counted_construction)
    plan = build_compression_plan(
        _full(),
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        _config(MosaicKVMethod.MOSAICKV_FULL, 0.5),
    )

    assert plan.construction.mode.value == "three_tier"
    assert plan.construction.prototypes
    assert plan.construction.residual_storage.payloads
    assert calls == 1
    assert exact_gather_modes == ["batched_layer"]


def test_prototype_budget_refinement_scans_only_transition_interval() -> None:
    assert _refined_budget_candidates(100, 60, (16, 16, 8)) == (92, 84, 76, 68)
    bounded = _refined_budget_candidates(10_000, 0, (1,), maximum_points=8)
    assert len(bounded) == 8
    assert bounded == tuple(sorted(set(bounded), reverse=True))


def test_dense_physical_byte_budget_supports_fixed_multitoken_blocks() -> None:
    key = np.arange(16 * 4, dtype=np.float32).reshape(1, 1, 16, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=4,
        mandatory_logical_positions=(15,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5),
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.5, 4),
    )

    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    assert dense.prompt_capacity == 8
    assert dense.target_bytes == full.active_bytes // 2
    assert dense.position_limits == {(0, 0): 8}
    assert dense.bytes_per_position == full.active_bytes // 16
    assert dense.minimum_block_representable_fill(4) == (
        dense.target_bytes - 3 * dense.bytes_per_position
    )


def test_three_tier_uniform_layout_may_underfill_hard_physical_cap() -> None:
    sequence = 67
    key = np.arange(sequence * 4, dtype=np.float32).reshape(1, 1, sequence, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(
            ModalitySpan(0, 12, Modality.TEXT),
            ModalitySpan(12, 60, Modality.IMAGE),
            ModalitySpan(60, 67, Modality.TEXT),
        ),
        block_size=16,
        mandatory_logical_positions=(66,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    target_positions = 37
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_PROTO, target_positions / sequence),
        cache=CacheConfig(
            full.active_bytes,
            BudgetUnit.BYTES,
            target_positions / sequence,
            16,
        ),
        prototypes=replace(
            _config(MosaicKVMethod.MOSAICKV_PROTO, 0.5).prototypes,
            group_size=15,
            max_position_span=256,
            budget_planning_mode="one_pass_anchor_cover",
        ),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
    )

    plan = build_compression_plan(
        full,
        None,
        {node_id: float(len(full.blocks) - node_id) for node_id in range(len(full.blocks))},
        _capabilities(safe=True),
        config,
        position_limits={(0, 0): target_positions},
    )

    assert plan.construction.mode.value == "three_tier"
    assert len(plan.construction.prototypes) == 1
    assert plan.construction.active_layouts[0].active_cache_length == 36
    assert plan.construction.active_layouts[0].active_cache_length < target_positions
    assert plan.construction.active_device_bytes < plan.active_budget_value
    assert "physical cap from 37 to 36 positions" in plan.construction.reason


@pytest.mark.parametrize(
    ("modality_spans", "required_node_ids", "physical_cap", "expected_exact"),
    (
        (
            (
                ModalitySpan(0, 8, Modality.TEXT),
                ModalitySpan(8, 72, Modality.IMAGE),
                ModalitySpan(72, 80, Modality.TEXT),
                ModalitySpan(80, 96, Modality.IMAGE),
            ),
            (0, 1, 2, 3),
            63,
            56,
        ),
        (
            (ModalitySpan(0, 96, Modality.IMAGE),),
            (0, 1, 2, 3),
            71,
            64,
        ),
    ),
)
def test_uniform_prototype_layout_steps_down_to_attainable_common_length(
    modality_spans: tuple[ModalitySpan, ...],
    required_node_ids: tuple[int, ...],
    physical_cap: int,
    expected_exact: int,
) -> None:
    key = np.arange(96 * 4, dtype=np.float32).reshape(1, 1, 96, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=modality_spans,
        block_size=16,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    graph = hf_runtime.build_evidence_graph(
        full,
        _config(MosaicKVMethod.MOSAICKV_PROTO, 0.5).graph,
    )

    exact_limits, final_length = _uniform_attainable_prototype_position_limits(
        graph,
        {(0, 0): physical_cap},
        {(0, 0): 1},
        block_size=16,
        required_node_ids=required_node_ids,
        prompt_anchor_node_ids=(),
        evidence_node_ids=(),
    )

    assert exact_limits == {(0, 0): expected_exact}
    assert final_length == expected_exact + 1
    assert final_length < physical_cap


def test_dense_physical_byte_budget_aligns_capacity_after_mandatory_fragments() -> None:
    key = np.arange(65 * 4, dtype=np.float32).reshape(1, 1, 65, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(
            ModalitySpan(0, 48, Modality.IMAGE),
            ModalitySpan(48, 49, Modality.TEXT),
            ModalitySpan(49, 65, Modality.IMAGE),
        ),
        block_size=16,
        mandatory_logical_positions=(48,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    requested_capacity = 48
    bytes_per_position = full.active_bytes // 65
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, requested_capacity / 65),
        cache=CacheConfig(
            requested_capacity * bytes_per_position,
            BudgetUnit.BYTES,
            requested_capacity / 65,
            16,
        ),
    )

    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    assert dense.prompt_capacity == 33
    assert dense.target_bytes == 33 * bytes_per_position
    assert dense.target_bytes <= config.cache.budget_value

    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(len(full.blocks) - node_id) for node_id in range(len(full.blocks))},
        _capabilities(safe=False),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
    )
    selected_positions = sum(block.position_count for block in plan.selection.selected_blocks)
    assert selected_positions == dense.prompt_capacity
    assert plan.selection.active_bytes == dense.target_bytes


def test_dense_budget_uses_common_reference_lattice_for_token_baseline() -> None:
    key = np.arange(65 * 4, dtype=np.float32).reshape(1, 1, 65, 4) + 1
    layers = ((key, np.flip(key, axis=-1).copy()),)
    spans = (
        ModalitySpan(0, 48, Modality.IMAGE),
        ModalitySpan(48, 49, Modality.TEXT),
        ModalitySpan(49, 65, Modality.IMAGE),
    )
    token_state = FullKVState.from_tensors(
        layers,
        modality_spans=spans,
        block_size=1,
        mandatory_logical_positions=tuple(range(58, 65)),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    reference_state = FullKVState.from_tensors(
        layers,
        modality_spans=spans,
        block_size=16,
        mandatory_logical_positions=(64,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    token_config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.75),
        cache=CacheConfig(
            token_state.active_bytes,
            BudgetUnit.BYTES,
            0.75,
            1,
            physical_budget_block_size=16,
            physical_budget_mandatory_policy="last_token",
        ),
    )
    reference_config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.75),
        cache=CacheConfig(
            reference_state.active_bytes,
            BudgetUnit.BYTES,
            0.75,
            16,
        ),
    )

    token_budget = _dense_physical_cache_budget(token_state, token_config)
    reference_budget = _dense_physical_cache_budget(reference_state, reference_config)

    assert token_budget is not None and reference_budget is not None
    assert token_budget.prompt_capacity == reference_budget.prompt_capacity
    assert token_budget.target_bytes == reference_budget.target_bytes


def test_dense_mask_free_selection_excludes_optional_partial_fragments() -> None:
    key = np.arange(64 * 4, dtype=np.float32).reshape(1, 1, 64, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(
            ModalitySpan(0, 15, Modality.IMAGE),
            ModalitySpan(15, 48, Modality.TEXT),
            ModalitySpan(48, 64, Modality.IMAGE),
        ),
        block_size=16,
        mandatory_logical_positions=(63,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.75),
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.75, 16),
        selection=replace(
            _config(MosaicKVMethod.MOSAICKV_EXACT, 0.75).selection,
            prompt_anchor_fraction=0.5,
        ),
    )
    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    assert dense.prompt_capacity == 48
    attention = {
        node_id: (100.0 if block.position_count < 16 else float(len(full.blocks) - node_id))
        for node_id, block in enumerate(full.blocks)
    }
    plan = build_compression_plan(
        full,
        _forecast(),
        attention,
        _capabilities(safe=False),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
        prompt_attention_by_node=attention,
    )

    assert sum(block.position_count for block in plan.selection.selected_blocks) == 48
    assert all(
        block.mandatory or block.position_count == 16 for block in plan.selection.selected_blocks
    )


def test_fidelity_formulas_match_known_attention_and_logit_patterns() -> None:
    values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    snapshot = CacheSnapshot(
        (CacheLayerSnapshot(values.clone(), values, -2),),
        tuple,
        "legacy_tuple",
        2,
        CachedKeyState.NOT_APPLICABLE,
    )

    class Adapter:
        @staticmethod
        def extract_past_key_values(value: object, *, clone: bool = True) -> object:
            del clone
            return value

    state = DecodeState(snapshot, torch.ones((1, 2)), 2, 2, 2, ())
    output = _attention_outputs(
        Adapter(),
        (torch.tensor([[[[0.25, 0.75]]]]),),
        state,
    )
    assert torch.allclose(output[0], torch.tensor([[[0.25, 1.5]]]))
    assert _attention_output_error(output, output) == pytest.approx(0.0)

    reference = torch.tensor([[3.0, 1.0, -2.0]])
    assert _logit_kl(reference, reference) == pytest.approx(0.0)
    assert _logit_kl(reference, torch.tensor([[1.0, 3.0, -2.0]])) > 0


@pytest.mark.parametrize("head_axis,sequence_axis", ((1, 2), (2, 1)))
def test_vectorized_dense_head_gather_preserves_layout_padding_and_tail(
    head_axis: int,
    sequence_axis: int,
) -> None:
    canonical = torch.arange(1 * 2 * 6 * 3, dtype=torch.float32).reshape(1, 2, 6, 3)
    tensor = canonical if head_axis == 1 else canonical.movedim((1, 2), (2, 1))
    packed, validity = _gather_dense_heads(
        tensor,
        head_axis=head_axis,
        sequence_axis=sequence_axis,
        positions=((4, 1), (0, 3, 2)),
        prompt_capacity=4,
        tail_tensor=tensor,
        old_prompt_capacity=5,
    )
    canonical_packed = packed if head_axis == 1 else packed.movedim((2, 1), (1, 2))
    expected = torch.zeros((1, 2, 5, 3))
    expected[:, 0, 0, :] = canonical[:, 0, 4, :]
    expected[:, 0, 1, :] = canonical[:, 0, 1, :]
    expected[:, 0, 4, :] = canonical[:, 0, 5, :]
    expected[:, 1, 0, :] = canonical[:, 1, 0, :]
    expected[:, 1, 1, :] = canonical[:, 1, 3, :]
    expected[:, 1, 2, :] = canonical[:, 1, 2, :]
    expected[:, 1, 4, :] = canonical[:, 1, 5, :]
    assert torch.equal(canonical_packed, expected)
    assert torch.equal(
        validity,
        torch.tensor([[True, True, False, False, True], [True, True, True, False, True]]),
    )


def test_fidelity_attention_accepts_installed_dynamic_cache_protocol() -> None:
    values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])

    class DynamicCacheLike:
        def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
            return ((values.clone(), values),)

    class Adapter:
        @staticmethod
        def extract_past_key_values(value: object, *, clone: bool = True) -> object:
            assert clone is False
            return value

        @staticmethod
        def _legacy_layers(
            cache: DynamicCacheLike,
        ) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], str]:
            return cache.to_legacy_cache(), "cache_object"

        @staticmethod
        def _cache_sequence_dimension(_value: torch.Tensor) -> int:
            return -2

    state = DecodeState(DynamicCacheLike(), torch.ones((1, 2)), 2, 2, 2, ())
    output = _attention_outputs(
        Adapter(),
        (torch.tensor([[[[0.25, 0.75]]]]),),
        state,
    )

    assert torch.allclose(output[0], torch.tensor([[[0.25, 1.5]]]))


def test_compact_trace_preserves_counts_hashes_and_generated_tokens() -> None:
    payload = {
        "selected_blocks": [{"node_id": 2}, {"node_id": 7}],
        "graph_edges": [{"source": 2, "target": 7}],
        "packed_slots": [{"slot": 0}],
        "forecast_statistics": {"heads": [{"layer": 0, "kv_head": 0}]},
        "generated_token_ids": [4, 5, 6],
        "repair_events": [{"triggered": False}],
    }

    _compact_trace_records(payload)  # type: ignore[arg-type]

    for key, count in (
        ("selected_blocks", 2),
        ("graph_edges", 1),
        ("packed_slots", 1),
    ):
        summary = payload[key]
        assert isinstance(summary, dict)
        assert summary["storage"] == "omitted_non_diagnostic"
        assert summary["record_count"] == count
        assert len(summary["sha256"]) == 64
    forecast = payload["forecast_statistics"]
    assert isinstance(forecast, dict)
    assert forecast["heads"]["record_count"] == 1
    assert payload["generated_token_ids"] == [4, 5, 6]
    assert payload["repair_events"] == [{"triggered": False}]


def test_streaming_trace_catalog_matches_post_materialization_compaction() -> None:
    records = [
        {"node_id": 2, "weight": 0.25, "type": "fallback"},
        {"node_id": 7, "weight": 0.75, "type": "semantic"},
    ]
    payload = {"graph_edges": list(records)}
    _compact_trace_records(payload)  # type: ignore[arg-type]

    assert _trace_catalog(iter(records), compact=True) == payload["graph_edges"]
    assert _trace_catalog(iter(records), compact=False) == records
    assert _trace_catalog(iter(()), compact=True) == []


def test_selection_decision_evidence_is_lossless_and_ordered() -> None:
    selection = SimpleNamespace(
        decisions=(
            SelectionDecision(
                node_id=0,
                selected=True,
                selection_rank=0,
                marginal_gain=1.25,
                marginal_gain_per_cost=0.625,
                cost=2,
                reason=SelectionReason.LAZY_GREEDY,
            ),
            SelectionDecision(
                node_id=1,
                selected=False,
                selection_rank=None,
                marginal_gain=0.125,
                marginal_gain_per_cost=0.0625,
                cost=2,
                reason=SelectionReason.NOT_SELECTED_LOWER_GAIN,
            ),
        )
    )

    evidence = _selection_decision_evidence(selection)  # type: ignore[arg-type]

    assert evidence["schema_version"] == 1
    assert evidence["record_count"] == 2
    assert evidence["records"] == [
        {
            "node_id": 0,
            "selected": True,
            "selection_rank": 0,
            "marginal_gain": 1.25,
            "marginal_gain_per_cost": 0.625,
            "cost": 2,
            "reason": "lazy_greedy",
        },
        {
            "node_id": 1,
            "selected": False,
            "selection_rank": None,
            "marginal_gain": 0.125,
            "marginal_gain_per_cost": 0.0625,
            "cost": 2,
            "reason": "not_selected_lower_gain",
        },
    ]
