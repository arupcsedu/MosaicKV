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
    _evidence_preservation_scores,
    _gather_dense_heads,
    _logit_kl,
    _mosaickv_effective_capacity,
    _plan_has_live_residual_repair,
    _refined_budget_candidates,
    _route_head_attention,
    _selection_decision_evidence,
    _short_horizon_candidate_evaluation,
    _trace_catalog,
    _uncertainty_exact_head_routes,
    _uniform_attainable_prototype_position_limits,
    _validate_combined_active_layouts,
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
from mosaickv.graph import BlockEvidenceMetadata
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


def test_spatial_stratified_prompt_evidence_covers_grid_before_duplicates() -> None:
    key = np.arange(8 * 2, dtype=np.float32).reshape(1, 1, 8, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        block_size=1,
        modality_spans=(ModalitySpan(0, 8, Modality.IMAGE, image_index=0, grid_shape=(2, 4)),),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    nodes = tuple(
        SimpleNamespace(
            node_id=node_id,
            block=block,
            evidence=BlockEvidenceMetadata.from_block(block),
        )
        for node_id, block in enumerate(full.blocks)
    )
    objective = SimpleNamespace(
        graph=SimpleNamespace(nodes=nodes),
        utilities=SimpleNamespace(
            for_node=lambda node_id: SimpleNamespace(local_utility=float(node_id) / 10.0)
        ),
    )
    prompt = {node_id: float(node_id + 1) for node_id in range(len(nodes))}

    scores = _evidence_preservation_scores(
        objective,  # type: ignore[arg-type]
        mode="spatial_stratified_prompt",
        prompt_attention_by_node=prompt,
        grid_rows=2,
        grid_columns=2,
    )

    first_per_stratum = (1, 3, 5, 7)
    second_per_stratum = (0, 2, 4, 6)
    assert min(scores[node_id] for node_id in first_per_stratum) > max(
        scores[node_id] for node_id in second_per_stratum
    )
    assert scores[1] > scores[0]


def test_spatial_stratified_prompt_evidence_requires_complete_nonnegative_scores() -> None:
    key = np.zeros((1, 1, 1, 2), dtype=np.float32)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        block_size=1,
        modality_spans=(ModalitySpan(0, 1, Modality.IMAGE, image_index=0, grid_shape=(1, 1)),),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    block = full.blocks[0]
    objective = SimpleNamespace(
        graph=SimpleNamespace(
            nodes=(
                SimpleNamespace(
                    node_id=0,
                    block=block,
                    evidence=BlockEvidenceMetadata.from_block(block),
                ),
            )
        ),
        utilities=SimpleNamespace(for_node=lambda _node_id: SimpleNamespace(local_utility=0.0)),
    )

    with pytest.raises(ValueError, match="cover every graph node"):
        _evidence_preservation_scores(
            objective,  # type: ignore[arg-type]
            mode="spatial_stratified_prompt",
            prompt_attention_by_node={},
        )


def test_routed_head_only_plan_bypasses_stable_graph_utility_and_selection() -> None:
    sequence = 8
    key = np.arange(2 * sequence * 4, dtype=np.float32).reshape(1, 2, sequence, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=1,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    base = _config(MosaicKVMethod.MOSAICKV_EXACT, 10 / 16)
    config = replace(
        base,
        cache=CacheConfig(16, BudgetUnit.BLOCKS, 10 / 16, 1),
        graph=replace(
            base.graph,
            similarity_shortlist_multiplier=2.0,
            candidate_descriptor_pooling=True,
            candidate_graph_mode="shortlist_rerank",
            uncertain_head_only=True,
            uncertain_head_fraction=0.5,
            uncertain_head_threshold=0.0,
        ),
        selection=replace(
            base.selection,
            algorithm="vectorized_coverage",
            minimum_stratum_fraction=0.0,
            coverage_correction_rounds=0,
            stable_head_prompt_routing=True,
            stable_head_uniform_fill=True,
            routed_head_only_planning=True,
        ),
        prototypes=replace(
            base.prototypes,
            uncertain_head_only=True,
            uncertain_head_fraction=0.5,
            uncertain_head_threshold=0.0,
        ),
    )
    prompt_attention = {
        node_id: (100.0 - node_id if block.kv_head == 1 else 1.0)
        for node_id, block in enumerate(full.blocks)
    }
    forecast_attention = {
        node_id: (100.0 - node_id if block.kv_head == 0 else 1.0)
        for node_id, block in enumerate(full.blocks)
    }

    plan = build_compression_plan(
        full,
        None,
        forecast_attention,
        _capabilities(safe=False),
        config,
        prompt_attention_by_node=prompt_attention,
        graph_head_uncertainty={(0, 0): 1.0, (0, 1): 0.0},
    )

    routed_ids = tuple(node_id for node_id, block in enumerate(full.blocks) if block.kv_head == 0)
    assert plan.optimization_node_ids == routed_ids
    assert plan.graph_head_routes == ((0, 0),)
    assert all(
        full.blocks[source].kv_head == 0 and full.blocks[target].kv_head == 0
        for source, target in zip(
            plan.graph.row_indices,
            plan.graph.column_indices,
            strict=True,
        )
    )
    stable_ids = tuple(node_id for node_id, block in enumerate(full.blocks) if block.kv_head == 1)
    assert all(
        plan.utilities.for_node(node_id).expected_attention_output_contribution == 0
        for node_id in stable_ids
    )
    assert all(
        plan.selection.decisions[node_id].reason is SelectionReason.PROMPT_ANCHOR
        for node_id in stable_ids
    )
    assert len(plan.selection.decisions) == len(full.blocks)
    assert plan.selection.budget_spent <= plan.selection.budget.value


def test_routed_head_only_plan_compacts_and_defers_source_audit() -> None:
    sequence = 16
    key = np.arange(2 * sequence * 4, dtype=np.float32).reshape(1, 2, sequence, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=1,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    base = _config(MosaicKVMethod.MOSAICKV_EXACT, 12 / 32)
    config = replace(
        base,
        cache=CacheConfig(32, BudgetUnit.BLOCKS, 12 / 32, 1),
        graph=replace(
            base.graph,
            similarity_shortlist_multiplier=2.0,
            candidate_descriptor_pooling=True,
            candidate_graph_mode="shortlist_rerank",
            uncertain_head_only=True,
            uncertain_head_fraction=0.5,
            uncertain_head_threshold=0.0,
        ),
        selection=replace(
            base.selection,
            algorithm="vectorized_coverage",
            minimum_stratum_fraction=0.0,
            coverage_correction_rounds=0,
            stable_head_prompt_routing=True,
            routed_head_only_planning=True,
        ),
        prototypes=replace(
            base.prototypes,
            uncertain_head_only=True,
            uncertain_head_fraction=0.5,
            uncertain_head_threshold=0.0,
        ),
    )
    prompt_attention = {
        node_id: (100.0 - node_id if block.kv_head == 1 else 1.0)
        for node_id, block in enumerate(full.blocks)
    }
    forecast_attention = {
        node_id: (100.0 - node_id if block.kv_head == 0 else 1.0)
        for node_id, block in enumerate(full.blocks)
    }

    plan = build_compression_plan(
        full,
        None,
        forecast_attention,
        _capabilities(safe=False),
        config,
        prompt_attention_by_node=prompt_attention,
        graph_head_uncertainty={(0, 0): 1.0, (0, 1): 0.0},
    )

    assert plan.uses_compact_node_ids
    assert plan.planner_source_node_ids is not None
    assert len(plan.graph.nodes) < len(full.blocks)
    assert tuple(node.node_id for node in plan.graph.nodes) == tuple(range(len(plan.graph.nodes)))
    assert tuple(node.source_id for node in plan.graph.nodes) == (plan.planner_source_node_ids)
    risks = hf_runtime._progressive_eviction_risks(plan)
    assert len(risks) == len(full.blocks)
    planner_sources = {node.source_id for node in plan.graph.nodes}
    stable_sources = set(range(len(full.blocks))) - planner_sources
    assert stable_sources
    assert all(risks[source_id] == prompt_attention[source_id] for source_id in stable_sources)
    assert hf_runtime._progressive_fixed_source_ids(plan) == frozenset(
        plan.source_node_id(node_id) for node_id in plan.selection.selected_node_ids
    )
    assert (
        tuple(sorted(plan.source_node_id(node_id) for node_id in plan.selection.selected_node_ids))
        == plan.construction.exact_node_ids
    )
    trace = hf_runtime._plan_trace(plan, capture_selection_evidence=True)
    evidence = trace["selection_decision_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["record_count"] == len(full.blocks)

    full_budget = hf_runtime.SelectionBudget(
        sum(block.byte_size for block in full.blocks),
        BudgetUnit.BYTES,
    )
    construction, progressive_selection, external_ids = (
        hf_runtime._progressive_exact_construction(
            plan,
            _capabilities(safe=False),
            config,
            range(len(full.blocks)),
            key_transform=None,
            active_budget=full_budget,
        )
    )
    assert construction.exact_node_ids == tuple(range(len(full.blocks)))
    assert not set(external_ids).intersection(planner_sources)
    assert {
        plan.source_node_id(node_id)
        for node_id in progressive_selection.selected_node_ids
    } == planner_sources

    removed_planner = min(planner_sources)
    removed_external = min(stable_sources)
    retained = set(range(len(full.blocks))) - {removed_planner, removed_external}
    reduced, reduced_selection, reduced_external = (
        hf_runtime._progressive_exact_construction(
            plan,
            _capabilities(safe=False),
            config,
            retained,
            key_transform=None,
            active_budget=full_budget,
        )
    )
    assert set(reduced.exact_node_ids) == retained
    assert removed_planner not in {
        plan.source_node_id(node_id) for node_id in reduced_selection.selected_node_ids
    }
    assert removed_external not in reduced_external
    assert not set(reduced_external).intersection(planner_sources)
    knapsack_bytes = sum(full.blocks[node_id].byte_size for node_id in retained)
    accounting = hf_runtime._exact_cache_storage_accounting(
        plan,
        retained,
        reduced,
        reduced_external,
        knapsack_bytes=knapsack_bytes,
        packed_physical_bytes=reduced.state.statistics.active_kv_bytes + 64,
    )
    assert accounting.logical_boundaries_aligned
    assert accounting.knapsack_bytes == knapsack_bytes
    assert accounting.unique_selected_source_bytes == knapsack_bytes
    assert accounting.planner_tier_bytes == reduced_selection.active_bytes
    assert accounting.external_exact_bytes == sum(
        full.blocks[node_id].byte_size for node_id in reduced_external
    )
    assert accounting.constructed_active_bytes == knapsack_bytes
    assert accounting.physical_padding_bytes == 64


def test_dense_routed_plan_keeps_stable_exact_blocks_outside_objective() -> None:
    sequence = 16
    key = np.arange(2 * sequence * 4, dtype=np.float32).reshape(1, 2, sequence, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=1,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    base = _config(MosaicKVMethod.MOSAICKV_EXACT, 12 / 32)
    config = replace(
        base,
        cache=CacheConfig(32, BudgetUnit.BLOCKS, 12 / 32, 1),
        graph=replace(
            base.graph,
            similarity_shortlist_multiplier=2.0,
            candidate_descriptor_pooling=True,
            candidate_graph_mode="shortlist_rerank",
            uncertain_head_only=True,
            uncertain_head_fraction=0.5,
            uncertain_head_threshold=0.0,
        ),
        selection=replace(
            base.selection,
            algorithm="vectorized_coverage",
            minimum_stratum_fraction=0.0,
            coverage_correction_rounds=0,
            stable_head_prompt_routing=True,
            routed_head_only_planning=True,
        ),
        prototypes=replace(
            base.prototypes,
            uncertain_head_only=True,
            uncertain_head_fraction=0.5,
            uncertain_head_threshold=0.0,
        ),
    )
    prompt_attention = {
        node_id: (100.0 - node_id if block.kv_head == 1 else 1.0)
        for node_id, block in enumerate(full.blocks)
    }
    forecast_attention = {
        node_id: (100.0 - node_id if block.kv_head == 0 else 1.0)
        for node_id, block in enumerate(full.blocks)
    }

    plan = build_compression_plan(
        full,
        None,
        forecast_attention,
        _capabilities(safe=False),
        config,
        position_limits={(0, 0): 6, (0, 1): 6},
        prompt_attention_by_node=prompt_attention,
        graph_head_uncertainty={(0, 0): 1.0, (0, 1): 0.0},
    )

    assert len(plan.graph.nodes) == sequence
    assert {node.block.kv_head for node in plan.graph.nodes} == {0}
    assert len(plan.external_exact_node_ids) == 6
    assert len(plan.external_exact_capacity_records) == 1
    assert plan.external_exact_capacity_records[0].shortfall_positions == 0
    assert all(full.blocks[node_id].kv_head == 1 for node_id in plan.external_exact_node_ids)
    assert len(plan.construction.exact_node_ids) == 12
    assert {
        layout.active_cache_length for layout in plan.construction.active_layouts
    } == {6}
    assert set(plan.external_exact_node_ids) <= set(plan.construction.exact_node_ids)
    assert plan.construction.active_budget.value == 12
    assert plan.selection.budget.value == 6
    trace = hf_runtime._plan_trace(plan, capture_selection_evidence=False)
    assert trace["planner_active_budget_value"] == 6
    byte_accounting = trace["byte_budget_accounting"]
    assert isinstance(byte_accounting, dict)
    assert byte_accounting["budget_unit"] == "blocks"
    assert byte_accounting["shared_hard_budget_bytes"] is None
    routing = trace["candidate_graph_routing"]
    assert isinstance(routing, dict)
    capacity = routing["external_stable_capacity"]
    assert isinstance(capacity, dict)
    assert capacity["selected_positions"] == 6
    assert capacity["shortfall_positions"] == 0

    scaled = build_compression_plan(
        full,
        None,
        forecast_attention,
        _capabilities(safe=False),
        replace(
            config,
            selection=replace(config.selection, physical_capacity_fraction=0.5),
        ),
        position_limits={(0, 0): 6, (0, 1): 6},
        prompt_attention_by_node=prompt_attention,
        graph_head_uncertainty={(0, 0): 1.0, (0, 1): 0.0},
    )
    assert scaled.shared_active_budget_value == 12
    assert scaled.active_budget_value == 6
    assert scaled.configured_physical_capacity_fraction == 0.5
    assert len(scaled.external_exact_node_ids) == 3
    assert scaled.external_exact_capacity_records[0].position_limit == 3
    assert len(scaled.construction.exact_node_ids) == 6
    assert scaled.selection.budget.value == 3


def test_incompatible_prototype_fallback_fills_routed_and_external_exact_heads() -> None:
    sizes = (6, 6, 8, 5)
    sequence = sum(sizes)
    key = np.arange(2 * sequence * 4, dtype=np.float32).reshape(1, 2, sequence, 4) + 1
    spans: list[ModalitySpan] = []
    offset = 0
    for image_index, size in enumerate(sizes):
        spans.append(
            ModalitySpan(
                offset,
                offset + size,
                Modality.IMAGE,
                image_index=image_index,
            )
        )
        offset += size
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=tuple(spans),
        block_size=8,
        mandatory_logical_positions=(sequence - 1,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    target_positions = 13
    base = _config(MosaicKVMethod.MOSAICKV_FULL, target_positions / sequence)
    config = replace(
        base,
        cache=replace(
            base.cache,
            budget_value=(full.active_bytes * target_positions) // sequence,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=target_positions / sequence,
            block_size=8,
            physical_budget_block_size=8,
            physical_budget_mandatory_policy="last_token",
        ),
        graph=replace(
            base.graph,
            similarity_shortlist_multiplier=2.0,
            candidate_descriptor_pooling=True,
            candidate_graph_mode="shortlist_rerank",
            uncertain_head_only=True,
            uncertain_head_fraction=0.5,
            uncertain_head_threshold=0.0,
        ),
        selection=replace(
            base.selection,
            algorithm="vectorized_coverage",
            minimum_stratum_fraction=0.0,
            coverage_correction_rounds=0,
            stable_head_prompt_routing=True,
            routed_head_only_planning=True,
        ),
        prototypes=replace(
            base.prototypes,
            uncertain_head_only=True,
            uncertain_head_fraction=0.5,
            uncertain_head_threshold=0.0,
            source_modalities=("image",),
            allowed_modality_pairs=("text:text",),
        ),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
    )
    dense = _dense_physical_cache_budget(full, config)
    assert dense is not None and dense.prompt_capacity == target_positions
    prompt_attention = {
        node_id: (100.0 - node_id if block.kv_head == 1 and block.position_count == 6 else 1.0)
        for node_id, block in enumerate(full.blocks)
    }

    plan = build_compression_plan(
        full,
        None,
        {node_id: 1.0 for node_id in range(len(full.blocks))},
        _capabilities(safe=True),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
        prompt_attention_by_node=prompt_attention,
        graph_head_uncertainty={(0, 0): 1.0, (0, 1): 0.0},
    )

    assert plan.construction.mode.value == "exact_only_incompatible_assignments"
    assert plan.external_exact_capacity_records[0].selected_positions == target_positions
    assert plan.external_exact_capacity_records[0].shortfall_positions == 0
    active_lengths = {
        (layout.layer, layout.kv_head): layout.active_cache_length
        for layout in plan.construction.active_layouts
    }
    assert active_lengths == {(0, 0): target_positions, (0, 1): target_positions}
    assert plan.construction.state.statistics.active_kv_bytes == dense.target_bytes


def test_external_stable_exact_blocks_coexist_with_routed_prototype_members() -> None:
    key = np.arange(2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=1,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    routed = tuple(node_id for node_id, block in enumerate(full.blocks) if block.kv_head == 0)
    external = tuple(node_id for node_id, block in enumerate(full.blocks) if block.kv_head == 1)[:1]
    base = _config(MosaicKVMethod.MOSAICKV_PROTO, 3 / 8)
    graph = hf_runtime.build_evidence_graph(
        full,
        base.graph,
        source_node_ids=routed,
    )
    utilities = hf_runtime.compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node={node_id: float(4 - node_id) for node_id in range(4)},
        attention_provenance="synthetic_external_exact_test",
        rope_aware=True,
    )
    objective = hf_runtime.BudgetedObjective(
        graph,
        utilities,
        replace(base.selection, stop_on_nonpositive_gain=False),
    )
    selection = hf_runtime.lazy_greedy_select(
        objective,
        hf_runtime.SelectionBudget(1, BudgetUnit.BLOCKS),
    )

    construction = hf_runtime.construct_three_tier_cache(
        full,
        graph,
        selection,
        _capabilities(safe=True),
        prototype_config=replace(
            base.prototypes,
            source_modalities=("text",),
            group_size=4,
            max_position_span=16,
        ),
        residual_config=ResidualConfig(enabled=False, require_pinned_memory=False),
        additional_exact_source_node_ids=external,
        active_budget=hf_runtime.SelectionBudget(3, BudgetUnit.BLOCKS),
    )

    assert construction.mode.value == "three_tier"
    assert set(external) <= set(construction.exact_node_ids)
    assert construction.prototypes[0].assigned_node_ids == (1, 2, 3)
    assert not set(external).intersection(construction.prototypes[0].assigned_node_ids)


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


def test_device_prototype_attention_reduction_matches_numpy_reference() -> None:
    attention = torch.tensor(
        [
            [
                [[0.2, 0.3, 0.5]],
                [[0.1, 0.4, 0.5]],
                [[0.6, 0.2, 0.2]],
                [[0.5, 0.3, 0.2]],
            ]
        ],
        dtype=torch.float32,
    )
    slots = ((7, 0, 0, 1), (8, 0, 1, 2))
    masks = (torch.ones((2, 3), dtype=torch.bool),)

    device = hf_runtime._prototype_attention_mass((attention,), slots, masks)
    reference = hf_runtime._prototype_attention_mass(
        (attention.numpy(),),
        slots,
        (masks[0].numpy(),),
    )

    assert device == pytest.approx(reference, abs=1e-7)


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


def test_similarity_shortlist_caps_nonmandatory_candidates_independently() -> None:
    full = _full()
    base = _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5)
    config = replace(
        base,
        graph=replace(
            base.graph,
            similarity_shortlist_multiplier=4.0,
            candidate_nonmandatory_fraction=0.25,
        ),
    )

    selected = hf_runtime._similarity_shortlist(
        full,
        config,
        source_budget=8,
        target_budget=4,
        prompt_attention_by_node={node_id: float(node_id) for node_id in range(8)},
    )

    # Six blocks are non-mandatory, so ceil(6 * .25) admits exactly two in
    # addition to mandatory source blocks 0 and 7.
    assert selected == (0, 5, 6, 7)


def test_selection_only_prototype_head_drops_unselected_blocks_without_merging() -> None:
    full = _full()
    base = _config(MosaicKVMethod.MOSAICKV_PROTO, 0.25)
    config = replace(
        base,
        prototypes=replace(base.prototypes, selection_only_head_routes=("0:0",)),
    )

    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        config,
    )

    assert len(plan.selection.selected_node_ids) == 2
    assert not plan.construction.prototypes
    assert plan.construction.state.exact.blocks == plan.selection.selected_blocks


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


def test_bounded_prototype_anchor_cover_selects_high_prompt_segments_within_limit() -> None:
    sequence = 48
    key = np.arange(sequence * 4, dtype=np.float32).reshape(1, 1, sequence, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(ModalitySpan(0, sequence, Modality.IMAGE),),
        block_size=4,
        mandatory_logical_positions=(sequence - 1,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    base = _config(MosaicKVMethod.MOSAICKV_PROTO, 8 / 12)
    config = replace(
        base,
        cache=CacheConfig(8, BudgetUnit.BLOCKS, 8 / 12, 4),
        prototypes=replace(
            base.prototypes,
            group_size=2,
            max_position_span=48,
            maximum_prototypes_per_head=2,
            source_modalities=("image",),
            budget_planning_mode="one_pass_bounded_anchor_cover",
        ),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
    )
    graph = hf_runtime.build_evidence_graph(full, replace(config.graph, enabled=False))
    forecast_attention = {node_id: 1.0 for node_id in range(len(full.blocks))}
    utilities = hf_runtime.compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node=forecast_attention,
        attention_provenance="synthetic_bounded_anchor_fixture",
        rope_aware=True,
    )
    objective = hf_runtime.BudgetedObjective(graph, utilities, config.selection)
    prompt_attention = {node_id: 1.0 for node_id in range(len(full.blocks))}
    prompt_attention.update({4: 50.0, 7: 100.0})

    anchors = hf_runtime._required_prototype_anchors(
        objective,
        config,
        position_fill_quantum=4,
        prompt_attention_by_node=prompt_attention,
    )

    assert anchors == (4, 7)


def test_bounded_prototype_plan_keeps_prompt_core_and_materializes_at_most_two_clusters() -> None:
    sequence = 48
    key = np.arange(sequence * 4, dtype=np.float32).reshape(1, 1, sequence, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(ModalitySpan(0, sequence, Modality.IMAGE),),
        block_size=4,
        mandatory_logical_positions=(sequence - 1,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    base = _config(MosaicKVMethod.MOSAICKV_PROTO, 8 / 12)
    config = replace(
        base,
        cache=CacheConfig(8, BudgetUnit.BLOCKS, 8 / 12, 4),
        selection=replace(base.selection, prompt_anchor_fraction=0.5),
        prototypes=replace(
            base.prototypes,
            group_size=2,
            max_position_span=48,
            maximum_prototypes_per_head=2,
            source_modalities=("image",),
            budget_planning_mode="one_pass_bounded_anchor_cover",
        ),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
    )
    prompt_attention = {node_id: 1.0 for node_id in range(len(full.blocks))}
    prompt_attention.update({4: 50.0, 7: 100.0})

    plan = build_compression_plan(
        full,
        None,
        prompt_attention,
        _capabilities(safe=True),
        config,
        prompt_attention_by_node=prompt_attention,
    )

    assert plan.construction.mode.value == "three_tier"
    assert len(plan.construction.prototypes) == 2
    assert {4, 7} <= set(plan.selection.selected_node_ids)
    assert len(plan.selection.selected_node_ids) + len(plan.construction.prototypes) <= 8
    represented = {
        node_id
        for prototype in plan.construction.prototypes
        for node_id in prototype.assigned_node_ids
    }
    eligible_unselected = set(range(len(full.blocks))) - set(plan.selection.selected_node_ids)
    assert represented < eligible_unselected
    assert "bounded prototype anchor cover assigned" in plan.construction.reason


def test_prompt_mass_audit_separates_exact_prototype_and_dropped_evidence() -> None:
    full = _full()
    scores = {node_id: float(8 - node_id) for node_id in range(len(full.blocks))}

    audit = hf_runtime._prompt_mass_audit(
        full,
        scores,
        exact_source_node_ids=(0,),
        prototype_source_node_ids=(1,),
    )

    assert audit["status"] == "measured_from_prompt_window_attention"
    assert audit["exact_block_count"] == 1
    assert audit["prototype_source_block_count"] == 1
    assert audit["dropped_block_count"] == 6
    assert audit["exact_prompt_mass_fraction"] == pytest.approx(8 / 36)
    assert audit["represented_prompt_mass_fraction"] == pytest.approx(15 / 36)


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


def test_prompt_exact_core_reserves_capacity_in_every_routed_head() -> None:
    sequence = 4
    key = np.arange(2 * sequence * 4, dtype=np.float32).reshape(1, 2, sequence, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=1,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    config = _config(MosaicKVMethod.MOSAICKV_PROTO, 0.5)
    graph = hf_runtime.build_evidence_graph(full, replace(config.graph, enabled=False))
    utilities = hf_runtime.compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node={node_id: 1.0 for node_id in range(len(full.blocks))},
        attention_provenance="synthetic_per_head_prompt_core",
        rope_aware=True,
    )
    objective = hf_runtime.BudgetedObjective(graph, utilities, config.selection)
    scores = {
        node_id: (100.0 - node_id if block.kv_head == 0 else 1.0 - node_id * 1e-3)
        for node_id, block in enumerate(full.blocks)
    }

    chosen = _bounded_reservation(
        objective,
        8,
        BudgetUnit.BLOCKS,
        0.5,
        scores,
        position_limits={(0, 0): 4, (0, 1): 4},
        per_head=True,
    )

    positions_by_head = {
        head: sum(
            full.blocks[node_id].position_count
            for node_id in chosen
            if full.blocks[node_id].kv_head == head
        )
        for head in (0, 1)
    }
    assert positions_by_head == {0: 2, 1: 2}


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


def test_byte_budget_materialization_trace_separates_cap_lattice_and_padding() -> None:
    full = _full()
    cap = full.active_bytes // 2 + 1
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.75),
        cache=CacheConfig(cap, BudgetUnit.BYTES, 0.75, 1),
    )
    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    payload = hf_runtime._byte_budget_materialization_trace(
        dense,
        config,
        source_prefill_kv_bytes=full.active_bytes,
        logical_prefill_active_kv_bytes=dense.target_bytes - dense.bytes_per_position,
        packed_prefill_active_kv_bytes=dense.target_bytes,
        packed_prompt_capacity=dense.prompt_capacity,
        plan=None,
    )

    assert payload is not None
    assert payload["configured_active_byte_cap"] == cap
    assert payload["requested_hard_budget_bytes"] == cap
    assert payload["dense_attainable_budget_bytes"] == dense.target_bytes
    assert payload["dense_quantization_underfill_bytes"] == 1
    assert payload["logical_underfill_vs_dense_budget_bytes"] == dense.bytes_per_position
    assert payload["packed_padding_bytes"] == dense.bytes_per_position
    assert payload["final_underfill_vs_dense_budget_bytes"] == 0
    assert payload["final_dense_budget_utilization"] == 1.0


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
    effective_target = 5 * dense.bytes_per_position
    assert dense.minimum_block_representable_fill(
        1,
        target_bytes=effective_target,
    ) == effective_target
    with pytest.raises(ValueError, match="within the dense hard cap"):
        dense.minimum_block_representable_fill(
            1,
            target_bytes=dense.target_bytes + dense.bytes_per_position,
        )


def test_common_eight_token_blocks_reduce_dense_lattice_underfill() -> None:
    sequence = 73
    requested_capacity = 52
    key = np.arange(sequence * 4, dtype=np.float32).reshape(1, 1, sequence, 4) + 1
    tensor_layers = ((key, np.flip(key, axis=-1).copy()),)
    states = {
        block_size: FullKVState.from_tensors(
            tensor_layers,
            block_size=block_size,
            mandatory_logical_positions=(sequence - 1,),
            cached_key_state=CachedKeyState.NOT_APPLICABLE,
        )
        for block_size in (16, 8)
    }
    bytes_per_position = states[16].active_bytes // sequence
    budgets = {}
    for block_size, state in states.items():
        base = _config(MosaicKVMethod.MOSAICKV_PROTO, requested_capacity / sequence)
        config = replace(
            base,
            cache=replace(
                base.cache,
                budget_value=requested_capacity * bytes_per_position,
                budget_unit=BudgetUnit.BYTES,
                retention_ratio=requested_capacity / sequence,
                block_size=block_size,
                physical_budget_block_size=block_size,
                physical_budget_mandatory_policy="last_token",
            ),
        )
        budgets[block_size] = _dense_physical_cache_budget(state, config)

    assert budgets[16] is not None
    assert budgets[8] is not None
    assert budgets[16].prompt_capacity == 41
    assert budgets[8].prompt_capacity == 49
    assert budgets[8].target_bytes - budgets[16].target_bytes == (8 * bytes_per_position)


def test_mosaickv_capacity_fraction_scales_every_dense_head_and_byte_budget() -> None:
    sequence = 67
    key = np.arange(2 * sequence * 4, dtype=np.float32).reshape(1, 2, sequence, 4) + 1
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
    base = _config(MosaicKVMethod.MOSAICKV_PROTO, 37 / sequence)
    config = replace(
        base,
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 37 / sequence, 16),
        selection=replace(base.selection, physical_capacity_fraction=0.70),
    )
    bytes_per_position = full.active_bytes // sequence
    shared_target = 37 * bytes_per_position

    effective, limits = _mosaickv_effective_capacity(
        full,
        config,
        shared_target,
        {(0, 0): 37, (0, 1): 37},
    )

    assert limits == {(0, 0): 25, (0, 1): 25}
    assert effective == 25 * bytes_per_position
    assert effective / shared_target < 0.70


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


def test_uniform_prototype_layout_ignores_unrouted_partial_fragment() -> None:
    key = np.arange(44 * 4, dtype=np.float32).reshape(1, 1, 44, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(
            ModalitySpan(0, 16, Modality.IMAGE),
            ModalitySpan(16, 28, Modality.TEXT),
            ModalitySpan(28, 44, Modality.IMAGE),
        ),
        block_size=16,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    graph = hf_runtime.build_evidence_graph(
        full,
        replace(_config(MosaicKVMethod.MOSAICKV_PROTO, 1.0).graph, enabled=False),
    )
    assert tuple(node.block.position_count for node in graph.nodes) == (16, 12, 16)

    exact_limits, final_length = _uniform_attainable_prototype_position_limits(
        graph,
        {(0, 0): 44},
        {},
        block_size=16,
        required_node_ids=(0,),
        prompt_anchor_node_ids=(),
        evidence_node_ids=(),
        optimization_node_ids=(0, 2),
    )

    assert exact_limits == {(0, 0): 32}
    assert final_length == 32


def test_routed_prototype_layout_uses_maskable_per_head_padding() -> None:
    key = np.arange(2 * 64 * 4, dtype=np.float32).reshape(1, 2, 64, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(ModalitySpan(0, 64, Modality.IMAGE),),
        block_size=16,
        mandatory_logical_positions=(63,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    graph = hf_runtime.build_evidence_graph(
        full,
        replace(_config(MosaicKVMethod.MOSAICKV_PROTO, 1.0).graph, enabled=False),
    )

    exact_limits, physical_length = _uniform_attainable_prototype_position_limits(
        graph,
        {(0, 0): 64, (0, 1): 64},
        {(0, 0): 2},
        block_size=16,
        required_node_ids=(),
        prompt_anchor_node_ids=(),
        evidence_node_ids=(),
        allow_head_padding=True,
    )

    assert exact_limits == {(0, 0): 48, (0, 1): 64}
    assert physical_length == 64
    assert exact_limits[(0, 0)] + 2 < physical_length


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


def test_routed_prototype_prompt_reservation_excludes_modality_fragments() -> None:
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
    base = _config(MosaicKVMethod.MOSAICKV_PROTO, 0.75)
    config = replace(
        base,
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.75, 16),
        selection=replace(base.selection, prompt_anchor_fraction=0.5),
        prototypes=replace(
            base.prototypes,
            group_size=31,
            max_position_span=128,
            source_modalities=("image",),
            maximum_prototypes_per_head=2,
        ),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
    )
    dense = _dense_physical_cache_budget(full, config)
    assert dense is not None
    attention = {
        node_id: (100.0 if block.position_count < 16 else float(len(full.blocks) - node_id))
        for node_id, block in enumerate(full.blocks)
    }

    plan = build_compression_plan(
        full,
        _forecast(),
        attention,
        _capabilities(safe=True),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
        prompt_attention_by_node=attention,
    )

    assert config.prototypes.source_modalities == ("image",)
    assert config.prototypes.maximum_prototypes_per_head == 2
    assert all(
        block.mandatory or block.position_count == full.block_size
        for block in plan.selection.selected_blocks
    )
    assert 0 not in plan.selection.selected_node_ids


def test_mixed_head_routed_prototypes_pack_with_auditable_validity_masks() -> None:
    sequence = 96
    key = torch.arange(2 * sequence * 4, dtype=torch.float32).reshape(1, 2, sequence, 4) + 1
    full = FullKVState.from_tensors(
        ((key, key.flip(-1)),),
        modality_spans=(
            ModalitySpan(0, 85, Modality.IMAGE),
            ModalitySpan(85, 96, Modality.TEXT),
        ),
        block_size=16,
        mandatory_logical_positions=(95,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    base = _config(MosaicKVMethod.MOSAICKV_PROTO, 75 / sequence)
    config = replace(
        base,
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 75 / sequence, 16),
        selection=replace(base.selection, prompt_anchor_fraction=0.75),
        prototypes=replace(
            base.prototypes,
            group_size=31,
            max_position_span=2048,
            source_modalities=("image",),
            selection_only_head_routes=("0:1",),
            maximum_prototypes_per_head=2,
        ),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
    )
    dense = _dense_physical_cache_budget(full, config)
    assert dense is not None and dense.prompt_capacity == 75
    attention = {node_id: float(len(full.blocks) - node_id) for node_id in range(len(full.blocks))}

    plan = build_compression_plan(
        full,
        None,
        attention,
        _capabilities(safe=True),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
        prompt_attention_by_node=attention,
    )

    assert plan.construction.mode.value == "three_tier"
    assert plan.construction.prototypes
    lengths = {
        (layout.layer, layout.kv_head): layout.active_cache_length
        for layout in plan.construction.active_layouts
    }
    assert lengths[(0, 0)] < lengths[(0, 1)]
    assert max(lengths.values()) <= dense.prompt_capacity
    adapter = SimpleNamespace(
        capabilities=SimpleNamespace(cached_key_state=CachedKeyState.NOT_APPLICABLE)
    )
    packed = hf_runtime.pack_runtime_cache(
        adapter,
        full,
        plan.construction.state,
        plan.construction.prototypes,
        tuple(record.prototype_id for record in plan.construction.prototypes),
        physical_prompt_capacity=max(lengths.values()),
        packing_mode="preallocated_layer",
    )

    assert not packed.all_slots_valid
    assert packed.prompt_capacity == max(lengths.values())
    assert tuple(packed.validity_masks[0].sum(dim=-1).tolist()) == (
        lengths[(0, 0)],
        lengths[(0, 1)],
    )
    assert not bool(packed.validity_masks[0][0, -1])
    assert bool(packed.validity_masks[0][1, -1])


def test_combined_routed_75_and_external_64_layout_uses_planned_padding() -> None:
    key = torch.arange(2 * 80 * 2, dtype=torch.float32).reshape(1, 2, 80, 2)
    full = FullKVState.from_tensors(
        ((key, key.flip(-1)),),
        block_size=16,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    external = hf_runtime.HeadLocalCapacityRecord(
        layer=0,
        kv_head=1,
        position_limit=75,
        selected_positions=64,
        shortfall_positions=11,
        selected_bytes=64 * 2 * 2 * key.element_size(),
    )

    maximum, distinct, padding = _validate_combined_active_layouts(
        {(0, 0): 75, (0, 1): 64},
        {(0, 0): 75},
        {(0, 0): 75},
        (external,),
    )

    assert maximum == 75
    assert distinct == {64, 75}
    assert padding
    payloads = {
        (0, head): tuple(
            (
                position,
                "exact",
                position,
                key[:, head : head + 1, position : position + 1, :],
                key[:, head : head + 1, position : position + 1, :].flip(-1),
            )
            for position in range(count)
        )
        for head, count in ((0, 75), (1, 64))
    }
    adapter = SimpleNamespace(
        capabilities=SimpleNamespace(cached_key_state=CachedKeyState.NOT_APPLICABLE)
    )
    packed = pack_runtime_payloads(
        adapter,
        full,
        payloads,
        (),
        physical_prompt_capacity=75,
        packing_mode="preallocated_layer",
    )

    assert packed.prompt_capacity == 75
    assert not packed.all_slots_valid
    assert tuple(packed.validity_masks[0].sum(dim=-1).tolist()) == (75, 64)
    assert bool(packed.validity_masks[0][0, -1])
    assert not bool(packed.validity_masks[0][1, -1])


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


def test_short_horizon_logit_gate_is_thresholded_and_fail_closed() -> None:
    passed = _short_horizon_candidate_evaluation(
        "pareto",
        kl_values=(0.01, 0.02),
        maximum_logit_differences=(0.5, 1.0),
        greedy_token_matches=(True, True),
        maximum_kl_threshold=0.02,
        maximum_logit_difference_threshold=1.0,
    )
    token_changed = _short_horizon_candidate_evaluation(
        "unsafe_token",
        kl_values=(0.0, 0.0),
        maximum_logit_differences=(0.0, 0.0),
        greedy_token_matches=(True, False),
        maximum_kl_threshold=0.02,
        maximum_logit_difference_threshold=1.0,
    )
    excessive_kl = _short_horizon_candidate_evaluation(
        "unsafe_kl",
        kl_values=(0.01, 0.0201),
        maximum_logit_differences=(0.5, 0.5),
        greedy_token_matches=(True, True),
        maximum_kl_threshold=0.02,
        maximum_logit_difference_threshold=1.0,
    )

    assert passed.passed
    assert passed.evaluated_steps == 2
    assert passed.maximum_kl == pytest.approx(0.02)
    assert passed.maximum_logit_difference == pytest.approx(1.0)
    assert not token_changed.passed
    assert not token_changed.greedy_token_match
    assert not excessive_kl.passed


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


def test_prompt_query_attention_retains_window_positions_and_pools_to_blocks() -> None:
    key = np.arange(1 * 2 * 4, dtype=np.float32).reshape(1, 2, 4, 1)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        block_size=2,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    attention = torch.arange(1, 1 + 4 * 3 * 4, dtype=torch.float32).reshape(1, 4, 3, 4)
    prefill = SimpleNamespace(attention_weights=(attention,))

    by_query = hf_runtime._prompt_query_token_attention_by_head(
        full,
        prefill,
        prompt_window=2,
    )
    averaged = hf_runtime._prompt_token_attention_by_head(
        full,
        prefill,
        prompt_window=2,
    )
    by_block = hf_runtime._block_query_attention_from_token_attention(
        full,
        by_query,
    )
    scalar_blocks = hf_runtime._block_attention_from_token_attention(
        full,
        averaged,
    )

    assert set(by_query) == {(0, 0), (0, 1)}
    assert all(values.shape == (2, 4) for values in by_query.values())
    for identity in by_query:
        np.testing.assert_allclose(by_query[identity].mean(axis=0), averaged[identity])
    for node_id, values in by_block.items():
        assert values.shape == (2,)
        assert float(values.mean()) == pytest.approx(scalar_blocks[node_id])
