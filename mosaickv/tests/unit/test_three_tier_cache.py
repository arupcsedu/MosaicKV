from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mosaickv.adapters.huggingface import (
    HuggingFaceMultimodalAdapter,
    InternVL25Adapter,
    Llava15Adapter,
    LlavaOneVisionAdapter,
    Qwen25VLAdapter,
)
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    QueryVectorState,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import PrototypeConfig, ResidualConfig, SelectionConfig, UtilityConfig
from mosaickv.graph import EdgeType, GraphDiagnostics, SparseEvidenceGraph
from mosaickv.graph.pooling import pool_block_descriptors
from mosaickv.prototypes import (
    PrototypeConstructionError,
    TierConstructionMode,
    construct_three_tier_cache,
)
from mosaickv.residual import (
    PinnedMemoryUnavailableError,
    ResidualStorageError,
    build_residual_storage,
    restore_residual_payload,
)
from mosaickv.selection import (
    BudgetedObjective,
    SelectionBudget,
    SelectionResult,
    compute_block_utilities,
    lazy_greedy_select,
)
from mosaickv.types import BudgetUnit, ResidualStorageDType


def _full(
    key_values: tuple[float, ...] = (10.0, 20.0, 30.0),
    value_values: tuple[float, ...] = (1.0, 3.0, 7.0),
    *,
    cached_key_state: CachedKeyState | str = CachedKeyState.NOT_APPLICABLE,
    logical_positions: tuple[int, ...] | None = None,
    original_length: int | None = None,
    next_decode_position: int | None = None,
) -> FullKVState:
    key = np.asarray(key_values, dtype=np.float16).reshape(1, 1, -1, 1)
    value = np.asarray(value_values, dtype=np.float16).reshape(1, 1, -1, 1)
    logical = logical_positions or tuple(range(len(key_values)))
    source_length = original_length or len(key_values)
    return FullKVState.from_tensors(
        ((key, value),),
        modality_spans=(ModalitySpan(0, source_length, Modality.TEXT),),
        block_size=1,
        logical_positions=logical,
        original_logical_sequence_length=source_length,
        next_decode_position=next_decode_position,
        cached_key_state=cached_key_state,
    )


def _graph(
    full: FullKVState,
    edges: tuple[tuple[int, int, float], ...] = ((1, 0, 0.25), (2, 0, 0.75)),
) -> SparseEvidenceGraph:
    nodes = pool_block_descriptors(full)
    return SparseEvidenceGraph(
        nodes=nodes,
        row_indices=tuple(edge[0] for edge in edges),
        column_indices=tuple(edge[1] for edge in edges),
        weights=tuple(edge[2] for edge in edges),
        edge_types=tuple(EdgeType.SEMANTIC_SIMILARITY for _edge in edges),
        diagnostics=GraphDiagnostics(
            node_count=len(nodes),
            edge_count=len(edges),
            connected_components=1,
            modality_mixing=0.0,
            average_degree=len(edges) / len(nodes),
            evidence_cluster_coverage=None,
            edge_counts=((EdgeType.SEMANTIC_SIMILARITY, len(edges)),),
            maximum_out_degree=max(
                (
                    sum(source == node_id for source, _target, _weight in edges)
                    for node_id in range(len(nodes))
                ),
                default=0,
            ),
            fallback_used=False,
        ),
    )


def _selection(
    graph: SparseEvidenceGraph,
    budget: int,
    probabilities: tuple[float, ...],
) -> SelectionResult:
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node=probabilities,
        attention_provenance="synthetic_rope_safe_fixture",
        rope_aware=True,
    )
    objective = BudgetedObjective(
        graph,
        utilities,
        SelectionConfig(lambda_g=0.0, lambda_m=0.0),
    )
    return lazy_greedy_select(objective, SelectionBudget(budget, BudgetUnit.BLOCKS))


def _safe_capabilities() -> AdapterCapabilities:
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


def test_manual_cluster_matches_expected_fp32_weighted_prototype() -> None:
    full = _full(logical_positions=(2, 4, 6), original_length=8, next_decode_position=9)
    graph = _graph(full)
    selection = _selection(graph, 2, (1.0, 0.0, 0.0))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        prototype_config=PrototypeConfig(group_size=2, max_position_span=8),
        residual_config=ResidualConfig(enabled=False),
    )

    assert result.mode is TierConstructionMode.THREE_TIER
    assert result.exact_node_ids == (0,)
    assert len(result.prototypes) == 1
    record = result.prototypes[0]
    assert record.anchor_node_id == 0
    assert record.assigned_node_ids == (1, 2)
    assert [member.raw_weight for member in record.members] == pytest.approx([0.25, 0.75])
    assert [member.normalized_weight for member in record.members] == pytest.approx([0.25, 0.75])
    prototype_key = result.state.prototypes.prototype_keys[0]
    prototype_value = result.state.prototypes.prototype_values[0]
    assert prototype_key.dtype == np.float16
    assert prototype_value.dtype == np.float16
    assert float(prototype_key.reshape(-1)[0]) == pytest.approx(27.5)
    assert float(prototype_value.reshape(-1)[0]) == pytest.approx(6.0)
    assert record.diagnostics.cluster_size == 2
    assert record.diagnostics.modality_composition == ((Modality.TEXT, 2),)
    assert record.diagnostics.position_span == 2
    assert record.diagnostics.active_bytes_saved == full.blocks[1].byte_size
    assert record.diagnostics.key_dispersion == pytest.approx(18.75)
    assert record.diagnostics.value_dispersion == pytest.approx(3.0)
    assert result.original_logical_sequence_length == 8
    assert result.next_decode_position == 9
    assert result.active_layouts[0].active_cache_length == 2
    assert result.active_layouts[0].prototype_anchor_positions == (2,)


def test_prototype_pools_each_multitoken_block_before_weighting() -> None:
    key = np.asarray([0.0, 2.0, 10.0, 14.0, 20.0, 24.0], dtype=np.float32).reshape(1, 1, 6, 1)
    value = (key + 100).copy()
    full = FullKVState.from_tensors(
        ((key, value),),
        block_size=2,
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    graph = _graph(full, ((1, 0, 0.25), (2, 0, 0.75)))
    selection = _selection(graph, 2, (1.0, 0.0, 0.0))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        prototype_config=PrototypeConfig(group_size=2),
        residual_config=ResidualConfig(enabled=False),
    )

    # Member block means are 12 and 22; graph weighting is 0.25 / 0.75.
    assert float(result.state.prototypes.prototype_keys[0].item()) == pytest.approx(19.5)
    assert float(result.state.prototypes.prototype_values[0].item()) == pytest.approx(119.5)
    assert result.prototypes[0].diagnostics.active_bytes_saved == 24


def test_lossless_residuals_preserve_and_index_every_original_position() -> None:
    full = _full()
    graph = _graph(full)
    selection = _selection(graph, 2, (1.0, 0.0, 0.0))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        prototype_config=PrototypeConfig(group_size=2),
        residual_config=ResidualConfig(
            storage_dtype=ResidualStorageDType.LOSSLESS,
            require_pinned_memory=False,
        ),
    )

    report = result.residual_storage
    assert len(report.payloads) == 2
    assert len(report.index) == 2
    assert {
        (item.layer, item.kv_head, item.prototype_id, item.original_position)
        for item in report.index
    } == {
        (0, 0, 0, 1),
        (0, 0, 0, 2),
    }
    assert report.cpu_bytes == sum(full.blocks[node_id].byte_size for node_id in (1, 2))
    assert report.lookup(0, 0, 0, 2).payload_index == 1
    with pytest.raises(KeyError, match="residual position does not exist"):
        report.lookup(0, 0, 0, 99)
    assert result.state.statistics.active_kv_bytes == (
        result.state.exact.active_bytes + result.state.prototypes.active_bytes
    )
    assert result.state.statistics.total_stored_bytes == (
        result.active_device_bytes + result.cpu_residual_bytes
    )
    for payload_index, node_id in enumerate((1, 2)):
        restored_key, restored_value = restore_residual_payload(report, payload_index, full)
        expected = full.gather_exact_blocks((full.blocks[node_id],))
        assert np.array_equal(restored_key, expected.key_blocks[0])
        assert np.array_equal(restored_value, expected.value_blocks[0])


@pytest.mark.parametrize(
    "storage_dtype",
    (ResidualStorageDType.FP16, ResidualStorageDType.INT8),
)
def test_numpy_residual_encodings_round_trip(storage_dtype: ResidualStorageDType) -> None:
    full = _full()
    report = build_residual_storage(
        full,
        {1: 0},
        ResidualConfig(storage_dtype=storage_dtype, require_pinned_memory=False),
    )

    restored_key, restored_value = restore_residual_payload(report, 0, full)
    expected = full.gather_exact_blocks((full.blocks[1],))
    tolerance = 0.05 if storage_dtype is ResidualStorageDType.INT8 else 0.0
    assert np.allclose(restored_key, expected.key_blocks[0], atol=tolerance)
    assert np.allclose(restored_value, expected.value_blocks[0], atol=tolerance)


@pytest.mark.parametrize("storage_dtype", (ResidualStorageDType.BF16, ResidualStorageDType.FP8))
def test_numpy_rejects_encodings_it_cannot_represent(
    storage_dtype: ResidualStorageDType,
) -> None:
    with pytest.raises(ResidualStorageError, match="requires a compatible PyTorch build"):
        build_residual_storage(
            _full(),
            {1: 0},
            ResidualConfig(storage_dtype=storage_dtype, require_pinned_memory=False),
        )


def test_pinned_memory_requirement_fails_closed_for_numpy() -> None:
    with pytest.raises(PinnedMemoryUnavailableError, match="requires torch tensors"):
        build_residual_storage(_full(), {1: 0}, ResidualConfig(require_pinned_memory=True))


@pytest.mark.parametrize(
    "adapter",
    (Llava15Adapter, Qwen25VLAdapter, LlavaOneVisionAdapter, InternVL25Adapter),
)
def test_current_supported_adapters_fall_back_to_exact_only(
    adapter: type[HuggingFaceMultimodalAdapter],
) -> None:
    full = _full(cached_key_state=CachedKeyState.POST_ROPE)
    graph = _graph(full)
    selection = _selection(graph, 1, (0.98, 0.01, 0.01))

    result = construct_three_tier_cache(full, graph, selection, adapter.capabilities)

    assert result.mode is TierConstructionMode.EXACT_ONLY_UNSAFE
    assert not result.safety.safe
    assert result.safety.source_cached_key_state is CachedKeyState.POST_ROPE
    assert result.state.exact.blocks == selection.selected_blocks
    assert not result.state.prototypes.source_blocks
    assert not result.state.residuals.source_blocks


def test_post_rope_merge_is_rejected_even_if_adapter_claims_support() -> None:
    full = _full(cached_key_state=CachedKeyState.POST_ROPE)
    graph = _graph(full)
    selection = _selection(graph, 1, (0.98, 0.01, 0.01))
    capabilities = replace(
        _safe_capabilities(),
        cached_key_state=CachedKeyState.POST_ROPE,
    )

    result = construct_three_tier_cache(full, graph, selection, capabilities)

    assert result.mode is TierConstructionMode.EXACT_ONLY_UNSAFE
    assert "never averages different RoPE phases" in result.reason


def test_unknown_source_rope_state_fails_closed() -> None:
    full = _full(cached_key_state="unverified")
    graph = _graph(full)
    selection = _selection(graph, 1, (0.98, 0.01, 0.01))

    result = construct_three_tier_cache(full, graph, selection, _safe_capabilities())

    assert result.mode is TierConstructionMode.EXACT_ONLY_UNSAFE
    assert result.safety.source_cached_key_state is CachedKeyState.UNKNOWN
    assert "unknown" in result.reason


def test_explicit_pre_rope_adapter_can_construct_prototypes() -> None:
    full = _full(cached_key_state=CachedKeyState.PRE_ROPE)
    graph = _graph(full)
    selection = _selection(graph, 2, (1.0, 0.0, 0.0))
    capabilities = replace(_safe_capabilities(), cached_key_state=CachedKeyState.PRE_ROPE)

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        capabilities,
        residual_config=ResidualConfig(enabled=False),
    )

    assert result.mode is TierConstructionMode.THREE_TIER
    assert result.safety.safe


def test_highest_weight_compatible_selected_anchor_wins() -> None:
    full = _full()
    graph = _graph(full, ((2, 0, 0.2), (2, 1, 0.9)))
    selection = _selection(graph, 3, (0.5, 0.5, 0.0))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        prototype_config=PrototypeConfig(group_size=1),
        residual_config=ResidualConfig(enabled=False),
    )

    assert result.exact_node_ids == (0, 1)
    assert result.prototypes[0].anchor_node_id == 1
    assert result.prototypes[0].members[0].raw_weight == pytest.approx(0.9)


def test_incompatible_assignment_falls_back_without_partial_prototypes() -> None:
    full = _full()
    graph = _graph(full)
    selection = _selection(graph, 1, (0.98, 0.01, 0.01))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        prototype_config=PrototypeConfig(max_position_span=0),
    )

    assert result.mode is TierConstructionMode.EXACT_ONLY_INCOMPATIBLE
    assert not result.prototypes
    assert not result.state.residuals.source_blocks


def test_prototypes_never_cross_kv_heads() -> None:
    key = np.asarray([1.0, 2.0], dtype=np.float32).reshape(1, 2, 1, 1)
    value = (key + 10).copy()
    full = FullKVState.from_tensors(((key, value),), block_size=1)
    graph = _graph(full, ((1, 0, 1.0),))
    selection = _selection(graph, 1, (1.0, 1.0))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        residual_config=ResidualConfig(enabled=False),
    )

    assert result.mode is TierConstructionMode.EXACT_ONLY_INCOMPATIBLE
    assert "no compatible selected graph anchor" in result.reason
    assert len(result.active_layouts) == 2
    assert result.active_layouts[1].active_cache_length == 0


def test_retention_one_bypasses_all_transformed_tiers() -> None:
    full = _full()
    graph = _graph(full)
    selection = _selection(graph, 3, (1.0, 1.0, 1.0))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        residual_config=ResidualConfig(require_pinned_memory=True),
        retention_ratio=1.0,
    )

    assert result.mode is TierConstructionMode.RETENTION_ONE
    assert result.state.is_retention_one
    assert not result.prototypes
    assert not result.residual_storage.payloads
    reconstructed = result.state.reconstruct_full_state(full)
    assert np.array_equal(reconstructed.layers[0].key, full.layers[0].key)
    assert np.array_equal(reconstructed.layers[0].value, full.layers[0].value)

    incomplete = _selection(graph, 1, (0.98, 0.01, 0.01))
    with pytest.raises(PrototypeConstructionError, match="requires every source block"):
        construct_three_tier_cache(
            full,
            graph,
            incomplete,
            _safe_capabilities(),
            retention_ratio=1.0,
        )


def test_selection_block_budget_rejects_prototype_growth() -> None:
    full = _full()
    graph = _graph(full)
    selection = _selection(graph, 1, (0.98, 0.01, 0.01))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        residual_config=ResidualConfig(enabled=False),
    )

    assert result.mode is TierConstructionMode.EXACT_ONLY_INCOMPATIBLE
    assert "active blocks" in result.reason


def test_explicit_byte_budget_rejects_prototype_growth() -> None:
    full = _full()
    graph = _graph(full)
    selection = _selection(graph, 2, (1.0, 0.0, 0.0))

    result = construct_three_tier_cache(
        full,
        graph,
        selection,
        _safe_capabilities(),
        residual_config=ResidualConfig(enabled=False),
        active_byte_budget=selection.active_bytes,
    )

    assert result.mode is TierConstructionMode.EXACT_ONLY_INCOMPATIBLE
    assert result.active_device_bytes <= selection.active_bytes
    assert "active bytes" in result.reason

    with pytest.raises(PrototypeConstructionError, match="exact blocks already exceed"):
        construct_three_tier_cache(
            full,
            graph,
            selection,
            _safe_capabilities(),
            residual_config=ResidualConfig(enabled=False),
            active_byte_budget=selection.active_bytes - 1,
        )


def test_torch_bfloat16_and_fp8_residual_storage_when_available() -> None:
    torch = pytest.importorskip("torch")
    key = torch.tensor([[[[1.0], [2.0]]]], dtype=torch.float32)
    value = torch.tensor([[[[3.0], [4.0]]]], dtype=torch.float32)
    full = FullKVState.from_tensors(((key, value),), block_size=1)
    formats = [ResidualStorageDType.BF16]
    if hasattr(torch, "float8_e4m3fn"):
        formats.append(ResidualStorageDType.FP8)
    for storage_dtype in formats:
        report = build_residual_storage(
            full,
            {1: 0},
            ResidualConfig(storage_dtype=storage_dtype, require_pinned_memory=False),
        )
        restored_key, restored_value = restore_residual_payload(report, 0, full)
        assert torch.allclose(restored_key, key[:, :, 1:2], atol=0.25, rtol=0)
        assert torch.allclose(restored_value, value[:, :, 1:2], atol=0.25, rtol=0)
