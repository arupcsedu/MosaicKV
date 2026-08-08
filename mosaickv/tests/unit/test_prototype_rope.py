from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from mosaickv.adapters.huggingface import Qwen25VLAdapter  # noqa: E402
from mosaickv.adapters.huggingface.types import CachedKeyState  # noqa: E402
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan  # noqa: E402
from mosaickv.config import (  # noqa: E402
    PrototypeConfig,
    ResidualConfig,
    SelectionConfig,
    UtilityConfig,
)
from mosaickv.graph import EdgeType, GraphDiagnostics, SparseEvidenceGraph  # noqa: E402
from mosaickv.graph.pooling import pool_block_descriptors  # noqa: E402
from mosaickv.prototypes import (  # noqa: E402
    TierConstructionMode,
    construct_three_tier_cache,
    refresh_prototype_after_promotions,
)
from mosaickv.prototypes.rope import RotaryPrototypeKeyTransform  # noqa: E402
from mosaickv.selection import (  # noqa: E402
    BudgetedObjective,
    SelectionBudget,
    compute_block_utilities,
    lazy_greedy_select,
)
from mosaickv.types import BudgetUnit  # noqa: E402


def _phase(sequence: int, dimension: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.linspace(0.0, 0.9, sequence * (dimension // 2), dtype=torch.float32).reshape(
        sequence, dimension // 2
    )
    angles = torch.cat((base, base), dim=-1).reshape(1, 1, sequence, dimension)
    return angles.cos(), angles.sin()


def _rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = int(value.shape[-1]) // 2
    rotated_half = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cos + rotated_half * sin


def test_rotary_key_transform_recovers_phase_neutral_weighted_mean() -> None:
    neutral = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 1.0, 5.0, 2.0],
            [7.0, 2.0, 1.0, 6.0],
        ],
        dtype=torch.float32,
    ).reshape(1, 1, 4, 4)
    cos, sin = _phase(4)
    post_rope = _rotate(neutral, cos, sin)
    full = FullKVState.from_tensors(
        ((post_rope, torch.zeros_like(post_rope)),),
        modality_spans=(ModalitySpan(0, 4, Modality.TEXT),),
        block_size=2,
        cached_key_state=CachedKeyState.POST_ROPE,
    )
    gathered = full.gather_exact_blocks(full.blocks)
    transform = RotaryPrototypeKeyTransform(cos, sin, "synthetic_exact_phase_v1")

    merged = transform.merge(
        gathered.key_blocks,
        full.blocks,
        (1.0, 3.0),
        sequence_dimension=-2,
        anchor_logical_position=0,
    )

    expected_neutral = neutral[..., :2, :].mean(dim=-2, keepdim=True) * 0.25
    expected_neutral += neutral[..., 2:, :].mean(dim=-2, keepdim=True) * 0.75
    expected = _rotate(expected_neutral, cos[..., :1, :], sin[..., :1, :])
    assert torch.allclose(merged.average_key_fp32, expected_neutral, atol=1e-5, rtol=0)
    assert torch.allclose(merged.prototype_key, expected, atol=1e-5, rtol=0)


def test_batched_anchor_phase_matches_individual_transform() -> None:
    cos, sin = _phase(6)
    transform = RotaryPrototypeKeyTransform(cos, sin, "synthetic_batched_phase_v1")
    neutral = torch.tensor(
        [
            [[[1.0, 2.0, 3.0, 4.0]]],
            [[[4.0, -1.0, 2.0, 0.5]]],
            [[[-2.0, 3.0, 0.25, 7.0]]],
        ],
        dtype=torch.float32,
    ).unsqueeze(1)
    reference = neutral[0].to(torch.float16)
    positions = (0, 3, 5)

    batched = transform.apply_anchor_phase_batch(neutral, positions, reference)
    individual = torch.stack(
        tuple(
            transform.apply_anchor_phase(neutral[index], position, reference)
            for index, position in enumerate(positions)
        )
    )

    assert batched.dtype == reference.dtype
    assert torch.equal(batched, individual)


def test_qwen_post_rope_construction_requires_and_accepts_exact_phase_transform() -> None:
    neutral = torch.tensor(
        [[[[-2.0, 1.0, 2.0, 3.0], [4.0, 2.0, -1.0, 5.0], [6.0, 7.0, 1.0, 2.0]]]]
    ).reshape(1, 1, 3, 4)
    cos, sin = _phase(3)
    post_rope = _rotate(neutral, cos, sin)
    full = FullKVState.from_tensors(
        ((post_rope, neutral.clone()),),
        modality_spans=(ModalitySpan(0, 3, Modality.TEXT),),
        block_size=1,
        cached_key_state=CachedKeyState.POST_ROPE,
    )
    nodes = pool_block_descriptors(full)
    graph = SparseEvidenceGraph(
        nodes,
        (1, 2),
        (0, 0),
        (0.25, 0.75),
        (EdgeType.SEMANTIC_SIMILARITY, EdgeType.SEMANTIC_SIMILARITY),
        GraphDiagnostics(
            node_count=3,
            edge_count=2,
            connected_components=1,
            modality_mixing=0.0,
            average_degree=2 / 3,
            evidence_cluster_coverage=None,
            edge_counts=((EdgeType.SEMANTIC_SIMILARITY, 2),),
            maximum_out_degree=1,
            fallback_used=False,
        ),
    )
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node=(0.98, 0.01, 0.01),
        attention_provenance="synthetic_post_rope_attention",
        rope_aware=True,
    )
    selection = lazy_greedy_select(
        BudgetedObjective(
            graph,
            utilities,
            SelectionConfig(lambda_g=0.0, lambda_m=0.0),
        ),
        SelectionBudget(1, BudgetUnit.BLOCKS),
    )
    # One exact anchor plus one prototype is the active budget. The selector
    # intentionally chose only the anchor so both remaining members can be
    # checked against the hand-computed weighted merge.
    selection = replace(selection, budget=SelectionBudget(2, BudgetUnit.BLOCKS))
    without_transform = construct_three_tier_cache(
        full,
        graph,
        selection,
        Qwen25VLAdapter.capabilities,
    )
    with_transform = construct_three_tier_cache(
        full,
        graph,
        selection,
        Qwen25VLAdapter.capabilities,
        prototype_config=PrototypeConfig(group_size=2),
        residual_config=ResidualConfig(enabled=False),
        key_transform=RotaryPrototypeKeyTransform(cos, sin, "synthetic_qwen_mrope_v1"),
    )

    assert without_transform.mode is TierConstructionMode.EXACT_ONLY_UNSAFE
    assert with_transform.mode is TierConstructionMode.THREE_TIER
    assert with_transform.safety.phase_transform_provenance == "synthetic_qwen_mrope_v1"
    expected_neutral = neutral[..., 1:2, :] * 0.25 + neutral[..., 2:3, :] * 0.75
    expected = _rotate(expected_neutral, cos[..., :1, :], sin[..., :1, :])
    assert torch.allclose(
        with_transform.state.prototypes.prototype_keys[0], expected, atol=1e-5, rtol=0
    )

    promoted = full.gather_exact_blocks((full.blocks[2],))
    refreshed = refresh_prototype_after_promotions(
        full,
        with_transform.prototypes[0],
        with_transform.state.prototypes.prototype_keys[0],
        with_transform.state.prototypes.prototype_values[0],
        {2: (promoted.key_blocks[0], promoted.value_blocks[0])},
        key_transform=RotaryPrototypeKeyTransform(cos, sin, "synthetic_qwen_mrope_v1"),
    )
    expected_remaining_key = _rotate(
        neutral[..., 1:2, :],
        cos[..., :1, :],
        sin[..., :1, :],
    )
    assert refreshed.record.assigned_node_ids == (1,)
    assert torch.allclose(refreshed.key, expected_remaining_key, atol=1e-4, rtol=0)
    assert torch.allclose(refreshed.value, neutral[..., 1:2, :], atol=1e-5, rtol=0)


def test_partial_repair_keeps_weighted_medoid_anchor_identity() -> None:
    neutral = torch.tensor(
        [[[[-2.0, 1.0, 2.0, 3.0], [4.0, 2.0, -1.0, 5.0], [6.0, 7.0, 1.0, 2.0]]]],
        dtype=torch.float32,
    )
    cos, sin = _phase(3)
    post_rope = _rotate(neutral, cos, sin)
    full = FullKVState.from_tensors(
        ((post_rope, neutral.clone()),),
        modality_spans=(ModalitySpan(0, 3, Modality.TEXT),),
        block_size=1,
        cached_key_state=CachedKeyState.POST_ROPE,
    )
    nodes = pool_block_descriptors(full)
    graph = SparseEvidenceGraph(
        nodes,
        (1, 2),
        (0, 0),
        (0.25, 0.75),
        (EdgeType.SEMANTIC_SIMILARITY, EdgeType.SEMANTIC_SIMILARITY),
        GraphDiagnostics(
            node_count=3,
            edge_count=2,
            connected_components=1,
            modality_mixing=0.0,
            average_degree=2 / 3,
            evidence_cluster_coverage=None,
            edge_counts=((EdgeType.SEMANTIC_SIMILARITY, 2),),
            maximum_out_degree=1,
            fallback_used=False,
        ),
    )
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node=(0.98, 0.01, 0.01),
        attention_provenance="synthetic_medoid_repair",
        rope_aware=True,
    )
    selection = lazy_greedy_select(
        BudgetedObjective(graph, utilities, SelectionConfig(lambda_g=0.0, lambda_m=0.0)),
        SelectionBudget(1, BudgetUnit.BLOCKS),
    )
    selection = replace(selection, budget=SelectionBudget(2, BudgetUnit.BLOCKS))
    transform = RotaryPrototypeKeyTransform(cos, sin, "synthetic_medoid_repair_v1")
    construction = construct_three_tier_cache(
        full,
        graph,
        selection,
        Qwen25VLAdapter.capabilities,
        prototype_config=PrototypeConfig(
            group_size=2,
            pooling_mode="forecast_attention_token",
            attention_bias_basis="forecast_attention_entropy",
            anchor_position_strategy="forecast_weighted_member_medoid",
        ),
        residual_config=ResidualConfig(enabled=False),
        key_transform=transform,
        prototype_token_weight_by_head={(0, 0): torch.tensor([0.01, 0.25, 0.75])},
    )
    record = construction.prototypes[0]
    assert record.anchor_logical_position == 2

    promoted = full.gather_exact_blocks((full.blocks[2],))
    refreshed = refresh_prototype_after_promotions(
        full,
        record,
        construction.state.prototypes.prototype_keys[0],
        construction.state.prototypes.prototype_values[0],
        {2: (promoted.key_blocks[0], promoted.value_blocks[0])},
        key_transform=transform,
    )

    assert refreshed.record.assigned_node_ids == (1,)
    assert refreshed.record.anchor_node_id == record.anchor_node_id
    assert refreshed.record.anchor_logical_position == record.anchor_logical_position
    expected = _rotate(neutral[..., 1:2, :], cos[..., 2:3, :], sin[..., 2:3, :])
    assert torch.allclose(refreshed.key, expected, atol=1e-4, rtol=0)
