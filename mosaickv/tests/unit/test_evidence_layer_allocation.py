"""Synthetic CPU fixtures; no pretrained-quality or measured-results claims."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import BoundedDecodeConfig, ConfigurationError, canonical_config, load_config
from mosaickv.selection.evidence_layer_allocation import (
    allocate_layer_slots,
    build_evidence_layer_plan,
    evidence_ranking,
)


def test_layer_allocator_spends_exact_bytes_and_responds_to_absolute_impact():
    counts = allocate_layer_slots(((1.0,) * 8, (10.0,) * 8), (2, 2), (6, 6), 8)
    assert counts == (2, 6)
    assert sum(counts) == 8
    assert allocate_layer_slots(((10.0,) * 8, (1.0,) * 8), (2, 2), (6, 6), 8) == (6, 2)
    assert allocate_layer_slots(((1.0,) * 8,) * 2, (2, 2), (6, 6), 8) == (6, 2)
    with pytest.raises(ValueError, match="exact global"):
        allocate_layer_slots(((1.0,) * 8,) * 2, (2, 2), (6, 6), 13)


def test_ranking_protects_mandatory_and_reserves_prompt_and_visual_evidence():
    score = np.array([0.0, 9, 8, 7, 6, 5, 4, 3, 2, 0.1])
    rank = evidence_ranking(score, (0, 9), (1, 2), ((3, 4), (5, 6)), 8)
    assert rank[:4] == (0, 9, 1, 3)
    assert sorted(rank) == list(range(10))
    assert evidence_ranking(score, (0, 9), (1, 2), ((3, 4),), 8) == rank


def test_plan_uses_no_labels_keeps_source_unchanged_and_retains_mandatory():
    key = torch.arange(32, dtype=torch.float32).reshape(1, 2, 8, 2)
    layers = ((key.clone(), key.clone()), (key.clone(), key.clone() * 10))
    saved = [(k.clone(), v.clone()) for k, v in layers]
    state = FullKVState.from_tensors(
        layers,
        block_size=1,
        mandatory_logical_positions=(0, 7),
        modality_spans=(
            ModalitySpan(0, 2, Modality.TEXT),
            ModalitySpan(2, 6, Modality.IMAGE, image_index=0),
            ModalitySpan(6, 8, Modality.TEXT),
        ),
    )
    attention = {(layer, head): np.ones(8) / 8 for layer in range(2) for head in range(2)}
    records = tuple(
        dict(layer=layer, kv_head=head, logical_position=p, tier="exact")
        for layer in range(2)
        for head in range(2)
        for p in (0, 1, 6, 7)
    )
    plan = build_evidence_layer_plan(
        state, attention, records, prompt_budget_bytes=state.active_bytes // 2, uniform_capacity=4
    )
    again = build_evidence_layer_plan(
        state, attention, records, prompt_budget_bytes=state.active_bytes // 2, uniform_capacity=4
    )
    assert plan == again
    assert plan.capacities == (2, 6)
    assert plan.prompt_bytes == state.active_bytes // 2
    for heads in plan.positions:
        assert len({len(row) for row in heads}) == 1
        for row in heads:
            assert row == tuple(sorted(set(row)))
            assert {0, 7} <= set(row)
    for original, after in zip(saved, layers, strict=True):
        for a, b in zip(original, after, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert plan.trace()["benchmark_labels_accessed"] is False


def test_fraction_is_common_byte_driven_and_legacy_identity_is_preserved():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs/development/llava_bounded_history_v1.yaml"
    cfg = load_config(path)
    assert canonical_config(cfg)["bounded_decode"] == {"generated_capacity": 32, "sink_tokens": 4}
    policy = BoundedDecodeConfig(32, 4, 0.2, "output_impact_v1")
    assert policy.resolved_capacity(40 * 2**20, 2**19) == 16
    assert policy.resolved_capacity(80 * 2**20, 2**19) == 32
    assert policy.resolved_capacity(32 * 2**20, 2**19) == 12
    assert "output_impact_layers" in policy.effective_method(cfg.method)
    assert (
        canonical_config(replace(cfg, bounded_decode=policy))["bounded_decode"]["history_fraction"]
        == 0.2
    )
    for fraction in [True, float("nan"), 0.0, 1.0, "0.2"]:
        with pytest.raises(ConfigurationError, match="history_fraction"):
            BoundedDecodeConfig(history_fraction=fraction)
