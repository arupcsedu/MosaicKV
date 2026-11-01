from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.protected_projection_source import (
    original_text_positions,
    protected_positions,
)


def test_protection_is_exactly_existing_local_policy_not_new_visual_or_text_quota():
    heads = ((*range(0, 64, 2), 63), tuple(range(1, 64, 2)))
    text = (0, 1, 22, 23, 62, 63)
    mask = protected_positions(heads, text, length=64)
    for h, row in enumerate(heads):
        expected = set(row[:4]) | set(row[-8:]) | (set(row) & set(text))
        assert set(mask[h].nonzero().flatten().tolist()) == expected
    assert mask[:, -1].all()
    with pytest.raises(ValueError):
        protected_positions(((0, 2),), text, length=64)


def test_metadata_requires_complete_exact_token_route_without_model_forward():
    prepared = SimpleNamespace(
        logical_sequence_length=8,
        model_inputs=dict(input_ids=torch.arange(8)[None], attention_mask=torch.ones(1, 8)),
        modality_map=(
            SimpleNamespace(start=0, end=2, modality="text"),
            SimpleNamespace(start=2, end=6, modality="image"),
            SimpleNamespace(start=6, end=8, modality="text"),
        ),
    )
    adapter = SimpleNamespace(prepare_inputs=lambda messages: prepared)
    text, digest = original_text_positions(adapter, (), expected_length=8)
    assert text == (0, 1, 6, 7) and len(digest) == 64
    with pytest.raises(ValueError):
        original_text_positions(adapter, (), expected_length=9)
    prepared.modality_map = prepared.modality_map[:-1]
    with pytest.raises(ValueError):
        original_text_positions(adapter, (), expected_length=8)


def test_metadata_model_has_no_real_weight_allocations_or_pretrained_model_load(monkeypatch):
    from transformers import AutoConfig, AutoProcessor, LlavaConfig

    from mosaickv.evaluation.protected_projection_source import metadata_adapter

    config = LlavaConfig(
        text_config=dict(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
        ),
        vision_config=dict(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
        ),
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    adapter = metadata_adapter("synthetic_configuration_only")
    assert adapter.device.type == "cpu"
    assert all(
        p.device.type == "meta" and p.dtype == torch.bfloat16 for p in adapter.model.parameters()
    )
