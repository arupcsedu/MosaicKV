"""Random-weight CPU tests for evidence-only capture changes."""

import json
from dataclasses import replace

import pytest
import torch
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel, compare_runtime_retention_one
from mosaickv.config import BoundedDecodeConfig, IncrementalHistoryConfig
from mosaickv.types import ForecastMode


@pytest.mark.parametrize("pooling", ["per_head", "shared_heads"])
def test_attention_mixture_changes_no_prefill_logit_or_cache_tensor(pooling):
    adapter = _llava()
    prepared = adapter.prepare_inputs(_runtime_request("mixture").messages)
    ordinary = adapter.prefill(prepared, capture_attentions=True, attention_query_window=2)
    mixed = adapter.prefill(
        prepared,
        capture_attentions=True,
        attention_query_window=2,
        attention_full_prompt_prior_weight=0.5,
        attention_full_prompt_prior_head_pooling=pooling,
    )
    assert torch.equal(ordinary.logits, mixed.logits)
    assert ordinary.next_token_id.equal(mixed.next_token_id)
    for a, b in zip(
        ordinary.state.past_key_values.layers, mixed.state.past_key_values.layers, strict=True
    ):
        assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(ordinary.attention_weights, mixed.attention_weights, strict=True)
    )


def test_layerwise_evidence_changes_no_model_state_or_retention_one(tmp_path):
    adapter = _llava()
    prepared = adapter.prepare_inputs(_runtime_request("layerwise-prefill").messages)
    layer_weights = tuple(
        0.5 if i % 2 == 0 else 1.0 for i in range(len(adapter._language_layers()))
    )
    ordinary = adapter.prefill(prepared, capture_attentions=True, attention_query_window=2)
    mixed = adapter.prefill(
        prepared,
        capture_attentions=True,
        attention_query_window=2,
        attention_full_prompt_prior_weight=0.5,
        attention_full_prompt_prior_head_pooling="shared_heads",
        attention_full_prompt_prior_layer_weights=layer_weights,
    )
    assert torch.equal(ordinary.logits, mixed.logits)
    for a, b in zip(
        ordinary.state.past_key_values.layers, mixed.state.past_key_values.layers, strict=True
    ):
        assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
    cfg = _config()
    cfg = replace(
        cfg,
        cache=replace(cfg.cache, retention_ratio=1.0),
        bounded_decode=BoundedDecodeConfig(
            4,
            0,
            incremental_history=IncrementalHistoryConfig(
                chunk_tokens=1,
                recent_tokens=1,
                policy="sink_recent",
                fixed_prompt_reservation=True,
                borrow_prefill_snapshot=True,
                full_prompt_prior_weight=0.5,
                full_prompt_prior_head_pooling="shared_heads",
                full_prompt_prior_layer_weights=layer_weights,
            ),
        ),
    )
    parity = compare_runtime_retention_one(adapter, prepared, cfg)
    assert parity.token_agreement == 1 and parity.maximum_logit_difference <= 1e-6
    compressed = replace(
        cfg,
        cache=replace(cfg.cache, retention_ratio=0.9999999999999999, budget_value=1536),
        forecasting=replace(cfg.forecasting, mode=ForecastMode.PROMPT_WINDOW, draft_steps=0),
    )
    request = replace(
        _runtime_request("layerwise-compressed"),
        generation_kwargs={"_mosaickv_capture_selection_fingerprints": True},
    )
    result = HuggingFaceMosaicKVModel(_llava(), compressed, trace_directory=tmp_path).generate(
        request
    )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["prompt_attention_evidence_basis"]["layer_weights"] == list(layer_weights)
    assert trace["selected_position_fingerprints"]
    assert "layerwise_v10" in result.effective_method
    assert result.metrics.peak_active_kv_storage_bytes <= 1536


@pytest.mark.parametrize("pooling", ["per_head", "shared_heads"])
def test_runtime_records_mixed_evidence_as_a_variant(tmp_path, pooling):
    config = _config()
    config = replace(
        config,
        cache=replace(config.cache, budget_value=1536),
        forecasting=replace(config.forecasting, mode=ForecastMode.PROMPT_WINDOW, draft_steps=0),
        bounded_decode=BoundedDecodeConfig(
            4,
            0,
            incremental_history=IncrementalHistoryConfig(
                1,
                1,
                "sink_recent",
                fixed_prompt_reservation=True,
                borrow_prefill_snapshot=True,
                full_prompt_prior_weight=0.5,
                full_prompt_prior_head_pooling=pooling,
            ),
        ),
    )
    out = HuggingFaceMosaicKVModel(_llava(), config, trace_directory=tmp_path).generate(
        _runtime_request("mixed-runtime")
    )
    assert "full_prompt_prior_v7" in out.effective_method
    assert out.metrics.peak_active_kv_storage_bytes <= config.cache.budget_value
    adapter = _llava()
    identity = replace(config, cache=replace(config.cache, retention_ratio=1.0))
    prepared = adapter.prepare_inputs(_runtime_request("mixed-identity").messages)
    parity = compare_runtime_retention_one(adapter, prepared, identity)
    assert parity.token_agreement == 1.0 and parity.maximum_logit_difference <= 1e-6
