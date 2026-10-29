"""Random-weight CPU tests for evidence-only capture changes."""

from dataclasses import replace

import torch
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel, compare_runtime_retention_one
from mosaickv.config import BoundedDecodeConfig, IncrementalHistoryConfig
from mosaickv.types import ForecastMode


def test_attention_mixture_changes_no_prefill_logit_or_cache_tensor():
    adapter = _llava()
    prepared = adapter.prepare_inputs(_runtime_request("mixture").messages)
    ordinary = adapter.prefill(prepared, capture_attentions=True, attention_query_window=2)
    mixed = adapter.prefill(
        prepared,
        capture_attentions=True,
        attention_query_window=2,
        attention_full_prompt_prior_weight=0.5,
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


def test_runtime_records_mixed_evidence_as_a_variant(tmp_path):
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
