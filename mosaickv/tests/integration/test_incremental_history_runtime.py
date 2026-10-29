"""Tiny random-weight multimodal model; no downloads or task-quality claim."""

import json
from dataclasses import replace

import pytest
import torch
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
from mosaickv.config import BoundedDecodeConfig, GenerationConfig, IncrementalHistoryConfig
from mosaickv.types import ForecastMode, MosaicKVMethod


@pytest.mark.parametrize("policy", ["sink_recent", "recent_importance"])
@pytest.mark.parametrize(
    "method", [MosaicKVMethod.MOSAICKV_EXACT, MosaicKVMethod.PROMPT_ATTENTION_TOPK]
)
def test_incremental_runtime_has_growing_physical_storage_but_bounded_peak(
    tmp_path, policy, method
):
    config = _config(method)
    config = replace(
        config,
        cache=replace(config.cache, budget_value=1024),
        forecasting=replace(config.forecasting, draft_steps=0, mode=ForecastMode.PROMPT_WINDOW),
        bounded_decode=BoundedDecodeConfig(
            3, 0, incremental_history=IncrementalHistoryConfig(1, 1, policy)
        ),
    )
    result = HuggingFaceMosaicKVModel(_llava(), config, trace_directory=tmp_path).generate(
        _runtime_request("incremental")
    )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert (
        result.metrics.initial_active_kv_storage_bytes
        < result.metrics.final_active_kv_storage_bytes
    )
    assert result.metrics.peak_active_kv_storage_bytes <= 1024
    assert result.metrics.generated_tokens == 20
    info = trace["bounded_decode"]
    assert info["owned_decode_KV_including_growth_peak"]["peak_storage_bytes"] <= 1024
    assert info["evicted_generated_positions_per_layer"] == [16, 16]
    assert info["next_decode_position"] == trace["source_kv_layout"][0]["key_shape"][-2] + 19
    assert "incremental_history" in result.effective_method
    assert trace["source_KV_retirement"]["remaining_source_tensor_objects"] == 0
    assert trace["teacher_forced_evaluation"]["enabled"] is False


def test_no_eviction_matches_ordinary_full_history_tokens(tmp_path):
    torch.manual_seed(53)
    adapter = _llava()
    config = replace(
        _config(),
        cache=replace(_config().cache, budget_value=4096),
        generation=GenerationConfig(max_new_tokens=5),
        bounded_decode=None,
    )
    a = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path / "reference").generate(
        _runtime_request("reference")
    )
    inc = replace(
        config,
        bounded_decode=BoundedDecodeConfig(
            16, 1, incremental_history=IncrementalHistoryConfig(1, 2)
        ),
    )
    b = HuggingFaceMosaicKVModel(adapter, inc, trace_directory=tmp_path / "incremental").generate(
        _runtime_request("incremental")
    )
    r = json.loads(next((tmp_path / "reference").rglob("*.json")).read_text())
    s = json.loads(next((tmp_path / "incremental").rglob("*.json")).read_text())
    assert r["generated_token_ids"] == s["generated_token_ids"]
    assert a.answer == b.answer
