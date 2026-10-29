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


def test_v2_generation_and_teacher_forcing_share_prompt_and_do_not_clone_source(tmp_path):
    from mosaickv.evaluation.model import EvaluationRequest

    config = replace(
        _config(),
        cache=replace(_config().cache, budget_value=1536),
        forecasting=replace(_config().forecasting, draft_steps=0, mode=ForecastMode.PROMPT_WINDOW),
        bounded_decode=BoundedDecodeConfig(
            4,
            0,
            incremental_history=IncrementalHistoryConfig(
                1,
                1,
                "recent_importance",
                fixed_prompt_reservation=True,
                borrow_prefill_snapshot=True,
                importance_refresh_tokens=2,
            ),
        ),
    )
    adapter = _llava()
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path)
    base = _runtime_request("v2-generation")
    flags = {"_mosaickv_capture_kv_ownership": True, "_mosaickv_capture_kv_transactions": True}
    request = EvaluationRequest(base.run_id, base.sample_id, base.task, base.messages, flags)
    result = runtime.generate(request)
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["source_snapshot_ownership"]["mode"] == "borrowed_read_only"
    assert trace["source_KV_retirement"]["remaining_source_tensor_objects"] == 0
    assert trace["request_KV_transactions"]["cache_update_calls"] >= 2 * 20
    assert result.metrics.peak_active_kv_storage_bytes <= 1536
    target = (3, 4, 5)
    second = EvaluationRequest(
        "v2-ppl",
        base.sample_id,
        base.task,
        base.messages,
        dict(flags, _mosaickv_teacher_forced_token_ids=target),
    )
    runtime.generate(second)
    traces = [json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    by_phase = {t["execution_phase"]: t for t in traces}
    a, b = by_phase["generation"], by_phase["teacher_forced"]
    assert a["packed_prefill_active_kv_bytes"] == b["packed_prefill_active_kv_bytes"]
    assert a["selected_blocks"] == b["selected_blocks"]
    assert b["teacher_forced_evaluation"]["evaluated_tokens"] == 3
    one = EvaluationRequest(
        "v2-one-token",
        base.sample_id,
        base.task,
        base.messages,
        dict(flags, _mosaickv_teacher_forced_token_ids=(3,)),
    )
    runtime.generate(one)
    t = json.loads(
        (
            tmp_path / "v2-one-token" / next((tmp_path / "v2-one-token").glob("*.json")).name
        ).read_text()
    )
    assert t["source_KV_retirement"]["remaining_source_tensor_objects"] == 0
    assert t["bounded_decode"]["status"] == "no_decode_updates"


def test_borrowing_does_not_change_prompt_tensors_or_generated_tokens(tmp_path):
    adapter = _llava()
    base = replace(
        _config(),
        cache=replace(_config().cache, budget_value=1536),
        forecasting=replace(_config().forecasting, draft_steps=0, mode=ForecastMode.PROMPT_WINDOW),
        bounded_decode=BoundedDecodeConfig(
            4,
            0,
            incremental_history=IncrementalHistoryConfig(
                1,
                1,
                "sink_recent",
                fixed_prompt_reservation=True,
            ),
        ),
    )
    for borrowed in (False, True):
        cfg = replace(
            base,
            bounded_decode=replace(
                base.bounded_decode,
                incremental_history=replace(
                    base.bounded_decode.incremental_history, borrow_prefill_snapshot=borrowed
                ),
            ),
        )
        HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path / str(borrowed)).generate(
            _runtime_request(str(borrowed))
        )
    a = json.loads(next((tmp_path / "False").rglob("*.json")).read_text())
    b = json.loads(next((tmp_path / "True").rglob("*.json")).read_text())
    assert a["generated_token_ids"] == b["generated_token_ids"]
    assert a["selected_blocks"] == b["selected_blocks"]


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
