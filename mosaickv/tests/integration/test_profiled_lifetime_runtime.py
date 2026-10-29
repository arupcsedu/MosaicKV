"""Random-weight CPU transport tests; no pretrained quality claims."""

import hashlib
import json
from dataclasses import replace

import pytest
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
from mosaickv.config import BoundedDecodeConfig, IncrementalHistoryConfig
from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.types import ForecastMode, MosaicKVMethod


@pytest.mark.parametrize(
    "method", [MosaicKVMethod.MOSAICKV_EXACT, MosaicKVMethod.PROMPT_ATTENTION_TOPK]
)
def test_profiled_layers_are_unpadded_and_identical_across_metric_phases(tmp_path, method):
    path = tmp_path / "profile.json"
    raw = json.dumps([0.1, 0.8]).encode()
    path.write_bytes(raw)
    config = replace(
        _config(method),
        cache=replace(_config(method).cache, budget_value=1280),
        forecasting=replace(
            _config(method).forecasting, draft_steps=0, mode=ForecastMode.PROMPT_WINDOW
        ),
        bounded_decode=BoundedDecodeConfig(
            4,
            0,
            incremental_history=IncrementalHistoryConfig(
                1,
                1,
                "sink_recent",
                fixed_prompt_reservation=True,
                borrow_prefill_snapshot=True,
                layer_profile_path=str(path),
                layer_profile_sha256=hashlib.sha256(raw).hexdigest(),
            ),
        ),
    )
    root = tmp_path / "traces"
    runtime = HuggingFaceMosaicKVModel(_llava(), config, trace_directory=root)
    request = _runtime_request("profile-generation")
    result = runtime.generate(request)
    a = json.loads(next(root.rglob("*.json")).read_text())
    assert a["packed_padding_kv_bytes"] == 0
    assert len(set(a["selected_layer_prompt_capacities"])) > 1
    assert result.metrics.peak_active_kv_storage_bytes <= 1280
    assert (
        result.metrics.initial_active_kv_storage_bytes
        < result.metrics.final_active_kv_storage_bytes
    )
    assert a["source_KV_retirement"]["remaining_source_tensor_objects"] == 0
    assert a["evidence_layer_allocation"]["not_official_PrefixKV"] is True
    runtime.generate(
        EvaluationRequest(
            "profile-ppl",
            request.sample_id,
            request.task,
            request.messages,
            {"_mosaickv_teacher_forced_token_ids": (3, 4, 5)},
        )
    )
    phases = {
        t["execution_phase"]: t for t in (json.loads(p.read_text()) for p in root.rglob("*.json"))
    }
    assert phases["generation"]["selected_blocks"] == phases["teacher_forced"]["selected_blocks"]
    assert "profiled_v3" in result.effective_method


def test_profile_hash_mismatch_fails_with_failure_trace(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("[0.1,0.8]")
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
                "sink_recent",
                fixed_prompt_reservation=True,
                layer_profile_path=str(path),
                layer_profile_sha256="a" * 64,
            ),
        ),
    )
    root = tmp_path / "traces"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        HuggingFaceMosaicKVModel(_llava(), config, trace_directory=root).generate(
            _runtime_request("bad")
        )
    trace = json.loads(next(root.rglob("*.json")).read_text())
    assert trace["status"] == "failed"
