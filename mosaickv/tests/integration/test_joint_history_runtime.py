"""CPU regressions for unequal history lengths and global-length plateaus."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.adapters.huggingface.incremental_history_cache import install_incremental_history
from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot
from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
from mosaickv.config import BoundedDecodeConfig, IncrementalHistoryConfig
from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.types import ForecastMode, MosaicKVMethod


def test_layerwise_history_handles_plateau_then_later_global_growth():
    adapter = _llava()
    prepared = adapter.prepare_inputs(_runtime_request("plateau").messages)
    prefill = adapter.prefill(prepared, capture_queries=False)
    state = prefill.state
    snapshot = gather_layer_local_snapshot(
        adapter, state.past_key_values, (((0, 1, 2, 3),), ((3,),))
    )
    state.past_key_values = adapter.inject_past_key_values(snapshot)
    state.active_cache_length = 4
    state.model_state["mosaickv_packed_all_valid"] = True
    install_incremental_history(
        adapter,
        state,
        generated_capacity=10,
        sink_tokens=0,
        cap_bytes=4096,
        config=IncrementalHistoryConfig(1, 1, "sink_recent"),
        layer_generated_capacities=(1, 10),
    )
    token = prefill.next_token_id
    lengths = []
    original = state.next_decode_position
    for _ in range(12):
        out = adapter.decode_one_token(token, state, capture_queries=False)
        state, token = out.state, out.next_token_id
        assert bool(torch.isfinite(out.logits).all())
        lengths.append(state.active_cache_length)
    assert lengths == [5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 11, 11]
    assert state.next_decode_position == original + 12
    assert all(
        max(x.generated_logical_positions) < state.next_decode_position
        for x in state.past_key_values.layers
    )


@pytest.mark.parametrize(
    "method", [MosaicKVMethod.MOSAICKV_EXACT, MosaicKVMethod.PROMPT_ATTENTION_TOPK]
)
def test_joint_profile_preserves_total_reserve_and_cap(tmp_path, method):
    profile = tmp_path / "profile.json"
    raw = b"[0.1,0.8]"
    profile.write_bytes(raw)
    config = replace(
        _config(method),
        cache=replace(_config(method).cache, budget_value=1536),
        forecasting=replace(
            _config(method).forecasting, mode=ForecastMode.PROMPT_WINDOW, draft_steps=0
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
                layer_profile_path=str(profile),
                layer_profile_sha256=hashlib.sha256(raw).hexdigest(),
                history_allocation="profile_weighted",
            ),
        ),
    )
    outdir = tmp_path / "traces"
    runtime = HuggingFaceMosaicKVModel(_llava(), config, trace_directory=outdir)
    request = _runtime_request("joint")
    result = runtime.generate(request)
    trace = json.loads(next(outdir.rglob("*.json")).read_text())
    caps = trace["bounded_decode"]["generated_capacities_per_layer"]
    assert len(set(caps)) > 1 and sum(caps) == 8 and min(caps) >= 2
    assert result.metrics.peak_active_kv_storage_bytes <= 1536
    assert trace["packed_padding_kv_bytes"] == 0
    assert trace["source_KV_retirement"]["remaining_source_tensor_objects"] == 0
    assert "joint_v4" in result.effective_method
    runtime.generate(
        EvaluationRequest(
            "joint-ppl",
            request.sample_id,
            request.task,
            request.messages,
            {"_mosaickv_teacher_forced_token_ids": (3, 4, 5)},
        )
    )
    phases = {
        t["execution_phase"]: t for t in (json.loads(p.read_text()) for p in outdir.rglob("*.json"))
    }
    assert phases["generation"]["selected_blocks"] == phases["teacher_forced"]["selected_blocks"]
