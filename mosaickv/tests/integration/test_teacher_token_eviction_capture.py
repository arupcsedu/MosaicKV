"""Random-weight CPU VLM: observe NLL without changing decoder/cache decisions."""

import json
from dataclasses import replace

import torch
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

import mosaickv.backends.hf_runtime as runtime_module
from mosaickv.config import BoundedDecodeConfig, IncrementalHistoryConfig
from mosaickv.evaluation.token_eviction import TeacherTokenCapture
from mosaickv.types import ForecastMode


def test_real_tiny_decode_keeps_nll_cache_positions_and_target_counts_exact(tmp_path):
    adapter = _llava()
    targets = tuple(range(3, 13))
    adapter.processor.tokenizer = lambda *_args, **_kwargs: {"input_ids": torch.tensor([targets])}
    cfg = _config()
    cfg = replace(
        cfg,
        cache=replace(cfg.cache, budget_value=1536),
        forecasting=replace(cfg.forecasting, mode=ForecastMode.PROMPT_WINDOW, draft_steps=0),
        bounded_decode=BoundedDecodeConfig(
            4,
            0,
            incremental_history=IncrementalHistoryConfig(
                chunk_tokens=1,
                recent_tokens=1,
                policy="sink_recent",
                fixed_prompt_reservation=True,
                borrow_prefill_snapshot=True,
            ),
        ),
    )
    runtime = runtime_module.HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path)
    ordinary = runtime.teacher_forced_nll(_runtime_request("ordinary"), "synthetic teacher")
    with TeacherTokenCapture(adapter, targets, runtime_module) as capture:
        observed = runtime.teacher_forced_nll(_runtime_request("observed"), "synthetic teacher")
    detail = capture.record()
    assert (
        observed.negative_log_likelihood_sum
        == ordinary.negative_log_likelihood_sum
        == detail["sum_nll"]
    )
    assert observed.evaluated_tokens == ordinary.evaluated_tokens == len(targets)
    assert (
        next(
            x["prediction_position"]
            for x in detail["records"]
            if any(x["generated_evictions_visible_per_layer"])
        )
        == 6
    )
    traces = [json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    assert len(traces) == 2
    for key in ["selected_blocks", "generated_token_ids", "cache_lifecycle"]:
        assert traces[0][key] == traces[1][key]
