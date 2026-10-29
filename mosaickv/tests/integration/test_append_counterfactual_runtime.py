"""Random-weight CPU VLM append-only intervention, not benchmark evidence."""

import hashlib
import json
from dataclasses import replace

import torch
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

import mosaickv.backends.hf_runtime as runtime_module
from mosaickv.config import BoundedDecodeConfig, IncrementalHistoryConfig
from mosaickv.evaluation.append_counterfactual import append_config, verify_intervention
from mosaickv.evaluation.token_eviction import TeacherTokenCapture
from mosaickv.types import ForecastMode


def test_same_selected_prompt_is_exact_before_eviction_and_never_evicts_in_append_mode(tmp_path):
    adapter, targets = _llava(), tuple(range(3, 13))
    adapter.processor.tokenizer = lambda *_a, **_k: {"input_ids": torch.tensor([targets])}
    cfg = _config()
    profile = tmp_path / "profile.json"
    profile.write_text("[0.0, 0.5]")
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
                layer_profile_path=str(profile),
                layer_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
            ),
        ),
    )
    captures, traces = [], []
    for name, c in (("fifo", cfg), ("append", append_config(cfg, len(targets), (64, 64)))):
        folder = tmp_path / name
        runtime = runtime_module.HuggingFaceMosaicKVModel(adapter, c, trace_directory=folder)
        with TeacherTokenCapture(adapter, targets, runtime_module) as cap:
            loss = runtime.teacher_forced_nll(_runtime_request(name), "synthetic teacher")
        detail = cap.record()
        assert detail["sum_nll"] == loss.negative_log_likelihood_sum
        captures.append(detail)
        traces.append(json.loads(next(folder.rglob("*.json")).read_text()))
    check = verify_intervention(captures[1], captures[0], traces[1], traces[0], history_capacity=9)
    assert check["identical_prefix_prediction_count"] == 5
    assert check["no_prompt_or_generated_evictions"]
