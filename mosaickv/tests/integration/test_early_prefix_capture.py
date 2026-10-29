"""Tiny random CPU architecture: fixed prefix, exact controls, hook cleanup."""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

import mosaickv.backends.hf_runtime as runtime_module
from mosaickv.config import BoundedDecodeConfig, IncrementalHistoryConfig
from mosaickv.evaluation.early_prefix import EarlyPrefixCapture, compare_layer_outputs
from mosaickv.types import ForecastMode, MosaicKVMethod


def test_read_only_capture_replays_logits_storage_and_selected_positions_exactly(tmp_path):
    adapter, targets = _llava(), tuple(range(3, 11))
    cfg = _config()
    cfg = replace(
        cfg,
        cache=replace(cfg.cache, budget_value=1920),
        forecasting=replace(cfg.forecasting, mode=ForecastMode.PROMPT_WINDOW, draft_steps=0),
        bounded_decode=BoundedDecodeConfig(
            8,
            0,
            incremental_history=IncrementalHistoryConfig(
                chunk_tokens=1, recent_tokens=1, policy="sink_recent", fixed_prompt_reservation=True
            ),
        ),
    )
    baseline = None
    original = (adapter.prefill, adapter.decode_one_token, runtime_module._state_from_packed)
    for observed in (False, True):
        folder = tmp_path / str(observed)
        runtime = runtime_module.HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=folder)
        with (
            patch.object(runtime, "_teacher_forced_target_ids", return_value=targets),
            EarlyPrefixCapture(
                adapter, runtime_module, targets, role="test", observed=observed
            ) as capture,
        ):
            runtime.teacher_forced_nll(_runtime_request(str(observed)), "model prefix sentinel")
        capture.validate_complete()
        assert original == (
            adapter.prefill,
            adapter.decode_one_token,
            runtime_module._state_from_packed,
        )
        trace = json.loads(next(folder.rglob("*.json")).read_text())
        current = (torch.stack(capture.logits), capture.positions, trace["cache_lifecycle"])
        if baseline is None:
            baseline = current
        else:
            assert torch.equal(baseline[0], current[0])
            assert baseline[1:] == current[1:]
            rows = compare_layer_outputs(
                capture.recorder.records["test"], capture.recorder.records["test"]
            )
            assert all(r["l2_error"] == 0 for r in rows)
        for layer in adapter._language_layers():
            assert not layer._forward_hooks and not layer._forward_pre_hooks
    with pytest.raises(ValueError, match="bounded"):
        EarlyPrefixCapture(adapter, runtime_module, list(range(33)), role="bad", observed=True)


def test_fullkv_attention_capture_keeps_all_logits_bitwise_and_decomposes_at_23(tmp_path):
    adapter, targets = _llava(), tuple(range(3, 35))
    cfg = _config()
    cfg = replace(
        cfg,
        method=MosaicKVMethod.FULLKV,
        bounded_decode=None,
        cache=replace(cfg.cache, retention_ratio=1.0),
        forecasting=replace(cfg.forecasting, mode=ForecastMode.PROMPT_WINDOW, draft_steps=0),
    )
    saved = None
    for observed in (False, True):
        runtime = runtime_module.HuggingFaceMosaicKVModel(
            adapter, cfg, trace_directory=tmp_path / str(observed)
        )
        cap = EarlyPrefixCapture(
            adapter,
            runtime_module,
            targets,
            role="full",
            observed=observed,
            reference_positions={"test": [[[0, 3]], [[0, 3]]]} if observed else None,
        )
        with patch.object(runtime, "_teacher_forced_target_ids", return_value=targets), cap:
            runtime.teacher_forced_nll(_runtime_request(str(observed)), "no labels")
        cap.validate_complete()
        logits = torch.stack(cap.logits)
        if saved is None:
            saved = logits
        else:
            assert torch.equal(saved, logits)
            assert len(cap.decompositions["test"]) == 2
            assert all(row["identity_max_abs_error"] < 1e-5 for row in cap.decompositions["test"])
