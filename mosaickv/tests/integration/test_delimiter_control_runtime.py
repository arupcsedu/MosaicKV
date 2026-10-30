"""Tiny random CPU VLM transaction; no download or benchmark-quality claim."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_bounded_history_runtime import _config
from test_hf_tiny_models import _llava, _runtime_request

import mosaickv.backends.hf_runtime as runtime_module
from mosaickv.config import BoundedDecodeConfig, IncrementalHistoryConfig
from mosaickv.evaluation.delimiter_control import DelimiterTransaction
from mosaickv.evaluation.early_prefix import EarlyPrefixCapture
from mosaickv.types import ForecastMode


def test_null_repacking_and_restoration_preserve_exact_bytes_positions_and_source(tmp_path):
    adapter, targets = _llava(), tuple(range(3, 11))
    adapter.processor.tokenizer = SimpleNamespace(
        decode=lambda ids: "ASSISTANT:" if ids == [4, 5] else "other"
    )
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
    saved = {}
    original = (adapter.prefill, adapter.decode_one_token, runtime_module._state_from_packed)
    for mode in ("anchor", "null_repack", "restore_delimiter"):
        folder = tmp_path / mode
        runtime = runtime_module.HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=folder)
        cap = EarlyPrefixCapture(adapter, runtime_module, targets, role=mode, observed=False)
        tx = DelimiterTransaction(adapter, runtime_module, mode=mode, prompt_window=2)
        with patch.object(runtime, "_teacher_forced_target_ids", return_value=targets), cap, tx:
            runtime.teacher_forced_nll(_runtime_request(mode), "model-generated prefix")
        cap.validate_complete()
        saved[mode] = (
            torch.stack(cap.logits),
            cap.positions,
            json.loads(next(folder.rglob("*.json")).read_text())["cache_lifecycle"],
        )
        assert original == (
            adapter.prefill,
            adapter.decode_one_token,
            runtime_module._state_from_packed,
        )
        if mode != "anchor":
            assert tx.record["source_KV_unchanged"]
            assert tx.record["initial_physical_bytes"] == tx.record["constructed_physical_bytes"]
            assert tx.record["padding_bytes"] == 0
            assert saved[mode][2] == saved["anchor"][2]
        if mode == "null_repack":
            assert torch.equal(saved[mode][0], saved["anchor"][0])
            assert saved[mode][1] == saved["anchor"][1]
        if mode == "restore_delimiter":
            assert all({2, 3}.issubset(h) for layer in cap.positions for h in layer)
            for e in tx.record["events"]:
                assert len(e["added_positions"]) == len(e["removed_positions"])
                assert 3 not in e["removed_positions"]


def test_hook_cleanup_on_missing_delimiter(tmp_path):
    adapter = _llava()
    adapter.processor.tokenizer = SimpleNamespace(decode=lambda _ids: "not a delimiter")
    original = (adapter.prefill, runtime_module._state_from_packed)
    import pytest

    with (
        pytest.raises(ValueError, match="cannot be identified"),
        DelimiterTransaction(adapter, runtime_module, mode="restore_delimiter"),
    ):
        adapter.prefill(adapter.prepare_inputs(_runtime_request("bad").messages))
    assert original == (adapter.prefill, runtime_module._state_from_packed)


def test_actual_restored_KV_is_verbatim_source_and_not_a_renumbered_slot(tmp_path):
    from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot
    from mosaickv.evaluation.delimiter_control import build_control, source_sha

    adapter = _llava()
    prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("source").messages))
    original, _ = adapter._legacy_layers(prefill.state.past_key_values)
    checksum = source_sha(original)
    positions = tuple(((0, 3),) for _ in original)
    snapshot = gather_layer_local_snapshot(adapter, prefill.state.past_key_values, positions)
    anchor = runtime_module.PackedRuntimeCache(
        snapshot,
        (),
        2,
        True,
        (),
        (),
        tuple(
            dict(layer=layer, kv_head=0, slot=slot, logical_position=p, tier="exact")
            for layer in range(len(original))
            for slot, p in enumerate((0, 3))
        ),
    )
    attention = tuple(torch.full((1, 2, 1, 4), 0.25) for _ in original)
    restored, record = build_control(
        adapter, prefill.state, anchor, attention, (2, 3), (3,), restore=True
    )
    assert record["changed_routes"] == len(original)
    assert record["source_KV_unchanged"]
    for layer, (source, packed) in enumerate(zip(original, restored.snapshot.layers, strict=True)):
        assert torch.equal(packed.key, source[0][:, :, [2, 3], :])
        assert torch.equal(packed.value, source[1][:, :, [2, 3], :])
        assert [r["logical_position"] for r in restored.slot_records if r["layer"] == layer] == [
            2,
            3,
        ]
    assert source_sha(original) == checksum
