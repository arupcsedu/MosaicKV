import json
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import native_storage_control as control


def test_substitutions_restored_after_failure(monkeypatch):
    original_construct = control.request_precision.construct_request_formats
    original_run = control.storage_runs.run

    def fail(*args, **kwargs):
        assert control.request_precision.construct_request_formats is control.construct_native
        assert control.storage_runs.run is control.run_native
        raise RuntimeError("intentional fixture failure")

    monkeypatch.setattr(control.request_precision, "complete_request", fail)
    with pytest.raises(RuntimeError, match="fixture"):
        control.complete_control(
            None,
            None,
            None,
            method="mosaickv_exact",
            mode="ordinary_selected_BF16",
            profile_path="unused",
            ratio=0.5,
        )
    assert control.request_precision.construct_request_formats is original_construct
    assert control.storage_runs.run is original_run
    assert not control._LOCK.locked()


def test_raw_history_does_not_reconstruct_source_history(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(2), {}

    def request(*args, **kwargs):
        assert kwargs["precision"] == "query3"
        control.storage_runs.run(
            source_history=True,
            consolidated_history=True,
            source_history_constructor="direct",
            sealed=True,
        )
        return torch.zeros(2), dict(online_variant=dict(generated_history="source"))

    monkeypatch.setattr(control.storage_runs, "run", run)
    monkeypatch.setattr(control.request_precision, "complete_request", request)
    _, record = control.complete_control(
        None,
        None,
        None,
        method="mosaickv_exact",
        mode="query3_raw_history",
        profile_path="unused",
        ratio=0.5,
    )
    assert calls == [
        dict(
            source_history=False,
            consolidated_history=False,
            source_history_constructor="reference",
            sealed=False,
        )
    ]
    assert record["effective_history_mode"] == "raw_BF16_append_only"
    assert control.storage_runs.run is run


def test_concurrent_or_unknown_controls_fail_before_mutation():
    with pytest.raises(ValueError, match="unregistered"):
        control.complete_control(
            None,
            None,
            None,
            method="mosaickv_exact",
            mode="unknown",
            profile_path="unused",
            ratio=0.5,
        )
    control._LOCK.acquire()
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            control.complete_control(
                None,
                None,
                None,
                method="mosaickv_exact",
                mode="query3_raw_history",
                profile_path="unused",
                ratio=0.5,
            )
    finally:
        control._LOCK.release()


def test_native_construction_is_private_exact_and_unpadded(tmp_path, monkeypatch):
    from mosaickv.adapters.huggingface.types import (
        CachedKeyState,
        CacheLayerSnapshot,
        CacheSnapshot,
    )
    from mosaickv.evaluation import storage_baselines

    source = tuple(
        (torch.arange(48).reshape(1, 2, 6, 4).bfloat16() + i, torch.ones(1, 2, 6, 4).bfloat16())
        for i in range(2)
    )
    originals = tuple(tuple(t.clone() for t in pair) for pair in source)
    template = CacheSnapshot(
        tuple(CacheLayerSnapshot(k, v, 2) for k, v in source),
        tuple,
        "tuple",
        6,
        CachedKeyState.POST_ROPE,
    )
    positions = (((0, 5), (1, 5)), ((0, 2, 5), (1, 3, 5)))
    expected_bytes = 5 * 2 * 4 * 2 * 2
    monkeypatch.setattr(control.request_precision, "online_frozen", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(
        storage_baselines,
        "baseline_payload",
        lambda *a, **kw: (positions, None, dict(source_slot_byte_cap=expected_bytes)),
    )
    adapter = SimpleNamespace(
        _legacy_layers=lambda _: (source, "tuple"), borrow_past_key_values=lambda _: template
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps([0.5, 0.5]))
    payloads, _ = control.construct_native(
        adapter,
        None,
        SimpleNamespace(state=SimpleNamespace(past_key_values=object())),
        None,
        None,
        "mosaickv_exact",
        str(path),
        0.5,
        "lossless",
    )
    snapshot = payloads[0].snapshot
    assert payloads[0].head_positions == positions
    assert snapshot.independent_layer_lengths and snapshot.active_sequence_length == 3
    for i, layer in enumerate(snapshot.layers):
        for new, old in ((layer.key, source[i][0]), (layer.value, source[i][1])):
            expected = torch.stack([old[0, h, list(p)] for h, p in enumerate(positions[i])])[None]
            assert torch.equal(new, expected) and new.data_ptr() != old.data_ptr()
    assert all(
        torch.equal(a, b)
        for old, new in zip(originals, source, strict=True)
        for a, b in zip(old, new, strict=True)
    )


def test_native_decoder_drops_snapshot_and_preserves_logical_position(monkeypatch):
    from mosaickv.adapters.huggingface.types import (
        CachedKeyState,
        CacheLayerSnapshot,
        CacheSnapshot,
    )
    from mosaickv.backends import hf_runtime

    pairs = tuple(
        (torch.ones(1, 2, n, 4).bfloat16(), torch.ones(1, 2, n, 4).bfloat16()) for n in (2, 3)
    )
    snapshot = CacheSnapshot(
        tuple(CacheLayerSnapshot(k, v, 2) for k, v in pairs),
        tuple,
        "tuple",
        3,
        CachedKeyState.POST_ROPE,
        True,
    )
    payload = control.NativePayload(snapshot, (((0, 1),) * 2, ((0, 1, 2),) * 2))
    state = SimpleNamespace(
        past_key_values=SimpleNamespace(pairs=pairs), next_decode_position=100, model_state={}
    )

    def build(adapter, original, packed):
        assert original.next_decode_position == 100
        assert packed.all_slots_valid and packed.snapshot is snapshot
        assert packed.layerwise_logical_positions == ((0, 1), (0, 1, 2))
        return state

    def decode(token, current, **kwargs):
        assert payload.snapshot is None
        current.past_key_values.pairs = tuple(
            tuple(torch.cat((t, t[..., :1, :]), 2) for t in pair)
            for pair in current.past_key_values.pairs
        )
        current.next_decode_position += 1
        return SimpleNamespace(logits=torch.zeros(1, 1, 8), next_token_id=0)

    monkeypatch.setattr(hf_runtime, "_state_from_packed", build)
    adapter = SimpleNamespace(
        _legacy_layers=lambda cache: (cache.pairs, "tuple"), decode_one_token=decode
    )
    prefill = SimpleNamespace(
        state=SimpleNamespace(next_decode_position=100),
        logits=torch.zeros(1, 1, 8),
        next_token_id=0,
    )
    logits, record = control.run_native(adapter, prefill, (payload,), tokens=128)
    assert logits.shape == (128, 8) and payload.snapshot is None
    assert state.next_decode_position == 227
    assert record["storage"]["initial_active_kv_storage_bytes"] == 160
    assert record["storage"]["final_active_kv_storage_bytes"] == 160 + 127 * 64
    assert not record["storage"]["KV_transaction_peak_certified"]


def test_missing_producers_are_explicit_not_silent(tmp_path, monkeypatch):
    import evicastkv_native_storage_control as runner

    root = tmp_path / "control"
    root.mkdir()
    plan = dict(source_sha="a" * 40, jobs=[dict(index=0), dict(index=1)], arms=runner.ARMS)
    monkeypatch.setattr(runner.previous.common, "inputs", lambda *args: plan)
    runner.analyze(root, "fixture")
    result = json.loads((root / "analysis/decision.json").read_text())
    assert not result["complete"] and result["scored_rows"] == 0
    assert result["expected_rows"] == 42 and len(result["failures"]) == 2
    assert not result["statistically_supported_superiority"]
