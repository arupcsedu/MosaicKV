"""Consumed paired systems study; pure schemas and actual tiny BF16 constructors."""

import copy
import json
import time
from types import SimpleNamespace

import evicastkv_lossless_streaming_timing as study
import pytest
import torch
from test_streaming_lossless_source_runner import TinyAdapter


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_frozen_pair_schedule_has_all_warmups_repetitions_and_reproducible_orders():
    first = study.orders(617)
    assert first == study.orders(617)
    assert first != study.orders(618)
    assert len(first) == 25
    assert [row["trial"] for row in first[:5]] == list(range(5))
    assert [row["trial"] for row in first[5:]] == list(range(20))
    assert all(sorted(row["method_order"]) == sorted(study.ARMS) for row in first)
    assert {tuple(row["method_order"]) for row in first} == {study.ARMS, study.ARMS[::-1]}


def fake_history():
    layer = SimpleNamespace(
        tail_used=15,
        source_sealed=[],
        history=tuple(torch.zeros(1, 4, 16, 128).bfloat16() for _ in range(2)),
        normalized_tail=torch.zeros(1, 16, 512).bfloat16(),
    )
    return SimpleNamespace(
        layers=[layer],
        phase_tail=(
            torch.zeros(1, 16, 128).bfloat16(),
            torch.zeros(1, 16, 128).bfloat16(),
            torch.arange(16),
        ),
        history_position_chunks=[],
        history_rotary_chunks=[],
        workspace=(),
    )


def test_history_digest_excludes_uninitialized_tail_capacity_but_checks_valid_values():
    cache = fake_history()
    before = study.history_hash(cache)
    for value in cache.layers[0].history:
        value[..., 15, :] = 99
    cache.layers[0].normalized_tail[:, 15] = 99
    for value in cache.phase_tail[:2]:
        value[:, 15] = 99
    cache.phase_tail[2][15] = 99
    assert study.history_hash(cache) == before
    cache.layers[0].history[0][..., 14, :] = 99
    assert study.history_hash(cache) != before


class Event:
    def record(self):
        self.stamp = time.perf_counter()

    def elapsed_time(self, other):
        return (other.stamp - self.stamp) * 1000


@pytest.mark.parametrize("audit", [False, True])
def test_actual_tiny_constructor_schedules_preserve_bits_disable_only_timing_observer(
    monkeypatch, audit
):
    import evicastkv_request_validation as runtime

    from mosaickv.evaluation import lossless_source_control, storage_runs

    monkeypatch.setattr(runtime, "load_messages", lambda payload: payload)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: Event())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1000000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2000000)
    seen, actual_constructor = [], lossless_source_control.construct

    def constructor(*args, **kwargs):
        # Keep metadata only: retaining an audit closure would itself retain
        # native control tensors and invalidate the retirement measurement.
        seen.append(kwargs.get("workspace_observer") is not None)
        return actual_constructor(*args, **kwargs)

    monkeypatch.setattr(lossless_source_control, "construct", constructor)
    monkeypatch.setattr(storage_runs, "cache_from_formats", lambda *args, **kwargs: fake_history())

    def decode(adapter, prefill, formats, **kwargs):
        assert kwargs["audit"] is audit
        assert kwargs["forced"] == tuple(range(128))
        assert prefill.state.past_key_values is None
        storage_runs.cache_from_formats(adapter, formats)
        size = sum(packet.component_bytes for packet in formats)
        return torch.zeros(128, 32), dict(
            storage=dict(
                initial_active_kv_storage_bytes=size,
                final_active_kv_storage_bytes=size + 100,
                peak_active_kv_storage_bytes=size + 200,
                history_updates_per_layer=[127] * 32,
                outstanding_layer_leases=0,
            )
        )

    monkeypatch.setattr(storage_runs, "run", decode)
    adapter, records = TinyAdapter(), []
    with torch.inference_mode():
        for arm in study.ARMS:
            logits, row = study.request(adapter, {}, arm, prefix=tuple(range(128)), audit=audit)
            assert logits.shape == (128, 32)
            assert row["constructor_all_word_validation"]
            assert row["no_per_ATen_audit_during_timing"] is (not audit)
            assert not row["whole_request_KV_peak_certified"]
            assert row["timing_peak_is_incomplete_snapshot"] is (not audit)
            assert row["private_source_objects_retired_after_response"]
            assert row["timing"]["full_request_wall_seconds"] > 0
            records.append(row)
    assert len(seen) == 4 and all(observer_present is audit for observer_present in seen)
    assert study.match_pair(*records, audited=audit)
    assert adapter.calls == [(False, False), (False, False)]


def record(arm, *, audit):
    return dict(
        arm=arm,
        audited_precheck=audit,
        storage=dict(
            initial_active_kv_storage_bytes=100,
            final_active_kv_storage_bytes=200,
            peak_active_kv_storage_bytes=400,
            history_updates_per_layer=[127] * 32,
            outstanding_layer_leases=0,
        ),
        source_packet_sha256="same_source",
        generated_history_valid_payload_sha256="same_history",
        persistent_format_bytes=80,
        original_native_KV_retired=True,
        constructor_all_word_validation=True,
        private_source_objects_retired_after_response=True,
    )


def test_only_audited_preflight_can_validate_declared_decode_peak_matching():
    a, b = (record(arm, audit=True) for arm in study.ARMS)
    assert study.match_pair(a, b, audited=True)
    b["storage"]["peak_active_kv_storage_bytes"] = 401
    with pytest.raises(ValueError, match="boundaries differ"):
        study.match_pair(a, b, audited=True)
    a["audited_precheck"] = b["audited_precheck"] = False
    assert study.match_pair(a, b, audited=False)
    b["storage"]["final_active_kv_storage_bytes"] += 1
    with pytest.raises(ValueError):
        study.match_pair(a, b, audited=False)


def complete_trials():
    ordering = study.orders(17)
    rows = []
    for block in ordering:
        for arm in block["method_order"]:
            row = dict(
                record(arm, audit=False), **block, status="completed", all_128_logits_bitwise=True
            )
            row["timing"] = dict(
                full_request_wall_seconds=1 if arm == study.ARMS[0] else 0.9,
                calling_thread_CPU_seconds=0.2,
                process_CPU_seconds=0.3,
                CUDA_stream_interval_ms=500,
            )
            rows.append(row)
    return rows, ordering


def test_no_intersection_of_failed_missing_duplicate_or_changed_observations():
    rows, ordering = complete_trials()
    groups = study.validate_trials(rows, ordering)
    assert all(len(values) == 20 for values in groups.values())
    cases = []
    missing = copy.deepcopy(rows)
    missing.pop()
    cases.append(missing)
    duplicate = copy.deepcopy(rows)
    duplicate[0] = duplicate[1]
    cases.append(duplicate)
    failed = copy.deepcopy(rows)
    failed[3]["status"] = "failed"
    cases.append(failed)
    nan = copy.deepcopy(rows)
    nan[4]["timing"]["full_request_wall_seconds"] = float("nan")
    cases.append(nan)
    for changed in cases:
        with pytest.raises(ValueError):
            study.validate_trials(changed, ordering)


def test_strict_latency_ratio_cannot_be_claimed_from_slower_or_equal_but_uncertain_pairs():
    from mosaickv.evaluation.paired_inline_latency import summarize

    assert summarize([1.0] * 20, [0.9] * 20, seed=13)["paired_latency_engineering_gate"]
    assert not summarize([1.0] * 20, [1.1] * 20, seed=13)["paired_latency_engineering_gate"]
    stats = summarize([1.0] * 20, [0.9, 1.1] * 10, seed=13)
    assert not stats["paired_latency_engineering_gate"]
    assert not stats["superiority_over_other_baselines_established"]


def test_failed_streaming_prerequisite_cannot_start_prepare(tmp_path):
    (tmp_path / study.streamed.GATE).write_text(json.dumps({"passed": False}))
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "worker").mkdir()
    (tmp_path / "worker/result.json").write_text("{}")
    with pytest.raises(ValueError, match="Actual passed immutable V13"):
        study.require_parent(tmp_path)
