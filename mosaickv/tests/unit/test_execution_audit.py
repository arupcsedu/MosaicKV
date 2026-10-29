"""Instrumentation must observe, never decide or alter cache semantics."""

from dataclasses import replace

import numpy as np
import pytest
from evicastkv_consumed_instrumentation import check_equivalence, logit_capture, replay_routing, sha
from test_hf_runtime import _capabilities, _config, _forecast, _full

from mosaickv.backends.hf_runtime import _uncertain_head_routes, build_compression_plan
from mosaickv.config import CacheConfig, PrototypeConfig
from mosaickv.execution_audit import audit_operation, audit_phase, collect_execution_audit
from mosaickv.types import BudgetUnit, MosaicKVMethod


def test_nested_spans_report_exclusive_not_double_counted(monkeypatch):
    import mosaickv.execution_audit as module

    ticks = iter([0.0, 1.0, 3.0, 5.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    with collect_execution_audit() as audit, audit_phase("parent"), audit_phase("child"):
        pass
    p, c = audit.payload()["phases"]
    assert p["inclusive_wall_seconds"] == 5
    assert p["exclusive_wall_seconds"] == 3
    assert c["exclusive_wall_seconds"] == 2
    assert c["parent"] == 0


def test_exception_and_nested_request_collectors_are_isolated():
    with collect_execution_audit() as outer:
        with pytest.raises(ValueError, match="fixture"), audit_phase("failure"):
            raise ValueError("fixture")
        with collect_execution_audit() as inner, audit_phase("inner"):
            pass
        with audit_phase("restored"):
            pass
    assert [p["operation"] for p in outer.payload()["phases"]] == ["failure", "restored"]
    assert outer.phases[0]["status"] == "failed"
    assert [p["operation"] for p in inner.payload()["phases"]] == ["inner"]
    with audit_phase("disabled"):
        pass
    assert len(outer.phases) == 2


def test_disabled_instrumentation_preserves_identity_and_exception():
    @audit_operation("identity")
    def identity(x):
        return x

    value = np.arange(8)
    assert identity(value) is value
    with collect_execution_audit() as audit:
        assert identity(value) is value
    assert len(audit.phases) == 1


def test_full_routing_scores_replay_without_changing_selection():
    scores = {(0, 4): 0.1, (1, 0): 0.9, (2, 0): 0.9, (3, 2): 0.4}
    kwargs = dict(fraction=0.5, threshold=0.0, label="fixture")
    expected = _uncertain_head_routes(scores, scores, **kwargs)
    with collect_execution_audit() as audit:
        selected = _uncertain_head_routes(scores, scores, **kwargs)
    assert expected == selected == ((1, 0), (2, 0))
    row = audit.routing[0]
    assert len(row["all_scores"]) == 4
    assert replay_routing(row) == [[1, 0], [2, 0]]
    row["selected"] = [[0, 4]]
    with pytest.raises(ValueError, match="does not replay"):
        replay_routing(row)


def test_routing_threshold_fallback_is_recorded_and_replayable():
    scores = {(0, 1): 0.1, (0, 2): 0.2}
    with collect_execution_audit() as audit:
        _uncertain_head_routes(scores, scores, fraction=0.1, threshold=0.9, label="fallback")
    assert replay_routing(audit.routing[0]) == [[0, 2]]


@pytest.mark.parametrize("mode", ["source_views", "batched_layer"])
def test_exact_tier_instrumentation_preserves_kv_memberships_and_bytes(mode):
    full = _full()
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5),
        method=MosaicKVMethod.MOSAICKV_EXACT,
        cache=CacheConfig(
            retention_ratio=0.5,
            block_size=1,
            budget_unit=BudgetUnit.BLOCKS,
            budget_value=4,
            exact_gather_mode=mode,
        ),
        prototypes=PrototypeConfig(enabled=False),
    )
    kwargs = dict(
        full_state=full,
        forecast=_forecast(),
        forecast_attention_by_node=dict.fromkeys(range(8), 0.125),
        capabilities=_capabilities(safe=True),
        config=cfg,
    )
    baseline = build_compression_plan(**kwargs)
    with collect_execution_audit() as audit:
        profiled = build_compression_plan(**kwargs)
    assert baseline.selection.selected_node_ids == profiled.selection.selected_node_ids
    first, second = baseline.construction.state, profiled.construction.state
    assert first.logical_positions == second.logical_positions
    assert first.statistics == second.statistics
    for a, b in zip(first.exact.key_blocks, second.exact.key_blocks, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(first.exact.value_blocks, second.exact.value_blocks, strict=True):
        np.testing.assert_array_equal(a, b)
    names = {p["operation"] for p in audit.phases}
    assert {
        "tier.complete_construction",
        "exact.tier_invariants",
        "exact.descriptor_gather_and_views",
        "state.create_and_account_storage",
        "state.membership_and_storage_invariants",
        "tier.active_layout_metadata",
    } <= names


def test_equivalence_rejects_storage_and_token_regressions():
    metrics = dict(
        initial_active_kv_storage_bytes=100,
        final_active_kv_storage_bytes=120,
        peak_active_kv_storage_bytes=120,
        residual_kv_bytes=0,
        repair_count=0,
    )
    control = dict(
        tokens=[1, 2], catalogs={"positions": [0, 3]}, metrics=metrics, maximum_logit_difference=0
    )
    check_equivalence(control, control, 0.0001)
    for candidate in (
        dict(control, tokens=[1, 3]),
        dict(control, maximum_logit_difference=0.1),
        dict(control, maximum_logit_difference=float("nan")),
        dict(control, metrics=dict(metrics, final_active_kv_storage_bytes=121)),
    ):
        with pytest.raises(ValueError):
            check_equivalence(control, candidate, 0.0001)


def test_logit_capture_retains_full_vocabulary_and_is_independent():
    import torch

    values = torch.arange(30, dtype=torch.float32).reshape(1, 3, 10)
    first, second = [], []
    capture = logit_capture(first)
    capture(None, None, values)
    np.testing.assert_array_equal(first[0], np.arange(20, 30))
    values.zero_()
    np.testing.assert_array_equal(first[0], np.arange(20, 30))
    assert not second
    with pytest.raises(ValueError, match="vocabulary"):
        logit_capture(second)(None, None, torch.zeros(1, 10))


def test_cuda_profile_uses_events_and_synchronizes_both_boundaries():
    from types import SimpleNamespace

    calls = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            calls.append("event")

        def elapsed_time(self, other):
            return 12.5

    fake = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, synchronize=lambda: calls.append("sync"))
    )
    with collect_execution_audit(torch_module=fake) as audit, audit_phase("fixture"):
        pass
    assert calls == ["sync", "event", "event", "sync"]
    assert audit.phases[0]["cuda_event_elapsed_seconds"] == 0.0125
    assert audit.payload()["profiler_contaminated"]


def test_manifest_hash_accepts_serialized_string_paths(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text("{}")
    assert sha(str(path)) == sha(path)


@pytest.mark.parametrize("registration_fails", [False, True])
def test_launch_registers_recognized_worker_and_preserves_callback_before_tick(
    tmp_path, monkeypatch, registration_fails
):
    import control_a100_schedule
    import evicastkv_consumed_instrumentation as module

    calls = []
    monkeypatch.setattr(module, "inputs", lambda *args: {})

    def submit(command, **kwargs):
        calls.append(command)
        return "12345\n" if len(calls) == 1 else "12346\n"

    monkeypatch.setattr(module.subprocess, "check_output", submit)

    def tick(*args):
        assert module.read(tmp_path / "submission.json")["cpu_analysis"] == "12346"
        if registration_fails:
            raise ValueError("synthetic registration failure")
        return {"registered": True}

    monkeypatch.setattr(control_a100_schedule, "tick", tick)
    if registration_fails:
        with pytest.raises(ValueError, match="synthetic registration"):
            module.submit(tmp_path, "fixture")
    else:
        module.submit(tmp_path, "fixture")
    assert "--job-name=mk-evicast-consumed-profile" in calls[0]
    assert "--dependency=afterany:12345" in calls[1]
    assert module.read(tmp_path / "submissions/gpu.json")["job_id"] == "12345"
