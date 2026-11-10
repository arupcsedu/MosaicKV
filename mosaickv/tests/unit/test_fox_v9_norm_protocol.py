"""Synthetic V9 protocol and scheduler checks; no real GPU or benchmark access."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import evicastkv_boundary_shape_validation as shape
import evicastkv_review_v9_norm as control
import pytest


def valid_scope():
    return dict(
        native_norm_boundary=True,
        ordinary_continuation_after_boundary=True,
        initial_cache=dict(bitwise_equal=True, maximum_absolute_difference=0.0),
        final_cache=dict(bitwise_equal=True, maximum_absolute_difference=0.0),
        operation_counts=dict(
            observed_prefill_layers=2,
            normalization_phase_entries=1,
            normalization_phase_completions=1,
            normalization_expected_calls=5,
            normalization_padded_calls=5,
            normalization_input_calls=2,
            normalization_post_attention_calls=2,
            normalization_final_calls=1,
            attention_padded_calls=2,
            linear_padded_calls=14,
        ),
    )


def test_complete_single_boundary_scope_with_exact_cache_passes():
    assert shape.norm_boundary_scope(valid_scope())


@pytest.mark.parametrize(
    "field,value",
    [
        ("normalization_phase_entries", 2),
        ("normalization_phase_completions", 0),
        ("normalization_expected_calls", 3),
        ("normalization_padded_calls", 10),
        ("normalization_input_calls", 1),
        ("normalization_post_attention_calls", 1),
        ("normalization_final_calls", 0),
        ("attention_padded_calls", 4),
        ("linear_padded_calls", 28),
        ("observed_prefill_layers", 0),
    ],
)
def test_incomplete_phase_or_modified_continuation_rejected(field, value):
    record = valid_scope()
    record["operation_counts"][field] = value
    with pytest.raises(ValueError, match="boundary incomplete"):
        shape.norm_boundary_scope(record)


def test_real_cache_difference_is_negative_not_discarded():
    record = valid_scope()
    record["final_cache"] = dict(bitwise_equal=False, maximum_absolute_difference=0.001)
    assert shape.norm_boundary_scope(record) is False
    record["final_cache"]["bitwise_equal"] = True
    with pytest.raises(ValueError, match="cache parity"):
        shape.norm_boundary_scope(record)


def test_norm_replay_evidence_required_before_materialization(tmp_path):
    with pytest.raises(ValueError, match="replay evidence"):
        shape.prepare(tmp_path / "new", tmp_path, tmp_path, native_norm_boundary=True)
    assert not (tmp_path / "new").exists()
    with pytest.raises(ValueError, match="explicit"):
        shape.prepare(tmp_path / "new", tmp_path, tmp_path, native_norm_boundary="yes")


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("both_saved_shape_pairs_reproduced", False),
        ("full_model_parity_established", True),
        ("original_first_logit_tolerance_unchanged", 0.01),
    ],
)
def test_inconclusive_or_relaxed_replay_cannot_authorize_variant(tmp_path, field, value):
    evidence = dict(
        complete=True,
        both_saved_shape_pairs_reproduced=True,
        full_model_parity_established=False,
        original_first_logit_tolerance_unchanged=1e-4,
    )
    evidence[field] = value
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="unrelaxed"):
        shape.prepare(
            tmp_path / "new", tmp_path, tmp_path, native_norm_boundary=True, norm_evidence=path
        )


def test_worker_propagates_optin_without_changing_historical_default(tmp_path, monkeypatch):
    original = shape.boundary.run_paths
    calls = []
    monkeypatch.setattr(shape, "run_paths", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(shape.boundary, "worker", lambda root: shape.boundary.run_paths(1, 2, 3))
    monkeypatch.setattr(shape, "validate_transport", lambda root: dict(native_norm_boundary=True))
    shape.worker(tmp_path)
    assert calls == [dict(native_norm_boundary=True)]
    assert shape.boundary.run_paths is original
    monkeypatch.setattr(shape, "validate_transport", lambda root: {})
    shape.worker(tmp_path)
    assert calls[-1] == dict(native_norm_boundary=False)


def test_run_paths_restores_all_operators_before_ordinary_continuation(monkeypatch):
    import torch

    events = []
    flags = dict(wrapped=False, normalization=False)

    @contextmanager
    def phase():
        flags["normalization"] = True
        try:
            yield
        finally:
            flags["normalization"] = False

    class Operations(dict):
        boundary_normalization = staticmethod(phase)

    @contextmanager
    def transport(*args, **kwargs):
        assert kwargs == dict(native_norm_boundary=True)
        flags["wrapped"] = True
        try:
            yield Operations()
        finally:
            flags["wrapped"] = False
            events.append("restored")

    class Capture:
        records, orders, total_bytes = {}, {}, 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def phase(self, name):
            yield

    def state(n):
        return SimpleNamespace(
            logical_sequence_length=n, active_cache_length=n, next_decode_position=n
        )

    def result(n):
        return SimpleNamespace(
            state=state(n), logits=torch.ones(1, 4), next_token_id=torch.tensor([1])
        )

    def prefill(prepared, **kwargs):
        assert flags["wrapped"] and not flags["normalization"]
        return result(prepared.logical_sequence_length)

    def decode(token, prior, **kwargs):
        events.append((flags["wrapped"], flags["normalization"]))
        return result(prior.logical_sequence_length + 1)

    adapter = SimpleNamespace(prefill=prefill, decode_one_token=decode)
    prepared, prefix = state(6), state(5)
    monkeypatch.setattr(shape, "shape_matched_transport", transport)
    monkeypatch.setattr(shape, "padded_prefix", lambda p: prepared)
    monkeypatch.setattr(shape, "remove_dummy_cache_slot", lambda *a: state(5))
    monkeypatch.setattr(shape, "cache_difference", lambda *a: dict(equal=True))
    monkeypatch.setattr(shape.boundary, "hold_back_last_text_token", lambda p: (prefix, 1))
    output = shape.run_paths(adapter, prepared, Capture(), native_norm_boundary=True)
    assert events == [(True, True), "restored"] + [(False, False)] * 30
    assert output["shape_transport_continuation"]["predictions"] == 16
    assert output["shape_transport_continuation"]["ordinary_continuation_after_boundary"]


def test_gpu_submission_uses_new_launcher_but_same_bounded_resource_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(control.shared.subprocess, "check_output", lambda *a, **kw: "100")
    control.submit(tmp_path, "norm", "norm-worker", [99], gpu=True)
    record = control.shared.read(tmp_path / "jobs/norm-request.json")
    command = record["command"]
    for item in (
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--time=02:00:00",
        "--reservation=bi_fox_dgx",
        "--dependency=afterany:99",
        "--job-name=fox9-norm",
    ):
        assert item in command
    assert command[-4].endswith("evicastkv_review_v9_norm.sbatch")


def test_failed_cpu_gate_never_prepares_model_or_admits_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="a" * 40))

    def failed(*args):
        raise ValueError("CPU suite failed")

    monkeypatch.setattr(control.shared, "cpu_gate", failed)
    monkeypatch.setattr(control, "module", lambda *a: pytest.fail("CPU gate bypassed"))
    monkeypatch.setattr(control, "finish", lambda root: dict(status="blocked"))
    assert control.prepare(tmp_path) == dict(status="blocked")


def test_admission_snapshot_failure_is_recorded_without_submission(tmp_path, monkeypatch):
    inventory = str(control.V8 / "freshness-continuation/aggregate/inventory.json")
    monkeypatch.setattr(
        control, "load", lambda root: dict(source_sha="a" * 40, bindings={inventory: "hash"})
    )
    monkeypatch.setattr(control.shared, "cpu_gate", lambda *a: dict(status="passed"))
    monkeypatch.setattr(
        control,
        "module",
        lambda name: SimpleNamespace(run=lambda *a, **kw: {}, prepare=lambda *a, **kw: {}),
    )

    def failed():
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(control.shared, "gpu_snapshot", failed)
    monkeypatch.setattr(control, "submit", lambda *a, **kw: pytest.fail("blind submission"))
    monkeypatch.setattr(control, "finish", lambda root: dict(status="blocked"))
    assert control.prepare(tmp_path) == dict(status="blocked")
    assert (
        "queue unavailable" in control.shared.read(tmp_path / "errors/gpu-admission.json")["error"]
    )


def test_gpu_worker_refuses_over_six_aggregate(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setattr(control.shared, "gpu_snapshot", lambda: dict(active_gpus=7))
    monkeypatch.setattr(control, "module", lambda *a: pytest.fail("ceiling bypassed"))
    with pytest.raises(ValueError, match="six-A100"):
        control.run(tmp_path, "norm-worker")


def test_analysis_failure_still_builds_review_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})

    def failed(root):
        raise ValueError("missing worker result")

    monkeypatch.setattr(control, "module", lambda name: SimpleNamespace(analyze=failed))
    monkeypatch.setattr(control, "finish", lambda root: dict(status="blocked"))
    assert control.run(tmp_path, "analyze")["status"] == "blocked"
    assert (
        "missing worker result"
        in control.shared.read(tmp_path / "errors/norm-analysis.json")["error"]
    )
