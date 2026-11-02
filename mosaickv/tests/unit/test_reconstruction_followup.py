from types import SimpleNamespace

import pytest
from evicastkv_reconstruction_followup import helper_gate

from mosaickv.evaluation.reconstruction_policy import current
from mosaickv.evaluation.reconstruction_verification import helper_trials, resources, spills_pass


def trials(ratio=0.8):
    return [
        dict(trial=t, warmup=t < 0, policy=p, cuda_stream_ms=1.0 if p == "original" else ratio)
        for t in range(-5, 20)
        for p in ("original", "reuse_headers")
    ]


def test_promotion_requires_complete_grid_and_registered_positive_helper_gain():
    assert helper_gate(trials(), "reuse_headers", 7)["promote"]
    assert not helper_gate(trials(0.995), "reuse_headers", 7)["promote"]
    assert not helper_gate(trials(1.2), "reuse_headers", 7)["promote"]
    assert helper_gate(trials(), "reuse_headers", 7) == helper_gate(trials(), "reuse_headers", 7)
    for bad in (trials()[:-1], [*trials()[:-1], trials()[0]]):
        with pytest.raises(ValueError):
            helper_gate(bad, "reuse_headers", 7)
    for bad in (float("nan"), 0, -1):
        with pytest.raises(ValueError):
            helper_gate(trials(bad), "reuse_headers", 7)


def test_resource_guard_rejects_unknown_or_increased_spills():
    assert not spills_pass({})
    assert spills_pass({"a": dict(baseline=dict(spills=0), candidate=dict(spills=0))})
    assert not spills_pass({"a": dict(baseline=dict(spills=0), candidate=dict(spills=1))})
    assert not spills_pass({"a": dict(baseline=dict(spills=0), candidate=dict(spills=None))})


def test_resource_audit_checks_original_once_and_restores_policy(monkeypatch):
    import torch

    from mosaickv.evaluation import shared_source_kernel as source

    calls = []

    def project(*args, **kwargs):
        calls.append(current())
        return dict(spills=0, registers=100)

    monkeypatch.setattr(source, "project", project)
    packet = SimpleNamespace(
        shape=(1, 32, 600, 128),
        source=SimpleNamespace(minimum=torch.ones(1)),
        source_head_group=4,
        source_row_tile=16,
    )
    attention = SimpleNamespace(k_proj=SimpleNamespace(weight=torch.ones(1)))
    with resources("reuse_headers") as records:
        assert source.project(packet, attention, (), component=0) is None
        assert source.project(packet, attention, (), component=0) is None
    assert calls == ["original", "reuse_headers", "reuse_headers"]
    assert current() == "original" and spills_pass(records)
    assert source.project is project


def test_helper_requires_owned_released_complete_state_before_gpu_use():
    with pytest.raises(ValueError, match="private"):
        helper_trials(SimpleNamespace(layers=[]), "reuse_headers", 7)


@pytest.mark.parametrize("promoted,expected", [(True, "reuse_both"), (False, "carry_literals")])
def test_one_bounded_followup_without_widening_cohort(monkeypatch, tmp_path, promoted, expected):
    import evicastkv_reconstruction_followup as driver

    calls = []
    plan = dict(study_root=str(tmp_path), attempt_index=0, source_sha="revision")
    monkeypatch.setattr(driver.common.previous.common, "inputs", lambda *a: plan)
    monkeypatch.setattr(
        driver.common.previous,
        "read",
        lambda p: (
            dict(complete=True, failures=[], candidate_promoted=promoted)
            if p.name == "decision.json"
            else dict(order=["reuse_headers", "carry_literals"])
        ),
    )
    monkeypatch.setattr(driver, "prepare_verification", lambda *a: calls.append(a))
    driver.advance(tmp_path / "first", "hash")
    assert calls == [(tmp_path, 1, expected)]


def test_second_attempt_stops_without_more_gpu_or_paper_claim(monkeypatch, tmp_path):
    import evicastkv_reconstruction_followup as driver

    monkeypatch.setattr(
        driver.common.previous.common,
        "inputs",
        lambda *a: dict(study_root=str(tmp_path), attempt_index=1, source_sha="revision"),
    )
    monkeypatch.setattr(
        driver.common.previous,
        "read",
        lambda p: dict(complete=True, failures=[], candidate_promoted=False),
    )
    monkeypatch.setattr(
        driver, "prepare_verification", lambda *a: pytest.fail("unbounded GPU request")
    )
    writes = []
    monkeypatch.setattr(driver, "write_once", lambda p, d: writes.append(d))
    driver.advance(tmp_path / "second", "hash")
    assert len(writes) == 1 and writes[0]["paper_superiority_established"] is False
    assert writes[0]["automatic_reader_attempts_completed"] == 2
