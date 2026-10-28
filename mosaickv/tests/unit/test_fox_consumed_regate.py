"""Synthetic orchestration contracts, not measured experimental results."""

import copy
import json

import evicastkv_consumed_regate as module
import pytest
from evicastkv_campaign import jobs_for, worker_time_limit

from mosaickv.evaluation.fox_campaign import TASKS, sha, write_once


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    parent, prior, source = (tmp_path / name for name in ("parent", "prior", "source"))
    for name in module.TRANSPORT_FILES:
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# synthetic source\n")
    samples = [
        dict(sample_id=f"{t}-{n:04d}", references=["not_a_selection_signal"])
        for t in TASKS
        for n in (1, 2, 3, 4, 5, 6, 14, 35)
    ]
    ledger = parent / "validation.json"
    write_once(ledger, dict(samples=samples))
    lock = dict(
        cohorts={
            "milebench": {
                "validation": dict(
                    path=str(ledger),
                    sha256=sha(ledger),
                    sample_ids=[s["sample_id"] for s in samples],
                )
            }
        },
        methods={},
        budgets_mib={},
        authorization={},
        resources={},
    )
    write_once(parent / "protocol_lock.json", lock)
    two = [s for s in samples if s["sample_id"] in module.CONSUMED_TWO]
    old_ledger = prior / "two.json"
    write_once(old_ledger, dict(samples=two))
    old_lock = copy.deepcopy(lock)
    old_lock.update(
        source_root=str(source),
        source_sha="synthetic",
        transport_regate=dict(parent_protocol_sha256=sha(parent / "protocol_lock.json")),
        arithmetic=dict(
            allow_fp16_reduced_precision_reduction=True,
            attention_layout_policy="valid_slot_decode",
        ),
    )
    old_lock["cohorts"]["milebench"]["validation"] = dict(
        path=str(old_ledger), sha256=sha(old_ledger), sample_ids=sorted(module.CONSUMED_TWO)
    )
    write_once(prior / "protocol_lock.json", old_lock)
    gate_path = prior / "analysis/correctness/GATE_correctness.json"
    gate = dict(
        status="passed",
        protocol_sha256=sha(prior / "protocol_lock.json"),
        source_sha="synthetic",
        blockers=[],
        tests_actually_executed=dict(expected=8, actual=8, completed=8, pairing_count=4),
    )
    write_once(gate_path, gate)
    monkeypatch.setattr(module, "clean_source", lambda *a: "synthetic")
    return parent, prior, source


def test_extension_reuses_two_and_runs_only_other_thirty(evidence):
    parent, prior, source = evidence
    lock, ledger = module.extension_plan(parent, prior, source=source)
    ids = [s["sample_id"] for s in ledger["samples"]]
    assert len(ids) == len(set(ids)) == 30
    assert not set(ids) & module.CONSUMED_TWO
    assert lock["transport_extension"]["task_scores_used"] is False
    assert not lock["authorization"]["automatic_stage_progression"]
    assert not lock["authorization"]["confirmation_authorized"]
    assert "lookm_reimpl" not in lock["methods"]["milebench"]
    lock["cohorts"]["milebench"]["validation"]["sample_ids"] = ids
    jobs = jobs_for(lock, "correctness")
    assert len(jobs) == 4
    assert sum(len(j["indices"]) * len(j["arms"]) for j in jobs) == 120
    assert worker_time_limit(lock["resources"]) == "00:45:00"


@pytest.mark.parametrize("change", ["failed", "incomplete", "wrong_protocol", "wrong_source"])
def test_failed_or_misbound_controls_rejected(evidence, change):
    parent, prior, source = evidence
    p = prior / "analysis/correctness/GATE_correctness.json"
    gate = json.loads(p.read_text())
    if change == "failed":
        gate["status"] = "failed"
    elif change == "incomplete":
        gate["tests_actually_executed"]["completed"] = 7
    elif change == "wrong_protocol":
        gate["protocol_sha256"] = "wrong"
    else:
        gate["source_sha"] = "wrong"
    p.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="gate must pass"):
        module.extension_plan(parent, prior, source=source)


def test_changed_runtime_cannot_reuse_controls(evidence):
    parent, prior, source = evidence
    other = source.parent / "different_source"
    for name in module.TRANSPORT_FILES:
        file = other / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# changed\n")
    with pytest.raises(ValueError, match="changed transport"):
        module.extension_plan(parent, prior, source=other)


def test_changed_ledger_rejected(evidence):
    parent, prior, source = evidence
    (parent / "validation.json").write_text("{}")
    with pytest.raises(ValueError, match="ledger changed"):
        module.extension_plan(parent, prior, source=source)


@pytest.mark.parametrize("minutes", [0, 121, -1, True, 1.5, "45"])
def test_invalid_registered_worker_limit(minutes):
    with pytest.raises(ValueError, match="wall_minutes"):
        worker_time_limit(dict(wall_minutes=minutes))


def test_default_and_short_registered_worker_limits():
    assert worker_time_limit({}) == "02:00:00"
    assert worker_time_limit(dict(wall_minutes=30)) == "00:30:00"
