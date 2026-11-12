"""Synthetic retry wiring: no historical scan, scheduler or model access."""

import json
from copy import deepcopy

import evicastkv_metadata_retry_v38 as driver
import pytest

from mosaickv.evaluation import coco_history_resume as prior


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    parent, root = tmp_path / "parent", tmp_path / "retry"
    source = "1e90ac0c1190809596a820c636b543473f55696f"
    origin = dict(
        source_sha=source,
        roots=["/synthetic"],
        seed_error_counts={str(i): 0 for i in range(8)},
        preserved_resolved_error_sha256s=[],
    )
    state = {key: [] for key in (*prior.SET_FIELDS, *prior.APPEND_FIELDS)}
    state.update({key: {} for key in prior.DICT_FIELDS})
    state.update({key: 0 for key in prior.COUNTER_FIELDS})
    state.update(
        frontier=[],
        traversal_complete=False,
        verification_index=0,
        verification_paths=["/synthetic/a"],
        errors=[],
        processed_files=1,
        projected_records=1,
        empty_parquet_projection={},
    )
    state["bindings"] = {"/synthetic/a": dict(signature=[])}
    state["seen_files"] = ["/synthetic/a"]
    path = parent / "state.json"
    save(path, state)
    heads = {
        str(i): dict(
            path=str(path),
            sha256=driver.sha(path),
            container=False,
            state_sha256=driver.digest(state),
        )
        for i in range(8)
    }
    gate = dict(
        source_sha=source,
        remaining_pending_items=0,
        complete=False,
        current_time_verification_epoch_performed=True,
        reason="failed_execution_partition",
        heads=heads,
    )
    save(parent / "registration.json", origin)
    save(parent / "GATE_global_history_v36.json", gate)
    monkeypatch.setattr(driver, "PARENT", parent)
    monkeypatch.setattr(
        driver, "PARENT_GATE_SHA", driver.sha(parent / "GATE_global_history_v36.json")
    )
    monkeypatch.setattr(driver, "clean_source", lambda *args: "b" * 40)
    monkeypatch.setattr(driver.old, "source_bridge", lambda: {})
    monkeypatch.setattr(driver, "cpu_gate", lambda *args: {})
    monkeypatch.setattr(driver.old, "put", save)
    return root, state, gate


def test_registration_preserves_saved_verification_heads(case):
    root, _state, gate = case
    plan = driver.register(root)
    assert plan["heads"] == gate["heads"]
    assert plan["original_traversal_replayed"] is False
    assert plan["scanner_seconds"] == 210
    assert plan["scanner_source_byte_bound"] == 2 * 2**30
    assert plan["fresh_confirmation_authorized"] is False
    assert driver.load(root) == plan


def test_loader_rejects_widened_scope(case):
    root, _, _ = case
    plan = driver.register(root)
    plan["roots"] = ["/elsewhere"]
    save(root / "registration.json", plan)
    with pytest.raises(ValueError, match="binding changed"):
        driver.load(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scanner_seconds", 211),
        ("fresh_confirmation_authorized", True),
        ("maximum_steps_per_owner", 7),
    ],
)
def test_loader_rejects_protocol_drift(case, field, value):
    root, _, _ = case
    plan = driver.register(root)
    plan[field] = value
    save(root / "registration.json", plan)
    with pytest.raises(ValueError):
        driver.load(root)


def test_retry_uses_saved_epoch_and_never_traverses(case, monkeypatch):
    root, _, _ = case
    driver.register(root)
    calls = []

    def advance(base, roots, scanner, *, owner_index):
        calls.append((deepcopy(base), roots, owner_index))
        after = dict(base, verification_index=1, traversal_complete=True)
        return dict(state=after, state_sha256=driver.digest(after))

    monkeypatch.setattr(driver.old, "modules", lambda: (object(),))
    monkeypatch.setattr(driver.verification, "advance_verification", advance)
    monkeypatch.setattr(driver.verification, "verify_verification", lambda *a, **k: True)
    decision = driver.worker(root, 3)
    assert decision["complete"] is True
    assert len(calls) == 1 and calls[0][2] == 3
    assert calls[0][0]["frontier"] == []
    assert decision["original_traversal_replayed"] is False
    assert decision["fresh_confirmation_authorized"] is False


@pytest.mark.parametrize("index", [True, -1, 8])
def test_invalid_original_owner_rejected(case, index):
    root, _, _ = case
    driver.register(root)
    with pytest.raises(ValueError, match="owner index"):
        driver.worker(root, index)


def test_cpu_retry_changes_infrastructure_only(case, monkeypatch):
    root, _, _ = case
    observed = []
    monkeypatch.setenv("PYTEST_ADDOPTS", "")
    closure = driver.importlib.import_module("evicastkv_review_closure_v2")
    monkeypatch.setattr(closure, "cpu_suite", lambda *a, **k: observed.append((a, k)))
    driver.cpu(root)
    assert observed[0][1] == dict(timeout_seconds=1200, capture_mode="sys")
    assert "faulthandler_timeout=60" in driver.os.environ["PYTEST_ADDOPTS"]


def test_cpu_failure_submits_no_orphan_dependents(case, monkeypatch):
    root, _, _ = case
    driver.register(root)

    def fail(*args):
        raise ValueError("incomplete CPU suite")

    monkeypatch.setattr(driver, "cpu_gate", fail)
    result = driver.dispatch(root)
    assert result["complete"] is False and result["worker_jobs_submitted"] == 0
    assert not (root / "jobs/verify-request.json").exists()


def test_passed_cpu_dispatches_only_verification_then_afterany_analysis(case, monkeypatch):
    root, _, _ = case
    driver.register(root)
    commands = []
    replies = iter(["123\n", "124\n"])

    def submit(command, **kwargs):
        commands.append(command)
        return next(replies)

    monkeypatch.setattr(driver.subprocess, "check_output", submit)
    result = driver.dispatch(root)
    assert result["verification_array"] == "123" and result["analysis_job"] == "124"
    assert "--array=0-7%8" in commands[0]
    assert "--dependency=afterany:123" in commands[1]
    assert not any("gres=" in part for command in commands for part in command)
    assert result["fresh_confirmation_authorized"] is False
