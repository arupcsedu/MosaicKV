"""Synthetic CPU state receipts: no historical scan, labels or inference."""

import json
from pathlib import Path

import evicastkv_history_segment_recovery as recovery
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return recovery.sha(path)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    parent, root = tmp_path / "V13-history", tmp_path / "derived-recovery"
    segment = parent / "segment-9"
    before = parent / "segment-8/combined_identity_inventory.json"
    identities = {key: [] for key in recovery.IDENTITIES}
    identities["sample_ids"] = ["previously-recovered-ID"]
    before_sha = put(before, dict(identities=identities))
    put(parent / "registration.json", dict(source_sha=recovery.OLD_SHA))
    plan_sha = put(
        segment / "plan.json",
        dict(
            source_sha=recovery.OLD_SHA,
            same_scanner_file_sha256=recovery.sha(Path(recovery.scanner.__file__)),
            parent=str(before.parent),
            parent_bindings={str(before): before_sha},
        ),
    )
    put(
        parent / "GATE_history_v13.json",
        dict(
            source_sha=recovery.OLD_SHA,
            status="blocked_history_continuation",
            stop_reason="advance_failed",
            error="ValueError: bound producer artifact exceeds registered byte limit",
            last_frontier=str(segment),
            bindings={str(segment / "plan.json"): plan_sha},
        ),
    )
    put(parent / "jobs/segment-9-submission.json", dict(job="12345"))
    for index in range(8):
        state = dict(
            {key: [] for key in recovery.IDENTITIES},
            errors=[
                dict(path=f"old-{index}.yaml", error="YAML requires separately reviewed projection")
            ],
            frontier=[dict(kind="directory", path=f"old-directory-{index}")],
            traversal_complete=False,
            processed_files=10 + index,
        )
        state["sample_ids"] = [f"consumed-{index}"]
        target = segment / "workers" / f"shard-{index}.json"
        expected = put(
            target,
            dict(
                source_sha=recovery.OLD_SHA,
                index=index,
                state=state,
                state_sha256=recovery.digest(state),
                files_added=2,
                records_added=3,
            ),
        )
        receipt = dict(index=index, frontier_complete=False, files_added=2, worker_sha256=expected)
        stdout = parent / "slurm" / f"segment-9-12345_{index}.out"
        stdout.parent.mkdir(exist_ok=True)
        stdout.write_text(f"clean_git_sha={recovery.OLD_SHA}\n{json.dumps(receipt)}\n")
    monkeypatch.setattr(recovery, "clean_source", lambda *args: "b" * 40)
    monkeypatch.setattr(recovery, "cpu_gate", lambda *args: dict(status="passed"))
    return dict(parent=parent, root=root, cpu=tmp_path)


def prepare(fixture):
    return recovery.prepare(fixture["root"], cpu_gate_root=fixture["cpu"], parent=fixture["parent"])


def test_recovers_all_actual_workers_preserving_old_failed_gate_and_parent_IDs(fixture):
    old_failure = fixture["parent"] / "GATE_history_v13.json"
    before = recovery.sha(old_failure)
    prepare(fixture)
    gate = recovery.run(fixture["root"])
    assert gate["missing_shards"] == [] and gate["pending_frontier_items"] == 8
    assert gate["original_and_new_coverage_errors"] == 8
    assert gate["newly_processed_files"] == 16
    assert gate["historical_worker_source_sha"] == recovery.OLD_SHA
    assert gate["source_sha"] == "b" * 40
    assert not gate["new_scan"] and not gate["fresh_confirmation_authorized"]
    assert recovery.sha(old_failure) == before
    inventory = recovery.Reader().read(fixture["root"] / "combined_identity_inventory.json")
    assert "previously-recovered-ID" in inventory["identities"]["sample_ids"]
    assert len(inventory["identities"]["sample_ids"]) == 9
    with pytest.raises(FileExistsError):
        recovery.run(fixture["root"])


def test_registered_larger_limit_is_still_finite(fixture, monkeypatch):
    monkeypatch.setattr(recovery, "MAX_FILE", 1)
    with pytest.raises(ValueError, match="byte limit"):
        prepare(fixture)
    assert not (fixture["root"] / "plan.json").exists()


def test_missing_receipt_blocks_before_accepting_any_new_scope(fixture):
    (fixture["parent"] / "slurm/segment-9-12345_4.out").unlink()
    with pytest.raises(FileNotFoundError):
        prepare(fixture)
    assert not fixture["root"].exists()


def test_wrong_clean_source_or_worker_hash_blocks(fixture):
    path = fixture["parent"] / "slurm/segment-9-12345_0.out"
    value = path.read_text()
    path.write_text(value.replace(recovery.OLD_SHA, "a" * 40))
    with pytest.raises(ValueError, match="clean producer"):
        prepare(fixture)


def test_bound_worker_mutation_after_preparation_is_detected(fixture):
    prepare(fixture)
    path = fixture["parent"] / "segment-9/workers/shard-7.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        recovery.run(fixture["root"])
    assert not (fixture["root"] / "GATE_history_segment_recovery.json").exists()


def test_worker_self_digest_cannot_be_wrong_despite_matching_receipt(fixture):
    path = fixture["parent"] / "segment-9/workers/shard-2.json"
    value = json.loads(path.read_text())
    value["state_sha256"] = "f" * 64
    expected = put(path, value)
    stdout = fixture["parent"] / "slurm/segment-9-12345_2.out"
    receipt = json.loads(stdout.read_text().splitlines()[1])
    receipt["worker_sha256"] = expected
    stdout.write_text(f"clean_git_sha={recovery.OLD_SHA}\n{json.dumps(receipt)}\n")
    prepare(fixture)
    with pytest.raises(ValueError, match="clean-source receipt"):
        recovery.run(fixture["root"])


def test_unexpected_receipt_payload_is_not_accepted_as_metadata(fixture):
    stdout = fixture["parent"] / "slurm/segment-9-12345_3.out"
    receipt = json.loads(stdout.read_text().splitlines()[1])
    receipt["answer"] = "PRIVATE_LABEL"
    stdout.write_text(f"clean_git_sha={recovery.OLD_SHA}\n{json.dumps(receipt)}\n")
    with pytest.raises(ValueError, match="unexpected worker receipt"):
        prepare(fixture)
    assert not fixture["root"].exists()


def test_wrong_parent_plan_and_mutated_original_failure_are_not_reinterpreted(fixture):
    prepare(fixture)
    old = fixture["parent"] / "GATE_history_v13.json"
    old.write_bytes(old.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        recovery.run(fixture["root"])


def test_total_budget_and_deadline_are_enforced(tmp_path, monkeypatch):
    path = tmp_path / "metadata.json"
    put(path, dict(value=1))
    reader = recovery.Reader()
    monkeypatch.setattr(recovery, "MAX_TOTAL", 1)
    with pytest.raises(ValueError, match="byte limit"):
        reader.read(path)
    monkeypatch.setattr(recovery, "MAX_TOTAL", 1024)
    reader.deadline = 0
    with pytest.raises(TimeoutError):
        reader.read(path)


def test_derived_output_limit_blocks_before_any_inventory_or_error_write(fixture, monkeypatch):
    monkeypatch.setattr(recovery, "MAX_OUTPUT", 1)
    prepare(fixture)
    with pytest.raises(ValueError, match="metadata export exceeds"):
        recovery.run(fixture["root"])
    assert not (fixture["root"] / "combined_identity_inventory.json").exists()
    assert not (fixture["root"] / "original_errors.json").exists()
