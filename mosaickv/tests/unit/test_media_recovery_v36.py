"""Synthetic CPU metadata catalog/privacy guards; never scans real history."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[2] / "scripts/evicastkv_media_recovery_v36.py"
    spec = importlib.util.spec_from_file_location("media_recovery_v36_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    owned = root / "mosaickv-backup"
    owned.mkdir()
    return root, owned


def test_catalog_reads_only_project_filename_metadata_and_no_bodies(
    protocol, tmp_path, monkeypatch
):
    root, owned = project(tmp_path)
    (owned / "source-snapshot.tar.gz").write_text("SECRET SOURCE BODY")
    private = root / "unrelated-private"
    private.mkdir()
    (private / "source-private.patch").write_text("PRIVATE")
    original = Path.read_bytes

    def deny_body(self):
        raise AssertionError("catalog must never open candidate contents")

    monkeypatch.setattr(Path, "read_bytes", deny_body)
    report = protocol.catalog(((root, True),))
    monkeypatch.setattr(Path, "read_bytes", original)
    assert report["complete"] is True
    assert len(report["candidates"]) == 1
    serialized = json.dumps(report)
    assert "source-snapshot.tar.gz" in serialized
    assert "SECRET SOURCE BODY" not in serialized and "unrelated-private" not in serialized
    assert report["candidate_bodies_read"] is False
    assert report["global_exhaustion_proof"] is False


def test_canonical_alias_deduplication_preserves_scope(protocol, tmp_path):
    root, owned = project(tmp_path)
    source = owned / "source.tar"
    source.write_bytes(b"fixture")
    (owned / "source-alias.tar").symlink_to(source)
    report = protocol.catalog(((root, True),))
    assert report["complete"] is True
    assert len(report["candidates"]) == 1
    assert report["duplicate_canonical_targets"] >= 1


@pytest.mark.parametrize("case", ("escape", "broken"))
def test_external_or_broken_backup_alias_is_rejected_without_reading(protocol, tmp_path, case):
    root, owned = project(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    (target / "source.tar").write_text("PRIVATE")
    (owned / "source-alias.tar").symlink_to(
        target / "source.tar" if case == "escape" else target / "missing.tar"
    )
    report = protocol.catalog(((root, True),))
    assert report["complete"] is False
    assert report["candidates"] == []
    assert report["errors"][0]["reason"] == (
        "canonical_target_outside_project_scope"
        if case == "escape"
        else "broken_or_unresolvable_alias"
    )
    assert "PRIVATE" not in json.dumps(report)


def test_shared_root_alias_cannot_enter_another_project(protocol, tmp_path):
    root, _owned = project(tmp_path)
    private = root / "different-project"
    private.mkdir()
    (private / "source-secret.tar").write_text("PRIVATE")
    (root / "mosaickv-backup-alias").symlink_to(private, target_is_directory=True)
    report = protocol.catalog(((root, True),))
    assert report["complete"] is False
    assert not any("source-secret" in item["path"] for item in report["candidates"])


def test_depth_entry_and_time_budgets_are_explicit_and_finite(protocol, tmp_path, monkeypatch):
    root, owned = project(tmp_path)
    deeper = owned / "deeper"
    deeper.mkdir()
    (deeper / "source.tar").write_text("not entered")
    report = protocol.catalog(((root, True),), max_entries=1)
    assert report["complete"] is False and report["stop_reason"] == "entry_limit"
    report = protocol.catalog(((root, True),))
    assert report["candidates"] == []
    ticks = iter((0, 31))
    monkeypatch.setattr(protocol.time, "monotonic", lambda: next(ticks))
    report = protocol.catalog(((root, True),))
    assert report["complete"] is False and report["stop_reason"] == "time_limit"


def test_optional_inventory_is_hashed_not_exported_and_alias_guarded(
    protocol, tmp_path, monkeypatch
):
    root, owned = project(tmp_path)
    path = owned / "combined_identity_inventory.json"
    path.write_text('{"answer":"PRIVATE"}')
    monkeypatch.setattr(protocol, "CATALOG_ROOTS", ((root, True),))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    report = protocol.bind_inventory(path, expected)
    assert report["sha256"] == expected and report["contents_exported"] is False
    assert "PRIVATE" not in json.dumps(report)
    with pytest.raises(ValueError, match="hash"):
        protocol.bind_inventory(path, "a" * 64)
    outside = tmp_path / "combined_identity_inventory.json"
    outside.write_text("PRIVATE")
    alias = owned / "typed_asset_inventory.json"
    alias.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        protocol.bind_inventory(alias, hashlib.sha256(outside.read_bytes()).hexdigest())


def test_cleanup_intent_cannot_enable_freshness_and_report_is_write_once(protocol, tmp_path):
    report = {name: False for name in protocol.REQUIRED_FALSE}
    path = tmp_path / "report.json"
    protocol.publish(path, report)
    with pytest.raises(FileExistsError):
        protocol.publish(path, report)
    for name in protocol.REQUIRED_FALSE:
        with pytest.raises(ValueError, match="cannot waive"):
            protocol.publish(tmp_path / (name + ".json"), dict(report, **{name: True}))


def test_bound_file_rejects_changes_and_limits(protocol, tmp_path):
    path = tmp_path / "cleanup.txt"
    path.write_text("fixture")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert protocol.bound_file(path, expected, maximum=20) == b"fixture"
    with pytest.raises(ValueError, match="bound"):
        protocol.bound_file(path, expected, maximum=1)
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        protocol.bound_file(path, expected, maximum=20)


def test_recover_reuses_hash_bound_prior_report_and_binds_twelve_absences(
    protocol, tmp_path, monkeypatch
):
    runs = tmp_path / "mosaickv-runs"
    runs.mkdir()
    monkeypatch.setattr(protocol, "RUNS", runs)
    monkeypatch.setattr(protocol, "clean_source", lambda source: "b" * 40)
    old = dict(
        pilot_roots=[dict(directory=str(runs / f"pilot-{i}")) for i in range(6)],
        previously_skipped_trace=dict(path="/unread-missing-207MB-trace.json", sha256="a" * 64),
    )
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(old))
    monkeypatch.setattr(protocol, "PARENT", parent)
    monkeypatch.setattr(protocol, "PARENT_SHA", hashlib.sha256(parent.read_bytes()).hexdigest())
    paths = [item["directory"] for item in old["pilot_roots"]]
    paths += [str(runs / "experiments" / name) for name in protocol.EXPERIMENTS]
    cleanup = tmp_path / "cleanup.txt"
    cleanup.write_text("\n".join(paths) + "\n")
    monkeypatch.setattr(protocol, "CLEANUP", cleanup)
    monkeypatch.setattr(protocol, "CLEANUP_SHA", hashlib.sha256(cleanup.read_bytes()).hexdigest())
    monkeypatch.setattr(protocol, "catalog", lambda: dict(complete=True, candidates=[]))
    report = protocol.recover()
    assert len(report["original_path_facts"]) == 12
    assert all(item["exists"] is False for item in report["original_path_facts"])
    assert all(item["cleanup_execution_proven"] is False for item in report["original_path_facts"])
    assert report["prior_gap_report"]["reused_without_trace_rescan"] is True
    assert report["previous_trace_projection_reused"] == old["previously_skipped_trace"]
    assert report["confirmation_blocked"] is True
    assert all(report[name] is False for name in protocol.REQUIRED_FALSE)
    with pytest.raises(ValueError, match="together"):
        protocol.recover(global_inventory=tmp_path / "missing.json")


def test_catalog_root_alias_outside_declared_parent_is_not_enumerated(protocol, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source-secret.tar").write_text("PRIVATE")
    alias = parent / "mosaickv-backup"
    alias.symlink_to(outside, target_is_directory=True)
    report = protocol.catalog(((alias, False),))
    assert report["complete"] is False
    assert report["candidates"] == []
    assert report["errors"][0]["reason"] == "catalog_root_alias_outside_declared_parent"
