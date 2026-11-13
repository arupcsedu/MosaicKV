"""Synthetic binding I/O: no historical files, row decoding, or credentials."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import evicastkv_recover_bindings_v49 as runner
import pytest


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "history"
    root.mkdir()
    monkeypatch.setattr(runner, "HISTORY", root.resolve())
    return root


def artifact(history, name="cache.parquet", raw=b"SYNTHETIC-NO-PARQUET-DECODER"):
    path = history / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def test_limits_and_two_actual_reviewed_writer_paths_remain_exact():
    assert runner.SECONDS == 210
    assert runner.MAX_BYTES == 2 * 2**30
    assert runner.LIVE_BYTES == 8 * 2**20
    assert runner.LIVE_NODES == 100000
    assert {(spec["commit"], spec["writer"]) for spec in runner.WRITERS.values()} == {
        (
            "d845493b4e051d82a836d569fe0ae1692483fa93",
            "mosaickv/scripts/evicastkv_append_counterfactual.py",
        ),
        (
            "11f1b545539d2925635aaa79bebfe74167a608e8",
            "mosaickv/scripts/evicastkv_review_closure_v2.py",
        ),
    }


def test_internal_alias_is_canonicalized_without_opening_file(history, monkeypatch):
    path = artifact(history)
    alias = history / "alias.parquet"
    alias.symlink_to(path)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("unexpected body read"))
    assert runner.allowed(alias) == path


@pytest.mark.parametrize(
    "kind", ["outside", "broken", "credential_target", "credential_name", "codex"]
)
def test_alias_and_credential_paths_fail_before_body_io(history, tmp_path, monkeypatch, kind):
    target = tmp_path / "outside.parquet"
    target.write_bytes(b"synthetic")
    alias = history / "alias.parquet"
    if kind == "outside":
        alias.symlink_to(target)
    elif kind == "broken":
        alias.symlink_to(history / "missing.parquet")
    elif kind == "credential_target":
        target = artifact(history, "credentials.json")
        alias.symlink_to(target)
    elif kind == "credential_name":
        alias = artifact(history, "auth.json")
    else:
        alias = artifact(history, ".codex/fixture.parquet")
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("credential/body opened"))
    with pytest.raises((ValueError, OSError)):
        runner.allowed(alias)


@pytest.mark.parametrize("map_name", runner.MAP_NAMES)
def test_recorded_artifact_maps_resolve_only_explicit_parquet_hashes(history, map_name):
    path = artifact(history)
    value = {map_name: {path.name: "a" * 64, "ignored.txt": "b" * 64, "bad.parquet": "bad"}}
    original = deepcopy(value)
    assert runner.artifact_entries(value, history) == [(path, "a" * 64)]
    assert value == original


def test_output_map_cannot_escape_through_a_symlink(history, tmp_path):
    target = tmp_path / "outside.parquet"
    target.write_bytes(b"synthetic")
    alias = history / "alias.parquet"
    alias.symlink_to(target)
    with pytest.raises(ValueError):
        runner.artifact_entries({"outputs": {str(alias): "a" * 64}}, history)


def test_parent_traversal_and_nonhash_entries_are_not_nominations(history):
    value = {"artifacts": {"../cache.parquet": "a" * 64, "cache.parquet": None, 4: "b" * 64}}
    assert runner.artifact_entries(value, history) == []


def test_metadata_hash_and_signature_are_bound(history):
    path = artifact(history, "manifest.json", b'{"source_sha":"' + b"a" * 40 + b'"}')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    result, actual, signature = runner.metadata(path, runner.Budget(), expected)
    assert result["source_sha"] == "a" * 40
    assert actual == expected
    assert signature == runner.geometry(path)
    with pytest.raises(ValueError, match=r"registered hash|metadata"):
        runner.metadata(path, runner.Budget(), "f" * 64)


def test_metadata_budget_refusal_precedes_any_body_open(history, monkeypatch):
    path = artifact(history, "manifest.json", b"{}")
    budget = runner.Budget()
    budget.input_bytes = runner.MAX_BYTES
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("over-budget body opened"))
    monkeypatch.setattr(
        runner.os, "open", lambda *a, **k: pytest.fail("over-budget descriptor opened")
    )
    with pytest.raises(ValueError, match="combined-read"):
        runner.metadata(path, budget)


def test_oversize_metadata_rejected_before_open(history, monkeypatch):
    path = artifact(history, "manifest.json", b"{}")
    monkeypatch.setattr(runner, "LIVE_BYTES", 1)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("oversized body opened"))
    monkeypatch.setattr(
        runner.os, "open", lambda *a, **k: pytest.fail("oversized descriptor opened")
    )
    with pytest.raises(ValueError, match=r"8-MiB|live|bound"):
        runner.metadata(path, runner.Budget())


def test_metadata_discards_benchmark_payload_before_json_value_decoding(history, monkeypatch):
    path = artifact(
        history,
        "report.json",
        json.dumps(
            {
                "source_sha": "a" * 40,
                "answer": "FORBIDDEN-BENCHMARK-ANSWER",
                "scores": {"payload": "FORBIDDEN-BENCHMARK-ANSWER"},
                "artifacts": {"cache.parquet": "b" * 64},
            }
        ).encode(),
    )
    original = json.loads

    def checked(raw, *args, **kwargs):
        value = raw.decode() if isinstance(raw, bytes) else raw
        assert "FORBIDDEN-BENCHMARK-ANSWER" not in value, "benchmark value decoder was reached"
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(json, "loads", checked)
    value, _, _ = runner.metadata(path, runner.Budget())
    assert value["source_sha"] == "a" * 40
    assert "answer" not in value and "scores" not in value


def test_metadata_descriptor_cannot_follow_postvalidation_credential_alias(
    history, tmp_path, monkeypatch
):
    path = artifact(history, "manifest.json", b"{}")
    protected = tmp_path / "credentials.json"
    protected.write_bytes(b"SYNTHETIC-CREDENTIAL-NEVER-READ")
    original = os.open
    calls = []

    def replaced(name, flags, *args, **kwargs):
        calls.append(flags)
        assert flags & os.O_NOFOLLOW
        path.unlink()
        path.symlink_to(protected)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", replaced)
    with pytest.raises((ValueError, OSError)):
        runner.metadata(path, runner.Budget())
    assert len(calls) == 1


def test_metadata_descriptor_geometry_is_checked_before_value_decoding(history, monkeypatch):
    path = artifact(history, "manifest.json", b"{}")
    signature = runner.geometry(path)
    signature[1] += 1
    monkeypatch.setattr(runner, "geometry", lambda *_args: signature)
    monkeypatch.setattr(
        runner.ProvenanceReader, "value", lambda *a, **k: pytest.fail("unbound descriptor decoded")
    )
    with pytest.raises(ValueError, match="descriptor"):
        runner.metadata(path, runner.Budget())


def test_metadata_mutation_during_projection_cannot_publish_a_claim(history, monkeypatch):
    path = artifact(history, "manifest.json", b"{}")
    original = runner.ProvenanceReader.value

    def modified(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kwargs.get("depth", 0) == 0:
            path.write_bytes(b'{"changed":true}')
        return result

    monkeypatch.setattr(runner.ProvenanceReader, "value", modified)
    with pytest.raises(ValueError, match=r"changed|byte bound"):
        runner.metadata(path, runner.Budget())


@pytest.mark.parametrize("raw", [b' \n {"source_sha":"' + b"a" * 40 + b'"}\t', b"{}\n"])
def test_standard_json_whitespace_does_not_change_projected_provenance(history, raw):
    path = artifact(history, "manifest.json", raw)
    value, actual, signature = runner.metadata(path, runner.Budget())
    assert actual == hashlib.sha256(raw).hexdigest()
    assert signature[2] == len(raw)
    assert value == ({"source_sha": "a" * 40} if b"source_sha" in raw else {})


def test_original_geometry_and_two_independent_hash_passes(history):
    path = artifact(history)
    row = dict(path=str(path), signature=runner.geometry(path))
    budget = runner.Budget()
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert runner.hash_pass(row, budget) == expected
    assert runner.hash_pass(row, budget) == expected
    assert budget.input_bytes == 2 * row["signature"][2]


def test_changed_original_signature_fails_before_hash_reader(history, monkeypatch):
    path = artifact(history)
    row = dict(path=str(path), signature=runner.geometry(path))
    row["signature"][2] += 1
    monkeypatch.setattr(
        runner.resume.reader.common, "hash_file", lambda *a, **k: pytest.fail("changed source read")
    )
    with pytest.raises(ValueError, match="geometry"):
        runner.hash_pass(row, runner.Budget())


def test_hash_budget_refusal_precedes_independent_source_read(history, monkeypatch):
    path = artifact(history)
    row = dict(path=str(path), signature=runner.geometry(path))
    budget = runner.Budget()
    budget.input_bytes = runner.MAX_BYTES
    monkeypatch.setattr(
        runner.resume.reader.common,
        "hash_file",
        lambda *a, **k: pytest.fail("over-budget source read"),
    )
    with pytest.raises(ValueError, match="combined-read"):
        runner.hash_pass(row, budget)


def test_unreviewed_source_sha_is_only_input_provenance(history, monkeypatch):
    metadata_path = history / "report.json"
    monkeypatch.setattr(
        runner, "git_blob", lambda *a, **k: pytest.fail("unreviewed source looked up")
    )
    result = runner.writer_claim(
        metadata_path, {"source_sha": "a" * 40}, "b" * 64, "c" * 64, runner.Budget(), {}
    )
    assert result["role"] == "input_source"
    assert result["source_evidence_verified"] is False
    assert result["writer_path"] is result["source_blob_sha256"] is None


def test_no_recorded_source_sha_remains_artifact_only(history):
    result = runner.writer_claim(
        history / "report.json",
        {"source_sha": "not-a-commit"},
        "b" * 64,
        "c" * 64,
        runner.Budget(),
        {},
    )
    assert result["role"] == "artifact_only"
    assert result["recorded_source_sha"] is None


@pytest.mark.parametrize(
    "failure", ["metadata_hash", "recorded_source", "missing_source", "changed_blob"]
)
def test_reviewed_writer_cannot_upgrade_mismatched_or_unavailable_evidence(monkeypatch, failure):
    path, spec = next(iter(runner.WRITERS.items()))
    recorded, metadata_sha = spec["commit"], spec["metadata_sha256"]
    sources = {}
    if failure == "metadata_hash":
        metadata_sha = "f" * 64
        monkeypatch.setattr(
            runner, "git_blob", lambda *a: pytest.fail("unbound provenance fetched")
        )
    elif failure == "recorded_source":
        recorded = "a" * 40
        monkeypatch.setattr(
            runner, "git_blob", lambda *a: pytest.fail("unbound provenance fetched")
        )
    elif failure == "missing_source":

        def missing(*_args):
            raise subprocess.CalledProcessError(128, ["git", "show"])

        monkeypatch.setattr(runner, "git_blob", missing)
    else:
        monkeypatch.setattr(runner, "git_blob", lambda *a: "f" * 64)
    with pytest.raises((ValueError, subprocess.SubprocessError)):
        runner.writer_claim(
            path, {"source_sha": recorded}, metadata_sha, "b" * 64, runner.Budget(), sources
        )
    assert sources == {}


@pytest.mark.parametrize("which", [0, 1])
def test_only_exact_reviewed_writer_closure_gets_runtime_role(monkeypatch, which):
    path, spec = list(runner.WRITERS.items())[which]
    calls = []

    def blob(commit, source_path, budget):
        calls.append((commit, source_path))
        return spec["blobs"][source_path]

    monkeypatch.setattr(runner, "git_blob", blob)
    sources = {}
    result = runner.writer_claim(
        path,
        {"source_sha": spec["commit"]},
        spec["metadata_sha256"],
        "b" * 64,
        runner.Budget(),
        sources,
    )
    assert result["role"] == "runtime_writer"
    assert result["source_evidence_verified"] is True
    assert result["writer_path"] == spec["writer"]
    assert result["source_blob_sha256"] == spec["blobs"][spec["writer"]]
    assert set(calls) == {(spec["commit"], name) for name in spec["blobs"]}


def test_git_blob_budget_is_checked_before_blob_body_subprocess(monkeypatch):
    calls = []

    def output(command, **_kwargs):
        calls.append(command)
        if command[1:3] == ["cat-file", "-t"]:
            return b"commit\n"
        if command[1:3] == ["cat-file", "-s"]:
            return b"5\n"
        pytest.fail("over-budget Git blob body fetched")

    monkeypatch.setattr(runner.subprocess, "check_output", output)
    budget = runner.Budget()
    budget.input_bytes = runner.MAX_BYTES
    with pytest.raises(ValueError, match="combined-read"):
        runner.git_blob("a" * 40, "mosaickv/scripts/synthetic.py", budget)
    assert len(calls) == 2


def test_expired_budget_never_authorizes_a_read(monkeypatch):
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0)
    budget = runner.Budget()
    monkeypatch.setattr(runner.time, "monotonic", lambda: 211)
    with pytest.raises(TimeoutError, match="210-second"):
        budget.authorize(0)


def test_afterany_missing_recovery_never_reads_original_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "clean_source", lambda *a: "a" * 40)
    monkeypatch.setattr(runner.resume.reader.common, "cpu_gate", lambda *a: None)
    monkeypatch.setattr(runner, "hash_pass", lambda *a: pytest.fail("analysis retried source read"))
    monkeypatch.setattr(
        runner, "metadata", lambda *a: pytest.fail("analysis reopened source metadata")
    )
    result = runner.analyze(tmp_path)
    assert result["status"] == "recovery_failed_or_incomplete"
    assert result["GPU_jobs"] == 0
    assert result["fresh_confirmation_authorized"] is False
    assert result["materialization_authorized"] is False


def test_aggregate_projected_metadata_remains_bounded(monkeypatch):
    runner.projected_bound({"a": [1, 2]})
    monkeypatch.setattr(runner, "LIVE_NODES", 3)
    with pytest.raises(ValueError, match="100,000 projected-node"):
        runner.projected_bound({"a": [1, 2]})


def test_projected_byte_cap_is_not_only_checked_at_export(monkeypatch):
    monkeypatch.setattr(runner, "LIVE_BYTES", 10)
    with pytest.raises(ValueError, match="8-MiB live projected"):
        runner.projected_bound({"hash": "a" * 64})
