"""Synthetic source-capsule I/O; no real archives, data, or old proofs are read."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import evicastkv_archive_recovery_v51 as runner
import pytest


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "history"
    root.mkdir()
    monkeypatch.setattr(runner.reader, "HISTORY", root.resolve())
    monkeypatch.setattr(runner, "HISTORY", root.resolve())
    return root


def file(history, name, raw):
    path = history / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_source_aliases_preserve_original_keys_dirty_true_and_exact_hashes(history):
    original = dict(
        analysis_source_sha="a" * 40,
        analysis_worktree_dirty=True,
        source_code_hashes={"/source/writer.py": "b" * 64},
        derived_artifact_hashes={"result.parquet": "c" * 64},
        generator_sha256="d" * 64,
        producer_source_sha="e" * 40,
        source_analysis_manifest_sha256="f" * 64,
    )
    path, expected = file(history, "report.json", json.dumps(original).encode())
    budget = runner.reader.Budget()
    assert runner.metadata(path, expected, budget) == original
    assert budget.input_bytes == path.stat().st_size


def test_benchmark_answers_and_scores_are_never_scalar_decoded(history, monkeypatch):
    marker = "FORBIDDEN_BENCHMARK_PAYLOAD"
    path, expected = file(
        history,
        "report.json",
        json.dumps(
            dict(
                analysis_source_sha="a" * 40,
                analysis_worktree_dirty=True,
                answers=[marker, {"answer": marker}],
                scores={"score": marker, "nested": [marker]},
                samples=[{"answer": marker, "task_score": 0.5}],
                source_code_hashes={"writer.py": "b" * 64},
            )
        ).encode(),
    )
    decoded = []
    original = runner.reader.lexical._decoded_scalar

    def observed(raw):
        assert marker.encode() not in raw, "benchmark value was scalar decoded"
        decoded.append(raw)
        return original(raw)

    monkeypatch.setattr(runner.reader.lexical, "_decoded_scalar", observed)
    result = runner.metadata(path, expected, runner.reader.Budget())
    assert result == dict(
        analysis_source_sha="a" * 40,
        analysis_worktree_dirty=True,
        source_code_hashes={"writer.py": "b" * 64},
    )
    assert decoded and all(marker.encode() not in raw for raw in decoded)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"analysis_worktree_dirty":true,"analysis_worktree_dirty":false}',
        b'{"source_code_hashes":{"writer.py":"'
        + b"a" * 64
        + b'","writer.py":"'
        + b"b" * 64
        + b'"}}',
        b'{"answers":[1],"answers":[2]}',
    ],
)
def test_duplicate_keys_even_in_discarded_payloads_fail_closed(history, raw):
    path, expected = file(history, "report.json", raw)
    with pytest.raises(ValueError, match="duplicate"):
        runner.metadata(path, expected, runner.reader.Budget())


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b'{"analysis_worktree_dirty":true,}',
        b'{"analysis_source_sha":',
        b'{"source_code_hashes":{"nested":{"bad":"map"}}}',
        b'{"analysis_source_sha":"HEAD"}',
        b"{} {}",
        b'{"analysis_worktree_dirty":true,"answer":"\\q"}',
        b'{"analysis_worktree_dirty":true,"answer":"\xff"}',
    ],
)
def test_malformed_or_unregistered_metadata_cannot_become_a_source_claim(history, raw):
    path, expected = file(history, "report.json", raw)
    with pytest.raises(ValueError):
        runner.metadata(path, expected, runner.reader.Budget())


@pytest.mark.parametrize("name", ["credentials.json", "auth.json", ".codex/report.json"])
def test_protected_metadata_is_rejected_before_descriptor_open(history, monkeypatch, name):
    path, expected = file(history, name, b"SYNTHETIC_SECRET_NEVER_READ")
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("credential opened"))
    with pytest.raises(ValueError, match="credential"):
        runner.metadata(path, expected, runner.reader.Budget())


@pytest.mark.parametrize("failure", ["bytes", "seconds", "metadata_size"])
def test_registered_limits_fail_before_open_without_becoming_larger(history, monkeypatch, failure):
    path, expected = file(history, "report.json", b'{"analysis_worktree_dirty":true}')
    budget = runner.reader.Budget()
    if failure == "bytes":
        budget.input_bytes = runner.reader.MAX_BYTES
    elif failure == "seconds":
        budget.started -= 211
    else:
        monkeypatch.setattr(runner.reader, "LIVE_BYTES", 1)
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("over-limit source opened"))
    with pytest.raises((ValueError, TimeoutError)):
        runner.metadata(path, expected, budget)


def test_complete_metadata_hash_is_required(history):
    path, _ = file(history, "report.json", b'{"analysis_worktree_dirty":true}')
    with pytest.raises(ValueError, match="hash or signature"):
        runner.metadata(path, "f" * 64, runner.reader.Budget())


def test_descriptor_signature_checked_before_any_metadata_decoding(history, monkeypatch):
    path, expected = file(history, "report.json", b'{"analysis_worktree_dirty":true}')
    signature = runner.reader.geometry(path)
    signature[1] += 1
    monkeypatch.setattr(runner.reader, "geometry", lambda *_: signature)
    monkeypatch.setattr(
        runner.CapsuleReader, "value", lambda *a, **k: pytest.fail("unbound value decoded")
    )
    with pytest.raises(ValueError, match="descriptor"):
        runner.metadata(path, expected, runner.reader.Budget())


def test_metadata_mutation_during_projection_cannot_publish_source_content(history, monkeypatch):
    path, expected = file(history, "report.json", b'{"analysis_worktree_dirty":true}')
    original = runner.CapsuleReader.value

    def mutated(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if kwargs.get("depth", 0) == 0:
            path.write_bytes(b'{"analysis_worktree_dirty":false}')
        return value

    monkeypatch.setattr(runner.CapsuleReader, "value", mutated)
    with pytest.raises(ValueError, match=r"changed|registered byte bound"):
        runner.metadata(path, expected, runner.reader.Budget())


def test_postvalidation_symlink_replacement_cannot_open_a_credential(history, monkeypatch):
    path, expected = file(history, "report.json", b'{"analysis_worktree_dirty":true}')
    protected, _ = file(history, "credentials.json", b"SYNTHETIC_SECRET")
    original = os.open

    def raced(name, flags, *args, **kwargs):
        assert flags & os.O_NOFOLLOW
        path.unlink()
        path.symlink_to(protected)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", raced)
    with pytest.raises((ValueError, OSError)):
        runner.metadata(path, expected, runner.reader.Budget())


def test_generator_opens_only_declared_python_twice_and_binds_original_hash(history, monkeypatch):
    raw = b'# SYNTHETIC GENERATOR; NEVER EXECUTE\nraise RuntimeError("do not run")\n'
    path, expected = file(history, "generate.py", raw)
    calls = []
    original = os.open

    def observed(name, flags, *args, **kwargs):
        assert Path(name) == path
        assert flags & os.O_NOFOLLOW
        calls.append(Path(name))
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", observed)
    budget = runner.reader.Budget()
    assert runner.generator_witness(path, expected, budget) == dict(
        path=str(path), sha256=expected, bytes=len(raw)
    )
    assert calls == [path, path]
    assert budget.input_bytes == 2 * len(raw)


def test_generator_wrong_original_hash_fails_after_two_independent_passes(history):
    path, _ = file(history, "generate.py", b"# synthetic source\n")
    budget = runner.reader.Budget()
    with pytest.raises(ValueError, match="original generator code hash"):
        runner.generator_witness(path, "f" * 64, budget)
    assert budget.input_bytes == 2 * path.stat().st_size


@pytest.mark.parametrize(
    "name", ["results.parquet", "report.json", "credentials.json", ".codex/generate.py"]
)
def test_generator_witness_never_opens_data_or_credentials(history, monkeypatch, name):
    path, expected = file(history, name, b"SYNTHETIC_NEVER_OPEN")
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("non-code body opened"))
    with pytest.raises(ValueError):
        runner.generator_witness(path, expected, runner.reader.Budget())


def test_generator_second_pass_mutation_is_not_hidden_by_the_first_hash(history, monkeypatch):
    path, expected = file(history, "generate.py", b"# original\n")
    original = os.open
    count = 0

    def mutated(name, flags, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            path.write_bytes(b"# MODIFIED SOURCE\n")
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", mutated)
    with pytest.raises(ValueError, match="descriptor"):
        runner.generator_witness(path, expected, runner.reader.Budget())
    assert count == 2


def test_generator_budget_refusal_precedes_either_code_open(history, monkeypatch):
    path, expected = file(history, "generate.py", b"# original\n")
    budget = runner.reader.Budget()
    budget.input_bytes = runner.reader.MAX_BYTES
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("over-budget source opened"))
    with pytest.raises(ValueError, match="combined-read"):
        runner.generator_witness(path, expected, budget)


def test_artifact_map_preserves_all_declared_hashes_without_opening_artifacts(history, monkeypatch):
    folder = history / "derived"
    declared = {
        "result.parquet": "a" * 64,
        "analysis_source.zip": "b" * 64,
        "/" + str(folder / "absolute.parquet").lstrip("/"): "c" * 64,
    }
    original = deepcopy(declared)
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("artifact body opened"))
    result = runner.artifact_map(declared, folder)
    assert result == {
        str(folder / "result.parquet"): "a" * 64,
        str(folder / "analysis_source.zip"): "b" * 64,
        str(folder / "absolute.parquet"): "c" * 64,
    }
    assert declared == original


def test_artifact_mount_aliases_are_lexically_normalized_and_collisions_rejected(monkeypatch):
    monkeypatch.setattr(runner, "HISTORY", Path("/sfs/weka/scratch/djy8hg/runs"))
    folder = Path("/sfs/weka/scratch/djy8hg/runs/example")
    aliases = {"/scratch/djy8hg/runs/example/a.parquet": "a" * 64}
    assert runner.artifact_map(aliases, folder) == {str(folder / "a.parquet"): "a" * 64}
    aliases[str(folder / "a.parquet")] = "a" * 64
    with pytest.raises(ValueError, match="duplicate"):
        runner.artifact_map(aliases, folder)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.parquet",
        ".codex/a.parquet",
        "auth.json",
        "credentials.json",
        "/home/user/results.parquet",
        "/scratch/another/runs/a.parquet",
    ],
)
def test_unsafe_or_unknown_artifact_scope_cannot_be_a_nomination(history, name):
    with pytest.raises(ValueError):
        runner.artifact_map({name: "a" * 64}, history)


@pytest.mark.parametrize(
    "values",
    [None, [], {"result.parquet": None}, {1: "a" * 64}, {"result.parquet": "not-original-hash"}],
)
def test_artifact_map_requires_explicit_original_sha_entries(history, values):
    with pytest.raises(ValueError):
        runner.artifact_map(values, history)


def test_afterany_analysis_records_missing_recovery_without_opening_old_proofs(
    tmp_path, monkeypatch
):
    root = tmp_path / "new"
    root.mkdir()
    monkeypatch.setattr(runner, "clean_source", lambda _: "a" * 40)
    monkeypatch.setattr(runner.reader.resume.reader.common, "cpu_gate", lambda *a: None)
    monkeypatch.setattr(
        runner.reader.resume.reader.common.history,
        "read",
        lambda *a: pytest.fail("missing proof was read"),
    )
    assert runner.analyze(root) == dict(
        status="archive_recovery_failed_or_incomplete",
        source_sha="a" * 40,
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )


def test_recovery_reuses_all_134_saved_artifact_transactions_without_parquet_opens(
    history, monkeypatch
):
    root = history / "new-output"
    root.mkdir()
    parent_dir = history / "parent"
    parent_dir.mkdir()
    (parent_dir / "recovered_bindings.json").write_bytes(b"{}")
    monkeypatch.setattr(runner, "PARENT", parent_dir)
    monkeypatch.setattr(runner, "clean_source", lambda _: "a" * 40)
    monkeypatch.setattr(runner.reader.resume.reader.common, "cpu_gate", lambda *a: None)
    rows = []
    for index in range(134):
        path, _ = file(history, f"targets/{index}.parquet", b"SYNTHETIC-NOT-READ")
        rows.append(
            dict(
                path=str(path),
                signature=runner.reader.geometry(path),
                status="artifact_binding_only",
            )
        )
    parent = dict(
        source_sha="b5722db36c7c04d63a8e9fc3062bc2d0e509c279",
        rows=rows,
        parquet_bodies_reopened=0,
        original_occurrence_counts={"net_unresolved_error_count": 135},
    )
    monkeypatch.setattr(runner.reader.resume.reader.common.history, "read", lambda *a: parent)
    report_path = history / "retro/report.json"
    archive_path = history / "retro/analysis_source.zip"
    manifest_path = history / "adhoc/manifest.json"
    monkeypatch.setattr(runner, "REPORT", report_path)
    monkeypatch.setattr(runner, "ZIP", archive_path)
    monkeypatch.setattr(runner, "MANIFEST", manifest_path)
    report = dict(
        analysis_source_sha="b" * 40,
        analysis_worktree_dirty=True,
        source_code_hashes={"/source/writer.py": "c" * 64},
        derived_artifact_hashes={
            **{row["path"]: "d" * 64 for row in rows[:3]},
            str(archive_path): runner.ZIP_SHA,
        },
    )
    manifest = dict(
        generator_sha256="e" * 64, outputs={rows[3]["path"]: "f" * 64}, producer_source_sha="b" * 40
    )
    metadata_calls = []

    def metadata(path, _expected, _budget):
        metadata_calls.append(path)
        assert path in {report_path, manifest_path}
        return deepcopy(report if path == report_path else manifest)

    monkeypatch.setattr(runner, "metadata", metadata)
    monkeypatch.setattr(runner, "inspect_zip", lambda *a, **k: {})
    monkeypatch.setattr(runner, "generator_witness", lambda *a: {})
    monkeypatch.setattr(runner, "validate_archive", lambda *a: {"kind": "dirty_source_archive"})
    monkeypatch.setattr(runner, "validate_generator", lambda *a: {"kind": "ad_hoc_generator"})
    monkeypatch.setattr(
        runner, "augment_binding", lambda row, *_: dict(row, status="synthetic_source_bound")
    )
    original_open = os.open

    def no_parquet_open(path, *args, **kwargs):
        assert Path(path).suffix != ".parquet", "completed artifact transaction was repeated"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", no_parquet_open)
    result = runner.recover(root)
    saved = json.loads((root / "recovered_bindings.json").read_text())
    assert result["recovered"] == 4
    assert saved["reused_hash_transactions"] == 134
    assert saved["parquet_bodies_reopened"] == saved["original_JSON_identity_proofs_replayed"] == 0
    assert saved["newly_resolved_occurrences"] == 0
    assert saved["fresh_confirmation_authorized"] is False
    assert saved["historical_code_executed"] is False
    assert saved["archive_extraction_to_disk"] is False
    assert metadata_calls == [report_path, manifest_path, report_path, manifest_path]


def test_all_registered_resource_limits_remain_unchanged():
    assert runner.reader.SECONDS == 210
    assert runner.reader.MAX_BYTES == 2 * 1024**3
    assert runner.reader.LIVE_BYTES == 8 * 1024**2
    assert runner.reader.LIVE_NODES == 100000
