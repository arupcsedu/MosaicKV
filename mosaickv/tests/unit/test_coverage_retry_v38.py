"""Finite resource scheduling regressions; no real historical proof wave."""

import copy
import json
from pathlib import Path

import evicastkv_coverage_retry_v38 as retry
import pytest


def row(name, size, kind="closed_parquet"):
    return dict(
        path=f"/scratch/history/{name}",
        kind=kind,
        signature=[1, 1, size, 1],
        original_error_sha256s=[retry.original.digest(name)],
    )


def test_v36_overflow_is_reproduced_without_relaxing_bound():
    rows = [row(str(i), 4 * 2**20) for i in range(2848)]
    before = copy.deepcopy(rows)
    result = retry.pack_wave(rows)
    assert (
        max(result["original_unconstrained_two_pass_worker_bytes"])
        > retry.original.MAX_WORKER_BYTES
    )
    assert (
        max(2 * count for count in result["worker_source_bytes"]) <= retry.original.MAX_WORKER_BYTES
    )
    assert len(result["queue"]) + len(result["registration_blocked"]) == 2848
    assert result["registration_blocked"]
    assert rows == before
    assert result["resource_bounds_raised"] is False
    assert all(item["original_error_unresolved"] for item in result["registration_blocked"])


def test_small_lexical_metadata_precedes_parquet_resources(monkeypatch):
    monkeypatch.setattr(retry.original, "SHARDS", 2)
    monkeypatch.setattr(retry.original, "MAX_WORKER_BYTES", 128)
    rows = [
        row("big-a", 64),
        row("big-b", 64),
        row("small-yaml", 8, "yaml_events"),
        row("small-controller", 8, "single_json_document"),
    ]
    result = retry.pack_wave(rows)
    assert {r["kind"] for r in result["queue"]} == {"yaml_events", "single_json_document"}
    assert len(result["registration_blocked"]) == 2
    assert all(item["reason"] == retry.DEFERRED_REASON for item in result["registration_blocked"])


def test_schedule_is_deterministic_and_preserves_every_error_hash():
    rows = [row(f"p-{i}", (i + 1) * 1000000) for i in range(200)]
    one, two = retry.pack_wave(rows), retry.pack_wave(list(reversed(rows)))
    assert one == two
    all_rows = one["queue"] + [item["row"] for item in one["registration_blocked"]]
    assert {r["path"] for r in all_rows} == {r["path"] for r in rows}
    assert {h for r in all_rows for h in r["original_error_sha256s"]} == {
        h for r in rows for h in r["original_error_sha256s"]
    }


@pytest.mark.parametrize("value", [True, -1, "3"])
def test_strict_file_size_required(value):
    with pytest.raises(ValueError, match="file size"):
        retry.pack_wave([row("bad", value)])


def test_duplicate_candidate_rejected():
    value = row("same", 1)
    with pytest.raises(ValueError, match="distinct"):
        retry.pack_wave([value, value])


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    original = retry.original
    history = tmp_path / "history"
    history.mkdir()
    yaml = history / "config.yaml"
    yaml.write_text("sample_id: synthetic-id\nprompt: PRIVATE_PROMPT\n")
    document = history / "controller.jsonl"
    document.write_text('{\n  "sample_id": "synthetic-controller"\n}\n')
    unknown = history / "summary.parquet"
    unknown.write_bytes(b"DO_NOT_READ_PARQUET_BODY")
    errors = [
        dict(path=str(yaml), error=original.yaml_events.YAML_ERROR),
        dict(path=str(document), error=original.JSONL_ERROR, line=0),
        dict(path=str(document), error=original.JSONL_ERROR, line=2),
        dict(path=str(unknown), error=original.derived.PARQUET_ERROR),
    ]
    preserved = tmp_path / "preserved_errors.json"
    preserved.write_text(json.dumps(errors))
    evidence = dict(
        original_errors_path=str(preserved),
        original_error_count=4,
        covered_error_hashes=[],
        bindings={str(preserved): original.sha(preserved)},
    )
    monkeypatch.setattr(
        original,
        "baseline",
        lambda: (copy.deepcopy(evidence), {original.digest(error): error for error in errors}),
    )
    monkeypatch.setattr(
        original,
        "EXPECTED_CATEGORIES",
        {
            original.yaml_events.YAML_ERROR: 1,
            original.JSONL_ERROR: 2,
            original.derived.PARQUET_ERROR: 1,
        },
    )
    monkeypatch.setattr(original.scanner, "ROOTS", (history,))
    monkeypatch.setattr(original, "OUTPUT_BASE", tmp_path)
    monkeypatch.setattr(original, "clean_source", lambda *args: "a" * 40)
    monkeypatch.setattr(retry, "clean_source", lambda *args: "a" * 40)
    monkeypatch.setattr(original, "cpu_gate", lambda *args: {})
    failed = tmp_path / "original-failure.json"
    failed.write_text(
        json.dumps(
            dict(
                complete=False,
                phase="prepare",
                error_type="ValueError",
                error_waivers=0,
                fresh_confirmation_authorized=False,
            )
        )
    )
    monkeypatch.setattr(retry, "FAILED_RECEIPT", failed)
    monkeypatch.setattr(retry, "FAILED_SHA", original.sha(failed))
    return tmp_path / "retry", preserved


def test_retry_retains_original_proof_semantics_and_denominators(prepared):
    root, preserved = prepared
    before = retry.original.sha(preserved)
    report = retry.prepare(root)
    assert report["workers"] == 8
    for index in range(8):
        retry.worker(root, index)
    gate = retry.analyze(root)
    assert gate["original_error_count"] == 4
    assert gate["newly_covered_error_hashes"] == 3
    assert gate["net_unresolved_after"] == 1
    assert gate["blocked_files"] == 1
    assert gate["resource_bounds_raised"] is False
    assert retry.original.sha(preserved) == before
    assert all(gate[field] is False for field in retry.original.FALSE_FIELDS)
    assert "PRIVATE_PROMPT" not in (root / "coverage_overlay.json").read_text()


def test_unknown_parquet_is_not_body_hashed(prepared, monkeypatch):
    root, _ = prepared
    read = retry.original.read_bytes

    def guarded(path, **kwargs):
        assert Path(path).suffix != ".parquet"
        return read(path, **kwargs)

    monkeypatch.setattr(retry.original, "read_bytes", guarded)
    assert retry.prepare(root)["blocked_files"] == 1


def test_original_failure_receipt_mutation_rejects_retry(prepared):
    root, _ = prepared
    retry.FAILED_RECEIPT.write_text("{}")
    with pytest.raises(ValueError, match="hash changed"):
        retry.prepare(root)
    assert not root.exists()


def test_worker_is_the_unchanged_original_delegate(monkeypatch):
    seen = []
    monkeypatch.setattr(retry.original, "worker", lambda root, index: seen.append((root, index)))
    retry.worker("/scratch/fixture", 3)
    assert seen == [("/scratch/fixture", 3)]


def test_no_automatic_second_wave_or_permission_broadening(prepared):
    root, _ = prepared
    retry.prepare(root)
    for index in range(8):
        retry.worker(root, index)
    gate = retry.analyze(root)
    assert gate["automatic_resubmission"] is False
    assert gate["resource_deferred_files"] == 0
    assert gate["fresh_confirmation_authorized"] is False
    with pytest.raises(ValueError, match="immutable"):
        retry.prepare(root)
