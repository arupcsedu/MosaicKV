"""Synthetic additive recovery: no historical body, auth, media, or model reads."""

import copy
import json
import time
from pathlib import Path

import evicastkv_coverage_resolution_v36 as recovery
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return recovery.sha(path)


@pytest.fixture
def case(tmp_path, monkeypatch):
    history = tmp_path / "history"
    history.mkdir()
    good = history / "config.yaml"
    good.write_text(
        "dataset:\n  id: COCO\n  split: validation\nsample_id: 123\nprompt: PRIVATE_PROMPT\n"
    )
    bad = history / "anchors.yaml"
    bad.write_text("dataset: &private COCO\nother: *private\n")
    parquet = history / "summary.parquet"
    parquet.write_bytes(b"NOT_PARQUET_DO_NOT_READ")
    controller = history / "model-controller.jsonl"
    controller.write_text(
        json.dumps(dict(initialized=True, sample_id="456", answer="PRIVATE_ANSWER"), indent=2)
    )
    errors = [
        dict(path=str(good), error=recovery.yaml_events.YAML_ERROR),
        dict(path=str(bad), error=recovery.yaml_events.YAML_ERROR),
        dict(path=str(parquet), error=recovery.derived.PARQUET_ERROR),
    ]
    errors.extend(
        dict(path=str(controller), error=recovery.JSONL_ERROR, line=offset) for offset in (0, 2, 15)
    )
    old = dict(path=str(history / "previous.json"), error="previously proven")
    errors.append(old)
    errors_path = tmp_path / "original/errors.json"
    put(errors_path, errors)
    before = recovery.sha(errors_path)
    saved = dict(
        original_errors_path=str(errors_path),
        original_error_count=len(errors),
        covered_error_hashes=[recovery.digest(old)],
        bindings={str(errors_path): before},
    )
    monkeypatch.setattr(
        recovery,
        "baseline",
        lambda: (copy.deepcopy(saved), {recovery.digest(error): error for error in errors}),
    )
    monkeypatch.setattr(recovery.scanner, "ROOTS", (history,))
    monkeypatch.setattr(recovery, "OUTPUT_BASE", tmp_path)
    monkeypatch.setattr(recovery, "clean_source", lambda *args: "a" * 40)
    monkeypatch.setattr(recovery, "cpu_gate", lambda *args: {})
    monkeypatch.setattr(
        recovery,
        "EXPECTED_CATEGORIES",
        {
            recovery.yaml_events.YAML_ERROR: 2,
            recovery.derived.PARQUET_ERROR: 1,
            recovery.JSONL_ERROR: 3,
        },
    )
    return dict(
        root=tmp_path / "recovery",
        history=history,
        errors=errors,
        original=errors_path,
        original_sha=before,
        good=good,
        controller=controller,
    )


def run(case):
    recovery.prepare(case["root"])
    for index in range(recovery.SHARDS):
        recovery.worker(case["root"], index)
    return recovery.analyze(case["root"])


def test_complete_denominator_and_original_errors_preserved(case):
    gate = run(case)
    assert gate["original_error_count"] == 7
    assert gate["unique_covered_before"] == 1
    assert gate["net_unresolved_before"] == 6
    assert gate["newly_covered_error_hashes"] == 4
    assert gate["net_unresolved_after"] == 2
    assert gate["proved_files"] == 2
    assert gate["blocked_files"] == 2
    assert gate["error_waivers"] == 0
    assert all(gate[name] is False for name in recovery.FALSE_FIELDS)
    assert recovery.sha(case["original"]) == case["original_sha"]
    assert len(recovery.read(case["original"])) == 7


def test_projection_does_not_export_benchmark_values(case):
    run(case)
    raw = (case["root"] / "coverage_overlay.json").read_text()
    assert "PRIVATE_ANSWER" not in raw and "PRIVATE_PROMPT" not in raw
    overlay = json.loads(raw)
    assert {v for proof in overlay["new_proofs"] for v in proof["identities"]["sample_ids"]} == {
        "123",
        "456",
        "COCO",
    }


def test_five_jsonl_lines_not_waived_single_document_bound(case):
    run(case)
    overlay = recovery.read(case["root"] / "coverage_overlay.json")
    proof = next(
        proof for proof in overlay["new_proofs"] if proof["kind"] == "single_json_document"
    )
    assert len(proof["original_error_sha256s"]) == 3
    assert proof["malformed_lines_skipped"] == 0
    assert proof["suffix_format_mismatch_preserved"] is True


def test_unknown_parquet_body_never_read(case, monkeypatch):
    original = recovery.read_bytes

    def guarded(path, **kwargs):
        assert Path(path).suffix != ".parquet"
        return original(path, **kwargs)

    monkeypatch.setattr(recovery, "read_bytes", guarded)
    result = run(case)
    assert result["net_unresolved_categories"][recovery.derived.PARQUET_ERROR] == 1


@pytest.mark.parametrize(
    "raw",
    [
        b'{"sample_id":"x"}\n{"sample_id":"y"}',
        b'{"sample_id":',
        b'{"sample_id":NaN}',
        b'{"sample_id":"x","sample_id":"y"}',
        b"[]",
    ],
)
def test_json_document_rejects_malformed_or_multiple_values(case, raw):
    case["controller"].write_bytes(raw)
    recovery.prepare(case["root"])
    plan = recovery.read(case["root"] / "plan.json")
    row = next(row for row in plan["queue"] if row["kind"] == "single_json_document")
    with pytest.raises((ValueError, RecursionError)):
        recovery.project_single_json(row, time.monotonic() + 2)


def test_yaml_anchor_remains_blocked(case):
    run(case)
    blocked = [
        item
        for path in (case["root"] / "workers").glob("*.json")
        for item in recovery.read(path)["blocked"]
    ]
    assert len(blocked) == 1
    assert blocked[0]["reason"] == "anchors_and_aliases_not_supported"


def test_mutated_registered_source_rejected(case):
    recovery.prepare(case["root"])
    plan = recovery.read(case["root"] / "plan.json")
    row = next(row for row in plan["queue"] if row["path"] == str(case["good"]))
    case["good"].write_text("sample_id: CHANGED\n")
    with pytest.raises(ValueError, match="signature"):
        recovery.resolve(row, plan["roots"], time.monotonic() + 2, {})


def test_mutated_parent_receipt_rejected(case):
    recovery.prepare(case["root"])
    case["original"].write_text("[]")
    with pytest.raises(ValueError, match="binding changed"):
        recovery.worker(case["root"], 0)


def test_worker_denominator_tampering_rejected(case):
    recovery.prepare(case["root"])
    for index in range(recovery.SHARDS):
        recovery.worker(case["root"], index)
    for path in (case["root"] / "workers").glob("*.json"):
        value = recovery.read(path)
        if value["committed"]:
            value["committed"].clear()
            path.write_text(json.dumps(value))
            break
    with pytest.raises(ValueError, match="denominator"):
        recovery.analyze(case["root"])


def test_missing_worker_never_promoted(case):
    recovery.prepare(case["root"])
    result = recovery.analyze(case["root"])
    assert result["missing_shards"] == list(range(recovery.SHARDS))
    assert result["proved_files"] == 0
    assert result["net_unresolved_after"] == 6
    assert result["fresh_confirmation_authorized"] is False


def test_write_once_run_rejected(case):
    recovery.prepare(case["root"])
    with pytest.raises(ValueError, match="write-once"):
        recovery.prepare(case["root"])
    recovery.worker(case["root"], 0)
    with pytest.raises(FileExistsError):
        recovery.worker(case["root"], 0)


def test_no_auth_body_or_alias_access(tmp_path):
    auth = tmp_path / ".codex/auth.json"
    auth.parent.mkdir()
    auth.write_bytes(b"DO_NOT_READ")
    alias = tmp_path / "safe.json"
    alias.symlink_to(auth)
    with pytest.raises(ValueError, match="credential"):
        recovery.canonical(auth, [tmp_path])
    with pytest.raises(ValueError, match="credential alias"):
        recovery.canonical(alias, [tmp_path])


def test_external_symlink_rejected(tmp_path):
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    inside.mkdir()
    outside.write_text("sample_id: private")
    alias = inside / "alias.yaml"
    alias.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        recovery.canonical(alias, [inside])


def test_global_inventory_requires_exact_hash_pair(case):
    with pytest.raises(ValueError, match="supplied together"):
        recovery.prepare(case["root"], global_inventory=case["original"])
    assert not case["root"].exists()


def test_exact_closed_writer_only_delegates_unchanged_helper(case, monkeypatch):
    path = case["history"] / "attempt/mechanism-terms/key/edges.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PARQUET")
    row = dict(
        path=str(path),
        canonical_path=str(path),
        kind="closed_parquet",
        signature=recovery.scanner.signature(path),
        current_file_sha256=None,
        original_error_sha256s=["d" * 64],
        index=0,
    )
    called = []

    def prove(actual, roots, deadline, cache):
        called.append((actual, roots, deadline, cache))
        return dict(
            path=str(path),
            signature=row["signature"],
            whole_file_sha256_before="e" * 64,
            whole_file_sha256_after="e" * 64,
            original_error_sha256s=row["original_error_sha256s"],
        )

    monkeypatch.setattr(recovery.derived, "prove", prove)
    result = recovery.resolve(row, [case["history"]], time.monotonic() + 2, {})
    assert len(called) == 1
    assert result["registration_sha256"] == recovery.digest(row)


def test_prepare_uses_eight_bounded_cpu_workers(case):
    recovery.prepare(case["root"])
    plan = recovery.read(case["root"] / "plan.json")
    assert plan["shard_count"] == 8
    assert plan["seconds_per_worker"] == 210
    assert plan["maximum_bytes_per_worker"] == 2 * 2**30
    assert len({row["path"] for row in plan["queue"]}) == len(plan["queue"])
    assert all(row["current_file_sha256"] is not None for row in plan["queue"])
