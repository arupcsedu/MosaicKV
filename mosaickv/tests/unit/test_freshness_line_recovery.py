from __future__ import annotations

import hashlib
import json
import time

import evicastkv_freshness_line_recovery as subject
import evicastkv_review_freshness_stream as scanner
import pytest


def bound(path, raw):
    path.write_bytes(raw)
    segments = [{"start": 0, "end": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]
    return {
        "signature": scanner.signature(path),
        "segments": segments,
        "segmented_content_root": subject.digest(segments),
    }


def test_lexical_projection_does_not_decode_answer_or_export_values(tmp_path, monkeypatch):
    path = tmp_path / "raw.jsonl"
    raw = b'{"sample_id":"s1","answer":"FORBIDDEN_SENTINEL","media_sha256":"' + b"a" * 64 + b'"}\n'
    binding = bound(path, raw)
    original = json.loads
    decoded = []

    def checked(value, *args, **kwargs):
        assert "FORBIDDEN_SENTINEL" not in value
        decoded.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(scanner.json, "loads", checked)
    result = subject.project_record(
        {"path": str(path), "offset": 0}, binding, [tmp_path], time.monotonic() + 30
    )
    assert decoded
    assert result["counts"]["sample_ids"] == 1
    assert result["counts"]["media_hashes"] == 1
    assert "FORBIDDEN_SENTINEL" not in repr(result)
    assert "s1" not in repr(result)
    assert not result["raw_or_answer_values_exported"]


def test_deterministic_selection_freezes_first_four_distinct_error_locations():
    rows = [
        {"path": f"/runs/{i}.jsonl", "offset": 0, "error": subject.PREFIX}
        for i in (5, 3, 1, 4, 2, 1)
    ]
    result = subject.select_errors({"errors": rows}, 4)
    assert [row["path"] for row in result] == [f"/runs/{i}.jsonl" for i in range(1, 5)]
    with pytest.raises(ValueError):
        subject.select_errors({"errors": rows}, 5)


def test_stale_signature_rejected(tmp_path):
    path = tmp_path / "raw.jsonl"
    binding = bound(path, b'{"id":"old"}\n')
    path.write_bytes(b'{"id":"newdifferent"}\n')
    with pytest.raises(ValueError, match="signature changed"):
        subject.project_record(
            {"path": str(path), "offset": 0}, binding, [tmp_path], time.monotonic() + 30
        )


def test_outside_or_symlink_escape_rejected(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.jsonl"
    binding = bound(outside, b'{"id":"x"}\n')
    alias = allowed / "alias.jsonl"
    alias.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        subject.project_record(
            {"path": str(alias), "offset": 0}, binding, [allowed], time.monotonic() + 30
        )


def test_too_large_is_explicit_not_replaced(tmp_path, monkeypatch):
    path = tmp_path / "raw.jsonl"
    binding = bound(path, b'{"id":"x"}\n')
    monkeypatch.setattr(subject, "MAX_BYTES", 2)
    with pytest.raises(ValueError, match="larger than 128 MiB"):
        subject.project_record(
            {"path": str(path), "offset": 0}, binding, [tmp_path], time.monotonic() + 30
        )


@pytest.mark.parametrize("raw", [b'{"answer":[1,]}\n', b'{"id":"x"}bad\n'])
def test_malformed_record_not_waived(tmp_path, raw):
    path = tmp_path / "raw.jsonl"
    binding = bound(path, raw)
    with pytest.raises(ValueError):
        subject.project_record(
            {"path": str(path), "offset": 0}, binding, [tmp_path], time.monotonic() + 30
        )


def test_deadline_rejected(tmp_path):
    path = tmp_path / "raw.jsonl"
    binding = bound(path, b'{"id":"x"}\n')
    with pytest.raises(TimeoutError, match="deadline"):
        subject.project_record(
            {"path": str(path), "offset": 0}, binding, [tmp_path], time.monotonic() - 1
        )


def test_offset_must_be_registered_record_boundary(tmp_path):
    path = tmp_path / "raw.jsonl"
    binding = bound(path, b'{"id":"x"}\n')
    with pytest.raises(ValueError, match="record boundary"):
        subject.project_record(
            {"path": str(path), "offset": 1}, binding, [tmp_path], time.monotonic() + 30
        )


def test_changed_segment_content_rejected_even_with_same_signature(tmp_path):
    path = tmp_path / "raw.jsonl"
    binding = bound(path, b'{"id":"x"}\n')
    binding["segments"][0]["sha256"] = "0" * 64
    binding["segmented_content_root"] = subject.digest(binding["segments"])
    with pytest.raises(ValueError, match="original recorded segment"):
        subject.project_record(
            {"path": str(path), "offset": 0}, binding, [tmp_path], time.monotonic() + 30
        )


def test_source_mutation_during_projection_rejected(tmp_path, monkeypatch):
    path = tmp_path / "raw.jsonl"
    binding = bound(path, b'{"id":"x"}\n')
    original = subject.identity_projection

    def mutate(raw, **kwargs):
        result = original(raw, **kwargs)
        path.write_bytes(b'{"id":"changed"}\n')
        return result

    monkeypatch.setattr(subject, "identity_projection", mutate)
    with pytest.raises(ValueError, match="signature changed"):
        subject.project_record(
            {"path": str(path), "offset": 0}, binding, [tmp_path], time.monotonic() + 30
        )


def test_preparation_uses_bound_checkpoint_without_projection(tmp_path, monkeypatch):
    path = tmp_path / "raw.jsonl"
    binding = bound(path, b'{"id":"x"}\n')
    state = {"bindings": {str(path): binding}}
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"state": state, "state_sha256": subject.digest(state)}))
    parent = tmp_path / "inventory.json"
    original = {
        "roots": [str(tmp_path)],
        "source_sha": "a" * 40,
        "errors": [{"path": str(path), "offset": 0, "error": subject.PREFIX}],
        "shard_bindings": {str(checkpoint): subject.sha(checkpoint)},
    }
    parent.write_text(json.dumps(original))
    monkeypatch.setattr(subject, "ROOTS", (tmp_path,))
    monkeypatch.setattr(subject, "identity_projection", lambda *args, **kwargs: pytest.fail())
    result = subject.preparation(parent, 4, subject.sha(parent))
    assert result["selected_records"][0]["source_binding"] == binding
    assert json.loads(parent.read_text()) == original
    assert not result["source_traversal_performed"]
    with pytest.raises(ValueError, match="hash binding"):
        subject.preparation(parent, 4, "0" * 64)


def test_execution_requires_cpu_and_inventory_bindings_before_any_access(tmp_path):
    with pytest.raises(ValueError, match="explicit CPU-suite"):
        subject.run(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_failure_slots_preserved_and_no_freshness_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "clean_source", lambda _: "a" * 40)
    monkeypatch.setattr(subject, "cpu_gate", lambda *args: {"status": "passed"})
    rows = [{"path": "/bad/raw.jsonl", "offset": i, "source_binding": None} for i in range(4)]
    monkeypatch.setattr(
        subject,
        "preparation",
        lambda *args: {
            "selected_records": rows,
            "canonical_roots": [str(tmp_path)],
            "original_error_count": 12,
            "original_errors_sha256": "f" * 64,
        },
    )
    gate = subject.run(tmp_path / "out", cpu_gate_root=tmp_path, inventory_sha256="0" * 64)
    assert gate["selected"] == 4
    assert gate["projected"] == 0
    assert gate["original_error_count"] == 12
    assert not gate["fresh_confirmation_authorized"]
    assert not gate["complete_scan"]
    assert all(row["status"] == "blocked_not_waived" for row in gate["results"])
