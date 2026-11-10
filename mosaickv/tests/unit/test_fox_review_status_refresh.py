from __future__ import annotations

import json

import evicastkv_review_status_refresh as subject
import pytest


def put(path, value):
    path.write_text(json.dumps(value))
    return path


def history():
    return {
        "requirements": [{"id": f"R{i:02}", "requirement": f"Review {i}"} for i in range(1, 11)],
        "evidence": {},
    }


def test_missing_evidence_is_not_measured_or_admissible(tmp_path):
    evidence, failures = subject.inventory({"mechanisms": tmp_path / "absent.json"})
    matrix = subject.build_matrix(history(), evidence, failures)
    assert matrix["summary"]["mechanism_rows"] is None
    assert matrix["requirements"][0]["measurement_status"].startswith("blocked")
    assert failures
    assert not matrix["superiority_established"]


def test_tampered_hash_rejected(tmp_path):
    path = put(tmp_path / "gate.json", {"status": "passed"})
    expected = subject.sha(path)
    put(path, {"status": "passed", "forged": True})
    evidence, failures = subject.inventory({"gate": {"path": path, "sha256": expected}})
    assert not evidence["gate"]["verified"]
    assert "hash mismatch" in failures[0]["error"]


def test_embedded_link_tamper_rejected(tmp_path):
    raw = put(tmp_path / "raw.json", {"rows": 3})
    parent = put(tmp_path / "gate.json", {"bindings": {str(raw): subject.sha(raw)}})
    put(raw, {"rows": 4})
    _, failures = subject.inventory({"gate": parent})
    assert failures


def test_reused_artifact_change_is_not_hidden_by_deduplication(tmp_path, monkeypatch):
    raw = put(tmp_path / "raw.json", {"rows": 3})
    original = subject.sha
    calls = 0

    def changing(path):
        nonlocal calls
        if subject.Path(path) == raw:
            calls += 1
            if calls > 2:
                return "0" * 64
        return original(path)

    monkeypatch.setattr(subject, "sha", changing)
    _, failures = subject.inventory({"first": raw, "second": raw})
    assert failures[0]["evidence_id"] == "second"
    assert "changed during inventory" in failures[0]["error"]


def test_negative_gate_is_preserved_and_counts_are_not_hardcoded(tmp_path):
    path = put(
        tmp_path / "source.json",
        {
            "actual_rows": 9,
            "all_source_variants_pass_registered_guard": False,
            "status": "complete",
        },
    )
    evidence, failures = subject.inventory({"source_prefix": path})
    matrix = subject.build_matrix(history(), evidence, failures)
    assert matrix["summary"]["source_fixed_prefix_rows"] == 9
    assert matrix["summary"]["source_all_registered_guards_pass"] is False
    assert evidence["source_prefix"]["verified"]
    assert not matrix["candidate_freeze_authorized"]


def test_completed_scheduler_or_positive_diagnostic_cannot_grant_paper_admission(tmp_path):
    gate = put(
        tmp_path / "fake.json",
        {"state": "COMPLETED", "superiority_established": True, "paper_campaign_authorized": True},
    )
    evidence, failures = subject.inventory({"current_numerical": gate})
    matrix = subject.build_matrix(history(), evidence, failures, diagnostics_present=True)
    assert not matrix["superiority_established"]
    assert not matrix["paper_campaign_authorized"]
    assert matrix["current_diagnostics_status"].startswith("provided")
    assert matrix["summary"]["historical_full_model_norm_parity"] is None


def test_readonly_export_write_once_and_disclosure_review(tmp_path, monkeypatch):
    historical = put(tmp_path / "old.json", history())
    before = historical.read_bytes()
    monkeypatch.setattr(subject, "HISTORICAL", historical)
    monkeypatch.setattr(subject, "SOURCES", {})
    root = tmp_path / "derived"
    gate = subject.run(root)
    assert gate["requirement_rows"] == 10
    assert historical.read_bytes() == before
    matrix = json.loads((root / "review_matrix.json").read_text())
    assert matrix["current_diagnostics_status"] == "pending_not_supplied"
    assert "all-author review" in matrix["proposed_LLM_disclosure_for_author_review"]
    assert not gate["author_review_or_approval_certified"]
    assert (root / "review_matrix.md").read_text().startswith("# Prof. Fox")
    with pytest.raises(FileExistsError):
        subject.run(root)


def test_optional_diagnostics_are_hash_bound(tmp_path, monkeypatch):
    historical = put(tmp_path / "old.json", history())
    current = put(tmp_path / "norm.json", {"full_model_parity": True})
    monkeypatch.setattr(subject, "HISTORICAL", historical)
    monkeypatch.setattr(subject, "SOURCES", {})
    subject.run(
        tmp_path / "derived",
        current_diagnostics={"numerical": {"path": current, "sha256": subject.sha(current)}},
    )
    matrix = json.loads((tmp_path / "derived/review_matrix.json").read_text())
    assert matrix["evidence"]["current_numerical"]["sha256"] == subject.sha(current)
    assert matrix["current_diagnostic_observations"]["current_numerical"] == {
        "full_model_parity": True
    }
    assert not matrix["paper_campaign_authorized"]


def test_historical_document_hash_is_verified_without_json_parsing(tmp_path, monkeypatch):
    document = tmp_path / "contract.md"
    document.write_text("Historical measurement scope.\n")
    old = history()
    old["evidence"]["contract"] = {"path": str(document), "sha256": subject.sha(document)}
    historical = put(tmp_path / "old.json", old)
    monkeypatch.setattr(subject, "HISTORICAL", historical)
    monkeypatch.setattr(subject, "SOURCES", {})
    gate = subject.run(tmp_path / "derived")
    assert gate["evidence_failures"] == []
    assert document.read_text() == "Historical measurement scope.\n"


def test_no_answer_or_gpu_dependency_imported():
    text = subject.Path(subject.__file__).read_text()
    assert "import torch" not in text
    assert "squeue" not in text
    assert "AutoModel" not in text


def test_bad_optional_input_does_not_create_output(tmp_path, monkeypatch):
    historical = put(tmp_path / "old.json", history())
    monkeypatch.setattr(subject, "HISTORICAL", historical)
    with pytest.raises(ValueError):
        subject.run(tmp_path / "derived", current_diagnostics={})
    assert not (tmp_path / "derived").exists()
