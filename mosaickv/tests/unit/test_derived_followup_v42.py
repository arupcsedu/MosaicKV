"""Fail-closed synthetic follow-up wiring; no historical inference or data."""

import evicastkv_derived_followup_v42 as driver
import pytest


def test_failed_CPU_gate_prevents_original_artifact_access(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "clean_source", lambda *_: "a" * 40)
    monkeypatch.setattr(
        driver, "cpu_gate", lambda *_: (_ for _ in ()).throw(ValueError("CPU gate"))
    )
    monkeypatch.setattr(driver.history, "read", lambda *_: pytest.fail("artifact accessed"))
    with pytest.raises(ValueError, match="CPU gate"):
        driver.recover(tmp_path)


def test_wrong_parent_source_blocks_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "clean_source", lambda *_: "a" * 40)
    monkeypatch.setattr(driver, "cpu_gate", lambda *_: {})
    monkeypatch.setattr(
        driver.history,
        "read",
        lambda *_: dict(
            source_sha="b" * 40,
            fresh_confirmation_authorized=False,
        ),
    )
    monkeypatch.setattr(driver.correction, "prove", lambda *_: pytest.fail("proof started"))
    with pytest.raises(ValueError, match="immutable V41"):
        driver.recover(tmp_path)


def test_already_resolved_occurrence_is_not_replayed(tmp_path, monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(driver, "clean_source", lambda *_: revision)
    monkeypatch.setattr(driver, "cpu_gate", lambda *_: {})
    row = dict(path=str(driver.correction.ORIGINAL), owner_index=0, historical_error_ordinal=2)

    def read(path, *args):
        if path.name == "decision.json":
            return dict(
                source_sha=driver.PARENT_SOURCE,
                fresh_confirmation_authorized=False,
                reconciled_history_sha256="c" * 64,
            )
        if path.name == "registration.json":
            return dict(source_sha=driver.PARENT_SOURCE, rows=[row])
        return dict(unresolved_occurrences=[])

    monkeypatch.setattr(driver.history, "read", read)
    monkeypatch.setattr(driver.correction, "prove", lambda *_: pytest.fail("repeated proof"))
    result = driver.recover(tmp_path)
    assert result["status"] == "already_resolved_no_retry"
    assert result["fresh_confirmation_authorized"] is False
