"""Synthetic recovery wiring; no historical sources or model access."""

from copy import deepcopy

import evicastkv_occurrence_recovery_v40 as driver
import pytest


def registration():
    return dict(
        owner_index=1,
        historical_error_ordinal=2,
        original_error_sha256="a" * 64,
        path="/fake/config.yaml",
        kind="yaml_events",
        signature=[1, 2, 5, 6],
        file_sha256="b" * 64,
    )


def test_selection_never_reinterprets_old_occurrences_as_new():
    old = dict(
        registration(),
        historical_error_ordinal=0,
        error="YAML requires separately reviewed lexical identity projection",
    )
    new = dict(old, historical_error_ordinal=2)
    large = dict(new, error="JSON exceeds bounded projection size; not silently skipped")
    assert driver.selected_occurrences(
        dict(unresolved_occurrences=[old, new, large]), {"1": 1}
    ) == [(new, "yaml_events")]


def test_receipt_is_bound_to_exact_semantic_proof():
    row = registration()
    proof = dict(
        path=row["path"],
        signature=row["signature"],
        original_error_sha256s=[row["original_error_sha256"]],
        whole_file_sha256_before="b" * 64,
        whole_file_sha256_after="b" * 64,
        kind=row["kind"],
        registration_sha256=driver.digest(driver.projection_row(row)),
        identities=dict(sample_ids=["id"], projected_records=1),
    )
    receipt = driver.receipt_for(row, proof)
    assert receipt["registration_sha256"] == driver.digest(row)
    assert receipt["identities"] == dict(sample_ids=["id"])
    changed = deepcopy(proof)
    changed["path"] = "/another/config.yaml"
    with pytest.raises(ValueError, match="exact occurrence"):
        driver.receipt_for(row, changed)
    changed = deepcopy(proof)
    changed["registration_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="exact occurrence"):
        driver.receipt_for(row, changed)


def test_CPU_failure_prevents_recovery_worker_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "load", lambda root: dict(source_sha="a" * 40))

    def fail(*args):
        raise ValueError("full CPU qualification missing")

    monkeypatch.setattr(driver, "cpu_gate", fail)
    with pytest.raises(ValueError, match="qualification missing"):
        driver.recover(tmp_path)
    assert not (tmp_path / "statuses.json").exists()


def test_unregistered_proof_path_is_rejected_before_open(tmp_path):
    with pytest.raises(ValueError, match="registered proof artifacts"):
        driver.checked_proof_path(tmp_path, registration(), "/unregistered/private.json")


def test_proof_alias_cannot_escape_registered_root(tmp_path):
    row = registration()
    path = tmp_path / f"proofs/{driver.digest(row)[:20]}.json"
    path.parent.mkdir()
    outside = tmp_path / "outside.json"
    outside.touch()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="registered proof artifacts"):
        driver.checked_proof_path(tmp_path, row, str(path))


@pytest.mark.parametrize("index", [True, -1, 8])
def test_invalid_worker_index_is_rejected(tmp_path, monkeypatch, index):
    monkeypatch.setattr(driver, "load", lambda root: dict(source_sha="a" * 40))
    monkeypatch.setattr(driver, "cpu_gate", lambda *args: {})
    with pytest.raises(ValueError, match="worker index"):
        driver.worker(tmp_path, index)
