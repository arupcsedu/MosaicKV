"""Synthetic provenance-only metadata tests; no historical artifact opens."""

import hashlib
import json

import evicastkv_manifest_recovery_v52 as runner
import pytest


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "history"
    root.mkdir()
    monkeypatch.setattr(runner.prior, "HISTORY", root.resolve())
    monkeypatch.setattr(runner.capsule, "HISTORY", root.resolve())
    return root


def fixture_file(history, raw, name="manifest.json"):
    path = history / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_new_map_preserves_original_field_and_source(history):
    original = dict(source_sha="a" * 40, artifact_sha256={"tier_operations.parquet": "b" * 64})
    path, expected = fixture_file(history, json.dumps(original).encode())
    budget = runner.prior.Budget()
    assert runner.metadata(path, expected, budget) == original
    assert budget.input_bytes == path.stat().st_size


def test_unknown_benchmark_scalar_values_are_never_decoded(history, monkeypatch):
    marker = "PRIVATE_BENCHMARK_VALUE_MUST_NOT_BE_DECODED"
    original = dict(
        source_sha="a" * 40,
        artifact_sha256={"result.parquet": "b" * 64},
        answers=[marker],
        predictions={"answer": marker},
        task_score=0.99,
    )
    path, expected = fixture_file(history, json.dumps(original).encode())
    decode = runner.prior.lexical._decoded_scalar
    seen = []

    def observed(raw):
        assert marker.encode() not in raw
        seen.append(raw)
        return decode(raw)

    monkeypatch.setattr(runner.prior.lexical, "_decoded_scalar", observed)
    assert runner.metadata(path, expected, runner.prior.Budget()) == {
        "source_sha": "a" * 40,
        "artifact_sha256": {"result.parquet": "b" * 64},
    }
    assert seen


@pytest.mark.parametrize(
    "raw",
    [
        b'{"artifact_sha256":{"a.parquet":{"nested":"bad"}}}',
        b'{"artifact_sha256":{"a.parquet":"not-a-hash"}}',
        b'{"artifact_sha256":{"a.parquet":"' + b"a" * 64 + b'","a.parquet":"' + b"b" * 64 + b'"}}',
        b'{"artifact_sha256":{},"artifact_sha256":{}}',
        b'{"source_sha":"HEAD"}',
        b'{"artifact_sha256":{}} {}',
        b"[]",
        b'{"answer":"\\q"}',
    ],
)
def test_malformed_provenance_fails_closed(history, raw):
    path, expected = fixture_file(history, raw)
    with pytest.raises(ValueError):
        runner.metadata(path, expected, runner.prior.Budget())


def test_source_hash_mismatch_is_rejected(history):
    path, _ = fixture_file(history, b"{}")
    with pytest.raises(ValueError, match="hash or signature"):
        runner.metadata(path, "0" * 64, runner.prior.Budget())


@pytest.mark.parametrize("name", ["credentials.json", "auth.json", ".codex/manifest.json"])
def test_credential_descriptor_is_never_opened(history, monkeypatch, name):
    path, expected = fixture_file(history, b"NEVER_READ", name)

    def forbidden(*args, **kwargs):
        pytest.fail("credential fixture was opened")

    monkeypatch.setattr(runner.os, "open", forbidden)
    with pytest.raises(ValueError, match="credential"):
        runner.metadata(path, expected, runner.prior.Budget())


def test_changed_descriptor_is_rejected(history, monkeypatch):
    path, expected = fixture_file(history, b"{}")
    original = runner.prior.geometry
    calls = 0

    def changed(target):
        nonlocal calls
        calls += 1
        value = original(target)
        return value if calls == 1 else [*value[:3], value[3] + 1]

    monkeypatch.setattr(runner.prior, "geometry", changed)
    with pytest.raises(ValueError, match="signature"):
        runner.metadata(path, expected, runner.prior.Budget())


def test_metadata_size_cap_is_unchanged(history, monkeypatch):
    path, expected = fixture_file(history, b"{}")
    monkeypatch.setattr(runner.prior, "LIVE_BYTES", 1)
    with pytest.raises(ValueError, match="8-MiB"):
        runner.metadata(path, expected, runner.prior.Budget())


def test_unchanged_deadline_is_enforced(history):
    path, expected = fixture_file(history, b"{}")
    budget = runner.prior.Budget()
    budget.started -= runner.prior.SECONDS + 1
    with pytest.raises(TimeoutError):
        runner.metadata(path, expected, budget)


def test_unchanged_combined_read_cap_is_enforced(history):
    path, expected = fixture_file(history, b"{}")
    budget = runner.prior.Budget()
    budget.input_bytes = runner.prior.MAX_BYTES
    with pytest.raises(ValueError, match="2-GiB"):
        runner.metadata(path, expected, budget)


def test_recovery_reuses_every_saved_transaction_and_preserves_non_targets(tmp_path, monkeypatch):
    from copy import deepcopy
    from pathlib import Path

    from test_manifest_bindings_v52 import arguments

    from mosaickv.evaluation.fox_campaign import digest

    cases = [arguments(i) for i in range(3)]
    previous = [case["previous"] for case in cases]
    inventory = [case["original_row"] for case in cases]
    for i in range(131):
        old = deepcopy(previous[0])
        old["path"] = f"/saved/source-{i}.parquet"
        # Non-target variants must be copied verbatim, not reclassified or
        # upgraded from snapshot/dirty/ad-hoc to a clean writer.
        old["schema"] = "preserved_existing_v51_binding"
        old["status"] = "dirty_archive_source_content_and_artifact_binding_verified"
        previous.append(old)
        row = deepcopy(inventory[0])
        row["path"] = old["path"]
        inventory.append(row)
    parent = dict(
        source_sha="96d933906ca2d578c929fba85facfdfb40b47cb6",
        rows=deepcopy(previous),
        parquet_bodies_reopened=0,
        original_occurrence_counts={"unresolved": 135},
    )
    original = dict(
        source_sha="555840326ecfb953a6c5b0e3665189d1bb4f0461",
        rows=deepcopy(previous),
        binding_claims={r["path"]: [] for r in previous},
    )
    before = deepcopy(parent)
    monkeypatch.setattr(runner, "clean_source", lambda _: "f" * 40)
    monkeypatch.setattr(runner.prior.resume.reader.common, "cpu_gate", lambda *_: None)

    def saved(path, *_):
        if path.parent == runner.PARENT:
            return parent
        if path.parent == runner.startup.PARENT:
            return original
        return dict(rows=inventory)

    monkeypatch.setattr(runner, "saved_json", saved)
    monkeypatch.setattr(runner.prior, "allowed", lambda p: Path(p))
    signatures = {r["path"]: r["signature"] for r in previous}
    monkeypatch.setattr(runner.prior, "geometry", lambda p: signatures[str(p)])
    monkeypatch.setattr(runner.prior, "REVIEWED", {})
    projections = {}
    for spec in runner.SPECS:
        matched = [
            case for case in cases if case["claim"]["metadata_path"] == spec["metadata_path"]
        ]
        projections[spec["metadata_path"]] = dict(
            source_sha=spec["source_sha"],
            artifact_sha256={
                Path(c["claim"]["metadata_path"]).parent / Path(c["previous"]["path"]).name: c[
                    "previous"
                ]["artifact_sha256"]
                for c in matched
            },
        )
    # Artifact maps use canonical strings in the actual projector.
    for projection in projections.values():
        projection["artifact_sha256"] = {
            str(k): v for k, v in projection["artifact_sha256"].items()
        }
    monkeypatch.setattr(runner, "metadata", lambda path, *_: deepcopy(projections[str(path)]))
    blobs = {spec["source_sha"]: spec["writer_sha256"] for spec in runner.SPECS}
    monkeypatch.setattr(
        runner.prior,
        "git_blob",
        lambda commit, path, *_: blobs[commit] if path == runner.WRITER else runner.HELPER_SHA,
    )
    reused = []

    def reuse(row, *_):
        reused.append(row["path"])
        return next((c["transaction"] for c in cases if c["previous"]["path"] == row["path"]), {})

    monkeypatch.setattr(runner.startup, "reused_transaction", reuse)

    def forbidden(*_, **__):
        pytest.fail("a historical Parquet body/hash proof was replayed")

    monkeypatch.setattr(runner.prior, "hash_pass", forbidden)
    monkeypatch.setattr(runner.prior.resume.reader.common, "hash_file", forbidden)
    monkeypatch.setattr(runner.prior.resume, "require_artifact_capacity", lambda *_: None)
    output = {}
    monkeypatch.setattr(runner, "write_once", lambda path, value: output.update({path.name: value}))
    result = runner.recover(tmp_path)
    report = output["recovered_bindings.json"]
    assert len(reused) == len(set(reused)) == 134
    assert result["recovered"] == 3
    assert parent == before
    assert all(
        digest(a) == digest(b) for a, b in zip(report["rows"][3:], previous[3:], strict=True)
    )
    assert (
        report["parquet_bodies_reopened"] == report["original_JSON_identity_proofs_replayed"] == 0
    )
    assert report["credential_bodies_read"] is False
    assert report["fresh_confirmation_authorized"] is False
