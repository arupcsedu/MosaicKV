"""Bounded synthetic orchestration checks; no scheduler, datasets or network."""

import hashlib
import time
from types import SimpleNamespace

import evicastkv_residual_recovery_v41 as driver
import evicastkv_source_recovery_v41 as sources
import pytest


def test_hash_pass_is_independent_and_signature_guarded(tmp_path):
    path = tmp_path / "synthetic.json"
    raw = b'{"sample_id":"fixture","answer":"PRIVATE"}'
    path.write_bytes(raw)
    signature = sources.signature(path)
    result = driver.hash_file(path, signature, time.monotonic() + 5)
    assert result["whole_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["independent_hash_pass"] is True
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        driver.hash_file(path, signature, time.monotonic() + 5)


def test_hash_pass_rejects_alias_before_body_read(tmp_path):
    file = tmp_path / "actual.json"
    file.write_bytes(b"{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(file)
    with pytest.raises(ValueError):
        driver.hash_file(alias, sources.signature(alias), time.monotonic() + 5)


def test_CPU_failure_prevents_canary_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "load", lambda root: dict(source_sha="a" * 40))
    monkeypatch.setattr(
        driver, "cpu_gate", lambda *a: (_ for _ in ()).throw(ValueError("CPU failed"))
    )
    monkeypatch.setattr(driver, "run_tasks", lambda *a, **k: pytest.fail("worker started"))
    with pytest.raises(ValueError, match="CPU failed"):
        driver.canary(tmp_path)


def test_negative_canary_blocks_bulk_JSON_not_small_recovery(tmp_path, monkeypatch):
    calls = []
    plan = dict(source_sha="a" * 40, rows=[dict(operation="json_stream"), dict(operation="small")])
    monkeypatch.setattr(driver, "load", lambda root: plan)
    monkeypatch.setattr(driver, "cpu_gate", lambda *a: {})
    monkeypatch.setattr(
        driver.history,
        "read",
        lambda *a: dict(source_sha="a" * 40, streaming_bulk_authorized=False),
    )
    monkeypatch.setattr(
        driver,
        "run_tasks",
        lambda root, p, indices, phase, label, **kw: calls.append((indices, phase)),
    )
    result = driver.bulk(tmp_path)
    assert calls == [([1], "prove")]
    assert result["fresh_confirmation_authorized"] is False
    assert result["status"] == "streaming_blocked_canary_failed"


@pytest.mark.parametrize("error", ["unsupported new error", "unknown identity waiver"])
def test_no_unregistered_error_category(error):
    with pytest.raises(ValueError):
        driver.operation(error)


def test_arbitrary_source_alias_rejected_before_target_open(tmp_path):
    row = dict(path=str(tmp_path / "alias"), signature=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="explicitly registered"):
        sources.prove_alias(row, [str(tmp_path)], time.monotonic() + 5)


def test_pinned_numeric_profile_tree_requires_exact_physical_contents(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    target = repo / "confs"
    target.mkdir(parents=True)
    listing = []
    for model in ("7b", "13b"):
        for ratio in range(10):
            name = f"prefixkv_llava-v1.5-{model}_0.{ratio}.json"
            raw = b"[0.25,0.75]"
            (target / name).write_bytes(raw)
            oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            listing.append(f"100644 blob {oid}\tconfs/{name}\0".encode())
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    row = dict(path=str(alias), signature=sources.signature(alias))

    def git(repo, args, deadline, **kw):
        return SimpleNamespace(
            returncode=0,
            stdout=(sources.PREFIX_COMMIT + "\n").encode()
            if args[0] == "rev-parse"
            else b"".join(listing),
        )

    monkeypatch.setattr(sources, "git", git)
    proof = sources.prove_alias(
        row, [str(tmp_path)], time.monotonic() + 5, aliases={str(alias)}, repo=repo
    )
    assert proof["source_hash_kind"] == "symlink_target_utf8"
    assert proof["configuration_files_verified"] == 20
    assert proof["old_unbound_historical_version_byte_identity_claimed"] is False
    (target / "unexpected-private-data.json").write_bytes(b"not read")
    with pytest.raises(ValueError, match="unexpected or untracked"):
        sources.prove_alias(
            row, [str(tmp_path)], time.monotonic() + 5, aliases={str(alias)}, repo=repo
        )


def test_remote_lookup_cannot_request_unregistered_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.subprocess, "run", lambda *a, **kw: pytest.fail("network invoked"))
    with pytest.raises(ValueError, match="exact registered"):
        sources.remote_source_attempt(tmp_path, ["f" * 40], deadline=time.monotonic() + 5)


def test_missing_remote_lookup_is_not_a_freshness_license(tmp_path):
    result = sources.remote_source_attempt(tmp_path, [], deadline=time.monotonic() + 5)
    assert result["attempted"] is False


def test_receipt_rejects_unbound_semantic_fallback(tmp_path):
    row = dict(operation="small", path="/fake/config.yaml", signature=[1, 2, 3, 4])
    proof = dict(path=row["path"], signature=row["signature"], proof_kind="arbitrary_fallback")
    with pytest.raises(ValueError, match="unregistered small"):
        driver.receipt(row, proof, tmp_path / "unopened.json")
