"""Synthetic startup I/O and reuse: no completed Parquet proof is repeated."""

from __future__ import annotations

import hashlib
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import evicastkv_startup_recovery_v50 as runner
import pytest
from test_parquet_binding_recovery_v49 import fixture as binding_fixture


def saved_pair(monkeypatch):
    row, transaction, claims, writers = binding_fixture()
    monkeypatch.setattr(runner.prior, "REVIEWED", writers)
    witness = runner.classify_binding(row, transaction, claims, reviewed_writers=writers)
    parent = dict(wall_seconds=7.5, binding_claims={row["path"]: claims})
    return row, witness, parent


def test_completed_witness_reuses_exact_bytes_and_observed_global_time_bound(monkeypatch):
    row, witness, parent = saved_pair(monkeypatch)
    original = deepcopy((row, witness, parent))
    result = runner.reused_transaction(row, witness, parent)
    assert (row, witness, parent) == original
    assert result == dict(
        path=row["path"],
        signature=row["signature"],
        before_sha256=witness["artifact_sha256"],
        after_sha256=witness["artifact_sha256"],
        bytes_read=2 * row["signature"][2],
        seconds=parent["wall_seconds"],
    )
    # This is deliberately the old global observed upper bound, not a newly
    # fabricated file-level timing or another source-body measurement.
    assert result["seconds"] == 7.5


def test_reuse_has_no_path_or_source_body_io(monkeypatch):
    row, witness, parent = saved_pair(monkeypatch)

    def forbidden(*_args, **_kwargs):
        pytest.fail("completed source body or proof replayed")

    monkeypatch.setattr("builtins.open", forbidden)
    for method in ("open", "read_bytes", "read_text", "stat", "lstat", "resolve"):
        monkeypatch.setattr(Path, method, forbidden)
    monkeypatch.setattr(runner.prior, "hash_pass", forbidden)
    monkeypatch.setattr(runner.prior, "metadata", forbidden)
    assert runner.reused_transaction(row, witness, parent)["bytes_read"] == 2468


@pytest.mark.parametrize(
    "change",
    [
        "failed_transaction",
        "missing_artifact_hash",
        "changed_artifact_hash",
        "changed_witness_producer",
        "changed_witness_claim_digest",
        "missing_claims",
        "changed_claim_artifact",
        "changed_claim_metadata",
        "changed_claim_producer",
        "missing_wall_time",
        "expired_wall_time",
        "nonfinite_wall_time",
        "changed_original_geometry",
    ],
)
def test_missing_or_modified_completed_proof_cannot_be_reconstructed(monkeypatch, change):
    row, witness, parent = saved_pair(monkeypatch)
    if change == "failed_transaction":
        witness["status"] = "binding_transaction_unqualified"
    elif change == "missing_artifact_hash":
        del witness["artifact_sha256"]
    elif change == "changed_artifact_hash":
        witness["artifact_sha256"] = "f" * 64
    elif change == "changed_witness_producer":
        witness["original_producer_commit"] = "f" * 40
    elif change == "changed_witness_claim_digest":
        witness["original_metadata_claim_sha256s"] = ["f" * 64]
    elif change == "missing_claims":
        del parent["binding_claims"][row["path"]]
    elif change == "changed_claim_artifact":
        parent["binding_claims"][row["path"]][0]["artifact_sha256"] = "f" * 64
    elif change == "changed_claim_metadata":
        parent["binding_claims"][row["path"]][0]["metadata_sha256"] = "f" * 64
    elif change == "changed_claim_producer":
        parent["binding_claims"][row["path"]][0]["recorded_source_sha"] = "f" * 40
    elif change == "missing_wall_time":
        del parent["wall_seconds"]
    elif change == "expired_wall_time":
        parent["wall_seconds"] = 211
    elif change == "nonfinite_wall_time":
        parent["wall_seconds"] = float("nan")
    else:
        row["signature"][2] += 1
    with pytest.raises((ValueError, KeyError)):
        runner.reused_transaction(row, witness, parent)


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "history"
    root.mkdir()
    monkeypatch.setattr(runner.prior, "HISTORY", root.resolve())
    return root


def startup_file(history, raw=None):
    path = history / "startup.out"
    raw = b"clean_git_sha=" + b"c" * 40 + b"\n" if raw is None else raw
    path.write_bytes(raw)
    return path, raw


def test_startup_descriptor_reads_only_the_55_byte_nominated_record(history):
    path, raw = startup_file(history)
    budget = runner.prior.Budget()
    assert runner.raw_startup(path, budget) == raw
    assert budget.input_bytes == 55


@pytest.mark.parametrize("size", [0, 54, 56, 500])
def test_wrong_startup_size_is_rejected_before_descriptor_open(history, monkeypatch, size):
    path, _ = startup_file(history, b"x" * size)
    monkeypatch.setattr(
        runner.os, "open", lambda *a, **k: pytest.fail("unregistered startup body opened")
    )
    with pytest.raises(ValueError, match="55-byte"):
        runner.raw_startup(path, runner.prior.Budget())


def test_startup_budget_is_authorized_before_opening(history, monkeypatch):
    path, _ = startup_file(history)
    budget = runner.prior.Budget()
    budget.input_bytes = runner.prior.MAX_BYTES
    monkeypatch.setattr(
        runner.os, "open", lambda *a, **k: pytest.fail("over-budget startup opened")
    )
    with pytest.raises(ValueError, match="combined-read"):
        runner.raw_startup(path, budget)


def test_startup_postvalidation_symlink_cannot_reach_credentials(history, tmp_path, monkeypatch):
    path, _ = startup_file(history)
    protected = tmp_path / "credentials.json"
    protected.write_bytes(b"SYNTHETIC-NOT-A-REAL-CREDENTIAL")
    original = os.open
    calls = []

    def changed(name, flags, *args, **kwargs):
        calls.append(flags)
        assert flags & os.O_NOFOLLOW
        path.unlink()
        path.symlink_to(protected)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", changed)
    with pytest.raises((OSError, ValueError)):
        runner.raw_startup(path, runner.prior.Budget())
    assert len(calls) == 1


def test_startup_descriptor_geometry_must_match_before_read(history, monkeypatch):
    path, _ = startup_file(history)
    signature = runner.prior.geometry(path)
    signature[1] += 1
    monkeypatch.setattr(runner.prior, "geometry", lambda *_args: signature)
    with pytest.raises(ValueError, match="descriptor"):
        runner.raw_startup(path, runner.prior.Budget())


def test_source_closure_uses_the_recorded_commit_and_each_exact_blob(monkeypatch):
    calls = []
    expected = {"mosaickv/scripts/analyzer.py": "a" * 64, "mosaickv/scripts/guard.sh": "b" * 64}

    def blob(commit, name, budget):
        calls.append((commit, name))
        return expected[name]

    monkeypatch.setattr(runner.prior, "git_blob", blob)
    result = runner.source_closure(
        "c" * 40,
        "mosaickv/scripts/analyzer.py",
        "a" * 64,
        "mosaickv/scripts/guard.sh",
        "b" * 64,
        runner.prior.Budget(),
    )
    assert result == expected
    assert calls == [("c" * 40, name) for name in expected]


@pytest.mark.parametrize("failure", ["writer_blob", "guard_blob", "unavailable"])
def test_mismatched_or_unavailable_source_closure_cannot_upgrade(monkeypatch, failure):
    def blob(commit, name, budget):
        if failure == "unavailable":
            raise subprocess.CalledProcessError(128, ["git", "show"])
        if (failure == "writer_blob" and name == "writer.py") or (
            failure == "guard_blob" and name == "guard.sh"
        ):
            return "f" * 64
        return "a" * 64 if name == "writer.py" else "b" * 64

    monkeypatch.setattr(runner.prior, "git_blob", blob)
    with pytest.raises((ValueError, subprocess.SubprocessError)):
        runner.source_closure(
            "c" * 40, "writer.py", "a" * 64, "guard.sh", "b" * 64, runner.prior.Budget()
        )


def test_accounting_uses_only_the_exact_registered_query(monkeypatch):
    calls = []
    raw = b"synthetic scheduler metadata\n"

    def output(command, **kwargs):
        calls.append((command, kwargs))
        return raw

    monkeypatch.setattr(runner.subprocess, "check_output", output)
    budget = runner.prior.Budget()
    assert runner.accounting("21671480", budget) == raw
    assert budget.input_bytes == len(raw)
    assert calls == [
        (
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                "21671480",
                "--format=JobIDRaw,User,State,ExitCode,WorkDir%2048,SubmitLine%16384",
            ],
            {"timeout": 15},
        )
    ]


def test_accounting_budget_refusal_precedes_scheduler_read(monkeypatch):
    budget = runner.prior.Budget()
    budget.input_bytes = runner.prior.MAX_BYTES
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **k: pytest.fail("over-budget scheduler read")
    )
    with pytest.raises(ValueError, match="combined-read"):
        runner.accounting("21671480", budget)


def test_afterany_missing_promotion_stays_blocked_without_body_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "clean_source", lambda *_args: "a" * 40)
    monkeypatch.setattr(runner.prior.resume.reader.common, "cpu_gate", lambda *_args: None)
    monkeypatch.setattr(runner.prior, "hash_pass", lambda *a: pytest.fail("Parquet body reopened"))
    monkeypatch.setattr(runner, "accounting", lambda *a: pytest.fail("startup retry launched"))
    result = runner.analyze(tmp_path)
    assert result["status"] == "startup_recovery_failed_or_incomplete"
    assert result["fresh_confirmation_authorized"] is False
    assert result["GPU_jobs"] == 0


def test_bound_startups_keep_original_reviewed_raw_record_hashes():
    assert [spec["job_id"] for spec in runner.STARTUPS] == ["21671480", "21748934", "21610690"]
    assert {spec["startup_sha256"] for spec in runner.STARTUPS} == {
        hashlib.sha256(
            b"clean_git_sha=" + b"fdeca177c4a3eaf77df53780b3586a4dc462365c" + b"\n"
        ).hexdigest(),
        hashlib.sha256(
            b"clean_git_sha=" + b"c4d733b7512aa6576ec37ddcbbbf6902e21a8096" + b"\n"
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    "token", [b"HF_TOKEN", b"api_key", b"password", b"secret", b"Authorization"]
)
def test_credential_bearing_accounting_is_never_persisted(token):
    with pytest.raises(ValueError, match="credential-bearing"):
        runner.preserved_accounting(b"sbatch --wrap=export " + token + b"=DO_NOT_PERSIST")


def test_public_reviewed_record_can_be_verified_offline():
    raw = b"21671480|djy8hg|COMPLETED|0:0|/scratch/work|sbatch public-command\n"
    assert runner.preserved_accounting(raw).encode() == raw
