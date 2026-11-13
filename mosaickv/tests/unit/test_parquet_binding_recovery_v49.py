"""Synthetic provenance facts; these tests never touch historical artifacts."""

from copy import deepcopy
from pathlib import Path

import pytest

from mosaickv.evaluation import parquet_binding_recovery_v49 as recovery
from mosaickv.evaluation.fox_campaign import digest


def fixture():
    registration = dict(
        owner_index=0,
        historical_error_ordinal=4,
        original_error_sha256="e" * 64,
        path="/scratch/runs/example/summary.parquet",
        operation="parquet",
        signature=[12, 34, 1234, 56],
    )
    row = dict(
        **registration,
        original_error=recovery.inventory.PARQUET_ERROR,
        original_registration_sha256=digest(registration),
        status="missing_artifact_binding",
        existing_artifact_sha256=None,
        original_producer_commit=None,
        original_producer_source_blob_sha256=None,
        original_producer_binding_status="missing_independent_producer_proof",
        source_body_read=False,
        occurrence_resolved=False,
    )
    transaction = dict(
        path=registration["path"],
        signature=list(registration["signature"]),
        before_sha256="a" * 64,
        after_sha256="a" * 64,
        bytes_read=2468,
        seconds=0.5,
    )
    claim = dict(
        metadata_path="/scratch/runs/example/manifest.json",
        metadata_sha256="b" * 64,
        artifact_sha256="a" * 64,
        role="runtime_writer",
        recorded_source_sha="c" * 40,
        source_blob_sha256="d" * 64,
        writer_path="mosaickv/scripts/original_writer.py",
        source_evidence_verified=True,
    )
    writers = {(claim["recorded_source_sha"], claim["writer_path"]): claim["source_blob_sha256"]}
    return row, transaction, [claim], writers


def run(row, transaction, claims, writers):
    return recovery.classify_binding(row, transaction, claims, reviewed_writers=writers)


def test_reviewed_original_runtime_writer_and_artifact_are_separately_bound():
    args = fixture()
    original = deepcopy(args)
    result = run(*args)
    assert args == original
    assert result["status"] == "runtime_producer_and_artifact_binding_verified"
    assert result["original_producer_commit"] == "c" * 40
    assert result["original_producer_source_blob_sha256"] == "d" * 64
    assert result["artifact_sha256"] == "a" * 64
    assert result["occurrence_resolved"] is False
    assert result["newly_resolved_occurrences"] == result["GPU_jobs"] == 0
    assert all(result[name] is False for name in recovery.contract.FALSE_FIELDS)


def test_classifier_is_pure_and_does_no_path_or_filesystem_access(monkeypatch):
    args = fixture()

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure classifier performed I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    for name in ("open", "stat", "lstat", "resolve", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, name, forbidden)
    assert run(*args)["status"] == "runtime_producer_and_artifact_binding_verified"


@pytest.mark.parametrize("role", ["input_source", "artifact_only"])
def test_input_checkpoint_or_current_hash_never_becomes_the_original_writer(role):
    row, transaction, claims, writers = fixture()
    claims[0]["role"] = role
    result = run(row, transaction, claims, writers)
    assert result["status"] == "artifact_binding_only"
    assert result["original_producer_commit"] is None


def test_current_two_hashes_without_an_original_claim_are_unrecovered():
    row, transaction, _, writers = fixture()
    result = run(row, transaction, [], writers)
    assert result["status"] == "unrecovered"
    assert result["original_producer_commit"] is None


def test_preexisting_bound_artifact_hash_still_does_not_establish_a_writer():
    row, transaction, _, writers = fixture()
    row.update(existing_artifact_sha256="a" * 64, status="existing_artifact_hash_only")
    result = run(row, transaction, [], writers)
    assert result["status"] == "artifact_binding_only"
    assert result["original_producer_commit"] is None


def test_committed_snapshot_is_not_runtime_generation_provenance():
    row, transaction, claims, writers = fixture()
    claims[0]["role"] = "committed_snapshot"
    result = run(row, transaction, claims, writers)
    assert result["status"] == "committed_snapshot_artifact_binding_verified"
    assert result["original_producer_commit"] is None
    assert result["committed_snapshot_bindings"] == [
        dict(commit="c" * 40, path=claims[0]["writer_path"], source_blob_sha256="d" * 64)
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "unverified",
        "unknown_writer",
        "wrong_blob",
        "missing_source",
        "missing_writer",
        "missing_blob",
    ],
)
def test_unavailable_or_unreviewed_source_cannot_upgrade_runtime_provenance(failure):
    row, transaction, claims, writers = fixture()
    if failure == "unverified":
        claims[0]["source_evidence_verified"] = False
    elif failure == "unknown_writer":
        writers = {}
    elif failure == "wrong_blob":
        writers = {next(iter(writers)): "f" * 64}
    else:
        key = {
            "missing_source": "recorded_source_sha",
            "missing_writer": "writer_path",
            "missing_blob": "source_blob_sha256",
        }[failure]
        claims[0][key] = None
    result = run(row, transaction, claims, writers)
    assert result["status"] == "artifact_binding_only"
    assert result["original_producer_commit"] is None
    assert result["insufficient_evidence_reasons"]


@pytest.mark.parametrize(
    "field", ["path", "signature", "before_sha256", "after_sha256", "bytes_read", "seconds"]
)
def test_transaction_requires_every_bound_field(field):
    row, transaction, claims, writers = fixture()
    del transaction[field]
    with pytest.raises(ValueError):
        run(row, transaction, claims, writers)


@pytest.mark.parametrize(
    "failure",
    [
        "path",
        "signature",
        "opening_hash",
        "closing_hash",
        "partial_bytes",
        "negative_bytes",
        "boolean_bytes",
        "timeout",
        "nan",
        "infinity",
        "boolean_seconds",
        "extra_field",
    ],
)
def test_changed_geometry_hashes_or_limits_fail_closed(failure):
    row, transaction, claims, writers = fixture()
    if failure == "path":
        transaction["path"] = "/scratch/runs/other/summary.parquet"
    elif failure == "signature":
        transaction["signature"][1] += 1
    elif failure in {"opening_hash", "closing_hash"}:
        transaction["before_sha256" if failure == "opening_hash" else "after_sha256"] = "f" * 64
    elif failure in {"partial_bytes", "negative_bytes", "boolean_bytes"}:
        transaction["bytes_read"] = {
            "partial_bytes": 1234,
            "negative_bytes": -1,
            "boolean_bytes": True,
        }[failure]
    elif failure == "extra_field":
        transaction["producer_sha"] = "c" * 40
    else:
        transaction["seconds"] = {
            "timeout": 210.001,
            "nan": float("nan"),
            "infinity": float("inf"),
            "boolean_seconds": True,
        }[failure]
    with pytest.raises(ValueError):
        run(row, transaction, claims, writers)


def test_complete_two_hash_reads_cannot_exceed_the_input_ceiling():
    row, transaction, claims, writers = fixture()
    row["signature"][2] = recovery.MAXIMUM_INPUT_BYTES // 2 + 1
    transaction["signature"] = list(row["signature"])
    transaction["bytes_read"] = 2 * row["signature"][2]
    registration = {
        name: row[name]
        for name in recovery.contract.OCCURRENCE_FIELDS | {"path", "signature", "operation"}
    }
    row["original_registration_sha256"] = digest(registration)
    with pytest.raises(ValueError, match="limits"):
        run(row, transaction, claims, writers)


@pytest.mark.parametrize(
    "field", ["artifact_sha256", "metadata_sha256", "source_blob_sha256", "recorded_source_sha"]
)
def test_malformed_original_hashes_or_commits_fail_closed(field):
    row, transaction, claims, writers = fixture()
    claims[0][field] = "not-a-commit-or-hash"
    with pytest.raises(ValueError):
        run(row, transaction, claims, writers)


def test_even_one_conflicting_original_artifact_hash_is_rejected():
    row, transaction, claims, writers = fixture()
    other = deepcopy(claims[0])
    other["artifact_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="conflicts"):
        run(row, transaction, [*claims, other], writers)


def test_conflicting_verified_producers_are_not_selected_arbitrarily():
    row, transaction, claims, writers = fixture()
    other = deepcopy(claims[0])
    other["recorded_source_sha"] = "e" * 40
    writers[(other["recorded_source_sha"], other["writer_path"])] = other["source_blob_sha256"]
    with pytest.raises(ValueError, match="conflicting original runtime"):
        run(row, transaction, [*claims, other], writers)


@pytest.mark.parametrize(
    "path",
    [
        "../writer.py",
        "mosaickv/../writer.py",
        "/absolute/writer.py",
        "mosaickv//writer.py",
        ".codex/auth.json",
        "credentials.json",
        "writer.py\x00",
        "writer\\file.py",
    ],
)
def test_malicious_writer_paths_fail_closed(path):
    row, transaction, claims, writers = fixture()
    claims[0]["writer_path"] = path
    with pytest.raises(ValueError):
        run(row, transaction, claims, writers)


@pytest.mark.parametrize(
    "path",
    [
        "relative.json",
        "/scratch/../manifest.json",
        "/scratch/.codex/auth.json",
        "/scratch/credentials.json",
        "/scratch//manifest.json",
        "/scratch/manifest.json\n",
    ],
)
def test_malicious_metadata_paths_cannot_supply_claims(path):
    row, transaction, claims, writers = fixture()
    claims[0]["metadata_path"] = path
    with pytest.raises(ValueError):
        run(row, transaction, claims, writers)


def test_dirty_tree_assertion_cannot_bypass_exact_claim_schema():
    row, transaction, claims, writers = fixture()
    claims[0]["git_dirty"] = False
    with pytest.raises(ValueError, match="schema"):
        run(row, transaction, claims, writers)


def test_claim_order_does_not_change_classification():
    row, transaction, claims, writers = fixture()
    second = deepcopy(claims[0])
    second.update(role="input_source", metadata_path="/scratch/runs/example/input.json")
    assert run(row, transaction, [*claims, second], writers) == run(
        row, transaction, [second, *claims], writers
    )


def test_claim_and_writer_counts_remain_bounded():
    row, transaction, claims, writers = fixture()
    with pytest.raises(ValueError, match="bounded"):
        run(row, transaction, claims * (recovery.MAXIMUM_CLAIMS + 1), writers)


@pytest.mark.parametrize("role", [[], {}, 7, None, "auditor"])
def test_invalid_claim_roles_are_rejected_with_useful_errors(role):
    row, transaction, claims, writers = fixture()
    claims[0]["role"] = role
    with pytest.raises(ValueError, match="recognized claim role"):
        run(row, transaction, claims, writers)


@pytest.mark.parametrize("verified", [1, None, "true"])
def test_source_verification_is_never_coerced_to_boolean(verified):
    row, transaction, claims, writers = fixture()
    claims[0]["source_evidence_verified"] = verified
    with pytest.raises(ValueError):
        run(row, transaction, claims, writers)


def test_unverified_snapshot_does_not_claim_a_verified_source():
    row, transaction, claims, writers = fixture()
    claims[0].update(role="committed_snapshot", source_evidence_verified=False)
    result = run(row, transaction, claims, writers)
    assert result["status"] == "artifact_binding_only"
    assert not result["committed_snapshot_bindings"]


def test_inclusive_registered_transaction_boundaries_are_valid():
    row, transaction, claims, writers = fixture()
    row["signature"][2] = recovery.MAXIMUM_INPUT_BYTES // 2
    transaction.update(
        signature=list(row["signature"]),
        bytes_read=recovery.MAXIMUM_INPUT_BYTES,
        seconds=recovery.MAXIMUM_SECONDS,
    )
    registration = {
        name: row[name]
        for name in recovery.contract.OCCURRENCE_FIELDS | {"path", "signature", "operation"}
    }
    row["original_registration_sha256"] = digest(registration)
    assert (
        run(row, transaction, claims, writers)["status"]
        == "runtime_producer_and_artifact_binding_verified"
    )


@pytest.mark.parametrize(
    "writers",
    [
        {("short", "writer.py"): "d" * 64},
        {("c" * 40, "../writer.py"): "d" * 64},
        {"not_a_tuple": "d" * 64},
        {("c" * 40, "writer.py"): "invalid"},
    ],
)
def test_reviewed_writer_registry_is_strict(writers):
    row, transaction, claims, _ = fixture()
    with pytest.raises(ValueError):
        run(row, transaction, claims, writers)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_index", True),
        ("historical_error_ordinal", -1),
        ("original_registration_sha256", "f" * 64),
        ("existing_artifact_sha256", "f" * 64),
        ("occurrence_resolved", True),
        ("source_body_read", True),
        ("original_producer_commit", "c" * 40),
    ],
)
def test_original_occurrence_cannot_be_rewritten_or_pre_promoted(field, value):
    row, transaction, claims, writers = fixture()
    row[field] = value
    with pytest.raises(ValueError):
        run(row, transaction, claims, writers)
