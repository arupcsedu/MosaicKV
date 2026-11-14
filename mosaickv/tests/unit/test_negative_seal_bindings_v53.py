"""Synthetic seal projections; never read historical YAML or artifact bodies."""

from copy import deepcopy
from pathlib import Path

import pytest
from test_parquet_binding_recovery_v49 import fixture

from mosaickv.evaluation import negative_seal_bindings_v53 as seals
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.parquet_binding_recovery_v49 import CLAIM_FIELDS, classify_binding

ROOT = "/scratch/djy8hg/runs/negative-example"
TARGET = ROOT + "/analysis/summary.parquet"
METADATA = "/scratch/djy8hg/workdir/seals/negative.yaml"
LEDGER_ROOT = "/scratch/djy8hg/workdir/immutable-ledger"


def projection():
    return dict(
        run_root=ROOT,
        classification=seals.CLASSIFICATION,
        immutable_result=True,
        frozen_candidate_created=False,
        artifacts={
            "summary": dict(path="analysis/summary.parquet", sha256="a" * 64),
            "driver": dict(path="scripts/driver.py", sha256="b" * 64),
            "external_script": dict(path="/scratch/djy8hg/workdir/driver.py", sha256="c" * 64),
        },
    )


def claim(projected=None, target=TARGET):
    return seals.artifact_claim(
        projection() if projected is None else projected, METADATA, "d" * 64, target
    )


def arguments(*, old_claim=False):
    row, transaction, _, registry = fixture()
    row["path"] = transaction["path"] = "/sfs/weka" + TARGET
    registration = {
        name: row[name]
        for name in seals.contract.OCCURRENCE_FIELDS | {"path", "signature", "operation"}
    }
    row["original_registration_sha256"] = digest(registration)
    original_claims = [dict(claim(), metadata_sha256="e" * 64)] if old_claim else []
    previous = classify_binding(row, transaction, original_claims, reviewed_writers=registry)
    return dict(
        previous=previous,
        original_row=row,
        transaction=transaction,
        original_claims=original_claims,
        claims=[claim()],
        reviewed_writers=registry,
    )


def test_exact_record_is_artifact_only_and_known_mount_alias_is_canonicalized():
    original = projection()
    saved = deepcopy(original)
    result = claim(original)
    assert original == saved
    assert set(result) == CLAIM_FIELDS
    assert result == dict(
        metadata_path="/sfs/weka" + METADATA,
        metadata_sha256="d" * 64,
        artifact_sha256="a" * 64,
        role="artifact_only",
        recorded_source_sha=None,
        source_blob_sha256=None,
        writer_path=None,
        source_evidence_verified=False,
    )
    original["artifacts"]["summary"]["path"] = "/sfs/weka" + TARGET
    assert claim(original, "/sfs/weka" + TARGET) == result


def qualified_projection():
    value = projection()
    value["immutable_ledger_root"] = LEDGER_ROOT
    value["artifacts"]["summary"]["root"] = "run_root"
    value["artifacts"]["driver"] = dict(
        root="immutable_ledger_root", path="scripts/driver.py", sha256="b" * 64
    )
    return value


@pytest.mark.parametrize("absolute", [False, True])
def test_qualified_v98_v99_roots_bind_only_their_exact_declared_paths(absolute):
    value = qualified_projection()
    if absolute:
        value["artifacts"]["summary"]["path"] = "/sfs/weka" + TARGET
        value["artifacts"]["driver"]["path"] = "/sfs/weka" + LEDGER_ROOT + "/scripts/driver.py"
    saved = deepcopy(value)
    assert claim(value) == claim()
    assert value == saved


def test_explicit_run_root_qualification_does_not_require_an_unused_ledger_root():
    value = projection()
    value["artifacts"]["summary"]["root"] = "run_root"
    assert claim(value) == claim()


@pytest.mark.parametrize(
    "change",
    [
        "unknown_root",
        "missing_declared_root",
        "invalid_root_type",
        "invalid_ledger_root",
        "unknown_declared_root",
        "absolute_root_conflict",
        "absolute_prefix_outside",
        "relative_ledger_target",
        "absolute_ledger_target",
        "outside_target",
        "cross_root_duplicate",
    ],
)
def test_qualified_roots_cannot_be_guessed_or_silently_reinterpreted(change):
    value, target = qualified_projection(), TARGET
    if change == "unknown_root":
        value["artifacts"]["driver"]["root"] = "source_root"
    elif change == "missing_declared_root":
        del value["immutable_ledger_root"]
    elif change == "invalid_root_type":
        value["artifacts"]["driver"]["root"] = ["immutable_ledger_root"]
    elif change == "invalid_ledger_root":
        value["immutable_ledger_root"] = "../ledger"
    elif change == "unknown_declared_root":
        value["source_root"] = LEDGER_ROOT
    elif change in {"absolute_root_conflict", "absolute_prefix_outside"}:
        value["artifacts"]["driver"]["path"] = (
            ROOT + "/scripts/driver.py"
            if change == "absolute_root_conflict"
            else LEDGER_ROOT + "-outside/scripts/driver.py"
        )
    elif change == "relative_ledger_target":
        value["artifacts"]["summary"]["root"] = "immutable_ledger_root"
    elif change == "absolute_ledger_target":
        value["artifacts"]["summary"].update(root="immutable_ledger_root", path=TARGET)
    elif change == "outside_target":
        target = LEDGER_ROOT + "/analysis/summary.parquet"
        value["artifacts"]["summary"]["root"] = "immutable_ledger_root"
    else:
        value["immutable_ledger_root"] = ROOT
        value["artifacts"]["duplicate"] = dict(
            root="immutable_ledger_root", path="analysis/summary.parquet", sha256="a" * 64
        )
    with pytest.raises(ValueError):
        claim(value, target)


@pytest.mark.parametrize("old_claim", [False, True])
def test_extension_reuses_original_transaction_without_changing_scientific_permissions(old_claim):
    values = arguments(old_claim=old_claim)
    saved = deepcopy(values)
    result = seals.augment_seal_binding(**values)
    assert values == saved
    assert result["schema"] == "historical_negative_seal_binding_v53"
    assert result["status"] == "artifact_binding_only"
    assert result["parent_binding_sha256"] == digest(values["previous"])
    assert result["original_negative_seal_claim_sha256s"] == [digest(values["claims"][0])]
    assert result["artifact_sha256"] == values["previous"]["artifact_sha256"]
    for name in (
        "original_producer_commit",
        "original_producer_writer_path",
        "original_producer_source_blob_sha256",
    ):
        assert result[name] is None
    assert result["reused_hash_transaction"] is True
    assert result["GPU_jobs"] == result["newly_resolved_occurrences"] == 0
    for name in (
        *seals.contract.FALSE_FIELDS,
        "occurrence_resolved",
        "canonical_eligible",
        "clean_runtime_execution_claimed",
    ):
        assert result[name] is False


def test_both_old_and_new_inputs_pass_through_unchanged_v49_classifier(monkeypatch):
    values = arguments(old_claim=True)
    calls = []

    def observed(*args, **kwargs):
        calls.append(deepcopy((args, kwargs)))
        return classify_binding(*args, **kwargs)

    monkeypatch.setattr(seals, "classify_binding", observed)
    seals.augment_seal_binding(**values)
    assert len(calls) == 2
    for args, kwargs in calls:
        assert args[:2] == (values["original_row"], values["transaction"])
        assert kwargs == {"reviewed_writers": values["reviewed_writers"]}
    assert calls[0][0][2] == values["original_claims"]
    assert calls[1][0][2] == values["original_claims"] + values["claims"]


def test_module_performs_no_filesystem_git_network_or_scheduler_access(monkeypatch):
    projected, values = projection(), arguments()

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure negative-seal validator performed I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("os.open", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    for method in ("open", "stat", "lstat", "resolve", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, method, forbidden)
    assert claim(projected)["role"] == "artifact_only"
    assert seals.augment_seal_binding(**values)["reused_hash_transaction"] is True


def test_multiple_independent_seals_can_bind_the_same_unchanged_artifact():
    values = arguments()
    values["claims"].append(dict(values["claims"][0], metadata_sha256="f" * 64))
    result = seals.augment_seal_binding(**values)
    assert result["original_negative_seal_claim_sha256s"] == sorted(map(digest, values["claims"]))
    assert result["status"] == "artifact_binding_only"


@pytest.mark.parametrize("bound", ["bytes", "nodes"])
def test_unchanged_projected_metadata_size_bounds_are_enforced(bound):
    value = projection()
    value["classification"] = "x" * (8 * 1024**2 + 1) if bound == "bytes" else [0] * 100001
    with pytest.raises(ValueError, match="bound"):
        claim(value)


@pytest.mark.parametrize(
    "change",
    [
        "extra_commit",
        "extra_protocol",
        "extra_run_sha",
        "extra_worker_sha",
        "missing_root",
        "wrong_label",
        "not_immutable",
        "boolean_as_integer",
        "candidate_created",
        "map_sequence",
        "empty_map",
        "entry_extra",
        "entry_missing",
        "bad_hash",
        "invalid_nontarget_hash",
        "traversal",
        "nontarget_traversal",
        "credential",
        "nontarget_credential",
        "duplicate",
        "conflict",
        "alias_duplicate",
        "not_exact_target",
        "target_outside",
        "prefix_outside",
        "not_parquet",
        "different_mount",
        "relative_root",
        "too_many",
        "long_path",
        "deep",
    ],
)
def test_projection_rejects_invalid_or_ambiguous_evidence_including_nontargets(change):
    value, target = projection(), TARGET
    if change.startswith("extra_"):
        value[change.removeprefix("extra_")] = "f" * 40
    elif change == "missing_root":
        del value["run_root"]
    elif change == "wrong_label":
        value["classification"] = "positive"
    elif change in {"not_immutable", "boolean_as_integer"}:
        value["immutable_result"] = False if change == "not_immutable" else 1
    elif change == "candidate_created":
        value["frozen_candidate_created"] = True
    elif change in {"map_sequence", "empty_map"}:
        value["artifacts"] = [] if change == "map_sequence" else {}
    elif change == "entry_extra":
        value["artifacts"]["summary"]["producer"] = "f" * 40
    elif change == "entry_missing":
        del value["artifacts"]["summary"]["sha256"]
    elif change in {"bad_hash", "invalid_nontarget_hash"}:
        value["artifacts"]["summary" if change == "bad_hash" else "driver"]["sha256"] = "bad"
    elif change in {"traversal", "nontarget_traversal", "credential", "nontarget_credential"}:
        entry = "driver" if change.startswith("nontarget_") else "summary"
        value["artifacts"][entry]["path"] = (
            "../summary.parquet" if change.endswith("traversal") else "private/.env"
        )
    elif change in {"duplicate", "conflict", "alias_duplicate"}:
        value["artifacts"]["duplicate"] = dict(
            path=TARGET if change != "alias_duplicate" else "/sfs/weka" + TARGET,
            sha256=("f" if change == "conflict" else "a") * 64,
        )
    elif change == "not_exact_target":
        target = ROOT + "/other/summary.parquet"
    elif change in {"target_outside", "prefix_outside"}:
        target = (ROOT + "-outside" if change == "prefix_outside" else "/other") + "/x.parquet"
        value["artifacts"]["summary"]["path"] = target
    elif change == "not_parquet":
        target = ROOT + "/analysis/summary.json"
    elif change == "different_mount":
        target = TARGET.replace("/scratch/djy8hg", "/sfs/gpfs/scratch/djy8hg")
    elif change == "relative_root":
        value["run_root"] = "runs/example"
    elif change == "too_many":
        value["artifacts"] = {
            str(i): dict(path=f"{i}.parquet", sha256="a" * 64)
            for i in range(seals.MAXIMUM_CLAIMS + 1)
        }
    elif change == "long_path":
        value["artifacts"]["driver"]["path"] = "x" * 4097
    else:
        nested = "x"
        for _ in range(66):
            nested = [nested]
        value["artifacts"]["driver"]["sha256"] = nested
    with pytest.raises(ValueError):
        claim(value, target)


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "prior_producer",
        "prior_writer",
        "prior_blob",
        "prior_science",
        "prior_digest",
        "old_claim_missing",
        "transaction_hash",
        "transaction_bytes",
        "transaction_time",
        "new_role",
        "new_source",
        "new_blob",
        "new_writer",
        "new_verification",
        "new_hash",
        "new_extra",
        "new_empty",
        "new_duplicate",
        "old_duplicate",
        "too_many",
    ],
)
def test_forged_previous_transaction_or_runtime_claims_cannot_advance(change):
    values = arguments(old_claim=True)
    previous, new = values["previous"], values["claims"][0]
    if change == "schema":
        previous["schema"] = "historical_original_manifest_binding_v52"
    elif change in {"prior_producer", "prior_writer", "prior_blob"}:
        previous[
            {
                "prior_producer": "original_producer_commit",
                "prior_writer": "original_producer_writer_path",
                "prior_blob": "original_producer_source_blob_sha256",
            }[change]
        ] = "f" * 40
    elif change == "prior_science":
        previous["fresh_confirmation_authorized"] = True
    elif change == "prior_digest":
        previous["artifact_sha256"] = "f" * 64
    elif change == "old_claim_missing":
        values["original_claims"] = []
    elif change.startswith("transaction_"):
        values["transaction"][
            {
                "transaction_hash": "after_sha256",
                "transaction_bytes": "bytes_read",
                "transaction_time": "seconds",
            }[change]
        ] = {"transaction_hash": "f" * 64, "transaction_bytes": 1, "transaction_time": 211}[change]
    elif change in {
        "new_role",
        "new_source",
        "new_blob",
        "new_writer",
        "new_verification",
        "new_hash",
        "new_extra",
    }:
        key, value = {
            "new_role": ("role", "runtime_writer"),
            "new_source": ("recorded_source_sha", "f" * 40),
            "new_blob": ("source_blob_sha256", "f" * 64),
            "new_writer": ("writer_path", "scripts/worker.py"),
            "new_verification": ("source_evidence_verified", 0),
            "new_hash": ("artifact_sha256", "f" * 64),
            "new_extra": ("seal_commit", "f" * 40),
        }[change]
        new[key] = value
    elif change == "new_empty":
        values["claims"] = []
    elif change == "new_duplicate":
        values["claims"].append(deepcopy(new))
    elif change == "old_duplicate":
        values["claims"] = deepcopy(values["original_claims"])
    else:
        values["claims"] *= seals.MAXIMUM_CLAIMS + 1
    with pytest.raises(ValueError):
        seals.augment_seal_binding(**values)
