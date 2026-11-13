"""Synthetic metadata only: residual inventory cannot open or waive artifacts."""

from copy import deepcopy
from pathlib import Path

import pytest

from mosaickv.evaluation import parquet_binding_inventory_v48 as inventory
from mosaickv.evaluation.fox_campaign import digest


def fixture():
    errors = [
        dict(path="/runs/a/summary.parquet", error=inventory.PARQUET_ERROR),
        dict(path="/runs/b/summary.parquet", error=inventory.PARQUET_ERROR),
        dict(
            path="/runs/fixture/.codex/auth.json",
            error="ValueError: truncated historical metadata JSON",
        ),
    ]
    raw = [
        dict(
            owner_index=owner,
            historical_error_ordinal=0,
            original_error_sha256=digest(error),
            **error,
        )
        for owner, error in enumerate(errors)
    ]
    registrations = [
        dict(
            **{name: row[name] for name in inventory.contract.OCCURRENCE_FIELDS | {"path"}},
            operation="small" if index == 2 else "parquet",
            signature=[1, index, 41, 3],
        )
        for index, row in enumerate(raw)
    ]
    parent = dict(
        source_sha="a" * 40,
        raw_error_records=raw,
        resolved_occurrences=[],
        unresolved_occurrences=deepcopy(raw),
        counts=dict(
            raw_error_count=3,
            resolved_error_count=0,
            net_unresolved_error_count=3,
            original_seed_error_count=3,
            newly_observed_error_count=0,
            unresolved_original_occurrences=3,
            unresolved_newly_observed_occurrences=0,
            resolved_newly_observed_occurrences=0,
            inherited_resolved_original_occurrences=0,
            newly_resolved_original_occurrences=0,
        ),
        source_bindings={"/metadata/parent.json": "a" * 64, errors[0]["path"]: "b" * 64},
        **dict.fromkeys(inventory.contract.FALSE_FIELDS, False),
    )
    return parent, registrations


def run(parent, registrations):
    return inventory.build_inventory(parent, registrations, auditor_source_sha="c" * 40)


def test_inventory_preserves_denominators_and_never_promotes_bindings():
    parent, registrations = fixture()
    original = deepcopy((parent, registrations))
    result = run(parent, registrations)
    assert (parent, registrations) == original
    assert all(result["counts"][key] == value for key, value in parent["counts"].items())
    assert result["counts"]["parquet_occurrences"] == 2
    assert result["counts"]["protected_occurrences"] == 1
    assert result["counts"]["existing_artifact_bindings"] == 1
    assert result["counts"]["missing_artifact_bindings"] == 1
    assert result["status_counts"] == dict(
        existing_artifact_hash_only=1, missing_artifact_binding=1, protected_credential_not_read=1
    )
    assert result["auditor_source_sha"] == "c" * 40
    assert result["parent_auditor_source_sha"] == "a" * 40
    assert result["rows"][0]["existing_artifact_sha256"] == "b" * 64
    assert all(row["original_producer_commit"] is None for row in result["rows"])
    assert (
        result["newly_resolved_occurrences"]
        == result["source_bodies_read"]
        == result["GPU_jobs"]
        == 0
    )
    assert all(result[name] is False for name in inventory.contract.FALSE_FIELDS)


def test_protected_fixture_and_all_paths_have_no_filesystem_operations(monkeypatch):
    parent, registrations = fixture()

    def forbidden(*_args, **_kwargs):
        pytest.fail("inventory attempted filesystem access")

    monkeypatch.setattr("builtins.open", forbidden)
    for method in ("open", "resolve", "stat", "lstat", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, method, forbidden)
    result = run(parent, registrations)
    assert result["rows"][2]["status"] == "protected_credential_not_read"
    assert result["rows"][2]["existing_artifact_sha256"] is None


def test_order_is_occurrence_canonical_not_registration_or_input_order():
    parent, registrations = fixture()
    expected = run(parent, registrations)
    parent["unresolved_occurrences"].reverse()
    parent["raw_error_records"].reverse()
    assert run(parent, list(reversed(registrations))) == expected


@pytest.mark.parametrize("hint", ["source_sha", "producer_commit", "schema", "filename"])
def test_schema_filename_and_auditor_hints_cannot_become_producer_proof(hint):
    parent, registrations = fixture()
    parent[hint] = "d" * 40 if hint != "schema" else {"fields": ["score"]}
    result = run(parent, registrations)
    assert all(row["original_producer_commit"] is None for row in result["rows"])
    assert result["producer_inferred_from_schema_filename_or_auditor"] is False


@pytest.mark.parametrize(
    "name",
    [
        "raw_error_count",
        "resolved_error_count",
        "net_unresolved_error_count",
        "original_seed_error_count",
        "newly_observed_error_count",
        "unresolved_original_occurrences",
        "unresolved_newly_observed_occurrences",
        "resolved_newly_observed_occurrences",
        "inherited_resolved_original_occurrences",
        "newly_resolved_original_occurrences",
    ],
)
def test_every_registered_denominator_conserves(name):
    parent, registrations = fixture()
    parent["counts"][name] += 1
    with pytest.raises(ValueError, match="denominators"):
        run(parent, registrations)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_registration",
        "duplicate_raw",
        "duplicate_unresolved",
        "missing_registration",
        "missing_raw",
        "unresolved_rewrite",
        "raw_rewrite",
        "wrong_hash",
        "changed_path",
        "boolean_owner",
        "boolean_signature",
        "wrong_operation",
        "extra_registration_field",
    ],
)
def test_changed_or_duplicate_occurrence_bindings_fail_closed(change):
    parent, registrations = fixture()
    if change == "duplicate_registration":
        registrations.append(deepcopy(registrations[0]))
    elif change == "duplicate_raw":
        parent["raw_error_records"].append(deepcopy(parent["raw_error_records"][0]))
    elif change == "duplicate_unresolved":
        parent["unresolved_occurrences"].append(deepcopy(parent["unresolved_occurrences"][0]))
    elif change == "missing_registration":
        registrations.pop(0)
    elif change == "missing_raw":
        parent["raw_error_records"].pop(0)
    elif change == "unresolved_rewrite":
        parent["unresolved_occurrences"][0]["path"] = "/runs/changed.parquet"
    elif change == "raw_rewrite":
        parent["raw_error_records"][0]["path"] = "/runs/changed.parquet"
    elif change == "wrong_hash":
        registrations[0]["original_error_sha256"] = "e" * 64
    elif change == "changed_path":
        registrations[0]["path"] = "/runs/changed.parquet"
    elif change == "boolean_owner":
        registrations[0]["owner_index"] = True
    elif change == "boolean_signature":
        registrations[0]["signature"][0] = True
    elif change == "wrong_operation":
        registrations[0]["operation"] = "json_stream"
    else:
        registrations[0]["producer_commit"] = "d" * 40
    with pytest.raises(ValueError):
        run(parent, registrations)


@pytest.mark.parametrize(
    "error",
    [
        "Parquet has no safely projectable",
        "unknown error",
        "ValueError: truncated historical metadata JSON",
    ],
)
def test_unknown_or_changed_error_categories_never_nominated(error):
    parent, registrations = fixture()
    for row in (parent["raw_error_records"][0], parent["unresolved_occurrences"][0]):
        row["error"] = error
        row["original_error_sha256"] = digest({"path": row["path"], "error": error})
    registrations[0]["original_error_sha256"] = parent["raw_error_records"][0][
        "original_error_sha256"
    ]
    with pytest.raises(ValueError, match="unreviewed"):
        run(parent, registrations)


@pytest.mark.parametrize(
    "change",
    ["fresh", "bad_binding", "bad_auditor", "conflicting_geometry", "unknown_protected_error"],
)
def test_scope_and_binding_guardrails(change):
    parent, registrations = fixture()
    if change == "fresh":
        parent["fresh_confirmation_authorized"] = True
    elif change == "bad_binding":
        parent["source_bindings"]["/runs/unknown.parquet"] = "bad"
    elif change == "bad_auditor":
        with pytest.raises(ValueError):
            inventory.build_inventory(parent, registrations, auditor_source_sha="bad")
        return
    elif change == "conflicting_geometry":
        registrations.append(dict(registrations[0], owner_index=7, signature=[1, 0, 42, 3]))
    else:
        raw = parent["raw_error_records"][2]
        raw["error"] = "unknown credential fixture"
        raw["original_error_sha256"] = digest({"path": raw["path"], "error": raw["error"]})
        parent["unresolved_occurrences"][2] = deepcopy(raw)
        registrations[2]["original_error_sha256"] = raw["original_error_sha256"]
    with pytest.raises(ValueError):
        run(parent, registrations)


def test_completed_registration_rows_are_not_renominated():
    parent, registrations = fixture()
    completed = parent["unresolved_occurrences"].pop(0)
    parent["resolved_occurrences"].append(deepcopy(completed))
    parent["counts"].update(
        resolved_error_count=1,
        net_unresolved_error_count=2,
        newly_resolved_original_occurrences=1,
        unresolved_original_occurrences=2,
    )
    result = run(parent, registrations)
    assert len(result["rows"]) == 2
    assert all(row["path"] != completed["path"] for row in result["rows"])


def test_parent_auditor_sha_requires_a_string_not_a_numeric_lookalike():
    parent, registrations = fixture()
    parent["source_sha"] = int("1" * 40)
    with pytest.raises(ValueError, match="auditor SHA"):
        run(parent, registrations)


def test_original_owner_ordinals_cannot_be_renumbered_consistently():
    parent, registrations = fixture()
    for row in (
        parent["raw_error_records"][0],
        parent["unresolved_occurrences"][0],
        registrations[0],
    ):
        row["historical_error_ordinal"] = 1
    with pytest.raises(ValueError, match="sequence has a gap"):
        run(parent, registrations)
