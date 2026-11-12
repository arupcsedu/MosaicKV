"""Synthetic occurrence/identity reconciliation; no filesystem or model access."""

from __future__ import annotations

import copy

import pytest

from mosaickv.evaluation import history_occurrence_reconciliation as core


def ids(**kwargs):
    return {name: kwargs.get(name, []) for name in core.IDENTITY_FIELDS}


def raw(path, error="YAML requires separately reviewed lexical identity projection"):
    return dict(path=path, error=error)


def wrap(error, owner, ordinal):
    return dict(
        owner_index=owner,
        historical_error_ordinal=ordinal,
        original_error_sha256=core.digest(error),
        **error,
    )


def proof(error, **kwargs):
    return dict(
        path=error["path"],
        original_error_sha256s=[core.digest(error)],
        whole_file_sha256_before="c" * 64,
        whole_file_sha256_after="c" * 64,
        registration_sha256="d" * 64,
        kind="yaml_events",
        resolved_as="unchanged_YAML_event_metadata_projected",
        identities=dict(
            **ids(
                sample_ids=["from-yaml"],
                media_hashes=["e" * 64],
                scopes=[["COCO", "", "val"]],
                scoped_identities=[["COCO", "", "val", "from-yaml"]],
            ),
            projected_records=1,
        ),
        **kwargs,
    )


@pytest.fixture
def inputs():
    a, b, c = (
        raw("/scratch/history/a.yaml"),
        raw("/scratch/history/b.yaml"),
        raw("/scratch/history/c.json"),
    )
    counts = {str(owner): 0 for owner in core.OWNERS} | {"0": 2, "1": 1}
    originals = [a, b, c]
    inherited = [core.digest(a)]
    # Same original error/hash repeated AFTER the immutable seed prefix.
    rows = [
        wrap(a, 0, 0),
        wrap(b, 0, 1),
        wrap(a, 0, 2),
        wrap(b, 0, 3),
        wrap(c, 1, 0),
        wrap(a, 1, 1),
    ]
    binding = {"/scratch/parent/receipt.json": "a" * 64}
    inventory = dict(
        source_sha="b" * 40,
        identities=ids(sample_ids=["already-excluded"], source_group_ids=["known-group"]),
        errors=rows,
        fresh_confirmation_authorized=False,
        complete_historical_id_inventory=False,
    )
    overlay = dict(
        source_sha=inventory["source_sha"],
        original_errors_preserved=True,
        parent_bindings=binding,
        covered_before_error_hashes=inherited,
        new_proofs=[proof(b)],
        fresh_confirmation_authorized=False,
    )
    return dict(
        global_inventory=inventory,
        seed_error_counts=counts,
        preserved_resolved_error_sha256s=inherited,
        original_errors=originals,
        source_bindings=binding,
        coverage_overlay=overlay,
    )


def test_new_identical_error_occurrences_never_waived(inputs):
    result = core.reconcile_occurrences(**inputs)
    assert result["counts"] == dict(
        raw_error_count=6,
        original_seed_error_count=3,
        newly_observed_error_count=3,
        inherited_resolved_original_occurrences=1,
        newly_resolved_original_occurrences=1,
        resolved_error_count=2,
        net_unresolved_error_count=4,
        unresolved_original_occurrences=1,
        unresolved_newly_observed_occurrences=3,
    )
    assert [
        (row["owner_index"], row["historical_error_ordinal"])
        for row in result["unresolved_occurrences"]
    ] == [(0, 2), (0, 3), (1, 0), (1, 1)]
    assert result["new_duplicate_occurrences_waived"] is False
    assert all(result[field] is False for field in core.FALSE_FIELDS)


def test_all_identity_exclusions_union_without_losing_originals(inputs):
    result = core.reconcile_occurrences(**inputs)
    assert result["identities"]["sample_ids"] == ["already-excluded", "from-yaml"]
    assert result["identities"]["source_group_ids"] == ["known-group"]
    assert result["identities"]["media_hashes"] == ["e" * 64]
    assert result["identities"]["scopes"] == [["COCO", "", "val"]]
    assert result["identity_addition_counts"]["sample_ids"] == 1
    assert result["preexisting_identity_exclusions_preserved"] is True


def test_inputs_and_raw_rows_remain_immutable(inputs):
    before = copy.deepcopy(inputs)
    result = core.reconcile_occurrences(**inputs)
    assert inputs == before
    assert result["raw_error_records"] == inputs["global_inventory"]["errors"]
    result["raw_error_records"][0]["error"] = "MUTATED RETURN VALUE"
    result["unresolved_occurrences"][0]["path"] = "/scratch/changed"
    result["source_bindings"].clear()
    assert inputs == before


def test_unavailable_new_overlay_leaves_original_unresolved(inputs):
    inputs["coverage_overlay"] = None
    result = core.reconcile_occurrences(**inputs)
    assert result["counts"]["newly_resolved_original_occurrences"] == 0
    assert result["counts"]["net_unresolved_error_count"] == 5
    assert result["identities"]["sample_ids"] == ["already-excluded"]


def test_inherited_identity_proofs_are_merged_as_exclusions_only(inputs):
    prior = proof(inputs["original_errors"][0])
    prior.pop("kind")
    prior.pop("registration_sha256")
    prior.pop("resolved_as")
    prior["identities"]["sample_ids"] = ["old-YAML-exclusion"]
    inputs["additional_identity_proofs"] = [prior]
    result = core.reconcile_occurrences(**inputs)
    assert result["identities"]["sample_ids"] == [
        "already-excluded",
        "from-yaml",
        "old-YAML-exclusion",
    ]
    assert result["counts"]["resolved_error_count"] == 2
    assert result["inherited_identity_proof_sha256s"] == [core.digest(prior)]


def test_old_proofs_accept_exact_singular_error_hash_binding(inputs):
    prior = proof(inputs["original_errors"][0])
    prior["original_error_sha256"] = prior.pop("original_error_sha256s")[0]
    inputs["additional_identity_proofs"] = [prior]
    assert core.reconcile_occurrences(**inputs)["counts"]["resolved_error_count"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_index", True),
        ("owner_index", 8),
        ("historical_error_ordinal", -1),
        ("historical_error_ordinal", False),
        ("original_error_sha256", "0" * 64),
    ],
)
def test_invalid_occurrence_wrapper_rejected(inputs, field, value):
    inputs["global_inventory"]["errors"][0][field] = value
    with pytest.raises(ValueError):
        core.reconcile_occurrences(**inputs)


def test_owner_or_ordinal_reassignment_does_not_reuse_old_proof(inputs):
    left, right = inputs["global_inventory"]["errors"][0], inputs["global_inventory"]["errors"][1]
    left["historical_error_ordinal"], right["historical_error_ordinal"] = 1, 0
    with pytest.raises(ValueError, match="prefix"):
        core.reconcile_occurrences(**inputs)


def test_original_error_drop_rejected(inputs):
    del inputs["global_inventory"]["errors"][0]
    with pytest.raises(ValueError, match="dropped"):
        core.reconcile_occurrences(**inputs)


def test_new_ordinal_gap_rejected(inputs):
    inputs["global_inventory"]["errors"][3]["historical_error_ordinal"] = 5
    with pytest.raises(ValueError, match="ordinal"):
        core.reconcile_occurrences(**inputs)


def test_duplicate_owner_ordinal_rejected(inputs):
    inputs["global_inventory"]["errors"].append(
        copy.deepcopy(inputs["global_inventory"]["errors"][0])
    )
    with pytest.raises(ValueError, match="duplicate"):
        core.reconcile_occurrences(**inputs)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d["seed_error_counts"].pop("7"),
        lambda d: d["seed_error_counts"].update({"7": True}),
        lambda d: d["seed_error_counts"].update({"7": -1}),
        lambda d: d["seed_error_counts"].update({"7": 1}),
        lambda d: d["original_errors"].append(copy.deepcopy(d["original_errors"][0])),
    ],
)
def test_seed_count_and_original_universe_validation(inputs, mutator):
    mutator(inputs)
    with pytest.raises(ValueError):
        core.reconcile_occurrences(**inputs)


def test_overlay_cannot_change_inherited_scope(inputs):
    inputs["coverage_overlay"]["covered_before_error_hashes"] = []
    with pytest.raises(ValueError, match="inherited resolution scope"):
        core.reconcile_occurrences(**inputs)


def test_overlay_original_proof_must_bind_exact_path(inputs):
    inputs["coverage_overlay"]["new_proofs"][0]["path"] = "/scratch/history/other.yaml"
    with pytest.raises(ValueError, match="proof path"):
        core.reconcile_occurrences(**inputs)


def test_overlay_cannot_resolve_error_outside_original_universe(inputs):
    inputs["coverage_overlay"]["new_proofs"][0]["original_error_sha256s"] = ["f" * 64]
    with pytest.raises(ValueError, match="escapes"):
        core.reconcile_occurrences(**inputs)


def test_missing_parent_proof_hash_rejected(inputs):
    inputs["source_bindings"] = {"/scratch/other/receipt.json": "f" * 64}
    with pytest.raises(ValueError, match="parent binding"):
        core.reconcile_occurrences(**inputs)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.update(whole_file_sha256_after="f" * 64),
        lambda p: p.update(kind="filename_exemption"),
        lambda p: p.update(resolved_as="arbitrary_pass"),
        lambda p: p.update(registration_sha256="invalid"),
        lambda p: p["identities"].update(projected_records=True),
        lambda p: p["identities"].update(media_hashes=["not-a-media-hash"]),
        lambda p: p["identities"].update(scopes=[["wrong_width"]]),
    ],
)
def test_malformed_or_unreviewed_proofs_rejected(inputs, mutator):
    mutator(inputs["coverage_overlay"]["new_proofs"][0])
    with pytest.raises(ValueError):
        core.reconcile_occurrences(**inputs)


def test_duplicate_additive_proof_rejected(inputs):
    inputs["coverage_overlay"]["new_proofs"] *= 2
    with pytest.raises(ValueError, match="overlaps"):
        core.reconcile_occurrences(**inputs)


def test_credential_body_cannot_become_a_proof_binding(inputs):
    inputs["source_bindings"] = {"/scratch/.codex/auth.json": "a" * 64}
    with pytest.raises(ValueError, match="credential"):
        core.reconcile_occurrences(**inputs)


@pytest.mark.parametrize("location", ["global_inventory", "coverage_overlay"])
def test_positive_freshness_claim_rejected_not_silently_normalized(inputs, location):
    inputs[location]["fresh_confirmation_authorized"] = True
    with pytest.raises(ValueError, match=r"freshness|scope"):
        core.reconcile_occurrences(**inputs)


def test_complete_resolution_still_does_not_authorize_freshness(inputs):
    inputs["global_inventory"]["errors"] = [
        row
        for row in inputs["global_inventory"]["errors"]
        if row["historical_error_ordinal"] < inputs["seed_error_counts"][str(row["owner_index"])]
    ]
    inputs["preserved_resolved_error_sha256s"].append(core.digest(inputs["original_errors"][2]))
    inputs["coverage_overlay"]["covered_before_error_hashes"] = inputs[
        "preserved_resolved_error_sha256s"
    ]
    result = core.reconcile_occurrences(**inputs)
    assert result["counts"]["net_unresolved_error_count"] == 0
    assert all(result[field] is False for field in core.FALSE_FIELDS)
    assert result["files_opened"] == result["scans_run"] == result["GPU_jobs_submitted"] == 0


def with_inventory(inputs):
    path = "/scratch/old-oversize/combined_identity_inventory.json"
    inputs["source_bindings"][path] = "f" * 64
    inventory = dict(
        source_sha="a" * 40,
        identities=ids(
            sample_ids=["old-oversize-exclusion"], source_group_ids=["old-source-group"]
        ),
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        fresh_confirmation_authorized=False,
    )
    inputs["additional_identity_inventories"] = [
        dict(path=path, sha256="f" * 64, inventory=inventory)
    ]
    return inventory


def test_hash_bound_old_inventory_adds_exclusions_not_error_waivers(inputs):
    inventory = with_inventory(inputs)
    expected = copy.deepcopy(inputs)
    result = core.reconcile_occurrences(**inputs)
    assert "old-oversize-exclusion" in result["identities"]["sample_ids"]
    assert "old-source-group" in result["identities"]["source_group_ids"]
    assert result["counts"]["resolved_error_count"] == 2
    assert result["counts"]["unresolved_newly_observed_occurrences"] == 3
    receipts = result["inherited_identity_inventory_receipts"]
    assert len(receipts) == 1
    assert receipts[0]["projected_inventory_sha256"] == core.digest(inventory)
    assert inputs == expected


@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d["additional_identity_inventories"][0].update(sha256="d" * 64),
        lambda d: d["additional_identity_inventories"][0].update(path="/scratch/unbound.json"),
        lambda d: d["additional_identity_inventories"].append(
            copy.deepcopy(d["additional_identity_inventories"][0])
        ),
        lambda d: d["additional_identity_inventories"][0]["inventory"].update(
            fresh_confirmation_authorized=True
        ),
        lambda d: d["additional_identity_inventories"][0]["inventory"]["identities"].pop(
            "sample_ids"
        ),
    ],
)
def test_inventory_receipt_binding_and_scope_must_validate(inputs, mutator):
    with_inventory(inputs)
    mutator(inputs)
    with pytest.raises(ValueError):
        core.reconcile_occurrences(**inputs)


def test_reconciliation_performs_no_filesystem_access(inputs, monkeypatch):
    import builtins
    import os

    def forbidden(*args, **kwargs):
        raise AssertionError("pure reconciliation attempted filesystem access")

    with monkeypatch.context() as context:
        context.setattr(builtins, "open", forbidden)
        context.setattr(os, "stat", forbidden)
        context.setattr(os, "scandir", forbidden)
        result = core.reconcile_occurrences(**inputs)
    assert result["files_opened"] == 0
