"""Pure synthetic receipt fixtures; no file, dataset, model or scheduler access."""

from copy import deepcopy

import pytest

from mosaickv.evaluation import residual_occurrence_proofs as reducer
from mosaickv.evaluation.fox_campaign import digest


def fixture():
    seeds = {str(owner): 2 if owner == 0 else 0 for owner in range(8)}
    errors = [
        dict(
            path="/runs/config.yaml",
            error="YAML requires separately reviewed lexical identity projection",
        ),
        dict(
            path="/runs/derived.parquet",
            error="Parquet has no safely projectable identity metadata columns",
        ),
    ]
    raw = [
        dict(
            owner_index=owner,
            historical_error_ordinal=ordinal,
            original_error_sha256=digest(errors[index]),
            **errors[index],
        )
        for owner, ordinal, index in ((0, 0, 0), (0, 1, 1), (1, 0, 0))
    ]
    identities = dict.fromkeys(reducer.contract.IDENTITY_FIELDS)
    identities = {key: [] for key in identities}
    identities["sample_ids"] = ["old-id"]
    report = dict(
        source_sha="a" * 40,
        input_metadata_sha256s=dict(seed_error_counts=digest(seeds)),
        source_bindings={"/metadata/parent.json": "a" * 64},
        raw_error_records=raw,
        resolved_occurrences=[
            dict(
                owner_index=0,
                historical_error_ordinal=1,
                original_error_sha256=raw[1]["original_error_sha256"],
                resolution_kind="inherited",
                additive_proof_sha256="b" * 64,
            )
        ],
        unresolved_occurrences=[raw[0], raw[2]],
        counts=dict(
            raw_error_count=3,
            original_seed_error_count=2,
            newly_observed_error_count=1,
            inherited_resolved_original_occurrences=1,
            newly_resolved_original_occurrences=0,
            resolved_error_count=1,
            net_unresolved_error_count=2,
            unresolved_original_occurrences=1,
            unresolved_newly_observed_occurrences=1,
        ),
        identities=identities,
        identity_addition_counts={key: 0 for key in identities},
        additive_proof_sha256s=["b" * 64],
        identity_proof_sha256s=[],
        unresolved_error_categories={errors[0]["error"]: 2},
        **dict.fromkeys(reducer.contract.FALSE_FIELDS, False),
    )
    return report, seeds


def proof(row, kind="yaml_events", identities=None):
    registration = {name: deepcopy(row[name]) for name in reducer.contract.OCCURRENCE_FIELDS}
    registration.update(path=row["path"], kind=kind, signature=[1, 2, 30, 4], file_sha256="f" * 64)
    receipt = dict(
        registration_sha256=digest(registration),
        kind=kind,
        signature=[1, 2, 30, 4],
        whole_file_sha256_before="f" * 64,
        whole_file_sha256_after="f" * 64,
    )
    if identities is not None:
        receipt["identities"] = identities
    return registration, receipt


def apply(report, seeds, registrations, receipts, **kwargs):
    return reducer.apply_proofs(
        report,
        registrations,
        receipts,
        source_bindings={"/metadata/receipts.json": "e" * 64},
        analysis_source_sha="c" * 40,
        seed_error_counts=seeds,
        **kwargs,
    )


def test_one_original_receipt_never_waives_identical_new_occurrence():
    parent, seeds = fixture()
    before = deepcopy(parent)
    reg, receipt = proof(parent["unresolved_occurrences"][0])
    result = apply(parent, seeds, [reg], [receipt])
    assert parent == before
    assert result["raw_error_records"] == parent["raw_error_records"]
    assert result["resolved_occurrences"][:1] == parent["resolved_occurrences"]
    assert result["unresolved_occurrences"] == [parent["raw_error_records"][2]]
    assert result["counts"]["unresolved_original_occurrences"] == 0
    assert result["counts"]["unresolved_newly_observed_occurrences"] == 1
    assert result["counts"]["newly_resolved_original_occurrences"] == 1
    assert result["counts"]["resolved_newly_observed_occurrences"] == 0


def test_distinct_registered_occurrences_can_share_one_source_transaction():
    parent, seeds = fixture()
    pairs = [proof(row) for row in parent["unresolved_occurrences"]]
    result = apply(parent, seeds, [p[0] for p in pairs], [p[1] for p in pairs])
    assert result["counts"]["resolved_error_count"] == 3
    assert result["counts"]["net_unresolved_error_count"] == 0
    assert result["counts"]["resolved_newly_observed_occurrences"] == 1
    assert result["raw_error_records"] == parent["raw_error_records"]
    assert all(result[field] is False for field in reducer.contract.FALSE_FIELDS)


def test_yaml_identity_additions_are_exclusions_and_preserve_prior_identity_rows():
    parent, seeds = fixture()
    identities = {key: [] for key in reducer.contract.IDENTITY_FIELDS}
    identities["sample_ids"] = ["new-id"]
    identities["scopes"] = [["dataset", "revision", "split"]]
    reg, receipt = proof(parent["unresolved_occurrences"][0], identities=identities)
    result = apply(parent, seeds, [reg], [receipt])
    assert result["identities"]["sample_ids"] == ["new-id", "old-id"]
    assert result["identity_addition_counts"]["sample_ids"] == 1
    assert result["identities"]["scopes"] == identities["scopes"]
    assert result["parent_report_sha256"] == digest(parent)


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "ordinal",
        "rawhash",
        "path",
        "kind",
        "signature",
        "before",
        "after",
        "registration",
        "extra",
        "missing",
    ],
)
def test_forged_or_mismatched_receipt_rejected(change):
    parent, seeds = fixture()
    reg, receipt = proof(parent["unresolved_occurrences"][0])
    if change == "owner":
        reg["owner_index"] = True
    elif change == "ordinal":
        reg["historical_error_ordinal"] = 100
    elif change == "rawhash":
        reg["original_error_sha256"] = "d" * 64
    elif change == "path":
        reg["path"] = "/runs/other.yaml"
    elif change == "kind":
        reg["kind"] = receipt["kind"] = "closed_parquet"
    elif change == "signature":
        receipt["signature"][0] = True
    elif change in {"before", "after"}:
        receipt[f"whole_file_sha256_{change}"] = "d" * 64
    elif change == "registration":
        receipt["registration_sha256"] = "d" * 64
    elif change == "extra":
        receipt["unreviewed_waiver"] = True
    else:
        del receipt["signature"]
    if change != "registration":
        receipt["registration_sha256"] = digest(reg)
    with pytest.raises(ValueError):
        apply(parent, seeds, [reg], [receipt])


def test_duplicate_or_already_resolved_occurrence_rejected():
    parent, seeds = fixture()
    reg, receipt = proof(parent["unresolved_occurrences"][0])
    with pytest.raises(ValueError):
        apply(parent, seeds, [reg, reg], [receipt, receipt])
    reg, receipt = proof(parent["raw_error_records"][1], kind="closed_parquet")
    with pytest.raises(ValueError):
        apply(parent, seeds, [reg], [receipt])


def test_source_binding_collision_and_inconsistent_shared_file_geometry_rejected():
    parent, seeds = fixture()
    reg, receipt = proof(parent["unresolved_occurrences"][0])
    parent["source_bindings"]["/metadata/receipts.json"] = "d" * 64
    with pytest.raises(ValueError, match="collision"):
        apply(parent, seeds, [reg], [receipt])
    parent, seeds = fixture()
    other_reg, other_receipt = proof(parent["unresolved_occurrences"][1])
    other_reg["signature"][1] = 99
    other_receipt["signature"][1] = 99
    other_receipt["registration_sha256"] = digest(other_reg)
    with pytest.raises(ValueError, match="geometry"):
        apply(parent, seeds, [reg, other_reg], [receipt, other_receipt])


def test_seed_count_hash_and_parent_partition_corruption_rejected():
    parent, seeds = fixture()
    bad = dict(seeds)
    bad["0"] = 1
    with pytest.raises(ValueError, match="boundaries"):
        apply(parent, bad, [], [])
    parent["unresolved_occurrences"].append(deepcopy(parent["unresolved_occurrences"][0]))
    with pytest.raises(ValueError, match="partition"):
        apply(parent, seeds, [], [])


def test_closed_parquet_proof_cannot_smuggle_new_identity_claim():
    parent, seeds = fixture()
    # Make the originally unresolved occurrence a Parquet diagnostic.
    row = parent["raw_error_records"][0]
    row.update(
        path="/runs/unresolved.parquet",
        error="Parquet has no safely projectable identity metadata columns",
    )
    row["original_error_sha256"] = digest({key: row[key] for key in ("path", "error")})
    identities = {key: [] for key in reducer.contract.IDENTITY_FIELDS}
    identities["sample_ids"] = ["invented"]
    reg, receipt = proof(row, kind="closed_parquet", identities=identities)
    with pytest.raises(ValueError, match="no-identity"):
        apply(parent, seeds, [reg], [receipt])
