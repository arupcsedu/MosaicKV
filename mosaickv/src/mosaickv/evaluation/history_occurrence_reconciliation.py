"""Pure occurrence-aware reconciliation of reviewed historical metadata proofs.

The caller verifies immutable file hashes before passing their decoded metadata.
This module performs no filesystem, dataset, model, or scheduler access. A proof
for an original error cannot waive a later observation of an identical error.
Projected identities are conservative exclusions, never freshness certificates.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from mosaickv.evaluation.fox_campaign import digest

OWNERS = tuple(range(8))
HASH = re.compile(r"[a-f0-9]{64}\Z")
SHA = re.compile(r"[a-f0-9]{40}\Z")
IDENTITY_FIELDS = (
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
)
NESTED_WIDTHS = {"scopes": 3, "scoped_identities": 4}
OCCURRENCE_FIELDS = {"owner_index", "historical_error_ordinal", "original_error_sha256"}
FALSE_FIELDS = (
    "globally_fresh",
    "complete_historical_id_inventory",
    "global_historical_inventory_complete",
    "historical_media_exposure_resolved",
    "source_group_disjointness_verified",
    "global_media_source_group_inventory_complete",
    "fresh_confirmation_authorized",
    "materialization_authorized",
    "paper_campaign_authorized",
)
KINDS = {
    "yaml_events": "unchanged_YAML_event_metadata_projected",
    "single_json_document": "complete_single_JSON_document_metadata_projected",
    "closed_parquet": "derived_no_request_or_media_identity",
}


def _hash(value, label):
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise ValueError(f"explicit SHA256 required: {label}")
    return value


def _hashes(values, label):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"explicit hash sequence required: {label}")
    result = [_hash(value, label) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate proof hash: {label}")
    return set(result)


def _path(value, label, *, credential_allowed=False):
    if not isinstance(value, str):
        raise ValueError(f"absolute metadata path required: {label}")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"absolute normalized metadata path required: {label}")
    if not credential_allowed and (
        ".codex" in path.parts or path.name.lower() in {"auth.json", "credentials.json"}
    ):
        raise ValueError("credential bodies cannot supply recovery proof bindings")
    return value


def _bindings(values):
    if not isinstance(values, Mapping) or not values:
        raise ValueError("nonempty hash-bound caller provenance is required")
    return {
        _path(path, "source binding"): _hash(expected, "source binding")
        for path, expected in values.items()
    }


def _counts(values):
    if not isinstance(values, Mapping) or set(values) != {str(owner) for owner in OWNERS}:
        raise ValueError("all eight original owner seed counts are required")
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError("strict nonnegative owner seed counts required")
    return dict(values)


def _raw_error(value):
    if not isinstance(value, dict) or OCCURRENCE_FIELDS & value.keys():
        raise ValueError("raw error cannot contain occurrence-wrapper fields")
    _path(value.get("path"), "raw error path", credential_allowed=True)
    if not isinstance(value.get("error"), str) or not value["error"]:
        raise ValueError("raw metadata error message required")
    digest(value)  # Strict finite JSON; never decode or reinterpret a payload.
    return value


def _identities(values, *, proof=False):
    expected = set(IDENTITY_FIELDS) | ({"projected_records"} if proof else set())
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("complete six-field identity projection required")
    result = {}
    for name in IDENTITY_FIELDS:
        rows = values[name]
        if not isinstance(rows, list):
            raise ValueError("explicit identity list required")
        if name in NESTED_WIDTHS:
            width = NESTED_WIDTHS[name]
            if any(
                not isinstance(row, (list, tuple))
                or len(row) != width
                or any(not isinstance(item, str) for item in row)
                for row in rows
            ):
                raise ValueError("identity scope tuple width/type differs")
            result[name] = {tuple(row) for row in rows}
        else:
            if any(not isinstance(row, str) for row in rows):
                raise ValueError("identity scalar type differs")
            if name == "media_hashes":
                for row in rows:
                    _hash(row, "media exclusion")
            result[name] = set(rows)
    if proof and (type(values["projected_records"]) is not int or values["projected_records"] < 0):
        raise ValueError("strict nonnegative projected record count required")
    return result


def _proof_hashes(proof):
    many, one = "original_error_sha256s" in proof, "original_error_sha256" in proof
    if many == one:
        raise ValueError("one unambiguous original-error proof binding required")
    return _hashes(
        proof["original_error_sha256s"] if many else [proof["original_error_sha256"]],
        "original-error proof",
    )


def _proof(proof, originals, *, additional=False):
    if not isinstance(proof, dict):
        raise ValueError("reviewed metadata proof dictionary required")
    path = _path(proof.get("path"), "projected proof path")
    before = _hash(proof.get("whole_file_sha256_before"), "proof before")
    if proof.get("whole_file_sha256_after") != before:
        raise ValueError("proof source bytes changed during transaction")
    hashes = _proof_hashes(proof)
    if not hashes or not hashes <= originals.keys():
        raise ValueError("proof escapes original error universe")
    if any(originals[value]["path"] != path for value in hashes):
        raise ValueError("proof path differs from its exact original error")
    identities = None
    if "identities" in proof:
        identities = _identities(proof["identities"], proof=True)
    if not additional:
        kind = proof.get("kind")
        if kind not in KINDS or proof.get("resolved_as") != KINDS[kind]:
            raise ValueError("unreviewed V36 recovery kind or result label")
        _hash(proof.get("registration_sha256"), "registered V36 proof")
        if kind in {"yaml_events", "single_json_document"} and identities is None:
            raise ValueError("projected metadata proof lost its identity exclusions")
        if kind == "single_json_document" and (
            proof.get("malformed_lines_skipped") != 0
            or proof.get("suffix_format_mismatch_preserved") is not True
        ):
            raise ValueError("single JSON proof cannot waive malformed JSONL lines")
        if kind == "closed_parquet" and proof.get("parquet_footer", {}).get("rows_decoded") != 0:
            raise ValueError("closed Parquet proof must retain metadata-only provenance")
    return hashes, identities


def reconcile_occurrences(
    global_inventory,
    *,
    seed_error_counts,
    preserved_resolved_error_sha256s,
    original_errors,
    source_bindings,
    coverage_overlay=None,
    additional_identity_proofs=(),
    additional_identity_inventories=(),
):
    """Reconcile only original occurrences; retain every new observation.

    ``original_errors`` is the V16 preserved list in original owner0..7 order.
    ``source_bindings`` contains paths/hashes already checked by the caller.
    Additional old identity proofs may add exclusions only when their original
    error hashes were already covered by the reviewed inherited receipts.
    """
    if not isinstance(global_inventory, dict) or not isinstance(
        global_inventory.get("errors"), list
    ):
        raise ValueError("complete projected global inventory dictionary required")
    if SHA.fullmatch(str(global_inventory.get("source_sha"))) is None:
        raise ValueError("global producer source SHA required")
    if any(global_inventory.get(field, False) is not False for field in FALSE_FIELDS):
        raise ValueError("unresolved inventory cannot carry positive freshness claims")
    counts = _counts(seed_error_counts)
    bindings = _bindings(source_bindings)
    identities = _identities(global_inventory.get("identities"))
    initial_identity_counts = {name: len(rows) for name, rows in identities.items()}
    if not isinstance(original_errors, list) or sum(counts.values()) != len(original_errors):
        raise ValueError("original raw errors do not match eight-owner seed prefix")
    originals = {}
    expected_prefix, offset = {}, 0
    for owner in OWNERS:
        count = counts[str(owner)]
        for ordinal, raw in enumerate(original_errors[offset : offset + count]):
            raw = _raw_error(raw)
            hashed = digest(raw)
            if hashed in originals:
                raise ValueError("original error hashes must remain unique")
            originals[hashed] = raw
            expected_prefix[(owner, ordinal)] = raw
        offset += count
    inherited = _hashes(preserved_resolved_error_sha256s, "reviewed inherited resolution")
    if not inherited <= originals.keys():
        raise ValueError("inherited resolution escapes original error scope")
    observed, owner_ordinals = {}, {owner: set() for owner in OWNERS}
    rows = global_inventory["errors"]
    for row in rows:
        if not isinstance(row, dict) or not row.keys() >= OCCURRENCE_FIELDS:
            raise ValueError("every global error needs owner, ordinal and raw-error SHA")
        owner, ordinal = row["owner_index"], row["historical_error_ordinal"]
        if type(owner) is not int or owner not in OWNERS or type(ordinal) is not int or ordinal < 0:
            raise ValueError("strict original-owner and nonnegative ordinal required")
        raw = _raw_error(
            {name: value for name, value in row.items() if name not in OCCURRENCE_FIELDS}
        )
        if _hash(row["original_error_sha256"], "global raw-error wrapper") != digest(raw):
            raise ValueError("global raw-error digest differs from occurrence wrapper")
        key = (owner, ordinal)
        if key in observed:
            raise ValueError("duplicate owner/ordinal observation")
        if ordinal < counts[str(owner)] and raw != expected_prefix[key]:
            raise ValueError("original per-owner error prefix was changed or reassigned")
        observed[key] = row
        owner_ordinals[owner].add(ordinal)
    for owner, ordinals in owner_ordinals.items():
        if ordinals != set(range(len(ordinals))) or len(ordinals) < counts[str(owner)]:
            raise ValueError("global inventory dropped an original error or ordinal")
    additive, proof_digests, identity_proof_digests = {}, [], []
    if coverage_overlay is not None:
        if not isinstance(coverage_overlay, dict):
            raise ValueError("hash-bound V36 coverage overlay required")
        if (
            coverage_overlay.get("source_sha") != global_inventory["source_sha"]
            or coverage_overlay.get("original_errors_preserved") is not True
            or any(coverage_overlay.get(field, False) is not False for field in FALSE_FIELDS)
        ):
            raise ValueError("coverage overlay producer or scientific scope differs")
        covered_before = _hashes(
            coverage_overlay.get("covered_before_error_hashes"), "overlay inherited"
        )
        if covered_before != inherited:
            raise ValueError("overlay changed inherited resolution scope")
        parent_bindings = _bindings(coverage_overlay.get("parent_bindings"))
        if any(bindings.get(path) != value for path, value in parent_bindings.items()):
            raise ValueError("overlay parent binding missing from reviewed caller provenance")
        proofs = coverage_overlay.get("new_proofs")
        if not isinstance(proofs, list):
            raise ValueError("explicit V36 proof list required")
        for proof in proofs:
            hashes, additions = _proof(proof, originals)
            if hashes & inherited or hashes & additive.keys():
                raise ValueError("additive proof overlaps inherited or another proof")
            hashed = digest(proof)
            proof_digests.append(hashed)
            additive.update({value: hashed for value in hashes})
            if additions is not None:
                for name in IDENTITY_FIELDS:
                    identities[name].update(additions[name])
                identity_proof_digests.append(hashed)
    if not isinstance(additional_identity_proofs, Sequence) or isinstance(
        additional_identity_proofs, (str, bytes)
    ):
        raise ValueError("explicit reviewed old identity-proof sequence required")
    additional_digests = set()
    for proof in additional_identity_proofs:
        hashes, additions = _proof(proof, originals, additional=True)
        if not hashes <= inherited or additions is None:
            raise ValueError("additional exclusions require reviewed inherited identity proofs")
        hashed = digest(proof)
        if hashed in additional_digests:
            raise ValueError("duplicate inherited identity proof supplied")
        additional_digests.add(hashed)
        for name in IDENTITY_FIELDS:
            identities[name].update(additions[name])
        identity_proof_digests.append(hashed)
    if not isinstance(additional_identity_inventories, Sequence) or isinstance(
        additional_identity_inventories, (str, bytes)
    ):
        raise ValueError("explicit reviewed identity-inventory receipts required")
    inventory_receipts, inventory_paths = [], set()
    for receipt in additional_identity_inventories:
        if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256", "inventory"}:
            raise ValueError("exact path/hash/inventory receipt required")
        path = _path(receipt["path"], "reviewed identity inventory")
        expected = _hash(receipt["sha256"], "reviewed identity inventory")
        if path in inventory_paths or bindings.get(path) != expected:
            raise ValueError("identity inventory is duplicated or lacks a reviewed source binding")
        inventory_paths.add(path)
        inventory = receipt["inventory"]
        if (
            not isinstance(inventory, dict)
            or SHA.fullmatch(str(inventory.get("source_sha"))) is None
            or any(inventory.get(field, False) is not False for field in FALSE_FIELDS)
        ):
            raise ValueError("reviewed identity inventory producer or scientific scope differs")
        additions = _identities(inventory.get("identities"))
        for name in IDENTITY_FIELDS:
            identities[name].update(additions[name])
        inventory_receipts.append(
            dict(path=path, sha256=expected, projected_inventory_sha256=digest(inventory))
        )
    resolved, unresolved, old_unresolved, new_unresolved = [], [], [], []
    for row in rows:
        owner, ordinal = row["owner_index"], row["historical_error_ordinal"]
        hashed = row["original_error_sha256"]
        old = ordinal < counts[str(owner)]
        # The occurrence boundary is decisive even when a new raw hash and path
        # exactly equal a reviewed old error. No path-based waiver exists.
        if old and (hashed in inherited or hashed in additive):
            resolved.append(
                dict(
                    owner_index=owner,
                    historical_error_ordinal=ordinal,
                    original_error_sha256=hashed,
                    resolution_kind="inherited_reviewed_proof"
                    if hashed in inherited
                    else "V36_additive_original_occurrence_proof",
                    additive_proof_sha256=additive.get(hashed),
                )
            )
        else:
            unresolved.append(copy.deepcopy(row))
            (old_unresolved if old else new_unresolved).append(row)
    normalized_identities = {
        name: [list(row) if name in NESTED_WIDTHS else row for row in sorted(identities[name])]
        for name in IDENTITY_FIELDS
    }
    source_partitions = dict(
        raw_error_count=len(rows),
        original_seed_error_count=len(original_errors),
        newly_observed_error_count=len(rows) - len(original_errors),
        inherited_resolved_original_occurrences=len(inherited),
        newly_resolved_original_occurrences=len(additive),
        resolved_error_count=len(resolved),
        net_unresolved_error_count=len(unresolved),
        unresolved_original_occurrences=len(old_unresolved),
        unresolved_newly_observed_occurrences=len(new_unresolved),
    )
    if len(resolved) + len(unresolved) != len(rows) or len(resolved) != len(inherited) + len(
        additive
    ):
        raise ValueError("raw/resolved/unresolved occurrence partition differs")
    return dict(
        schema="evicastkv_occurrence_safe_identity_reconciliation_v37",
        source_sha=global_inventory["source_sha"],
        source_bindings=copy.deepcopy(bindings),
        input_metadata_sha256s=dict(
            global_inventory=digest(global_inventory),
            original_errors=digest(original_errors),
            seed_error_counts=digest(seed_error_counts),
            inherited_resolution_hashes=digest(sorted(inherited)),
            coverage_overlay=None if coverage_overlay is None else digest(coverage_overlay),
            additional_identity_proofs=digest(list(additional_identity_proofs)),
        ),
        counts=source_partitions,
        raw_error_records=copy.deepcopy(rows),
        resolved_occurrences=resolved,
        unresolved_occurrences=unresolved,
        unresolved_error_categories=dict(Counter(row["error"] for row in unresolved)),
        identities=normalized_identities,
        identity_addition_counts={
            name: len(identities[name]) - initial_identity_counts[name] for name in IDENTITY_FIELDS
        },
        additive_proof_sha256s=sorted(proof_digests),
        inherited_identity_proof_sha256s=sorted(additional_digests),
        inherited_identity_inventory_receipts=inventory_receipts,
        identity_proof_sha256s=sorted(identity_proof_digests),
        raw_input_errors_preserved=True,
        original_owner_ordinals_preserved=True,
        preexisting_identity_exclusions_preserved=True,
        new_duplicate_occurrences_waived=False,
        original_error_hash_does_not_authorize_new_observations=True,
        error_waivers=0,
        files_opened=0,
        scans_run=0,
        GPU_jobs_submitted=0,
        **dict.fromkeys(FALSE_FIELDS, False),
    )
