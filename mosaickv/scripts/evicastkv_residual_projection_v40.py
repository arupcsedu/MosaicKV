#!/usr/bin/env python3
"""Narrow V40 per-occurrence metadata proofs; no global freshness authority.

YAML uses the unchanged strict event parser. Only the exactly reviewed new
V6 graph producer is added to the unchanged closed-writer transaction. The
caller registers owner/ordinal/error bindings and enforces wave resource caps.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import evicastkv_coverage_resolution_v36 as common
import evicastkv_freshness_yaml_events as yaml_events
import evicastkv_native_v4_provenance_recovery as native
import evicastkv_reviewed_producers_v39 as reviewed

from mosaickv.evaluation.fox_campaign import digest

PRODUCER = "7cf31bf285ea7b5f4ae7d7caa615921f2e89f417"
VARIANTS = reviewed.WORKER_VARIANTS | {PRODUCER: reviewed.BASE_TEMPLATES[reviewed.WORKER]}
REGISTRY_SHA = digest(
    dict(
        base_commit=reviewed.BASE,
        base_templates=reviewed.BASE_TEMPLATES,
        exact_worker_variants=VARIANTS,
    )
)
_LOCK = threading.Lock()
_WORKERS = {}


@contextmanager
def _context():
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("V40 proof dispatch requires process-isolated execution")
    try:
        with (
            patch.object(reviewed, "WORKER_VARIANTS", VARIANTS),
            patch.object(reviewed, "REGISTRY_SHA", REGISTRY_SHA),
            patch.object(native, "PRODUCER", PRODUCER),
        ):
            yield
    finally:
        _LOCK.release()


def _path(row, roots):
    kind = row.get("kind")
    if kind not in {"yaml_events", "closed_parquet"}:
        raise ValueError("only registered YAML events and new V6 closed terms are permitted")
    path = common.canonical(row["path"], roots)
    if common.scanner.signature(path) != row["signature"]:
        raise ValueError("registered source signature changed")
    if kind == "yaml_events" and path.suffix not in {".yml", ".yaml"}:
        raise ValueError("actual YAML suffix required")
    if kind == "closed_parquet" and (
        path.name not in reviewed.original.NAMES or path.parent.parent.name != "mechanism-terms"
    ):
        raise ValueError("exact original closed-writer layout required")
    return path


def _expected(row):
    first, second = row.get("file_sha256"), row.get("current_file_sha256")
    if first is not None and second is not None and first != second:
        raise ValueError("conflicting registered current-file hashes")
    result = first if first is not None else second
    if not isinstance(result, str) or reviewed.HASH.fullmatch(result) is None:
        raise ValueError("registered exact current-file SHA256 required")
    return result


def expected_file_sha(row, roots, deadline):
    """Hash small YAML, or bind Parquet hash from its actual provenance chain.

    No Parquet body is read. The unchanged worker/record/audit/raw-terms chain
    checks the exact new producer; its eight AST constraints run in prove().
    """
    reviewed.original.check(deadline)
    path = _path(row, roots)
    if row["kind"] == "yaml_events":
        _raw, result = common.read_bytes(path, bound=yaml_events.MAX_BYTES)
    else:
        with _context():
            result = native.register_pair(path, roots, deadline, _WORKERS, {})
    if common.scanner.signature(path) != row["signature"]:
        raise ValueError("source changed across metadata registration")
    if (
        row.get("file_sha256") is not None or row.get("current_file_sha256") is not None
    ) and result != _expected(row):
        raise ValueError("current metadata differs from an already registered hash")
    reviewed.original.check(deadline)
    return result


def _tag(kind):
    return dict(
        schema="evicastkv_residual_projection_v40",
        kind=kind,
        additional_exact_producer=PRODUCER if kind == "closed_parquet" else None,
        reviewed_four_producer_registry_sha256=REGISTRY_SHA if kind == "closed_parquet" else None,
        original_parser_and_closed_writer_unchanged=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )


def prove(row, roots, deadline, cache=None):
    path = _path(row, roots)
    expected = _expected(row)
    if row["kind"] == "yaml_events":
        proof = yaml_events.project_file(dict(row, path=str(path)), deadline)
        proof["resolved_as"] = "unchanged_YAML_event_metadata_projected"
    else:
        with _context():
            worker, _ = reviewed.original.metadata(
                path.parent.parent.parent / "worker.json", roots, deadline
            )
            if worker.get("source_sha") != PRODUCER:
                raise ValueError("V40 covers only the exactly registered new V6 producer")
            proof = reviewed.prove(row, roots, deadline, cache)
    if (
        proof["whole_file_sha256_before"] != expected
        or proof["whole_file_sha256_after"] != expected
    ):
        raise ValueError("actual proof bytes differ from the registered current source")
    proof = dict(
        proof,
        kind=row["kind"],
        registration_sha256=digest(row),
        registered_current_file_sha256=expected,
        residual_projection=_tag(row["kind"]),
    )
    validate_proof(proof)
    return proof


def validate_proof(proof):
    kind = proof.get("kind")
    if kind not in {"yaml_events", "closed_parquet"} or proof.get("residual_projection") != _tag(
        kind
    ):
        raise ValueError("exact V40 projection scope required")
    expected = proof.get("registered_current_file_sha256")
    if (
        not isinstance(expected, str)
        or reviewed.HASH.fullmatch(expected) is None
        or proof.get("whole_file_sha256_before") != expected
        or proof.get("whole_file_sha256_after") != expected
        or not isinstance(proof.get("registration_sha256"), str)
        or reviewed.HASH.fullmatch(proof["registration_sha256"]) is None
    ):
        raise ValueError("registered current-source transaction differs")
    if kind == "closed_parquet":
        if proof.get("source_proof", {}).get("recorded_producer_commit") != PRODUCER:
            raise ValueError("the actual new V6 producer must remain explicit")
        with _context():
            reviewed.validate_proof(proof)
    else:
        if (
            Path(proof.get("path", "")).suffix not in {".yaml", ".yml"}
            or proof.get("resolved_as") != "unchanged_YAML_event_metadata_projected"
            or set(proof.get("identities", {}))
            != set(yaml_events.IDENTITIES) | {"projected_records"}
            or proof.get("old_unbound_historical_version_byte_identity_claimed") is not False
            or not isinstance(proof.get("projected_metadata_sha256"), str)
            or reviewed.HASH.fullmatch(proof["projected_metadata_sha256"]) is None
        ):
            raise ValueError("unchanged strict YAML projection proof required")
        identities = proof["identities"]
        signature = proof.get("signature")
        if (
            any(not isinstance(identities[name], list) for name in yaml_events.IDENTITIES)
            or type(identities["projected_records"]) is not int
            or identities["projected_records"] < 0
            or not isinstance(signature, (list, tuple))
            or len(signature) != 4
            or any(type(value) is not int or value < 0 for value in signature)
            or signature[2] > yaml_events.MAX_BYTES
        ):
            raise ValueError("registered YAML identity/signature structure differs")
        hashes = proof.get("original_error_sha256s")
        if (
            not isinstance(hashes, list)
            or not hashes
            or any(
                not isinstance(value, str) or reviewed.HASH.fullmatch(value) is None
                for value in hashes
            )
            or len(set(hashes)) != len(hashes)
        ):
            raise ValueError("exact original-error proof bindings required")
    return True
