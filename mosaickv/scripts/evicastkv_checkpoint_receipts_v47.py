#!/usr/bin/env python3
"""Exact receipt sidecars, without changing global exclusions or checkpoint caps.

Only redundant inline receipt payloads move out of the checkpoint. All six
global identity lists, raw errors, proof digests and inherited artifact hashes
remain untouched. Consumers validate the external receipts, not discard them.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path

from mosaickv.evaluation import metadata_recovery_v41 as core
from mosaickv.evaluation.fox_campaign import digest

MAXIMUM_BYTES = 512 * 2**20
INLINE = "metadata_recovery_v41_receipts"
REFERENCES = "external_metadata_receipt_refs"
MANIFEST = "external_metadata_receipt_manifest"
REFERENCE_FIELDS = {
    "ordinal",
    "path",
    "sha256",
    "bytes",
    "semantic_sha256",
    "registration_sha256",
    "contains_identities",
    "identity_counts",
}


def _check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("registered receipt-checkpoint deadline reached")


def _limit(maximum_bytes):
    if type(maximum_bytes) is not int or not 0 < maximum_bytes <= MAXIMUM_BYTES:
        raise ValueError("receipt/checkpoint limit must remain at or below 512 MiB")


def _context(report):
    if any(report.get(name) is not False for name in core.contract.FALSE_FIELDS):
        raise ValueError("receipt externalization cannot widen scientific scope")
    if not isinstance(report.get(INLINE, []), list):
        raise ValueError("explicit inline receipt sequence required")
    known = core.contract._identities(report["identities"])
    bindings = core.contract._bindings(report["source_bindings"])
    proofs = core.contract._hashes(report["additive_proof_sha256s"], "additive receipts")
    identity_proofs = core.contract._hashes(report["identity_proof_sha256s"], "identity receipts")
    return known, bindings, proofs, identity_proofs


def _validate_receipt(receipt, context):
    known, bindings, _proofs, _identity_proofs = context
    if not isinstance(receipt, dict) or set(receipt) not in (
        core.RECEIPT_FIELDS,
        core.RECEIPT_FIELDS | {"identities"},
    ):
        raise ValueError("exact full metadata receipt schema required")
    signature = receipt["signature"]
    if (
        receipt["operation"] not in core.OPERATIONS
        or not isinstance(signature, list)
        or len(signature) != 4
        or any(type(value) is not int or value < 0 for value in signature)
        or any(receipt.get(name) is not False for name in core.contract.FALSE_FIELDS)
        or receipt["old_unbound_historical_version_byte_identity_claimed"] is not False
    ):
        raise ValueError("receipt source signature or scientific scope differs")
    registration = core.contract._hash(receipt["registration_sha256"], "receipt registration")
    core.contract._path(receipt["path"], "receipt source", credential_allowed=True)
    proof = core.contract._path(receipt["proof_path"], "receipt semantic proof")
    proof_sha = core.contract._hash(receipt["proof_sha256"], "receipt semantic proof")
    first = core.contract._hash(receipt["whole_file_sha256_before"], "first source hash")
    second = core.contract._hash(receipt["whole_file_sha256_after"], "second source hash")
    expected_kind = (
        "symlink_target_utf8" if receipt["operation"] == "source_alias" else "file_bytes"
    )
    if (
        first != second
        or receipt["source_hash_kind"] != expected_kind
        or bindings.get(proof) != proof_sha
    ):
        raise ValueError("receipt hashes or existing semantic-proof binding differ")
    counts = dict.fromkeys(core.contract.IDENTITY_FIELDS, 0)
    if "identities" in receipt:
        values = core.contract._identities(receipt["identities"])
        if any(not rows <= known[name] for name, rows in values.items()):
            raise ValueError("only identities already present in global exclusions may move")
        counts = {name: len(receipt["identities"][name]) for name in counts}
    return registration, counts


def _manifest_refs(report):
    refs, manifest = report.get(REFERENCES, []), report.get(MANIFEST)
    if not isinstance(refs, list):
        raise ValueError("ordered external receipt references required")
    if any(not isinstance(ref, dict) or set(ref) != REFERENCE_FIELDS for ref in refs):
        raise ValueError("exact external receipt reference required")
    if manifest is None:
        if refs:
            raise ValueError("external receipt manifest is missing")
        return refs
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "receipt_count", "ordered_semantic_sha256"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or type(manifest["receipt_count"]) is not int
        or manifest["receipt_count"] != len(refs)
        or manifest["ordered_semantic_sha256"]
        != digest([ref.get("semantic_sha256") for ref in refs])
    ):
        raise ValueError("receipt count, schema or ordered semantic digest differs")
    return refs


def _read_reference(ref, ordinal, context, maximum_bytes, deadline):
    _check(deadline)
    if not isinstance(ref, dict) or set(ref) != REFERENCE_FIELDS:
        raise ValueError("exact external receipt reference required")
    if (
        type(ref["ordinal"]) is not int
        or ref["ordinal"] != ordinal
        or type(ref["bytes"]) is not int
        or not 0 < ref["bytes"] <= maximum_bytes
        or type(ref["contains_identities"]) is not bool
        or not isinstance(ref["identity_counts"], dict)
        or set(ref["identity_counts"]) != set(core.contract.IDENTITY_FIELDS)
        or any(type(value) is not int or value < 0 for value in ref["identity_counts"].values())
    ):
        raise ValueError("external receipt order, count or byte bound differs")
    name = core.contract._path(ref["path"], "external receipt")
    expected = core.contract._hash(ref["sha256"], "external receipt")
    semantic = core.contract._hash(ref["semantic_sha256"], "full receipt semantic digest")
    core.contract._hash(ref["registration_sha256"], "external receipt registration")
    known, bindings, proofs, identity_proofs = context
    if bindings.get(name) != expected or semantic not in proofs:
        raise ValueError("external receipt is missing its immutable binding or semantic proof")
    if ref["contains_identities"] and semantic not in identity_proofs:
        raise ValueError("external identity receipt is not a registered identity proof")
    path = Path(name)
    if path.resolve(strict=True) != path.absolute():
        raise ValueError("external receipt path cannot be a symlink alias")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != ref["bytes"]:
        raise ValueError("external receipt file size/type differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    raw, hashed = bytearray(), hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())

        def geometry(info):
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

        if geometry(opened) != geometry(before):
            raise ValueError("external receipt descriptor differs")
        while True:
            _check(deadline)
            part = stream.read(min(2**20, maximum_bytes - len(raw) + 1))
            if not part:
                break
            raw.extend(part)
            if len(raw) > maximum_bytes:
                raise ValueError("bounded full receipt input exceeded")
            hashed.update(part)
        if geometry(os.fstat(stream.fileno())) != geometry(before) or geometry(
            path.lstat()
        ) != geometry(before):
            raise ValueError("external receipt changed during validation")
    if (
        len(raw) != ref["bytes"]
        or hashed.hexdigest() != expected
        or raw[-1:] != b"\n"
        or hashlib.sha256(memoryview(raw)[:-1]).hexdigest() != semantic
    ):
        raise ValueError("external receipt byte/hash/semantic digest differs")

    def fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate external receipt metadata field")
            value[key] = item
        return value

    receipt = json.loads(raw, object_pairs_hook=fields)
    registration, counts = _validate_receipt(receipt, (known, bindings, proofs, identity_proofs))
    if (
        registration != ref["registration_sha256"]
        or counts != ref["identity_counts"]
        or ("identities" in receipt) is not ref["contains_identities"]
    ):
        raise ValueError("external reference differs from its full receipt schema/counts")
    _check(deadline)
    return receipt


def _write_receipt(path, receipt, maximum_bytes, deadline, allowed_semantics):
    temporary = None
    size, semantic = 0, hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            encoder = json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":"))
            for fragment in encoder.iterencode(receipt):
                _check(deadline)
                raw = fragment.encode()
                size += len(raw)
                if size + 1 > maximum_bytes:
                    raise ValueError("bounded full receipt output exceeded")
                stream.write(raw)
                semantic.update(raw)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        content = semantic.copy()
        content.update(b"\n")
        if semantic.hexdigest() not in allowed_semantics:
            raise ValueError("written full receipt semantic digest is not registered")
        os.link(temporary, path)
        return size + 1, content.hexdigest(), semantic.hexdigest()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validated_receipts(report, context, maximum_bytes, deadline):
    seen, semantics = set(), []
    for ordinal, ref in enumerate(_manifest_refs(report)):
        receipt = _read_reference(ref, ordinal, context, maximum_bytes, deadline)
        if ref["path"] in seen:
            raise ValueError("duplicate external receipt artifact")
        seen.add(ref["path"])
        semantics.append(ref["semantic_sha256"])
        yield receipt
    for receipt in report.get(INLINE, []):
        _check(deadline)
        _validate_receipt(receipt, context)
        semantic = digest(receipt)
        if semantic not in context[2] or ("identities" in receipt and semantic not in context[3]):
            raise ValueError("inline full receipt semantic digest is not registered")
        semantics.append(semantic)
        yield receipt
    _complete_receipt_set(report, semantics)


def _complete_receipt_set(report, semantics):
    expected = {
        row["additive_proof_sha256"]
        for row in report.get("resolved_occurrences", [])
        if str(row.get("resolution_kind", "")).startswith("V41_occurrence_specific_")
    }
    if expected and (len(semantics) != len(set(semantics)) or set(semantics) != expected):
        raise ValueError("external receipts do not cover every resolved metadata proof")


def iter_receipts(report, *, maximum_bytes=MAXIMUM_BYTES, deadline=None):
    """Restore the exact ordered receipt sequence in the coordinator, one at a time."""
    _limit(maximum_bytes)
    yield from _validated_receipts(report, _context(report), maximum_bytes, deadline)


def validate_external_receipts(report, *, maximum_bytes=MAXIMUM_BYTES, deadline=None):
    """Verify every binding, full receipt and identity subset without altering it."""
    count = sum(1 for _ in iter_receipts(report, maximum_bytes=maximum_bytes, deadline=deadline))
    return dict(
        receipt_count=count,
        external_receipt_count=len(report.get(REFERENCES, [])),
        inline_receipt_count=len(report.get(INLINE, [])),
        all_global_identities_inline=True,
    )


def compact_report(report, output_root, *, maximum_bytes=MAXIMUM_BYTES, deadline=None):
    """Return an equivalent checkpoint with only redundant receipt payloads external.

    Future global identity growth is NOT hidden: the caller must still use its
    unchanged 512-MiB checkpoint writer and stop if global exclusions alone fail.
    """
    _limit(maximum_bytes)
    _check(deadline)
    context = _context(report)
    root = Path(output_root).resolve(strict=True)
    if not root.is_relative_to(Path("/scratch/djy8hg").resolve(strict=True)) or not root.is_dir():
        raise ValueError("receipt archive must be an existing scratch directory")
    refs = list(_manifest_refs(report))
    seen = set()
    for ordinal, ref in enumerate(refs):
        _read_reference(ref, ordinal, context, maximum_bytes, deadline)
        if ref["path"] in seen:
            raise ValueError("duplicate external receipt artifact")
        seen.add(ref["path"])
    result = dict(report)
    result["source_bindings"] = dict(report["source_bindings"])
    for receipt in report.get(INLINE, []):
        registration, counts = _validate_receipt(receipt, context)
        ordinal = len(refs)
        path = root / f"receipt-{ordinal:05d}-{registration[:12]}.json"
        allowed = context[2] & context[3] if "identities" in receipt else context[2]
        size, file_sha, semantic = _write_receipt(path, receipt, maximum_bytes, deadline, allowed)
        if (
            str(path) in result["source_bindings"]
            and result["source_bindings"][str(path)] != file_sha
        ):
            raise ValueError("immutable external receipt binding collision")
        refs.append(
            dict(
                ordinal=ordinal,
                path=str(path),
                sha256=file_sha,
                bytes=size,
                semantic_sha256=semantic,
                registration_sha256=registration,
                contains_identities="identities" in receipt,
                identity_counts=counts,
            )
        )
        result["source_bindings"][str(path)] = file_sha
    result[INLINE] = []
    _complete_receipt_set(report, [ref["semantic_sha256"] for ref in refs])
    result[REFERENCES] = refs
    result[MANIFEST] = dict(
        schema_version=1,
        receipt_count=len(refs),
        ordered_semantic_sha256=digest([ref["semantic_sha256"] for ref in refs]),
    )
    _check(deadline)
    return result
