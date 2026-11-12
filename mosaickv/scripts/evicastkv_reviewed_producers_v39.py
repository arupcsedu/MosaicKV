#!/usr/bin/env python3
"""Three exact reviewed graph-diagnostic producers; no schema or error waiver.

The two older workers differ in correctness/optimization orchestration, not
their mechanism-term writer. Their actual complete worker ASTs are retained;
the other seven source constraints and the original provenance/footer/hash
transaction remain unchanged. Calls are process-isolated, not threaded.
"""

from __future__ import annotations

import copy
import re
import threading
from pathlib import PurePosixPath
from unittest.mock import patch

import evicastkv_derived_artifact_proofs as original

from mosaickv.evaluation.fox_campaign import digest

WORKER = "mosaickv/scripts/evicastkv_worker.py::run_worker"
BASE = "cd5c3ca625c19c3df095dc58dccc9fa8bfc850f6"
BASE_TEMPLATES = dict(original.TEMPLATES)
WORKER_VARIANTS = {
    BASE: "293b7caddc068967db2a7d5b3c70bf00538744ab5739947edf8d09db34806106",
    "8985f03e561edfd4919e1f80cd85bd6e45ff3557": (
        "7ee6b0cbfce0fb29baa2d0e548ee96beb6534ebe624a202462d234c373433eb6"
    ),
    "88cee84432b98778e941847e0e2532b534fad64f": (
        "8c9cd65172e1f2cb8ea03a20895e677009164b8570ef411f3c7427790bd373af"
    ),
}
REGISTRY_SHA = digest(
    dict(base_commit=BASE, base_templates=BASE_TEMPLATES, exact_worker_variants=WORKER_VARIANTS)
)
HASH = re.compile(r"[a-f0-9]{64}\Z")
_LOCK = threading.Lock()
_ATTESTED_SOURCES = {}


def reviewed_ast_map(revision):
    if not isinstance(revision, str) or revision not in WORKER_VARIANTS:
        raise ValueError("recorded source is not one of the three reviewed exact commits")
    return BASE_TEMPLATES | {WORKER: WORKER_VARIANTS[revision]}


def _source(proof):
    if not isinstance(proof, dict):
        raise ValueError("actual reviewed producer proof required")
    revision = proof.get("recorded_producer_commit")
    expected = reviewed_ast_map(revision)
    blobs = proof.get("source_blob_sha256s")
    if (
        proof.get("reviewed_template_commit") != revision
        or proof.get("reviewed_AST_sha256s") != expected
        or proof.get("historical_source_executed") is not False
        or not isinstance(blobs, dict)
        or set(blobs) != {key.split("::")[0] for key in expected}
        or any(
            not isinstance(value, str) or HASH.fullmatch(value) is None for value in blobs.values()
        )
    ):
        raise ValueError("recorded producer AST/blob template proof differs")
    return revision


def _registry(revision):
    return dict(
        schema="evicastkv_reviewed_closed_producers_v39",
        registry_sha256=REGISTRY_SHA,
        actual_producer_commit=revision,
        actual_worker_AST_sha256=WORKER_VARIANTS[revision],
        unchanged_other_AST_constraints=7,
        filename_only_exemption=False,
        provenance_schema_hash_guards_unchanged=True,
    )


def validate_proof(proof):
    """Pure analyzer validation; source identity is actual, never relabeled cd5."""
    if not isinstance(proof, dict):
        raise ValueError("complete graph-diagnostic proof dictionary required")
    revision = _source(proof.get("source_proof"))
    registry = proof.get("reviewed_producer_registry")
    if registry != _registry(revision) and not (revision == BASE and registry is None):
        raise ValueError("actual reviewed producer registry binding differs")
    path = PurePosixPath(str(proof.get("path", "")))
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != proof.get("path")
        or path.name not in original.NAMES
        or path.parent.parent.name != "mechanism-terms"
    ):
        raise ValueError("reviewed closed artifact layout required")
    owner, key = path.parent.parent.parent, path.parent.name
    required = {
        str(owner / "worker.json"),
        str(owner / "records" / f"{key}.json"),
        str(owner / "mechanisms" / f"{key}.json"),
    }
    bindings = proof.get("bindings")
    before, after = proof.get("whole_file_sha256_before"), proof.get("whole_file_sha256_after")
    if (
        not isinstance(before, str)
        or HASH.fullmatch(before) is None
        or after != before
        or not isinstance(bindings, dict)
        or set(bindings) != required
        or any(
            not isinstance(value, str) or HASH.fullmatch(value) is None
            for value in bindings.values()
        )
        or proof.get("resolved_as") != "derived_no_request_or_media_identity"
        or proof.get("reason") != "closed_internal_graph_metadata_producer_and_footer_verified"
        or proof.get("input_independence_or_global_freshness_claimed") is not False
    ):
        raise ValueError("closed provenance/whole-file transaction differs")
    hashes, signature = proof.get("original_error_sha256s"), proof.get("signature")
    if (
        not isinstance(hashes, list)
        or not hashes
        or any(not isinstance(value, str) or HASH.fullmatch(value) is None for value in hashes)
        or len(hashes) != len(set(hashes))
        or not isinstance(signature, (tuple, list))
        or len(signature) != 4
        or any(type(value) is not int or value < 0 for value in signature)
    ):
        raise ValueError("original error/signature binding differs")
    footer = proof.get("parquet_footer")
    if not isinstance(footer, dict) or not isinstance(footer.get("fields"), list):
        raise ValueError("exact closed footer proof required")
    fields, schema = footer["fields"], original.SCHEMAS[path.name]
    if (
        any(
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(not isinstance(item, str) for item in value)
            for value in fields
        )
        or len(fields) != len(schema)
        or {name for name, _kind in fields} != set(schema)
        or any(kind not in schema[name] for name, kind in fields)
        or footer.get("schema_sha256") != digest(fields)
        or not isinstance(footer.get("serialized_schema_sha256"), str)
        or HASH.fullmatch(footer["serialized_schema_sha256"]) is None
        or type(footer.get("rows_decoded")) is not int
        or footer["rows_decoded"] != 0
        or type(footer.get("rows_from_footer")) is not int
        or footer["rows_from_footer"] < 0
    ):
        raise ValueError("closed footer schema or zero-row-decoding guard differs")
    return True


def prove(row, roots, deadline, cache=None):
    """Run the unchanged transaction after a metadata-bound exact-source lookup."""
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("reviewed producer dispatch requires process-isolated execution")
    try:
        path = original.canonical(row["path"], roots)
        if path.name not in original.NAMES or path.parent.parent.name != "mechanism-terms":
            raise ValueError("exact original closed-writer layout required")
        owner = path.parent.parent.parent
        worker, _worker_sha = original.metadata(owner / "worker.json", roots, deadline)
        revision = worker.get("source_sha")
        templates = reviewed_ast_map(revision)
        with (
            patch.object(original, "TEMPLATES", templates),
            patch.object(original, "REFERENCE_PRODUCER", revision),
        ):
            if revision not in _ATTESTED_SOURCES:
                actual = original.producer_proof(revision, deadline)
                if _source(actual) != revision:
                    raise ValueError("actual source proof differs from selected recorded producer")
                _ATTESTED_SOURCES[revision] = copy.deepcopy(actual)
            actual = _ATTESTED_SOURCES[revision]
            cache = {} if cache is None else cache
            if not isinstance(cache, dict):
                raise ValueError("explicit process-local producer cache dictionary required")
            if revision in cache and cache[revision] != actual:
                raise ValueError(
                    "caller source cache conflicts with independently attested producer"
                )
            cache[revision] = copy.deepcopy(actual)
            proof = original.prove(row, roots, deadline, cache)
        if proof.get("source_proof") != actual:
            raise ValueError("transaction producer changed after source selection")
        proof["reviewed_producer_registry"] = _registry(revision)
        validate_proof(proof)
        return proof
    finally:
        _LOCK.release()
