#!/usr/bin/env python3
"""Finite producer-bound proofs for two closed graph diagnostic schemas.

No tree walk, Parquet row decoding, historical-source execution or error waiver.
An exact basename only nominates a saved error: the complete provenance chain,
reviewed producer AST, footer schema and stable whole-file SHA must all pass.
Original scan errors remain immutable; this exports a separate coverage proof.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_review_freshness_stream as scanner
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
SHARDS, SECONDS, MAX_FILES = 8, 240, 1500
MAX_ARTIFACT_BYTES, MAX_WORKER_BYTES = 512 * 2**20, 8 * 2**30
PARQUET_ERROR = "Parquet has no safely projectable identity metadata columns"
NAMES = frozenset({"edges.parquet", "raw_marginal_terms.parquet"})
GATES = ("GATE_freshness_frontier_fork.json", "GATE_freshness_empty_parquet_frontier.json")
REFERENCE_PRODUCER = "cd5c3ca625c19c3df095dc58dccc9fa8bfc850f6"
# AST digests reviewed at REFERENCE_PRODUCER; no historical code is imported.
TEMPLATES = {
    "mosaickv/scripts/evicastkv_worker.py::run_worker": (
        "293b7caddc068967db2a7d5b3c70bf00538744ab5739947edf8d09db34806106"
    ),
    "mosaickv/src/mosaickv/evaluation/fox_mechanisms.py::mechanism_audit.plan": (
        "937e2ac2e8383df9491c5cce35378721bfa6a5a7ba364a810ad075948f77dc58"
    ),
    "mosaickv/src/mosaickv/graph/types.py::PooledBlockDescriptor.source_id": (
        "5f18f4502e33a5f7e4f8063fc0936a1deb2711d7b421474844d3827bd87b11b6"
    ),
    "mosaickv/src/mosaickv/graph/types.py::EdgeType": (
        "324ad63837150f28838140309d070b42953693bc47ee7d8dc8b9f66b602e4c38"
    ),
    "mosaickv/src/mosaickv/selection/types.py::BlockUtility": (
        "17772fea156659830c2d378d81143660552bc16e0ffbeb793bd2bb73b9f534ee"
    ),
    "mosaickv/src/mosaickv/selection/types.py::SelectionDecision": (
        "5ca2a742efda0b56218f72c65d91a9d81266f288fddee54f547e79a52bcef2ee"
    ),
    "mosaickv/src/mosaickv/selection/types.py::SelectionReason": (
        "de7aa9924e6acbc0c176e69f47c9780672902e223289a7c0c2dc5baafec01cab"
    ),
    "mosaickv/src/mosaickv/cache_state.py::Modality": (
        "7d2697f5c5b2a13b7a997eab9677591908296d60247fa5bc377846acc5645e5f"
    ),
}
UTILITY = (
    "forecast_attention_probability",
    "value_novelty",
    "expected_attention_output_contribution",
    "graph_centrality",
    "singleton_coverage_gain",
    "modality_rarity",
    "redundancy_penalty",
    "mandatory_priority",
    "local_utility",
)
SCHEMAS = {
    "edges.parquet": {
        "source": {"int64"},
        "target": {"int64"},
        "weight": {"double"},
        "edge_type": {"large_string", "string"},
    },
    "raw_marginal_terms.parquet": {
        "source_id": {"int64"},
        "layer": {"int64"},
        "head": {"int64"},
        "modality": {"large_string", "string"},
        "node_id": {"int64"},
        **{name: {"double"} for name in UTILITY},
        "selection_node_id": {"int64"},
        "selection_selected": {"bool"},
        "selection_selection_rank": {"int64", "double", "null"},
        "selection_marginal_gain": {"double"},
        "selection_marginal_gain_per_cost": {"double"},
        "selection_cost": {"int64"},
        "selection_reason": {"large_string", "string"},
    },
}


def read(path, expected=None):
    return read_json(read_bytes(path, bound=128 * 2**20, expected=expected)[0])


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("finite derived-artifact proof deadline")


def canonical(path, roots):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not any(path.is_relative_to(Path(r)) for r in roots):
        raise ValueError("derived proof input escapes the declared historical roots")
    return path


def metadata(path, roots, deadline, expected=None):
    path = canonical(path, roots)
    check(deadline)
    raw, actual = read_bytes(path, bound=8 * 2**20, expected=expected)
    # Only provenance scalars plus the two exact artifact-map key forms survive.
    # All benchmark subtrees are still discarded lexically before JSON decoding.
    fields = {
        "path",
        "sha256",
        "source_sha",
        "mechanism_audit_path",
        "mechanism_audit_sha256",
        "status",
    }

    def allowed(key):
        return key in fields or (Path(key).is_absolute() and Path(key).name in NAMES)

    with patch.object(scanner, "safe_key", allowed):
        value = scanner.identity_projection(raw, budget=SimpleNamespace(deadline=deadline))
    return value, actual


def ast_digest(raw, symbol):
    node = ast.parse(raw)
    for component in symbol.split("."):
        found = [
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and child.name == component
        ]
        if len(found) != 1:
            raise ValueError("reviewed producer symbol unavailable or ambiguous")
        node = found[0]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def producer_proof(revision, deadline):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("exact recorded producer commit required")
    check(deadline)
    object_type = subprocess.check_output(
        ["git", "cat-file", "-t", revision],
        cwd=SOURCE,
        timeout=max(0.01, min(10, deadline - time.monotonic())),
        stderr=subprocess.DEVNULL,
    )
    if object_type.strip() != b"commit":
        raise ValueError("recorded producer object is not a Git commit")
    blobs, proofs = {}, {}
    for spec, expected in TEMPLATES.items():
        check(deadline)
        path, symbol = spec.split("::")
        if path not in blobs:
            raw = subprocess.check_output(
                ["git", "show", f"{revision}:{path}"],
                cwd=SOURCE,
                timeout=max(0.01, min(10, deadline - time.monotonic())),
                stderr=subprocess.DEVNULL,
            )
            if len(raw) > 2 * 2**20:
                raise ValueError("producer source exceeds reviewed size bound")
            blobs[path] = raw
        if ast_digest(blobs[path], symbol) != expected:
            raise ValueError("recorded producer differs from the reviewed closed writer/schema")
        proofs[spec] = expected
    return dict(
        recorded_producer_commit=revision,
        reviewed_template_commit=REFERENCE_PRODUCER,
        source_blob_sha256s={p: hashlib.sha256(v).hexdigest() for p, v in blobs.items()},
        reviewed_AST_sha256s=proofs,
        historical_source_executed=False,
    )


def hash_file(path, expected, registered_signature, deadline):
    if scanner.signature(path) != registered_signature:
        raise ValueError("artifact signature changed")
    if registered_signature[2] > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds finite per-file bound")
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            check(deadline)
            raw = stream.read(2**20)
            if not raw:
                break
            hashed.update(raw)
    actual = hashed.hexdigest()
    if actual != expected or scanner.signature(path) != registered_signature:
        raise ValueError("bound artifact SHA/signature differs")
    return actual


def schema_proof(path):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    fields = [(field.name, str(field.type)) for field in parquet.schema_arrow]
    expected = SCHEMAS[path.name]
    if (
        len(fields) != len(expected)
        or {name for name, _ in fields} != expected.keys()
        or any(kind not in expected[name] for name, kind in fields)
    ):
        raise ValueError("unexpected closed diagnostic schema; no identity inference permitted")
    if set(parquet.schema_arrow.metadata or {}) - {b"pandas"}:
        raise ValueError("unexpected diagnostic footer metadata")
    return dict(
        fields=fields,
        schema_sha256=digest(fields),
        serialized_schema_sha256=hashlib.sha256(parquet.schema_arrow.serialize()).hexdigest(),
        rows_from_footer=parquet.metadata.num_rows,
        rows_decoded=0,
    )


def prove(row, roots, deadline, source_cache=None):
    path = canonical(row["path"], roots)
    if path.name not in NAMES or scanner.signature(path) != row["signature"]:
        raise ValueError("registered candidate artifact differs")
    # Exact reviewed writer layout, not an ancestor search or directory glob.
    if path.parent.parent.name != "mechanism-terms":
        raise ValueError("artifact is not in the reviewed worker's exact output layout")
    owner, key = path.parent.parent.parent, path.parent.name
    worker_path = canonical(owner / "worker.json", roots)
    record_path = canonical(owner / "records" / f"{key}.json", roots)
    audit_path = canonical(owner / "mechanisms" / f"{key}.json", roots)
    worker, worker_sha = metadata(worker_path, roots, deadline)
    matches = [
        r
        for r in worker.get("records", [])
        if isinstance(r, dict)
        and isinstance(r.get("path"), str)
        and canonical(r["path"], roots) == record_path
    ]
    if len(matches) != 1 or not scanner.HASH.fullmatch(matches[0].get("sha256", "")):
        raise ValueError("worker must bind exactly one immutable result record")
    record, record_sha = metadata(record_path, roots, deadline, matches[0]["sha256"])
    if canonical(
        record.get("mechanism_audit_path", ""), roots
    ) != audit_path or not scanner.HASH.fullmatch(record.get("mechanism_audit_sha256", "")):
        raise ValueError("result record does not bind the expected mechanism audit")
    audit, audit_sha = metadata(audit_path, roots, deadline, record["mechanism_audit_sha256"])
    terms = audit.get("raw_terms", {})
    if not isinstance(terms, dict) or len(terms) != 2:
        raise ValueError("exact two-artifact producer hash map required")
    members = {}
    for name, expected in terms.items():
        target = canonical(name, roots)
        if target.parent != path.parent or target.name not in NAMES or target in members:
            raise ValueError("producer map escapes closed artifact pair or repeats an alias")
        if not isinstance(expected, str) or not scanner.HASH.fullmatch(expected):
            raise ValueError("producer map lacks an exact artifact SHA")
        members[target] = expected
    if {p.name for p in members} != NAMES or path not in members:
        raise ValueError("producer map does not bind the registered diagnostic artifact")
    revision = worker.get("source_sha")
    if record.get("source_sha") != revision:
        raise ValueError("worker and record producer commits differ")
    source_cache = source_cache if source_cache is not None else {}
    if revision not in source_cache:
        source_cache[revision] = producer_proof(revision, deadline)
    before = hash_file(path, members[path], row["signature"], deadline)
    footer = schema_proof(path)
    after = hash_file(path, members[path], row["signature"], deadline)
    # Rebind provenance after the file transaction: no proof escapes mutation.
    for value, expected in (
        (worker_path, worker_sha),
        (record_path, record_sha),
        (audit_path, audit_sha),
    ):
        read_bytes(value, bound=8 * 2**20, expected=expected)
    check(deadline)
    return dict(
        path=str(path),
        original_error_sha256s=row["original_error_sha256s"],
        resolved_as="derived_no_request_or_media_identity",
        reason="closed_internal_graph_metadata_producer_and_footer_verified",
        source_proof=source_cache[revision],
        parquet_footer=footer,
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
        signature=row["signature"],
        bindings={
            str(worker_path): worker_sha,
            str(record_path): record_sha,
            str(audit_path): audit_sha,
        },
        input_independence_or_global_freshness_claimed=False,
    )


def prepare(root, *, cpu_gate_root, parent, parent_gate_filename=None):
    root, parent = Path(root).resolve(), Path(parent).resolve(strict=True)
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch proof root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    available = [name for name in GATES if (parent / name).is_file()]
    gate_name = parent_gate_filename or (available[0] if len(available) == 1 else None)
    if gate_name not in GATES:
        raise ValueError("exact reviewed historical state gate required")
    gate_path, plan_path = parent / gate_name, parent / "plan.json"
    gate, old = read(gate_path), read(plan_path)
    if gate["missing_shards"] or gate["source_sha"] != old["source_sha"]:
        raise ValueError("all source-matched parent states required")
    roots = [str(Path(r).resolve(strict=True)) for r in old["roots"]]
    bound = {str(gate_path): sha(gate_path), str(plan_path): sha(plan_path)}
    grouped, errors = defaultdict(list), []
    indices = []
    for name, expected in gate["worker_bindings"].items():
        worker_path = Path(name).resolve(strict=True)
        match = re.fullmatch(r"shard-(\d+)\.json", worker_path.name)
        if worker_path.parent != parent / "workers" or match is None:
            raise ValueError("only the exact eight historical state artifacts are admissible")
        indices.append(int(match.group(1)))
        saved = read(worker_path, expected)
        if (
            saved["source_sha"] != old["source_sha"]
            or digest(saved["state"]) != saved["state_sha256"]
            or ("index" in saved and saved["index"] != int(match.group(1)))
        ):
            raise ValueError("parent worker state source/digest differs")
        bound[str(worker_path)] = expected
        for error in saved["state"]["errors"]:
            errors.append(error)
            if error.get("error") == PARQUET_ERROR and Path(error["path"]).name in NAMES:
                grouped[error["path"]].append(digest(error))
    if sorted(indices) != list(range(SHARDS)) or len(grouped) > MAX_FILES:
        raise ValueError("incomplete states or finite artifact queue bound exceeded")
    queue, seen = [], set()
    for name, original in sorted(grouped.items()):
        try:
            path = canonical(name, roots)
            if path in seen:
                raise ValueError("duplicate canonical artifact aliases require explicit review")
            seen.add(path)
            registered, initial_error = scanner.signature(path), None
        except (OSError, ValueError) as error:
            path, registered, initial_error = Path(name), None, type(error).__name__
        queue.append(
            dict(
                path=str(path),
                original_error_sha256s=sorted(original),
                signature=registered,
                registration_error=initial_error,
                index=int(hashlib.sha256(str(path).encode()).hexdigest(), 16) % SHARDS,
            )
        )
    write_once(root / "original_errors.json", errors)
    plan = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        parent=str(parent),
        parent_gate_filename=gate_name,
        bindings=bound,
        roots=roots,
        queue=queue,
        shard_count=SHARDS,
        seconds_per_worker=SECONDS,
        maximum_worker_bytes=MAX_WORKER_BYTES,
        reviewed_source_templates=TEMPLATES,
        original_error_count=len(errors),
        original_errors_sha256=sha(root / "original_errors.json"),
        original_errors_preserved=True,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_finite_derived_proof_queue",
        files=len(queue),
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    root = Path(root)
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    if plan["reviewed_source_templates"] != TEMPLATES:
        raise ValueError("registered source proof policy differs")
    for name, expected in plan["bindings"].items():
        read_bytes(name, bound=128 * 2**20, expected=expected)
    read(root / "original_errors.json", plan["original_errors_sha256"])
    return plan


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered one-CPU shard index required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["seconds_per_worker"]
    queue = [row for row in plan["queue"] if row["index"] == index]
    committed, blocked, pending, source_cache, charged = [], [], [], {}, 0
    for number, row in enumerate(queue):
        try:
            check(deadline)
            if row["registration_error"]:
                raise ValueError("registered source unavailable or canonical alias ambiguous")
            size = 2 * row["signature"][2]
            if charged + size > plan["maximum_worker_bytes"]:
                raise TimeoutError("finite worker source-byte limit")
            charged += size
            committed.append(prove(row, plan["roots"], deadline, source_cache))
        except TimeoutError:
            pending.extend(queue[number:])
            break
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RecursionError,
            subprocess.SubprocessError,
        ) as error:
            blocked.append(dict(row=row, reason=type(error).__name__))
    output = dict(
        source_sha=plan["source_sha"],
        index=index,
        committed=committed,
        blocked=blocked,
        pending=pending,
        source_bytes_charged=charged,
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
    )
    target = root / "workers" / f"shard-{index}.json"
    write_once(target, output)
    return dict(
        index=index,
        proved=len(committed),
        blocked=len(blocked),
        pending=len(pending),
        sha256=sha(target),
    )


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    expected = {row["path"]: row for row in plan["queue"]}
    if len(expected) != len(plan["queue"]):
        raise ValueError("ambiguous canonical candidate aliases")
    proofs, blocked, pending, missing, bindings = [], [], [], [], {}
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        saved = read(path)
        if saved["source_sha"] != plan["source_sha"] or saved["index"] != index:
            raise ValueError("proof worker source/index differs")
        classified = (
            [r["path"] for r in saved["committed"]]
            + [r["row"]["path"] for r in saved["blocked"]]
            + [r["path"] for r in saved["pending"]]
        )
        if len(classified) != len(set(classified)) or set(classified) != {
            r["path"] for r in plan["queue"] if r["index"] == index
        }:
            raise ValueError("registered artifact omitted or multiply classified")
        for proof in saved["committed"]:
            row = expected[proof["path"]]
            if (
                proof["original_error_sha256s"] != row["original_error_sha256s"]
                or proof["signature"] != row["signature"]
                or proof["whole_file_sha256_before"] != proof["whole_file_sha256_after"]
                or proof["resolved_as"] != "derived_no_request_or_media_identity"
                or proof["source_proof"]["reviewed_AST_sha256s"] != TEMPLATES
            ):
                raise ValueError("derived proof differs from registered evidence/policy")
            artifact = canonical(proof["path"], plan["roots"])
            if scanner.signature(artifact) != row["signature"]:
                raise ValueError("proved artifact changed before aggregation")
            for name, expected_sha in proof["bindings"].items():
                read_bytes(canonical(name, plan["roots"]), bound=8 * 2**20, expected=expected_sha)
            if digest(schema_proof(artifact)) != digest(proof["parquet_footer"]):
                raise ValueError("proved footer changed before aggregation")
        if any(r["row"] != expected[r["row"]["path"]] for r in saved["blocked"]) or any(
            r != expected[r["path"]] for r in saved["pending"]
        ):
            raise ValueError("unresolved artifact registration changed")
        proofs.extend(saved["committed"])
        blocked.extend(saved["blocked"])
        pending.extend(saved["pending"])
        bindings[str(path)] = sha(path)
    write_once(
        root / "coverage_overlay.json",
        dict(
            proofs=proofs,
            original_errors_preserved=True,
            original_errors_sha256=plan["original_errors_sha256"],
            worker_bindings=bindings,
            fresh_confirmation_authorized=False,
        ),
    )
    report = dict(
        status="derived_proof_queue_complete"
        if not (missing or blocked or pending)
        else "derived_proof_queue_incomplete",
        source_sha=plan["source_sha"],
        candidate_files=len(expected),
        proved_files=len(proofs),
        resolved_original_error_hashes=len(
            {value for proof in proofs for value in proof["original_error_sha256s"]}
        ),
        original_error_count=plan["original_error_count"],
        blocked_files=len(blocked),
        pending_files=len(pending),
        missing_shards=missing,
        worker_bindings=bindings,
        original_errors_preserved=True,
        original_errors_sha256=plan["original_errors_sha256"],
        coverage_overlay_sha256=sha(root / "coverage_overlay.json"),
        error_waivers=0,
        complete_historical_identity_coverage=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "GATE_derived_artifact_proofs.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-gate-filename")
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(
            args.root,
            cpu_gate_root=args.cpu_gate_root,
            parent=args.parent,
            parent_gate_filename=args.parent_gate_filename,
        )
    elif args.mode == "worker":
        result = worker(args.root, args.index)
    else:
        result = analyze(args.root)
    print(json.dumps(result))
