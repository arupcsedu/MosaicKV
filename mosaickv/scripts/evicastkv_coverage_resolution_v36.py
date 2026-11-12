#!/usr/bin/env python3
"""Finite additive metadata recovery of the verified 10,015-error V18 remainder.

Unchanged YAML-event and producer/schema proofs are reused. A .jsonl suffix
mismatch is resolved only when the entire bounded file is one strict JSON
document. No line is skipped, no auth body is read, and no freshness is granted.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import evicastkv_derived_artifact_proofs as derived
import evicastkv_freshness_yaml_events as yaml_events
import evicastkv_provenance_import as imported
import evicastkv_review_freshness_stream as scanner
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OUTPUT_BASE = Path("/scratch/djy8hg/workdir")
SHARDS, SECONDS, MAX_WORKER_BYTES = 8, 210, 2 * 2**30
SMALL_JSON = 32768
GATE = "GATE_coverage_resolution_v36.json"
JSONL_ERROR = "malformed historical JSONL: ValueError"
KINDS = {
    yaml_events.YAML_ERROR: "yaml_events",
    derived.PARQUET_ERROR: "closed_parquet",
    JSONL_ERROR: "single_json_document",
}
EXPECTED_CATEGORIES = {yaml_events.YAML_ERROR: 6335, derived.PARQUET_ERROR: 2988, JSONL_ERROR: 45}
FALSE_FIELDS = (
    "globally_fresh",
    "historical_media_exposure_resolved",
    "source_group_disjointness_verified",
    "fresh_confirmation_authorized",
    "materialization_authorized",
    "paper_campaign_authorized",
)


def false_scope():
    return dict.fromkeys(FALSE_FIELDS, False)


def read(path, expected=None):
    return read_json(read_bytes(Path(path), bound=128 * 2**20, expected=expected)[0])


def baseline():
    """Verify the completed importer; never repeat its historical proof work."""
    verified = imported.validate_saved(imported.PARENT)
    plan = read_json(verified["raws"]["plan.json"])
    overlay = read_json(verified["raws"]["coverage_overlay.json"])
    covered = set(plan["covered_error_hashes"])
    covered.update(h for proof in overlay["new_proofs"] for h in proof["original_error_sha256s"])
    errors_path = Path(plan["original_errors_path"])
    errors = read(errors_path, plan["bindings"][str(errors_path)])
    by_hash = {digest(error): error for error in errors}
    if (
        len(by_hash) != 39861
        or not covered <= by_hash.keys()
        or len(by_hash.keys() - covered) != 10015
    ):
        raise ValueError("verified original/net V18 error denominators differ")
    bindings = {
        str(imported.PARENT / name): value for name, value in imported.EXPECTED_HASHES.items()
    }
    bindings[str(errors_path)] = sha(errors_path)
    return dict(
        original_errors_path=str(errors_path),
        original_error_count=len(errors),
        covered_error_hashes=sorted(covered),
        bindings=bindings,
    ), by_hash


def canonical(path, roots):
    path = Path(path)
    # Check the lexical path BEFORE resolving it or opening anything.
    if ".codex" in path.parts or path.name.lower() in {"auth.json", "credentials.json"}:
        raise ValueError("credential paths are outside metadata recovery scope")
    target = path.resolve(strict=True)
    if not target.is_file() or not any(target.is_relative_to(Path(root)) for root in roots):
        raise ValueError("metadata path escapes registered historical roots")
    if ".codex" in target.parts or target.name.lower() in {"auth.json", "credentials.json"}:
        raise ValueError("credential alias is outside metadata recovery scope")
    return target


def nominate(errors, covered):
    groups = defaultdict(list)
    for error_hash, error in errors.items():
        if error_hash not in covered and error.get("error") in KINDS:
            groups[(error["path"], KINDS[error["error"]])].append(error_hash)
    return groups


def prepare(root, *, cpu_gate_root=None, global_inventory=None, global_inventory_sha256=None):
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(OUTPUT_BASE.resolve()):
        raise ValueError("new write-once scratch resolution root required")
    source = clean_source(SOURCE)
    cpu_gate_root = Path(cpu_gate_root or root.parent).resolve()
    cpu_gate(cpu_gate_root, source)
    saved, errors = baseline()
    covered = set(saved["covered_error_hashes"])
    categories = Counter(e["error"] for h, e in errors.items() if h not in covered)
    if any(categories[name] != expected for name, expected in EXPECTED_CATEGORIES.items()):
        raise ValueError("registered remaining-category denominators differ")
    roots = [str(Path(path).resolve(strict=True)) for path in scanner.ROOTS]
    groups, registrations, rejected = nominate(errors, covered), [], []
    deadline = time.monotonic() + SECONDS
    for (name, kind), hashes in sorted(groups.items()):
        row = dict(path=name, kind=kind, original_error_sha256s=sorted(hashes))
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("finite registration deadline")
            path = canonical(name, roots)
            row.update(canonical_path=str(path), signature=scanner.signature(path))
            if kind == "closed_parquet":
                # No arbitrary Parquet body hashing/decoding. The unchanged proof
                # may nominate ONLY its exact two reviewed writer layouts.
                if path.name not in derived.NAMES or path.parent.parent.name != "mechanism-terms":
                    raise ValueError("no existing closed-writer proof for this Parquet layout")
                row["current_file_sha256"] = None  # Producer map is bound by worker proof.
            else:
                maximum = yaml_events.MAX_BYTES if kind == "yaml_events" else SMALL_JSON
                _raw, current = read_bytes(path, bound=maximum)
                if scanner.signature(path) != row["signature"]:
                    raise ValueError("metadata changed during registration")
                row["current_file_sha256"] = current
            registrations.append(row)
        except (OSError, ValueError, TimeoutError) as error:
            rejected.append(dict(row=row, reason=type(error).__name__, proof_not_available=True))
    # Greedy byte balancing is deterministic; each file is registered once.
    costs = [0] * SHARDS
    queue = []
    for row in sorted(registrations, key=lambda r: (-r["signature"][2], r["path"])):
        index = min(range(SHARDS), key=lambda value: (costs[value], value))
        costs[index] += row["signature"][2]
        queue.append(dict(row, index=index))
    if any(2 * count > MAX_WORKER_BYTES for count in costs):
        raise ValueError("registered two-pass source bytes exceed per-worker bound")
    for module in (
        Path(__file__),
        Path(yaml_events.__file__),
        Path(derived.__file__),
        Path(scanner.__file__),
        Path(imported.__file__),
    ):
        saved["bindings"][str(module.resolve())] = sha(module)
    if (global_inventory is None) != (global_inventory_sha256 is None):
        raise ValueError("global inventory path/hash must be supplied together")
    if global_inventory is not None:
        path = Path(global_inventory).resolve(strict=True)
        if not path.is_relative_to(OUTPUT_BASE.resolve()):
            raise ValueError("global inventory must remain in project scratch")
        read_bytes(path, bound=128 * 2**20, expected=global_inventory_sha256)
        saved["bindings"][str(path)] = global_inventory_sha256
    plan = dict(
        **saved,
        source_sha=source,
        cpu_gate_root=str(cpu_gate_root),
        roots=roots,
        queue=queue,
        registration_blocked=rejected,
        nominated_files=len(groups),
        expected_category_errors=EXPECTED_CATEGORIES,
        shard_count=SHARDS,
        seconds_per_worker=SECONDS,
        maximum_bytes_per_worker=MAX_WORKER_BYTES,
        worker_source_bytes=costs,
        original_errors_preserved=True,
        error_waivers=0,
        **false_scope(),
    )
    write_once(root / "plan.json", plan)
    return dict(
        workers=SHARDS,
        registered_files=len(queue),
        blocked_files=len(rejected),
        plan_sha256=sha(root / "plan.json"),
        **false_scope(),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    if plan["shard_count"] != SHARDS or plan["seconds_per_worker"] != SECONDS:
        raise ValueError("registered CPU bounds differ")
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("registered parent, code, or inventory binding changed")
    return plan


def project_single_json(row, deadline):
    """A complete JSON document is not a forgiving JSONL line parser."""
    path = Path(row["canonical_path"])
    if scanner.signature(path) != row["signature"]:
        raise ValueError("registered JSON document signature changed")
    raw, before = read_bytes(path, bound=SMALL_JSON, expected=row["current_file_sha256"])
    # Existing strict lexical parser rejects duplicate keys/truncation/nonfinite
    # identity scalars, and redacts benchmark subtrees before decoding them.
    from types import SimpleNamespace

    value = scanner.identity_projection(raw, budget=SimpleNamespace(deadline=deadline))
    if not isinstance(value, dict):
        raise ValueError("single controller JSON must be an object")
    identities = {key: [] for key in yaml_events.IDENTITIES} | {"projected_records": 0}
    scanner.collect(value, identities, path)
    _raw, after = read_bytes(path, bound=SMALL_JSON, expected=before)
    if scanner.signature(path) != row["signature"] or time.monotonic() >= deadline:
        raise ValueError("JSON document changed or deadline exceeded")
    return dict(
        path=row["path"],
        original_error_sha256s=row["original_error_sha256s"],
        signature=row["signature"],
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
        projected_metadata_sha256=digest(value),
        identities=identities,
        suffix_format_mismatch_preserved=True,
        resolved_as="complete_single_JSON_document_metadata_projected",
        malformed_lines_skipped=0,
    )


def resolve(row, roots, deadline, cache):
    path = canonical(row["path"], roots)
    if str(path) != row["canonical_path"] or scanner.signature(path) != row["signature"]:
        raise ValueError("registered canonical source/signature changed")
    if row["kind"] == "yaml_events":
        proof = yaml_events.project_file(dict(row, path=str(path)), deadline)
        proof["path"] = row["path"]
        proof["resolved_as"] = "unchanged_YAML_event_metadata_projected"
    elif row["kind"] == "closed_parquet":
        proof = derived.prove(dict(row, path=str(path)), roots, deadline, cache)
        proof["path"] = row["path"]
    elif row["kind"] == "single_json_document":
        proof = project_single_json(row, deadline)
    else:
        raise ValueError("unregistered proof kind")
    current = row["current_file_sha256"]
    if proof["whole_file_sha256_before"] != proof["whole_file_sha256_after"] or (
        current is not None and proof["whole_file_sha256_before"] != current
    ):
        raise ValueError("proof differs from registered current source bytes")
    return dict(proof, kind=row["kind"], registration_sha256=digest(row))


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered one-CPU worker index required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline, used, cache = time.monotonic() + SECONDS, 0, {}
    rows = [row for row in plan["queue"] if row["index"] == index]
    committed, pending, blocked = [], [], []
    for number, row in enumerate(rows):
        if time.monotonic() >= deadline or used + 2 * row["signature"][2] > MAX_WORKER_BYTES:
            pending.extend(rows[number:])
            break
        used += 2 * row["signature"][2]
        try:
            committed.append(resolve(row, plan["roots"], deadline, cache))
        except TimeoutError:
            pending.extend(rows[number:])
            break
        except (OSError, ValueError, RecursionError, UnicodeError) as error:
            blocked.append(dict(row=row, reason=getattr(error, "reason", type(error).__name__)))
    output = dict(
        source_sha=plan["source_sha"],
        index=index,
        plan_sha256=sha(root / "plan.json"),
        committed=committed,
        pending=pending,
        blocked=blocked,
        two_pass_source_bytes_budgeted=used,
        original_errors_preserved=True,
        error_waivers=0,
        **false_scope(),
    )
    path = root / "workers" / f"shard-{index}.json"
    write_once(path, output)
    return dict(
        index=index,
        committed=len(committed),
        blocked=len(blocked),
        pending=len(pending),
        worker_sha256=sha(path),
    )


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    by_path = {row["path"]: row for row in plan["queue"]}
    proofs, pending, blocked = [], [], list(plan["registration_blocked"])
    missing, workers = [], {}
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        value = read(path)
        if (
            value["source_sha"] != plan["source_sha"]
            or value["index"] != index
            or value["plan_sha256"] != sha(root / "plan.json")
            or value["two_pass_source_bytes_budgeted"] > MAX_WORKER_BYTES
            or any(value[field] for field in FALSE_FIELDS)
        ):
            raise ValueError("worker plan/source/bounds/scope differs")
        names = [item["path"] for item in value["committed"] + value["pending"]]
        names += [item["row"]["path"] for item in value["blocked"]]
        expected = {row["path"] for row in plan["queue"] if row["index"] == index}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("registered denominator dropped or duplicated")
        for proof in value["committed"]:
            row = by_path[proof["path"]]
            if (
                proof["registration_sha256"] != digest(row)
                or proof["original_error_sha256s"] != row["original_error_sha256s"]
                or proof["kind"] != row["kind"]
                or proof["signature"] != row["signature"]
                or proof["whole_file_sha256_before"] != proof["whole_file_sha256_after"]
                or (
                    row["current_file_sha256"] is not None
                    and proof["whole_file_sha256_before"] != row["current_file_sha256"]
                )
            ):
                raise ValueError("proof registration/source bytes differ")
            if row["kind"] == "closed_parquet":
                if (
                    proof["source_proof"]["reviewed_AST_sha256s"] != derived.TEMPLATES
                    or proof["parquet_footer"]["rows_decoded"] != 0
                ):
                    raise ValueError("closed producer proof differs")
            else:
                if set(proof["identities"]) != set(yaml_events.IDENTITIES) | {"projected_records"}:
                    raise ValueError("unexpected projected identity fields")
            proofs.append(proof)
        for row in value["pending"] + [item["row"] for item in value["blocked"]]:
            if row != by_path[row["path"]]:
                raise ValueError("unresolved registration changed")
        pending.extend(value["pending"])
        blocked.extend(value["blocked"])
        workers[str(path)] = sha(path)
    missing_files = sum(row["index"] in missing for row in plan["queue"])
    if len(proofs) + len(pending) + len(blocked) + missing_files != plan["nominated_files"]:
        raise ValueError("nominated denominator not conserved")
    original = read(plan["original_errors_path"], plan["bindings"][plan["original_errors_path"]])
    original_hashes = {digest(error) for error in original}
    before = set(plan["covered_error_hashes"])
    new = [h for proof in proofs for h in proof["original_error_sha256s"]]
    if len(new) != len(set(new)) or set(new) & before or not set(new) <= original_hashes:
        raise ValueError("proof coverage overlaps or escapes original errors")
    remaining = [error for error in original if digest(error) not in before | set(new)]
    report = dict(
        original_error_count=len(original),
        unique_covered_before=len(before),
        net_unresolved_before=len(original_hashes - before),
        newly_covered_error_hashes=len(new),
        net_unresolved_after=len(remaining),
        net_unresolved_categories=dict(Counter(error["error"] for error in remaining)),
        current_duplicate_observations_waived=False,
        original_errors_preserved=True,
        error_waivers=0,
        external_media_source_groups_resolved=False,
        **false_scope(),
    )
    overlay = dict(
        source_sha=plan["source_sha"],
        parent_bindings=plan["bindings"],
        covered_before_error_hashes=sorted(before),
        new_proofs=proofs,
        original_errors_path=plan["original_errors_path"],
        original_errors_preserved=True,
        **false_scope(),
    )
    write_once(root / "coverage_overlay.json", overlay)
    write_once(root / "net_error_report.json", report)
    gate = dict(
        **report,
        status="bounded_additive_metadata_recovery_complete"
        if not (missing or pending or blocked)
        else "bounded_additive_metadata_recovery_partial",
        source_sha=plan["source_sha"],
        registered_files=len(plan["queue"]),
        nominated_files=plan["nominated_files"],
        proved_files=len(proofs),
        blocked_files=len(blocked),
        pending_files=len(pending),
        missing_files=missing_files,
        missing_shards=missing,
        worker_bindings=workers,
        coverage_overlay_sha256=sha(root / "coverage_overlay.json"),
        net_error_report_sha256=sha(root / "net_error_report.json"),
    )
    write_once(root / GATE, gate)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--global-inventory", type=Path)
    parser.add_argument("--global-inventory-sha256")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    result = (
        prepare(
            args.root,
            cpu_gate_root=args.cpu_gate_root,
            global_inventory=args.global_inventory,
            global_inventory_sha256=args.global_inventory_sha256,
        )
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    import json

    print(json.dumps(result, sort_keys=True, allow_nan=False))
