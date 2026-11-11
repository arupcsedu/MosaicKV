#!/usr/bin/env python3
"""Finite producer-bound recovery of 678 native-v4 graph diagnostics.

This reuses the unchanged closed-schema validator, not an identity-free
filename exemption. Parquet bodies are streamed only for complete SHA checks;
no rows, images, benchmark answers, or model outputs are decoded. Original
negative history and all successful older proofs remain immutable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import evicastkv_derived_artifact_proofs as derived
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
HISTORY = Path("/scratch/djy8hg/workdir/fox-review-v16/run-20261111-2/history")
PRIOR = Path("/scratch/djy8hg/workdir/fox-review-v15/run-20261111-1/derived-2")
YAML = Path("/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1/yaml")
OVERSIZE = Path("/scratch/djy8hg/workdir/fox-review-v17/run-20261111-1/oversize")
RECIPE = Path("/scratch/djy8hg/workdir/fox-review-v11/run-20261110-1/freshness")
NATIVE_ROOT = Path("/scratch/djy8hg/runs/evicastkv/fox-review-v4-native-coco-20261109-mechanisms")
OUTPUT_BASE = Path("/scratch/djy8hg")
HISTORY_SOURCE = "84900a6a402e97c7d58499a722c159cfc86e46c7"
PRIOR_SOURCE = "279c4f6336bd852d2ccd8408d519e1ddfa2a53b1"
YAML_SOURCE = "76d4ced1a10c6746bad73f564e84210522355ef9"
PRODUCER = "cd5c3ca625c19c3df095dc58dccc9fa8bfc850f6"
EXPECTED_PRIOR, EXPECTED_FILES, EXPECTED_OWNERS = 2170, 678, 8
SHARDS, SECONDS = 2, 780
MAX_SOURCE_BYTES = 3 * 2**30
GATE = "GATE_native_v4_provenance_recovery.json"


def read(path, expected=None):
    return read_json(read_bytes(Path(path), bound=128 * 2**20, expected=expected)[0])


def bound_read(path, bindings, expected=None):
    path = Path(path).resolve(strict=True)
    value = read(path, expected)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError("parent metadata changed after stable read")
    if bindings.setdefault(str(path), actual) != actual:
        raise ValueError("conflicting canonical parent aliases")
    return value


def historical_evidence():
    """Finite saved error/proof metadata only; no historical result-body reads."""
    bindings = {}
    hg = bound_read(HISTORY / "GATE_history_closure_v16.json", bindings)
    if (
        hg["source_sha"] != HISTORY_SOURCE
        or hg["last_segment"] != 4
        or hg["stop_reason"] != "maximum_four_segments"
        or hg["missing_shards"]
        or hg["fresh_confirmation_authorized"]
        or not hg["original_errors_preserved"]
    ):
        raise ValueError("actual immutable final four-segment history required")
    last = (HISTORY / "segment-4").resolve(strict=True)
    decision = bound_read(last / "decision.json", bindings, hg["last_decision_sha256"])
    if decision["source_sha"] != HISTORY_SOURCE:
        raise ValueError("final history decision producer differs")
    errors_path = last / "preserved_errors.json"
    errors = bound_read(errors_path, bindings, decision["errors_sha256"])
    errors_by_hash = {digest(error): error for error in errors}
    if len(errors_by_hash) != len(errors):
        raise ValueError("original error hashes duplicated")
    prior_gate = bound_read(PRIOR / "GATE_derived_artifact_proofs.json", bindings)
    old = bound_read(
        PRIOR / "coverage_overlay.json", bindings, prior_gate["coverage_overlay_sha256"]
    )
    imported = old["proofs"]
    if (
        len(imported) != EXPECTED_PRIOR
        or prior_gate["source_sha"] != PRIOR_SOURCE
        or prior_gate["proved_files"] != EXPECTED_PRIOR
        or prior_gate["missing_shards"]
        or prior_gate["blocked_files"]
        or prior_gate["unattempted_files"]
        or prior_gate["pending_files"]
    ):
        raise ValueError("complete immutable prior proof receipt required")
    covered = set()
    for proof in imported:
        if (
            proof["source_proof"]["reviewed_AST_sha256s"] != derived.TEMPLATES
            or proof["whole_file_sha256_before"] != proof["whole_file_sha256_after"]
            or proof["resolved_as"] != "derived_no_request_or_media_identity"
        ):
            raise ValueError("prior closed-schema proof differs")
        covered.update(proof["original_error_sha256s"])
    if len(covered) != EXPECTED_PRIOR:
        raise ValueError("prior error proof partition differs")
    yaml_gate = bound_read(YAML / "GATE_freshness_yaml_events.json", bindings)
    if (
        yaml_gate["missing_shards"]
        or yaml_gate["source_sha"] != YAML_SOURCE
        or len(yaml_gate["worker_bindings"]) != 8
    ):
        raise ValueError("all original YAML projection receipts required")
    yaml_hashes = set()
    for name, expected in yaml_gate["worker_bindings"].items():
        worker = bound_read(name, bindings, expected)
        if worker["source_sha"] != YAML_SOURCE:
            raise ValueError("YAML receipt producer differs")
        for proof in worker["committed"]:
            if proof["whole_file_sha256_before"] != proof["whole_file_sha256_after"]:
                raise ValueError("YAML projection transaction differs")
            yaml_hashes.update(proof["original_error_sha256s"])
    covered.update(yaml_hashes)
    if len(yaml_hashes) != yaml_gate["projected_files"]:
        raise ValueError("YAML proof denominator differs")
    over_gate = bound_read(OVERSIZE / "GATE_pending_oversize_recovery.json", bindings)
    over = bound_read(
        OVERSIZE / "coverage_overlay.json", bindings, over_gate["coverage_overlay_sha256"]
    )
    for proof in over["imported_proofs"] + over["new_proofs"]:
        if proof["whole_file_sha256_before"] != proof["whole_file_sha256_after"]:
            raise ValueError("oversize projection transaction differs")
        covered.add(proof["original_error_sha256"])
    recipe_gate = bound_read(RECIPE / "GATE_freshness_recipe_recovery.json", bindings)
    recipes = bound_read(
        RECIPE / "indexed_coverage_overlay.json", bindings, recipe_gate["coverage_overlay_sha256"]
    )
    recipe_hashes = {p["original_error_sha256"] for p in recipes}
    if not recipe_hashes <= yaml_hashes or not covered <= errors_by_hash.keys():
        raise ValueError("historical proof scope or recipe/YAML subset differs")
    return dict(
        bindings=bindings,
        imported_proofs=imported,
        covered_error_hashes=sorted(covered),
        original_errors_path=str(errors_path),
        original_error_count=len(errors),
        pending_frontier_items=hg["pending_frontier_items"],
        recipe_proofs_already_in_yaml=len(recipe_hashes),
    ), errors_by_hash


def register_pair(path, roots, deadline, cache, bindings):
    """Register the exact worker→record→audit→two artifact SHA chain, metadata only."""
    path = derived.canonical(path, roots)
    if path.parent.parent.name != "mechanism-terms" or path.name not in derived.NAMES:
        raise ValueError("exact closed writer layout required")
    owner, key = path.parent.parent.parent, path.parent.name
    if owner in cache:
        worker, worker_sha = cache[owner]
    else:
        cache[owner] = derived.metadata(owner / "worker.json", roots, deadline)
        worker, worker_sha = cache[owner]
    if worker.get("source_sha") != PRODUCER:
        raise ValueError("only the exactly reviewed native-v4 producer is authorized")
    record_path, audit_path = (
        owner / "records" / f"{key}.json",
        owner / "mechanisms" / f"{key}.json",
    )
    matches = [
        r
        for r in worker.get("records", [])
        if isinstance(r, dict)
        and isinstance(r.get("path"), str)
        and derived.canonical(r["path"], roots) == record_path
    ]
    if len(matches) != 1:
        raise ValueError("worker does not bind one exact immutable result record")
    record, record_sha = derived.metadata(record_path, roots, deadline, matches[0]["sha256"])
    if (
        record.get("source_sha") != PRODUCER
        or derived.canonical(record.get("mechanism_audit_path", ""), roots) != audit_path
    ):
        raise ValueError("record producer or audit path differs")
    audit, audit_sha = derived.metadata(
        audit_path, roots, deadline, record["mechanism_audit_sha256"]
    )
    terms = audit.get("raw_terms", {})
    if not isinstance(terms, dict) or len(terms) != 2:
        raise ValueError("exact two-artifact SHA map required")
    members = {str(derived.canonical(p, roots)): h for p, h in terms.items()}
    if (
        len(members) != 2
        or {Path(p).name for p in members} != derived.NAMES
        or any(Path(p).parent != path.parent for p in members)
        or any(
            not isinstance(h, str) or not derived.scanner.HASH.fullmatch(h)
            for h in members.values()
        )
    ):
        raise ValueError("registered producer artifact map differs")
    for p, expected in (
        (owner / "worker.json", worker_sha),
        (record_path, record_sha),
        (audit_path, audit_sha),
    ):
        if bindings.setdefault(str(p), expected) != expected:
            raise ValueError("conflicting registered provenance binding")
    return members[str(path)]


def prepare(root, *, cpu_gate_root):
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(OUTPUT_BASE.resolve()):
        raise ValueError("new write-once scratch output root required")
    revision = clean_source(SOURCE)
    cpu_gate(Path(cpu_gate_root), revision)
    evidence, errors = historical_evidence()
    covered = set(evidence["covered_error_hashes"])
    native = NATIVE_ROOT.resolve(strict=True)
    candidates = defaultdict(list)
    for error_sha, error in errors.items():
        path = Path(error["path"])
        if (
            error_sha not in covered
            and error.get("error") == derived.PARQUET_ERROR
            and path.is_relative_to(native)
            and path.name in derived.NAMES
        ):
            candidates[str(path)].append(error_sha)
    if len(candidates) != EXPECTED_FILES:
        raise ValueError("exact 678-file native-v4 pending scope required")
    roots, cache = [str(native)], {}
    deadline = time.monotonic() + 600
    producer = derived.producer_proof(PRODUCER, deadline)
    queue, costs = [], [0] * SHARDS
    rows = []
    for name, original in sorted(candidates.items()):
        path = derived.canonical(name, roots)
        registered = derived.scanner.signature(path)
        if registered[2] > derived.MAX_ARTIFACT_BYTES:
            raise ValueError("artifact exceeds unchanged per-file validator bound")
        expected = register_pair(path, roots, deadline, cache, evidence["bindings"])
        rows.append(
            dict(
                path=str(path),
                original_error_sha256s=sorted(original),
                signature=registered,
                expected_artifact_sha256=expected,
            )
        )
    if len(cache) != EXPECTED_OWNERS or sum(r["signature"][2] for r in rows) > MAX_SOURCE_BYTES:
        raise ValueError("registered native-v4 owner/total-byte boundary differs")
    for row in sorted(rows, key=lambda r: (-r["signature"][2], r["path"])):
        index = min(range(SHARDS), key=lambda i: (costs[i], i))
        costs[index] += row["signature"][2]
        queue.append(dict(row, index=index))
    evidence["bindings"].update(
        {
            str(Path(p).resolve()): sha(Path(p))
            for p in (
                __file__,
                derived.__file__,
                derived.scanner.__file__,
            )
        }
    )
    write_once(
        root / "plan.json",
        dict(
            **evidence,
            source_sha=revision,
            cpu_gate_root=str(Path(cpu_gate_root).resolve()),
            queue=queue,
            roots=roots,
            producer=producer,
            worker_source_bytes=costs,
            shard_count=SHARDS,
            seconds_per_worker=SECONDS,
            historical_roots_restarted=False,
            original_errors_preserved=True,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
            automatic_resubmission=False,
        ),
    )
    return dict(
        workers=SHARDS,
        files=len(queue),
        worker_source_bytes=costs,
        imported_proofs=EXPECTED_PRIOR,
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("registered current source or parent/provenance evidence changed")
    return plan


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered CPU index required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["seconds_per_worker"]
    queue = [r for r in plan["queue"] if r["index"] == index]
    committed, pending, blocked, cache = [], [], [], {}
    for position, row in enumerate(queue):
        try:
            derived.check(deadline)
            proof = derived.prove(row, plan["roots"], deadline, cache)
            if (
                proof["whole_file_sha256_before"] != row["expected_artifact_sha256"]
                or proof["source_proof"]["recorded_producer_commit"] != PRODUCER
                or any(plan["bindings"].get(p) != h for p, h in proof["bindings"].items())
            ):
                raise ValueError("actual strict proof differs from registered producer transaction")
            committed.append(proof)
        except TimeoutError:
            pending.extend(queue[position:])
            break
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            blocked.append(dict(row=row, error=type(exc).__name__))
    path = root / "workers" / f"shard-{index}.json"
    write_once(
        path,
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=sha(root / "plan.json"),
            index=index,
            committed=committed,
            pending=pending,
            blocked=blocked,
            scheduler_job_id=os.getenv("SLURM_JOB_ID"),
            fresh_confirmation_authorized=False,
        ),
    )
    return dict(
        index=index,
        committed=len(committed),
        pending=len(pending),
        blocked=len(blocked),
        worker_sha256=sha(path),
    )


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    expected = {r["path"]: r for r in plan["queue"]}
    proofs, pending, blocked, missing, bindings = [], [], [], [], {}
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        result = read(path)
        if (
            result["source_sha"] != plan["source_sha"]
            or result["index"] != index
            or result["plan_sha256"] != sha(root / "plan.json")
        ):
            raise ValueError("worker source/index/plan mismatch")
        bindings[str(path)] = sha(path)
        names = [r["path"] for r in result["committed"] + result["pending"]]
        names += [r["row"]["path"] for r in result["blocked"]]
        assigned = {p for p, row in expected.items() if row["index"] == index}
        if len(names) != len(set(names)) or set(names) != assigned:
            raise ValueError("registered candidates dropped or duplicated")
        for row in result["pending"] + [r["row"] for r in result["blocked"]]:
            if row != expected[row["path"]]:
                raise ValueError("unresolved registration differs")
        for proof in result["committed"]:
            row = expected[proof["path"]]
            footer = proof["parquet_footer"]
            fields = footer["fields"]
            schema = derived.SCHEMAS[Path(proof["path"]).name]
            if (
                proof["original_error_sha256s"] != row["original_error_sha256s"]
                or proof["signature"] != row["signature"]
                or proof["whole_file_sha256_before"] != row["expected_artifact_sha256"]
                or proof["whole_file_sha256_after"] != row["expected_artifact_sha256"]
                or proof["source_proof"] != plan["producer"]
                or proof["resolved_as"] != "derived_no_request_or_media_identity"
                or footer["rows_decoded"] != 0
                or len(fields) != len(schema)
                or {name for name, _kind in fields} != set(schema)
                or any(kind not in schema[name] for name, kind in fields)
                or footer["schema_sha256"] != digest(fields)
                or any(plan["bindings"].get(p) != h for p, h in proof["bindings"].items())
            ):
                raise ValueError("strict producer/schema proof differs from registration")
            proofs.append(proof)
        pending.extend(result["pending"])
        blocked.extend(result["blocked"])
    missing_files = sum(r["index"] in missing for r in plan["queue"])
    if len(proofs) + len(pending) + len(blocked) + missing_files != EXPECTED_FILES:
        raise ValueError("original 678-file denominator not conserved")
    resolved = {h for p in proofs for h in p["original_error_sha256s"]}
    before = set(plan["covered_error_hashes"])
    if resolved & before:
        raise ValueError("new proofs overlap previously resolved original errors")
    errors = read(plan["original_errors_path"], plan["bindings"][plan["original_errors_path"]])
    combined_coverage = before | resolved
    unresolved = [e for e in errors if digest(e) not in combined_coverage]
    report = dict(
        original_error_count=len(errors),
        unique_covered_before=len(before),
        newly_covered_error_hashes=len(resolved),
        net_unresolved_after=len(unresolved),
        net_unresolved_categories=dict(Counter(e["error"] for e in unresolved)),
        pending_frontier_items=plan["pending_frontier_items"],
        producer_recipe_subset_of_yaml=plan["recipe_proofs_already_in_yaml"],
        scientific_scope="provenance_coverage_only_not_identity_or_media_freshness",
    )
    write_once(root / "net_error_report.json", report)
    write_once(
        root / "coverage_overlay.json",
        dict(
            source_sha=plan["source_sha"],
            imported_proofs=plan["imported_proofs"],
            new_proofs=proofs,
            parent_bindings=plan["bindings"],
            original_errors_path=plan["original_errors_path"],
            original_errors_preserved=True,
            fresh_confirmation_authorized=False,
        ),
    )
    gate = dict(
        status="native_v4_provenance_recovery_complete"
        if not (missing or pending or blocked)
        else "native_v4_provenance_recovery_incomplete",
        source_sha=plan["source_sha"],
        registered_files=EXPECTED_FILES,
        newly_proved_files=len(proofs),
        previously_proved_files=len(plan["imported_proofs"]),
        pending_files=len(pending),
        blocked_files=len(blocked),
        missing_files=missing_files,
        missing_shards=missing,
        worker_bindings=bindings,
        coverage_overlay_sha256=sha(root / "coverage_overlay.json"),
        net_error_report_sha256=sha(root / "net_error_report.json"),
        **report,
        original_errors_preserved=True,
        error_waivers=0,
        roots_restarted=False,
        external_video_journals_unresolved=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        automatic_resubmission=False,
    )
    write_once(root / GATE, gate)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    result = (
        prepare(args.root, cpu_gate_root=args.cpu_gate_root)
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(result))
