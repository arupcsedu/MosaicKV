#!/usr/bin/env python3
"""Finite metadata recovery from V40; never an inference/freshness waiver."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path

import evicastkv_global_history_v36 as history
import evicastkv_json_projection_v41 as streaming
import evicastkv_parquet_residuals_v41 as parquet
import evicastkv_small_residuals_v41 as small
import evicastkv_source_recovery_v41 as source_recovery
from evicastkv_occurrence_recovery_v40 import ROOT_PLAN, ROOT_PLAN_SHA, SEED, checked_roots
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation import metadata_recovery_v41 as core
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v40/run-20261112-1")
PARENT_HASHES = {
    "decision.json": "6ae7de26da8de309838d8682abddcdad98b7e312686882724dd556834ec30d21",
    "reconciled_history.json": "f670e729c43fbc4e038bcc6c5ee01aa4bfcadf49863317121419073b323c5ebe",
}
SLOTS, SECONDS, MAX_BYTES = 8, 210, 2 * 2**30


def operation(error):
    for name, kinds in core.OPERATIONS.items():
        if error in ((kinds,) if isinstance(kinds, str) else kinds):
            return name
    raise ValueError("unregistered residual category")


def register(root):
    revision = clean_source(SOURCE)
    parent = history.read(
        PARENT / "reconciled_history.json", PARENT_HASHES["reconciled_history.json"]
    )
    decision = history.read(PARENT / "decision.json", PARENT_HASHES["decision.json"])
    if (
        parent["counts"]["net_unresolved_error_count"] != 1434
        or decision["resolved_occurrences"] != 274
    ):
        raise ValueError("exact completed V40 checkpoint required")
    seeds = history.read(SEED)["seed_error_counts"]
    if digest(seeds) != parent["input_metadata_sha256s"]["seed_error_counts"]:
        raise ValueError("original owner boundaries changed")
    roots, rows = checked_roots(), []
    for error in parent["unresolved_occurrences"]:
        row = {name: error[name] for name in core.contract.OCCURRENCE_FIELDS | {"path"}}
        path = Path(row["path"])
        # Only lstat here: no credential bodies or external alias targets.
        row.update(operation=operation(error["error"]), signature=source_recovery.signature(path))
        if row["operation"] != "source_alias" and (
            not stat.S_ISREG(path.lstat().st_mode)
            or not any(path.resolve(strict=True).is_relative_to(Path(p)) for p in roots)
        ):
            raise ValueError("registered regular file must remain inside historical roots")
        rows.append(row)
    if len(rows) != 1434 or sum(r["operation"] == "json_stream" for r in rows) != 1261:
        raise ValueError("exact original residual denominator required")
    indices = [i for i, r in enumerate(rows) if r["operation"] == "json_stream"]
    canaries = [
        min(indices, key=lambda i: (rows[i]["signature"][2], i)),
        max(indices, key=lambda i: (rows[i]["signature"][2], -i)),
    ]
    bindings = {str(PARENT / k): v for k, v in PARENT_HASHES.items()}
    bindings.update({str(SEED): sha(SEED), str(ROOT_PLAN): ROOT_PLAN_SHA})
    plan = dict(
        source_sha=revision,
        source_bridge=history.source_bridge(),
        bindings=bindings,
        rows=rows,
        rows_sha256=digest(rows),
        roots=roots,
        seed_error_counts=seeds,
        json_canary_indices=canaries,
        original_traversal_replayed=False,
        prior_successful_proofs_replayed=False,
        worker_seconds=SECONDS,
        worker_maximum_bytes=MAX_BYTES,
        maximum_parallel_workers=SLOTS,
        bulk_wall_seconds=5400,
        metadata_only=True,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "registration.json", plan)
    return plan


def load(root, *, expected=None, full=True):
    plan = history.read(root / "registration.json", expected)
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["rows"]) != plan["rows_sha256"] or len(plan["rows"]) != 1434:
        raise ValueError("registered occurrences changed")
    if full:
        if history.source_bridge() != plan["source_bridge"] or checked_roots() != plan["roots"]:
            raise ValueError("source or historical scope changed")
        for path, expected_hash in plan["bindings"].items():
            if sha(Path(path)) != expected_hash:
                raise ValueError("immutable parent metadata changed")
    return plan


def cpu(root):
    plan = register(root)
    os.environ["PYTEST_ADDOPTS"] = "-vv --durations=20 --durations-min=1 -o faulthandler_timeout=60"
    from evicastkv_review_closure_v2 import cpu_suite

    cpu_suite(root / "cpu/full-cpu", timeout_seconds=1200, capture_mode="sys")
    return cpu_gate(root / "cpu", plan["source_sha"])


def artifact(root, index, phase):
    return root / f"tasks/{index:04d}-{phase}.json"


def hash_file(path, signature, deadline):
    if signature[2] > MAX_BYTES:
        raise ValueError("registered one-pass byte bound exceeded")
    if source_recovery.signature(path) != signature or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("registered hash-pass source changed")
    hashed, total = hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        first = os.fstat(stream.fileno())
        if streaming._signature(first) != signature:
            raise ValueError("hash-pass descriptor differs")
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("independent hash-pass deadline")
            raw = stream.read(min(2**20, MAX_BYTES - total + 1))
            if not raw:
                break
            total += len(raw)
            if total > MAX_BYTES:
                raise ValueError("independent hash-pass byte bound")
            hashed.update(raw)
        if streaming._signature(os.fstat(stream.fileno())) != signature:
            raise ValueError("hash-pass descriptor changed")
    if total != signature[2] or source_recovery.signature(path) != signature:
        raise ValueError("hash-pass length/signature changed")
    return dict(
        path=str(path),
        signature=signature,
        bytes_read=total,
        whole_file_sha256=hashed.hexdigest(),
        independent_hash_pass=True,
    )


def worker(root, index, phase, plan_sha):
    plan = load(root, expected=plan_sha, full=False)
    cpu_gate(root / "cpu", plan["source_sha"])
    if type(index) is not int or index not in range(len(plan["rows"])):
        raise ValueError("registered task index required")
    row, started = plan["rows"][index], time.monotonic()
    deadline = started + SECONDS
    try:
        if row["operation"] == "json_stream":
            if phase == "parse":
                proof = streaming.project_file(Path(row["path"]), row["signature"], deadline)
            elif phase == "hash":
                proof = hash_file(Path(row["path"]), row["signature"], deadline)
            else:
                raise ValueError("JSON requires two independent registered passes")
        elif phase == "prove":
            if row["operation"] == "small":
                proof = small.prove(row, plan["roots"], deadline)
            elif row["operation"] == "parquet":
                proof = parquet.prove(row, plan["roots"], deadline)
            else:
                proof = source_recovery.prove_alias(row, plan["roots"], deadline)
        else:
            raise ValueError("unsupported proof phase")
        result = dict(status="passed", proof=proof)
    except (ValueError, OSError, TimeoutError, RecursionError, UnicodeError) as error:
        result = dict(status="blocked", error_type=type(error).__name__, error=str(error))
    result.update(
        source_sha=plan["source_sha"],
        registration_sha256=digest(row),
        index=index,
        phase=phase,
        plan_sha256=plan_sha,
        wall_seconds=time.monotonic() - started,
    )
    history.put(artifact(root, index, phase), result)
    return {k: v for k, v in result.items() if k != "proof"}


def checked_task(root, plan, index, phase):
    path = artifact(root, index, phase)
    if not path.is_file():
        return None
    if path.resolve(strict=True) != path.absolute():
        raise ValueError("task artifact alias outside registered output")
    result = history.read(path)
    if (
        result["source_sha"] != plan["source_sha"]
        or result["index"] != index
        or result["phase"] != phase
        or result["plan_sha256"] != sha(root / "registration.json")
        or result["registration_sha256"] != digest(plan["rows"][index])
    ):
        raise ValueError("task artifact source/occurrence differs")
    return result


def run_tasks(root, plan, indices, phase, label, *, maximum_wall=5400):
    end = time.monotonic() + maximum_wall
    plan_sha = sha(root / "registration.json")

    def one(index):
        if checked_task(root, plan, index, phase) is not None:
            return dict(index=index, reused=True)
        if time.monotonic() >= end - 240:
            return dict(index=index, deferred=True)
        path = root / f"logs/task-{index:04d}-{phase}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            try:
                result = subprocess.run(
                    [
                        "/scratch/djy8hg/env/mosaickv/bin/python",
                        str(Path(__file__)),
                        "worker",
                        "--root",
                        str(root),
                        "--index",
                        str(index),
                        "--phase",
                        phase,
                        "--plan-sha",
                        plan_sha,
                    ],
                    cwd=SOURCE,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=240,
                    check=False,
                )
                return dict(index=index, returncode=result.returncode)
            except subprocess.TimeoutExpired:
                return dict(index=index, timed_out=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=SLOTS) as pool:
        results = list(pool.map(one, indices))
    write_once(root / f"stages/{label}.json", dict(phase=phase, tasks=results))
    return results


def json_pair(root, plan, index):
    first, second = (
        checked_task(root, plan, index, "parse"),
        checked_task(root, plan, index, "hash"),
    )
    if (
        first is None
        or second is None
        or first["status"] != "passed"
        or second["status"] != "passed"
    ):
        raise ValueError("both independent JSON passes must complete successfully")
    a, b, row = first["proof"], second["proof"], plan["rows"][index]
    if (
        a["whole_file_sha256"] != b["whole_file_sha256"]
        or a["bytes_read"] != row["signature"][2]
        or b["bytes_read"] != row["signature"][2]
        or a["hash_passes"] != 1
        or a["independent_second_pass_verified"] is not False
        or b["independent_hash_pass"] is not True
        or any(
            a[key] != row["signature"]
            for key in (
                "signature_before",
                "signature_after",
                "fd_signature_before",
                "fd_signature_after",
            )
        )
        or b["signature"] != row["signature"]
    ):
        raise ValueError("independent JSON hash/signature/length transaction differs")
    return dict(
        path=row["path"],
        signature=row["signature"],
        proof_kind="strict_streaming_JSON_two_pass",
        whole_file_sha256_before=a["whole_file_sha256"],
        whole_file_sha256_after=b["whole_file_sha256"],
        identities=a["identities"],
        parser_statistics=a["statistics"],
        two_independent_bounded_passes=True,
        task_artifacts={
            str(artifact(root, index, p)): sha(artifact(root, index, p)) for p in ("parse", "hash")
        },
        old_unbound_historical_version_byte_identity_claimed=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )


def canary(root):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    indices = plan["json_canary_indices"]
    run_tasks(root, plan, indices, "parse", "canary-parse")
    passed = [
        i for i in indices if (checked_task(root, plan, i, "parse") or {}).get("status") == "passed"
    ]
    run_tasks(root, plan, passed, "hash", "canary-hash")
    pairs, errors = [], []
    for index in indices:
        try:
            pairs.append(dict(index=index, proof_sha256=digest(json_pair(root, plan, index))))
        except ValueError as error:
            errors.append(dict(index=index, error=str(error)))
    result = dict(
        source_sha=plan["source_sha"],
        required_indices=indices,
        passed=pairs,
        failures=errors,
        streaming_bulk_authorized=len(pairs) == len(indices),
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )
    write_once(root / "canary.json", result)
    return result


def bulk(root):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    canaries = history.read(root / "canary.json")
    non_json = [i for i, r in enumerate(plan["rows"]) if r["operation"] != "json_stream"]
    run_tasks(root, plan, non_json, "prove", "small-proofs", maximum_wall=900)
    if (
        canaries["streaming_bulk_authorized"] is not True
        or canaries["source_sha"] != plan["source_sha"]
    ):
        result = dict(status="streaming_blocked_canary_failed", fresh_confirmation_authorized=False)
    else:
        for index in plan["json_canary_indices"]:
            json_pair(root, plan, index)
        indices = [i for i, r in enumerate(plan["rows"]) if r["operation"] == "json_stream"]
        started = time.monotonic()
        run_tasks(root, plan, indices, "parse", "bulk-parse", maximum_wall=3600)
        passed = [
            i
            for i in indices
            if (checked_task(root, plan, i, "parse") or {}).get("status") == "passed"
        ]
        remaining = max(240, plan["bulk_wall_seconds"] - (time.monotonic() - started))
        run_tasks(root, plan, passed, "hash", "bulk-hash", maximum_wall=remaining)
        result = dict(status="bounded_bulk_completed", fresh_confirmation_authorized=False)
    write_once(root / "bulk.json", result)
    return result


def receipt(row, proof, proof_path):
    if proof["path"] != row["path"] or digest(proof["signature"]) != digest(row["signature"]):
        raise ValueError("semantic proof source differs from registered occurrence")
    if any(proof.get(k) is True for k in core.contract.FALSE_FIELDS):
        raise ValueError("semantic proof attempted to widen scope")
    if any(
        proof.get(k) is True
        for k in (
            "old_unbound_historical_version_byte_identity_claimed",
            "old_historical_byte_claim",
        )
    ):
        raise ValueError("unbound historical byte identity cannot be inferred")
    if row["operation"] == "small" and proof.get("proof_kind") not in {
        "bounded_parser_event_DAG_identity_projection",
        "stable_zero_byte_file_no_body_read",
        "actual_passed_test_closed_scalar_fixture",
        "nonfinite_aggregate_redacted_metadata_only",
        "existing_derived_recovery_original_truncation_preserved",
    }:
        raise ValueError("unregistered small-file semantic proof")
    if row["operation"] == "parquet" and (
        proof.get("resolved_as") != "closed_current_empty_Arrow_footer_no_data_pages"
        or proof.get("source_proof", {}).get("rows") != 0
        or proof.get("source_proof", {}).get("columns") != 0
    ):
        raise ValueError("unregistered Parquet semantic proof")
    if row["operation"] == "source_alias" and (
        proof.get("proof_kind") != "pinned_upstream_source_alias"
        or proof.get("source_revision") != source_recovery.PREFIX_COMMIT
        or proof.get("configuration_files_verified") != 20
    ):
        raise ValueError("unregistered source-alias semantic proof")
    before = proof.get("whole_file_sha256_before", proof.get("full_file_sha256_before"))
    after = proof.get("whole_file_sha256_after", proof.get("full_file_sha256_after"))
    result = dict(
        registration_sha256=digest(row),
        operation=row["operation"],
        path=row["path"],
        signature=row["signature"],
        proof_path=str(proof_path),
        proof_sha256=sha(proof_path),
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
        source_hash_kind=proof.get("source_hash_kind", "file_bytes"),
        old_unbound_historical_version_byte_identity_claimed=False,
        **dict.fromkeys(core.contract.FALSE_FIELDS, False),
    )
    if proof.get("identities") is not None:
        result["identities"] = {
            k: v for k, v in proof["identities"].items() if k != "projected_records"
        }
    return result


def analyze(root):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    registrations, receipts, failures, bindings = [], [], [], dict(plan["bindings"])
    bindings[str(root / "registration.json")] = sha(root / "registration.json")
    for index, row in enumerate(plan["rows"]):
        try:
            if row["operation"] == "json_stream":
                proof = json_pair(root, plan, index)
            else:
                record = checked_task(root, plan, index, "prove")
                if record is None or record["status"] != "passed":
                    raise ValueError(
                        "no completed semantic proof" if record is None else record["error"]
                    )
                proof = record["proof"]
            if source_recovery.signature(Path(row["path"])) != row["signature"]:
                raise ValueError("source changed after its completed proof")
            path = root / f"proofs/{index:04d}.json"
            write_once(path, dict(proof, registration_sha256=digest(row)))
            item = receipt(row, proof, path)
            bindings[str(path)] = sha(path)
            for field in ("bindings", "source_bindings", "task_artifacts"):
                for name, expected in proof.get(field, {}).items():
                    candidate = Path(name)
                    if ".codex" in candidate.parts or candidate.name.lower() in {
                        "auth.json",
                        "credentials.json",
                    }:
                        continue  # Stat-only credential proof stays in its bound receipt.
                    if name in bindings and bindings[name] != expected:
                        raise ValueError("immutable proof input binding collision")
                    bindings[name] = expected
            registrations.append(row)
            receipts.append(item)
        except (ValueError, OSError, KeyError) as error:
            failures.append(dict(index=index, registration_sha256=digest(row), error=str(error)))
    parent = history.read(
        PARENT / "reconciled_history.json", PARENT_HASHES["reconciled_history.json"]
    )
    result = core.apply_metadata_proofs(
        parent,
        registrations,
        receipts,
        source_sha=plan["source_sha"],
        bindings=bindings,
        seed_error_counts=plan["seed_error_counts"],
    )
    history.put(root / "reconciled_history.json", result)
    write_once(root / "failures.json", dict(registered_failures=failures))
    decision = dict(
        source_sha=plan["source_sha"],
        registered_occurrences=len(plan["rows"]),
        resolved_occurrences=len(receipts),
        unresolved_registered_occurrences=len(failures),
        error_counts=result["counts"],
        fresh_confirmation_authorized=False,
        historical_media_exposure_resolved=False,
        source_group_provenance_complete=False,
        materialization_authorized=False,
        GPU_jobs=0,
        reconciled_history_sha256=sha(root / "reconciled_history.json"),
    )
    write_once(root / "decision.json", decision)
    return decision


def media(root):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    deadline = time.monotonic() + SECONDS
    report = source_recovery.recover_sources(SOURCE, deadline=deadline)
    report["isolated_remote_source_attempt"] = source_recovery.remote_source_attempt(
        root, report["missing_exact_source_commits"], deadline=deadline
    )
    write_once(root / "media-source-recovery.json", dict(source_sha=plan["source_sha"], **report))
    return dict(
        inspected_databases=len(report["inspected_databases"]),
        historical_media_exposure_resolved=False,
        fresh_confirmation_authorized=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "canary", "bulk", "analyze", "media", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--phase", choices=("parse", "hash", "prove"))
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("new write-once scratch output root required")
    try:
        result = (
            worker(root, args.index, args.phase, args.plan_sha)
            if args.mode == "worker"
            else globals()[args.mode](root)
        )
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}-{args.index}.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
