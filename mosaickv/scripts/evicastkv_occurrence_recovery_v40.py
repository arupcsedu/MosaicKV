#!/usr/bin/env python3
"""Recover only the 274 new, already identified YAML/graph occurrences."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

import evicastkv_global_history_v36 as history
import evicastkv_residual_projection_v40 as projection
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.residual_occurrence_proofs import apply_proofs

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v39/run-20261112-1")
PARENT_HASHES = {
    "decision.json": "1ac85e988cc772957b954ab2b502009a8e97bf031051bc4964cba46e68b19fe5",
    "reconciled_history.json": "9379abbc0d6d6551f10a4c360c8d3e819f2d6f5e31972fb663839ceb598bf3ad",
}
SEED = Path("/scratch/djy8hg/workdir/fox-review-v38/run-20261111-1/registration.json")
ROOT_PLAN = Path("/scratch/djy8hg/workdir/fox-review-v38/run-20261111-1/coverage/plan.json")
ROOT_PLAN_SHA = "f451828fb75dadbc495a59eb36805c2c2d8beafbdc9421212009203362c01907"


def selected_occurrences(parent, counts):
    rows = []
    for row in parent["unresolved_occurrences"]:
        if row["historical_error_ordinal"] < counts[str(row["owner_index"])]:
            continue
        kind = {
            "Parquet has no safely projectable identity metadata columns": "closed_parquet",
            "YAML requires separately reviewed lexical identity projection": "yaml_events",
        }.get(row["error"])
        if kind is not None:
            rows.append((row, kind))
    return rows


def checked_roots():
    roots = history.read(ROOT_PLAN, ROOT_PLAN_SHA)["roots"]
    if roots != [str(path.resolve(strict=True)) for path in projection.common.scanner.ROOTS]:
        raise ValueError("canonical historical roots differ from immutable scanner scope")
    return roots


def register(root):
    revision = clean_source(SOURCE)
    parent = history.read(
        PARENT / "reconciled_history.json", PARENT_HASHES["reconciled_history.json"]
    )
    decision = history.read(PARENT / "decision.json", PARENT_HASHES["decision.json"])
    seed = history.read(SEED)
    counts = seed["seed_error_counts"]
    if (
        decision["error_counts"]["net_unresolved_error_count"] != 1708
        or digest(counts) != parent["input_metadata_sha256s"]["seed_error_counts"]
    ):
        raise ValueError("exact V39 residual checkpoint and original owner boundaries required")
    roots = checked_roots()
    rows, costs, deadline = [], [0] * 8, time.monotonic() + 210
    for error, kind in selected_occurrences(parent, counts):
        path = projection.common.canonical(error["path"], roots)
        record = dict(
            owner_index=error["owner_index"],
            historical_error_ordinal=error["historical_error_ordinal"],
            original_error_sha256=error["original_error_sha256"],
            path=str(path),
            kind=kind,
            signature=projection.common.scanner.signature(path),
        )
        record["file_sha256"] = projection.expected_file_sha(record, roots, deadline)
        rows.append(record)
    if len(rows) != 274 or sum(r["kind"] == "closed_parquet" for r in rows) != 256:
        raise ValueError("only 256 new graph and 18 new YAML occurrences may be recovered")
    assignments = []
    for row in sorted(rows, key=lambda r: (-r["signature"][2], r["path"])):
        index = min(range(8), key=lambda i: (costs[i], i))
        costs[index] += 2 * row["signature"][2]
        assignments.append(dict(registration=row, index=index))
    if max(costs) > 2 * 2**30:
        raise ValueError("unchanged 2-GiB two-pass worker bound exceeded")
    bindings = {str(PARENT / k): v for k, v in PARENT_HASHES.items()}
    bindings[str(SEED)] = sha(SEED)
    bindings[str(ROOT_PLAN)] = ROOT_PLAN_SHA
    plan = dict(
        source_sha=revision,
        bindings=bindings,
        rows=assignments,
        roots=roots,
        seed_error_counts=counts,
        source_bridge=history.source_bridge(),
        two_pass_worker_bytes=costs,
        original_traversal_replayed=False,
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )
    write_once(root / "registration.json", plan)
    return plan


def load(root):
    plan = history.read(root / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["source_bridge"] != history.source_bridge():
        raise ValueError("source closure changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("bound prior evidence changed")
    if plan["roots"] != checked_roots():
        raise ValueError("registered historical roots changed")
    return plan


def cpu(root):
    register(root)
    os.environ["PYTEST_ADDOPTS"] = "-vv --durations=20 --durations-min=1 -o faulthandler_timeout=60"
    from evicastkv_review_closure_v2 import cpu_suite

    cpu_suite(root / "cpu/full-cpu", timeout_seconds=1200, capture_mode="sys")
    return cpu_gate(root / "cpu", clean_source(SOURCE))


def receipt_for(registration, proof):
    if (
        proof["path"] != registration["path"]
        or proof["signature"] != registration["signature"]
        or proof["kind"] != registration["kind"]
        or proof["registration_sha256"] != digest(projection_row(registration))
        or proof["original_error_sha256s"] != [registration["original_error_sha256"]]
    ):
        raise ValueError("semantic proof differs from its exact occurrence registration")
    receipt = dict(
        registration_sha256=digest(registration),
        kind=registration["kind"],
        signature=proof["signature"],
        whole_file_sha256_before=proof["whole_file_sha256_before"],
        whole_file_sha256_after=proof["whole_file_sha256_after"],
    )
    if "identities" in proof:
        receipt["identities"] = {
            k: v for k, v in proof["identities"].items() if k != "projected_records"
        }
    return receipt


def projection_row(registration):
    return dict(
        registration,
        current_file_sha256=registration["file_sha256"],
        original_error_sha256s=[registration["original_error_sha256"]],
    )


def checked_proof_path(root, registration, recorded):
    expected = root / f"proofs/{digest(registration)[:20]}.json"
    if Path(recorded) != expected or Path(recorded).resolve(strict=True) != expected.absolute():
        raise ValueError("only exact registered proof artifacts may be opened")
    return expected


def worker(root, index):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("registered worker index required")
    rows = [r["registration"] for r in plan["rows"] if r["index"] == index]
    deadline, used, cache = time.monotonic() + 210, 0, {}
    receipts, failures = [], []
    for registration in rows:
        try:
            amount = 2 * registration["signature"][2]
            if used + amount > 2 * 2**30 or time.monotonic() >= deadline:
                raise TimeoutError("unchanged worker time/byte bound")
            used += amount
            row = projection_row(registration)
            proof = projection.prove(row, plan["roots"], deadline, cache)
            projection.validate_proof(proof)
            receipt = receipt_for(registration, proof)
            name = digest(registration)[:20]
            path = root / f"proofs/{name}.json"
            write_once(path, proof)
            receipts.append(
                dict(
                    registration=registration,
                    receipt=receipt,
                    proof_path=str(path),
                    proof_sha256=sha(path),
                )
            )
        except (ValueError, OSError, TimeoutError) as error:
            failures.append(
                dict(
                    registration_sha256=digest(registration),
                    error_type=type(error).__name__,
                    error=str(error),
                )
            )
    result = dict(
        source_sha=plan["source_sha"],
        index=index,
        plan_sha256=sha(root / "registration.json"),
        receipts=receipts,
        failures=failures,
        two_pass_bytes_budgeted=used,
    )
    write_once(root / f"workers/{index}.json", result)
    return dict(proved=len(receipts), unresolved=len(failures))


def recover(root):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])

    def run(index):
        path = root / f"worker-{index}.log"
        with path.open("x") as stream:
            try:
                completed = subprocess.run(
                    [
                        "/scratch/djy8hg/env/mosaickv/bin/python",
                        str(Path(__file__)),
                        "worker",
                        "--root",
                        str(root),
                        "--index",
                        str(index),
                    ],
                    cwd=SOURCE,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=420,
                    check=False,
                )
                return dict(index=index, returncode=completed.returncode)
            except subprocess.TimeoutExpired:
                return dict(index=index, returncode=None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(run, range(8)))
    write_once(root / "statuses.json", dict(workers=statuses))
    registrations, receipts, bindings = [], [], dict(plan["bindings"])
    for index in range(8):
        path = root / f"workers/{index}.json"
        if not path.is_file():
            continue
        result = history.read(path)
        if (
            result["source_sha"] != plan["source_sha"]
            or result["index"] != index
            or result["plan_sha256"] != sha(root / "registration.json")
            or result["two_pass_bytes_budgeted"] > 2 * 2**30
        ):
            raise ValueError("worker source/index/plan/bound changed")
        expected = {
            digest(r["registration"]): r["registration"]
            for r in plan["rows"]
            if r["index"] == index
        }
        observed = [digest(r["registration"]) for r in result["receipts"]]
        observed += [r["registration_sha256"] for r in result["failures"]]
        if len(observed) != len(set(observed)) or set(observed) != set(expected):
            raise ValueError("registered occurrence denominator was dropped or duplicated")
        for item in result["receipts"]:
            registration = item["registration"]
            if registration != expected[digest(registration)]:
                raise ValueError("registered occurrence changed")
            proof_path = checked_proof_path(root, registration, item["proof_path"])
            proof = history.read(proof_path, item["proof_sha256"])
            projection.validate_proof(proof)
            if item["receipt"] != receipt_for(registration, proof):
                raise ValueError("receipt does not describe the bound semantic proof")
            bindings[item["proof_path"]] = item["proof_sha256"]
            registrations.append(registration)
            receipts.append(item["receipt"])
        bindings[str(path)] = sha(path)
    parent = history.read(
        PARENT / "reconciled_history.json", PARENT_HASHES["reconciled_history.json"]
    )
    report = apply_proofs(
        parent,
        registrations,
        receipts,
        source_bindings=bindings,
        analysis_source_sha=plan["source_sha"],
        seed_error_counts=plan["seed_error_counts"],
    )
    history.put(root / "reconciled_history.json", report)
    result = dict(
        source_sha=plan["source_sha"],
        registered_occurrences=274,
        resolved_occurrences=len(receipts),
        unresolved_registered_occurrences=274 - len(receipts),
        worker_statuses=statuses,
        error_counts=report["counts"],
        fresh_confirmation_authorized=False,
        source_group_provenance_complete=False,
        GPU_jobs=0,
        reconciled_history_sha256=sha(root / "reconciled_history.json"),
    )
    write_once(root / "decision.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "recover", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only output required")
    try:
        result = (
            cpu(root)
            if args.mode == "cpu"
            else worker(root, args.index)
            if args.mode == "worker"
            else recover(root)
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
