#!/usr/bin/env python3
"""Two canaries then at most two CPU proof waves; never repeat successful work."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

import evicastkv_coverage_resolution_v36 as original
import evicastkv_coverage_retry_v38 as packing
import evicastkv_global_history_v36 as history
import evicastkv_reviewed_producers_v39 as producers
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.history_occurrence_reconciliation import reconcile_occurrences

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v38/run-20261111-1")
BINDINGS = {
    "decision.json": "8fdf139f19e1b6b040d62716a501092ed38f2120b965d4262574e87c9cec7e80",
    "reconciled_history.json": "74888a9eda60255a7ff303e1c86568bdb1819b4d6c4064d0bd9f0ad508a346eb",
    "coverage/coverage_overlay.json": (
        "7e690a5642ebfa70661c790bb479f4cf11bd243d0d36631a99bd822bf888ceb0"
    ),
    "coverage/GATE_coverage_retry_v38.json": (
        "fbced9236f3e7d63e5837725238f08d714e128e39389dafb553e77e516e39c49"
    ),
}


def plan_rows(old_plan):
    selected = [r for r in old_plan["queue"] if r["kind"] == "closed_parquet"]
    selected += [r["row"] for r in old_plan["registration_blocked"] if r.get("resource_deferred")]
    if len({r["path"] for r in selected}) != len(selected):
        raise ValueError("duplicate retry candidate")
    return sorted(selected, key=lambda r: r["path"])


def register(root):
    revision = clean_source(SOURCE)
    inputs = {name: history.read(PARENT / name, expected) for name, expected in BINDINGS.items()}
    if (
        inputs["decision.json"]["verification_complete"] is not True
        or inputs["decision.json"]["missing_workers"]
        or inputs["decision.json"]["error_counts"]["net_unresolved_error_count"] != 4556
    ):
        raise ValueError("the completed V38 checkpoint is required")
    old_plan = history.read(PARENT / "coverage/plan.json")
    gate = inputs["coverage/GATE_coverage_retry_v38.json"]
    if len(inputs["coverage/coverage_overlay.json"]["new_proofs"]) != gate["proved_files"]:
        raise ValueError("preserved proof denominator changed")
    for path, expected in gate["worker_bindings"].items():
        history.read(path, expected)
    rows = plan_rows(old_plan)
    if len(rows) != 2848:
        raise ValueError("only failed or resource-deferred reviewed-layout candidates may retry")
    bindings = {str(PARENT / k): v for k, v in BINDINGS.items()}
    bindings[str(PARENT / "coverage/plan.json")] = sha(PARENT / "coverage/plan.json")
    plan = dict(
        source_sha=revision,
        rows=rows,
        roots=old_plan["roots"],
        bindings=bindings,
        source_bridge=history.source_bridge(),
        maximum_waves=2,
        workers=8,
        maximum_worker_two_pass_bytes=2 * 2**30,
        worker_seconds=210,
        completed_YAML_JSON_proofs_imported_not_repeated=gate["proved_files"],
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )
    write_once(root / "registration.json", plan)
    return plan


def load(root):
    plan = history.read(root / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["source_bridge"] != history.source_bridge():
        raise ValueError("reviewed source closure changed")
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("immutable completed evidence changed")
    if (
        plan["rows"] != plan_rows(history.read(PARENT / "coverage/plan.json"))
        or plan["maximum_waves"] != 2
        or plan["workers"] != 8
        or plan["maximum_worker_two_pass_bytes"] != 2 * 2**30
        or plan["worker_seconds"] != 210
        or plan["fresh_confirmation_authorized"] is not False
    ):
        raise ValueError("registered retry scope or bounds changed")
    return plan


def cpu(root):
    register(root)
    os.environ["PYTEST_ADDOPTS"] = "-vv --durations=20 --durations-min=1 -o faulthandler_timeout=60"
    from evicastkv_review_closure_v2 import cpu_suite

    cpu_suite(root / "cpu/full-cpu", timeout_seconds=1200, capture_mode="sys")
    return cpu_gate(root / "cpu", clean_source(SOURCE))


def prove_row(row, roots, deadline, cache):
    proof = producers.prove(row, roots, deadline, cache)
    producers.validate_proof(proof)
    return dict(proof, kind="closed_parquet", registration_sha256=digest(row))


def worker(root, wave, index):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    spec = history.read(root / f"wave-{wave}/plan.json")
    rows = [r for r in spec["queue"] if r["index"] == index]
    if spec["source_sha"] != plan["source_sha"] or not 0 <= index < 8 or wave not in (0, 1):
        raise ValueError("registered wave and worker index required")
    used, cache, proofs, blocked, pending = 0, {}, [], [], []
    deadline = time.monotonic() + 210
    for offset, row in enumerate(rows):
        amount = 2 * row["signature"][2]
        if time.monotonic() >= deadline or used + amount > 2 * 2**30:
            pending.extend(rows[offset:])
            break
        used += amount
        try:
            proofs.append(prove_row(row, plan["roots"], deadline, cache))
        except TimeoutError:
            pending.extend(rows[offset:])
            break
        except (OSError, ValueError) as error:
            blocked.append(dict(row=row, error_type=type(error).__name__, error=str(error)))
    result = dict(
        source_sha=plan["source_sha"],
        wave=wave,
        index=index,
        plan_sha256=sha(root / f"wave-{wave}/plan.json"),
        proofs=proofs,
        blocked=blocked,
        pending=pending,
        two_pass_bytes_budgeted=used,
    )
    write_once(root / f"wave-{wave}/worker-{index}.json", result)
    return dict(proved=len(proofs), blocked=len(blocked), pending=len(pending))


def run_wave(root, number, rows):
    packed = packing.pack_wave(rows)
    folder = root / f"wave-{number}"
    write_once(folder / "plan.json", dict(source_sha=clean_source(SOURCE), queue=packed["queue"]))

    def run(index):
        with (folder / f"worker-{index}.log").open("x") as output:
            try:
                done = subprocess.run(
                    [
                        "/scratch/djy8hg/env/mosaickv/bin/python",
                        str(Path(__file__)),
                        "worker",
                        "--root",
                        str(root),
                        "--wave",
                        str(number),
                        "--index",
                        str(index),
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=420,
                    check=False,
                )
                return dict(index=index, returncode=done.returncode)
            except subprocess.TimeoutExpired:
                return dict(index=index, returncode=None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(run, range(8)))
    write_once(folder / "statuses.json", dict(workers=statuses))
    proofs, remaining = [], [r["row"] for r in packed["registration_blocked"]]
    by_path = {r["path"]: r for r in packed["queue"]}
    for index in range(8):
        path = folder / f"worker-{index}.json"
        expected = {r["path"] for r in packed["queue"] if r["index"] == index}
        if not path.is_file():
            remaining.extend(by_path[p] for p in sorted(expected))
            continue
        result = history.read(path)
        if (
            result["source_sha"] != clean_source(SOURCE)
            or result["index"] != index
            or result["wave"] != number
            or result["plan_sha256"] != sha(folder / "plan.json")
            or result["two_pass_bytes_budgeted"] > 2 * 2**30
        ):
            raise ValueError("worker proof source, index or budget changed")
        observed = [p["path"] for p in result["proofs"] + result["pending"]]
        observed += [p["row"]["path"] for p in result["blocked"]]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError("worker omitted or duplicated a candidate")
        for proof in result["proofs"]:
            row = by_path[proof["path"]]
            producers.validate_proof(proof)
            if (
                proof["registration_sha256"] != digest(row)
                or proof["original_error_sha256s"] != row["original_error_sha256s"]
                or proof["signature"] != row["signature"]
            ):
                raise ValueError("proof changed its original error/signature binding")
        proofs.extend(result["proofs"])
        # Terminal incompatible proofs are not automatically retried unchanged.
        remaining.extend(result["pending"])
    return proofs, remaining


def continue_recovery(root):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    representatives, cache, owner_metadata = {}, {}, {}
    for row in plan["rows"]:
        owner = Path(row["path"]).parent.parent.parent
        if owner not in owner_metadata:
            metadata, _ = original.derived.metadata(
                owner / "worker.json", plan["roots"], time.monotonic() + 10
            )
            owner_metadata[owner] = metadata.get("source_sha")
        revision = owner_metadata[owner]
        producers.reviewed_ast_map(revision)
        representatives.setdefault(revision, row)
    if len(representatives) != 2:
        raise ValueError("exactly the two reviewed historical producer groups required")
    canaries = []
    for revision, row in sorted(representatives.items()):
        proof = prove_row(row, plan["roots"], time.monotonic() + 60, cache)
        write_once(root / f"canaries/{revision}.json", proof)
        canaries.append(proof)
    write_once(
        root / "canaries.json",
        dict(proofs=canaries, complete=True, fresh_confirmation_authorized=False),
    )
    proved = {p["path"]: p for p in canaries}
    todo = [r for r in plan["rows"] if r["path"] not in proved]
    for wave in range(2):
        if not todo:
            break
        completed, todo = run_wave(root, wave, todo)
        proved.update((p["path"], p) for p in completed)
    imported = history.read(
        PARENT / "coverage/coverage_overlay.json", BINDINGS["coverage/coverage_overlay.json"]
    )
    prior = history.read(PARENT / "reconciled_history.json", BINDINGS["reconciled_history.json"])
    bindings = dict(prior["source_bindings"], **plan["bindings"])
    for proof in proved.values():
        bindings.update(proof["bindings"])
    combined = dict(
        imported,
        source_sha=plan["source_sha"],
        parent_bindings=bindings,
        new_proofs=imported["new_proofs"] + list(proved.values()),
    )
    history.put(root / "coverage_overlay.json", combined)
    seed_path = Path(
        "/scratch/djy8hg/workdir/fox-review-v36/run-20261111-1/history/registration.json"
    )
    parent_registration = history.read(PARENT / "registration.json")
    seed = history.read(seed_path, parent_registration["parent_registration_sha256"])
    original_path = imported["original_errors_path"]
    originals = history.read(original_path, imported["parent_bindings"][original_path])
    value = dict(
        source_sha=plan["source_sha"],
        identities=prior["identities"],
        errors=prior["raw_error_records"],
    )
    report = reconcile_occurrences(
        value,
        seed_error_counts=seed["seed_error_counts"],
        preserved_resolved_error_sha256s=seed["preserved_resolved_error_sha256s"],
        original_errors=originals,
        source_bindings=bindings,
        coverage_overlay=combined,
    )
    history.put(root / "reconciled_history.json", report)
    result = dict(
        source_sha=plan["source_sha"],
        canaries_passed=2,
        retry_candidates=len(plan["rows"]),
        proved_retry_files=len(proved),
        unresolved_retry_candidates=len(plan["rows"]) - len(proved),
        imported_successful_proof_files=len(imported["new_proofs"]),
        unprocessed_resource_or_time_deferred=len(todo),
        error_counts=report["counts"],
        fresh_confirmation_authorized=False,
        source_group_provenance_complete=False,
        original_traversal_replayed=False,
        GPU_jobs=0,
    )
    write_once(root / "decision.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "continue", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", type=int)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only output required")
    try:
        result = (
            cpu(root)
            if args.mode == "cpu"
            else worker(root, args.wave, args.index)
            if args.mode == "worker"
            else continue_recovery(root)
        )
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}-{args.wave}-{args.index}.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
