#!/usr/bin/env python3
"""Recover only unresolved JSON; preserve V42 and stop on resource failures."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import evicastkv_residual_recovery_v41 as worker
from evicastkv_review_closure_v2 import cpu_suite

from mosaickv.evaluation import metadata_recovery_v41 as core
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v42/run-20261112-1")
PARENT_SHA = {
    "decision.json": "4b7a733e0aedf8285e545a6d10f61f30422655e6f9891832fdf766eea376bb89",
    "reconciled_history.json": "bc72a0403f8aafc6495792045b146c338a4d878f1792a0a67ba39e943de3a709",
}
OLD = Path("/scratch/djy8hg/workdir/fox-review-v41/run-20261112-1")
OLD_PLAN_SHA = "94282140db11f00c0cf8605196a3fd34553cfd7cceb35df2f3daaab5762112b5"
RESOURCE_ERRORS = {
    "bounded projected metadata capacity exceeded",
    "bounded streaming metadata deadline reached",
}


def cpu(root):
    revision = clean_source(SOURCE)
    os.environ["PYTEST_ADDOPTS"] = "-vv --durations=20 --durations-min=1 -o faulthandler_timeout=60"
    cpu_suite(root / "cpu/full-cpu", timeout_seconds=1200, capture_mode="sys")
    return worker.cpu_gate(root / "cpu", revision)


def prepare(root):
    revision = clean_source(SOURCE)
    worker.cpu_gate(root / "cpu", revision)
    decision = worker.history.read(PARENT / "decision.json", PARENT_SHA["decision.json"])
    parent = worker.history.read(
        PARENT / "reconciled_history.json", PARENT_SHA["reconciled_history.json"]
    )
    if (
        decision["source_sha"] != "6ae1e119b0c39ef858267b17c60e329bede00657"
        or parent["counts"]["net_unresolved_error_count"] != 1364
    ):
        raise ValueError("exact completed V42 recovery required")
    old = worker.history.read(OLD / "registration.json", OLD_PLAN_SHA)
    unresolved = {core._key(r): r for r in parent["unresolved_occurrences"]}
    targets = []
    for i, row in enumerate(old["rows"]):
        error = unresolved.get(core._key(row))
        if error is not None and row["operation"] == "json_stream":
            if (
                error["path"] != row["path"]
                or error["original_error_sha256"] != row["original_error_sha256"]
            ):
                raise ValueError("original occurrence binding changed")
            targets.append(i)
    if len(targets) != 1229:
        raise ValueError("only 1229 unresolved JSON occurrences may be scheduled")
    canaries = []
    for name in sorted(RESOURCE_ERRORS):
        candidates = []
        for i in targets:
            path = worker.artifact(OLD, i, "parse")
            if not path.is_file():
                continue
            result = worker.history.read(path)
            if (
                result["source_sha"] != "d37fc8daf75af02d66ce35cf76abe0d7c3a5446b"
                or result["registration_sha256"] != digest(old["rows"][i])
                or result["plan_sha256"] != OLD_PLAN_SHA
            ):
                raise ValueError("prior failure does not bind its original source/occurrence")
            if result.get("error") == name:
                candidates.append(i)
        if candidates:
            canaries.append(max(candidates, key=lambda i: (old["rows"][i]["signature"][2], -i)))
    if len(canaries) != 2 or len(set(canaries)) != 2:
        raise ValueError("one consumed deadline and one metadata-capacity canary required")
    bindings = dict(old["bindings"])
    bindings.update({str(PARENT / name): value for name, value in PARENT_SHA.items()})
    bindings[str(OLD / "registration.json")] = OLD_PLAN_SHA
    plan = dict(
        old,
        source_sha=revision,
        source_bridge=worker.history.source_bridge(),
        bindings=bindings,
        retry_indices=targets,
        json_canary_indices=canaries,
        reused_parent_source_sha=decision["source_sha"],
        successful_prior_JSON_occurrences_replayed=False,
        non_JSON_occurrences_retried=False,
        maximum_new_parse_wave_size=8,
        resource_failure_stops_subsequent_waves=True,
    )
    write_once(root / "registration.json", plan)
    return dict(
        retry_occurrences=len(targets), canaries=canaries, fresh_confirmation_authorized=False
    )


def canary(root):
    result = worker.canary(root)
    if result["streaming_bulk_authorized"]:
        plan = worker.load(root)
        for index in plan["json_canary_indices"]:
            qualified_pair(root, plan, index)
    return result


def qualified_pair(root, plan, index):
    result = worker.json_pair(root, plan, index)
    first = worker.checked_task(root, plan, index, "parse")["proof"]
    second = worker.checked_task(root, plan, index, "hash")["proof"]
    if (
        first["path"] != plan["rows"][index]["path"]
        or second["path"] != plan["rows"][index]["path"]
        or type(first["hash_passes"]) is not int
    ):
        raise ValueError("independent passes do not bind the registered input")
    return result


def bulk(root):
    plan = worker.load(root)
    worker.cpu_gate(root / "cpu", plan["source_sha"])
    gate = worker.history.read(root / "canary.json")
    if gate["streaming_bulk_authorized"] is not True or gate["source_sha"] != plan["source_sha"]:
        result = dict(status="blocked_canary_failed", fresh_confirmation_authorized=False)
        write_once(root / "bulk.json", result)
        return result
    for index in plan["json_canary_indices"]:
        qualified_pair(root, plan, index)
    started = time.monotonic()
    reason = "all_registered_waves_processed"
    waves = 0
    for offset in range(0, len(plan["retry_indices"]), 8):
        remaining = plan["bulk_wall_seconds"] - (time.monotonic() - started)
        if remaining < 600:
            reason = "registered_wall_bound_deferred"
            break
        indices = plan["retry_indices"][offset : offset + 8]
        worker.run_tasks(
            root, plan, indices, "parse", f"bounded-parse-wave-{waves}", maximum_wall=remaining
        )
        records = [worker.checked_task(root, plan, i, "parse") for i in indices]
        passed = [i for i, r in zip(indices, records, strict=True) if r and r["status"] == "passed"]
        worker.run_tasks(
            root, plan, passed, "hash", f"bounded-hash-wave-{waves}", maximum_wall=remaining
        )
        waves += 1
        if any(
            r is None or (r.get("status") != "passed" and r.get("error") in RESOURCE_ERRORS)
            for r in records
        ):
            reason = "resource_failure_stopped_later_waves"
            break
    result = dict(
        source_sha=plan["source_sha"],
        status=reason,
        completed_waves=waves,
        worker_seconds=210,
        worker_maximum_bytes=2 * 2**30,
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )
    write_once(root / "bulk.json", result)
    return result


def analyze(root):
    plan = worker.load(root)
    worker.cpu_gate(root / "cpu", plan["source_sha"])
    bindings = dict(plan["bindings"])
    bindings[str(root / "registration.json")] = sha(root / "registration.json")
    rows, receipts, failures = [], [], []
    for index in plan["retry_indices"]:
        row = plan["rows"][index]
        try:
            proof = qualified_pair(root, plan, index)
            if any(k in proof and proof[k] is not False for k in core.contract.FALSE_FIELDS):
                raise ValueError("proof scope widened")
            path = root / f"proofs/{index:04d}.json"
            write_once(path, dict(proof, registration_sha256=digest(row)))
            item = worker.receipt(row, proof, path)
            rows.append(row)
            receipts.append(item)
            bindings[str(path)] = sha(path)
            for name, value in proof["task_artifacts"].items():
                bindings[name] = value
        except (ValueError, OSError, KeyError) as error:
            failures.append(dict(index=index, registration_sha256=digest(row), error=str(error)))
    parent = worker.history.read(
        PARENT / "reconciled_history.json", PARENT_SHA["reconciled_history.json"]
    )
    report = core.apply_metadata_proofs(
        parent,
        rows,
        receipts,
        source_sha=plan["source_sha"],
        bindings=bindings,
        seed_error_counts=plan["seed_error_counts"],
    )
    worker.history.put(root / "reconciled_history.json", report)
    write_once(root / "failures.json", dict(unresolved_registered_occurrences=failures))
    result = dict(
        source_sha=plan["source_sha"],
        newly_resolved_JSON_occurrences=len(receipts),
        error_counts=report["counts"],
        parent_source_sha=parent["source_sha"],
        successful_prior_proofs_replayed=False,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        historical_media_exposure_resolved=False,
        materialization_authorized=False,
        reconciled_history_sha256=sha(root / "reconciled_history.json"),
    )
    write_once(root / "decision.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "prepare", "canary", "bulk", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("write-once scratch output required")
    try:
        print(globals()[args.mode](root))
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                fresh_confirmation_authorized=False,
            ),
        )
        raise
