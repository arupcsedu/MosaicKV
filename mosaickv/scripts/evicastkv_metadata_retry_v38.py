#!/usr/bin/env python3
"""Finite CPU verification/coverage retry; completed traversal is never replayed."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import os
import subprocess
from pathlib import Path

import evicastkv_global_history_v36 as old
import evicastkv_history_reconcile_v37 as reconcile
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation import history_verification_retry as verification
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.history_occurrence_reconciliation import reconcile_occurrences

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v36/run-20261111-1/history")
PARENT_GATE_SHA = "74e5c94d319740029b77614dc02d760d9e6909863a7e6cfa8c568096fe0cf0e7"
MAX_STEPS = 6


def register(root):
    revision = clean_source(SOURCE)
    gate = old.read(PARENT / "GATE_global_history_v36.json", PARENT_GATE_SHA)
    parent = old.read(PARENT / "registration.json")
    if (
        gate["remaining_pending_items"] != 0
        or gate["complete"] is not False
        or gate["source_sha"] != "1e90ac0c1190809596a820c636b543473f55696f"
        or parent["source_sha"] != gate["source_sha"]
        or not gate["current_time_verification_epoch_performed"]
        or gate["reason"] != "failed_execution_partition"
        or set(gate["heads"]) != {str(i) for i in range(8)}
    ):
        raise ValueError("exact traversal-exhausted failed verification checkpoint required")
    plan = dict(
        source_sha=revision,
        parent_gate_sha256=PARENT_GATE_SHA,
        parent_registration_sha256=sha(PARENT / "registration.json"),
        heads=gate["heads"],
        roots=parent["roots"],
        seed_error_counts=parent["seed_error_counts"],
        preserved_resolved_error_sha256s=parent["preserved_resolved_error_sha256s"],
        source_bridge=old.source_bridge(),
        original_owner_count=8,
        maximum_steps_per_owner=MAX_STEPS,
        scanner_seconds=210,
        scanner_source_byte_bound=2 * 2**30,
        original_traversal_replayed=False,
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )
    write_once(root / "registration.json", plan)
    return plan


def load(root):
    plan = old.read(root / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    parent_gate = old.read(PARENT / "GATE_global_history_v36.json", PARENT_GATE_SHA)
    parent = old.read(PARENT / "registration.json", plan["parent_registration_sha256"])
    if (
        plan["parent_gate_sha256"] != PARENT_GATE_SHA
        or plan["heads"] != parent_gate["heads"]
        or plan["roots"] != parent["roots"]
        or plan["seed_error_counts"] != parent["seed_error_counts"]
        or plan["preserved_resolved_error_sha256s"] != parent["preserved_resolved_error_sha256s"]
        or plan["maximum_steps_per_owner"] != MAX_STEPS
        or plan["scanner_seconds"] != 210
        or plan["scanner_source_byte_bound"] != 2 * 2**30
        or plan["original_traversal_replayed"] is not False
        or plan["fresh_confirmation_authorized"] is not False
        or sha(PARENT / "GATE_global_history_v36.json") != plan["parent_gate_sha256"]
        or sha(PARENT / "registration.json") != plan["parent_registration_sha256"]
        or plan["source_bridge"] != old.source_bridge()
    ):
        raise ValueError("saved source/checkpoint binding changed")
    return plan


def cpu(root):
    register(root)
    # Reporting verbosity identifies the exact last test if infrastructure stalls.
    os.environ["PYTEST_ADDOPTS"] = "-vv --durations=20 --durations-min=1 -o faulthandler_timeout=60"
    closure = importlib.import_module("evicastkv_review_closure_v2")
    closure.cpu_suite(root / "cpu/full-cpu", timeout_seconds=1200, capture_mode="sys")
    return cpu_gate(root / "cpu", clean_source(SOURCE))


def dispatch(root):
    """Admit no worker array if the prerequisite fails; leave no orphan jobs."""
    plan = load(root)
    try:
        cpu_gate(root / "cpu", plan["source_sha"])
    except (OSError, ValueError, KeyError) as error:
        result = dict(
            stage="cpu",
            complete=False,
            error_type=type(error).__name__,
            worker_jobs_submitted=0,
            source_sha=plan["source_sha"],
            fresh_confirmation_authorized=False,
        )
        write_once(root / "decision.json", result)
        return result

    def submit(label, mode, options):
        command = [
            "sbatch",
            "--parsable",
            f"--job-name=fox38-{label}",
            f"--output={root}/slurm/{label}-%A_%a.out",
            f"--error={root}/slurm/{label}-%A_%a.err",
            *options,
            str(SOURCE / "mosaickv/slurm/fox_metadata_retry_v38.sbatch"),
            str(SOURCE),
            str(root),
            mode,
        ]
        write_once(root / f"jobs/{label}-request.json", dict(command=command))
        identity = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
        if not identity.isdigit():
            raise ValueError("invalid scheduler identity")
        write_once(root / f"jobs/{label}-submission.json", dict(job_id=identity, command=command))
        return identity

    workers = submit(
        "verify", "worker", ["--array=0-7%8", "--cpus-per-task=1", "--mem=8G", "--time=00:50:00"]
    )
    analysis = submit(
        "closeout", "analyze", [f"--dependency=afterany:{workers}", "--mem=64G", "--time=00:40:00"]
    )
    result = dict(
        source_sha=plan["source_sha"],
        verification_array=workers,
        analysis_job=analysis,
        original_traversal_replayed=False,
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )
    write_once(root / "dispatch.json", result)
    return result


def worker(root, index):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("original owner index 0..7 required")
    head = plan["heads"][str(index)]
    write_once(
        root / f"verification/owner-{index}/claim.json",
        dict(job_id=os.getenv("SLURM_JOB_ID"), source_sha=plan["source_sha"], head=head),
    )
    for step in range(MAX_STEPS):
        base, _ = old.state(head)
        if base["frontier"]:
            raise ValueError("verification retry must never restart historical traversal")
        if base["traversal_complete"]:
            break
        scanner = old.modules()[0]
        result = verification.advance_verification(
            base, tuple(Path(p) for p in plan["roots"]), scanner, owner_index=index
        )
        verification.verify_verification(base, result, owner_index=index)
        path = root / f"verification/owner-{index}/step-{step:03d}.json"
        old.put(path, dict(source_sha=plan["source_sha"], **result))
        head = dict(
            path=str(path), sha256=sha(path), container=True, state_sha256=result["state_sha256"]
        )
        if result["state"]["traversal_complete"]:
            break
        if result["state"]["verification_index"] == base["verification_index"]:
            raise ValueError("verification cursor made no progress")
    state, _ = old.state(head)
    decision = dict(
        source_sha=plan["source_sha"],
        owner=index,
        head=head,
        complete=state["traversal_complete"],
        summary=old.owner_summary(state),
        original_traversal_replayed=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / f"verification/owner-{index}/decision.json", decision)
    return decision


def coverage(root, cpu_root, inventory):
    module = importlib.import_module("evicastkv_coverage_retry_v38")
    prepared = module.prepare(
        root,
        cpu_gate_root=cpu_root,
        global_inventory=inventory,
        global_inventory_sha256=sha(inventory),
    )
    script = SOURCE / "mosaickv/scripts/evicastkv_coverage_retry_v38.py"

    def run(index):
        with (root / f"worker-{index}.log").open("x") as stream:
            try:
                done = subprocess.run(
                    [
                        "/scratch/djy8hg/env/mosaickv/bin/python",
                        str(script),
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
                return dict(index=index, returncode=done.returncode, timeout=False)
            except subprocess.TimeoutExpired:
                return dict(index=index, returncode=None, timeout=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(run, range(8)))
    write_once(root / "statuses.json", dict(prepared=prepared, workers=statuses))
    module.analyze(root)
    return old.read(root / "coverage_overlay.json")


def analyze(root):
    plan = load(root)
    cpu_gate(root / "cpu", plan["source_sha"])
    identities, errors, missing, owners, bindings = {}, [], [], {}, {}
    for owner in range(8):
        head = plan["heads"][str(owner)]
        # Keep valid partial progress; no failed owner is restarted from seed.
        for step in range(MAX_STEPS):
            path = root / f"verification/owner-{owner}/step-{step:03d}.json"
            if not path.is_file():
                break
            base, _ = old.state(head)
            record = old.read(path)
            if record["source_sha"] != plan["source_sha"]:
                raise ValueError("verification transition producer changed")
            verification.verify_verification(base, record, owner_index=owner)
            head = dict(
                path=str(path),
                sha256=sha(path),
                container=True,
                state_sha256=record["state_sha256"],
            )
            bindings[str(path)] = head["sha256"]
        path = root / f"verification/owner-{owner}/decision.json"
        if path.is_file():
            decision = old.read(path)
            if decision["source_sha"] != plan["source_sha"] or decision["owner"] != owner:
                raise ValueError("verification decision producer/owner changed")
            if decision["head"] != head:
                raise ValueError("terminal decision differs from validated checkpoint chain")
            bindings[str(path)] = sha(path)
        else:
            # Missing workers remain missing; original exclusions are still retained.
            missing.append(owner)
        state, _ = old.state(head)
        owners[str(owner)] = old.owner_summary(state)
        bindings[head["path"]] = head["sha256"]
        for key in (
            "sample_ids",
            "coco_ids",
            "media_hashes",
            "source_group_ids",
            "scopes",
            "scoped_identities",
        ):
            identities.setdefault(key, set()).update(
                tuple(v) if isinstance(v, list) else v for v in state[key]
            )
        errors.extend(
            dict(
                owner_index=owner,
                historical_error_ordinal=i,
                original_error_sha256=digest(row),
                **row,
            )
            for i, row in enumerate(state["errors"])
        )
    inventory = root / "combined_identity_inventory.json"
    value = dict(
        source_sha=plan["source_sha"],
        identities={k: sorted(v) for k, v in identities.items()},
        errors=errors,
        complete_historical_id_inventory=False,
        fresh_confirmation_authorized=False,
    )
    old.put(inventory, value)
    bindings[str(inventory)] = sha(inventory)
    bindings[str(root / "registration.json")] = sha(root / "registration.json")
    overlay = None
    try:
        overlay = coverage(root / "coverage", root / "cpu", inventory)
        bindings[str(root / "coverage/coverage_overlay.json")] = sha(
            root / "coverage/coverage_overlay.json"
        )
        bindings.update(overlay["parent_bindings"])
    except Exception as error:
        write_once(
            root / "coverage-error.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                fresh_confirmation_authorized=False,
            ),
        )
    verified = reconcile.prior.validate_saved(reconcile.prior.PARENT)
    old_plan = json.loads(verified["raws"]["plan.json"])
    bindings.update(verified["bindings"])
    originals = reconcile.metadata(
        old_plan["original_errors_path"],
        bindings,
        old_plan["bindings"][old_plan["original_errors_path"]],
    )
    extra = reconcile.imported_identity_proofs(verified, bindings)
    inventories = reconcile.imported_identity_inventories(verified, bindings)
    report = reconcile_occurrences(
        value,
        seed_error_counts=plan["seed_error_counts"],
        preserved_resolved_error_sha256s=plan["preserved_resolved_error_sha256s"],
        original_errors=originals,
        source_bindings=bindings,
        coverage_overlay=overlay,
        additional_identity_proofs=extra,
        additional_identity_inventories=inventories,
    )
    old.put(root / "reconciled_history.json", report)
    result = dict(
        source_sha=plan["source_sha"],
        missing_workers=missing,
        owners=owners,
        verification_complete=not missing and all(v["traversal_complete"] for v in owners.values()),
        error_counts=report["counts"],
        reconciled_history_sha256=sha(root / "reconciled_history.json"),
        original_traversal_replayed=False,
        source_group_provenance_complete=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        GPU_jobs=0,
        remaining_external_dependency="historical media/source-group records not recovered",
    )
    write_once(root / "decision.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "dispatch", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("write-once output must remain in project scratch")
    try:
        result = (
            dispatch(root)
            if args.mode == "dispatch"
            else (
                cpu(root)
                if args.mode == "cpu"
                else worker(root, args.index)
                if args.mode == "worker"
                else analyze(root)
            )
        )
    except Exception as error:
        path = root / f"failure-{args.mode}-{args.index}.json"
        write_once(
            path,
            dict(
                error_type=type(error).__name__,
                error=str(error),
                source_sha=clean_source(SOURCE),
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
