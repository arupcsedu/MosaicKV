#!/usr/bin/env python3
"""Finite continuation of all 8,481 saved historical cursors, then verification.

Original owner indices, projections and scan budgets are unchanged. Completed
V35 work is imported, not repeated. Worker deltas avoid eight copies of each
large base state. Historical errors are retained, never waived or relabeled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from mosaickv.evaluation import global_history_resume as core
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
BASE = "d0e47fda4ce1c46e029836cfa45ee9dd53894b10"
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v16/run-20261111-2/history/segment-4")
DECISION_SHA = "9063ff8e19e6a551fc9aa72b2bb5d2edd535d63cf9d6b02111ff0c2794654f58"
ONE = Path(
    "/scratch/djy8hg/workdir/fox-history-failed7-retry/run-20261111-1/subset/workers/shard-1.json"
)
ONE_SHA = "ac2c7e2fa5b2eb6d74f87450647e6b2ab7552c3d5b9f2bf53881992086d4645e"
SEVEN = Path(
    "/scratch/djy8hg/workdir/fox-review-v35/run-20261111-1/history/round-04/merged-state.json"
)
V35 = SEVEN.parents[1]
V35_GATE_SHA = "7f338bc1da928cd58a032043401f71c6609cdc3ff31031c26cd7be495030fec4"
MAX_STATE, SLOTS, MAX_ROUNDS, MAX_WALL = 512 * 2**20, 32, 40, 14100
EXPECTED_PENDING = [77, 1685, 0, 0, 0, 1781, 0, 4938]
POLICY = dict(
    original_owners=list(range(8)),
    original_owner_count=8,
    initial_pending_items=8481,
    completed_506_roots_not_repeated=True,
    parallel_CPU_slots=SLOTS,
    scanner_seconds=210,
    scanner_bytes=2 * 2**30,
    scanner_line_bytes=8 * 2**20,
    outer_worker_seconds=600,
    maximum_rounds=MAX_ROUNDS,
    maximum_wall_seconds=MAX_WALL,
    maximum_checkpoint_bytes=MAX_STATE,
    maximum_new_artifact_bytes=64 * 2**30,
    current_time_verification_epoch_required=True,
    credential_body_policy="exclude_as_unresolved_without_reading",
    metadata_only=True,
    GPU_jobs=0,
    automatic_failed_worker_retry=False,
    fresh_confirmation_authorized=False,
    materialization_authorized=False,
)


def modules():
    old = importlib.import_module("evicastkv_coco_history_subset")
    return old.modules()


def read(path, expected=None):
    from evicastkv_freshness_producer_canary import read_bytes, read_json

    return read_json(read_bytes(Path(path), expected=expected, bound=MAX_STATE)[0])


def put(path, value):
    modules()[2].bounded_write(Path(path), value, MAX_STATE, compact=True)


def source_bridge():
    raw = subprocess.check_output(
        ["git", "diff", "--name-status", f"{BASE}..HEAD"], cwd=SOURCE, text=True, timeout=15
    )
    additions = []
    for line in raw.splitlines():
        action, path = line.split("\t", 1)
        if action != "A" or not path.startswith("mosaickv/"):
            raise ValueError("only additive audit/admission tooling may reuse prior controls")
        additions.append(path)
    return dict(base_sha=BASE, additions=additions, scanner=modules()[2].bridge())


def state(head):
    record = read(head["path"], head["sha256"])
    value = record["state"] if head["container"] else record
    value, normalization = core.normalize_seed(value)
    if digest(value) != head["state_sha256"]:
        raise ValueError("owner state digest differs from immutable head")
    return value, normalization


def preservation(old, new):
    """Import only append-only progress; never drop historical exclusions."""
    from mosaickv.evaluation import coco_history_resume as legacy

    for name in legacy.SET_FIELDS:
        if not legacy._values(old, name) <= legacy._values(new, name):
            raise ValueError("import lost historical identity/path exclusions")
    for name in legacy.APPEND_FIELDS:
        if new[name][: len(old[name])] != old[name]:
            raise ValueError("import rewrote original historical error/receipt prefix")
    for name in (*legacy.DICT_FIELDS, "empty_parquet_projection"):
        if any(key not in new[name] or new[name][key] != value for key, value in old[name].items()):
            raise ValueError("import rewrote a historical binding")
    for name in legacy.COUNTER_FIELDS:
        if type(new[name]) is not int or new[name] < old[name]:
            raise ValueError("import lost historical progress")
    if any(
        new[name] != old[name]
        for name in ("verification_paths", "verification_index", "traversal_complete")
    ):
        raise ValueError("import changed original verification state")


def saved_resolution():
    """Validate prior proof transactions once, without redoing their scans."""
    importer = importlib.import_module("evicastkv_provenance_import")
    recovery = importer.validate_saved(importer.PARENT)
    plan = json.loads(recovery["raws"]["plan.json"])
    overlay = json.loads(recovery["raws"]["coverage_overlay.json"])
    covered = set(plan["covered_error_hashes"])
    covered.update(h for proof in overlay["new_proofs"] for h in proof["original_error_sha256s"])
    if len(covered) != 29846:
        raise ValueError("completed original coverage resolution denominator differs")
    return dict(
        bindings=recovery["bindings"],
        covered_error_hashes=sorted(covered),
        original_source_sha=recovery["gate"]["source_sha"],
    )


def register(root, cpu_gate_root, original_review_root=None):
    revision = clean_source(SOURCE)
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("new immutable scratch root required")
    from evicastkv_review_followups import cpu_gate

    from mosaickv.evaluation import coco_history_subset as subset

    cpu = cpu_gate(Path(cpu_gate_root), revision)
    decision_path = PARENT / "decision.json"
    decision = read(decision_path, DECISION_SHA)
    previous_gate = read(V35 / "GATE_history_resume_v35.json", V35_GATE_SHA)
    previous_decision = read(SEVEN.parent / "decision.json")
    if (
        previous_gate.get("reviewed_pending_items") != 0
        or previous_gate.get("complete") is not True
    ):
        raise ValueError("completed 506-item continuation is not admissible")
    bindings = {
        str(decision_path): DECISION_SHA,
        str(V35 / "GATE_history_resume_v35.json"): V35_GATE_SHA,
        str(SEVEN.parent / "decision.json"): sha(SEVEN.parent / "decision.json"),
    }
    seeds, normalizations, seed_error_counts = {}, {}, {}
    counts = []
    for owner in range(8):
        path = PARENT / "workers" / f"shard-{owner}.json"
        expected = decision["worker_bindings"][str(path.resolve(strict=True))]
        original = read(path, expected)
        if original["index"] != owner or original["state_sha256"] != digest(original["state"]):
            raise ValueError("original owner/state binding differs")
        old, _ = core.normalize_seed(original["state"])
        bindings[str(path)] = expected
        container = True
        if owner == 1:
            path, expected = ONE, ONE_SHA
            imported = read(path, expected)
            split = subset.split_frontier(original, 1)
            subset.verify_continuation(original, split, imported["state"], imported["summary"])
            imported_state = imported["state"]
        elif owner == 7:
            path, expected, container = SEVEN, previous_decision["state_file_sha256"], False
            imported_state = read(path, expected)
            deferred = [row["item"] for row in subset.split_frontier(original, 7)["deferred"]]
            if imported_state["frontier"] != deferred:
                raise ValueError("completed selected roots reappeared or deferred cursors changed")
        else:
            imported_state = original["state"]
        value, normalization = core.normalize_seed(imported_state)
        preservation(old, value)
        bindings[str(path)] = expected
        seeds[str(owner)] = dict(
            path=str(path), sha256=expected, container=container, state_sha256=digest(value)
        )
        normalizations[str(owner)] = normalization
        seed_error_counts[str(owner)] = len(value["errors"])
        counts.append(len(value["frontier"]))
    if counts != EXPECTED_PENDING or sum(counts) != 8481:
        raise ValueError("the exact 8,481 pending original cursors differ")
    for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
        path = Path(cpu_gate_root) / "full-cpu" / name
        bindings[str(path)] = sha(path)
    # Reuse completed, hash-bound coverage proofs. Retain every raw error;
    # resolutions apply only to its original occurrence, never a later error
    # with the same path/message after source mutation.
    recovery = saved_resolution()
    covered = recovery["covered_error_hashes"]
    bindings.update(recovery["bindings"])
    scanner = modules()[0]
    if original_review_root is not None:
        review_module = importlib.import_module("evicastkv_original_complete_admission_v36")
        review_module.load_review(original_review_root)
        review_path = (
            Path(original_review_root).resolve(strict=True) / "original_complete_review.json"
        )
        bindings[str(review_path)] = sha(review_path)
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_bridge=source_bridge(),
            policy=POLICY,
            roots=[str(path.resolve(strict=True)) for path in scanner.ROOTS],
            seeds=seeds,
            seed_normalizations=normalizations,
            seed_error_counts=seed_error_counts,
            preserved_resolved_error_sha256s=sorted(covered),
            original_resolution_producer_sha=recovery["original_source_sha"],
            bindings=bindings,
            cpu_gate_root=str(Path(cpu_gate_root).resolve()),
            cpu_gate=cpu,
            pending_by_owner=counts,
            current_time_verification_not_yet_performed=True,
            original_review_root=str(Path(original_review_root).resolve())
            if original_review_root
            else None,
        ),
    )
    return dict(registered=True, pending_items=8481, CPU_slots=SLOTS, GPU_jobs=0)


def load(root):
    plan = read(Path(root) / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["policy"]) != digest(POLICY) or plan["source_bridge"] != source_bridge():
        raise ValueError("original scanner or immutable execution policy changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable provenance/control input changed")
    return plan


def allocate(counts, slots=SLOTS):
    """Give each active owner one slot, then balance pending items per slot."""
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("strict nonnegative pending counts required")
    active = {owner: 1 for owner, count in sorted(counts.items()) if count}
    if len(active) > slots:
        raise ValueError("not enough execution slots for active original owners")
    while active and sum(active.values()) < slots:
        eligible = [owner for owner in active if active[owner] < counts[owner]]
        if not eligible:
            break
        owner = min(eligible, key=lambda key: (-counts[key] / active[key], key))
        active[owner] += 1
    return active


def worker(root, number, index):
    root = Path(root).resolve(strict=True)
    plan = load(root)
    folder = root / f"round-{number:03d}"
    spec = read(folder / "plan.json")
    task = spec["tasks"][index]
    if spec["source_sha"] != plan["source_sha"] or task["index"] != index:
        raise ValueError("task/source index binding changed")
    if spec["roots_sha256"] != digest(plan["roots"]):
        raise ValueError("worker global roots differ from immutable registration")
    write_once(
        folder / "locks" / f"task-{index}.json",
        dict(job_id=os.getenv("SLURM_JOB_ID"), pid=os.getpid(), task=task),
    )
    base, _ = state(spec["heads"][str(task["owner"])])
    scanner, empty, _history = modules()
    allowed = tuple(Path(path) for path in plan["roots"])
    if spec["phase"] == "traverse":
        result = core.advance_partition(
            base,
            task["queue"],
            plan["roots"],
            allowed,
            scanner,
            empty.empty_projection_parquet,
            owner_index=task["owner"],
        )
        result = core.compact_result(base, task["queue"], result, owner_index=task["owner"])
    elif spec["phase"] == "verify":
        result = core.advance_verification(base, allowed, scanner, owner_index=task["owner"])
    else:
        raise ValueError("unknown bounded history phase")
    put(folder / "workers" / f"task-{index}.json", dict(source_sha=plan["source_sha"], **result))
    return result["summary"]


def run_task(root, number, index):
    folder = Path(root) / f"round-{number:03d}"
    command = [
        "/scratch/djy8hg/env/mosaickv/bin/python",
        str(Path(__file__).resolve()),
        "worker",
        "--root",
        str(root),
        "--round",
        str(number),
        "--index",
        str(index),
    ]
    started = time.monotonic()
    with (
        (folder / "logs" / f"task-{index}.out").open("x") as out,
        (folder / "logs" / f"task-{index}.err").open("x") as err,
    ):
        try:
            done = subprocess.run(
                command, cwd=SOURCE, stdout=out, stderr=err, timeout=600, check=False
            )
            result = dict(returncode=done.returncode, timeout=False)
        except subprocess.TimeoutExpired:
            result = dict(returncode=None, timeout=True)
    record = dict(index=index, command=command, wall_seconds=time.monotonic() - started, **result)
    write_once(folder / "jobs" / f"task-{index}.json", record)
    return record


def owner_summary(value):
    return dict(
        pending_items=len(value["frontier"]),
        errors=len(value["errors"]),
        processed_files=value["processed_files"],
        projected_records=value["projected_records"],
        verification_index=value["verification_index"],
        verification_paths=None
        if value["verification_paths"] is None
        else len(value["verification_paths"]),
        traversal_complete=value["traversal_complete"],
    )


def unresolved_errors(records, covered, seed_error_counts):
    """Old proof hashes cannot waive a new same-path error occurrence."""
    return [
        row
        for row in records
        if (
            row["historical_error_ordinal"] >= seed_error_counts[str(row["owner_index"])]
            or row["original_error_sha256"] not in covered
        )
    ]


def controller(root):
    root = Path(root).resolve(strict=True)
    plan = load(root)
    write_once(
        root / "controller.json",
        dict(
            source_sha=plan["source_sha"],
            job_id=os.getenv("SLURM_JOB_ID"),
            host=os.uname().nodename,
            no_polling=True,
        ),
    )
    heads, phase, epochs = dict(plan["seeds"]), "traverse", False
    started, reason, complete, rounds = time.monotonic(), "registered_bound", False, 0
    for number in range(MAX_ROUNDS):
        if time.monotonic() - started > MAX_WALL - 900:
            break
        used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if used + SLOTS * 2 * MAX_STATE > POLICY["maximum_new_artifact_bytes"]:
            reason = "registered_artifact_byte_bound"
            break
        summaries = {owner: owner_summary(state(head)[0]) for owner, head in heads.items()}
        if phase == "traverse" and all(not row["pending_items"] for row in summaries.values()):
            phase, epochs = "verify", True
            for owner, head in list(heads.items()):
                value, _ = state(head)
                epoch, receipt = core.begin_verification_epoch(value)
                path = root / "verification-epochs" / f"owner-{owner}.json"
                put(path, epoch)
                write_once(path.with_suffix(".receipt.json"), dict(origin=head, **receipt))
                heads[owner] = dict(
                    path=str(path), sha256=sha(path), container=False, state_sha256=digest(epoch)
                )
            summaries = {owner: owner_summary(state(head)[0]) for owner, head in heads.items()}
        if phase == "verify" and all(row["traversal_complete"] for row in summaries.values()):
            complete, reason = True, "all_original_owners_traversed_and_current_time_verified"
            break
        folder = root / f"round-{number:03d}"
        for name in ("logs", "jobs", "workers", "locks", "owners"):
            (folder / name).mkdir(parents=True, exist_ok=False)
        tasks = []
        if phase == "traverse":
            capacities = allocate(
                {int(owner): row["pending_items"] for owner, row in summaries.items()}
            )
            for owner, parts in capacities.items():
                value, _ = state(heads[str(owner)])
                divided = core.partition(value, value["frontier"], 0, parts, owner_index=owner)
                for queue in divided["queues"]:
                    tasks.append(dict(index=len(tasks), owner=owner, queue=queue))
        else:
            tasks = [
                dict(index=index, owner=int(owner))
                for index, owner in enumerate(
                    owner for owner, row in summaries.items() if not row["traversal_complete"]
                )
            ]
        write_once(
            folder / "plan.json",
            dict(
                source_sha=plan["source_sha"],
                heads=heads,
                phase=phase,
                tasks=tasks,
                roots_sha256=digest(plan["roots"]),
            ),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=SLOTS) as executor:
            statuses = list(
                executor.map(
                    lambda index, current=number: run_task(root, current, index), range(len(tasks))
                )
            )
        if any(row["returncode"] != 0 for row in statuses):
            reason = "failed_execution_partition"
            write_once(folder / "failure.json", dict(statuses=statuses, automatic_retry=False))
            break
        old_heads = dict(heads)
        for owner in sorted({row["owner"] for row in tasks}):
            value, _ = state(heads[str(owner)])
            owned = [row for row in tasks if row["owner"] == owner]
            results, queues, worker_bindings = [], [], {}
            for task in owned:
                path = folder / "workers" / f"task-{task['index']}.json"
                record = read(path)
                if record["source_sha"] != plan["source_sha"]:
                    raise ValueError("worker source differs")
                record = {key: value for key, value in record.items() if key != "source_sha"}
                worker_bindings[str(path)] = sha(path)
                if phase == "traverse":
                    queues.append(task["queue"])
                    results.append(
                        core.expand_result(value, task["queue"], record, owner_index=owner)
                    )
                else:
                    core.verify_verification(value, record, owner_index=owner)
                    results.append(record)
            if phase == "traverse":
                merged = core.merge(value, queues, results, [], owner_index=owner)
                path = folder / "owners" / f"owner-{owner}.json"
                put(path, merged)
                container = False
            else:
                if len(results) != 1:
                    raise ValueError("one current verification cursor per owner required")
                merged, path, container = (
                    results[0]["state"],
                    Path(next(iter(worker_bindings))),
                    True,
                )
            heads[str(owner)] = dict(
                path=str(path), sha256=sha(path), container=container, state_sha256=digest(merged)
            )
            write_once(
                folder / "owners" / f"owner-{owner}.receipt.json",
                dict(
                    origin=old_heads[str(owner)],
                    new_head=heads[str(owner)],
                    workers=worker_bindings,
                    phase=phase,
                    summary=owner_summary(merged),
                ),
            )
        rounds += 1
        summaries = {owner: owner_summary(state(head)[0]) for owner, head in heads.items()}
        decision = dict(
            round=number,
            phase=phase,
            heads=heads,
            summaries=summaries,
            pending_items=sum(row["pending_items"] for row in summaries.values()),
            coverage_errors=sum(row["errors"] for row in summaries.values()),
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
        )
        write_once(folder / "decision.json", decision)
        print(
            json.dumps(
                {
                    key: decision[key]
                    for key in ("round", "phase", "pending_items", "coverage_errors")
                }
            ),
            flush=True,
        )
        if all(heads[owner]["state_sha256"] == old_heads[owner]["state_sha256"] for owner in heads):
            reason = "no_checkpoint_progress"
            break
    inventory, errors = {}, []
    for owner, head in heads.items():
        value, _ = state(head)
        for name in (
            "sample_ids",
            "coco_ids",
            "media_hashes",
            "source_group_ids",
            "scopes",
            "scoped_identities",
        ):
            nested = name in {"scopes", "scoped_identities"}
            inventory.setdefault(name, set()).update(
                tuple(row) if nested else row for row in value[name]
            )
        errors.extend(
            dict(
                owner_index=int(owner),
                historical_error_ordinal=index,
                original_error_sha256=digest(row),
                **row,
            )
            for index, row in enumerate(value["errors"])
        )
    unresolved = unresolved_errors(
        errors, set(plan["preserved_resolved_error_sha256s"]), plan["seed_error_counts"]
    )
    put(
        root / "combined_identity_inventory.json",
        dict(
            source_sha=plan["source_sha"],
            heads=heads,
            identities={key: sorted(values) for key, values in inventory.items()},
            errors=errors,
            complete_historical_id_inventory=False,
            media_source_groups_certified=False,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
        ),
    )
    summaries = {owner: owner_summary(state(head)[0]) for owner, head in heads.items()}
    complete = epochs and all(row["traversal_complete"] for row in summaries.values())
    if complete:
        reason = "all_original_owners_traversed_and_current_time_verified"
    gate = dict(
        source_sha=plan["source_sha"],
        complete=complete,
        reason=reason,
        rounds_completed=rounds,
        initial_pending_items=8481,
        remaining_pending_items=sum(row["pending_items"] for row in summaries.values()),
        current_time_verification_epoch_performed=epochs,
        owners=summaries,
        heads=heads,
        historical_coverage_errors=len(errors),
        preserved_prior_resolutions=len(errors) - len(unresolved),
        net_unresolved_coverage_errors=len(unresolved),
        error_categories=dict(Counter(row.get("error", "unknown") for row in unresolved)),
        complete_declared_identity_field_scope=complete and not unresolved,
        global_historical_inventory_complete=False,
        complete_historical_id_inventory=False,
        global_media_source_group_inventory_complete=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        GPU_jobs_submitted=0,
        wall_seconds=time.monotonic() - started,
        inventory_sha256=sha(root / "combined_identity_inventory.json"),
    )
    write_once(root / "GATE_global_history_v36.json", gate)
    return gate


def closeout(root):
    """Afterany evidence analysis; incomplete gates never submit confirmation."""
    root = Path(root).resolve(strict=True)
    plan = load(root)
    folder = root / "review"
    folder.mkdir(exist_ok=False)
    inventory = root / "combined_identity_inventory.json"
    coverage_status = recover_coverage(root, folder, plan, inventory)
    media_module = importlib.import_module("evicastkv_media_recovery_v36")
    media = media_module.recover(
        global_inventory=inventory if inventory.is_file() else None,
        global_inventory_sha256=sha(inventory) if inventory.is_file() else None,
    )
    media_module.publish(folder / "media-gap.json", media)
    gate_path = root / "GATE_global_history_v36.json"
    gate = read(gate_path) if gate_path.is_file() else None
    proof_directory = root.parent / "qualified-proofs"
    kinds = ("correctness", "metric", "byte_matching", "baseline_fidelity", "history", "media")
    paths = {kind: proof_directory / f"{kind}.json" for kind in kinds}
    regate_directory = root.parent / "consumed-regate" / "analysis"
    for kind in kinds[:4]:
        actual = regate_directory / f"qualified_{kind}_proof.json"
        if actual.is_file():
            paths[kind] = actual
    missing = [kind for kind, path in paths.items() if not path.is_file()]
    review_root = plan.get("original_review_root")
    freeze = None
    frozen_path = Path(review_root) / "selected_config.json" if review_root else None
    if review_root and not any(kind in missing for kind in kinds[:4]) and not frozen_path.exists():
        review_module = importlib.import_module("evicastkv_original_complete_admission_v36")
        review_module.freeze_candidate(review_root, {kind: paths[kind] for kind in kinds[:4]})
    if not missing and review_root:
        review_module = importlib.import_module("evicastkv_original_complete_admission_v36")
        freeze = review_module.finalize(review_root, paths)
    result = dict(
        source_sha=plan["source_sha"],
        metadata_analysis_complete=True,
        global_history_gate=gate,
        additive_coverage_recovery=coverage_status,
        media_recovery_sha256=sha(folder / "media-gap.json"),
        original_default_review=review_root,
        missing_qualified_proofs=missing,
        freeze_decision=freeze,
        candidate_parameters_frozen=bool(frozen_path and frozen_path.is_file()),
        status="blocked_missing_or_negative_confirmation_prerequisites",
        media_source_groups_certified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        GPU_confirmation_jobs_submitted=0,
        negative_successor_selection_gates_unchanged=True,
        failed_jobs_or_missing_evidence_not_replaced_with_passes=True,
    )
    write_once(folder / "closeout.json", result)
    return result


def recover_coverage(root, folder, plan, inventory):
    """One bounded CPU proof wave; preserve unsupported and failed records."""
    module = importlib.import_module("evicastkv_coverage_resolution_v36")
    output = folder / "coverage"
    try:
        prepared = module.prepare(
            output,
            cpu_gate_root=plan["cpu_gate_root"],
            global_inventory=inventory if inventory.is_file() else None,
            global_inventory_sha256=sha(inventory) if inventory.is_file() else None,
        )
    except Exception as error:
        result = dict(
            complete=False,
            phase="prepare",
            error_type=type(error).__name__,
            fresh_confirmation_authorized=False,
            error_waivers=0,
        )
        write_once(folder / "coverage-infrastructure-error.json", result)
        return result
    logs = folder / "coverage-logs"
    logs.mkdir(exist_ok=False)

    def run(index):
        command = [
            "/scratch/djy8hg/env/mosaickv/bin/python",
            str(SOURCE / "mosaickv/scripts/evicastkv_coverage_resolution_v36.py"),
            "worker",
            "--root",
            str(output),
            "--index",
            str(index),
        ]
        with (logs / f"worker-{index}.log").open("x") as stream:
            try:
                done = subprocess.run(
                    command,
                    cwd=SOURCE,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=420,
                    check=False,
                )
                status = dict(returncode=done.returncode, timeout=False)
            except subprocess.TimeoutExpired:
                status = dict(returncode=None, timeout=True)
        receipt = dict(index=index, command=command, **status)
        write_once(logs / f"worker-{index}.json", receipt)
        return receipt

    with concurrent.futures.ThreadPoolExecutor(max_workers=SLOTS) as executor:
        statuses = list(executor.map(run, range(8)))
    try:
        gate = module.analyze(output)
        result = dict(
            prepared=prepared,
            workers=statuses,
            gate=gate,
            gate_sha256=sha(output / module.GATE),
            complete=not (gate["missing_shards"] or gate["pending_files"] or gate["blocked_files"]),
            fresh_confirmation_authorized=False,
        )
    except Exception as error:
        result = dict(
            complete=False,
            phase="analysis",
            error_type=type(error).__name__,
            workers=statuses,
            fresh_confirmation_authorized=False,
            error_waivers=0,
        )
    write_once(folder / "coverage-wave.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "controller", "worker", "closeout"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--cpu-gate-root")
    parser.add_argument("--original-review-root")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "register":
        result = register(args.root, args.cpu_gate_root, args.original_review_root)
    elif args.mode == "worker":
        result = worker(args.root, args.round, args.index)
    elif args.mode == "closeout":
        result = closeout(args.root)
    else:
        result = controller(args.root)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
