#!/usr/bin/env python3
"""Two finite CPU history workers; no GPU, sampling, or freshness promotion."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
from pathlib import Path

from mosaickv.evaluation import coco_history_subset as subset
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
GLOBAL_GATE = subset.PARENT.parent / "GATE_history_closure_v16.json"
GLOBAL_GATE_SHA = "08c7897ffa7ad45be4eb7bbed13b8e3607056ebd16b0168928b18346d160ba34"
DECISION_SHA = "9063ff8e19e6a551fc9aa72b2bb5d2edd535d63cf9d6b02111ff0c2794654f58"
INVENTORY_SHA = "368c4fd8414e62af7cc4389eafb834d6d32988d857c8d9c23bb7763792e21102"


def read(path, expected=None, *, bound=256 * 2**20):
    canary = importlib.import_module("evicastkv_freshness_producer_canary")
    return canary.read_json(canary.read_bytes(path, expected=expected, bound=bound)[0])


def modules():
    scanner = importlib.import_module("evicastkv_review_freshness_stream")
    empty = importlib.import_module("evicastkv_freshness_empty_parquet_continue")
    history = importlib.import_module("evicastkv_history_closure_v16")
    if any(
        not Path(module.__file__).resolve().is_relative_to(SOURCE)
        for module in (scanner, empty, history)
    ):
        raise ValueError("historical scanner imports must use the current immutable source")
    return scanner, empty, history


def empirical_controls(whole, coco, *, whole_producer_sha, coco_producer_sha):
    """These are engineering admission gates, never quality/freshness gates."""
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in (whole_producer_sha, coco_producer_sha)
    ):
        raise ValueError("both empirical gate producer SHAs must be explicitly pinned")
    if (
        whole.get("source_sha") != whole_producer_sha
        or whole.get("complete") is not True
        or whole.get("source_fidelity_128_pass") is not True
        or whole.get("whole_request_KV_byte_match_pass") is not True
        or whole.get("whole_request_KV_peak_certified") is not True
        or whole.get("failures") != []
    ):
        raise ValueError("whole-request fidelity/storage admission gate is incomplete or negative")
    if (
        coco.get("source_sha") != coco_producer_sha
        or coco.get("complete") is not True
        or coco.get("corrected_boundary_128_pass") is not True
        or coco.get("failures") != []
        or coco.get("source_scope") != "FP16_GQA_COCO_FullKV_evaluation_only_boundary_control"
    ):
        raise ValueError("COCO numerical boundary admission gate is incomplete or negative")
    return dict(
        whole_request_engineering_gate_passed=True,
        COCO_boundary_gate_passed=True,
        fresh_confirmation_authorized=False,
    )


def register(
    root,
    *,
    execute=False,
    cpu_gate_root=None,
    whole_gate=None,
    coco_gate=None,
    whole_producer_sha=None,
    coco_producer_sha=None,
):
    revision = clean_source(SOURCE)
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve(strict=True)):
        raise ValueError("new immutable scratch subset root required")
    if type(execute) is not bool:
        raise ValueError("explicit execution scope required")
    scanner, _empty, history = modules()
    source_bridge = history.bridge()
    bindings = {
        str(GLOBAL_GATE): GLOBAL_GATE_SHA,
        str(subset.PARENT / "decision.json"): DECISION_SHA,
        str(subset.PARENT / "combined_identity_inventory.json"): INVENTORY_SHA,
    }
    gate = read(GLOBAL_GATE, GLOBAL_GATE_SHA)
    decision = read(subset.PARENT / "decision.json", DECISION_SHA)
    if (
        gate.get("pending_frontier_items") != subset.GLOBAL_PARENT_FRONTIER
        or gate.get("frontier_exhausted") is not False
        or decision.get("identity_inventory_sha256") != INVENTORY_SHA
    ):
        raise ValueError("preserved parent global incomplete frontier differs")
    selections, deferred_count = {}, 0
    for index, spec in subset.SPECS.items():
        path = subset.PARENT / "workers" / f"shard-{index}.json"
        parent = read(path, spec["worker_sha256"])
        split = subset.split_frontier(parent, index)
        selections[str(index)] = split
        deferred_count += len(split["deferred"])
        bindings[str(path)] = spec["worker_sha256"]
    passed = None
    if execute:
        if any(value is None for value in (cpu_gate_root, whole_gate, coco_gate)):
            raise ValueError(
                "CPU, whole-request, and COCO gate artifacts are required before scanning"
            )
        followups = importlib.import_module("evicastkv_review_followups")
        cpu = followups.cpu_gate(Path(cpu_gate_root), revision)
        whole_path, coco_path = (
            Path(whole_gate).resolve(strict=True),
            Path(coco_gate).resolve(strict=True),
        )
        passed = empirical_controls(
            read(whole_path),
            read(coco_path),
            whole_producer_sha=whole_producer_sha,
            coco_producer_sha=coco_producer_sha,
        )
        for path in (whole_path, coco_path):
            bindings[str(path)] = sha(path)
        for name in ("cpu-tests.json", "cpu-tests.xml", "cpu-tests.log"):
            path = Path(cpu_gate_root) / "full-cpu" / name
            bindings[str(path)] = sha(path)
    else:
        cpu = None
    for module in (scanner, _empty, history, subset):
        path = Path(module.__file__).resolve()
        bindings[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            source_sha=revision,
            parent_producer_sha=subset.PARENT_SOURCE,
            execute=execute,
            scope="exact_521_reviewed_COCO_pending_directories_only",
            cpu_gate_root=str(cpu_gate_root) if cpu_gate_root else None,
            cpu_gate=cpu,
            whole_gate=str(whole_gate) if whole_gate else None,
            coco_gate=str(coco_gate) if coco_gate else None,
            whole_producer_sha=whole_producer_sha,
            coco_producer_sha=coco_producer_sha,
            empirical_admission=passed,
            source_bridge=source_bridge,
            bindings=bindings,
            original_owner_indices=[1, 7],
            original_shard_count=8,
            selected_initial_directories=521,
            original_global_pending=9002,
            unselected_global_pending_preserved=8481,
            deferred_in_selected_parent_states=deferred_count,
            other_original_workers_bound_by_parent_decision=decision["worker_bindings"],
            roots=[str(path.resolve(strict=True)) for path in scanner.ROOTS],
            maximum_worker_seconds=subset.SECONDS,
            maximum_worker_bytes=subset.MAX_BYTES,
            automatic_resubmission=False,
            maximum_GPU_workers=0,
            metadata_only=True,
            answers_or_predictions_exported=False,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
        ),
    )
    for index, split in selections.items():
        write_once(root / "queues" / f"shard-{index}.json", split)
    return dict(registered=True, execution_enabled=execute, GPU_jobs=0, jobs_submitted=0)


def load(root):
    root = Path(root).resolve(strict=True)
    plan = read(root / "plan.json", bound=4 * 2**20)
    clean_source(SOURCE, plan["source_sha"])
    if (
        plan.get("execute") is not True
        or plan.get("metadata_only") is not True
        or plan.get("original_owner_indices") != [1, 7]
        or plan.get("original_shard_count") != 8
        or plan.get("maximum_worker_seconds") != subset.SECONDS
        or plan.get("maximum_worker_bytes") != subset.MAX_BYTES
        or plan.get("fresh_confirmation_authorized") is not False
        or plan.get("materialization_authorized") is not False
        or plan.get("maximum_GPU_workers") != 0
    ):
        raise ValueError("only explicitly admitted bounded historical CPU scope may execute")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable subset input/source binding changed")
    empirical_controls(
        read(plan["whole_gate"]),
        read(plan["coco_gate"]),
        whole_producer_sha=plan["whole_producer_sha"],
        coco_producer_sha=plan["coco_producer_sha"],
    )
    scanner, _empty, history = modules()
    if plan["roots"] != [str(path.resolve(strict=True)) for path in scanner.ROOTS] or (
        plan["source_bridge"] != history.bridge()
    ):
        raise ValueError("original root ownership or unchanged scanner bridge differs")
    return plan


def worker(root, index):
    root = Path(root).resolve(strict=True)
    plan = load(root)
    if type(index) is not int or index not in subset.SPECS:
        raise ValueError("do not renumber original shard indices 1 and 7")
    write_once(
        root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID"), index=index)
    )
    parent_path = subset.PARENT / "workers" / f"shard-{index}.json"
    parent = read(parent_path, subset.SPECS[index]["worker_sha256"])
    selected = read(root / "queues" / f"shard-{index}.json", bound=8 * 2**20)
    if selected != subset.split_frontier(parent, index):
        raise ValueError("registered selected/deferred queues differ from their immutable parent")
    scanner, empty, history = modules()
    state, summary = subset.advance_subset(
        parent,
        selected,
        tuple(Path(p) for p in plan["roots"]),
        scanner=scanner,
        empty_projection_parquet=empty.empty_projection_parquet,
    )
    result = dict(
        source_sha=plan["source_sha"],
        parent_source_sha=subset.PARENT_SOURCE,
        index=index,
        state=state,
        state_sha256=digest(state),
        summary=summary,
        parent_worker_sha256=subset.SPECS[index]["worker_sha256"],
        fresh_confirmation_authorized=False,
    )
    path = root / "workers" / f"shard-{index}.json"
    history.bounded_write(path, result, 256 * 2**20, compact=True)
    return dict(index=index, worker_sha256=sha(path), summary=summary)


def analyze(root):
    root = Path(root).resolve(strict=True)
    plan = load(root)
    records, missing, bindings = [], [], {}
    for index in (1, 7):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        record = read(path)
        if (
            record.get("source_sha") != plan["source_sha"]
            or record.get("index") != index
            or record.get("parent_worker_sha256") != subset.SPECS[index]["worker_sha256"]
            or record.get("state_sha256") != digest(record["state"])
            or record["state"].get("traversal_complete") is not False
        ):
            raise ValueError("subset worker state/producer or global completeness label differs")
        parent = read(
            subset.PARENT / "workers" / f"shard-{index}.json", subset.SPECS[index]["worker_sha256"]
        )
        selected = subset.split_frontier(parent, index)
        subset.verify_continuation(parent, selected, record["state"], record["summary"])
        records.append(record)
        bindings[str(path)] = sha(path)
    old = read(subset.PARENT / "combined_identity_inventory.json", INVENTORY_SHA)
    identities = copy.deepcopy(old["identities"])
    for name in identities:
        nested = name in {"scopes", "scoped_identities"}
        values = {tuple(row) if nested else row for row in identities[name]}
        for record in records:
            values.update(tuple(row) if nested else row for row in record["state"][name])
        identities[name] = sorted(values)
    output = root / "combined_identity_inventory.json"
    write_once(
        output,
        dict(
            source_sha=plan["source_sha"],
            parent_inventory_sha256=INVENTORY_SHA,
            original_parent_source_sha=subset.PARENT_SOURCE,
            identities=identities,
            complete_historical_id_inventory=False,
            fresh_confirmation_authorized=False,
            subset_worker_bindings=bindings,
        ),
    )
    report = dict(
        status="finite_COCO_subset_continuation_not_global_freshness",
        source_sha=plan["source_sha"],
        missing_original_worker_indices=missing,
        selected_initial_directories=521,
        unselected_original_global_pending_preserved=8481,
        summaries=[record["summary"] for record in records],
        subset_frontier_exhausted=not missing
        and all(r["summary"]["subset_frontier_exhausted"] for r in records),
        new_coverage_errors=sum(len(r["summary"]["new_errors"]) for r in records),
        identity_inventory_sha256=sha(output),
        worker_bindings=bindings,
        original_errors_preserved=True,
        global_historical_inventory_complete=False,
        globally_fresh=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        GPU_jobs_submitted=0,
    )
    write_once(root / "GATE_COCO_history_subset.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("register", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--cpu-gate-root", type=Path)
    p.add_argument("--whole-gate", type=Path)
    p.add_argument("--coco-gate", type=Path)
    p.add_argument("--whole-producer-sha")
    p.add_argument("--coco-producer-sha")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "register":
        result = register(
            a.root,
            execute=a.execute,
            cpu_gate_root=a.cpu_gate_root,
            whole_gate=a.whole_gate,
            coco_gate=a.coco_gate,
            whole_producer_sha=a.whole_producer_sha,
            coco_producer_sha=a.coco_producer_sha,
        )
    elif a.mode == "worker":
        result = worker(a.root, a.index)
    else:
        result = analyze(a.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
