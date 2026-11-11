#!/usr/bin/env python3
"""One V11-state continuation; bypass only provably empty Parquet projection.

No original source/state/error is mutated and no unsupported schema is waived.
An empty selected Arrow schema cannot yield identities: materializing millions
of empty dictionaries merely inflated the old projected-record counter. Raw
file hashing and immutable signatures remain mandatory and resumable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import evicastkv_review_freshness_stream as scanner
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
SHARDS, SECONDS, LINE_BOUND = 8, 210, 8 * 2**20
ORIGINAL_PARQUET = scanner._parquet
IDENTITIES = (
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
)


def read(path, expected=None):
    return read_json(read_bytes(path, bound=128 * 2**20, expected=expected)[0])


def empty_projection_parquet(path, item, state, budget):
    """Same identity sets; explicit counter difference only for empty rows."""
    import pyarrow.parquet as pq

    scanner._stat_guard(path, item["signature"])
    parquet = pq.ParquetFile(path)
    columns = [field.name for field in parquet.schema_arrow if scanner._arrow_safe(field)]
    if columns:
        return ORIGINAL_PARQUET(path, item, state, budget)
    if "columns" not in item:
        item["columns"] = []
        unsafe = [
            f.name
            for f in parquet.schema_arrow
            if f.name in scanner.MEDIA_KEYS | scanner.ID_KEYS | scanner.SCOPE_KEYS
        ]
        if unsafe:
            scanner.error(
                state, path, "unprojectable nested/binary identity or media columns", columns=unsafe
            )
        scanner.error(state, path, "Parquet has no safely projectable identity metadata columns")
    elif item["columns"] != []:
        raise ValueError("saved Parquet projection differs from unchanged source schema")
    group, batch = item.get("row_group", 0), item.get("batch", 0)
    if not 0 <= group <= parquet.num_row_groups or type(batch) is not int or batch < 0:
        raise ValueError("invalid saved Parquet projection cursor")
    if "empty_projection_rows_not_enumerated" not in item:
        done = sum(parquet.metadata.row_group(i).num_rows for i in range(group))
        if group < parquet.num_row_groups:
            done += min(batch * 512, parquet.metadata.row_group(group).num_rows)
        item["empty_projection_rows_not_enumerated"] = parquet.metadata.num_rows - done
        schema = [(f.name, str(f.type)) for f in parquet.schema_arrow]
        state.setdefault("empty_parquet_projection", {})[str(path)] = dict(
            schema_sha256=digest(schema),
            safely_projected_columns=[],
            source_rows_from_footer=parquet.metadata.num_rows,
            legacy_rows_already_enumerated=done,
            rows_not_enumerated=item["empty_projection_rows_not_enumerated"],
            identity_effect="none_empty_projection_proved",
            original_schema_error_preserved=True,
        )
    item.update(row_group=parquet.num_row_groups, batch=0)
    offset, segments = item.get("hash_offset", 0), item.get("segments", [])
    cursor = 0
    for segment in segments:
        if segment["start"] != cursor or segment["end"] <= cursor:
            raise ValueError("saved source-hash segments overlap or have a gap")
        cursor = segment["end"]
    if cursor != offset or offset > item["signature"][2]:
        raise ValueError("saved Parquet hash cursor differs from complete segment prefix")
    start, hashed = offset, hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        while budget.available():
            raw = stream.read(scanner.CHUNK)
            if not raw:
                item["done"] = True
                break
            hashed.update(raw)
            offset += len(raw)
            budget.bytes += len(raw)
    item["hash_offset"] = offset
    if offset > start:
        item.setdefault("segments", []).append(
            dict(start=start, end=offset, sha256=hashed.hexdigest())
        )
    scanner._stat_guard(path, item["signature"])
    return item.get("done", False)


def prepare(root, *, cpu_gate_root, parent, parent_gate_filename=None):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch continuation root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(cpu_gate_root, revision)
    gates = ("GATE_freshness_frontier_fork.json", "GATE_freshness_empty_parquet_frontier.json")
    available = [name for name in gates if (parent / name).is_file()]
    gate_name = parent_gate_filename or (available[0] if len(available) == 1 else None)
    if gate_name not in gates:
        raise ValueError("one explicitly reviewed continuation parent gate required")
    parent_plan_path, gate_path = parent / "plan.json", parent / gate_name
    parent_plan, gate = read(parent_plan_path), read(gate_path)
    if (
        gate["missing_shards"]
        or gate["source_sha"] != parent_plan["source_sha"]
        or sha(Path(scanner.__file__)) != parent_plan["same_scanner_file_sha256"]
    ):
        raise ValueError("all parent states and unchanged scanner source required")
    if (
        gate_name == gates[1]
        and parent_plan.get("optimization") != "empty_safe_Parquet_columns_only"
    ):
        raise ValueError("continuation parent uses an unreviewed optimization policy")
    indexes = {}
    for name in gate["worker_bindings"]:
        path = Path(name).resolve(strict=True)
        match = re.fullmatch(r"shard-(\d+)\.json", Path(name).name)
        if match is None or path.parent != (parent / "workers").resolve():
            raise ValueError("only exact parent worker artifacts may define continued states")
        indexes[name] = int(match.group(1))
    if sorted(indexes.values()) != list(range(SHARDS)):
        raise ValueError("every parent shard index must be imported exactly once")
    bound = {str(parent_plan_path): sha(parent_plan_path), str(gate_path): sha(gate_path)}
    inventory_path = parent / "combined_identity_inventory.json"
    read(inventory_path, gate["identity_inventory_sha256"])
    bound[str(inventory_path)] = gate["identity_inventory_sha256"]
    imports = []
    for name, expected in sorted(gate["worker_bindings"].items()):
        saved = read(name, expected)
        index = indexes[name]
        if (
            saved["source_sha"] != parent_plan["source_sha"]
            or digest(saved["state"]) != saved["state_sha256"]
            or (gate_name == gates[1] and saved.get("index") != index)
        ):
            raise ValueError("V11 state/hash source differs")
        imports.append((index, saved["state"]))
        bound[name] = expected
    for index, state in imports:
        write_once(
            root / "imports" / f"shard-{index}.json", dict(state=state, state_sha256=digest(state))
        )
    plan = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        parent=str(parent),
        parent_source_sha=parent_plan["source_sha"],
        parent_gate_filename=gate_name,
        parent_state_generation=(
            "V11_finished_worker_states_not_CP005"
            if gate_name == gates[0]
            else "continued_finished_worker_states_no_restart"
        ),
        same_scanner_file_sha256=sha(Path(scanner.__file__)),
        parent_bindings=bound,
        import_bindings={
            str(root / "imports" / f"shard-{i}.json"): sha(root / "imports" / f"shard-{i}.json")
            for i in range(SHARDS)
        },
        roots=parent_plan["roots"],
        seconds_per_worker=SECONDS,
        maximum_bytes=2 * 2**30,
        MAX_LINE=LINE_BOUND,
        shard_count=SHARDS,
        segment_count=1,
        optimization="empty_safe_Parquet_columns_only",
        original_projected_record_counter_preserved=True,
        future_empty_rows_not_added_to_projected_records=True,
        original_errors_preserved=True,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_one_finished_state_segment",
        workers=SHARDS,
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in {**plan["parent_bindings"], **plan["import_bindings"]}.items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable V11 continuation evidence changed")
    return plan


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered one-CPU shard index required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    imported = read(root / "imports" / f"shard-{index}.json")
    state = copy.deepcopy(imported["state"])
    if digest(state) != imported["state_sha256"]:
        raise ValueError("imported exact V11 state digest differs")
    before = dict(
        files=state["processed_files"],
        records=state["projected_records"],
        frontier=len(state["frontier"]),
        errors=len(state["errors"]),
    )
    with (
        patch.object(scanner, "_parquet", empty_projection_parquet),
        patch.object(scanner, "MAX_LINE", plan["MAX_LINE"]),
    ):
        scanner.advance(
            state,
            tuple(Path(p) for p in plan["roots"]),
            shard_index=index,
            shard_count=SHARDS,
            budget=scanner.Budget(
                seconds=plan["seconds_per_worker"], maximum_bytes=plan["maximum_bytes"]
            ),
        )
    result = dict(
        source_sha=plan["source_sha"],
        index=index,
        state=state,
        state_sha256=digest(state),
        before=before,
        files_added=state["processed_files"] - before["files"],
        records_added=state["projected_records"] - before["records"],
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
    )
    path = root / "workers" / f"shard-{index}.json"
    write_once(path, result)
    return dict(
        index=index,
        frontier_complete=state["traversal_complete"],
        files_added=result["files_added"],
        worker_sha256=sha(path),
    )


def analyze(root, *, recovery_root=None, yaml_root=None):
    root = Path(root).resolve()
    plan = load(root)
    original = read(Path(plan["parent"]) / "combined_identity_inventory.json")
    merged = {
        key: set(tuple(v) if isinstance(v, list) else v for v in original["identities"][key])
        for key in IDENTITIES
    }
    missing, states, bound, branch_bindings = [], [], {}, {}
    branch_status = {}
    for branch, gate_name, label in (
        (recovery_root, "GATE_grouped_offset_recovery.json", "grouped_offset"),
        (yaml_root, "GATE_freshness_yaml_events.json", "YAML_event"),
    ):
        if branch is None:
            continue
        recovery_root = Path(branch).resolve()
        recovery_plan = read(recovery_root / "plan.json")
        recovery_gate = read(recovery_root / gate_name)
        if recovery_gate["source_sha"] != plan["source_sha"] or Path(
            recovery_plan["parent"]
        ).resolve() != Path(plan["parent"]):
            raise ValueError("metadata recovery branch does not share this source/parent")
        recovered = read(
            recovery_root / "combined_identity_inventory.json",
            recovery_gate["identity_inventory_sha256"],
        )
        for key in IDENTITIES:
            merged[key].update(
                tuple(v) if isinstance(v, list) else v for v in recovered["identities"][key]
            )
        for name in (
            "plan.json",
            gate_name,
            "combined_identity_inventory.json",
        ):
            path = recovery_root / name
            branch_bindings[str(path)] = sha(path)
        branch_status[label] = recovery_gate["status"]
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        result = read(path)
        if (
            result["source_sha"] != plan["source_sha"]
            or result["index"] != index
            or digest(result["state"]) != result["state_sha256"]
        ):
            raise ValueError("continued state source/hash/index differs")
        state = result["state"]
        states.append(result)
        bound[str(path)] = sha(path)
        for key in IDENTITIES:
            merged[key].update(tuple(v) if isinstance(v, list) else v for v in state[key])
    write_once(
        root / "combined_identity_inventory.json",
        dict(
            identities={k: sorted(v) for k, v in merged.items()},
            source_sha=plan["source_sha"],
            worker_bindings=bound,
            recovery_bindings=branch_bindings,
            complete_historical_id_inventory=False,
            cross_dataset_media_certified=False,
            fresh_confirmation_authorized=False,
        ),
    )
    report = dict(
        status="finite_frontier_segment_finished_not_freshness_certified",
        source_sha=plan["source_sha"],
        missing_shards=missing,
        frontier_exhausted=not missing and all(r["state"]["traversal_complete"] for r in states),
        newly_processed_files=sum(r["files_added"] for r in states),
        newly_projected_records=sum(r["records_added"] for r in states),
        pending_frontier_items=sum(len(r["state"]["frontier"]) for r in states),
        original_and_new_coverage_errors=sum(len(r["state"]["errors"]) for r in states),
        empty_projection_rows_not_enumerated=sum(
            x["rows_not_enumerated"]
            for r in states
            for x in r["state"].get("empty_parquet_projection", {}).values()
        ),
        worker_bindings=bound,
        identity_inventory_sha256=sha(root / "combined_identity_inventory.json"),
        grouped_offset_recovery_status=branch_status.get("grouped_offset"),
        YAML_event_recovery_status=branch_status.get("YAML_event"),
        recovery_bindings=branch_bindings,
        original_errors_preserved=True,
        error_waivers=0,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "GATE_freshness_empty_parquet_frontier.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-gate-filename")
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--yaml-root", type=Path)
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
        result = analyze(args.root, recovery_root=args.recovery_root, yaml_root=args.yaml_root)
    print(json.dumps(result))
