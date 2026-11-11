#!/usr/bin/env python3
"""One finite, source-transactional recovery of registered historical offsets.

The original states/errors remain immutable. This consumes only their saved
offset queue: no tree discovery, answer decoding, new cohort, or retry loop.
Each source is verified once before and once after its registered records.
No projected identity escapes a failed/interrupted file transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import evicastkv_freshness_line_recovery as lines
from evicastkv_freshness_frontier_fork import canonical_bindings
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
SHARDS, SECONDS = 8, 240
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


def empty_identities():
    return {name: [] for name in IDENTITIES} | {"projected_records": 0}


def project_group(rows, binding, roots, deadline):
    """Atomic source group, identical lexical projection and collection rules."""
    if not rows:
        raise ValueError("nonempty registered offset group required")
    path = lines.canonical(rows[0]["path"], roots)
    offsets = [row["offset"] for row in rows]
    if any(type(offset) is not int or offset < 0 for offset in offsets) or offsets != sorted(
        set(offsets)
    ):
        raise ValueError("unique sorted nonnegative registered offsets required")
    if any(lines.canonical(row["path"], roots) != path for row in rows):
        raise ValueError("a transaction cannot combine different source files")
    before = lines.hash_source(path, binding, deadline)
    observations, projected_records = [], []
    with lines._descriptor(path, binding["signature"]) as stream:
        for row in rows:
            lines.check_deadline(deadline)
            offset = row["offset"]
            if offset >= binding["signature"][2]:
                raise ValueError("registered offset outside source")
            if offset:
                stream.seek(offset - 1)
                if stream.read(1) != b"\n":
                    raise ValueError("registered offset is not a JSONL record boundary")
            stream.seek(offset)
            raw = stream.readline(lines.MAX_BYTES + 1)
            if not raw.strip() or len(raw) > lines.MAX_BYTES:
                raise ValueError("registered record is empty or exceeds the existing strict bound")
            statistics = {}
            projected = lines.identity_projection(
                raw, budget=SimpleNamespace(deadline=deadline), statistics=statistics
            )
            identities = empty_identities()
            lines.collect(projected, identities, path)
            projected_records.append(dict(path=str(path), offset=offset, identities=identities))
            observations.append(
                dict(
                    path=str(path),
                    offset=offset,
                    original_error_sha256=row["original_error_sha256"],
                    line_sha256=hashlib.sha256(raw).hexdigest(),
                    line_bytes=len(raw),
                    projected_metadata_sha256=digest(projected),
                    statistics=statistics,
                )
            )
    lines.check_deadline(deadline)
    after = lines.hash_source(path, binding, deadline)
    if after != before or any(lines.canonical(row["path"], roots) != path for row in rows):
        raise ValueError("source changed during grouped recovery")
    # Local metadata becomes observable only after the final whole-file hash.
    return dict(
        path=str(path),
        status="verified_group_transaction",
        records=projected_records,
        record_proofs=observations,
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
        signature=binding["signature"],
        raw_or_answer_values_exported=False,
    )


def prepare(root, *, cpu_gate_root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch grouped-recovery root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(cpu_gate_root, revision)
    original_plan_path = parent / "plan.json"
    original = read(original_plan_path)
    bound = {str(original_plan_path): sha(original_plan_path)}
    for name, expected in {**original["bindings"], **original["auxiliary_bindings"]}.items():
        read_bytes(name, bound=128 * 2**20, expected=expected)
        bound[name] = expected
    auxiliary = canonical_bindings(original["auxiliary_bindings"])
    for path in (parent / "oversize_queue.json", parent / "recovery_source_bindings.json"):
        if auxiliary.get(str(path)) != sha(path):
            raise ValueError(
                "parent registration does not bind its original offset queue/source inventory"
            )
        bound[str(path)] = auxiliary[str(path)]
    gate_path = parent / "GATE_freshness_frontier_fork.json"
    gate = read(gate_path)
    if gate["source_sha"] != original["source_sha"] or gate["missing_shards"]:
        raise ValueError("all registered parent worker outputs and their matching gate required")
    bound[str(gate_path)] = sha(gate_path)
    inventory_path = parent / "combined_identity_inventory.json"
    read(inventory_path, gate["identity_inventory_sha256"])
    bound[str(inventory_path)] = gate["identity_inventory_sha256"]
    if sorted(Path(name).name for name in gate["worker_bindings"]) != [
        f"shard-{index:03d}.json" for index in range(SHARDS)
    ]:
        raise ValueError("all eight registered parent workers required")
    recovered = set()
    for name, expected in gate["worker_bindings"].items():
        worker = read(name, expected)
        if worker["source_sha"] != original["source_sha"]:
            raise ValueError("parent worker source differs")
        bound[name] = expected
        proofs = {
            (r["path"], r["offset"]): r
            for r in worker["recovery_observations"]
            if r["status"] == "projected_registered_metadata_only"
        }
        for record in worker["recovered_metadata"]:
            key = record["path"], record["offset"]
            proof = proofs.get(key)
            if (
                proof is None
                or proof["whole_file_sha256_before"] != proof["whole_file_sha256_after"]
                or not proof.get("line_sha256")
            ):
                raise ValueError("previous offset recovery lacks its complete source proof")
            if key in recovered:
                raise ValueError("previous recovery offset duplicated across workers")
            recovered.add(key)
    original_rows = read(parent / "oversize_queue.json")
    queue_keys = {(r["path"], r["offset"]) for r in original_rows}
    if len(queue_keys) != len(original_rows) or not recovered <= queue_keys:
        raise ValueError("original offset queue or previous recovery identities differ")
    grouped = defaultdict(list)
    for row in original_rows:
        if (row["path"], row["offset"]) not in recovered:
            grouped[row["path"]].append(row)
    groups = [
        dict(
            path=name,
            rows=sorted(rows, key=lambda r: r["offset"]),
            index=int(hashlib.sha256(name.encode()).hexdigest(), 16) % SHARDS,
        )
        for name, rows in sorted(grouped.items())
    ]
    bindings_path = parent / "recovery_source_bindings.json"
    bindings = read(bindings_path)
    plan = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        parent_source_sha=original["source_sha"],
        parent=str(parent),
        bindings=bound,
        original_offsets=len(original_rows),
        previously_recovered_offsets=len(recovered),
        pending_offsets=sum(len(g["rows"]) for g in groups),
        groups=groups,
        roots=original["roots"],
        source_bindings=bindings,
        unchanged_projector_sha256=sha(Path(lines.identity_projection.__code__.co_filename)),
        unchanged_collector_sha256=sha(Path(lines.collect.__code__.co_filename)),
        seconds_per_worker=SECONDS,
        shard_count=SHARDS,
        segment_count=1,
        automatic_resubmission=False,
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_bounded_grouped_offset_recovery",
        groups=len(groups),
        pending_offsets=plan["pending_offsets"],
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable grouped-recovery parent evidence changed")
    return plan


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered CPU shard index required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["seconds_per_worker"]
    groups = [g for g in plan["groups"] if g["index"] == index]
    committed, pending, blocked = [], [], []
    for number, group in enumerate(groups):
        try:
            lines.check_deadline(deadline)
            binding = plan["source_bindings"].get(group["path"])
            if binding is None:
                raise ValueError("complete original source binding unavailable")
            committed.append(
                project_group(group["rows"], binding, [Path(r) for r in plan["roots"]], deadline)
            )
        except TimeoutError:
            pending.extend(groups[number:])
            break
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            blocked.append(dict(group=group, error=f"{type(exc).__name__}: {exc}"))
    output = dict(
        source_sha=plan["source_sha"],
        parent_source_sha=plan["parent_source_sha"],
        index=index,
        committed_groups=committed,
        pending_groups=pending,
        blocked_groups=blocked,
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
    )
    path = root / "workers" / f"shard-{index}.json"
    write_once(path, output)
    return dict(
        index=index,
        committed_groups=len(committed),
        pending_groups=len(pending),
        blocked_groups=len(blocked),
        worker_sha256=sha(path),
    )


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    inventory = read(Path(plan["parent"]) / "combined_identity_inventory.json")
    merged = {
        key: set(tuple(v) if isinstance(v, list) else v for v in inventory["identities"][key])
        for key in IDENTITIES
    }
    pending, blocked, recovered, missing, bindings = [], [], [], [], {}
    expected_groups = {g["path"]: g for g in plan["groups"]}
    seen = set()
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        result = read(path)
        if result["source_sha"] != plan["source_sha"] or result["index"] != index:
            raise ValueError("grouped worker source/index differs")
        bindings[str(path)] = sha(path)
        assigned = {name for name, group in expected_groups.items() if group["index"] == index}
        classified = [group["path"] for group in result["committed_groups"]]
        for group in [*result["pending_groups"], *(r["group"] for r in result["blocked_groups"])]:
            name = group["path"]
            if name not in expected_groups or group != expected_groups[name]:
                raise ValueError("unresolved group differs from its registered offsets")
            classified.append(name)
        if len(classified) != len(set(classified)) or set(classified) != assigned:
            raise ValueError("registered worker groups dropped or multiply classified")
        pending.extend(result["pending_groups"])
        blocked.extend(result["blocked_groups"])
        for group in result["committed_groups"]:
            name = group["path"]
            if (
                name in seen
                or name not in expected_groups
                or expected_groups[name]["index"] != index
                or group["whole_file_sha256_before"] != group["whole_file_sha256_after"]
            ):
                raise ValueError("grouped transaction duplicates or differs from registered source")
            seen.add(name)
            expected = {r["offset"] for r in expected_groups[name]["rows"]}
            if (
                len(group["records"]) != len(expected)
                or {r["offset"] for r in group["records"]} != expected
                or any(r["path"] != name for r in group["records"])
                or group["signature"] != plan["source_bindings"][name]["signature"]
            ):
                raise ValueError("committed source does not cover its exact registered offsets")
            proofs = {r["offset"]: r for r in group["record_proofs"]}
            if (
                len(proofs) != len(expected)
                or set(proofs) != expected
                or any(
                    proofs[r["offset"]]["original_error_sha256"] != r["original_error_sha256"]
                    for r in expected_groups[name]["rows"]
                )
            ):
                raise ValueError("registered record proof set differs")
            recovered.extend(group["records"])
            for record in group["records"]:
                for key in IDENTITIES:
                    merged[key].update(
                        tuple(v) if isinstance(v, list) else v for v in record["identities"][key]
                    )
    unresolved = {g["path"] for g in pending} | {g["group"]["path"] for g in blocked}
    if seen & unresolved or (not missing and seen | unresolved != set(expected_groups)):
        raise ValueError("registered groups dropped or multiply classified")
    combined = dict(
        identities={k: sorted(v) for k, v in merged.items()},
        source_sha=plan["source_sha"],
        worker_bindings=bindings,
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "combined_identity_inventory.json", combined)
    report = dict(
        status="bounded_grouped_recovery_complete"
        if not (missing or pending or blocked)
        else "bounded_grouped_recovery_incomplete",
        source_sha=plan["source_sha"],
        original_offsets=plan["original_offsets"],
        previously_recovered_offsets=plan["previously_recovered_offsets"],
        newly_recovered_offsets=len(recovered),
        pending_offsets=sum(len(g["rows"]) for g in pending),
        blocked_offsets=sum(len(g["group"]["rows"]) for g in blocked),
        missing_worker_offsets=sum(len(g["rows"]) for g in plan["groups"] if g["index"] in missing),
        missing_shards=missing,
        worker_bindings=bindings,
        identity_inventory_sha256=sha(root / "combined_identity_inventory.json"),
        original_errors_preserved=True,
        error_waivers=0,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    if (
        sum(
            report[key]
            for key in (
                "previously_recovered_offsets",
                "newly_recovered_offsets",
                "pending_offsets",
                "blocked_offsets",
                "missing_worker_offsets",
            )
        )
        != report["original_offsets"]
    ):
        raise ValueError("original offset denominator is not conserved")
    write_once(root / "GATE_grouped_offset_recovery.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.root, cpu_gate_root=args.cpu_gate_root, parent=args.parent)
    elif args.mode == "worker":
        result = worker(args.root, args.index)
    else:
        result = analyze(args.root)
    print(json.dumps(result))
