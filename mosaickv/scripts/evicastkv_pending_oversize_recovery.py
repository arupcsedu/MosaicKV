#!/usr/bin/env python3
"""One finite recovery of the 74 still-pending, already registered JSONL offsets.

No discovery, rescan, historical-error deletion, cohort selection, or freshness
authorization. The unchanged grouped projector releases metadata only after its
whole-file before/after transaction succeeds. Earlier successful bodies are not
read again; their hash-bound proofs and identities are imported explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import evicastkv_freshness_grouped_offsets as grouped
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
GROUPED = Path("/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1/grouped")
HISTORY = Path("/scratch/djy8hg/workdir/fox-review-v16/run-20261111-2/history")
GROUPED_SOURCE = "76d4ced1a10c6746bad73f564e84210522355ef9"
FORK_SOURCE = "1aab16c63eee2f063194e6db8c9b3b5af466c3a1"
HISTORY_SOURCE = "84900a6a402e97c7d58499a722c159cfc86e46c7"
OUTPUT_BASE = Path("/scratch/djy8hg")
SHARDS, SECONDS = 2, 240
EXPECTED_GROUPS, EXPECTED_PENDING, EXPECTED_PREVIOUS, EXPECTED_ORIGINAL = 15, 74, 1387, 1461
MAX_TOTAL_SOURCE_BYTES, MAX_METADATA_BYTES = 256 * 2**20, 128 * 2**20
GATE = "GATE_pending_oversize_recovery.json"


def read(path, expected=None):
    return read_json(read_bytes(Path(path), bound=MAX_METADATA_BYTES, expected=expected)[0])


def key(row):
    return row["path"], row["offset"]


def bound_read(path, bindings, expected=None):
    path = Path(path).resolve(strict=True)
    value = read(path, expected)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError("parent metadata changed after stable read")
    old = bindings.setdefault(str(path), actual)
    if old != actual:
        raise ValueError("conflicting canonical parent aliases")
    return value


def checked_proof(proof, row, before, after):
    if (
        key(proof) != key(row)
        or proof["original_error_sha256"] != row["original_error_sha256"]
        or before != after
        or not isinstance(before, str)
        or len(before) != 64
        or not isinstance(proof.get("line_sha256"), str)
        or len(proof["line_sha256"]) != 64
        or not isinstance(proof.get("projected_metadata_sha256"), str)
    ):
        raise ValueError("offset lacks complete immutable source and lexical-projection proof")
    return dict(
        proof,
        whole_file_sha256_before=before,
        whole_file_sha256_after=after,
    )


def inspect_parents(grouped_root=GROUPED, history_root=HISTORY):
    """Read finite bound metadata only; never opens historical result bodies."""
    grouped_root, history_root = Path(grouped_root).resolve(), Path(history_root).resolve()
    bindings = {}
    plan = bound_read(grouped_root / "plan.json", bindings)
    gate = bound_read(grouped_root / "GATE_grouped_offset_recovery.json", bindings)
    if (
        gate["source_sha"] != plan["source_sha"]
        or plan["source_sha"] != GROUPED_SOURCE
        or gate["missing_shards"]
    ):
        raise ValueError("all eight original grouped workers and matching gate required")
    for name, expected in plan["bindings"].items():
        path = Path(name).resolve(strict=True)
        read_bytes(path, bound=MAX_METADATA_BYTES, expected=expected)
        if bindings.setdefault(str(path), expected) != expected:
            raise ValueError("conflicting canonical parent aliases")
    scanner = Path(grouped.lines.identity_projection.__code__.co_filename)
    if plan["unchanged_projector_sha256"] != sha(scanner) or plan[
        "unchanged_collector_sha256"
    ] != sha(scanner):
        raise ValueError("registered lexical projection or collection semantics changed")
    fork = Path(plan["parent"]).resolve(strict=True)
    queue = bound_read(fork / "oversize_queue.json", bindings)
    rows = {key(row): row for row in queue}
    if len(rows) != len(queue) or len(queue) != EXPECTED_ORIGINAL:
        raise ValueError("original offset queue denominator differs")
    source_bindings = bound_read(fork / "recovery_source_bindings.json", bindings)
    if source_bindings != plan["source_bindings"]:
        raise ValueError("original grouped source bindings differ")
    fork_gate = bound_read(fork / "GATE_freshness_frontier_fork.json", bindings)
    if fork_gate["missing_shards"] or fork_gate["source_sha"] != FORK_SOURCE:
        raise ValueError("original fork workers incomplete")
    prior = {}
    if len(fork_gate["worker_bindings"]) != 8 or len(gate["worker_bindings"]) != 8:
        raise ValueError("exactly eight original workers per producer required")
    for name, expected in fork_gate["worker_bindings"].items():
        result = bound_read(name, bindings, expected)
        if result["source_sha"] != FORK_SOURCE:
            raise ValueError("original fork worker source differs")
        observations = {
            key(r): r
            for r in result["recovery_observations"]
            if r["status"] == "projected_registered_metadata_only"
        }
        for record in result["recovered_metadata"]:
            identifier = key(record)
            proof = observations[identifier]
            if identifier in prior:
                raise ValueError("prior offset proof duplicated")
            prior[identifier] = checked_proof(
                proof,
                rows[identifier],
                proof["whole_file_sha256_before"],
                proof["whole_file_sha256_after"],
            )
    registered = {g["path"]: g for g in plan["groups"]}
    pending, classified = [], set()
    for name, expected in gate["worker_bindings"].items():
        result = bound_read(name, bindings, expected)
        if result["source_sha"] != plan["source_sha"] or result["blocked_groups"]:
            raise ValueError("unexpected original grouped worker source or blocked records")
        for group in result["pending_groups"]:
            if registered.get(group["path"]) != group:
                raise ValueError("pending group differs from original registration")
            pending.append(group)
        for group in result["committed_groups"]:
            expected_rows = registered[group["path"]]["rows"]
            proofs = {key(p): p for p in group["record_proofs"]}
            if len(proofs) != len(expected_rows) or set(proofs) != {key(r) for r in expected_rows}:
                raise ValueError("prior grouped proof set differs")
            for row in expected_rows:
                identifier = key(row)
                if identifier in prior:
                    raise ValueError("prior recovery proofs overlap")
                prior[identifier] = checked_proof(
                    proofs[identifier],
                    row,
                    group["whole_file_sha256_before"],
                    group["whole_file_sha256_after"],
                )
        names = [g["path"] for g in result["pending_groups"] + result["committed_groups"]]
        if len(names) != len(set(names)) or classified.intersection(names):
            raise ValueError("original groups multiply classified")
        classified.update(names)
    pending_keys = [key(row) for g in pending for row in g["rows"]]
    if (
        classified != set(registered)
        or len(pending) != EXPECTED_GROUPS
        or len(pending_keys) != EXPECTED_PENDING
        or len(set(pending_keys)) != len(pending_keys)
        or len(prior) != EXPECTED_PREVIOUS
        or set(prior) & set(pending_keys)
        or set(prior) | set(pending_keys) != set(rows)
        or gate["previously_recovered_offsets"] + gate["newly_recovered_offsets"] != len(prior)
        or gate["pending_offsets"] != len(pending_keys)
    ):
        raise ValueError("pending/prior/original offset partition differs")
    historical = bound_read(history_root / "GATE_history_closure_v16.json", bindings)
    if (
        historical["last_segment"] != 4
        or historical["source_sha"] != HISTORY_SOURCE
        or historical["missing_shards"]
        or historical["stop_reason"] != "maximum_four_segments"
        or historical["fresh_confirmation_authorized"]
        or not historical["original_errors_preserved"]
        or historical["roots_restarted"]
    ):
        raise ValueError("actual finite four-segment history closure required")
    last = history_root / "segment-4"
    decision = bound_read(last / "decision.json", bindings, historical["last_decision_sha256"])
    if decision["source_sha"] != historical["source_sha"]:
        raise ValueError("final history producer mismatch")
    errors = bound_read(last / "preserved_errors.json", bindings, decision["errors_sha256"])
    error_hashes = {digest(error) for error in errors}
    if any(rows[k]["original_error_sha256"] not in error_hashes for k in rows):
        raise ValueError("an original offset error is absent from preserved final history")
    inventory_paths = [
        last / "combined_identity_inventory.json",
        grouped_root / "combined_identity_inventory.json",
    ]
    for path, expected in zip(
        inventory_paths,
        [historical["identity_inventory_sha256"], gate["identity_inventory_sha256"]],
        strict=True,
    ):
        bound_read(path, bindings, expected)
    costs = {}
    for group in pending:
        path = grouped.lines.canonical(group["path"], [Path(r) for r in plan["roots"]])
        binding = source_bindings[str(path)]
        if grouped.lines.signature(path) != binding["signature"]:
            raise ValueError("pending source signature changed")
        if grouped.digest(binding["segments"]) != binding["segmented_content_root"]:
            raise ValueError("pending source segment binding differs")
        costs[str(path)] = binding["signature"][2]
    if sum(costs.values()) > MAX_TOTAL_SOURCE_BYTES:
        raise ValueError("registered finite total source-byte budget exceeded")
    load = [0] * SHARDS
    assignments = []
    for group in sorted(pending, key=lambda g: (-costs[g["path"]], g["path"])):
        index = min(range(SHARDS), key=lambda i: (load[i], i))
        assignments.append(dict(group, index=index, original_index=group["index"]))
        load[index] += costs[group["path"]]
    return dict(
        bindings=bindings,
        groups=assignments,
        source_bindings={p: source_bindings[p] for p in costs},
        roots=plan["roots"],
        imported_proofs=[prior[k] for k in sorted(prior)],
        inventory_paths=list(map(str, inventory_paths)),
        preserved_errors_path=str(last / "preserved_errors.json"),
        historical_gate=str(history_root / "GATE_history_closure_v16.json"),
        pending_frontier_items=historical["pending_frontier_items"],
        original_and_new_coverage_errors=len(errors),
        source_bytes=sum(costs.values()),
        worker_source_bytes=load,
        original_offsets=len(rows),
        previously_recovered_offsets=len(prior),
    )


def prepare(root, *, cpu_gate_root):
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(OUTPUT_BASE.resolve()):
        raise ValueError("new write-once scratch output root required")
    revision = clean_source(SOURCE)
    cpu_gate(Path(cpu_gate_root), revision)
    evidence = inspect_parents(GROUPED, HISTORY)
    code = {
        str(Path(p).resolve()): sha(Path(p))
        for p in (
            __file__,
            grouped.__file__,
            grouped.lines.__file__,
            grouped.lines.collect.__code__.co_filename,
        )
    }
    plan = dict(
        **evidence,
        source_sha=revision,
        cpu_gate_root=str(Path(cpu_gate_root).resolve()),
        code_bindings=code,
        shard_count=SHARDS,
        seconds_per_worker=SECONDS,
        original_errors_preserved=True,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        workers=SHARDS,
        pending_offsets=EXPECTED_PENDING,
        pending_files=EXPECTED_GROUPS,
        source_bytes=evidence["source_bytes"],
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in {**plan["bindings"], **plan["code_bindings"]}.items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable parent or current policy binding changed")
    return plan


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < plan["shard_count"]:
        raise ValueError("registered worker index required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["seconds_per_worker"]
    groups = [g for g in plan["groups"] if g["index"] == index]
    committed, pending, blocked = [], [], []
    for number, group in enumerate(groups):
        try:
            grouped.lines.check_deadline(deadline)
            committed.append(
                grouped.project_group(
                    group["rows"],
                    plan["source_bindings"][group["path"]],
                    [Path(r) for r in plan["roots"]],
                    deadline,
                )
            )
        except TimeoutError:
            pending.extend(groups[number:])
            break
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            # Exception type only: parser messages may contain source payloads.
            blocked.append(dict(group=group, error=type(exc).__name__))
    result = dict(
        source_sha=plan["source_sha"],
        plan_sha256=sha(root / "plan.json"),
        index=index,
        committed_groups=committed,
        pending_groups=pending,
        blocked_groups=blocked,
        scheduler_job_id=os.getenv("SLURM_JOB_ID"),
        fresh_confirmation_authorized=False,
    )
    path = root / "workers" / f"shard-{index}.json"
    write_once(path, result)
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
    merged = {k: set() for k in grouped.IDENTITIES}

    def merge(identities):
        for k in grouped.IDENTITIES:
            merged[k].update(tuple(v) if isinstance(v, list) else v for v in identities[k])

    for name in plan["inventory_paths"]:
        merge(read(name)["identities"])
    expected = {g["path"]: g for g in plan["groups"]}
    proofs, missing, pending, blocked, bound = [], [], [], [], {}
    for index in range(plan["shard_count"]):
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
        bound[str(path)] = sha(path)
        unresolved = result["pending_groups"] + [r["group"] for r in result["blocked_groups"]]
        for group in unresolved:
            if expected.get(group["path"]) != group:
                raise ValueError("unresolved group registration differs")
        classified = [g["path"] for g in result["committed_groups"] + unresolved]
        assigned = {name for name, g in expected.items() if g["index"] == index}
        if len(classified) != len(set(classified)) or set(classified) != assigned:
            raise ValueError("worker dropped or duplicated a registered source group")
        pending.extend(result["pending_groups"])
        blocked.extend(result["blocked_groups"])
        for transaction in result["committed_groups"]:
            name = transaction["path"]
            group = expected[name]
            if (
                transaction["signature"] != plan["source_bindings"][name]["signature"]
                or transaction["status"] != "verified_group_transaction"
                or transaction["raw_or_answer_values_exported"] is not False
            ):
                raise ValueError("transaction source signature or redaction status differs")
            registered = {key(r): r for r in group["rows"]}
            records = {key(r): r for r in transaction["records"]}
            observed = {key(r): r for r in transaction["record_proofs"]}
            if (
                len(records) != len(transaction["records"])
                or len(observed) != len(transaction["record_proofs"])
                or set(records) != set(registered)
                or set(observed) != set(registered)
            ):
                raise ValueError("transaction record/proof offset set differs")
            for identifier, row in registered.items():
                proofs.append(
                    checked_proof(
                        observed[identifier],
                        row,
                        transaction["whole_file_sha256_before"],
                        transaction["whole_file_sha256_after"],
                    )
                )
                merge(records[identifier]["identities"])
    missing_offsets = sum(len(g["rows"]) for g in expected.values() if g["index"] in missing)
    pending_count = sum(len(g["rows"]) for g in pending)
    blocked_count = sum(len(g["group"]["rows"]) for g in blocked)
    if len(proofs) + pending_count + blocked_count + missing_offsets != EXPECTED_PENDING:
        raise ValueError("pending offset denominator not conserved")
    write_once(
        root / "coverage_overlay.json",
        dict(
            source_sha=plan["source_sha"],
            imported_proofs=plan["imported_proofs"],
            new_proofs=proofs,
            original_errors_path=plan["preserved_errors_path"],
            parent_bindings=plan["bindings"],
            original_errors_preserved=True,
            error_waivers=0,
        ),
    )
    write_once(
        root / "combined_identity_inventory.json",
        dict(
            source_sha=plan["source_sha"],
            identities={k: sorted(v) for k, v in merged.items()},
            complete_historical_id_inventory=False,
            cross_dataset_media_certified=False,
            fresh_confirmation_authorized=False,
        ),
    )
    gate = dict(
        status="pending_oversize_recovery_complete"
        if not (missing or pending or blocked)
        else "pending_oversize_recovery_incomplete",
        source_sha=plan["source_sha"],
        original_offsets=plan["original_offsets"],
        previously_recovered_offsets=plan["previously_recovered_offsets"],
        newly_recovered_offsets=len(proofs),
        total_recovered_offsets=plan["previously_recovered_offsets"] + len(proofs),
        pending_offsets=pending_count,
        blocked_offsets=blocked_count,
        missing_worker_offsets=missing_offsets,
        missing_shards=missing,
        worker_bindings=bound,
        plan_sha256=sha(root / "plan.json"),
        identity_inventory_sha256=sha(root / "combined_identity_inventory.json"),
        coverage_overlay_sha256=sha(root / "coverage_overlay.json"),
        pending_frontier_items=plan["pending_frontier_items"],
        original_and_new_coverage_errors=plan["original_and_new_coverage_errors"],
        original_errors_preserved=True,
        error_waivers=0,
        roots_restarted=False,
        automatic_resubmission=False,
        external_video_journals_unresolved=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
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
