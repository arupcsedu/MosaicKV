#!/usr/bin/env python3
"""Bounded disk-backed JSON continuation; exclude every qualified old pair."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import time
from pathlib import Path

import evicastkv_bounded_json_v44 as bounded
import evicastkv_identity_sink_v44 as identities
import evicastkv_reader_continuation_v43 as previous
import evicastkv_residual_recovery_v41 as common

from mosaickv.evaluation import metadata_recovery_v41 as core
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v44-bounded/run-20261112-f4a2362-1")
BOUND = {
    "decision.json": "40eb4ffe89a64a218d094d515b595a686f8daedd8e1854f0000e8eda0016ea75",
    "reconciled_history.json": "ab03e2a49b15204f2f0bfbdc808f1651b54c5b82df58d4e565466dad02e987ed",
    "registration.json": "aa94f0f9c0a890ba0101c79681b85bcbf08656877daa63b210eaa7c837ec893b",
}
SLOTS, SECONDS, BYTE_BOUND = 8, 210, 2 * 2**30


def eligible(parent, rows):
    unresolved = {core._key(row): row for row in parent["unresolved_occurrences"]}
    result = []
    for index, row in enumerate(rows):
        error = unresolved.get(core._key(row))
        if error is not None and row["operation"] == "json_stream":
            if (
                error["path"] != row["path"]
                or error["original_error_sha256"] != row["original_error_sha256"]
            ):
                raise ValueError("original occurrence binding differs")
            result.append(index)
    return result


def cpu(root):
    return previous.cpu(root)


def verify_reused_pairs(parent, rows, targets, roots):
    """Check saved receipts and stat sources; never read qualified old inputs."""
    completed = [
        i
        for i, row in enumerate(rows)
        if row["operation"] == "json_stream"
        and i not in set(targets)
        and all(
            str(common.artifact(previous.OLD.resolve(), i, phase)) in parent["source_bindings"]
            for phase in ("parse", "hash")
        )
    ]
    if len(completed) != 32:
        raise ValueError("exactly 32 qualified old JSON pair receipt bindings must remain excluded")
    result = {}
    for index in completed:
        row = rows[index]
        path = bounded.common.canonical(row["path"], roots)
        if str(path) != row["path"] or digest(bounded.geometry(path.lstat())) != digest(
            row["signature"]
        ):
            raise ValueError("old qualified source changed; do not silently replay it")
        for phase in ("parse", "hash"):
            receipt = common.artifact(previous.OLD.resolve(), index, phase)
            expected = parent["source_bindings"].get(str(receipt))
            if expected is None:
                raise ValueError("old completed pair lost its immutable receipt binding")
            record = common.history.read(receipt, expected)
            if (
                record["status"] != "passed"
                or record["index"] != index
                or record["phase"] != phase
                or record["registration_sha256"] != digest(row)
            ):
                raise ValueError("saved old receipt differs from its qualified occurrence")
            result[str(receipt)] = expected
    return result


def select_canaries(rows, targets):
    """Use only still-unresolved, already consumed resource failures."""
    result = []
    for category in sorted(previous.RESOURCE_ERRORS):
        candidates = []
        for index in targets:
            path = common.artifact(previous.OLD.resolve(), index, "parse")
            if not path.is_file():
                continue
            record = common.history.read(path)
            if (
                record.get("source_sha") != "d37fc8daf75af02d66ce35cf76abe0d7c3a5446b"
                or record.get("plan_sha256") != previous.OLD_PLAN_SHA
                or record.get("registration_sha256") != digest(rows[index])
            ):
                raise ValueError("consumed canary failure lost its original binding")
            if record.get("status") == "blocked" and record.get("error") == category:
                candidates.append(index)
        if not candidates:
            raise ValueError("one still-unresolved consumed failure per resource category required")
        result.append(max(candidates, key=lambda i: (rows[i]["signature"][2], -i)))
    if len(result) != 2 or len(set(result)) != 2:
        raise ValueError("two distinct consumed resource canaries required")
    return result


def verify_completed_source_signatures(rows, targets, roots):
    """Stat every inherited JSON source without reopening any source body."""
    retry = set(targets)
    count = 0
    for index, row in enumerate(rows):
        if row["operation"] != "json_stream" or index in retry:
            continue
        path = bounded.common.canonical(row["path"], roots)
        if str(path) != row["path"] or digest(bounded.geometry(path.lstat())) != digest(
            row["signature"]
        ):
            raise ValueError("completed source changed; stat-only proof reuse blocked")
        count += 1
    return count


def prepare(root):
    revision = clean_source(SOURCE)
    common.cpu_gate(root / "cpu", revision)
    parent = common.history.read(
        PARENT / "reconciled_history.json", BOUND["reconciled_history.json"]
    )
    decision = common.history.read(PARENT / "decision.json", BOUND["decision.json"])
    old = common.history.read(PARENT / "registration.json", BOUND["registration.json"])
    targets = eligible(parent, old["rows"])
    if (
        len(targets) != 1202
        or parent["counts"]["net_unresolved_error_count"] != 1337
        or decision["newly_resolved_JSON_occurrences"] != 1
        or decision["source_sha"] != "f4a2362009aafeb40b24b97ad0e13d5bfa27a695"
        or parent["source_sha"] != decision["source_sha"]
    ):
        raise ValueError("exact completed strict V44 checkpoint required")
    canaries = select_canaries(old["rows"], targets)
    bindings = dict(old["bindings"])
    bindings.update({str(PARENT / name): value for name, value in BOUND.items()})
    bindings.update(verify_reused_pairs(parent, old["rows"], targets, old["roots"]))
    completed = verify_completed_source_signatures(old["rows"], targets, old["roots"])
    if completed != 59:
        raise ValueError("exactly 59 completed JSON sources must remain excluded")
    bindings.update(
        {
            str(common.artifact(previous.OLD.resolve(), i, "parse")): sha(
                common.artifact(previous.OLD.resolve(), i, "parse")
            )
            for i in canaries
        }
    )
    plan = dict(
        old,
        source_sha=revision,
        source_bridge=common.history.source_bridge(),
        bindings=bindings,
        retry_indices=targets,
        json_canary_indices=canaries,
        worker_seconds=SECONDS,
        worker_maximum_bytes=BYTE_BOUND,
        live_metadata_bytes=8 * 2**20,
        live_metadata_nodes=100000,
        metadata_externalization=True,
        stop_after_any_unqualified_wave=True,
        successful_prior_JSON_occurrences_replayed=False,
        closing_hash_failure_stops_subsequent_waves=True,
        live_signature_required_at_reconciliation=True,
        inherited_qualified_JSON_pairs=32,
        inherited_strict_bounded_JSON_pairs=1,
        inherited_JSON_sources_signature_verified=completed,
        inherited_completed_record_JSON_proofs=26,
        inherited_record_proofs_are_unchanged_whole_file_resource_passes=False,
        qualified_old_input_bytes_reread=0,
        whole_projection_deadline_resets=0,
        separate_fragment_deadlines_permitted=False,
        noncanonical_record_diagnostic_source_sha="0d45f555141bf6f2760cba73bef924c952a69f47",
        noncanonical_record_diagnostic_tasks_relabelled_as_strict_proofs=False,
    )
    for key in (
        "algorithms",
        "target_shard_bytes",
        "record_array_source_IO_boundary",
        "whole_JSON_source_IO_boundary",
        "identity_output_maximum_bytes",
        "live_projection_bound_unchanged",
    ):
        plan.pop(key, None)
    plan["resolved_reader_policy"] = dict(
        worker_seconds=SECONDS,
        input_bytes=BYTE_BOUND,
        parallel_workers=SLOTS,
        live_metadata_bytes=8 * 2**20,
        live_metadata_nodes=100000,
        spill_bytes=bounded.projection.SPILL_BYTES,
        spill_nodes=bounded.projection.SPILL_NODES,
        source_chunk_bytes=256 * 2**10,
        flat_object_bytes=bounded.projection.FLAT_BYTES,
        plain_scalar_member_runs=True,
        learned_scalar_layouts=bounded.projection.MAXIMUM_LAYOUTS,
        learned_scalar_layout_bytes=bounded.projection.MAXIMUM_LAYOUT_BYTES,
        learned_scalar_layout_fields=bounded.projection.MAXIMUM_LAYOUT_FIELDS,
        learned_scalar_layouts_per_key=bounded.projection.MAXIMUM_LAYOUTS_PER_KEY,
        identity_shard_target_bytes=identities.TARGET_BYTES,
        identity_shard_maximum_bytes=identities.MAX_SHARD_BYTES,
        identity_shard_maximum_nodes=identities.MAX_SHARD_NODES,
        maximum_shard_references=identities.MAX_REFERENCES,
        SQLite_reservation_bytes=identities.SQLITE_RESERVATION,
        identity_hot_cache_bytes=identities.HOT_CACHE_BYTES,
        identity_hot_cache_eviction="exact_generation_reset_with_SQL_fallback",
        independent_hash_pass=True,
        stop_after_any_unqualified_wave=True,
        whole_projection_deadline_resets=0,
        separate_fragment_deadlines_permitted=False,
    )
    plan["resolved_configuration"] = dict(
        schema="strict_historical_reader_v45",
        reader_policy=plan["resolved_reader_policy"],
        registered_rows_sha256=plan["rows_sha256"],
        retry_indices=targets,
        canary_indices=canaries,
        immutable_parent_artifacts=BOUND,
    )
    plan["config_sha"] = digest(plan["resolved_configuration"])
    write_once(root / "registration.json", plan)
    return dict(retry_occurrences=len(targets), fresh_confirmation_authorized=False)


def worker(root, index, phase, plan_sha):
    plan = common.load(root, expected=plan_sha, full=False)
    common.cpu_gate(root / "cpu", plan["source_sha"])
    if type(index) is not int or index not in plan["retry_indices"]:
        raise ValueError("completed old proofs and non-JSON records cannot be scheduled")
    row, started = plan["rows"][index], time.monotonic()
    try:
        if phase == "parse":
            output = root / f"identities/{index:04d}"
            output.mkdir(parents=True, mode=0o700, exist_ok=False)
            proof = bounded.project_file(row, plan["roots"], started + SECONDS, output)
        elif phase == "hash":
            proof = common.hash_file(Path(row["path"]), row["signature"], started + SECONDS)
        else:
            raise ValueError("two independent parse/hash phases required")
        result = dict(status="passed", proof=proof)
    except Exception as error:
        result = dict(
            status="blocked",
            error_type=type(error).__name__,
            error="unqualified bounded JSON pass",
            failure_statistics=getattr(error, "statistics", {}),
        )
    result.update(
        source_sha=plan["source_sha"],
        index=index,
        phase=phase,
        registration_sha256=digest(row),
        plan_sha256=plan_sha,
        wall_seconds=time.monotonic() - started,
    )
    common.history.put(common.artifact(root, index, phase), result)
    return {key: value for key, value in result.items() if key != "proof"}


def run_tasks(root, plan, indices, phase, label):
    plan_sha = sha(root / "registration.json")

    def one(index):
        if common.checked_task(root, plan, index, phase) is not None:
            return dict(index=index, reused=True)
        path = root / f"logs/task-{index:04d}-{phase}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            try:
                result = subprocess.run(
                    [
                        "/scratch/djy8hg/env/mosaickv/bin/python",
                        __file__,
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


def shard_values(root, index, proof):
    counts = dict.fromkeys(identities.KINDS, 0)
    refs, seen, last = proof["identity_shards"], set(), {}
    if len(refs) > identities.MAX_REFERENCES or set(proof["identity_counts"]) != set(counts):
        raise ValueError("bounded complete six-kind shard references required")
    if any(type(value) is not int or value < 0 for value in proof["identity_counts"].values()):
        raise ValueError("strict nonnegative identity counts required")
    output = (root / f"identities/{index:04d}").resolve(strict=True)
    for ref in refs:
        path = Path(ref["path"])
        if (
            not path.is_relative_to(output)
            or path.resolve(strict=True) != path.absolute()
            or str(path) in seen
        ):
            raise ValueError("only unique registered identity shards may be read")
        seen.add(str(path))
        shard = identities.read_shard(ref)
        kind = shard["kind"]
        for value in shard["values"]:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8", "backslashreplace"
            )
            if kind in last and encoded <= last[kind]:
                raise ValueError("identity shards must be strictly ordered and duplicate-free")
            last[kind] = encoded
            counts[kind] += 1
            yield kind, tuple(value) if kind in core.contract.NESTED_WIDTHS else value
    if counts != proof["identity_counts"]:
        raise ValueError("identity shard counts differ from completed worker")


def pair(root, plan, index):
    first = common.checked_task(root, plan, index, "parse")
    second = common.checked_task(root, plan, index, "hash")
    if not first or not second or first["status"] != "passed" or second["status"] != "passed":
        raise ValueError("both complete independent passes required")
    for record in (first, second):
        elapsed = record.get("wall_seconds")
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or not 0 <= elapsed <= SECONDS
        ):
            raise ValueError("one unchanged whole-pass 210-second deadline required")
    a, b, row = first["proof"], second["proof"], plan["rows"][index]
    resources = (
        (a.get("total_input_bytes"), BYTE_BOUND),
        (a.get("statistics", {}).get("maximum_live_projected_metadata_bytes"), 8 * 2**20),
        (a.get("statistics", {}).get("maximum_live_projected_metadata_nodes"), 100000),
    )
    if any(type(value) is not int or not 0 <= value <= bound for value, bound in resources):
        raise ValueError("strict bounded resource counters required")
    if any(k in a and a[k] is not False for k in core.contract.FALSE_FIELDS):
        raise ValueError("proof scope widened")
    if (
        a.get("parser_schema") != "evicastkv_bounded_external_JSON_v44"
        or a.get("complete_document_validated") is not True
        or a.get("all_spill_replays_verified") is not True
        or a.get("forbidden_value_scalars_JSON_decoded") is not False
        or any(
            k in a and a[k] is not False
            for k in (
                "old_unbound_historical_version_byte_identity_claimed",
                "historical_numeric_validity_claimed",
            )
        )
    ):
        raise ValueError("unregistered parser or incomplete semantic proof")
    if (
        a["whole_file_sha256"] != b["whole_file_sha256"]
        or a["bytes_read"] != row["signature"][2]
        or b["bytes_read"] != row["signature"][2]
        or type(a["hash_passes"]) is not int
        or a["hash_passes"] != 1
        or a["independent_second_pass_verified"] is not False
        or b["independent_hash_pass"] is not True
        or any(
            digest(a[k]) != digest(row["signature"])
            for k in (
                "signature_before",
                "signature_after",
                "fd_signature_before",
                "fd_signature_after",
            )
        )
        or digest(b["signature"]) != digest(row["signature"])
        or a["path"] != row["path"]
        or b["path"] != row["path"]
        or a["total_input_bytes"] > BYTE_BOUND
        or a["total_input_bytes"] < a["bytes_read"]
        or a["statistics"]["maximum_live_projected_metadata_bytes"] > 8 * 2**20
        or a["statistics"]["maximum_live_projected_metadata_nodes"] > 100000
    ):
        raise ValueError("source, independent hash, length or resource transaction differs")
    actual = bounded.common.canonical(row["path"], plan["roots"])
    if str(actual) != row["path"] or digest(bounded.geometry(actual.lstat())) != digest(
        row["signature"]
    ):
        raise ValueError("live source signature changed after proof")
    # Fully consume verification even when callers need no identities.
    for _ in shard_values(root, index, a):
        pass
    return dict(
        a,
        signature=row["signature"],
        whole_file_sha256_before=a["whole_file_sha256"],
        whole_file_sha256_after=b["whole_file_sha256"],
        task_artifacts={
            str(common.artifact(root, index, p)): sha(common.artifact(root, index, p))
            for p in ("parse", "hash")
        },
        old_unbound_historical_version_byte_identity_claimed=False,
    )


def wave(root, plan, indices, label):
    run_tasks(root, plan, indices, "parse", label + "-parse")
    passed = [
        i
        for i in indices
        if (common.checked_task(root, plan, i, "parse") or {}).get("status") == "passed"
    ]
    run_tasks(root, plan, passed, "hash", label + "-hash")
    qualified, failed = [], []
    for index in indices:
        try:
            pair(root, plan, index)
            qualified.append(index)
        except (ValueError, OSError, KeyError):
            failed.append(index)
    return qualified, failed


def exclusion_delta(root, index, proof, known):
    """A failed shard must never mutate the cumulative exclusion state."""
    delta = {kind: set() for kind in identities.KINDS}
    for kind, value in shard_values(root, index, proof):
        if value not in known[kind]:
            delta[kind].add(value)
    return delta


def canary(root):
    plan = common.load(root)
    common.cpu_gate(root / "cpu", plan["source_sha"])
    indices = plan["json_canary_indices"]
    passed, failed = wave(root, plan, indices, "canary")
    result = dict(
        source_sha=plan["source_sha"],
        required_indices=indices,
        passed=passed,
        failures=failed,
        streaming_bulk_authorized=not failed,
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )
    write_once(root / "canary.json", result)
    return result


def bulk(root):
    plan = common.load(root)
    common.cpu_gate(root / "cpu", plan["source_sha"])
    gate = common.history.read(root / "canary.json")
    if gate["streaming_bulk_authorized"] is not True or gate["source_sha"] != plan["source_sha"]:
        result = dict(status="blocked_canary_failed", fresh_confirmation_authorized=False)
    else:
        for i in plan["json_canary_indices"]:
            pair(root, plan, i)
        indices = [i for i in plan["retry_indices"] if i not in plan["json_canary_indices"]]
        started, waves = time.monotonic(), 0
        reason = "registered_work_complete"
        for offset in range(0, len(indices), 8):
            if time.monotonic() - started > plan["bulk_wall_seconds"] - 600:
                reason = "registered_wall_bound_deferred"
                break
            _, failed = wave(root, plan, indices[offset : offset + 8], f"wave-{waves}")
            waves += 1
            if failed:
                reason = "unqualified_pass_stopped_later_waves"
                break
        result = dict(status=reason, completed_waves=waves, fresh_confirmation_authorized=False)
    write_once(root / "bulk.json", result)
    return result


def analyze(root):
    plan = common.load(root)
    common.cpu_gate(root / "cpu", plan["source_sha"])
    parent = common.history.read(
        PARENT / "reconciled_history.json", BOUND["reconciled_history.json"]
    )
    known = core.contract._identities(parent["identities"])
    bindings = dict(plan["bindings"])
    bindings[str(root / "registration.json")] = sha(root / "registration.json")
    rows, receipts, failures = [], [], []
    for index in plan["retry_indices"]:
        row = plan["rows"][index]
        try:
            proof = pair(root, plan, index)
            delta = exclusion_delta(root, index, proof, known)
            values = {
                kind: [list(v) for v in sorted(items)]
                if kind in core.contract.NESTED_WIDTHS
                else sorted(items)
                for kind, items in delta.items()
            }
            proof.update(
                identity_exclusion_mode="full_shards_with_parent_relative_delta_receipt",
                original_errors_preserved=True,
                added_exclusion_counts={k: len(v) for k, v in delta.items()},
            )
            path = root / f"proofs/{index:04d}.json"
            write_once(path, proof)
            receipt = dict(
                registration_sha256=digest(row),
                operation="json_stream",
                path=row["path"],
                signature=row["signature"],
                proof_path=str(path),
                proof_sha256=sha(path),
                whole_file_sha256_before=proof["whole_file_sha256_before"],
                whole_file_sha256_after=proof["whole_file_sha256_after"],
                source_hash_kind="file_bytes",
                identities=values,
                old_unbound_historical_version_byte_identity_claimed=False,
                **dict.fromkeys(core.contract.FALSE_FIELDS, False),
            )
            bindings[str(path)] = sha(path)
            bindings.update(proof["task_artifacts"])
            bindings.update({ref["path"]: ref["sha256"] for ref in proof["identity_shards"]})
            rows.append(row)
            receipts.append(receipt)
            for kind, items in delta.items():
                known[kind].update(items)
        except (ValueError, OSError, KeyError) as error:
            failures.append(
                dict(
                    index=index,
                    error_type=type(error).__name__,
                    error="no complete qualified bounded transaction",
                )
            )
    report = core.apply_metadata_proofs(
        parent,
        rows,
        receipts,
        source_sha=plan["source_sha"],
        bindings=bindings,
        seed_error_counts=plan["seed_error_counts"],
    )
    common.history.put(root / "reconciled_history.json", report)
    write_once(root / "failures.json", dict(unresolved_occurrences=failures))
    result = dict(
        source_sha=plan["source_sha"],
        newly_resolved_JSON_occurrences=len(rows),
        error_counts=report["counts"],
        successful_prior_proofs_replayed=False,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        historical_media_exposure_resolved=False,
        reconciled_history_sha256=sha(root / "reconciled_history.json"),
    )
    write_once(root / "decision.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "prepare", "canary", "bulk", "analyze", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--phase", choices=("parse", "hash"))
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only output root required")
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
                error="bounded continuation failed closed",
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
