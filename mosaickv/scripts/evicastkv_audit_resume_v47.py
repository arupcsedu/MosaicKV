#!/usr/bin/env python3
"""Recover V46 proofs and resume only unqualified historical CPU transactions."""

from __future__ import annotations

import argparse
import gc
import json
import os
import stat
import time
from pathlib import Path

import evicastkv_checkpoint_receipts_v47 as archives
import evicastkv_identity_layout_v47 as layout
import evicastkv_reader_continuation_v44 as reader

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD_SOURCE = Path("/scratch/djy8hg/workdir/fox-review-v46-reader-code").resolve()
OLD = Path("/scratch/djy8hg/workdir/fox-review-v46/run-20261112-d4cfd5b-1").resolve()
OLD_SHA = "d4cfd5b72cc835bcb6de84743f74c92a08515643"
OLD_PLAN_SHA = "61cc462faaf9e2665ad53d24392904e902a1b88b40397f92f72789dce801218b"
MAX_ARTIFACT_BYTES = reader.common.history.POLICY["maximum_new_artifact_bytes"]


def _bind(bindings, values):
    for path, expected in values.items():
        if path in bindings and bindings[path] != expected:
            raise ValueError("immutable recovery binding collision")
    bindings.update(values)


def _load(root):
    plan = reader.common.load(root)
    reader.common.cpu_gate(root / "cpu", plan["source_sha"])
    return plan


def _parent(path, expected):
    report = layout.unpack_identities(reader.common.history.read(path, expected))
    archives.validate_external_receipts(report)
    return report


def artifact_bytes(root):
    total, pending = 0, [root]
    while pending:
        directory = pending.pop()
        if not directory.exists():
            continue
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                else:
                    raise ValueError("only regular immutable output artifacts are allowed")
    return total


def require_artifact_capacity(root, reserve):
    if type(reserve) is not int or reserve < 0:
        raise ValueError("explicit nonnegative artifact reservation required")
    if artifact_bytes(root) + reserve > MAX_ARTIFACT_BYTES:
        raise ValueError("unchanged aggregate artifact ceiling would be exceeded")


def archive(report, directory, budget_root):
    reserve = reader.common.history.MAX_STATE + sum(
        layout.canonical_json_size(receipt) + 1 for receipt in report.get(archives.INLINE, [])
    )
    require_artifact_capacity(budget_root, reserve)
    directory.mkdir(parents=True, exist_ok=False)
    return archives.compact_report(report, directory)


def _save(root, report, budget_root):
    directory = root / "receipts"
    compact = archive(report, directory, budget_root)
    encoded = layout.pack_identities(compact)
    reader.common.history.put(root / "reconciled_history.json", encoded)
    return encoded


def reconcile(
    producer, plan, root, parent, bindings, revision, *, expected_passes=None, budget_root=None
):
    """Read only saved proof metadata; never open any historical source body."""
    known = reader.core.contract._identities(parent["identities"])
    rows, receipts, failures = [], [], []
    passed = []
    for index in plan["retry_indices"]:
        row = plan["rows"][index]
        try:
            proof = reader.pair(producer, plan, index)
            delta = reader.exclusion_delta(producer, index, proof, known)
            values = {
                kind: [list(v) for v in sorted(items)]
                if kind in reader.core.contract.NESTED_WIDTHS
                else sorted(items)
                for kind, items in delta.items()
            }
            proof.update(
                identity_exclusion_mode="full_shards_with_parent_relative_delta_receipt",
                original_errors_preserved=True,
                added_exclusion_counts={k: len(v) for k, v in delta.items()},
            )
            if producer == OLD:
                saved = OLD / f"proofs/{index:04d}.json"
                recorded = reader.common.history.read(saved)
                if digest(recorded) != digest(proof):
                    raise ValueError("saved V46 proof differs from its original raw task pair")
                _bind(bindings, {str(saved): sha(saved)})
            proof.update(producer_source_sha=plan["source_sha"], analysis_source_sha=revision)
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
                **dict.fromkeys(reader.core.contract.FALSE_FIELDS, False),
            )
            proposed = {str(path): sha(path)}
            proposed.update(proof["task_artifacts"])
            proposed.update({ref["path"]: ref["sha256"] for ref in proof["identity_shards"]})
            _bind(bindings, proposed)
            rows.append(row)
            receipts.append(receipt)
            passed.append(index)
            for kind, items in delta.items():
                known[kind].update(items)
        except (ValueError, OSError, KeyError) as error:
            first = reader.common.checked_task(producer, plan, index, "parse")
            failures.append(
                dict(
                    index=index,
                    error_type=type(error).__name__,
                    status="failed_or_incomplete" if first is not None else "deferred",
                    error="no complete qualified bounded transaction",
                )
            )
    del known
    gc.collect()
    if expected_passes is not None and len(rows) != expected_passes:
        raise ValueError("all expected completed proof pairs must be recovered before publication")
    report = reader.core.apply_metadata_proofs(
        parent,
        rows,
        receipts,
        source_sha=revision,
        bindings=bindings,
        seed_error_counts=plan["seed_error_counts"],
    )
    _save(root, report, root if budget_root is None else budget_root)
    write_once(root / "failures.json", dict(unqualified_registered_occurrences=failures))
    result = dict(
        source_sha=revision,
        producer_source_sha=plan["source_sha"],
        newly_resolved_JSON_occurrences=len(rows),
        passed_indices=passed,
        error_counts=report["counts"],
        successful_prior_proofs_replayed=False,
        historical_source_bodies_reparsed=0,
        maximum_checkpoint_bytes=reader.common.history.MAX_STATE,
        maximum_new_artifact_bytes=MAX_ARTIFACT_BYTES,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        historical_media_exposure_resolved=False,
        reconciled_history_sha256=sha(root / "reconciled_history.json"),
    )
    write_once(root / "decision.json", result)
    return result


def cpu(root):
    return reader.cpu(root)


def recover(root):
    revision = clean_source(SOURCE)
    reader.common.cpu_gate(root / "cpu", revision)
    clean_source(OLD_SOURCE, OLD_SHA)
    reader.common.cpu_gate(OLD / "cpu", OLD_SHA)
    plan = reader.common.history.read(OLD / "registration.json", OLD_PLAN_SHA)
    if plan["source_sha"] != OLD_SHA or digest(plan["rows"]) != plan["rows_sha256"]:
        raise ValueError("exact immutable V46 producer registration required")
    parent = reader.common.history.read(
        reader.PARENT / "reconciled_history.json", reader.BOUND["reconciled_history.json"]
    )
    target = root / "recovery"
    target.mkdir(parents=True, exist_ok=False)
    normalized = target / "normalized-parent-receipts"
    parent = archive(parent, normalized, root)
    normalized_path = target / "normalized_parent.json"
    reader.common.history.put(normalized_path, layout.pack_identities(parent))
    bindings = dict(plan["bindings"])
    _bind(
        bindings,
        {
            str(OLD / "registration.json"): OLD_PLAN_SHA,
            str(OLD / "bulk.json"): sha(OLD / "bulk.json"),
            str(OLD / "failure-analyze-None.json"): sha(OLD / "failure-analyze-None.json"),
            str(normalized_path): sha(normalized_path),
            str(reader.PARENT / "reconciled_history.json"): reader.BOUND["reconciled_history.json"],
        },
    )
    return reconcile(
        OLD, plan, target, parent, bindings, revision, expected_passes=33, budget_root=root
    )


def prepare(root):
    revision = clean_source(SOURCE)
    reader.common.cpu_gate(root / "cpu", revision)
    recovered = root / "recovery"
    decision = reader.common.history.read(recovered / "decision.json")
    if (
        decision["source_sha"] != revision
        or decision["producer_source_sha"] != OLD_SHA
        or decision["newly_resolved_JSON_occurrences"] != 33
        or decision["error_counts"]["net_unresolved_error_count"] != 1303
        or decision["historical_source_bodies_reparsed"] != 0
    ):
        raise ValueError("complete proof-only recovery is required before remaining work")
    parent = _parent(recovered / "reconciled_history.json", decision["reconciled_history_sha256"])
    old = reader.common.history.read(OLD / "registration.json", OLD_PLAN_SHA)
    targets = reader.eligible(parent, old["rows"])
    if len(targets) != 1168 or set(targets) & set(decision["passed_indices"]):
        raise ValueError("completed proofs must not re-enter the retry set")
    if reader.verify_completed_source_signatures(old["rows"], targets, old["roots"]) != 93:
        raise ValueError("all 93 completed JSON sources must remain excluded")
    first = reader.common.checked_task(OLD, old, 114, "parse")
    if not first or first["status"] != "blocked" or 114 not in targets:
        raise ValueError("the one consumed failed V46 parse is the only retry canary")
    bindings = dict(parent["source_bindings"])
    _bind(
        bindings,
        {
            str(recovered / "decision.json"): sha(recovered / "decision.json"),
            str(recovered / "reconciled_history.json"): decision["reconciled_history_sha256"],
            str(OLD / "registration.json"): OLD_PLAN_SHA,
            str(reader.common.artifact(OLD, 114, "parse")): sha(
                reader.common.artifact(OLD, 114, "parse")
            ),
        },
    )
    plan = dict(
        old,
        source_sha=revision,
        source_bridge=reader.common.history.source_bridge(),
        bindings=bindings,
        retry_indices=targets,
        json_canary_indices=[114],
        inherited_JSON_sources_signature_verified=93,
        inherited_strict_bounded_JSON_pairs=35,
        inherited_completed_record_JSON_proofs=26,
        inherited_qualified_JSON_pairs=32,
        stop_after_any_unqualified_wave=True,
    )
    policy = dict(
        plan["resolved_reader_policy"],
        stop_learning_after_layout_byte_cap_refusal=True,
        full_receipts_in_hash_bound_sidecars=True,
        exact_scope_factored_checkpoint=True,
        maximum_checkpoint_bytes=reader.common.history.MAX_STATE,
        maximum_new_artifact_bytes=MAX_ARTIFACT_BYTES,
    )
    plan["resolved_reader_policy"] = policy
    plan["resolved_configuration"] = dict(
        schema="strict_historical_reader_v47",
        reader_policy=policy,
        registered_rows_sha256=plan["rows_sha256"],
        retry_indices=targets,
        canary_indices=[114],
        recovered_parent_sha256=decision["reconciled_history_sha256"],
    )
    plan["config_sha"] = digest(plan["resolved_configuration"])
    write_once(root / "registration.json", plan)
    return dict(retry_occurrences=len(targets), successful_source_bodies_reparsed=0, GPU_jobs=0)


def analyze(root):
    plan = _load(root)
    recovered = root / "recovery"
    decision = reader.common.history.read(recovered / "decision.json")
    parent = _parent(recovered / "reconciled_history.json", decision["reconciled_history_sha256"])
    bindings = dict(plan["bindings"])
    _bind(bindings, {str(root / "registration.json"): sha(root / "registration.json")})
    target = root / "analysis"
    target.mkdir(parents=True, exist_ok=False)
    return reconcile(root, plan, target, parent, bindings, plan["source_sha"], budget_root=root)


def bulk(root):
    plan = _load(root)
    gate = reader.common.history.read(root / "canary.json")
    if gate["source_sha"] != plan["source_sha"] or gate["streaming_bulk_authorized"] is not True:
        result = dict(status="blocked_canary_failed", fresh_confirmation_authorized=False)
    else:
        for index in plan["json_canary_indices"]:
            reader.pair(root, plan, index)
        indices = [i for i in plan["retry_indices"] if i not in plan["json_canary_indices"]]
        started, waves = time.monotonic(), 0
        reason = "registered_work_complete"
        for offset in range(0, len(indices), reader.SLOTS):
            if time.monotonic() - started > plan["bulk_wall_seconds"] - 600:
                reason = "registered_wall_bound_deferred"
                break
            # Reserve conservative per-worker output plus a complete checkpoint.
            # This reduces admission rather than widening the 64-GiB ceiling.
            reserve = reader.SLOTS * 4 * 2**30 + reader.common.history.MAX_STATE
            if artifact_bytes(root) + reserve > MAX_ARTIFACT_BYTES:
                reason = "registered_artifact_bound_deferred"
                break
            _, failed = reader.wave(
                root, plan, indices[offset : offset + reader.SLOTS], f"wave-{waves}"
            )
            waves += 1
            if failed:
                reason = "unqualified_pass_stopped_later_waves"
                break
        result = dict(
            status=reason,
            completed_waves=waves,
            maximum_new_artifact_bytes=MAX_ARTIFACT_BYTES,
            fresh_confirmation_authorized=False,
        )
    write_once(root / "bulk.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "recover", "prepare", "canary", "bulk", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only immutable output root required")
    try:
        result = (
            getattr(reader, args.mode)(root)
            if args.mode == "canary"
            else globals()[args.mode](root)
        )
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error="CPU audit continuation failed closed",
                GPU_jobs=0,
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
