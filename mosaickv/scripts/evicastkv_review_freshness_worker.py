#!/usr/bin/env python3
"""Finite CPU wrapper for the V7 sharded historical-identity inventory.

At most three 480-second scanner invocations run per worker. No polling,
submission, model access, identity selection or authorization is performed.
Incomplete/negative scans are retained explicitly, never promoted to freshness.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import evicastkv_review_freshness_stream as stream

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
MAX_INVOCATIONS = 3
SECONDS_PER_INVOCATION = 480


def _paths(root, index):
    return root / "shards" / f"shard-{index:03d}", root / "workers" / f"worker-{index:03d}.json"


def _root(root):
    root = Path(root).resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("freshness wrapper requires a scratch output root")
    if stream.within(root, tuple(path.resolve() for path in stream.ROOTS)):
        raise ValueError("freshness outputs must remain outside scanned historical roots")
    return root


def _gate_valid(gate, index, shards, revision):
    if not isinstance(gate, dict) or type(gate.get("traversal_complete")) is not bool:
        raise ValueError("scanner did not emit a classified traversal gate")
    if gate.get("shard_index") != index or gate.get("shard_count") != shards:
        raise ValueError("scanner gate has different shard ownership")
    if gate.get("source_sha") != revision:
        raise ValueError("scanner gate has different immutable source")
    if gate.get("fresh_confirmation_authorized") is not False:
        raise ValueError("identity inventory cannot authorize confirmation")


def worker(root, index, *, shards=8):
    """Run one bounded shard and publish one write-once worker summary."""
    if not 0 <= index < shards <= 256:
        raise ValueError("valid shard index/count required, at most 256")
    root = _root(root)
    shard_root, summary_path = _paths(root, index)
    if summary_path.exists():
        raise ValueError("immutable worker summary already exists")
    revision = clean_source(SOURCE)
    started = time.monotonic()
    attempts = []
    final_gate, failure = None, None
    recovered_terminal = False
    try:
        checkpoints = sorted(shard_root.glob("checkpoint-*.json"))
        if checkpoints:
            # A scheduler interruption may happen after checkpoint/gate
            # publication but before the worker summary. Recover only a bound
            # terminal gate; do not ask run() to resume a terminal shard.
            last = checkpoints[-1]
            gate_path = shard_root / last.name.replace("checkpoint-", "GATE_chunk-")
            if gate_path.is_file():
                saved = json.loads(gate_path.read_text())
                _gate_valid(saved, index, shards, revision)
                if saved["traversal_complete"]:
                    if saved.get("checkpoint_sha256") != sha(last):
                        raise ValueError("saved terminal gate checkpoint binding changed")
                    final_gate, recovered_terminal = saved, True
        if final_gate is None:
            for number in range(MAX_INVOCATIONS):
                resume = bool(list(shard_root.glob("checkpoint-*.json")))
                gate = stream.run(
                    shard_root,
                    shard_index=index,
                    shard_count=shards,
                    seconds=SECONDS_PER_INVOCATION,
                    resume=resume,
                )
                _gate_valid(gate, index, shards, revision)
                attempts.append(dict(number=number, resume=resume, gate=gate))
                final_gate = gate
                if gate["traversal_complete"]:
                    break
    except Exception as exc:
        # Configuration, source mutation, malformed checkpoint, permission,
        # parser/import errors and unclassified failures are not retried.
        failure = dict(type=type(exc).__name__, message=str(exc), retryable=False)
    terminal = bool(final_gate and final_gate["traversal_complete"])
    if failure is not None:
        status = "blocked_nonretryable_error"
    elif not terminal:
        status = "incomplete_invocation_budget_exhausted"
    elif final_gate.get("errors", 0):
        status = "completed_with_coverage_errors_freshness_blocked"
    else:
        status = "completed_declared_scope_freshness_blocked"
    summary = dict(
        schema_version=1,
        source_sha=revision,
        worker_source_sha256=sha(Path(__file__)),
        scanner_source_sha256=sha(Path(stream.__file__)),
        root=str(root),
        shard_root=str(shard_root),
        shard_index=index,
        shard_count=shards,
        status=status,
        maximum_invocations=MAX_INVOCATIONS,
        seconds_per_invocation=SECONDS_PER_INVOCATION,
        invocations=len(attempts),
        attempts=attempts,
        final_gate=final_gate,
        traversal_complete=terminal,
        nonretryable_error=failure,
        recovered_terminal_checkpoint=recovered_terminal,
        wall_seconds=time.monotonic() - started,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        complete_historical_id_inventory=False,
        models_loaded=False,
        cohort_selected=False,
    )
    write_once(summary_path, summary)
    return summary


def aggregate(root, *, shards=8):
    """Merge only complete registered shards; publish missing/failure evidence."""
    if not 1 <= shards <= 256:
        raise ValueError("valid shard count required, at most 256")
    root = _root(root)
    output = root / "GATE_freshness_workers.json"
    if output.exists():
        raise ValueError("immutable aggregate summary already exists")
    revision = clean_source(SOURCE)
    missing, incomplete, failed, summaries, shard_roots, bindings = [], [], [], [], [], {}
    for index in range(shards):
        shard_root, summary_path = _paths(root, index)
        shard_roots.append(shard_root)
        if not summary_path.is_file():
            missing.append(index)
            continue
        try:
            summary = json.loads(summary_path.read_text())
            bindings[str(summary_path)] = sha(summary_path)
            if (
                summary.get("shard_index"),
                summary.get("shard_count"),
                summary.get("source_sha"),
            ) != (
                index,
                shards,
                revision,
            ):
                raise ValueError("worker summary ownership/source differs")
            if summary.get("shard_root") != str(shard_root):
                raise ValueError("worker summary has a different checkpoint root")
            if summary.get("nonretryable_error"):
                failed.append(dict(index=index, failure=summary["nonretryable_error"]))
            if not summary.get("traversal_complete"):
                incomplete.append(index)
            summaries.append(summary)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failed.append(
                dict(index=index, failure=dict(type=type(exc).__name__, message=str(exc)))
            )
    merged, merge_error = None, None
    if not missing and not incomplete and not failed and len(summaries) == shards:
        try:
            merged = stream.aggregate(root / "aggregate", shard_roots)
        except Exception as exc:
            merge_error = dict(type=type(exc).__name__, message=str(exc), retryable=False)
    result = dict(
        schema_version=1,
        source_sha=revision,
        expected_shards=shards,
        status="merged_declared_scope_freshness_blocked"
        if merged
        else "blocked_missing_or_incomplete_shards",
        missing_shards=missing,
        incomplete_shards=incomplete,
        failed_shards=failed,
        worker_summary_bindings=bindings,
        merged_gate=merged,
        merge_error=merge_error,
        complete_declared_identity_field_scope=bool(
            merged and merged.get("complete_declared_identity_field_scope")
        ),
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        models_loaded=False,
        cohort_selected=False,
    )
    write_once(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "worker" and args.index is None:
        parser.error("worker requires --index")
    result = (
        worker(args.root, args.index, shards=args.shards)
        if args.mode == "worker"
        else aggregate(args.root, shards=args.shards)
    )
    print(json.dumps(result))
