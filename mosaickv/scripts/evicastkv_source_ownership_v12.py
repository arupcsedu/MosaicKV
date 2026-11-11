#!/usr/bin/env python3
"""Versioned trace-first/indexed observer for the same consumed source request.

This does not edit or resume V11, alter its source codec, or request another
GPU automatically. Its separate protocol exists only for an explicitly
authorized retry if V11 cannot preserve a complete raw trace.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import time
import zlib
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import evicastkv_source_ownership_closure as original
from evicastkv_source_trace_adjudication import indexed_conservative_report

from mosaickv.evaluation.fox_campaign import digest, sha, write_once

VARIANT = "source_owner_trace_first_indexed_generation_v12"
POLICY = dict(
    original.POLICY,
    observer_variant=VARIANT,
    archive_actual_trace_before_attribution=True,
    generation_match="same_interval_predicate_indexed_by_address",
    archive_gzip_compresslevel=1,
)
MAX_TRACE_JSON_BYTES = 2 * 2**30


class _BoundedWriter:
    def __init__(self, stream):
        self.stream, self.bytes = stream, 0

    def write(self, value):
        self.bytes += len(value.encode())
        if self.bytes > MAX_TRACE_JSON_BYTES:
            raise ValueError("bounded raw ownership trace archive exceeds 2 GiB")
        return self.stream.write(value)


def _gzip_once(path, value):
    """Publish only a complete gzip member, never a truncated apparent trace."""
    path = Path(path)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".source-owner-", dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
        with gzip.open(
            temporary, "wt", compresslevel=POLICY["archive_gzip_compresslevel"]
        ) as stream:
            writer = _BoundedWriter(stream)
            json.dump(value, writer, sort_keys=True, allow_nan=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        return writer.bytes
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def classify_retry(parent, *, expected_job_id, accounting_text):
    """One narrowly justified TIMEOUT rescue, not a retry on generic audit errors.

    Caller must preserve the exact sacct response and returned decision in its
    immutable registration. This function never submits or changes a Slurm job.
    """
    parent = Path(parent)
    result = dict(
        rerun=False,
        one_retry_only=True,
        accounting_sha256=hashlib.sha256(accounting_text.encode()).hexdigest(),
        actual_GPU_retry_submitted=False,
    )
    try:
        protocol = original.read(parent / "protocol.json")
        original.clean_source(Path(protocol["source_root"]), protocol["source_sha"])
        if protocol["source_sha"] != "1aab16c63eee2f063194e6db8c9b3b5af466c3a1":
            raise ValueError("only the registered V11 producer is eligible")
        for path, expected in protocol["bindings"].items():
            if sha(Path(path)) != expected:
                raise ValueError("registered V11 source/evidence hash changed")
        manifest = original.read(parent / "worker/manifest.json")
        job = str(expected_job_id)
        if (
            not job.isascii()
            or not job.isdigit()
            or not accounting_text.isascii()
            or manifest.get("slurm_job_id") != job
            or manifest.get("source_sha") != protocol["source_sha"]
        ):
            raise ValueError("scheduler proof differs from immutable V11 worker identity")
        rows = [
            line.rstrip("|").split("|") for line in accounting_text.splitlines() if line.strip()
        ]
        if rows and rows[0][0].lower() == "jobidraw":
            header = [column.lower() for column in rows.pop(0)]
            if header != [
                "jobidraw",
                "state",
                "elapsedraw",
                "timelimitraw",
            ]:
                raise ValueError("concrete four-field sacct accounting schema required")
        top = [row for row in rows if row[0] == job]
        if len(top) != 1 or any(row[0] != job and not row[0].startswith(job + ".") for row in rows):
            raise ValueError("exactly one matching top-level Slurm job required")
        row = top[0]
        if (
            len(row) != 4
            or row[1] != "TIMEOUT"
            or not row[2].isdigit()
            or not row[3].isdigit()
            or int(row[3]) != protocol["wall_minutes"]
            or int(row[2]) < int(row[3]) * 60
        ):
            raise ValueError("proven TIMEOUT at the registered wall limit required")
        result.update(
            proven_timeout=True,
            job_id=job,
            elapsed_seconds=int(row[2]),
            original_minutes=int(row[3]),
            protocol_sha256=sha(parent / "protocol.json"),
            manifest_sha256=sha(parent / "worker/manifest.json"),
        )
        completed = {}
        for name in ("raw.json", "logits.pt"):
            artifact = parent / "worker" / name
            if (
                not artifact.is_file()
                or artifact.is_symlink()
                or artifact.resolve().parent != (parent / "worker").resolve()
            ):
                raise ValueError("completed model-request raw record and logits are required")
            completed[name] = sha(artifact)
        raw = original.read(parent / "worker/raw.json")
        if (
            raw.get("arm") != original.ARM
            or raw.get("record", {}).get("output_tokens") != 128
            or raw.get("counters", {}).get("updates") != 127
            or raw.get("record", {}).get("storage", {}).get("history_updates_per_layer")
            != [127] * 32
        ):
            raise ValueError("complete registered 128-prediction model request required")
        result.update(completed_model_request=True, completed_request_sha256=completed)
        finality = [
            parent / "worker" / name
            for name in ("result.json", "allocator_ledger.json.gz", "declared_owners.json")
        ]
        result["missing_finality"] = [path.name for path in finality if not path.exists()]
        final = parent / "worker/result.json"
        if final.exists():
            saved = original.read(final)
            if saved["source_sha"] != protocol["source_sha"]:
                raise ValueError("V11 final result source changed")
            for name, expected in saved["outputs"].items():
                if Path(name).name != name or sha(parent / "worker" / name) != expected:
                    raise ValueError("V11 finalized raw artifact hash changed")
        trace = parent / "worker/allocator_trace.json.gz"
        if not trace.exists():
            result["raw_trace_status"] = "missing"
        else:
            if not trace.is_file() or trace.resolve().parent != (parent / "worker").resolve():
                raise ValueError("raw trace is not a regular in-scope archive")
            before = trace.stat()
            result["raw_trace_sha256"] = sha(trace)
            count = 0
            deadline = time.monotonic() + 120
            try:
                with gzip.open(trace, "rb") as stream:
                    while True:
                        if time.monotonic() >= deadline or count > MAX_TRACE_JSON_BYTES:
                            raise ValueError(
                                "bounded archive inspection exceeded; not a retry waiver"
                            )
                        block = stream.read(2**20)
                        if not block:
                            break
                        count += len(block)
                result["raw_trace_status"] = "complete_gzip_CPU_recovery_only"
            except EOFError:
                result["raw_trace_status"] = "EOF_truncated_gzip"
            result["uncompressed_bytes_observed"] = count
            after = trace.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or sha(trace) != result["raw_trace_sha256"]:
                raise ValueError("raw archive changed during inspection")
        result["rerun"] = bool(
            result["missing_finality"]
            and result["raw_trace_status"] in {"missing", "EOF_truncated_gzip"}
        )
        result["status"] = (
            "one_trace_first_retry_eligible"
            if result["rerun"]
            else "CPU_recovery_or_existing_finality_no_GPU_retry"
        )
    except (OSError, ValueError, KeyError, TypeError, zlib.error) as error:
        result.update(
            rerun=False,
            status="blocked_unverified_retry_scope",
            error=f"{type(error).__name__}: {error}",
        )
    return result


def archive_before_attribution(root, capture, observations, events, snapshot):
    """Raw metadata only: no model values/answers and no declared roles added."""
    root = Path(root)
    if snapshot is None:
        raise ValueError("actual final allocator snapshot required before attribution")
    target = root / "worker/allocator_trace_pre_attribution.json.gz"
    payload = dict(
        observer_variant=VARIANT,
        actual_events=events,
        allocator_records=capture["records"],
        stack_table=capture["stack_table"],
        clock_calibration=capture.get("clock_calibration"),
        complete_request=capture["complete_request"],
        allocator_history_complete=capture["allocator_history_complete"],
        capture_metadata=capture,
        final_allocator_snapshot=snapshot,
        installed_and_local_source_hashes=original.stack_source_hashes(capture["stack_table"]),
        inherited_stack_classifications_not_admitted=True,
    )
    raw_bytes = _gzip_once(target, payload)
    owners_path = root / "worker/declared_owners_pre_attribution.json"
    write_once(owners_path, dict(observations=observations, observer_variant=VARIANT))
    receipt = dict(
        observer_variant=VARIANT,
        source_sha=original.read(root / "protocol.json")["source_sha"],
        protocol_sha256=sha(root / "protocol.json"),
        trace_path=str(target),
        trace_sha256=sha(target),
        uncompressed_bytes=raw_bytes,
        declared_owners_sha256=sha(owners_path),
        actual_events_sha256=digest(events),
        allocation_records_sha256=digest(capture["records"]),
        final_allocator_snapshot_sha256=digest(snapshot),
        capture_sha256=digest(capture),
        complete_request=capture["complete_request"],
        allocator_history_complete=capture["allocator_history_complete"],
        attribution_not_yet_run=True,
        new_roles_assigned=0,
        complete_gzip_published_atomically=True,
        exact_whole_request_KV_peak_certified=False,
    )
    write_once(root / "worker/trace_pre_attribution_receipt.json", receipt)
    return receipt


@contextmanager
def _policy():
    with patch.object(original, "POLICY", POLICY):
        yield


def prepare(root, parent=original.PARENT):
    with _policy():
        return original.prepare(root, parent)


def worker(root, index=0):
    from mosaickv.evaluation import request_ownership

    holders = []
    constructor = request_ownership._RequestSession

    def session(*args, **kwargs):
        value = constructor(*args, **kwargs)
        holders.append(value)
        return value

    def reduce(capture, observations, *, events):
        if len(holders) != 1:
            raise ValueError("one actual allocator observer required")
        archive_before_attribution(root, capture, observations, events, holders[0].snapshot)
        return indexed_conservative_report(capture, observations, events=events)

    with (
        _policy(),
        patch.object(request_ownership, "_RequestSession", session),
        patch.object(original, "conservative_report", reduce),
    ):
        return original.worker(root, index)


def analyze(root):
    with _policy():
        outcome = original.analyze(root)
    receipt = original.read(Path(root) / "worker/trace_pre_attribution_receipt.json")
    if sha(Path(receipt["trace_path"])) != receipt["trace_sha256"]:
        raise ValueError("trace-first raw archive changed")
    write_once(
        Path(root) / "GATE_source_ownership_v12.json",
        dict(
            observer_variant=VARIANT,
            parent_gate_sha256=sha(Path(root) / "GATE_source_ownership_closure.json"),
            trace_first_receipt_sha256=sha(
                Path(root) / "worker/trace_pre_attribution_receipt.json"
            ),
            raw_trace_saved_before_attribution=True,
            exact_interval_predicate_unchanged=True,
            source_whole_request_KV_peak_certified=False,
            whole_device_memory_certified=False,
            source_gate=outcome,
            fresh_confirmation_authorized=False,
            superiority_established=False,
        ),
    )
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=original.PARENT)
    args = parser.parse_args()
    value = (
        prepare(args.root, args.parent)
        if args.mode == "prepare"
        else globals()[args.mode](args.root)
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
