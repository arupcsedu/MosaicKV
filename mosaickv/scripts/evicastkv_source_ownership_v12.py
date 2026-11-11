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
from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_source_ownership_closure as original
from evicastkv_source_trace_adjudication import indexed_conservative_report
from evicastkv_source_trace_codec import LIMITS as CODEC_LIMITS
from evicastkv_source_trace_codec import document_digests, encode

from mosaickv.evaluation.fox_campaign import sha, write_once

VARIANT = "source_owner_trace_first_indexed_generation_v13_lossless_metadata_DAG"
POLICY = dict(
    original.POLICY,
    observer_variant=VARIANT,
    archive_actual_trace_before_attribution=True,
    generation_match="same_interval_predicate_indexed_by_address",
    archive_gzip_compresslevel=1,
    archive_codec="source_allocator_metadata_DAG_v1",
    archive_codec_limits=CODEC_LIMITS,
)
MAX_TRACE_JSON_BYTES = 2 * 2**30
ARCHIVE_FAILURE_PARENT = Path("/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1/ownership")


class _BoundedWriter:
    def __init__(self, stream):
        self.stream, self.bytes = stream, 0

    def write(self, value):
        self.bytes += len(value.encode())
        if self.bytes > MAX_TRACE_JSON_BYTES:
            raise ValueError(
                f"bounded raw ownership trace archive exceeds 2 GiB; attempted_bytes={self.bytes}"
            )
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
        if temporary.stat().st_size > CODEC_LIMITS["compressed_bytes"]:
            raise ValueError("bounded compressed ownership trace exceeds 512 MiB")
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


def classify_archive_failure(parent):
    """One serialization-only retry proof, not admission on a generic failure."""
    parent = Path(parent)
    protocol = original.read(parent / "protocol.json")
    if protocol["source_sha"] != "76d4ced1a10c6746bad73f564e84210522355ef9":
        raise ValueError("only the registered V12 archive-cap failure is eligible")
    original.clean_source(Path(protocol["source_root"]), protocol["source_sha"])
    result = original.read(parent / "worker/result.json")
    manifest = original.read(parent / "worker/manifest.json")
    if (
        result["source_sha"] != protocol["source_sha"]
        or manifest["source_sha"] != protocol["source_sha"]
        or result["status"] != "failed"
        or result["continuation_updates"] != 127
        or result["failures"]
        != [dict(error="ownership reduction: bounded raw ownership trace archive exceeds 2 GiB")]
        or not str(manifest.get("slurm_job_id", "")).isdigit()
    ):
        raise ValueError("exact completed-model V12 serialization-cap failure required")
    bindings = dict(protocol["bindings"])
    for name, expected in result["outputs"].items():
        path = parent / "worker" / name
        if (
            Path(name).name != name
            or not path.is_file()
            or path.is_symlink()
            or sha(path) != expected
        ):
            raise ValueError("immutable V12 failure artifacts changed")
        bindings[str(path)] = expected
    for path in (parent / "protocol.json", parent / "worker/result.json"):
        bindings[str(path)] = sha(path)
    if any(sha(Path(path)) != expected for path, expected in bindings.items()):
        raise ValueError("registered archive-failure source/evidence binding changed")
    if not {"manifest.json", "raw.json", "logits.pt", "declared_owners.json"} <= set(
        result["outputs"]
    ):
        raise ValueError("completed model and declared-owner evidence required")
    if any(
        (parent / "worker" / name).exists()
        for name in (
            "allocator_trace_pre_attribution.json.gz",
            "trace_pre_attribution_receipt.json",
            "allocator_trace.json.gz",
            "allocator_ledger.json.gz",
        )
    ):
        raise ValueError("an existing allocator archive requires CPU recovery, not another GPU")
    raw = original.read(parent / "worker/raw.json")
    if (
        raw["arm"] != original.ARM
        or raw["record"]["output_tokens"] != 128
        or raw["counters"]["updates"] != 127
        or raw["record"]["storage"]["history_updates_per_layer"] != [127] * 32
    ):
        raise ValueError("complete original 128-prediction model request required")
    return dict(
        status="one_lossless_serialization_retry_registered",
        one_retry_only=True,
        prior_root=str(parent),
        producer_sha=protocol["source_sha"],
        bindings=bindings,
        prior_registered_guard_pass=result["registered_guard_pass"],
        prior_source_repeat_all_tokens_agree=result["repeat_all_tokens_agree"],
        numerical_thresholds_unchanged=True,
        numerical_pass_not_claimed=True,
        actual_GPU_retry_submitted=False,
        archive_limits=CODEC_LIMITS,
    )


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
    document = encode(payload)
    raw_bytes = _gzip_once(target, document)
    semantic_hashes = document_digests(
        document,
        tuple(
            (key,)
            for key in (
                "actual_events",
                "allocator_records",
                "final_allocator_snapshot",
                "capture_metadata",
            )
        ),
    )
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
        actual_events_sha256=semantic_hashes[("actual_events",)],
        allocation_records_sha256=semantic_hashes[("allocator_records",)],
        final_allocator_snapshot_sha256=semantic_hashes[("final_allocator_snapshot",)],
        capture_sha256=semantic_hashes[("capture_metadata",)],
        archive_codec=POLICY["archive_codec"],
        archive_codec_statistics=document["statistics"],
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
    proof = classify_archive_failure(ARCHIVE_FAILURE_PARENT)
    old_write = original.write_once

    def bound_write(path, value):
        if Path(path) == Path(root) / "protocol.json":
            path_proof = Path(root) / "archive_failure_proof.json"
            old_write(path_proof, proof)
            value = dict(
                value,
                archive_failure_proof_sha256=sha(path_proof),
                bindings={
                    **value["bindings"],
                    **proof["bindings"],
                    str(path_proof): sha(path_proof),
                },
            )
        return old_write(path, value)

    with _policy(), patch.object(original, "write_once", bound_write):
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

    def bounded_dump(value, stream, **kwargs):
        if isinstance(value, dict) and {"actual_events", "allocator_records"} <= set(value):
            value = encode(value)
            stream_bound = _BoundedWriter(stream)
            json.dump(value, stream_bound, **kwargs)
            stream.flush()
            filename = getattr(getattr(stream, "buffer", None), "name", None)
            if (
                filename is not None
                and Path(filename).stat().st_size > CODEC_LIMITS["compressed_bytes"] - 32
            ):
                raise ValueError("bounded compressed standard trace exceeds 512 MiB")
            return None
        return json.dump(value, stream, **kwargs)

    scoped_json = SimpleNamespace(**vars(json))
    scoped_json.dump = bounded_dump

    with (
        _policy(),
        patch.object(request_ownership, "_RequestSession", session),
        patch.object(original, "conservative_report", reduce),
        patch.object(original, "json", scoped_json),
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
