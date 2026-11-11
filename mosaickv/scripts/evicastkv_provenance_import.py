#!/usr/bin/env python3
"""Verify and import one completed provenance recovery, never execute it again.

Only bounded saved receipts, their metadata/code bindings, and CPU-suite
receipts are read. No Parquet body, row decoder, producer proof function, scan,
model, or scheduler is called. Original producer identity is preserved in a
byte-identical gate; the new reader identity exists only in reuse.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v18/run-20261111-2/provenance")
PARENT_SOURCE_ROOT = Path("/scratch/djy8hg/workdir/fox-review-v18-retry-code")
OUTPUT_BASE = Path("/scratch/djy8hg/workdir")
PARENT_SOURCE = "079bd85cf228738cb15340f2de45e10ee33a3b4b"
PRODUCER = "cd5c3ca625c19c3df095dc58dccc9fa8bfc850f6"
GATE = "GATE_native_v4_provenance_recovery.json"
EXPECTED_HASHES = {
    GATE: "f31742a87a432b9d144dc623016d58572e0adbcbe2f4d79563c4dfa77edd0173",
    "coverage_overlay.json": "7954615544f3ec9c25d9bae7561ba49b856fce658ee0a741287d21e8587a9185",
    "net_error_report.json": "1f3f47a7c5b5f363c1beaaeb9cdf774d47c843ccbab082cce0ea03fb1cd9a6a8",
    "plan.json": "87e37f0bce6b0d6fb8bba09d8041f9974416858df739c0690795b34a23baa778",
    "workers/shard-0.json": "9a2137b98ff20c9b85be5495b689f3006123eff3987f26ff0359f9f1ccfcd100",
    "workers/shard-1.json": "9b06d28fde3157b8793f69117504d8f83f699894cac2d671b4f28d498992a432",
}
PARENT_CPU_GATE_SHA = "826e3330badcbe64f0b4afa19de3104f0fde98af9b6df9518c15be14ee2ebcb6"
PARENT_CPU_SUITE_SHA = "2e5ee4be5892434c57400af83a593b52019676108ae37a95dec3234511004043"
EXPECTED = dict(
    registered=678,
    imported=2170,
    raw=39861,
    before=29168,
    net=10015,
    frontier=9002,
    recipe_subset=9403,
    cpu_tests=6744,
)
NET_FIELDS = (
    "original_error_count",
    "unique_covered_before",
    "newly_covered_error_hashes",
    "net_unresolved_after",
    "net_unresolved_categories",
    "pending_frontier_items",
    "producer_recipe_subset_of_yaml",
    "scientific_scope",
)
MAX_RECEIPT_BYTES, MAX_BOUND_BYTES = 128 * 2**20, 16 * 2**20
MAX_TOTAL_BINDING_BYTES = 512 * 2**20


def stable(path, expected, *, bound=MAX_RECEIPT_BYTES):
    return read_bytes(Path(path), bound=bound, expected=expected)[0]


def copy_once(path, raw):
    """Atomic exact-byte copy, refusing existing destinations and race winners."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_saved(parent):
    """Return the verified, fixed completed receipts without redoing a proof."""
    parent = Path(parent).resolve(strict=True)
    if parent != PARENT.resolve(strict=True):
        raise ValueError("only the exact completed V18 provenance parent is authorized")
    clean_source(PARENT_SOURCE_ROOT, PARENT_SOURCE)
    raws = {name: stable(parent / name, expected) for name, expected in EXPECTED_HASHES.items()}
    records = {name: read_json(raw) for name, raw in raws.items()}
    gate, plan, overlay, net = (
        records[name]
        for name in (GATE, "plan.json", "coverage_overlay.json", "net_error_report.json")
    )
    if (
        gate["source_sha"] != PARENT_SOURCE
        or plan["source_sha"] != PARENT_SOURCE
        or gate["status"] != "native_v4_provenance_recovery_complete"
        or gate["registered_files"] != EXPECTED["registered"]
        or gate["newly_proved_files"] != EXPECTED["registered"]
        or gate["previously_proved_files"] != EXPECTED["imported"]
        or any(
            gate[k] for k in ("pending_files", "blocked_files", "missing_files", "missing_shards")
        )
        or gate["fresh_confirmation_authorized"]
        or gate["materialization_authorized"]
        or gate["error_waivers"] != 0
        or not gate["original_errors_preserved"]
        or not gate["external_video_journals_unresolved"]
        or gate["roots_restarted"]
        or gate["coverage_overlay_sha256"] != EXPECTED_HASHES["coverage_overlay.json"]
        or gate["net_error_report_sha256"] != EXPECTED_HASHES["net_error_report.json"]
    ):
        raise ValueError("completed parent gate or original endpoint denominators differ")
    cpu_root = Path(plan["cpu_gate_root"]).resolve(strict=True)
    if cpu_root != parent.parent:
        raise ValueError("parent CPU receipt scope differs")
    cpu_raw = stable(cpu_root / "GATE_cpu_suite.json", PARENT_CPU_GATE_SHA)
    cpu = cpu_gate(cpu_root, PARENT_SOURCE)
    if (
        read_json(cpu_raw) != cpu
        or cpu["counts"] != dict(tests=EXPECTED["cpu_tests"], failures=0, errors=0, skipped=0)
        or cpu["cpu_suite_sha256"] != PARENT_CPU_SUITE_SHA
    ):
        raise ValueError("complete original 6744-test CPU receipt differs")
    if (
        plan["shard_count"] != 2
        or len(plan["queue"]) != EXPECTED["registered"]
        or len(plan["imported_proofs"]) != EXPECTED["imported"]
        or plan["producer"]["recorded_producer_commit"] != PRODUCER
        or plan["producer"]["historical_source_executed"] is not False
        or plan["original_errors_preserved"] is not True
        or plan["fresh_confirmation_authorized"]
        or plan["materialization_authorized"]
        or overlay["source_sha"] != PARENT_SOURCE
        or digest(overlay["imported_proofs"]) != digest(plan["imported_proofs"])
        or overlay["parent_bindings"] != plan["bindings"]
        or overlay["original_errors_path"] != plan["original_errors_path"]
    ):
        raise ValueError("original plan or imported-proof lineage differs")
    expected = {r["path"]: r for r in plan["queue"]}
    if len(expected) != EXPECTED["registered"]:
        raise ValueError("parent candidate paths duplicated")
    proofs = []
    for index in range(2):
        name = f"workers/shard-{index}.json"
        path = str((parent / name).resolve(strict=True))
        worker = records[name]
        if (
            gate["worker_bindings"].get(path) != EXPECTED_HASHES[name]
            or worker["source_sha"] != PARENT_SOURCE
            or worker["index"] != index
            or worker["plan_sha256"] != EXPECTED_HASHES["plan.json"]
            or worker["pending"]
            or worker["blocked"]
            or worker["fresh_confirmation_authorized"]
        ):
            raise ValueError("complete source-matched worker receipt differs")
        assigned = {p for p, row in expected.items() if row["index"] == index}
        actual = [p["path"] for p in worker["committed"]]
        if len(actual) != len(set(actual)) or set(actual) != assigned:
            raise ValueError("worker dropped, duplicated or reassigned registered candidates")
        for proof in worker["committed"]:
            row = expected[proof["path"]]
            if (
                proof["signature"] != row["signature"]
                or proof["original_error_sha256s"] != row["original_error_sha256s"]
                or proof["whole_file_sha256_before"] != row["expected_artifact_sha256"]
                or proof["whole_file_sha256_after"] != row["expected_artifact_sha256"]
                or proof["source_proof"] != plan["producer"]
                or proof["resolved_as"] != "derived_no_request_or_media_identity"
                or proof["parquet_footer"]["rows_decoded"] != 0
                or any(plan["bindings"].get(p) != h for p, h in proof["bindings"].items())
            ):
                raise ValueError("saved strict proof differs from its original transaction")
            proofs.append(proof)
    if len(gate["worker_bindings"]) != 2 or digest(overlay["new_proofs"]) != digest(proofs):
        raise ValueError("aggregate proof set differs from both worker receipts")
    new_hashes = {h for proof in proofs for h in proof["original_error_sha256s"]}
    before = set(plan["covered_error_hashes"])
    if (
        len(new_hashes) != EXPECTED["registered"]
        or len(before) != EXPECTED["before"]
        or new_hashes & before
        or any(net[k] != gate[k] for k in NET_FIELDS)
        or net["original_error_count"] != EXPECTED["raw"]
        or net["net_unresolved_after"] != EXPECTED["net"]
        or net["pending_frontier_items"] != EXPECTED["frontier"]
        or net["producer_recipe_subset_of_yaml"] != EXPECTED["recipe_subset"]
        or net["unique_covered_before"] != len(before)
        or net["newly_covered_error_hashes"] != len(new_hashes)
        or net["original_error_count"] - len(before) - len(new_hashes)
        != net["net_unresolved_after"]
        or sum(net["net_unresolved_categories"].values()) != net["net_unresolved_after"]
    ):
        raise ValueError("historical-error denominator or retrospective net report differs")
    bindings = {str((parent / n).resolve()): h for n, h in EXPECTED_HASHES.items()}
    total = 0
    for name, expected_hash in plan["bindings"].items():
        path = Path(name)
        if path.suffix not in {".json", ".py"}:
            raise ValueError(
                "only registered metadata/code hashing is permitted; no data-body reads"
            )
        total += path.stat().st_size
        if total > MAX_TOTAL_BINDING_BYTES:
            raise ValueError("finite parent metadata verification byte bound exceeded")
        stable(path, expected_hash, bound=MAX_BOUND_BYTES)  # Hash only, never decode result values.
        bindings[str(path)] = expected_hash
    bindings[str(cpu_root / "GATE_cpu_suite.json")] = PARENT_CPU_GATE_SHA
    for name in ("cpu-tests.json", "cpu-tests.log", "cpu-tests.xml"):
        path = cpu_root / "full-cpu" / name
        bindings[str(path)] = sha(path)
    # Recheck receipt hashes after validation, before any copy becomes observable.
    for name, expected_hash in EXPECTED_HASHES.items():
        stable(parent / name, expected_hash)
    return dict(
        gate=gate, raws=raws, bindings=bindings, original_cpu=cpu, verified_metadata_bytes=total
    )


def import_completed(root, parent=PARENT):
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(OUTPUT_BASE.resolve()):
        raise ValueError("new write-once scratch import root required")
    reader = clean_source(SOURCE)
    current_cpu = cpu_gate(root.parent, reader)
    verified = validate_saved(parent)
    parent = Path(parent).resolve(strict=True)
    copied = {}
    for name in ("coverage_overlay.json", "net_error_report.json", GATE):
        raw = verified["raws"][name]
        copy_once(root / name, raw)
        copied[str(root / name)] = hashlib.sha256(raw).hexdigest()
    reuse = dict(
        status="verified_completed_provenance_import",
        reader_source_sha=reader,
        original_source_sha=PARENT_SOURCE,
        original_parent=str(parent),
        parent_bindings=verified["bindings"],
        copied_artifacts=copied,
        current_cpu_gate=current_cpu,
        original_cpu_gate=verified["original_cpu"],
        verified_metadata_bytes=verified["verified_metadata_bytes"],
        imported_registered_files=EXPECTED["registered"],
        imported_prior_proofs=EXPECTED["imported"],
        new_proof_executions=0,
        parquet_body_reads=0,
        new_scans=0,
        parent_gate_copied_byte_for_byte=True,
        old_producer_relabelled=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    write_once(root / "reuse.json", reuse)
    return dict(
        status=reuse["status"],
        original_source_sha=PARENT_SOURCE,
        reader_source_sha=reader,
        gate_sha256=sha(root / GATE),
        reuse_sha256=sha(root / "reuse.json"),
        new_proof_executions=0,
        net_unresolved_after=verified["gate"]["net_unresolved_after"],
        fresh_confirmation_authorized=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", type=Path, default=PARENT)
    args = parser.parse_args()
    print(json.dumps(import_completed(args.root, args.parent)))
