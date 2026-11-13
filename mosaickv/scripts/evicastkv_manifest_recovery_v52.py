#!/usr/bin/env python3
"""Recover source-declared hash-map bindings without replaying artifact proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import ClassVar

import evicastkv_archive_recovery_v51 as capsule
import evicastkv_recover_bindings_v49 as prior
import evicastkv_startup_recovery_v50 as startup

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
COMPLETED_BINDINGS = Path("/scratch/djy8hg/workdir/fox-review-v52/run-20261113-b3a8d9f-1")
COMPLETED_BINDINGS_SHA = "e41b0be40ca048ea85d955343832c3d999791f3c23ead0f07151cee5983a4a3b"
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v51/run-20261113-96d9339-1")
PARENT_SHA = "9d8b04fe42ee5ea487391616230eaccb3f541b1dadb7c605c94c50ec71f59411"
WRITER = "mosaickv/scripts/evicastkv_consumed_instrumentation.py"
HELPER = "mosaickv/src/mosaickv/evaluation/fox_campaign.py"
HELPER_SHA = "4fa92bd9908d48ab6b040e09475d87a982521a1e934532bc5bca4e615ce0d421"
SPECS = (
    dict(
        metadata_path=str(
            prior.HISTORY
            / "evicastkv/consumed-routing-tier-diagnostic-20261029-v1/analysis/manifest.json"
        ),
        metadata_sha256="a302f875904e64982a77d0f2fe17de3d6f59d90e1532b3ee230d2f473ed0d8a9",
        source_sha="da58a3370a92fc6378ecc77c26bf6ae22887547c",
        writer_sha256="39a1b4caee3e66751cc29e550de2fb2090e73f7f03379f72559aefc7c7a81c71",
    ),
    dict(
        metadata_path=str(
            prior.HISTORY
            / "evicastkv/compact-exact-views-consumed-20261029-v1/analysis/manifest.json"
        ),
        metadata_sha256="62851f2c14ed0878b3aadcd6dfd216190428a39aa89e238a029c3c8bcf10ee20",
        source_sha="7472d7ccf9bb670b6921398cada04f6cfd2a5f85",
        writer_sha256="19a778e6453dfc6ff70cd1ab4260d4c313cf8b822a2a3c272d3736f0c0bf1e1b",
    ),
)


class ManifestReader(capsule.CapsuleReader):
    """Only the reviewed flat hash field is new; benchmark values stay opaque."""

    FIELDS = capsule.CapsuleReader.FIELDS | frozenset(("artifact_sha256",))
    ALIASES: ClassVar[dict[str, str]] = {
        **capsule.CapsuleReader.ALIASES,
        "artifact_sha256": "outputs",
    }


def metadata(path, expected, budget):
    """Hash a source-nominated metadata object while decoding provenance only."""
    path = prior.allowed(path)
    before = prior.geometry(path)
    if before[2] > prior.LIVE_BYTES:
        raise ValueError("unchanged 8-MiB metadata bound exceeded")
    budget.authorize(before[2] + 1)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if prior.lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("metadata descriptor differs")
        parser = ManifestReader(
            stream,
            deadline=budget.started + prior.SECONDS,
            maximum_bytes=max(1, before[2]),
            chunk_bytes=prior.lexical.CHUNK_BYTES,
        )
        try:
            parser.spaces()
            if parser.peek() != 123:
                raise ValueError("one provenance object required")
            result = parser.value()
            parser.spaces()
            if parser.peek() is not None:
                raise ValueError("trailing provenance payload")
            actual = parser.hash.hexdigest()
        finally:
            budget.add(parser.stats["bytes_read"])
        if prior.lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("metadata descriptor changed")
    if (
        prior.geometry(path) != before
        or actual != expected
        or parser.stats["bytes_read"] != before[2]
    ):
        raise ValueError("original metadata hash or signature differs")
    prior.projected_bound(result)
    return result


def saved_json(path, expected, budget):
    """Read only already-written audit packets, never a historical source body."""
    size = path.stat().st_size
    if size > prior.LIVE_BYTES:
        raise ValueError("unchanged saved-metadata bound exceeded")
    budget.authorize(size)
    result = prior.resume.reader.common.history.read(path, expected)
    budget.add(size)
    prior.projected_bound(result)
    return result


def recover(root):
    from mosaickv.evaluation.manifest_bindings_v52 import augment_manifest_binding

    revision = clean_source(SOURCE)
    prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    budget = prior.Budget()
    parent = saved_json(PARENT / "recovered_bindings.json", PARENT_SHA, budget)
    original = saved_json(startup.PARENT / "recovered_bindings.json", startup.PARENT_SHA, budget)
    inventory = saved_json(prior.PARENT / "inventory.json", prior.BOUND["inventory.json"], budget)
    if (
        parent["source_sha"] != "96d933906ca2d578c929fba85facfdfb40b47cb6"
        or len(parent["rows"]) != 134
        or parent["parquet_bodies_reopened"] != 0
        or original["source_sha"] != "555840326ecfb953a6c5b0e3665189d1bb4f0461"
    ):
        raise ValueError("complete immutable V49/V51 proof-reuse packets required")
    old_rows = {row["path"]: row for row in original["rows"]}
    inventory_rows = {
        row["path"]: row for row in inventory["rows"] if row["operation"] == "parquet"
    }
    current = {row["path"]: row for row in parent["rows"]}
    if set(old_rows) != set(inventory_rows) or set(current) != set(old_rows):
        raise ValueError("all 134 registered transactions must remain present")
    registry, claims, evidence, closures = {}, {}, [], {}
    for spec in SPECS:
        path = Path(spec["metadata_path"])
        value = metadata(path, spec["metadata_sha256"], budget)
        if value.get("source_sha") != spec["source_sha"]:
            raise ValueError("actual writer source differs from reviewed original manifest")
        blobs = {
            WRITER: prior.git_blob(spec["source_sha"], WRITER, budget),
            HELPER: prior.git_blob(spec["source_sha"], HELPER, budget),
        }
        if blobs != {WRITER: spec["writer_sha256"], HELPER: HELPER_SHA}:
            raise ValueError("reviewed actual writer/clean-source guard closure differs")
        registry[(spec["source_sha"], WRITER)] = spec["writer_sha256"]
        outputs = capsule.artifact_map(value["artifact_sha256"], path.parent)
        selected = {name: recorded for name, recorded in outputs.items() if name in current}
        for name, recorded in selected.items():
            if name in claims:
                raise ValueError("duplicate original artifact binding")
            claims[name] = dict(
                metadata_path=str(path),
                metadata_sha256=spec["metadata_sha256"],
                artifact_sha256=recorded,
                role="runtime_writer",
                recorded_source_sha=spec["source_sha"],
                writer_path=WRITER,
                source_blob_sha256=spec["writer_sha256"],
                source_evidence_verified=True,
            )
            closures[name] = blobs
        # Independent closing metadata transaction; not an identity-proof replay.
        if metadata(path, spec["metadata_sha256"], budget) != value:
            raise ValueError("original metadata changed")
        evidence.append(dict(spec, source_blobs=blobs, projected_original_hash_map=outputs))
    if len(claims) != 3:
        raise ValueError("all three originally declared artifact links must qualify")
    results, promoted = [], []
    for previous in parent["rows"]:
        name = previous["path"]
        path = prior.allowed(name)
        if str(path) != name or prior.geometry(path) != previous["signature"]:
            raise ValueError("saved artifact signature changed; body replay is not authorized")
        # Revalidate the source-checked V49 transaction; no target bytes opened.
        transaction = startup.reused_transaction(inventory_rows[name], old_rows[name], original)
        if name not in claims:
            results.append(previous)
            continue
        results.append(
            augment_manifest_binding(
                previous,
                inventory_rows[name],
                transaction,
                original["binding_claims"][name],
                claims[name],
                source_closure=closures[name],
                previous_reviewed_writers=prior.REVIEWED,
                reviewed_writers=registry,
            )
        )
        promoted.append(name)
    result = dict(
        source_sha=revision,
        parent_report_path=str(PARENT / "recovered_bindings.json"),
        parent_report_sha256=PARENT_SHA,
        rows=results,
        recovered_paths=promoted,
        status_counts=dict(Counter(row["status"] for row in results)),
        original_metadata_bindings=evidence,
        supplemental_claims=claims,
        original_occurrence_counts=parent["original_occurrence_counts"],
        reused_hash_transactions=134,
        reused_timing_kind="V49_whole_recovery_observed_upper_bound_not_new_per_file_measurement",
        parquet_bodies_reopened=0,
        original_JSON_identity_proofs_replayed=0,
        credential_bodies_read=False,
        historical_code_executed=False,
        newly_resolved_occurrences=0,
        combined_input_bytes=budget.input_bytes,
        wall_seconds=budget.check(),
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        historical_media_exposure_resolved=False,
        paper_campaign_authorized=False,
    )
    prior.projected_bound(result)
    prior.resume.require_artifact_capacity(root, prior.LIVE_BYTES)
    write_once(root / "recovered_bindings.json", result)
    return dict(
        status_counts=result["status_counts"],
        recovered=len(promoted),
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
    )


def analyze(root):
    revision = clean_source(SOURCE)
    prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    path = root / "recovered_bindings.json"
    if not path.exists():
        return dict(
            status="binding_recovery_failed_or_incomplete",
            source_sha=revision,
            fresh_confirmation_authorized=False,
            GPU_jobs=0,
        )
    report = saved_json(path, None, prior.Budget())
    if (
        report["source_sha"] != revision
        or report["parquet_bodies_reopened"] != 0
        or report["reused_hash_transactions"] != 134
    ):
        raise ValueError("saved proof reuse required")
    engineering_path = root / "engineering_review.json"
    engineering_status = "failed_or_incomplete"
    engineering_sha = None
    if engineering_path.exists():
        engineering_report = saved_json(engineering_path, None, prior.Budget())
        if (
            engineering_report.get("audit_source_sha") != revision
            or engineering_report.get("superiority_established") is not False
            or engineering_report.get("fresh_confirmation_authorized") is not False
        ):
            raise ValueError("independent review audit source/scope differs")
        engineering_status = "complete_registered_scopes_only"
        engineering_sha = sha(engineering_path)
    return dict(
        source_sha=revision,
        report_sha256=sha(path),
        status_counts=report["status_counts"],
        recovered_paths=report["recovered_paths"],
        reused_hash_transactions=134,
        newly_resolved_occurrences=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
        engineering_review_status=engineering_status,
        engineering_review_sha256=engineering_sha,
        GPU_jobs=0,
    )


def original_git_difference(command, budget):
    """Bound time AND bytes before retaining Git's exact original name list."""
    budget.authorize(prior.LIVE_BYTES)
    deadline = min(budget.started + prior.SECONDS, time.monotonic() + 15)
    proc = subprocess.Popen(command, cwd=SOURCE, stdout=subprocess.PIPE)
    parts, total = [], 0
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("bounded original Git bridge inspection deadline")
            events = selector.select(min(remaining, 0.1))
            if not events:
                continue
            raw = os.read(proc.stdout.fileno(), min(64 * 1024, prior.LIVE_BYTES + 1 - total))
            if not raw:
                break
            budget.add(len(raw))
            total += len(raw)
            if total > prior.LIVE_BYTES:
                raise ValueError("unchanged Git bridge live-byte bound exceeded")
            parts.append(raw)
        proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        if proc.returncode != 0:
            raise ValueError("original Git bridge command failed")
        return b"".join(parts)
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def gate_capsule(path, expected, budget):
    """Open only the exact, reviewed small numerical/control gate paths."""
    from mosaickv.evaluation import review_engineering_matrix_v52 as matrix

    def normalized(value):
        value = Path(value)
        # Only the already reviewed scratch-root alias is permitted. Do not
        # resolve allowlist leaves: a substituted symlink must not rewrite scope.
        alias = Path("/scratch/djy8hg")
        owner = prior.HISTORY.parent
        if value.is_relative_to(alias):
            return owner / value.relative_to(alias)
        if value.is_relative_to(owner):
            return value
        raise ValueError("gate path is outside the reviewed scratch root")

    def protected(value):
        return ".codex" in value.parts or value.name.casefold() in {
            "auth.json",
            "credentials.json",
            ".env",
            ".netrc",
        }

    path = Path(path)
    paths = [Path(v["path"]) for v in matrix.KNOWN_BINDINGS.values()]
    paths.append(Path(matrix.KNOWN_BINDINGS["coco"]["path"]).parent.parent / "plan.json")
    registered = {normalized(value) for value in paths}
    nominated = normalized(path)
    if (
        nominated not in registered
        or nominated.suffix != ".json"
        or protected(path)
        or protected(nominated)
    ):
        raise ValueError("gate capsule is outside the exact read-only scope")
    if path.is_symlink():
        raise ValueError("registered gate leaves must not be symlinks")
    canonical = path.resolve(strict=True)
    if (
        canonical != nominated
        or protected(canonical)
        or not stat.S_ISREG(canonical.lstat().st_mode)
    ):
        raise ValueError("regular gate at its original canonical path required")
    before = prior.geometry(canonical)
    if before[2] > prior.LIVE_BYTES:
        raise ValueError("unchanged gate live-byte limit exceeded")
    budget.authorize(before[2] + 1)
    fd = os.open(canonical, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if prior.lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("gate descriptor differs")
        raw = stream.read(before[2] + 1)
        budget.add(len(raw))
        if prior.lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("gate descriptor changed")
    if len(raw) != before[2] or prior.geometry(canonical) != before:
        raise ValueError("gate capsule changed")
    matrix._read(raw, expected)
    return raw


def engineering(root):
    """Current read-only engineering evidence, not another model experiment."""
    from mosaickv.evaluation import review_engineering_matrix_v52 as matrix

    revision = clean_source(SOURCE)
    prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    budget = prior.Budget()
    raw = {
        name: gate_capsule(Path(spec["path"]), spec["sha256"], budget)
        for name, spec in matrix.KNOWN_BINDINGS.items()
    }
    plan_path = Path(matrix.KNOWN_BINDINGS["coco"]["path"]).parent.parent / "plan.json"
    plan_raw = gate_capsule(plan_path, matrix.COCO_PLAN_SHA, budget)
    proof = matrix._read(plan_raw, matrix.COCO_PLAN_SHA)["execution_base_proof"]
    # Check the ORIGINAL registered V33→V34 bridge, not the current auditor.
    command = [
        "git",
        "diff",
        "--name-status",
        "--no-renames",
        "-z",
        matrix.PRODUCER,
        matrix.NUMERICAL_PRODUCER,
    ]
    diff = original_git_difference(command, budget)
    parts = diff.decode("utf-8").split("\0")
    if parts.pop() != "" or len(parts) % 2:
        raise ValueError("incomplete original Git bridge record")
    if any(status != "A" for status in parts[::2]):
        raise ValueError("original source bridge changed previously tracked files")
    if proof["added_paths"] != parts[1::2]:
        raise ValueError("actual original additions differ from registered bridge")
    transport = "mosaickv/src/mosaickv/evaluation/fox_shape_transport.py"
    for source in (matrix.PRODUCER, matrix.NUMERICAL_PRODUCER):
        if prior.git_blob(source, transport, budget) != matrix.TRANSPORT_SHA:
            raise ValueError("original V9 numerical transport differs")
    result = matrix.validate_engineering_gates(raw, source_bridge=proof)
    # This recovery already passed. Link it, rather than repeat its metadata
    # transactions or relabel its original producer as the corrective auditor.
    bindings = saved_json(
        COMPLETED_BINDINGS / "recovered_bindings.json", COMPLETED_BINDINGS_SHA, budget
    )
    if (
        bindings["source_sha"] != "b3a8d9fe4e3240e4766d4762d430354cdade4cdc"
        or bindings["parquet_bodies_reopened"] != 0
        or bindings["reused_hash_transactions"] != 134
        or bindings["newly_resolved_occurrences"] != 0
    ):
        raise ValueError("complete unchanged binding-recovery receipt required")
    result["reused_binding_recovery"] = dict(
        path=str(COMPLETED_BINDINGS / "recovered_bindings.json"),
        sha256=COMPLETED_BINDINGS_SHA,
        execution_source_sha=bindings["source_sha"],
        status_counts=bindings["status_counts"],
        reused_hash_transactions=134,
        new_binding_recovery_executed=False,
    )
    result.update(
        audit_source_sha=revision,
        original_source_bridge_git_diff_sha256=hashlib.sha256(diff).hexdigest(),
        original_plan_path=str(plan_path),
        original_plan_sha256=matrix.COCO_PLAN_SHA,
        source_bridge_git_verified=True,
        gate_capsules_only=True,
        newly_resolved_occurrences=0,
        parquet_bodies_reopened=0,
        original_JSON_identity_proofs_replayed=0,
        credential_bodies_read=False,
        combined_input_bytes=budget.input_bytes,
        wall_seconds=budget.check(),
    )
    prior.projected_bound(result)
    prior.resume.require_artifact_capacity(root, prior.LIVE_BYTES)
    write_once(root / "engineering_review.json", result)
    return dict(
        source_sha=revision,
        report_sha256=sha(root / "engineering_review.json"),
        engineering_controls_passed_in_registered_scopes=True,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "recover", "engineering", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only immutable output required")
    try:
        result = prior.resume.cpu(root) if args.mode == "cpu" else globals()[args.mode](root)
        if args.mode == "analyze":
            write_once(root / "decision.json", result)
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error="manifest binding recovery failed closed",
                fresh_confirmation_authorized=False,
                GPU_jobs=0,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
