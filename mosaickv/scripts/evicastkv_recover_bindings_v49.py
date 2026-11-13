#!/usr/bin/env python3
"""Recover recorded Parquet byte bindings, without semantic/freshness waivers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import time
from collections import Counter
from pathlib import Path

import evicastkv_audit_resume_v47 as resume
import evicastkv_json_projection_v41 as lexical

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.parquet_binding_recovery_v49 import classify_binding

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v48/run-20261113-ec314f5-1")
BOUND = {
    "inventory.json": "bac083d2209ecc242c5e8e107577eb4e52a1aeada3e246e5e6148a07eaba65b4",
    "decision.json": "3cf4296881303d8553147c4b5e91eb4064f04e48b367dba37c7be2bd5bac5658",
}
SECONDS, MAX_BYTES = 210, 2 * 2**30
LIVE_BYTES, LIVE_NODES = 8 * 2**20, 100000
HISTORY = Path("/scratch/djy8hg/runs").resolve()
HASH = re.compile(r"[a-f0-9]{64}\Z")
COMMIT = re.compile(r"[a-f0-9]{40}\Z")
METADATA_NAMES = (
    "manifest.json",
    "report.json",
    "GATE_analysis.json",
    "provenance.json",
    "metadata.json",
    "label_blind_calibration.json",
    "robust_fidelity_calibration.json",
)
MAP_NAMES = ("artifacts", "outputs", "derived_artifact_hashes")
ATT = HISTORY / "mosaickv/vllm-baseline-sweep-20261014-v1/source_attestation.json"
ATT_SHA = "205a00d9ada0148d3e5f01a92cad9eb1c56ba31ea374fe4b2a6fd6cfbb6a5e03"
SNAPSHOT = "a7b0b1a45426fd34553085d9f9af031d30fb05da"
SNAPSHOT_ROOTS = (
    HISTORY / "mosaickv/vllm-baseline-sweep-20261014-v1/source",
    HISTORY / "mosaickv/prefixkv-evidence-20261014-v1/source",
)
WRITERS = {
    HISTORY / "evicastkv/append-history-counterfactual-20261029-v1/analysis/manifest.json": {
        "metadata_sha256": "3ffa26abde3d0cdc43b53bf9854e742c4874c671c98d5ee7c82a032e9b404711",
        "commit": "d845493b4e051d82a836d569fe0ae1692483fa93",
        "writer": "mosaickv/scripts/evicastkv_append_counterfactual.py",
        "blobs": {
            "mosaickv/scripts/evicastkv_append_counterfactual.py": (
                "59947d901ca6987e91881e37897a55f1c68a79df16268e80108a0dbe1a30ccb8"
            ),
            "mosaickv/scripts/evicastkv_lifetime_history.py": (
                "5edfa011ae82d12742acb8e5994e708cd232df13df9a106ba60ed7b599fe38a6"
            ),
            "mosaickv/src/mosaickv/evaluation/fox_campaign.py": (
                "4fa92bd9908d48ab6b040e09475d87a982521a1e934532bc5bca4e615ce0d421"
            ),
        },
    },
    HISTORY
    / "evicastkv/fox-review-20261028-writing-closure-v2-report-retry1/writing/GATE_analysis.json": {
        "metadata_sha256": "f292626a08c478206fb22a1dae5475217ee892077871ed584cba5ff907ce6e10",
        "commit": "11f1b545539d2925635aaa79bebfe74167a608e8",
        "writer": "mosaickv/scripts/evicastkv_review_closure_v2.py",
        "blobs": {
            "mosaickv/scripts/evicastkv_review_closure_v2.py": (
                "2f48a86c5a003f54406686cdec252dc85e633aa142ae98e22bce1702960b6b9c"
            ),
            "mosaickv/src/mosaickv/evaluation/fox_campaign.py": (
                "4fa92bd9908d48ab6b040e09475d87a982521a1e934532bc5bca4e615ce0d421"
            ),
        },
    },
}
REVIEWED = {(v["commit"], v["writer"]): v["blobs"][v["writer"]] for v in WRITERS.values()}


class Budget:
    def __init__(self):
        self.started, self.input_bytes = time.monotonic(), 0

    def check(self):
        elapsed = time.monotonic() - self.started
        if not math.isfinite(elapsed) or elapsed > SECONDS:
            raise TimeoutError("unchanged 210-second binding-recovery deadline")
        return elapsed

    def add(self, amount):
        self.check()
        self.input_bytes += amount
        if self.input_bytes > MAX_BYTES:
            raise ValueError("unchanged 2-GiB combined-read limit")

    def authorize(self, amount):
        self.check()
        if type(amount) is not int or amount < 0 or self.input_bytes + amount > MAX_BYTES:
            raise ValueError("unchanged 2-GiB combined-read limit")


def allowed(path):
    path = Path(path)
    if ".codex" in path.parts or path.name.lower() in {"auth.json", "credentials.json"}:
        raise ValueError("credential fixture is outside binding recovery")
    target = path.resolve(strict=True)
    if (
        ".codex" in target.parts
        or target.name.lower() in {"auth.json", "credentials.json"}
        or not target.is_relative_to(HISTORY)
        or not stat.S_ISREG(target.lstat().st_mode)
    ):
        raise ValueError("regular canonical historical metadata/artifact required")
    return target


def geometry(path):
    info = path.lstat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


class ProvenanceReader(lexical.Reader):
    """Decode only root provenance scalars and flat SHA-256 maps."""

    FIELDS = frozenset((*MAP_NAMES, "files", "source_sha", "source_git_sha", "canonical_eligible"))

    def value(self, *, key="", keep=True, depth=0, classification=None, dataset_object=False):
        self.tick()
        if depth > lexical.MAXIMUM_DEPTH:
            raise ValueError("registered JSON nesting bound exceeded")
        self.stats["maximum_depth"] = max(self.stats["maximum_depth"], depth)
        self.spaces()
        byte = self.peek()
        keep = keep and (depth != 1 or key in self.FIELDS)
        if byte is None:
            raise ValueError("truncated provenance JSON")
        if byte == 123:
            if keep and depth > 1:
                raise ValueError("provenance SHA maps must have scalar values")
            if depth == 1 and key not in {*MAP_NAMES, "files"}:
                keep = False
            return self.object(keep=keep, depth=depth, dataset_object=False)
        if byte == 91:
            return self.array(
                key=key, keep=False, retain=False, depth=depth, classification=(False, False)
            )
        if byte == 34:
            raw = self.string(keep)
            if not keep:
                self.stats["discarded_scalar_values"] += 1
                return lexical.DROP
            width = 40 if depth == 1 and key in {"source_sha", "source_git_sha"} else 64
            if re.fullmatch(rb'"[a-f0-9]{' + str(width).encode() + rb'}"', raw) is None:
                raise ValueError("only registered hash strings may be decoded as provenance")
            value = lexical._decoded_scalar(raw)
            self.charge(len(raw) + len(key))
            self.stats["scalar_values_decoded"] += 1
            return value
        return self.atom(keep and depth == 1 and key == "canonical_eligible", key, True)


def metadata(path, budget, expected=None):
    path = allowed(path)
    before = geometry(path)
    if before[2] > LIVE_BYTES:
        raise ValueError("metadata file exceeds unchanged 8-MiB bound")
    # This is a new targeted provenance extraction, not another identity proof.
    budget.authorize(before[2] + 1)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("metadata descriptor differs from registered file")
        reader = ProvenanceReader(
            stream,
            deadline=budget.started + SECONDS,
            maximum_bytes=max(1, before[2]),
            chunk_bytes=lexical.CHUNK_BYTES,
        )
        try:
            reader.spaces()
            if reader.peek() != 123:
                raise ValueError("provenance metadata must be one JSON object")
            result = reader.value()
            reader.spaces()
            if reader.peek() is not None:
                raise ValueError("trailing provenance payload")
            actual = reader.hash.hexdigest()
        finally:
            budget.add(reader.stats["bytes_read"])
        if lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("metadata descriptor changed during projection")
    if (
        reader.stats["bytes_read"] != before[2]
        or geometry(path) != before
        or (expected and actual != expected)
    ):
        raise ValueError("historical metadata changed or differs from its registered hash")
    result = {} if result is lexical.DROP else result
    return result, actual, before


def git_blob(revision, path, budget):
    if COMMIT.fullmatch(revision or "") is None or path.startswith("/") or ".." in path.split("/"):
        raise ValueError("exact local Git commit and repository-relative blob required")
    budget.check()
    kind = subprocess.check_output(["git", "cat-file", "-t", revision], cwd=SOURCE, timeout=15)
    if kind.strip() != b"commit":
        raise ValueError("recorded producer is not an available Git commit")
    spec = f"{revision}:{path}"
    size = int(subprocess.check_output(["git", "cat-file", "-s", spec], cwd=SOURCE, timeout=15))
    if size > LIVE_BYTES:
        raise ValueError("recorded source blob exceeds unchanged live bound")
    budget.authorize(size)
    raw = subprocess.check_output(["git", "show", spec], cwd=SOURCE, timeout=15)
    budget.add(len(raw))
    if len(raw) != size:
        raise ValueError("Git blob length differs")
    return hashlib.sha256(raw).hexdigest()


def writer_claim(path, value, metadata_sha, artifact_sha, budget, sources):
    recorded = value.get("source_sha")
    recorded = recorded if isinstance(recorded, str) and COMMIT.fullmatch(recorded) else None
    spec = WRITERS.get(path)
    role, writer, source_sha, verified = "artifact_only", None, None, False
    if spec is not None:
        if metadata_sha != spec["metadata_sha256"] or recorded != spec["commit"]:
            raise ValueError("reviewed historical writer metadata binding differs")
        if recorded not in sources:
            blobs = {p: git_blob(recorded, p, budget) for p in spec["blobs"]}
            if blobs != spec["blobs"]:
                raise ValueError("reviewed writer/source-guard closure differs")
            sources[recorded] = blobs
        role, writer = "runtime_writer", spec["writer"]
        source_sha, verified = sources[recorded][writer], True
    elif recorded is not None:
        # Unreviewed source_sha fields may describe inference inputs, not writers.
        role = "input_source"
    return dict(
        metadata_path=str(path),
        metadata_sha256=metadata_sha,
        artifact_sha256=artifact_sha,
        role=role,
        recorded_source_sha=recorded,
        source_blob_sha256=source_sha,
        writer_path=writer,
        source_evidence_verified=verified,
    )


def artifact_entries(value, folder):
    entries = []
    for field in MAP_NAMES:
        mapping = value.get(field)
        if not isinstance(mapping, dict):
            continue
        for name, expected in mapping.items():
            if (
                not isinstance(name, str)
                or not isinstance(expected, str)
                or not HASH.fullmatch(expected)
            ):
                continue
            path = Path(name)
            if path.suffix != ".parquet" or ".." in path.parts:
                continue
            path = path if path.is_absolute() else folder / path
            entries.append((allowed(path), expected))
    return entries


def hash_pass(row, budget):
    path = allowed(row["path"])
    if str(path) != row["path"] or geometry(path) != row["signature"]:
        raise ValueError("original registered Parquet geometry changed")
    budget.authorize(row["signature"][2])
    result = resume.reader.common.hash_file(path, row["signature"], budget.started + SECONDS)
    budget.add(result["bytes_read"])
    return result["whole_file_sha256"]


def projected_bound(value):
    pending, nodes = [value], 0
    while pending:
        item = pending.pop()
        nodes += 1
        if nodes > LIVE_NODES:
            raise ValueError("unchanged 100,000 projected-node bound")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    if len(json.dumps(value).encode()) > LIVE_BYTES:
        raise ValueError("unchanged 8-MiB live projected-metadata bound")


def recover(root):
    revision = clean_source(SOURCE)
    resume.reader.common.cpu_gate(root / "cpu", revision)
    budget = Budget()
    budget.authorize(sum((PARENT / name).stat().st_size for name in BOUND))
    parent = {
        name: resume.reader.common.history.read(PARENT / name, expected)
        for name, expected in BOUND.items()
    }
    budget.add(sum((PARENT / name).stat().st_size for name in BOUND))
    inventory = parent["inventory.json"]
    if (
        inventory["source_sha"] != "ec314f5b24125d4437ea4a64b47f051677aafa80"
        or inventory["counts"]["parquet_occurrences"] != 134
    ):
        raise ValueError("exact qualified V48 residual inventory required")
    rows = [row for row in inventory["rows"] if row["operation"] == "parquet"]
    sources = {}
    claims = {row["path"]: [] for row in rows}
    evidence, unqualified = [], []
    # Only immediate, explicitly named metadata siblings are nominated.
    candidates = sorted(
        {
            Path(row["path"]).parent / name
            for row in rows
            for name in METADATA_NAMES
            if (Path(row["path"]).parent / name).is_file()
        }
    )
    for candidate in candidates:
        try:
            value, actual, signature = metadata(candidate, budget)
            path = allowed(candidate)
            matched = 0
            for target, expected in artifact_entries(value, path.parent):
                if str(target) not in claims:
                    continue
                claims[str(target)].append(
                    writer_claim(path, value, actual, expected, budget, sources)
                )
                matched += 1
            evidence.append(
                dict(path=str(path), sha256=actual, signature=signature, matched_targets=matched)
            )
            projected_bound([claims, evidence, sources])
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            unqualified.append(
                dict(
                    path=str(candidate),
                    error_type=type(error).__name__,
                    status="metadata_not_qualified",
                )
            )
    # The attestation is a snapshot-content proof, never a clean execution claim.
    attestation, att_sha, att_signature = metadata(ATT, budget, ATT_SHA)
    if attestation.get("source_git_sha") != SNAPSHOT:
        raise ValueError("exact recorded snapshot commit required")
    for row in rows:
        path = Path(row["path"])
        for base in SNAPSHOT_ROOTS:
            if not path.is_relative_to(base):
                continue
            relative = str(path.relative_to(base))
            expected = attestation["files"].get(relative)
            if not isinstance(expected, str) or not HASH.fullmatch(expected):
                continue
            blob = git_blob(SNAPSHOT, relative, budget)
            if blob != expected:
                raise ValueError("recorded snapshot file differs from committed Git blob")
            claims[str(path)].append(
                dict(
                    metadata_path=str(ATT),
                    metadata_sha256=att_sha,
                    artifact_sha256=expected,
                    role="committed_snapshot",
                    recorded_source_sha=SNAPSHOT,
                    source_blob_sha256=blob,
                    writer_path=relative,
                    source_evidence_verified=True,
                )
            )
    evidence.append(
        dict(
            path=str(ATT),
            sha256=att_sha,
            signature=att_signature,
            snapshot_canonical_eligible=attestation.get("canonical_eligible"),
        )
    )
    projected_bound([claims, evidence, sources])
    witnesses = []
    for row in rows:
        try:
            started = time.monotonic()
            before, after = hash_pass(row, budget), hash_pass(row, budget)
            transaction = dict(
                path=row["path"],
                signature=row["signature"],
                before_sha256=before,
                after_sha256=after,
                bytes_read=2 * row["signature"][2],
                seconds=time.monotonic() - started,
            )
            witness = classify_binding(
                row, transaction, claims[row["path"]], reviewed_writers=REVIEWED
            )
        except (ValueError, OSError, TimeoutError) as error:
            witness = dict(
                status="binding_transaction_unqualified",
                path=row["path"],
                owner_index=row["owner_index"],
                historical_error_ordinal=row["historical_error_ordinal"],
                original_error_sha256=row["original_error_sha256"],
                error_type=type(error).__name__,
                original_producer_commit=None,
                occurrence_resolved=False,
                fresh_confirmation_authorized=False,
                materialization_authorized=False,
            )
        write_once(
            root / "witnesses" / f"{row['owner_index']}-{row['historical_error_ordinal']}.json",
            witness,
        )
        witnesses.append(witness)
    # A final stable metadata check binds the complete transaction, not just discovery.
    for item in evidence:
        _, actual, signature = metadata(Path(item["path"]), budget, item["sha256"])
        if actual != item["sha256"] or signature != item["signature"]:
            raise ValueError("historical metadata changed before closing the transaction")
    elapsed = budget.check()
    result = dict(
        source_sha=revision,
        parent_bindings={str(PARENT / k): v for k, v in BOUND.items()},
        rows=witnesses,
        binding_claims=claims,
        evidence=evidence,
        unqualified_metadata=unqualified,
        status_counts=dict(Counter(w["status"] for w in witnesses)),
        original_occurrence_counts=inventory["counts"],
        newly_resolved_occurrences=0,
        targeted_provenance_metadata_files=len(evidence),
        combined_input_bytes=budget.input_bytes,
        wall_seconds=elapsed,
        verified_writer_closures=sources,
        completed_JSON_identity_proofs_replayed=0,
        credential_bodies_read=False,
        parquet_rows_decoded=0,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        historical_media_exposure_resolved=False,
        semantic_closed_writer_proofs_completed=False,
    )
    projected_bound(result)
    resume.require_artifact_capacity(root, LIVE_BYTES)
    write_once(root / "recovered_bindings.json", result)
    return dict(
        status_counts=result["status_counts"], fresh_confirmation_authorized=False, GPU_jobs=0
    )


def analyze(root):
    revision = clean_source(SOURCE)
    resume.reader.common.cpu_gate(root / "cpu", revision)
    path = root / "recovered_bindings.json"
    if not path.is_file():
        return dict(
            source_sha=revision,
            status="recovery_failed_or_incomplete",
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
            GPU_jobs=0,
        )
    result = resume.reader.common.history.read(path)
    if result["source_sha"] != revision or result["newly_resolved_occurrences"] != 0:
        raise ValueError("binding discovery cannot silently resolve semantic coverage")
    return dict(
        source_sha=revision,
        recovered_bindings_sha256=sha(path),
        status_counts=result["status_counts"],
        original_occurrence_counts=result["original_occurrence_counts"],
        newly_resolved_occurrences=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        GPU_jobs=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "recover", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only write-once output root required")
    try:
        result = resume.cpu(root) if args.mode == "cpu" else globals()[args.mode](root)
        if args.mode == "analyze":
            write_once(root / "decision.json", result)
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error="CPU binding recovery failed closed",
                fresh_confirmation_authorized=False,
                GPU_jobs=0,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
