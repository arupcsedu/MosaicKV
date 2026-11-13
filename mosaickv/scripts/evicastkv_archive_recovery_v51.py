#!/usr/bin/env python3
"""Verify originally declared source capsules; never rerun artifact body proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import ClassVar

import evicastkv_recover_bindings_v49 as reader
from evicastkv_source_zip_v51 import inspect_zip

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.source_archive_bindings_v51 import (
    augment_binding,
    validate_archive,
    validate_generator,
)

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v50/run-20261113-b5722db-1")
PARENT_SHA = "939df6cbef46858cb59820772ff9d251a904c0edc472aee8dc82859455abe471"
HISTORY = reader.HISTORY
RETRO = HISTORY / "mosaickv/coco-cross-budget-retrospective-20261022-v2"
REPORT = RETRO / "report.json"
REPORT_SHA = "3586040e4830c3443b1d1ab8a2495e7ae0250a5916d66530396ec8bd6c9232bf"
ZIP = RETRO / "analysis_source.zip"
ZIP_SHA = "5acdc804fb0d83314fb08c0dd9cd238c4b095e4192dfed8786ea918bcb3ea8c5"
ADHOC = HISTORY / "evicastkv/lossless-lifetime-consumed-20261030-v4/derived_summary_v1"
MANIFEST = ADHOC / "manifest.json"
MANIFEST_SHA = "16121a6f6f28b0f1d901dc0220a0441556a8e621f58979f9e1fcd10477c4b476"


class CapsuleReader(reader.ProvenanceReader):
    """Extend the established lexical whitelist, not its resource limits."""

    FIELDS = reader.ProvenanceReader.FIELDS | frozenset(
        (
            "analysis_source_sha",
            "analysis_worktree_dirty",
            "source_code_hashes",
            "generator_sha256",
            "producer_source_sha",
            "source_analysis_manifest_sha256",
        )
    )
    ALIASES: ClassVar[dict[str, str]] = {
        "analysis_source_sha": "source_sha",
        "analysis_worktree_dirty": "canonical_eligible",
        "source_code_hashes": "files",
        "producer_source_sha": "source_sha",
    }

    def value(self, *, key="", keep=True, depth=0, classification=None, dataset_object=False):
        selected = self.ALIASES.get(key, key) if depth == 1 else key
        return super().value(
            key=selected,
            keep=keep,
            depth=depth,
            classification=classification,
            dataset_object=dataset_object,
        )


def metadata(path, expected, budget):
    path = reader.allowed(path)
    before = reader.geometry(path)
    if before[2] > reader.LIVE_BYTES:
        raise ValueError("unchanged metadata bound exceeded")
    budget.authorize(before[2] + 1)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if reader.lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("registered metadata descriptor differs")
        parser = CapsuleReader(
            stream,
            deadline=budget.started + 210,
            maximum_bytes=max(1, before[2]),
            chunk_bytes=256 * 2**10,
        )
        try:
            parser.spaces()
            if parser.peek() != 123:
                raise ValueError("one metadata object required")
            result = parser.value()
            parser.spaces()
            if parser.peek() is not None:
                raise ValueError("trailing metadata payload")
            actual = parser.hash.hexdigest()
        finally:
            budget.add(parser.stats["bytes_read"])
        if reader.lexical._signature(os.fstat(stream.fileno())) != before:
            raise ValueError("metadata changed during projection")
    if (
        reader.geometry(path) != before
        or parser.stats["bytes_read"] != before[2]
        or actual != expected
    ):
        raise ValueError("original metadata hash or signature differs")
    if not isinstance(result, dict):
        raise ValueError("registered provenance projection required")
    return result


def artifact_map(values, folder):
    if not isinstance(values, dict):
        raise ValueError("registered artifact hash map required")
    result = {}
    for name, expected in values.items():
        if not isinstance(name, str) or reader.HASH.fullmatch(str(expected)) is None:
            raise ValueError("explicit original artifact hash entry required")
        path = Path(name)
        if str(path) != name or (
            ".." in path.parts
            or ".codex" in path.parts
            or path.name in {"auth.json", "credentials.json"}
        ):
            raise ValueError("protected or noncanonical artifact metadata path")
        path = path if path.is_absolute() else folder / path
        # Lexical alias normalization; missing historical non-target outputs are
        # not permission to open, regenerate or drop them from the original map.
        if str(path).startswith("/scratch/djy8hg/"):
            path = Path("/sfs/weka/scratch/djy8hg") / path.relative_to("/scratch/djy8hg")
        if not path.is_relative_to(HISTORY) or str(path) in result:
            raise ValueError("out-of-scope or duplicate artifact binding")
        result[str(path)] = expected
    return result


def generator_witness(path, expected, budget):
    path = reader.allowed(path)
    before = reader.geometry(path)
    if path.suffix != ".py" or before[2] > reader.LIVE_BYTES:
        raise ValueError("only bounded declared generator code may be opened")
    hashes = []
    for _ in range(2):
        budget.authorize(before[2] + 1)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            if reader.lexical._signature(os.fstat(stream.fileno())) != before:
                raise ValueError("generator descriptor differs")
            raw = stream.read(before[2] + 1)
            budget.add(len(raw))
            if (
                len(raw) != before[2]
                or reader.lexical._signature(os.fstat(stream.fileno())) != before
            ):
                raise ValueError("generator changed during independent hash transaction")
        hashes.append(hashlib.sha256(raw).hexdigest())
    if reader.geometry(path) != before or hashes != [expected, expected]:
        raise ValueError("original generator code hash differs")
    return dict(path=str(path), sha256=expected, bytes=before[2])


def recover(root):
    revision = clean_source(SOURCE)
    reader.resume.reader.common.cpu_gate(root / "cpu", revision)
    budget = reader.Budget()
    path = PARENT / "recovered_bindings.json"
    budget.authorize(path.stat().st_size)
    parent = reader.resume.reader.common.history.read(path, PARENT_SHA)
    budget.add(path.stat().st_size)
    if (
        parent["source_sha"] != "b5722db36c7c04d63a8e9fc3062bc2d0e509c279"
        or len(parent["rows"]) != 134
        or parent["parquet_bodies_reopened"] != 0
    ):
        raise ValueError("complete immutable V50 proof-reuse ledger required")
    report = metadata(REPORT, REPORT_SHA, budget)
    report["derived_artifact_hashes"] = artifact_map(
        report["derived_artifact_hashes"], REPORT.parent
    )
    if report["derived_artifact_hashes"].get(str(ZIP)) != ZIP_SHA:
        raise ValueError("original declared source ZIP hash differs")
    witness = inspect_zip(ZIP, ZIP_SHA, budget, allowed_root=HISTORY)
    archive = validate_archive(
        {
            key: report[key]
            for key in (
                "analysis_source_sha",
                "analysis_worktree_dirty",
                "source_code_hashes",
                "derived_artifact_hashes",
            )
        },
        witness,
    )
    reader.resume.require_artifact_capacity(root, reader.LIVE_BYTES)
    write_once(
        root / "source_archive_witness.json",
        dict(
            source_sha=revision,
            original_metadata_sha256=REPORT_SHA,
            witness=witness,
            closure=archive,
            fresh_confirmation_authorized=False,
            GPU_jobs=0,
        ),
    )
    manifest = metadata(MANIFEST, MANIFEST_SHA, budget)
    manifest["outputs"] = artifact_map(manifest["outputs"], MANIFEST.parent)
    generator = generator_witness(ADHOC / "generate.py", manifest["generator_sha256"], budget)
    adhoc = validate_generator(
        {key: manifest[key] for key in ("generator_sha256", "outputs", "producer_source_sha")},
        generator,
    )
    reader.resume.require_artifact_capacity(root, reader.LIVE_BYTES)
    write_once(
        root / "generator_source_witness.json",
        dict(
            source_sha=revision,
            original_metadata_sha256=MANIFEST_SHA,
            witness=generator,
            closure=adhoc,
            fresh_confirmation_authorized=False,
            GPU_jobs=0,
        ),
    )
    evidence = ((report["derived_artifact_hashes"], archive), (manifest["outputs"], adhoc))
    results, recovered_paths = [], []
    for previous_row in parent["rows"]:
        row = previous_row
        for outputs, closure in evidence:
            if row["path"] in outputs:
                # Reuse the qualified V49/V50 transaction. No Parquet is opened
                # or hashed; historical bytes come from the immutable witness.
                actual = reader.allowed(row["path"])
                if str(actual) != row["path"] or reader.geometry(actual) != row["signature"]:
                    raise ValueError("completed source signature changed; do not replay it")
                row = augment_binding(previous_row, outputs, closure)
                recovered_paths.append(row["path"])
        results.append(row)
    if len(recovered_paths) != 4:
        raise ValueError("the three dirty-archive and one ad-hoc source links must all qualify")
    # Independently close the two metadata transactions too, without rerunning
    # their already completed JSON identity/exclusion proofs.
    metadata(REPORT, REPORT_SHA, budget)
    metadata(MANIFEST, MANIFEST_SHA, budget)
    result = dict(
        source_sha=revision,
        parent_report_path=str(path),
        parent_report_sha256=PARENT_SHA,
        rows=results,
        source_closures=[archive, adhoc],
        archive_witness=witness,
        generator_witness=generator,
        original_metadata_bindings={str(REPORT): REPORT_SHA, str(MANIFEST): MANIFEST_SHA},
        recovered_paths=recovered_paths,
        status_counts=dict(Counter(x["status"] for x in results)),
        original_occurrence_counts=parent["original_occurrence_counts"],
        newly_resolved_occurrences=0,
        reused_hash_transactions=134,
        parquet_bodies_reopened=0,
        original_JSON_identity_proofs_replayed=0,
        credential_bodies_read=False,
        historical_code_executed=False,
        archive_extraction_to_disk=False,
        combined_input_bytes=budget.input_bytes,
        wall_seconds=budget.check(),
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        historical_media_exposure_resolved=False,
        paper_campaign_authorized=False,
    )
    reader.projected_bound(result)
    reader.resume.require_artifact_capacity(root, reader.LIVE_BYTES)
    write_once(root / "recovered_bindings.json", result)
    return dict(
        status_counts=result["status_counts"],
        recovered=len(recovered_paths),
        fresh_confirmation_authorized=False,
        GPU_jobs=0,
    )


def analyze(root):
    revision = clean_source(SOURCE)
    reader.resume.reader.common.cpu_gate(root / "cpu", revision)
    path = root / "recovered_bindings.json"
    if not path.exists():
        return dict(
            status="archive_recovery_failed_or_incomplete",
            source_sha=revision,
            fresh_confirmation_authorized=False,
            GPU_jobs=0,
        )
    report = reader.resume.reader.common.history.read(path)
    if report["source_sha"] != revision or report["parquet_bodies_reopened"] != 0:
        raise ValueError("archive recovery must reuse the completed artifact proofs")
    return dict(
        source_sha=revision,
        report_sha256=sha(path),
        recovered_paths=report["recovered_paths"],
        status_counts=report["status_counts"],
        reused_hash_transactions=134,
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
        raise ValueError("scratch-only immutable output required")
    try:
        result = reader.resume.cpu(root) if args.mode == "cpu" else globals()[args.mode](root)
        if args.mode == "analyze":
            write_once(root / "decision.json", result)
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error="source capsule recovery failed closed",
                fresh_confirmation_authorized=False,
                GPU_jobs=0,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
