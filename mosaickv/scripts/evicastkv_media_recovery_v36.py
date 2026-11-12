#!/usr/bin/env python3
"""Bind new cleanup/backup facts to V35 without rescanning video or waiving gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import deque
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v35/run-20261111-1/history/review/media-gap.json")
PARENT_SHA = "2f50562f890f63cf654c657257c771dfe1080aa979d3014baec5a491255e81d4"
CLEANUP = Path("/scratch/djy8hg/cleanup_targets.txt")
CLEANUP_SHA = "97a98f5f4c105208aee8c88cb6140b74c36290ecda01caad51346128112d7afe"
RUNS = Path("/scratch/djy8hg/runs/mosaickv")
EXPERIMENTS = (
    "main-quality-preflight-canaries-v5-da10309",
    "main-quality-preflight-canaries-v9-4d0a456",
    "main-quality-preflight-canaries-v11-da3223a",
    "main-quality-preflight-video-v6-45d03a6",
    "main-quality-preflight-video-v7-a05b393",
    "main-quality-preflight-video-v8-0be3133",
)
CATALOG_ROOTS = (
    (RUNS / "archives", False),
    (Path("/scratch/djy8hg/workdir"), True),
    (Path("/scratch/djy8hg/worktrees"), True),
    (Path("/scratch/djy8hg/tmp"), True),
    (Path("/scratch/djy8hg/audits"), True),
    (Path("/scratch/djy8hg/project"), True),
)
PROJECT_NAME = re.compile(r"^(?:mosaickv|evicastkv|fox)(?:[-_.].*|$)", re.I)
RECOVERY_NAME = re.compile(
    r"source|snapshot|backup|archive|pilot|preflight|cleanup|delet|migrat", re.I
)
RECOVERY_SUFFIX = re.compile(
    r"\.(?:bundle|patch|diff|tar(?:\.(?:gz|xz|bz2|zst))?|zip|json|txt|log)$", re.I
)
SKIP_DIRECTORIES = {
    ".git",
    "media",
    "images",
    "videos",
    "datasets",
    "models",
    "weights",
    "hub",
    "data",
}
REQUIRED_FALSE = (
    "globally_fresh",
    "historical_media_exposure_resolved",
    "source_group_disjointness_verified",
    "materialization_authorized",
    "fresh_confirmation_authorized",
    "paper_campaign_authorized",
)


def signature(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def bound_file(path, expected, *, maximum):
    if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise ValueError("explicit SHA-256 required")
    path = Path(path)
    before = signature(path)
    if before[2] > maximum:
        raise ValueError("registered metadata byte bound exceeded")
    value = path.read_bytes()
    if signature(path) != before or hashlib.sha256(value).hexdigest() != expected:
        raise ValueError("immutable metadata hash or identity changed")
    return value


def _allowed(path, roots):
    for root, shared in roots:
        if path.is_relative_to(root):
            relative = path.relative_to(root)
            if not shared or not relative.parts or PROJECT_NAME.fullmatch(relative.parts[0]):
                return True
    return False


def catalog(roots=CATALOG_ROOTS, *, max_entries=20000, max_seconds=30, maximum_depth=2):
    """Filename/stat metadata only; explicitly project-scoped and non-global."""
    if (
        type(max_entries) is not int
        or not 1 <= max_entries <= 20000
        or not 0 < max_seconds <= 30
        or maximum_depth != 2
    ):
        raise ValueError("registered depth-two, 20000-entry, 30-second bounds required")
    begun = time.monotonic()
    allowed, missing, errors, candidates = [], [], [], []
    for root, shared in roots:
        root = Path(root)
        if not root.exists():
            missing.append(str(root))
        else:
            canonical = root.resolve(strict=True)
            if not canonical.is_relative_to(root.parent.resolve(strict=True)):
                errors.append(
                    dict(path=str(root), reason="catalog_root_alias_outside_declared_parent")
                )
                continue
            allowed.append((canonical, bool(shared)))
    pending = deque((root, 0, shared) for root, shared in allowed)
    visited, seen, examined, skipped, duplicates = set(), set(), 0, 0, 0
    stopped = None
    while pending and stopped is None:
        path, depth, shared = pending.popleft()
        if path in visited:
            duplicates += 1
            continue
        visited.add(path)
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if time.monotonic() - begun > max_seconds:
                        stopped = "time_limit"
                        break
                    if examined >= max_entries:
                        stopped = "entry_limit"
                        break
                    examined += 1
                    if depth == 0 and shared and not PROJECT_NAME.fullmatch(entry.name):
                        skipped += 1
                        continue
                    eligible = bool(
                        RECOVERY_NAME.search(entry.name) and RECOVERY_SUFFIX.search(entry.name)
                    ) or entry.name in {"provenance.json", "manifest.json", ".git"}
                    is_directory = entry.is_dir(follow_symlinks=False)
                    is_alias = entry.is_symlink()
                    if (
                        not is_directory
                        and not eligible
                        and not (depth == 0 and shared and is_alias)
                    ):
                        skipped += 1
                        continue
                    try:
                        canonical = Path(entry.path).resolve(strict=True)
                    except (FileNotFoundError, RuntimeError, OSError) as exc:
                        errors.append(
                            dict(
                                path=entry.path,
                                reason="broken_or_unresolvable_alias",
                                error_type=type(exc).__name__,
                            )
                        )
                        continue
                    if not _allowed(canonical, allowed):
                        errors.append(
                            dict(path=entry.path, reason="canonical_target_outside_project_scope")
                        )
                        continue
                    if canonical in seen:
                        duplicates += 1
                        continue
                    seen.add(canonical)
                    info = canonical.stat()
                    if canonical.is_dir():
                        if entry.name == ".git":
                            candidates.append(
                                dict(
                                    path=entry.path,
                                    canonical_path=str(canonical),
                                    kind="project_git_database_directory",
                                    body_read=False,
                                )
                            )
                        elif (
                            depth + 1 < maximum_depth
                            and entry.name not in SKIP_DIRECTORIES
                            and canonical.name not in SKIP_DIRECTORIES
                        ):
                            pending.append((canonical, depth + 1, False))
                    elif eligible:
                        candidates.append(
                            dict(
                                path=entry.path,
                                canonical_path=str(canonical),
                                kind="recovery_filename_metadata_only",
                                bytes=info.st_size,
                                mtime_ns=info.st_mtime_ns,
                                body_read=False,
                            )
                        )
        except OSError as exc:
            errors.append(
                dict(
                    path=str(path), reason="directory_metadata_error", error_type=type(exc).__name__
                )
            )
    return dict(
        complete=stopped is None and not errors,
        stop_reason=stopped,
        errors=errors,
        root_paths=[str(root) for root, _shared in allowed],
        missing_root_paths=missing,
        maximum_depth=maximum_depth,
        maximum_entries=max_entries,
        maximum_seconds=max_seconds,
        examined_entries=examined,
        unselected_or_nonrecovery_entries_skipped=skipped,
        duplicate_canonical_targets=duplicates,
        candidates=candidates,
        candidate_bodies_read=False,
        media_decoded=False,
        global_exhaustion_proof=False,
        archival_presence_is_not_historical_exposure_proof=True,
    )


def bind_inventory(path, expected):
    path = Path(path)
    if path.name not in {"combined_identity_inventory.json", "typed_asset_inventory.json"}:
        raise ValueError("only explicitly named projected historical inventories may be bound")
    canonical = path.resolve(strict=True)
    allowed = [(root.resolve(), shared) for root, shared in CATALOG_ROOTS]
    if not _allowed(canonical, allowed):
        raise ValueError("inventory alias escapes project-owned metadata scope")
    value = bound_file(path, expected, maximum=128 * 2**20)
    return dict(
        path=str(path),
        canonical_path=str(canonical),
        sha256=expected,
        bytes=len(value),
        contents_exported=False,
        media_exposure_proof_inferred=False,
    )


def recover(*, global_inventory=None, global_inventory_sha256=None):
    revision = clean_source(SOURCE)
    if (global_inventory is None) != (global_inventory_sha256 is None):
        raise ValueError("global inventory path and SHA must be supplied together")
    previous = json.loads(bound_file(PARENT, PARENT_SHA, maximum=2**20))
    cleanup = bound_file(CLEANUP, CLEANUP_SHA, maximum=65536).decode("utf-8").splitlines()
    cleanup_lines = {name: index + 1 for index, name in enumerate(cleanup)}
    roots = [(item["directory"], "pilot_index") for item in previous["pilot_roots"]]
    roots += [(str(RUNS / "experiments" / name), "experiment_outputs") for name in EXPERIMENTS]
    if len(roots) != 12 or len(set(name for name, _kind in roots)) != 12:
        raise ValueError("all twelve original index/output roots required")
    path_facts = []
    for name, kind in roots:
        if name not in cleanup_lines:
            raise ValueError("original missing root is not bound by the reviewed cleanup list")
        path = Path(name)
        fact = dict(
            path=name,
            kind=kind,
            cleanup_list_line=cleanup_lines[name],
            cleanup_intent_recorded=True,
            cleanup_execution_proven=False,
            lexists=os.path.lexists(path),
            exists=path.exists(),
        )
        if fact["lexists"]:
            try:
                fact["canonical_path"] = str(path.resolve(strict=True))
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                fact["path_resolution_error"] = type(exc).__name__
        path_facts.append(fact)
    inventory = (
        None
        if global_inventory is None
        else bind_inventory(global_inventory, global_inventory_sha256)
    )
    found = catalog()
    report = dict(
        source_sha=revision,
        status="bounded_cleanup_and_backup_recovery_not_freshness_certification",
        complete=found["complete"],
        prior_gap_report=dict(
            path=str(PARENT), sha256=PARENT_SHA, reused_without_trace_rescan=True
        ),
        cleanup_intent_evidence=dict(
            path=str(CLEANUP),
            sha256=CLEANUP_SHA,
            bytes=CLEANUP.stat().st_size,
            intent_not_execution_receipt=True,
        ),
        original_path_facts=path_facts,
        bounded_project_backup_catalog=found,
        new_global_inventory_binding=inventory,
        previous_trace_projection_reused=previous["previously_skipped_trace"],
        existing_missing_source_roots_and_worktree_fingerprint_not_waived=True,
        new_proof_links_require_separate_validation=True,
        no_exposure_certified=False,
        source_groups_remain_unresolved=True,
        confirmation_blocked=True,
        globally_fresh=False,
        historical_media_exposure_resolved=False,
        source_group_disjointness_verified=False,
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
        GPU_jobs_submitted=0,
        images_videos_decoded=0,
        models_loaded=False,
    )
    return report


def publish(output, report):
    if any(report.get(name) is not False for name in REQUIRED_FALSE):
        raise ValueError("recovery evidence cannot waive unresolved source-group/freshness gates")
    write_once(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-inventory", type=Path)
    parser.add_argument("--global-inventory-sha256")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/scratch/djy8hg").resolve()):
        parser.error("immutable scratch output required; never home")
    report = recover(
        global_inventory=args.global_inventory, global_inventory_sha256=args.global_inventory_sha256
    )
    publish(output, report)
    print(
        json.dumps(
            dict(
                output=str(output),
                report_sha256=digest(report),
                status=report["status"],
                globally_fresh=False,
            )
        )
    )


if __name__ == "__main__":
    main()
