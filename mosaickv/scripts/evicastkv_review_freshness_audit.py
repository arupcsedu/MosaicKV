#!/usr/bin/env python3
"""Bounded metadata-only historical coverage audit, never a freshness license.

Canonical small JSON manifests are scanned recursively without reading raw
prediction/scheduler logs, tensor packets or images. Benchmark-value fields
are lexically redacted before JSON decoding. Exhausting this explicitly
limited scope is distinct from exhausting all prior identity/media evidence.
No IDs are selected, materialized, scored or declared fresh by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from evicastkv_boundary_operators import _STRING, _skip_json_value

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
RUNS = Path("/scratch/djy8hg/runs")
ROOTS = (RUNS / "mosaickv", RUNS / "evicastkv")
AVAILABILITY = (
    RUNS / "evicastkv/fox-review-20261028-writing-closure-v2/coco_availability/report.json"
)
RESOLUTIONS = RUNS / "evicastkv/fox-review-20261028-provenance-reviewed-v1/root_resolutions.json"
REVIEWED = {
    str(AVAILABILITY): "154c5f7cb9913f933c49a78c0c43d171b4b4189260f0286f23e63e35f47b5a3f",
    str(RESOLUTIONS): "d1505bcd7fa6feb090e558e95f9691fd49a17f8102a9e7835a7e2d3ccbfd417b",
    str(
        AVAILABILITY.parent / "extraction_command.json"
    ): "ea31f5ae181950f2906922e39d8e820041b45158029f397d0e39dbdec86815bf",
}
NAMES = frozenset(
    {
        "plan.json",
        "protocol.json",
        "protocol_lock.json",
        "materialization.json",
        "ledger.json",
        "registration.json",
        "historical_ids.json",
        "prior-materialization-exclusions.json",
        "controller-base.json",
        "launch.json",
        "submission.json",
    }
)
FORBIDDEN = frozenset(
    {
        "answer",
        "answers",
        "reference",
        "references",
        "score",
        "scores",
        "task_score",
        "predictions",
        "prediction",
        "response",
        "responses",
        "prompt",
        "choices",
        "teacher_forcing_target",
        "logits",
        "attention",
        "pixel_values",
        "image",
        "images",
        "video",
        "generated_tokens",
        "token_ids",
        "failure_identity",
        "error",
        "errors",
    }
)
COCO = re.compile(r"COCO_(?:val|train)2014_\d{12}\.jpg")
SHA256 = re.compile(r"[0-9a-f]{64}")


def metadata_projection(raw):
    """Forbidden values are skipped lexically, never constructed as JSON objects."""
    text = raw.decode("utf-8")
    pieces, last, cursor = [], 0, 0
    while cursor < len(text):
        if text[cursor] != '"':
            cursor += 1
            continue
        match = _STRING.match(text, cursor)
        if match is None:
            raise ValueError("malformed canonical metadata JSON string")
        end, colon = match.end(), match.end()
        while colon < len(text) and text[colon].isspace():
            colon += 1
        if colon < len(text) and text[colon] == ":" and json.loads(match.group()) in FORBIDDEN:
            value_end = _skip_json_value(text, colon + 1)
            pieces.extend((text[last : colon + 1], "null"))
            last, cursor = value_end, value_end
        else:
            cursor = end
    pieces.append(text[last:])
    return json.loads("".join(pieces))


def canonical_name(name):
    return name in NAMES or name.endswith("-ledger.json") or name.endswith("_ledger.json")


class Budget:
    def __init__(self, *, seconds=600, maximum_entries=300000, maximum_bytes=128 * 2**20):
        if not 0 < seconds <= 900 or maximum_entries < 1 or maximum_bytes < 1:
            raise ValueError("positive CPU bounds with at most fifteen minutes required")
        self.deadline = time.monotonic() + seconds
        self.maximum_entries, self.maximum_bytes = maximum_entries, maximum_bytes
        self.entries, self.bytes = 0, 0

    def check(self, *, entries=0, bytes_count=0):
        self.entries += entries
        self.bytes += bytes_count
        if (
            time.monotonic() > self.deadline
            or self.entries > self.maximum_entries
            or self.bytes > self.maximum_bytes
        ):
            raise TimeoutError("registered metadata audit time/entry/byte bound reached")


def within(path, roots):
    return any(path.is_relative_to(root) for root in roots)


def enumerate_canonical(roots, budget):
    """Strict canonical symlinks; aliases deduplicated before traversal/read."""
    allowed = tuple(Path(root).resolve(strict=True) for root in roots)
    if len(set(allowed)) != len(allowed) or any(not root.is_dir() for root in allowed):
        raise ValueError("distinct existing run-directory roots required")
    queue, seen_dirs, files, errors, aliases = list(allowed), set(), {}, [], []
    ignored_payloads, complete = 0, True
    try:
        while queue:
            budget.check()
            directory = queue.pop()
            if directory in seen_dirs:
                continue
            seen_dirs.add(directory)
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        budget.check(entries=1)
                        try:
                            path = Path(entry.path)
                            target = path.resolve(strict=True) if entry.is_symlink() else path
                            if not within(target, allowed):
                                raise ValueError("symlink target outside allowed historical roots")
                            if entry.is_symlink():
                                aliases.append(dict(alias=str(path), canonical_target=str(target)))
                            if target.is_dir():
                                if target not in seen_dirs:
                                    queue.append(target)
                            elif canonical_name(path.name) or canonical_name(target.name):
                                if not target.is_file():
                                    raise ValueError("canonical metadata must be a regular file")
                                files[str(target)] = target
                            elif target.suffix in {".json", ".jsonl", ".parquet", ".yaml", ".yml"}:
                                ignored_payloads += 1
                        except (OSError, RuntimeError, ValueError) as error:
                            errors.append(
                                dict(path=entry.path, error=f"{type(error).__name__}: {error}")
                            )
            except OSError as error:
                errors.append(dict(path=str(directory), error=f"{type(error).__name__}: {error}"))
    except TimeoutError as error:
        complete = False
        errors.append(dict(error=str(error)))
    return dict(
        files=sorted(files),
        errors=errors,
        aliases=aliases,
        complete=complete,
        ignored_structured_payloads=ignored_payloads,
        visited_directories=len(seen_dirs),
        pending_directories=len(queue),
    )


def read_stable(path, budget, *, maximum_file_bytes=8 * 2**20):
    before = path.stat()
    if before.st_size > maximum_file_bytes:
        raise ValueError("canonical manifest exceeds registered per-file byte bound")
    budget.check(bytes_count=before.st_size)
    with path.open("rb") as stream:
        raw = stream.read(maximum_file_bytes + 1)
    after = path.stat()
    if len(raw) != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("canonical manifest changed during bounded read")
    return metadata_projection(raw), hashlib.sha256(raw).hexdigest()


def identities(value):
    """Return conservative observed identity/media metadata, not eligibility."""
    coco, sample_ids, media_hashes = set(), set(), set()

    def visit(item, *, parent_key="", in_media=False):
        if isinstance(item, dict):
            for key, child in item.items():
                if key not in FORBIDDEN:
                    visit(child, parent_key=key, in_media=in_media or key == "media")
        elif isinstance(item, list):
            for child in item:
                visit(child, parent_key=parent_key, in_media=in_media)
        elif isinstance(item, str):
            coco.update(COCO.findall(item))
            if parent_key in {
                "sample_id",
                "sample_ids",
                "excluded_ids",
                "selected_ids",
                "COCO_consumed_ids",
                "MileBench_consumed_ids",
            }:
                sample_ids.add(item)
            if (parent_key == "sha256" and in_media) or parent_key in {
                "media_sha256",
                "image_sha256",
                "media_group_ids",
            }:
                candidate = item.removeprefix("input-media:").removeprefix("original-image:")
                if SHA256.fullmatch(candidate):
                    media_hashes.add(candidate)

    visit(value)
    return coco, sample_ids, media_hashes


def reviewed_evidence(budget, *, bindings=REVIEWED):
    loaded = {}
    for name, expected in bindings.items():
        value, actual = read_stable(Path(name), budget)
        if actual != expected:
            raise ValueError("reviewed provenance-resolution evidence changed")
        loaded[name] = value
    availability, resolutions = loaded[str(AVAILABILITY)], loaded[str(RESOLUTIONS)]
    if availability.get("unresolved_workload_roots"):
        raise ValueError("reviewed later root classifications are not resolved")
    recovered, bound = [], dict(bindings)
    for classification in availability["classifications"]:
        evidence = Path(classification["evidence"])
        expected = availability["bound_files"].get(str(evidence), bindings.get(str(evidence)))
        if evidence.stat().st_size > 8 * 2**20:
            raise ValueError("classification evidence exceeds small-metadata scope")
        budget.check(bytes_count=evidence.stat().st_size)
        if expected is None or sha(evidence) != expected:
            raise ValueError("reviewed classification lacks unchanged evidence")
        bound[str(evidence)] = expected
        recovered.append(classification)
    previous_ids = set()
    for path, expected in availability["bound_files"].items():
        if Path(path).name not in {"mosaickv-inventory.json", "evicastkv-inventory.json"}:
            continue
        value, actual = read_stable(Path(path), budget)
        if actual != expected:
            raise ValueError("saved exclusion inventory changed")
        bound[path] = actual
        previous_ids.update(identities(value)[1])
        previous_ids.update(identities(value)[0])
    previous_ids.update(sid for row in resolutions for sid in row["excluded_ids"])
    return dict(
        classifications=recovered,
        prior_excluded_ids=sorted(previous_ids),
        bindings=bound,
        old_id_only_available=availability["id_only_available"],
        old_availability_is_not_media_certification=True,
    )


def audit(roots, *, budget=None, previous=None):
    budget = budget or Budget()
    started = time.monotonic()
    first = enumerate_canonical(roots, budget)
    bindings, errors = {}, list(first["errors"])
    coco, samples, media = set(), set(), set()
    for name in first["files"]:
        try:
            value, file_sha = read_stable(Path(name), budget)
            bindings[name] = file_sha
            observed_coco, observed_samples, observed_media = identities(value)
            coco.update(observed_coco)
            samples.update(observed_samples)
            media.update(observed_media)
        except (TimeoutError, OSError, ValueError, UnicodeError) as error:
            errors.append(dict(path=name, error=f"{type(error).__name__}: {error}"))
            if isinstance(error, TimeoutError):
                break
    # Detect manifest additions/removals as well as content changes. Both passes
    # share the same finite budget; exhaustion is a documented negative audit.
    second = enumerate_canonical(roots, budget)
    errors.extend(second["errors"])
    names_stable = first["files"] == second["files"]
    if not names_stable:
        errors.append(dict(error="canonical manifest set changed during audit"))
    for name, expected in bindings.items():
        try:
            _value, current_sha = read_stable(Path(name), budget)
            if current_sha != expected:
                raise ValueError("canonical manifest changed after identity projection")
        except (TimeoutError, OSError, ValueError) as error:
            errors.append(dict(path=name, error=f"{type(error).__name__}: {error}"))
            if isinstance(error, TimeoutError):
                break
    previous = previous or dict(classifications=[], prior_excluded_ids=[], bindings={})
    for sid in previous["prior_excluded_ids"]:
        samples.add(sid)
        coco.update(COCO.findall(sid))
    complete = first["complete"] and second["complete"] and names_stable and not errors
    return dict(
        status="metadata_audit_complete_confirmation_blocked"
        if complete
        else "metadata_audit_incomplete_confirmation_blocked",
        scope_roots=[str(Path(root).resolve()) for root in roots],
        canonical_manifest_scope_complete=complete,
        canonical_manifest_name_rule=[*sorted(NAMES), "*-ledger.json", "*_ledger.json"],
        canonical_manifest_set_stable=names_stable,
        canonical_manifests_discovered=len(first["files"]),
        canonical_manifests_read=len(bindings),
        bound_files=bindings,
        canonical_bindings_sha256=digest(bindings),
        prior_reviewed_bindings=previous["bindings"],
        recovered_classifications=previous["classifications"],
        prior_resolved_roots_not_reintroduced_as_ambiguities=True,
        observed_coco_ids=sorted(coco),
        observed_sample_ids=sorted(samples),
        observed_media_hashes=sorted(media),
        symlink_aliases=first["aliases"],
        errors=errors,
        unparsed_structured_payloads=first["ignored_structured_payloads"],
        visited_directories=first["visited_directories"],
        pending_directories=first["pending_directories"],
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        global_source_group_aliases_certified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        cohort_selected=False,
        models_loaded=False,
        benchmark_values_lexically_redacted=True,
        raw_prediction_logs_parsed=False,
        wall_seconds=time.monotonic() - started,
        bounded_bytes_read=budget.bytes,
        coverage_gaps=[
            "Canonical-name scope excludes prediction JSONL/Parquet and noncanonical legacy "
            "manifests; full historical ID coverage is not certified.",
            "Observed SHA-256 metadata is not exhaustive underlying-image/pixel/source-group "
            "coverage across all datasets.",
            "Video/episode/document global aliases are not certified; absence of observed "
            "overlap is not freshness.",
            "The old 36,408 ID-only available COCO count is not a fresh-media certificate.",
        ],
    )


def run(root):
    revision = clean_source(SOURCE)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch output directory required")
    budget = Budget()
    previous = reviewed_evidence(budget)
    report = audit(ROOTS, budget=budget, previous=previous)
    report["source_sha"] = revision
    write_once(root / "inventory.json", report)
    gate = dict(
        status="passed_metadata_audit_freshness_blocked"
        if report["canonical_manifest_scope_complete"]
        else "blocked_metadata_audit_incomplete",
        source_sha=revision,
        inventory_sha256=sha(root / "inventory.json"),
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        models_loaded=False,
        coverage_gaps=report["coverage_gaps"],
    )
    write_once(root / "GATE_freshness_audit.json", gate)
    write_once(
        root / "readiness.json",
        dict(
            status="confirmation_blocked",
            **{
                key: report[key]
                for key in (
                    "coverage_gaps",
                    "fresh_confirmation_authorized",
                    "cohort_selected",
                    "cross_dataset_media_certified",
                )
            },
        ),
    )
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root)))
