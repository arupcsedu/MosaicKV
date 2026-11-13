"""Exact scope factoring for bounded historical-identity checkpoints.

Only the repeated scoped-ID layout changes. Global sample IDs, all other
identity fields, proof digests, occurrence rows and permission flags remain
unchanged. The ordinary reader rejects the encoded dictionary rather than
mistaking it for an empty identity list; callers must explicitly unpack it.

This module performs no file, scheduler, model or dataset access. The existing
512-MiB checkpoint writer remains the final size authority.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from bisect import bisect_left
from collections import Counter

KIND = "sample_scope_difference_v1"
IDENTITY_FIELDS = {
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
}
ENCODED_FIELDS = {
    "kind",
    "scope",
    "sample_ids_count",
    "sample_ids_sha256",
    "scoped_identities_count",
    "scoped_identities_sha256",
    "omitted_sample_ids",
    "additional_rows",
}
HASH = re.compile(r"[a-f0-9]{64}\Z")


def _measurement(value):
    """Actual canonical JSON bytes and hash, without a whole-output string."""
    encoder = json.JSONEncoder(
        sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    count, hashed = 0, hashlib.sha256()
    for fragment in encoder.iterencode(value):
        raw = fragment.encode("utf-8")
        count += len(raw)
        hashed.update(raw)
    return count, hashed.hexdigest()


def canonical_json_size(value):
    """Expose the exact compact writer boundary for CPU-only verification."""
    return _measurement(value)[0]


def _strings(values, label):
    if not isinstance(values, list):
        raise ValueError(f"explicit ordered string list required: {label}")
    previous = None
    for value in values:
        if not isinstance(value, str) or (previous is not None and value <= previous):
            raise ValueError(f"strict ordered duplicate-free strings required: {label}")
        previous = value


def _rows(values, width, label):
    if not isinstance(values, list):
        raise ValueError(f"explicit ordered row list required: {label}")
    previous = None
    for row in values:
        if (
            not isinstance(row, list)
            or len(row) != width
            or any(not isinstance(value, str) for value in row)
        ):
            raise ValueError(f"exact string tuple width required: {label}")
        current = tuple(row)
        if previous is not None and current <= previous:
            raise ValueError(f"strict ordered duplicate-free rows required: {label}")
        previous = current


def _parts(report):
    if not isinstance(report, dict):
        raise ValueError("explicit historical report dictionary required")
    identities = report.get("identities")
    if not isinstance(identities, dict) or set(identities) != IDENTITY_FIELDS:
        raise ValueError("complete six-field identity dictionary required")
    _strings(identities["sample_ids"], "sample_ids")
    return identities


def _contains(values, value):
    index = bisect_left(values, value)
    return index < len(values) and values[index] == value


def _new_report(report, identities, scoped):
    result, replacement = dict(report), dict(identities)
    replacement["scoped_identities"] = scoped
    result["identities"] = replacement
    return result


def pack_identities(report):
    """Factor the largest scope only when its actual serialized JSON shrinks.

    No identity is removed. An inherited encoded layout is validated before
    reuse. Other report fields retain their exact values and object references.
    """
    identities = _parts(report)
    rows = identities["scoped_identities"]
    if isinstance(rows, dict):
        unpack_identities(report)  # Never silently inherit a stale or malformed layout.
        return dict(report)
    _rows(rows, 4, "scoped_identities")
    if not rows:
        return dict(report)
    counts = Counter(tuple(row[:3]) for row in rows)
    scope = min(counts, key=lambda candidate: (-counts[candidate], candidate))
    samples = identities["sample_ids"]
    omitted, additional, sample_index = [], [], 0
    # Rows of the selected scope form one ordered ID subsequence. Merge it
    # against the already ordered sample table, without another large ID set.
    for row in rows:
        if tuple(row[:3]) != scope:
            additional.append(row)
            continue
        value = row[3]
        while sample_index < len(samples) and samples[sample_index] < value:
            omitted.append(samples[sample_index])
            sample_index += 1
        if sample_index < len(samples) and samples[sample_index] == value:
            sample_index += 1
        else:
            additional.append(row)  # A base-scope ID outside the global table.
    omitted.extend(samples[sample_index:])
    old_size, old_hash = _measurement(rows)
    encoded = dict(
        kind=KIND,
        scope=list(scope),
        sample_ids_count=len(samples),
        sample_ids_sha256=_measurement(samples)[1],
        scoped_identities_count=len(rows),
        scoped_identities_sha256=old_hash,
        omitted_sample_ids=omitted,
        additional_rows=additional,
    )
    if canonical_json_size(encoded) >= old_size:
        return dict(report)
    return _new_report(report, identities, encoded)


def unpack_identities(report):
    """Validate and reconstruct exactly the canonical legacy scoped-ID rows."""
    identities = _parts(report)
    encoded, samples = identities["scoped_identities"], identities["sample_ids"]
    if isinstance(encoded, list):
        _rows(encoded, 4, "scoped_identities")
        return dict(report)
    if (
        not isinstance(encoded, dict)
        or set(encoded) != ENCODED_FIELDS
        or encoded.get("kind") != KIND
    ):
        raise ValueError("strict sample-scope difference layout required")
    scope = encoded["scope"]
    if (
        not isinstance(scope, list)
        or len(scope) != 3
        or any(not isinstance(value, str) for value in scope)
    ):
        raise ValueError("exact three-string base scope required")
    for name in ("sample_ids_count", "scoped_identities_count"):
        if type(encoded[name]) is not int or encoded[name] < 0:
            raise ValueError("strict nonnegative identity counts required")
    for name in ("sample_ids_sha256", "scoped_identities_sha256"):
        if not isinstance(encoded[name], str) or HASH.fullmatch(encoded[name]) is None:
            raise ValueError("explicit canonical identity SHA256 required")
    if (
        encoded["sample_ids_count"] != len(samples)
        or encoded["sample_ids_sha256"] != _measurement(samples)[1]
    ):
        raise ValueError("encoded scope refers to a stale sample table")
    omitted, additional = encoded["omitted_sample_ids"], encoded["additional_rows"]
    _strings(omitted, "omitted_sample_ids")
    _rows(additional, 4, "additional_rows")
    if any(not _contains(samples, value) for value in omitted):
        raise ValueError("omitted IDs must belong to the unchanged sample table")
    for row in additional:
        if row[:3] == scope and _contains(samples, row[3]):
            raise ValueError("additional rows overlap or redundantly replace the base scope")
    count = len(samples) - len(omitted) + len(additional)
    if count != encoded["scoped_identities_count"]:
        raise ValueError("decoded scoped-ID count differs")

    def base_rows():
        index = 0
        for value in samples:
            if index < len(omitted) and omitted[index] == value:
                index += 1
            else:
                yield [*scope, value]

    rows = list(heapq.merge(base_rows(), additional))
    # The disjoint ordered inputs make the merge canonical and duplicate-free.
    if _measurement(rows)[1] != encoded["scoped_identities_sha256"]:
        raise ValueError("decoded scoped-ID semantic hash differs")
    return _new_report(report, identities, rows)
