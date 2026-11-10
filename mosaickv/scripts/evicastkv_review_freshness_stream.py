#!/usr/bin/env python3
"""Resumable, sharded historical identity inventory; never a freshness license.

Unlike the preceding canonical-name audit, this reads evidence while walking,
does not skip trace directories, and projects JSONL/Parquet identity columns.
Every invocation is finite and publishes a new immutable checkpoint. Resuming
an unchanged shard consumes the saved frontier rather than enumerating again.
No benchmark values, candidate selection, model loading or submission occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat as stat_module
import time
from collections import deque
from pathlib import Path

from evicastkv_boundary_operators import _STRING, _skip_json_value
from evicastkv_review_freshness_audit import ROOTS, canonical_name, reviewed_evidence

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
VERSION = 1
MAX_LINE = 2**20
MAX_JSON = 32 * 2**20
CHUNK = 2**20
MAX_DIRECTORY_ENTRIES = 500000
COCO = re.compile(r"COCO_(?:val|train)2014_\d{12}\.jpg")
HASH = re.compile(r"[0-9a-f]{64}")
STRUCTURED = {".json", ".jsonl", ".ndjson", ".parquet", ".yaml", ".yml"}
ID_KEYS = {
    "id",
    "sample_id",
    "sample_ids",
    "ids",
    "excluded_ids",
    "selected_ids",
    "consumed_ids",
    "COCO_consumed_ids",
    "MileBench_consumed_ids",
    "image_id",
    "image_ids",
    "question_id",
    "question_ids",
    "row_id",
    "source_group_id",
    "source_group_ids",
    "episode_id",
    "video_id",
    "document_id",
    "media_group_ids",
}
SCOPE_KEYS = {
    "dataset",
    "dataset_id",
    "dataset_name",
    "dataset_revision",
    "dataset_config",
    "configuration",
    "configuration_name",
    "config_name",
    "split",
    "task",
    "task_name",
    "revision",
    "subset",
    "subtask",
    "path",
    "name",
}
MEDIA_KEYS = {
    "media",
    "media_path",
    "media_paths",
    "image_path",
    "image_paths",
    "filename",
    "file_name",
    "original_path",
    "relative_path",
    "sha256",
    "media_sha256",
    "image_sha256",
    "content_sha256",
    "source_media_sha256",
    "media_group_ids",
}
FORBIDDEN = {
    "answer",
    "answers",
    "reference",
    "references",
    "caption",
    "captions",
    "question",
    "questions",
    "prompt",
    "prompts",
    "text",
    "content",
    "choices",
    "score",
    "scores",
    "task_score",
    "prediction",
    "predictions",
    "response",
    "responses",
    "output",
    "outputs",
    "completion",
    "completions",
    "logits",
    "attention",
    "attentions",
    "pixel_values",
    "image",
    "images",
    "video",
    "generated_tokens",
    "token_ids",
    "teacher_forcing_target",
    "nll",
    "loss",
    "error",
    "errors",
    "failure_identity",
    "ground_truth",
    "target",
    "label",
}


def safe_key(key):
    return (
        key in ID_KEYS | SCOPE_KEYS | MEDIA_KEYS
        or key.endswith(("_path", "_paths", "_sha256", "_revision"))
    ) and key not in FORBIDDEN


def identity_projection(raw):
    """Retain only whitelisted metadata leaves before constructing JSON objects.

    Unknown containers remain traversable, but unknown scalar values and every
    forbidden subtree are skipped lexically. Thus a novel prediction-text key
    cannot leak its value merely because it was absent from a blacklist.
    """
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    def spaces(cursor):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    def value(cursor, key=""):
        cursor = spaces(cursor)
        if cursor >= len(text):
            raise ValueError("truncated historical metadata JSON")
        if key in FORBIDDEN:
            return "null", _skip_json_value(text, cursor)
        if text[cursor] == "{":
            pieces = ["{"]
            cursor = spaces(cursor + 1)
            first = True
            keys = set()
            while cursor < len(text) and text[cursor] != "}":
                match = _STRING.match(text, cursor)
                if match is None:
                    raise ValueError("invalid historical metadata object key")
                child_key = json.loads(match.group())
                if child_key in keys:
                    raise ValueError("ambiguous duplicate historical metadata key")
                keys.add(child_key)
                colon = spaces(match.end())
                if colon >= len(text) or text[colon] != ":":
                    raise ValueError("historical metadata object lacks colon")
                projected, cursor = value(colon + 1, child_key)
                pieces.extend(("" if first else ",", match.group(), ":", projected))
                first = False
                cursor = spaces(cursor)
                if cursor < len(text) and text[cursor] == ",":
                    cursor = spaces(cursor + 1)
                    if cursor >= len(text) or text[cursor] == "}":
                        raise ValueError("trailing comma in historical metadata")
                elif cursor >= len(text) or text[cursor] != "}":
                    raise ValueError("invalid historical metadata object delimiter")
            if cursor >= len(text):
                raise ValueError("unterminated historical metadata object")
            pieces.append("}")
            return "".join(pieces), cursor + 1
        if text[cursor] == "[":
            pieces = ["["]
            cursor = spaces(cursor + 1)
            first = True
            while cursor < len(text) and text[cursor] != "]":
                projected, cursor = value(cursor, key)
                pieces.extend(("" if first else ",", projected))
                first = False
                cursor = spaces(cursor)
                if cursor < len(text) and text[cursor] == ",":
                    cursor = spaces(cursor + 1)
                    if cursor >= len(text) or text[cursor] == "]":
                        raise ValueError("trailing comma in historical metadata")
                elif cursor >= len(text) or text[cursor] != "]":
                    raise ValueError("invalid historical metadata list delimiter")
            if cursor >= len(text):
                raise ValueError("unterminated historical metadata list")
            pieces.append("]")
            return "".join(pieces), cursor + 1
        end = _skip_json_value(text, cursor)
        return text[cursor:end] if safe_key(key) else "null", end

    projected, end = value(0)
    if spaces(end) != len(text):
        raise ValueError("trailing historical JSON payload")
    return json.loads(projected)


def signature(path):
    stat = path.stat()
    # GPFS/Weka directory allocation size is not an entry-set signature and
    # may settle independently of metadata writes. Entries/mtime are guarded
    # separately; file size remains strict for actual source payloads.
    size = None if stat_module.S_ISDIR(stat.st_mode) else stat.st_size
    return [stat.st_dev, stat.st_ino, size, stat.st_mtime_ns]


def within(path, roots):
    return any(path.is_relative_to(root) for root in roots)


def owner(path, roots, count):
    for root in roots:
        if path.is_relative_to(root):
            relative = path.relative_to(root)
            # All descendants of a canonical top-level run have one owner.
            item = str(root / relative.parts[0]) if relative.parts else str(root)
            return int(hashlib.sha256(item.encode()).hexdigest(), 16) % count
    raise ValueError("canonical target outside allowed historical roots")


def scheduler_log(path):
    return "attempts" in path.parts and path.name in {"stdout.jsonl", "stderr.jsonl"}


class Budget:
    def __init__(self, *, seconds=1200, maximum_entries=500000, maximum_bytes=512 * 2**20):
        if not 0 < seconds <= 1500 or maximum_entries < 1 or maximum_bytes < MAX_LINE:
            raise ValueError("positive bounds: at most 1500 seconds and at least one line required")
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.maximum_entries, self.maximum_bytes = maximum_entries, maximum_bytes
        self.entries = self.bytes = 0

    def available(self):
        return (
            time.monotonic() < self.deadline
            and self.entries < self.maximum_entries
            and self.bytes < self.maximum_bytes
        )

    def check(self, *, entries=0, bytes_count=0):
        """Compatibility with the previously reviewed provenance-only loader."""
        self.entries += entries
        self.bytes += bytes_count
        if not self.available():
            raise TimeoutError("reviewed provenance exceeds finite streaming audit budget")


def collect(projected, state, path):
    """Conservative identities are exclusions, not eligibility or alias proofs."""
    cached = state.get("_identity_sets")
    samples, coco, hashes, groups, scopes, scoped = cached or (
        set(state["sample_ids"]),
        set(state["coco_ids"]),
        set(state["media_hashes"]),
        set(state["source_group_ids"]),
        {tuple(item) for item in state["scopes"]},
        {tuple(item) for item in state["scoped_identities"]},
    )

    def walk(item, context=(), key="", in_media=False):
        if isinstance(item, dict):
            local = dict(context)
            for field in (
                "dataset_id",
                "dataset_name",
                "dataset",
                "task",
                "split",
                "dataset_config",
            ):
                if isinstance(item.get(field), (str, int)):
                    local[field] = str(item[field])
            dataset = item.get("dataset")
            if isinstance(dataset, dict):
                for field in ("id", "name", "path"):
                    if isinstance(dataset.get(field), str):
                        local["dataset_id"] = dataset[field]
                        break
                for field in ("split", "configuration", "config_name"):
                    if isinstance(dataset.get(field), str):
                        local[field] = dataset[field]
            for child_key, child in item.items():
                walk(child, local, child_key, in_media or child_key == "media")
        elif isinstance(item, list):
            for child in item:
                walk(child, context, key, in_media)
        elif isinstance(item, (str, int)) and not isinstance(item, bool):
            item = str(item)
            coco.update(COCO.findall(item))
            if key in ID_KEYS:
                samples.add(item)
                scope = dict(context)
                namespace = scope.get(
                    "dataset_id",
                    scope.get("dataset_name", scope.get("dataset", scope.get("task", ""))),
                )
                configuration = scope.get("dataset_config", scope.get("configuration", ""))
                split = scope.get("split", "")
                scopes.add((namespace, configuration, split))
                scoped.add((namespace, configuration, split, item))
                if key in {
                    "source_group_id",
                    "source_group_ids",
                    "episode_id",
                    "video_id",
                    "document_id",
                }:
                    groups.add(item)
            if key in {
                "media_sha256",
                "image_sha256",
                "content_sha256",
                "source_media_sha256",
                "media_group_ids",
            } or (key == "sha256" and in_media):
                candidate = item.removeprefix("input-media:").removeprefix("original-image:")
                if HASH.fullmatch(candidate):
                    hashes.add(candidate)

    walk(projected)
    if cached is None:
        state.update(
            sample_ids=sorted(samples),
            coco_ids=sorted(coco),
            media_hashes=sorted(hashes),
            source_group_ids=sorted(groups),
            scopes=sorted(scopes),
            scoped_identities=sorted(scoped),
        )
    state["projected_records"] += 1


def error(state, path, message, **fields):
    state["errors"].append(dict(path=str(path), error=message, **fields))


def _stat_guard(path, expected):
    actual = signature(path)
    if actual != expected:
        raise ValueError(
            f"historical source changed since traversal/read checkpoint: {expected} -> {actual}"
        )


def _jsonl(path, item, state, budget):
    """Resume at a record boundary; raw segment hashes bind all consumed bytes."""
    _stat_guard(path, item["signature"])
    offset = item.get("offset", 0)
    start, hashed = offset, hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        while budget.available():
            raw = stream.readline(MAX_LINE)
            if not raw:
                item["done"] = True
                break
            budget.bytes += len(raw)
            hashed.update(raw)
            item["offset"] = stream.tell()
            continuation = item.get("skipping_line", False)
            too_long = len(raw) == MAX_LINE and not raw.endswith(b"\n")
            if continuation or too_long:
                if not continuation:
                    error(
                        state,
                        path,
                        "JSONL record exceeds identity-projection line bound",
                        offset=offset,
                    )
                item["skipping_line"] = too_long
            elif raw.strip():
                try:
                    collect(identity_projection(raw), state, path)
                except (ValueError, UnicodeError, RecursionError) as exc:
                    error(
                        state,
                        path,
                        f"malformed historical JSONL: {type(exc).__name__}",
                        offset=offset,
                    )
            offset = stream.tell()
    if offset > start:
        item.setdefault("segments", []).append(
            dict(start=start, end=offset, sha256=hashed.hexdigest())
        )
    _stat_guard(path, item["signature"])
    return item.get("done", False)


def _arrow_safe(field):
    import pyarrow as pa

    kind = field.type
    if pa.types.is_struct(kind):
        return all(_arrow_safe(child) for child in kind)
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        return _arrow_safe(pa.field(field.name, kind.value_type))
    return safe_key(field.name) and not (pa.types.is_binary(kind) or pa.types.is_large_binary(kind))


def _parquet(path, item, state, budget):
    import pyarrow.parquet as pq

    _stat_guard(path, item["signature"])
    parquet = pq.ParquetFile(path)
    columns = [field.name for field in parquet.schema_arrow if _arrow_safe(field)]
    unsafe = [
        field.name
        for field in parquet.schema_arrow
        if field.name in MEDIA_KEYS | ID_KEYS | SCOPE_KEYS and field.name not in columns
    ]
    if "columns" not in item:
        item["columns"] = columns
        if unsafe:
            error(
                state, path, "unprojectable nested/binary identity or media columns", columns=unsafe
            )
        if not columns:
            error(state, path, "Parquet has no safely projectable identity metadata columns")
    group = item.get("row_group", 0)
    batch_offset = item.get("batch", 0)
    while group < parquet.num_row_groups and budget.available():
        batches = parquet.iter_batches(batch_size=512, row_groups=[group], columns=columns)
        for number, batch in enumerate(batches):
            if number < batch_offset:
                continue
            if not budget.available():
                item.update(row_group=group, batch=number)
                return False
            for row in batch.to_pylist():
                collect(row, state, path)
            budget.bytes += batch.nbytes
            item.setdefault("projected_batches", []).append(digest(batch.to_pylist()))
            item.update(row_group=group, batch=number + 1)
        group += 1
        batch_offset = 0
        item.update(row_group=group, batch=0)
    if group < parquet.num_row_groups:
        return False
    # Bind the complete source bytes in finite raw-hash segments, never load
    # answer columns. A segmented content root is not mislabeled file SHA-256.
    offset = item.get("hash_offset", 0)
    start, hashed = offset, hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        while budget.available():
            raw = stream.read(CHUNK)
            if not raw:
                item["done"] = True
                break
            hashed.update(raw)
            offset += len(raw)
            budget.bytes += len(raw)
    item["hash_offset"] = offset
    if offset > start:
        item.setdefault("segments", []).append(
            dict(start=start, end=offset, sha256=hashed.hexdigest())
        )
    _stat_guard(path, item["signature"])
    return item.get("done", False)


def _process_file(item, state, budget):
    path = Path(item["path"])
    _stat_guard(path, item["signature"])
    if path.suffix in {".jsonl", ".ndjson"}:
        complete = _jsonl(path, item, state, budget)
    elif path.suffix == ".parquet":
        complete = _parquet(path, item, state, budget)
    elif path.suffix == ".json":
        if path.stat().st_size > MAX_JSON:
            error(state, path, "JSON exceeds bounded projection size; not silently skipped")
            return True
        raw = path.read_bytes()
        budget.bytes += len(raw)
        collect(identity_projection(raw), state, path)
        _stat_guard(path, item["signature"])
        item["sha256"] = hashlib.sha256(raw).hexdigest()
        complete = True
    else:
        error(state, path, "YAML requires separately reviewed lexical identity projection")
        return True
    if complete:
        state["bindings"][str(path)] = dict(
            signature=item["signature"],
            sha256=item.get("sha256"),
            segments=item.get("segments", []),
            segmented_content_root=digest(item.get("segments", [])),
            content_hash_kind="whole_file_sha256"
            if "sha256" in item
            else "sha256_of_segment_manifest",
            projected_batches=item.get("projected_batches", []),
        )
        state["processed_files"] += 1
    return complete


def _directory(item, state, pending, budget, roots, shard_index, shard_count):
    path = Path(item["path"])
    _stat_guard(path, item["signature"])
    position, found, prefix = item.get("position", 0), [], hashlib.sha256()
    last = 0
    # Weka/GPFS may return a different readdir order across unchanged opens.
    # Resume against sorted names, not an assumed stable filesystem order.
    names = []
    with os.scandir(path) as entries:
        for entry in entries:
            names.append(entry.name)
            if len(names) > MAX_DIRECTORY_ENTRIES:
                raise ValueError("directory exceeds bounded deterministic listing capacity")

    def order(name):
        return not canonical_name(name), Path(name).suffix not in STRUCTURED, name

    for index, name in enumerate(sorted(names, key=order)):
        encoded = (name + "\0").encode()
        prefix.update(encoded)
        if index < position:
            if index + 1 == position and prefix.hexdigest() != item["prefix_sha256"]:
                raise ValueError("directory enumeration prefix changed during resume")
            last = index + 1
            continue
        if not budget.available():
            # This not-yet-consumed name is excluded from the saved prefix.
            break
        budget.entries += 1
        found.append(name)
        last = index + 1
        item["prefix_sha256"] = prefix.hexdigest()
    else:
        item["done"] = True
    if not item.get("done"):
        item["position"] = last
        pending.appendleft(item)
    files, directories = [], []
    for name in sorted(found, key=lambda name: (not canonical_name(name), name)):
        alias = path / name
        try:
            canonical = alias.resolve(strict=True)
            if not within(canonical, roots):
                raise ValueError("symlink target outside allowed historical roots")
            if alias.is_symlink():
                state["aliases"].append(dict(alias=str(alias), canonical_target=str(canonical)))
            if owner(canonical, roots, shard_count) != shard_index:
                continue
            if canonical.is_dir():
                if str(canonical) not in state["_seen_directories"]:
                    state["_seen_directories"].add(str(canonical))
                    directories.append(
                        dict(kind="directory", path=str(canonical), signature=signature(canonical))
                    )
            elif canonical.is_file():
                if scheduler_log(alias) or scheduler_log(canonical):
                    if str(canonical) in state["_seen_files"]:
                        continue
                    state["_seen_files"].add(str(canonical))
                    state["scheduler_logs_excluded"].append(str(canonical))
                elif canonical.suffix in STRUCTURED:
                    if str(canonical) in state["_seen_files"]:
                        continue
                    state["_seen_files"].add(str(canonical))
                    files.append(
                        dict(kind="file", path=str(canonical), signature=signature(canonical))
                    )
                else:
                    state["binary_or_unstructured_files"] += 1
        except (ValueError, OSError, RuntimeError) as exc:
            error(state, alias, f"{type(exc).__name__}: {exc}")
    for directory in directories:
        pending.append(directory)
    for file in reversed(files):
        pending.appendleft(file)
    _stat_guard(path, item["signature"])
    state["directories"][str(path)] = item["signature"]


def initial_state(roots):
    return dict(
        frontier=[
            dict(kind="directory", path=str(path), signature=signature(path)) for path in roots
        ],
        seen_directories=[str(path) for path in roots],
        seen_files=[],
        directories={},
        bindings={},
        errors=[],
        aliases=[],
        scheduler_logs_excluded=[],
        sample_ids=[],
        coco_ids=[],
        media_hashes=[],
        source_group_ids=[],
        scopes=[],
        scoped_identities=[],
        projected_records=0,
        processed_files=0,
        binary_or_unstructured_files=0,
        verification_paths=None,
        verification_index=0,
    )


def advance(state, roots, *, shard_index, shard_count, budget):
    """Process metadata immediately; a pause always preserves its exact frontier."""
    pending = deque(state["frontier"])
    state["_seen_files"] = set(state["seen_files"])
    state["_seen_directories"] = set(state["seen_directories"])
    state["_identity_sets"] = (
        set(state["sample_ids"]),
        set(state["coco_ids"]),
        set(state["media_hashes"]),
        set(state["source_group_ids"]),
        {tuple(item) for item in state["scopes"]},
        {tuple(item) for item in state["scoped_identities"]},
    )
    while pending and budget.available():
        item = pending.popleft()
        try:
            if item["kind"] == "directory":
                _directory(item, state, pending, budget, roots, shard_index, shard_count)
            elif not _process_file(item, state, budget):
                pending.appendleft(item)
        except (
            ValueError,
            OSError,
            UnicodeError,
            RuntimeError,
            RecursionError,
            ImportError,
        ) as exc:
            error(state, item["path"], f"{type(exc).__name__}: {exc}")
    state["frontier"] = list(pending)
    state["seen_files"] = sorted(state.pop("_seen_files"))
    state["seen_directories"] = sorted(state.pop("_seen_directories"))
    identity_sets = state.pop("_identity_sets")
    for name, values in zip(
        (
            "sample_ids",
            "coco_ids",
            "media_hashes",
            "source_group_ids",
            "scopes",
            "scoped_identities",
        ),
        identity_sets,
        strict=True,
    ):
        state[name] = sorted(values)
    # Detect additions/removals and mutation even after a file was consumed in
    # an earlier checkpoint. A changing historical run produces a negative
    # audit, not a self-updating supposedly immutable inventory.
    if not pending:
        paths = state["verification_paths"]
        if paths is None:
            paths = sorted(state["directories"]) + sorted(state["bindings"])
            state["verification_paths"] = paths
        while state["verification_index"] < len(paths) and budget.available():
            name = paths[state["verification_index"]]
            expected = state["directories"].get(
                name, state["bindings"].get(name, {}).get("signature")
            )
            try:
                _stat_guard(Path(name), expected)
            except (ValueError, OSError) as exc:
                error(state, name, f"{type(exc).__name__}: {exc}")
            state["verification_index"] += 1
            budget.entries += 1
    state["traversal_complete"] = (
        not pending
        and state["verification_paths"] is not None
        and (state["verification_index"] == len(state["verification_paths"]))
    )
    return state


def run(
    root,
    *,
    roots=ROOTS,
    shard_index=0,
    shard_count=16,
    seconds=1200,
    maximum_entries=500000,
    maximum_bytes=512 * 2**20,
    resume=False,
    reviewed_previous=True,
):
    """Run one finite CPU chunk; same shard root may resume, never overwrite.

    Canonical execution requires clean source. Each fresh shard root is outside
    all scanned roots. ``aggregate`` publishes a separate final inventory only
    after every registered shard has a terminal checkpoint.
    """
    revision = clean_source(SOURCE)
    root = Path(root).resolve()
    allowed = tuple(Path(path).resolve(strict=True) for path in roots)
    if not root.is_relative_to(Path("/scratch/djy8hg").resolve()) or within(root, allowed):
        raise ValueError("write-once scratch output must be outside scanned historical roots")
    if len(set(allowed)) != len(allowed) or any(not path.is_dir() for path in allowed):
        raise ValueError("distinct existing historical roots required")
    if any(left != right and left.is_relative_to(right) for left in allowed for right in allowed):
        raise ValueError("overlapping historical roots would duplicate shard ownership")
    if not 0 <= shard_index < shard_count <= 256:
        raise ValueError("registered shard index/count required, at most 256")
    budget = Budget(seconds=seconds, maximum_entries=maximum_entries, maximum_bytes=maximum_bytes)
    if resume:
        provenance = json.loads((root / "reviewed_provenance.json").read_text())
    else:
        if root.exists():
            raise ValueError("new write-once shard output root required")
        provenance = (
            reviewed_evidence(budget)
            if reviewed_previous
            else dict(classifications=[], prior_excluded_ids=[], bindings={})
        )
    registration = dict(
        schema_version=VERSION,
        source_sha=revision,
        source_sha256=sha(Path(__file__)),
        roots=[str(path) for path in allowed],
        shard_index=shard_index,
        shard_count=shard_count,
        reviewed_previous=reviewed_previous,
        reviewed_provenance_digest=digest(provenance),
        projection="whitelisted_identity_leaves_v1",
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
    if resume:
        saved = json.loads((root / "registration.json").read_text())
        if saved != registration:
            raise ValueError("resume source/roots/shard/projection registration changed")
        checkpoints = sorted(root.glob("checkpoint-*.json"))
        if not checkpoints:
            raise ValueError("resume requires a complete prior checkpoint")
        previous = json.loads(checkpoints[-1].read_text())
        if previous["registration_sha256"] != sha(root / "registration.json"):
            raise ValueError("checkpoint registration content binding changed")
        state = previous["state"]
        if digest(state) != previous["state_sha256"]:
            raise ValueError("checkpoint content binding changed")
        if state.get("traversal_complete"):
            raise ValueError("terminal shard cannot be resumed or overwritten")
        number, parent_hash = len(checkpoints), sha(checkpoints[-1])
    else:
        write_once(root / "registration.json", registration)
        write_once(root / "reviewed_provenance.json", provenance)
        state = initial_state(allowed)
        state["sample_ids"] = provenance["prior_excluded_ids"]
        state["coco_ids"] = sorted(
            {match for sid in state["sample_ids"] for match in COCO.findall(sid)}
        )
        number, parent_hash = 0, None
    advance(state, allowed, shard_index=shard_index, shard_count=shard_count, budget=budget)
    checkpoint = dict(
        number=number,
        registration_sha256=sha(root / "registration.json"),
        parent_checkpoint_sha256=parent_hash,
        state=state,
        state_sha256=digest(state),
        wall_seconds=time.monotonic() - budget.started,
        bounded_bytes_read=budget.bytes,
        bounded_entries=budget.entries,
        seconds_limit=seconds,
        maximum_entries=maximum_entries,
        maximum_bytes=maximum_bytes,
    )
    path = root / f"checkpoint-{number:06d}.json"
    write_once(path, checkpoint)
    gate = dict(
        status="complete_identity_inventory_freshness_blocked"
        if state["traversal_complete"] and not state["errors"]
        else "complete_with_coverage_errors_freshness_blocked"
        if state["traversal_complete"]
        else "checkpointed_resume_required",
        source_sha=revision,
        shard_index=shard_index,
        shard_count=shard_count,
        checkpoint=str(path),
        checkpoint_sha256=sha(path),
        traversal_complete=state["traversal_complete"],
        processed_files=state["processed_files"],
        projected_records=state["projected_records"],
        pending_work_items=len(state["frontier"]),
        errors=len(state["errors"]),
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        complete_declared_identity_field_scope=state["traversal_complete"] and not state["errors"],
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        global_source_group_aliases_certified=False,
        benchmark_values_lexically_redacted=True,
        cohort_selected=False,
        models_loaded=False,
    )
    write_once(root / f"GATE_chunk-{number:06d}.json", gate)
    return gate


def aggregate(root, shard_roots):
    """Merge exact shard coverage; explicitly do not infer certified freshness."""
    root = Path(root)
    if root.exists():
        raise ValueError("new immutable aggregate directory required")
    registrations, states, bindings, errors = [], [], {}, []
    for shard_root in map(Path, shard_roots):
        registration = json.loads((shard_root / "registration.json").read_text())
        provenance = json.loads((shard_root / "reviewed_provenance.json").read_text())
        if digest(provenance) != registration["reviewed_provenance_digest"]:
            raise ValueError("reviewed provenance binding changed")
        checkpoints = sorted(shard_root.glob("checkpoint-*.json"))
        if not checkpoints:
            raise ValueError("all registered shards require a checkpoint")
        chain_parent = None
        for number, path in enumerate(checkpoints):
            checkpoint = json.loads(path.read_text())
            if checkpoint["registration_sha256"] != sha(shard_root / "registration.json"):
                raise ValueError("checkpoint registration content binding changed")
            if (
                checkpoint["number"] != number
                or checkpoint["parent_checkpoint_sha256"] != chain_parent
            ):
                raise ValueError("checkpoint chain incomplete or changed")
            if digest(checkpoint["state"]) != checkpoint["state_sha256"]:
                raise ValueError("checkpoint content binding changed")
            chain_parent = sha(path)
        registrations.append(registration)
        states.append(checkpoint["state"])
        bindings[str(checkpoints[-1])] = chain_parent
        errors.extend(checkpoint["state"]["errors"])
    if not registrations:
        raise ValueError("registered shards required")
    first = registrations[0]
    expected = first["shard_count"]
    if sorted(item["shard_index"] for item in registrations) != list(range(expected)):
        raise ValueError("every shard required exactly once")
    for registration in registrations:
        if {key: value for key, value in registration.items() if key != "shard_index"} != {
            key: value for key, value in first.items() if key != "shard_index"
        }:
            raise ValueError("shards have different scope/source/projection registrations")
    complete = all(state["traversal_complete"] for state in states)
    inventory = dict(
        source_sha=first["source_sha"],
        roots=first["roots"],
        shard_bindings=bindings,
        traversal_complete=complete,
        complete_declared_identity_field_scope=complete and not errors,
        complete_historical_id_inventory=False,
        sample_ids=sorted({item for state in states for item in state["sample_ids"]}),
        coco_ids=sorted({item for state in states for item in state["coco_ids"]}),
        media_hashes=sorted({item for state in states for item in state["media_hashes"]}),
        source_group_ids=sorted({item for state in states for item in state["source_group_ids"]}),
        scopes=sorted({tuple(item) for state in states for item in state["scopes"]}),
        scoped_identities=sorted(
            {tuple(item) for state in states for item in state["scoped_identities"]}
        ),
        processed_files=sum(state["processed_files"] for state in states),
        projected_records=sum(state["projected_records"] for state in states),
        errors=errors,
        recovered_classifications=json.loads(
            (Path(shard_roots[0]) / "reviewed_provenance.json").read_text()
        )["classifications"],
        cross_dataset_media_certified=False,
        global_source_group_aliases_certified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        cohort_selected=False,
        models_loaded=False,
        coverage_gaps=[
            "ID/metadata inventory does not certify decoded-pixel or underlying media equivalence.",
            "Cross-dataset aliases and global episode/video/document mappings remain unresolved.",
            "Redacted image/images fields can contain unindexed media; source-group certification "
            "is separate.",
            "Errors, unsupported YAML/nested Parquet, growing roots and bounded files block "
            "complete ID coverage.",
            "Undeclared identity aliases and ID-bearing binary packets require separate "
            "reviewed mappings.",
            "No new cohort has been selected, materialized or authorized by this inventory.",
        ],
    )
    write_once(root / "inventory.json", inventory)
    gate = dict(
        status="declared_identity_inventory_complete_confirmation_blocked"
        if inventory["complete_declared_identity_field_scope"]
        else "identity_inventory_incomplete_confirmation_blocked",
        inventory_sha256=sha(root / "inventory.json"),
        **{
            key: inventory[key]
            for key in (
                "complete_historical_id_inventory",
                "complete_declared_identity_field_scope",
                "traversal_complete",
                "processed_files",
                "projected_records",
                "cross_dataset_media_certified",
                "fresh_confirmation_authorized",
                "materialization_authorized",
            )
        },
    )
    write_once(root / "GATE_freshness_stream.json", gate)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-roots", type=Path, nargs="*")
    args = parser.parse_args()
    result = (
        aggregate(args.root, args.shard_roots or [])
        if args.mode == "aggregate"
        else run(
            args.root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            seconds=args.seconds,
            resume=args.resume,
        )
    )
    print(json.dumps(result))
