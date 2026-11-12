#!/usr/bin/env python3
"""Disk-backed, bounded replay of the unchanged historical identity collector.

Only the already redacted projection enters this module. Replay occurs after
EOF so dataset fields following an array retain their legacy scope semantics.
The worker returns bounded shard references, never an unbounded identity set.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import sys
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import evicastkv_review_freshness_stream as legacy

KINDS = (
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
)
WIDTHS = {"scopes": 3, "scoped_identities": 4}
SCOPE_FIELDS = ("dataset_id", "dataset_name", "dataset", "task", "split", "dataset_config")
GROUP_KEYS = {"source_group_id", "source_group_ids", "episode_id", "video_id", "document_id"}
HASH_KEYS = {
    "media_sha256",
    "image_sha256",
    "content_sha256",
    "source_media_sha256",
    "media_group_ids",
}
MAX_SHARD_BYTES, MAX_SHARD_NODES = 8 * 2**20, 100000
TARGET_BYTES, MAX_REFERENCES = 256 * 2**10, 4096
SQLITE_RESERVATION = 512 * 2**10
HOT_CACHE_BYTES = 128 * 2**10


def _scratch_directory(path):
    path = Path(path)
    allowed = Path("/scratch/djy8hg").resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(allowed) or not resolved.is_dir():
        raise ValueError("identity workspace/output must be an existing scratch directory")
    return resolved


@contextmanager
def _owned(reader, amount, nodes=1):
    reader.acquire(amount, nodes)
    try:
        yield
    finally:
        reader.release(amount, nodes)


@contextmanager
def _encoded(value, reader):
    # At most four bounded source scalars form a scoped identity. Charge both
    # Python representations plus SQLite's parameter copy while they are live.
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # JSON permits escaped unpaired surrogates. Keep those as ASCII escapes,
    # rather than rejecting an identity accepted by the unchanged collector.
    payload = text.encode("utf-8", "backslashreplace")
    amount = sys.getsizeof(text) + sys.getsizeof(payload) + 2 * len(payload) + 128
    with _owned(reader, amount):
        yield payload


def _nodes(value):
    return 1 + len(value) if isinstance(value, (tuple, list)) else 1


class _ExactHotCache:
    """Small exact cache; bounded generation resets never change membership."""

    def __init__(self, reader, maximum=HOT_CACHE_BYTES):
        if type(maximum) is not int or not 0 <= maximum <= HOT_CACHE_BYTES:
            raise ValueError("identity hot cache must remain within 128 KiB")
        self.reader, self.maximum = reader, maximum
        self.entries = OrderedDict() if maximum >= sys.getsizeof(OrderedDict()) else None
        self.bytes = self.nodes = self.peak_bytes = self.peak_entries = 0
        if self.entries is not None:
            self.bytes, self.nodes = sys.getsizeof(self.entries), 1
            reader.acquire(self.bytes, self.nodes)
            self.peak_bytes = self.bytes

    def contains(self, kind, value):
        if self.entries is None:
            return False
        key = (kind, value)
        if key not in self.entries:
            return False
        self.entries.move_to_end(key)
        return True

    def add(self, kind, value):
        if self.entries is None:
            return
        key = (kind, value)
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        # Charge retained scalar objects even when their source tree still owns
        # another reference: cache ownership must outlive a released spill child.
        fields = value if isinstance(value, tuple) else (value,)
        amount = (
            sys.getsizeof(key)
            + (sys.getsizeof(value) if isinstance(value, tuple) else 0)
            + sum(sys.getsizeof(field) for field in fields)
            + 512
        )
        nodes = 4 + len(fields)
        base = sys.getsizeof(OrderedDict())
        if amount + base > self.maximum:
            return  # A large identity remains exclusively in the bounded database.
        if self.bytes + amount > self.maximum:
            # Itemwise OrderedDict eviction leaves its large native hash table
            # allocated. Reset the generation so a subsequent large identity
            # cannot coexist with an uncharged table-capacity high-water mark.
            self.entries.clear()
            current = sys.getsizeof(self.entries)
            self.reader.release(self.bytes - current, self.nodes - 1)
            self.bytes, self.nodes = current, 1
        self.reader.acquire(amount, nodes)
        self.entries[key] = (amount, nodes)
        self.bytes += amount
        self.nodes += nodes
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.peak_entries = max(self.peak_entries, len(self.entries))

    def close(self):
        if self.entries is not None:
            self.entries.clear()
            self.entries = None
            self.reader.release(self.bytes, self.nodes)
            self.bytes = self.nodes = 0


def _walk(node, reader, emit, context=None, key="", in_media=False):
    reader.check_deadline()
    context = {} if context is None else context
    if isinstance(node, dict):
        local = None
        created = []
        for field in SCOPE_FIELDS:
            value = node.get(field)
            # Deliberately preserve bool-as-int for scope fields, but not IDs.
            if isinstance(value, (str, int)):
                if local is None:
                    local = dict(context)
                converted = str(value)
                local[field] = converted
                if not isinstance(value, str):
                    created.append(converted)
        dataset = node.get("dataset")
        if isinstance(dataset, dict):
            if local is None:
                local = dict(context)
            for field in ("id", "name", "path"):
                if isinstance(dataset.get(field), str):
                    local["dataset_id"] = dataset[field]
                    break
            for field in ("split", "configuration", "config_name"):
                if isinstance(dataset.get(field), str):
                    local[field] = dataset[field]
        if local is None:
            # Ordinary profiler objects do not introduce a scope. Reuse the
            # immutable ancestor mapping rather than copying/charging it again.
            for child_key, child in node.items():
                _walk(child, reader, emit, context, child_key, in_media or child_key == "media")
            return
        amount = sys.getsizeof(local) + sum(sys.getsizeof(value) for value in created)
        with _owned(reader, amount, 1 + len(created)):
            for child_key, child in node.items():
                _walk(child, reader, emit, local, child_key, in_media or child_key == "media")
        return
    if isinstance(node, list):
        for child in node:
            _walk(child, reader, emit, context, key, in_media)
        return
    # Import lazily: the parser can use this collector without a module cycle.
    from evicastkv_spill_projection_v44 import SpilledArray

    if isinstance(node, SpilledArray):
        children = reader.iter_spilled(node)
        try:
            for child in children:
                _walk(child, reader, emit, context, key, in_media)
        finally:
            children.close()
        return
    if not isinstance(node, (str, int)) or isinstance(node, bool):
        return
    if isinstance(node, str):
        _leaf(node, context, key, in_media, emit)
    else:
        value = str(node)
        with _owned(reader, sys.getsizeof(value)):
            _leaf(value, context, key, in_media, emit)


def _leaf(value, context, key, in_media, emit):
    for match in legacy.COCO.finditer(value):
        emit("coco_ids", match.group())
    if key in legacy.ID_KEYS:
        emit("sample_ids", value)
        namespace = context.get(
            "dataset_id",
            context.get("dataset_name", context.get("dataset", context.get("task", ""))),
        )
        configuration = context.get("dataset_config", context.get("configuration", ""))
        scope = (namespace, configuration, context.get("split", ""))
        emit("scopes", scope)
        emit("scoped_identities", (*scope, value))
        if key in GROUP_KEYS:
            emit("source_group_ids", value)
    if key in HASH_KEYS or (key == "sha256" and in_media):
        candidate = value.removeprefix("input-media:").removeprefix("original-image:")
        if legacy.HASH.fullmatch(candidate):
            emit("media_hashes", candidate)


class _ShardWriter:
    def __init__(self, output, reader, target):
        self.output, self.reader, self.target = output, reader, target
        self.refs, self.counts = [], dict.fromkeys(KINDS, 0)
        self.stream = None
        self.kind = None
        self.count = self.size = self.nodes = 0
        self.hash = None
        self.path = None

    def _write(self, raw):
        self.reader.check_deadline()
        self.stream.write(raw)
        self.hash.update(raw)
        self.size += len(raw)

    def _open(self, kind):
        if len(self.refs) >= MAX_REFERENCES:
            raise ValueError("bounded identity shard reference capacity exceeded")
        self.kind = kind
        self.path = self.output / f"identities-{len(self.refs):05d}.json"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(fd, "wb")
        self.hash = hashlib.sha256()
        self.count = self.size = 0
        self.nodes = 8
        self._write(b'{"version":1,"kind":' + json.dumps(kind).encode() + b',"values":[')

    def finish(self):
        if self.stream is None:
            return
        self._write(b"]}")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.stream = None
        reference = dict(
            path=str(self.path),
            sha256=self.hash.hexdigest(),
            bytes=self.size,
            kind=self.kind,
            count=self.count,
        )
        # These references remain live in the returned receipt. The caller
        # retains their reader ownership until it publishes/discards the receipt.
        self.reader.acquire(len(json.dumps(reference).encode()) + 256, 9)
        self.refs.append(reference)

    def add(self, kind, raw, nodes):
        self.reader.check_deadline()
        if len(raw) + 128 > MAX_SHARD_BYTES or nodes + 8 > MAX_SHARD_NODES:
            raise ValueError("individual identity exceeds bounded shard capacity")
        projected = self.size + len(raw) + (1 if self.count else 0) + 2
        if self.stream is not None and (
            kind != self.kind
            or (self.count and projected > self.target)
            or projected > MAX_SHARD_BYTES
            or self.nodes + nodes > MAX_SHARD_NODES
        ):
            self.finish()
        if self.stream is None:
            self._open(kind)
        if self.count:
            self._write(b",")
        self._write(raw)
        self.count += 1
        self.nodes += nodes
        self.counts[kind] += 1


def collect_to_shards(
    node,
    reader,
    workspace,
    output_root,
    *,
    target_bytes=TARGET_BYTES,
    hot_cache_bytes=HOT_CACHE_BYTES,
):
    """Replay the projected root exactly once and emit deduplicated typed shards.

    ``reader`` must share its 8-MiB/100k live-object accounting and 210-second,
    source-plus-auxiliary 2-GiB budget with projection and spool replay.
    """
    if type(target_bytes) is not int or not 128 <= target_bytes <= MAX_SHARD_BYTES:
        raise ValueError("registered finite shard target required")
    workspace, output = _scratch_directory(workspace), _scratch_directory(output_root)
    reader.check_deadline()
    private = Path(tempfile.mkdtemp(prefix="identity-sink-", dir=workspace))
    database = private / "identities.sqlite3"
    writer = _ShardWriter(output, reader, target_bytes)
    statistics = dict(
        insert_attempts=0,
        duplicate_identities=0,
        sqlite_export_bytes=0,
        sqlite_insert_calls=0,
        hot_cache_hits=0,
    )
    connection = cache = None
    try:
        with _owned(reader, SQLITE_RESERVATION, 16):
            connection = sqlite3.connect(database, cached_statements=8)
            for command in (
                "PRAGMA journal_mode=OFF",
                "PRAGMA synchronous=OFF",
                "PRAGMA temp_store=FILE",
                "PRAGMA cache_size=-256",
                "PRAGMA mmap_size=0",
            ):
                connection.execute(command)
            connection.set_progress_handler(lambda: _progress(reader), 256)
            connection.execute(
                "CREATE TABLE ids(kind TEXT NOT NULL,value BLOB NOT NULL,nodes INTEGER NOT NULL,"
                "PRIMARY KEY(kind,value)) WITHOUT ROWID"
            )
            cache = _ExactHotCache(reader, hot_cache_bytes)

            def emit(kind, value):
                reader.check_deadline()
                statistics["insert_attempts"] += 1
                if cache.contains(kind, value):
                    statistics["hot_cache_hits"] += 1
                    statistics["duplicate_identities"] += 1
                    return
                with _encoded(value, reader) as raw:
                    before = connection.total_changes
                    statistics["sqlite_insert_calls"] += 1
                    connection.execute(
                        "INSERT OR IGNORE INTO ids(kind,value,nodes) VALUES(?,?,?)",
                        (kind, raw, _nodes(value)),
                    )
                    statistics["duplicate_identities"] += int(connection.total_changes == before)
                cache.add(kind, value)

            _walk(node, reader, emit)
            statistics.update(
                maximum_hot_cache_bytes=cache.peak_bytes,
                maximum_hot_cache_entries=cache.peak_entries,
            )
            cache.close()
            reader.assert_spills_verified()
            connection.commit()
            cursor = connection.execute("SELECT kind,value,nodes FROM ids ORDER BY kind,value")
            while True:
                reader.check_deadline()
                row = cursor.fetchone()
                if row is None:
                    break
                kind, raw, nodes = row
                with _owned(reader, sys.getsizeof(raw) + 128, 3):
                    reader.consume_aux_bytes(len(raw))
                    statistics["sqlite_export_bytes"] += len(raw)
                    writer.add(kind, raw, nodes)
            writer.finish()
    except sqlite3.OperationalError as error:
        # SQLite converts exceptions in progress callbacks into 'interrupted'.
        # Re-raise the registered deadline error when that caused interruption.
        reader.check_deadline()
        raise ValueError("bounded identity sink SQL operation failed") from error
    finally:
        statistics.update(
            maximum_hot_cache_bytes=cache.peak_bytes if cache is not None else 0,
            maximum_hot_cache_entries=cache.peak_entries if cache is not None else 0,
        )
        # Publish once, including when parsing/replay hits its unchanged deadline.
        # The wrapper reads these counters when it constructs a failure trace.
        reader.stats.update(
            identity_sink_sqlite_insert_calls=statistics["sqlite_insert_calls"],
            identity_sink_insert_attempts=statistics["insert_attempts"],
            identity_sink_hot_cache_hits=statistics["hot_cache_hits"],
            identity_sink_maximum_hot_cache_bytes=statistics["maximum_hot_cache_bytes"],
            identity_sink_maximum_hot_cache_entries=statistics["maximum_hot_cache_entries"],
        )
        if cache is not None:
            cache.close()
        if connection is not None:
            connection.close()
        if writer.stream is not None:
            writer.stream.close()
        # Only our exclusively created temporary database, never input files.
        database.unlink(missing_ok=True)
        private.rmdir()
    return dict(
        identity_shards=writer.refs,
        identity_counts=writer.counts,
        projected_records=1,
        stats=statistics,
    )


def _progress(reader):
    reader.check_deadline()
    return 0


def read_shard(reference):
    """Validate one bounded shard for the separate, higher-memory reducer."""
    if set(reference) != {"path", "sha256", "bytes", "kind", "count"}:
        raise ValueError("exact identity shard reference required")
    path = Path(reference["path"])
    if (
        reference["kind"] not in KINDS
        or type(reference["bytes"]) is not int
        or not 0 < reference["bytes"] <= MAX_SHARD_BYTES
        or type(reference["count"]) is not int
        or reference["count"] <= 0
        or path.resolve(strict=True) != path.absolute()
        or not stat.S_ISREG(path.lstat().st_mode)
    ):
        raise ValueError("invalid bounded identity shard reference")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read(reference["bytes"] + 1)
    if len(raw) != reference["bytes"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("identity shard byte/hash mismatch")

    def object_fields(pairs):
        values = {}
        for key, item in pairs:
            if key in values:
                raise ValueError("duplicate identity shard metadata field")
            values[key] = item
        return values

    value = json.loads(raw, object_pairs_hook=object_fields)
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "kind", "values"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["kind"] != reference["kind"]
        or not isinstance(value["values"], list)
        or len(value["values"]) != reference["count"]
    ):
        raise ValueError("identity shard typed metadata differs")
    width = WIDTHS.get(value["kind"])
    for identity in value["values"]:
        if width is None:
            valid = isinstance(identity, str)
        else:
            valid = (
                isinstance(identity, list)
                and len(identity) == width
                and all(isinstance(field, str) for field in identity)
            )
        if not valid:
            raise ValueError("identity shard has invalid typed identity")
    if 8 + sum(_nodes(identity) for identity in value["values"]) > MAX_SHARD_NODES:
        raise ValueError("identity shard exceeds node capacity")
    return value
