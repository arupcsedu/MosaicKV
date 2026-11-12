#!/usr/bin/env python3
"""Bounded identity projection with private, typed external array storage.

The V43 grammar, redaction and scope-preserving projection are unchanged. Only
completed array children may leave the live tree. They are written as tagged
JSON, never pickle, and replayed one child at a time after the parent object's
late dataset fields are known. Source JSON cannot create ``SpilledArray``.

All source and auxiliary reads share the original byte/deadline allowance.
Auxiliary bytes are reserved when written, so spooling cannot silently add an
unbounded second input. A consumer must exhaust every spill and call
``assert_spills_verified`` before admitting its collected identities.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import evicastkv_json_projection_v41 as base

SPILL_BYTES = 2**20
SPILL_NODES = 12500
REPLAY_CHUNK = 65536
DESCRIPTOR_BYTES = 256
REGISTRY_BYTES = 192
NAME = re.compile(r"array-[0-9]{8}\.jsonl\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FLAT_BYTES = 16384
MAXIMUM_LAYOUTS = 128
MAXIMUM_LAYOUT_BYTES = 256 * 2**10
MAXIMUM_LAYOUT_FIELDS = 64
MAXIMUM_LAYOUTS_PER_KEY = 4
_PLAIN = rb'"[^"\\\x00-\x1f]*"'
_WS = rb"[ \t\r\n]*"
_MEMBER = _PLAIN + _WS + rb":" + _WS + rb"(?:" + _PLAIN + rb"|" + base.SCALAR_PATTERN + rb")"
FLAT_RUN = re.compile(_MEMBER + rb"(?:" + _WS + rb"," + _WS + _MEMBER + rb")*" + _WS + rb"(?=[,}])")
FLAT_MEMBER = re.compile(
    rb"("
    + _PLAIN
    + rb")"
    + _WS
    + rb":"
    + _WS
    + rb"("
    + _PLAIN
    + rb"|"
    + base.SCALAR_PATTERN
    + rb")"
)
PRIVATE_ID_INTEGER = re.compile(rb'\["d",\[\["id",\["s",(-?(?:0|[1-9][0-9]*))\]\]\]\]\n')


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


@dataclass(frozen=True)
class SpilledArray:
    """An internal capability, not a dictionary accepted from historical JSON."""

    filename: str
    sha256: str
    count: int
    bytes_count: int
    signature: tuple[int, int, int, int]


@dataclass(frozen=True)
class _ScalarLayout:
    pattern: object
    field_count: int
    hashes: frozenset
    retained: tuple
    discarded: int


class InputBudget:
    """One immutable ceiling for source reads plus every auxiliary replay."""

    def __init__(self, maximum, deadline):
        if type(maximum) is not int or not 0 < maximum <= base.MAXIMUM_BYTES:
            raise ValueError("registered combined input byte bound required")
        self.maximum, self.max, self.deadline = maximum, maximum, deadline
        self.used, self.reserved = 0, 0

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("bounded streaming metadata deadline reached")
        if self.used + self.reserved > self.maximum:
            raise ValueError("combined source/auxiliary input byte bound exceeded")

    def consume(self, amount, *, auxiliary=False):
        if type(amount) is not int or amount < 0:
            raise ValueError("nonnegative bounded input amount required")
        self.check()
        credit = min(amount, self.reserved) if auxiliary else 0
        if self.used + self.reserved + amount - credit > self.maximum:
            raise ValueError("combined source/auxiliary input byte bound exceeded")
        self.reserved -= credit
        self.used += amount
        self.check()

    def reserve(self, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError("nonnegative auxiliary reservation required")
        self.check()
        if self.used + self.reserved + amount > self.maximum:
            raise ValueError("combined source/auxiliary input byte bound exceeded")
        self.reserved += amount


class _Source:
    def __init__(self, stream, owner):
        self.stream, self.owner = stream, owner

    def read(self, amount):
        self.owner.check_deadline()
        result = self.stream.read(amount)
        if isinstance(result, bytes):
            self.owner.budget.consume(len(result))
        self.owner.check_deadline()
        return result


class _Auxiliary:
    def __init__(self, stream, owner):
        self.stream, self.owner = stream, owner

    def read(self, amount):
        self.owner.check_deadline()
        result = self.stream.read(amount)
        self.owner.consume_aux_bytes(len(result))
        return result


def _parts(value):
    """A prefix-free typed JSON encoding; source lists cannot forge a spill."""
    if isinstance(value, SpilledArray):
        yield '["p",'
        yield json.dumps(
            [value.filename, value.sha256, value.count, value.bytes_count],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        yield "]"
    elif isinstance(value, dict):
        yield '["d",['
        for index, (key, child) in enumerate(value.items()):
            if index:
                yield ","
            yield "["
            yield json.dumps(key, ensure_ascii=False)
            yield ","
            yield from _parts(child)
            yield "]"
        yield "]]"
    elif isinstance(value, list):
        yield '["l",['
        for index, child in enumerate(value):
            if index:
                yield ","
            yield from _parts(child)
        yield "]]"
    elif value is None or type(value) in (str, int, float, bool):
        yield '["s",'
        yield json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        yield "]"
    else:
        raise ValueError("unsupported internal projection value")


def descriptor_digest(value):
    """Hash the deterministic typed representation without materializing it."""
    hashed = hashlib.sha256()
    for part in _parts(value):
        hashed.update(part.encode("utf-8", "backslashreplace"))
    return hashed.hexdigest()


class ProjectionReader(base.Reader):
    """V43 reader with bounded, scratch-only external array children."""

    def __init__(self, stream, *, deadline, maximum_bytes, chunk_bytes, workspace):
        self.budget = InputBudget(maximum_bytes, deadline)
        super().__init__(
            _Source(stream, self),
            deadline=deadline,
            maximum_bytes=maximum_bytes,
            chunk_bytes=chunk_bytes,
        )
        root = Path("/scratch/djy8hg").resolve(strict=True)
        parent = Path(workspace).resolve(strict=True)
        if not parent.is_relative_to(root) or not parent.is_dir():
            raise ValueError("private projection workspace must be under scratch")
        self.workspace = Path(tempfile.mkdtemp(prefix="projection-v44-", dir=parent))
        self.directory_fd = os.open(self.workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.registry, self.verified_spills = {}, set()
        self._persistent_bytes, self._persistent_nodes = 0, 0
        self._spool_counter, self._closed = 0, False
        self.scalar_layouts, self.scalar_layout_bytes, self.scalar_layout_count = {}, 0, 0
        self.stats.update(
            spill_files=0,
            spilled_array_children=0,
            auxiliary_bytes_written=0,
            auxiliary_bytes_read=0,
            completed_spill_replays=0,
            private_id_integer_fastpath_hits=0,
            flat_objects=0,
            flat_object_fields=0,
            scalar_member_runs=0,
            scalar_run_fields=0,
            scalar_layout_hits=0,
            scalar_layout_misses=0,
            scalar_layout_fields=0,
            scalar_layout_entries=0,
            scalar_layout_bytes=0,
        )

    def check_deadline(self):
        self.budget.check()

    def check(self):
        super().check()
        self.budget.check()

    def consume_aux_bytes(self, amount):
        self.budget.consume(amount, auxiliary=True)
        self.stats["auxiliary_bytes_read"] += amount

    def acquire(self, bytes_count, nodes_count=1):
        if any(type(item) is not int or item < 0 for item in (bytes_count, nodes_count)):
            raise ValueError("nonnegative live ownership required")
        self.metadata_bytes += bytes_count
        self.metadata_nodes += nodes_count
        self.stats["maximum_live_projected_metadata_bytes"] = max(
            self.stats["maximum_live_projected_metadata_bytes"], self.metadata_bytes
        )
        self.stats["maximum_live_projected_metadata_nodes"] = max(
            self.stats["maximum_live_projected_metadata_nodes"], self.metadata_nodes
        )
        if (
            self.metadata_bytes > base.MAXIMUM_METADATA
            or self.metadata_nodes > base.MAXIMUM_METADATA_NODES
        ):
            raise ValueError("bounded projected metadata capacity exceeded")
        self.check_deadline()

    def charge(self, amount):
        self.acquire(amount, 1)

    def release(self, bytes_count, nodes_count=1):
        if (
            any(type(item) is not int or item < 0 for item in (bytes_count, nodes_count))
            or bytes_count > self.metadata_bytes
            or nodes_count > self.metadata_nodes
        ):
            raise ValueError("invalid live ownership release")
        self.metadata_bytes -= bytes_count
        self.metadata_nodes -= nodes_count

    @contextlib.contextmanager
    def ownership(self, bytes_count, nodes_count=1):
        self.acquire(bytes_count, nodes_count)
        try:
            yield
        finally:
            self.release(bytes_count, nodes_count)

    def _mark(self):
        return (
            self.metadata_bytes,
            self.metadata_nodes,
            self._persistent_bytes,
            self._persistent_nodes,
        )

    def _owned_since(self, mark):
        return (
            self.metadata_bytes - mark[0] - self._persistent_bytes + mark[2],
            self.metadata_nodes - mark[1] - self._persistent_nodes + mark[3],
        )

    def _persistent(self, amount, nodes=1):
        self.acquire(amount, nodes)
        self._persistent_bytes += amount
        self._persistent_nodes += nodes

    def _new_spool(self):
        self.check_deadline()
        self._spool_counter += 1
        name = f"array-{self._spool_counter:08d}.jsonl"
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.directory_fd,
        )
        self.acquire(REPLAY_CHUNK, 0)
        return dict(
            name=name,
            stream=os.fdopen(fd, "wb", buffering=0),
            hash=hashlib.sha256(),
            count=0,
            buffer=bytearray(),
            buffer_owned=True,
        )

    def _flush(self, spool):
        buffer = spool["buffer"]
        view = memoryview(buffer)
        cursor = 0
        while cursor < len(view):
            self.check_deadline()
            amount = spool["stream"].write(view[cursor:])
            if not amount:
                raise ValueError("private spill write did not advance")
            cursor += amount
        spool["hash"].update(view)
        del view
        buffer.clear()

    def _write(self, spool, part):
        # Preserve UTF-8 scalars at their source-sized representation. Lone
        # surrogate code points remain JSON escapes rather than invalid UTF-8.
        raw = part.encode("utf-8", "backslashreplace")
        with self.ownership(2 * len(raw), 0):
            self.budget.reserve(len(raw))
            cursor = 0
            while cursor < len(raw):
                amount = min(len(raw) - cursor, REPLAY_CHUNK - len(spool["buffer"]))
                spool["buffer"].extend(raw[cursor : cursor + amount])
                cursor += amount
                if len(spool["buffer"]) == REPLAY_CHUNK:
                    self._flush(spool)
            self.stats["auxiliary_bytes_written"] += len(raw)

    def _write_child(self, spool, child):
        for part in _parts(child):
            self.check_deadline()
            self._write(spool, part)
        self._write(spool, "\n")
        spool["count"] += 1
        self.stats["spilled_array_children"] += 1

    def _finish_spool(self, spool):
        self._flush(spool)
        info = os.fstat(spool["stream"].fileno())
        spool["stream"].close()
        self.release(REPLAY_CHUNK, 0)
        spool["buffer_owned"] = False
        result = SpilledArray(
            spool["name"], spool["hash"].hexdigest(), spool["count"], info.st_size, _signature(info)
        )
        self._persistent(REGISTRY_BYTES)
        self.registry[result.filename] = result
        self.stats["spill_files"] += 1
        self.acquire(DESCRIPTOR_BYTES, 1)
        return result

    def array(self, *, key, keep, retain, depth, classification):
        self.expect(91)
        children, costs, spool = deque(), deque(), None
        self.spaces()
        if self.peek() == 93:
            self.pos += 1
            return base.DROP
        try:
            while True:
                if not retain and self.fill():
                    match = base.NUMERIC_RUN.match(self.buffer, self.pos)
                    if match is not None:
                        amount = self.buffer.count(b",", self.pos, match.end())
                        self.stats["numeric_run_values"] += amount
                        self.stats["discarded_scalar_values"] += amount
                        self.stats["numeric_run_blocks"] += 1
                        self.pos = match.end()
                        self.check()
                        self.spaces()
                        if self.peek() == 93:
                            raise ValueError("trailing comma in JSON array")
                mark = self._mark()
                child = self.value(
                    key=key, keep=keep, depth=depth + 1, classification=classification
                )
                if child is not base.DROP:
                    self.charge(40)
                    cost = self._owned_since(mark)
                    if spool is None:
                        children.append(child)
                        costs.append(cost)
                        child = base.DROP
                        if self.metadata_bytes >= SPILL_BYTES or self.metadata_nodes >= SPILL_NODES:
                            spool = self._new_spool()
                            while children:
                                previous, owned = children.popleft(), costs.popleft()
                                self._write_child(spool, previous)
                                previous = base.DROP
                                self.release(*owned)
                    else:
                        self._write_child(spool, child)
                        child = base.DROP
                        self.release(*cost)
                    child = base.DROP
                self.spaces()
                byte = self.peek()
                if byte == 93:
                    self.pos += 1
                    if spool is not None:
                        return self._finish_spool(spool)
                    return list(children) if children else base.DROP
                self.expect(44)
                self.spaces()
                if self.peek() == 93:
                    raise ValueError("trailing comma in JSON array")
        finally:
            if spool is not None and not spool["stream"].closed:
                spool["stream"].close()
            if spool is not None and spool["buffer_owned"]:
                spool["buffer"].clear()
                self.release(REPLAY_CHUNK, 0)
                spool["buffer_owned"] = False

    def _object_key(self, raw_key, keep, keys):
        hashed, key, classification = self.key_info(raw_key, keep)
        if hashed in keys:
            raise ValueError("ambiguous duplicate JSON object key")
        keys.add(hashed)
        self.active_keys += 1
        self.stats["object_keys"] += 1
        self.stats["maximum_active_keys"] = max(self.stats["maximum_active_keys"], self.active_keys)
        if self.active_keys > base.MAXIMUM_ACTIVE_KEYS:
            raise ValueError("bounded active object-key capacity exceeded")
        return key, classification

    def _flat_field(self, member, *, keep, depth, dataset_object, keys):
        self.tick()
        if depth + 1 > base.MAXIMUM_DEPTH:
            raise ValueError("registered JSON nesting bound exceeded")
        self.stats["maximum_depth"] = max(self.stats["maximum_depth"], depth + 1)
        raw_key, raw_value = member.groups()
        key, classification = self._object_key(raw_key, keep, keys)
        self.stats["flat_object_fields"] += 1
        self.stats["scalar_run_fields"] += 1
        field_keep = keep and key not in base.legacy.FORBIDDEN
        retain = field_keep and classification[0]
        essential = field_keep and (
            classification[1] or (dataset_object and key in base.DATASET_FIELDS)
        )
        if not retain:
            self.stats["discarded_scalar_values"] += 1
            return key, base.DROP
        return key, self._retained_field(raw_key, key, raw_value, essential)

    def _retained_field(self, raw_key, key, raw_value, essential):
        if raw_value.startswith(b'"'):
            if not essential and b"COCO_" not in raw_value:
                self.stats["irrelevant_safe_scalars_pruned"] += 1
                return base.DROP
            value = base._decoded_scalar(raw_value)
            if not essential and base.legacy.COCO.search(value) is None:
                self.stats["irrelevant_safe_scalars_pruned"] += 1
                return base.DROP
            self.stats["scalar_values_decoded"] += 1
        else:
            value = base._decoded_scalar(raw_value)
            self.stats["scalar_values_decoded"] += 1
            if not essential:
                self.stats["irrelevant_safe_scalars_pruned"] += 1
                return base.DROP
            if value is None:
                return base.DROP
        self.charge(len(raw_value) + len(key))
        self.charge(len(raw_key))
        return value

    def _layout_key(self, keep, dataset_object, end):
        first = base.PLAIN_STRING.match(self.buffer, self.pos, end)
        return None if first is None else (first.group(), keep, dataset_object)

    def _remember_layout(self, start, end, *, keep, dataset_object):
        """Learn only a run whose ordinary parser already validated every field."""
        if not MAXIMUM_LAYOUTS or self.scalar_layout_count >= MAXIMUM_LAYOUTS:
            return
        if (
            self.metadata_bytes + 2 * MAXIMUM_LAYOUT_BYTES > base.MAXIMUM_METADATA
            or self.metadata_nodes + MAXIMUM_LAYOUT_FIELDS * 5 + 3 > base.MAXIMUM_METADATA_NODES
        ):
            return  # Optional compilation cannot crowd out the source tree.
        members = list(FLAT_MEMBER.finditer(self.buffer, start, end))
        if not 2 <= len(members) <= MAXIMUM_LAYOUT_FIELDS:
            return
        dispatch = (members[0].group(1), keep, dataset_object)
        bucket = self.scalar_layouts.get(dispatch, ())
        if len(bucket) >= MAXIMUM_LAYOUTS_PER_KEY:
            return
        parts, retained, hashes = [], [], set()
        for member in members:
            raw_key = member.group(1)
            hashed, key, classification = self.key_info(raw_key, keep)
            hashes.add(hashed)
            retain = keep and key not in base.legacy.FORBIDDEN and classification[0]
            value = rb"(?:" + _PLAIN + rb"|" + base.SCALAR_PATTERN + rb")"
            if retain:
                value = b"(" + value + b")"
                essential = classification[1] or (dataset_object and key in base.DATASET_FIELDS)
                retained.append((raw_key, key, essential))
            parts.append(re.escape(raw_key) + _WS + rb":" + _WS + value)
        expression = (_WS + rb"," + _WS).join(parts) + _WS + rb"(?=[,}])"
        if len(expression) > FLAT_BYTES or self.scalar_layout_bytes >= MAXIMUM_LAYOUT_BYTES:
            return
        # This is the pinned CPython regex compiler, deliberately bypassing
        # re.compile's process-global cache. Optional layouts never widen the
        # input grammar or become a prerequisite for successful parsing.
        compiler = getattr(re, "_compiler", None)
        if (
            compiler is None
            or self.metadata_bytes + 2 * MAXIMUM_LAYOUT_BYTES > base.MAXIMUM_METADATA
            or self.metadata_nodes + MAXIMUM_LAYOUT_FIELDS * 5 + 3 > base.MAXIMUM_METADATA_NODES
        ):
            return
        with self.ownership(MAXIMUM_LAYOUT_BYTES, MAXIMUM_LAYOUT_FIELDS * 3):
            pattern = compiler.compile(expression, 0)
            amount = (
                sys.getsizeof(pattern)
                + len(expression)
                + sum(len(member.group(1)) * 2 + 192 for member in members)
                + 512
            )
            if self.scalar_layout_bytes + amount > MAXIMUM_LAYOUT_BYTES:
                return
            layout = _ScalarLayout(
                pattern,
                len(members),
                frozenset(hashes),
                tuple(retained),
                len(members) - len(retained),
            )
            if any(existing.pattern.pattern == expression for existing in bucket):
                return
            self._persistent(amount, 3 + len(members) * 2)
            self.scalar_layouts.setdefault(dispatch, []).append(layout)
            self.scalar_layouts[dispatch].sort(key=lambda entry: -entry.field_count)
            self.scalar_layout_bytes += amount
            self.scalar_layout_count += 1
            self.stats["scalar_layout_entries"] = self.scalar_layout_count
            self.stats["scalar_layout_bytes"] = self.scalar_layout_bytes

    def _cached_run(self, *, keep, dataset_object, depth, keys, result, end):
        if not self.scalar_layout_count:
            return False
        dispatch = self._layout_key(keep, dataset_object, end)
        for layout in self.scalar_layouts.get(dispatch, ()):
            self.check_deadline()
            match = layout.pattern.match(self.buffer, self.pos, end)
            if match is None:
                continue
            if not keys.isdisjoint(layout.hashes):
                raise ValueError("ambiguous duplicate JSON object key")
            if depth + 1 > base.MAXIMUM_DEPTH:
                raise ValueError("registered JSON nesting bound exceeded")
            keys.update(layout.hashes)
            self.active_keys += layout.field_count
            self.stats["maximum_active_keys"] = max(
                self.stats["maximum_active_keys"], self.active_keys
            )
            if self.active_keys > base.MAXIMUM_ACTIVE_KEYS:
                raise ValueError("bounded active object-key capacity exceeded")
            self.stats["maximum_depth"] = max(self.stats["maximum_depth"], depth + 1)
            previous = self.work
            self.work += layout.field_count
            if self.work // 1024 != previous // 1024:
                self.check()
            for counter in (
                "object_keys",
                "flat_object_fields",
                "scalar_run_fields",
                "scalar_layout_fields",
            ):
                self.stats[counter] += layout.field_count
            self.stats["discarded_scalar_values"] += layout.discarded
            self.stats["scalar_layout_hits"] += 1
            self.stats["scalar_member_runs"] += 1
            for group, (raw_key, key, essential) in enumerate(layout.retained, 1):
                child = self._retained_field(raw_key, key, match.group(group), essential)
                if child is not base.DROP:
                    result[key] = child
            self.pos = match.end()
            self.check_deadline()
            return True
        self.stats["scalar_layout_misses"] += 1
        return False

    def object(self, *, keep, depth, dataset_object=False):
        """Accelerate complete scalar runs around ordinary nested/escaped fields.

        A run must end at a verified comma or closing brace inside a 16-KiB
        window. One duplicate-key set spans all runs and fallback fields. The
        fallback is the exact V43 field grammar, not an unvalidated skip.
        """
        self.expect(123)
        result, keys, all_flat = {}, set(), True
        self.spaces()
        if self.peek() == 125:
            self.pos += 1
            self.stats["flat_objects"] += 1
            return base.DROP
        try:
            while True:
                end = min(len(self.buffer), self.pos + FLAT_BYTES)
                cached = self._cached_run(
                    keep=keep,
                    dataset_object=dataset_object,
                    depth=depth,
                    keys=keys,
                    result=result,
                    end=end,
                )
                run = None if cached else FLAT_RUN.match(self.buffer, self.pos, end)
                if run is not None:
                    self.check_deadline()
                    self.stats["scalar_member_runs"] += 1
                    start = self.pos
                    for member in FLAT_MEMBER.finditer(self.buffer, self.pos, run.end()):
                        key, child = self._flat_field(
                            member, keep=keep, depth=depth, dataset_object=dataset_object, keys=keys
                        )
                        if child is not base.DROP:
                            result[key] = child
                    self._remember_layout(
                        start, run.end(), keep=keep, dataset_object=dataset_object
                    )
                    self.pos = run.end()
                elif not cached:
                    all_flat = False
                    raw_key = self.string(True)
                    key, classification = self._object_key(raw_key, keep, keys)
                    self.spaces()
                    self.expect(58)
                    if dataset_object and key in base.DATASET_FIELDS:
                        classification = (classification[0], True)
                    child = self.value(
                        key=key,
                        keep=keep,
                        depth=depth + 1,
                        classification=classification,
                        dataset_object=key == "dataset",
                    )
                    if child is not base.DROP:
                        self.charge(len(raw_key))
                        result[key] = child
                self.spaces()
                if self.peek() == 125:
                    self.pos += 1
                    if all_flat:
                        self.stats["flat_objects"] += 1
                    self.check_deadline()
                    return result if result else base.DROP
                self.expect(44)
                self.spaces()
                if self.peek() == 125:
                    raise ValueError("trailing comma in JSON object")
        finally:
            self.active_keys -= len(keys)

    def _validate_descriptor(self, descriptor):
        if (
            not isinstance(descriptor, SpilledArray)
            or type(descriptor.filename) is not str
            or NAME.fullmatch(descriptor.filename) is None
            or type(descriptor.sha256) is not str
            or SHA256.fullmatch(descriptor.sha256) is None
            or type(descriptor.count) is not int
            or descriptor.count <= 0
            or type(descriptor.bytes_count) is not int
            or descriptor.bytes_count <= 0
            or self.registry.get(descriptor.filename) != descriptor
        ):
            raise ValueError("unregistered private spill descriptor")

    def iter_spilled(self, descriptor):
        self._validate_descriptor(descriptor)
        self.check_deadline()
        fd = os.open(descriptor.filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.directory_fd)
        with os.fdopen(fd, "rb", buffering=0) as stream:
            before = os.fstat(stream.fileno())
            path_before = os.stat(
                descriptor.filename, dir_fd=self.directory_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or _signature(before) != descriptor.signature
                or _signature(path_before) != descriptor.signature
                or before.st_nlink != 1
            ):
                raise ValueError("private spill signature mismatch")
            with self.ownership(REPLAY_CHUNK + 6, 0):
                replay = _ReplayReader(_Auxiliary(stream, self), self)
                count = 0
                while True:
                    replay.spaces()
                    if replay.peek() is None:
                        break
                    mark = self._mark()
                    child = replay.node()
                    owned = self._owned_since(mark)
                    count += 1
                    if count > descriptor.count:
                        raise ValueError("private spill count mismatch")
                    try:
                        yield child
                    finally:
                        child = base.DROP
                        self.release(*owned)
                after = os.fstat(stream.fileno())
                path_after = os.stat(
                    descriptor.filename, dir_fd=self.directory_fd, follow_symlinks=False
                )
                if (
                    count != descriptor.count
                    or replay.stats["bytes_read"] != descriptor.bytes_count
                    or replay.hash.hexdigest() != descriptor.sha256
                    or _signature(after) != descriptor.signature
                    or _signature(path_after) != descriptor.signature
                ):
                    raise ValueError("private spill hash/count/signature mismatch")
                self.check_deadline()
                if descriptor.filename not in self.verified_spills:
                    self._persistent(96)
                    self.verified_spills.add(descriptor.filename)
                self.stats["completed_spill_replays"] += 1

    def assert_spills_verified(self):
        self.check_deadline()
        if len(self.registry) != len(self.verified_spills) or any(
            name not in self.verified_spills for name in self.registry
        ):
            raise ValueError("private spill replay incomplete")

    def descriptor_digest(self, value):
        hashed = hashlib.sha256()
        for part in _parts(value):
            self.check_deadline()
            raw = part.encode("utf-8", "backslashreplace")
            with self.ownership(2 * len(raw), 0):
                hashed.update(raw)
        return hashed.hexdigest()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            with os.scandir(self.directory_fd) as entries:
                for entry in entries:
                    if NAME.fullmatch(entry.name) is None:
                        raise ValueError("unexpected private workspace entry; refusing cleanup")
                    os.unlink(entry.name, dir_fd=self.directory_fd)
        finally:
            os.close(self.directory_fd)
        self.workspace.rmdir()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _ReplayReader(base.Reader):
    """Decode the private tagged grammar directly, never build a tagged tree."""

    def __init__(self, stream, owner):
        self.owner = owner
        super().__init__(
            stream,
            deadline=owner.deadline,
            maximum_bytes=owner.budget.maximum,
            chunk_bytes=REPLAY_CHUNK,
        )

    def check(self):
        super().check()
        self.owner.check_deadline()

    def _punctuation(self, byte):
        self.spaces()
        self.expect(byte)
        self.spaces()

    def _string(self):
        self.spaces()
        raw = self.string(True)
        return base._decoded_scalar(raw), len(raw)

    def _scalar(self, key):
        self.spaces()
        if self.peek() == 34:
            value, size = self._string()
            self.owner.acquire(size + len(key), 1)
            return value
        result = self.atom(True, key, True)
        size = self.metadata_bytes
        self.metadata_bytes = 0
        self.metadata_nodes = 0
        if result is base.DROP:
            raise ValueError("null is not retained private metadata")
        self.owner.acquire(size, 1)
        return result

    def _descriptor(self):
        self._punctuation(91)
        filename, _ = self._string()
        self._punctuation(44)
        sha256, _ = self._string()
        values = []
        for _ in range(2):
            self._punctuation(44)
            value = self.atom(True, "", True)
            self.metadata_bytes = self.metadata_nodes = 0
            if type(value) is not int:
                raise ValueError("invalid private spill integer")
            values.append(value)
        self._punctuation(93)
        registered = self.owner.registry.get(filename)
        if registered is None:
            raise ValueError("unregistered private spill descriptor")
        descriptor = SpilledArray(filename, sha256, *values, registered.signature)
        self.owner._validate_descriptor(descriptor)
        self.owner.acquire(DESCRIPTOR_BYTES, 1)
        return descriptor

    def node(self, *, key="", depth=0):
        self.owner.check_deadline()
        if depth > base.MAXIMUM_DEPTH:
            raise ValueError("registered JSON nesting bound exceeded")
        # Only our canonical, complete private ID packet is accelerated. Hash,
        # source binding, count, read limits and replay ownership remain in the
        # enclosing iterator. Any different codec, escaped/string/float value,
        # split packet or malformed suffix uses the unchanged strict decoder.
        end = min(len(self.buffer), self.pos + base.MAXIMUM_SCALAR + 32)
        packet = PRIVATE_ID_INTEGER.match(self.buffer, self.pos, end)
        if packet is not None:
            raw = packet.group(1)
            if len(raw) > base.MAXIMUM_SCALAR:
                raise ValueError("registered scalar byte bound exceeded")
            if depth + 1 > base.MAXIMUM_DEPTH:
                raise ValueError("registered JSON nesting bound exceeded")
            if self.active_keys + 1 > base.MAXIMUM_ACTIVE_KEYS:
                raise ValueError("bounded active object-key capacity exceeded")
            value = int(raw)
            # Exact logical cost of the ordinary scalar(id) and dict-key path:
            # digits + len('id') at the scalar, then len('"id"') at the key.
            self.owner.acquire(len(raw) + 6, 2)
            self.pos = packet.end()
            self.owner.stats["private_id_integer_fastpath_hits"] += 1
            return {"id": value}
        self._punctuation(91)
        tag, _ = self._string()
        self._punctuation(44)
        if tag == "s":
            result = self._scalar(key)
        elif tag == "p":
            result = self._descriptor()
        elif tag in {"d", "l"}:
            self._punctuation(91)
            result = {} if tag == "d" else []
            keys = set()
            try:
                if self.peek() != 93:
                    while True:
                        if tag == "d":
                            self._punctuation(91)
                            child_key, key_size = self._string()
                            if child_key in keys:
                                raise ValueError("duplicate private projection key")
                            keys.add(child_key)
                            self.active_keys += 1
                            if self.active_keys > base.MAXIMUM_ACTIVE_KEYS:
                                raise ValueError("bounded active object-key capacity exceeded")
                            self._punctuation(44)
                            child = self.node(key=child_key, depth=depth + 1)
                            self._punctuation(93)
                            self.owner.acquire(key_size, 1)
                            result[child_key] = child
                        else:
                            child = self.node(key=key, depth=depth + 1)
                            self.owner.acquire(40, 1)
                            result.append(child)
                        self.spaces()
                        if self.peek() == 93:
                            break
                        self._punctuation(44)
                self._punctuation(93)
            finally:
                self.active_keys -= len(keys)
        else:
            raise ValueError("invalid private projection tag")
        self._punctuation(93)
        return result


def iter_spilled(descriptor, reader):
    """Compatibility helper for consumers preferring a module-level iterator."""
    yield from reader.iter_spilled(descriptor)
