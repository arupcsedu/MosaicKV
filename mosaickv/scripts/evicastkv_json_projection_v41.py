#!/usr/bin/env python3
"""One-pass bounded JSON syntax/hash verification and identity-only projection.

Discarded value scalars are validated lexically, never passed to json.loads.
Unknown tensor arrays do not build lists of null placeholders. UTF-8 decoding
checks raw input chunks, not JSON scalar values. This is not a freshness gate;
an independent second hash pass must be supplied by the coordinator.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import re
import stat
import time
from pathlib import Path

import evicastkv_review_freshness_stream as legacy

from mosaickv.evaluation.fox_campaign import digest

MAXIMUM_BYTES = 2 * 2**30
CHUNK_BYTES = 256 * 2**10
MAXIMUM_CHUNK = 2**20
MAXIMUM_SCALAR = 2**20
MAXIMUM_METADATA = 8 * 2**20
MAXIMUM_METADATA_NODES = 100000
MAXIMUM_ACTIVE_KEYS = 100000
MAXIMUM_DEPTH = 64
IDENTITIES = (
    "sample_ids",
    "coco_ids",
    "media_hashes",
    "source_group_ids",
    "scopes",
    "scoped_identities",
)
WS = re.compile(rb"[ \t\r\n]*")
SPECIAL = re.compile(rb'["\\\x00-\x1f]')
PLAIN_STRING = re.compile(rb'"[^"\\\x00-\x1f]*"')
ATOM = re.compile(rb"[A-Za-z0-9+.\-]+")
SCALAR_PATTERN = rb"(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
SCALAR = re.compile(SCALAR_PATTERN)
NUMERIC_RUN = re.compile(rb"(?:" + SCALAR_PATTERN + rb"[ \t\r\n]*,[ \t\r\n]*)+")
HEX = re.compile(rb"[0-9a-fA-F]{4}\Z")
ESCAPES = {
    ord('"'): 34,
    ord("\\"): 92,
    ord("/"): 47,
    ord("b"): 8,
    ord("f"): 12,
    ord("n"): 10,
    ord("r"): 13,
    ord("t"): 9,
}
DROP = object()
SPACE_BYTES = frozenset((9, 10, 13, 32))
SCOPE_FIELDS = frozenset(
    ("dataset_id", "dataset_name", "dataset", "task", "split", "dataset_config")
)
DATASET_FIELDS = frozenset(("id", "name", "path", "split", "configuration", "config_name"))
MEDIA_FIELDS = frozenset(
    (
        "media_sha256",
        "image_sha256",
        "content_sha256",
        "source_media_sha256",
        "media_group_ids",
        "sha256",
    )
)
ESSENTIAL_FIELDS = legacy.ID_KEYS | SCOPE_FIELDS | MEDIA_FIELDS
MAXIMUM_KEY_CACHE_ENTRIES = 4096
MAXIMUM_KEY_CACHE_BYTES = 512 * 2**10


def _same_projection(left, right):
    """Exact typed structural equality, not a digest-only deduplication claim."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_projection(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_projection(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float):
        return left.hex() == right.hex()
    return left == right


def _decoded_scalar(raw):
    """Only object keys and retained metadata reach this scalar decoder."""
    result = json.loads(raw)
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("nonfinite retained metadata scalar")
    return result


def _key_hash(raw):
    """Normalize escaped keys without constructing discarded subtree values."""
    hashed, cursor, end = hashlib.sha256(), 1, len(raw) - 1
    while cursor < end:
        slash = raw.find(b"\\", cursor, end)
        limit = end if slash < 0 else slash
        if limit > cursor:
            hashed.update(raw[cursor:limit].decode("utf-8").encode("utf-32-be", "surrogatepass"))
        if slash < 0:
            break
        escape = raw[slash + 1]
        if escape == ord("u"):
            code = int(raw[slash + 2 : slash + 6], 16)
            cursor = slash + 6
            if (
                0xD800 <= code <= 0xDBFF
                and raw[cursor : cursor + 2] == b"\\u"
                and cursor + 6 <= end
            ):
                low = int(raw[cursor + 2 : cursor + 6], 16)
                if 0xDC00 <= low <= 0xDFFF:
                    code = 0x10000 + ((code - 0xD800) << 10) + low - 0xDC00
                    cursor += 6
        else:
            code, cursor = ESCAPES[escape], slash + 2
        hashed.update(code.to_bytes(4, "big"))
    return hashed.digest()


class Reader:
    def __init__(self, stream, *, deadline, maximum_bytes, chunk_bytes):
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (float, int))
            or not math.isfinite(deadline)
            or type(maximum_bytes) is not int
            or not 0 < maximum_bytes <= MAXIMUM_BYTES
            or type(chunk_bytes) is not int
            or not 1 <= chunk_bytes <= MAXIMUM_CHUNK
        ):
            raise ValueError("registered finite deadline, byte and chunk bounds required")
        self.stream, self.deadline, self.maximum, self.chunk = (
            stream,
            deadline,
            maximum_bytes,
            chunk_bytes,
        )
        self.buffer, self.pos, self.eof = b"", 0, False
        self.hash = hashlib.sha256()
        self.utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self.work, self.active_keys, self.metadata_bytes, self.metadata_nodes = 0, 0, 0, 0
        self.key_cache, self.key_cache_bytes = {}, 0
        self.stats = dict(
            bytes_read=0,
            read_calls=0,
            maximum_buffer_bytes=0,
            maximum_read_request=0,
            numeric_run_values=0,
            numeric_run_blocks=0,
            scalar_values_decoded=0,
            discarded_scalar_values=0,
            object_keys=0,
            maximum_active_keys=0,
            maximum_depth=0,
            deadline_checks=0,
            array_children_deduplicated=0,
            released_duplicate_metadata_bytes=0,
            released_duplicate_metadata_nodes=0,
            maximum_live_projected_metadata_bytes=0,
            maximum_live_projected_metadata_nodes=0,
            key_cache_hits=0,
            key_cache_entries=0,
            key_cache_bytes=0,
            in_buffer_strings=0,
            in_buffer_atoms=0,
            irrelevant_safe_scalars_pruned=0,
        )

    def check(self):
        self.stats["deadline_checks"] += 1
        if time.monotonic() >= self.deadline:
            raise TimeoutError("bounded streaming metadata deadline reached")

    def tick(self):
        self.work += 1
        if self.work % 1024 == 0:
            self.check()

    def fill(self, minimum=1):
        while len(self.buffer) - self.pos < minimum and not self.eof:
            self.check()
            rest = self.buffer[self.pos :]
            remaining = self.maximum - self.stats["bytes_read"]
            request = min(self.chunk, remaining + 1)
            raw = self.stream.read(request)
            self.stats["read_calls"] += 1
            self.stats["maximum_read_request"] = max(self.stats["maximum_read_request"], request)
            if not isinstance(raw, bytes) or len(raw) > request:
                raise ValueError("binary stream violated bounded read contract")
            if len(raw) > remaining:
                raise ValueError("stream exceeds registered byte bound")
            self.stats["bytes_read"] += len(raw)
            self.hash.update(raw)
            try:
                self.utf8.decode(raw, final=not raw)  # Raw UTF-8, never JSON-unescape values.
            except UnicodeError:
                raise ValueError("invalid UTF-8 in registered JSON stream") from None
            self.buffer, self.pos = rest + raw, 0
            self.stats["maximum_buffer_bytes"] = max(
                self.stats["maximum_buffer_bytes"], len(self.buffer)
            )
            if not raw:
                self.eof = True
        return len(self.buffer) - self.pos >= minimum

    def peek(self):
        if self.pos < len(self.buffer):
            return self.buffer[self.pos]
        return self.buffer[self.pos] if self.fill() else None

    def spaces(self):
        while self.pos < len(self.buffer) or self.fill():
            if self.buffer[self.pos] not in SPACE_BYTES:
                return
            end = WS.match(self.buffer, self.pos).end()
            self.pos = end
            if self.pos < len(self.buffer):
                return

    def expect(self, byte):
        if self.pos >= len(self.buffer) and not self.fill():
            raise ValueError("invalid JSON punctuation or truncated input")
        if self.buffer[self.pos] != byte:
            raise ValueError("invalid JSON punctuation or truncated input")
        self.pos += 1

    def string(self, capture):
        if self.pos < len(self.buffer):
            match = PLAIN_STRING.match(self.buffer, self.pos)
            if match is not None:
                start, end = self.pos, match.end()
                if end - start > MAXIMUM_SCALAR:
                    raise ValueError("registered scalar byte bound exceeded")
                self.pos = end
                self.stats["in_buffer_strings"] += 1
                return self.buffer[start:end] if capture else None
        self.expect(34)
        pieces, length = ([b'"'] if capture else None), 1
        while True:
            self.tick()
            if not self.fill():
                raise ValueError("unterminated JSON string")
            match = SPECIAL.search(self.buffer, self.pos)
            end = len(self.buffer) if match is None else match.start()
            if capture:
                pieces.append(self.buffer[self.pos : end])
            length += end - self.pos
            self.pos = end
            if length > MAXIMUM_SCALAR:
                raise ValueError("registered scalar byte bound exceeded")
            if match is None:
                continue
            byte = self.buffer[self.pos]
            if byte == 34:
                self.pos += 1
                length += 1
                if length > MAXIMUM_SCALAR:
                    raise ValueError("registered scalar byte bound exceeded")
                if capture:
                    pieces.append(b'"')
                    return b"".join(pieces)
                return None
            if byte != 92 or not self.fill(2):
                raise ValueError("invalid JSON string control or escape")
            escape = self.buffer[self.pos + 1]
            size = 6 if escape == ord("u") else 2
            if (escape not in ESCAPES and escape != ord("u")) or not self.fill(size):
                raise ValueError("invalid or truncated JSON escape")
            raw = self.buffer[self.pos : self.pos + size]
            if escape == ord("u") and HEX.fullmatch(raw[2:]) is None:
                raise ValueError("invalid JSON Unicode escape")
            if capture:
                pieces.append(raw)
            self.pos += size
            length += size

    def atom(self, capture, key, essential):
        match = ATOM.match(self.buffer, self.pos)
        if match is not None and match.end() < len(self.buffer):
            raw = self.buffer[self.pos : match.end()]
            self.pos = match.end()
            self.stats["in_buffer_atoms"] += 1
        else:
            pieces, length = [], 0
            while True:
                self.tick()
                if not self.fill():
                    break
                match = ATOM.match(self.buffer, self.pos)
                if match is None:
                    break
                raw = self.buffer[self.pos : match.end()]
                pieces.append(raw)
                length += len(raw)
                self.pos = match.end()
                if length > MAXIMUM_SCALAR:
                    raise ValueError("registered scalar byte bound exceeded")
                if self.pos < len(self.buffer):
                    break
            raw = b"".join(pieces)
        if len(raw) > MAXIMUM_SCALAR:
            raise ValueError("registered scalar byte bound exceeded")
        if not raw or SCALAR.fullmatch(raw) is None:
            raise ValueError("invalid JSON scalar")
        if capture:
            self.stats["scalar_values_decoded"] += 1
            value = _decoded_scalar(raw)
            if not essential:
                self.stats["irrelevant_safe_scalars_pruned"] += 1
                return DROP
            if value is None:
                return DROP
            self.charge(len(raw) + len(key))
            return value
        self.stats["discarded_scalar_values"] += 1
        return DROP

    def charge(self, amount):
        self.metadata_bytes += amount
        self.metadata_nodes += 1
        self.stats["maximum_live_projected_metadata_bytes"] = max(
            self.stats["maximum_live_projected_metadata_bytes"], self.metadata_bytes
        )
        self.stats["maximum_live_projected_metadata_nodes"] = max(
            self.stats["maximum_live_projected_metadata_nodes"], self.metadata_nodes
        )
        if self.metadata_bytes > MAXIMUM_METADATA or self.metadata_nodes > MAXIMUM_METADATA_NODES:
            raise ValueError("bounded projected metadata capacity exceeded")

    def key_info(self, raw, keep):
        cached = self.key_cache.get(raw)
        if cached is not None:
            self.stats["key_cache_hits"] += 1
            if not keep:
                return cached[0], "", (False, False)
            if cached[1] is not None:
                return cached
        hashed = cached[0] if cached is not None else _key_hash(raw)
        if keep:
            key = raw[1:-1].decode("utf-8") if b"\\" not in raw else _decoded_scalar(raw)
            classification = (legacy.safe_key(key), key in ESSENTIAL_FIELDS)
        else:
            key, classification = None, (False, False)
        result = (hashed, key, classification)
        cost = len(raw) * 2 + 64
        if (
            len(raw) <= 4096
            and len(self.key_cache) < MAXIMUM_KEY_CACHE_ENTRIES
            and self.key_cache_bytes + cost <= MAXIMUM_KEY_CACHE_BYTES
        ):
            if cached is None:
                self.key_cache_bytes += cost
            self.key_cache[raw] = result
            self.stats["key_cache_entries"] = len(self.key_cache)
            self.stats["key_cache_bytes"] = self.key_cache_bytes
        return (hashed, "", (False, False)) if not keep else result

    def value(self, *, key="", keep=True, depth=0, classification=None, dataset_object=False):
        self.tick()
        if depth > MAXIMUM_DEPTH:
            raise ValueError("registered JSON nesting bound exceeded")
        self.stats["maximum_depth"] = max(self.stats["maximum_depth"], depth)
        self.spaces()
        byte = self.peek()
        if byte is None:
            raise ValueError("truncated JSON value")
        keep = keep and key not in legacy.FORBIDDEN
        if classification is None:
            classification = (legacy.safe_key(key), key in ESSENTIAL_FIELDS)
        retain = keep and classification[0]
        essential = keep and classification[1]
        if byte == 123:
            return self.object(keep=keep, depth=depth, dataset_object=dataset_object)
        if byte == 91:
            return self.array(
                key=key, keep=keep, retain=retain, depth=depth, classification=classification
            )
        if byte == 34:
            raw = self.string(retain)
            if retain:
                if not essential and b"COCO_" not in raw and b"\\" not in raw:
                    self.stats["irrelevant_safe_scalars_pruned"] += 1
                    return DROP
                value = _decoded_scalar(raw)
                if not essential and legacy.COCO.search(value) is None:
                    self.stats["irrelevant_safe_scalars_pruned"] += 1
                    return DROP
                self.charge(len(raw) + len(key))
                self.stats["scalar_values_decoded"] += 1
                return value
            self.stats["discarded_scalar_values"] += 1
            return DROP
        return self.atom(retain, key, essential)

    def object(self, *, keep, depth, dataset_object=False):
        self.expect(123)
        result, keys = {}, set()
        self.spaces()
        if self.peek() == 125:
            self.pos += 1
            return DROP
        try:
            while True:
                raw_key = self.string(True)
                hashed, key, classification = self.key_info(raw_key, keep)
                if hashed in keys:
                    raise ValueError("ambiguous duplicate JSON object key")
                keys.add(hashed)
                self.active_keys += 1
                self.stats["object_keys"] += 1
                self.stats["maximum_active_keys"] = max(
                    self.stats["maximum_active_keys"], self.active_keys
                )
                if self.active_keys > MAXIMUM_ACTIVE_KEYS:
                    raise ValueError("bounded active object-key capacity exceeded")
                self.spaces()
                self.expect(58)
                if dataset_object and key in DATASET_FIELDS:
                    classification = (classification[0], True)
                child = self.value(
                    key=key,
                    keep=keep,
                    depth=depth + 1,
                    classification=classification,
                    dataset_object=key == "dataset",
                )
                if child is not DROP:
                    self.charge(len(raw_key))
                    result[key] = child
                self.spaces()
                byte = self.peek()
                if byte == 125:
                    self.pos += 1
                    break
                self.expect(44)
                self.spaces()
                if self.peek() == 125:
                    raise ValueError("trailing comma in JSON object")
            return result if result else DROP
        finally:
            self.active_keys -= len(keys)

    def array(self, *, key, keep, retain, depth, classification):
        self.expect(91)
        result, seen = [], {}
        self.spaces()
        if self.peek() == 93:
            self.pos += 1
            return DROP
        while True:
            if not retain and self.fill():
                # The run includes ONLY complete scalar+comma pairs. A token
                # split across a chunk is handled by atom(), never accepted early.
                match = NUMERIC_RUN.match(self.buffer, self.pos)
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
            before_bytes, before_nodes = self.metadata_bytes, self.metadata_nodes
            child = self.value(key=key, keep=keep, depth=depth + 1, classification=classification)
            if child is not DROP:
                fingerprint = digest(child)
                matches = seen.get(fingerprint, ())
                duplicate = False
                for index in matches:
                    self.tick()
                    if _same_projection(child, result[index]):
                        duplicate = True
                        break
                if duplicate:
                    # Only identical completed children of THIS array share an
                    # inherited context. collect() uses sets and counts the
                    # complete document once; no exclusion or scope is lost.
                    # Release the temporary child before returning its live
                    # capacity. Parsing/hash/work/deadline counters never reset.
                    child = DROP
                    self.stats["array_children_deduplicated"] += 1
                    self.stats["released_duplicate_metadata_bytes"] += (
                        self.metadata_bytes - before_bytes
                    )
                    self.stats["released_duplicate_metadata_nodes"] += (
                        self.metadata_nodes - before_nodes
                    )
                    self.metadata_bytes, self.metadata_nodes = before_bytes, before_nodes
                else:
                    self.charge(40)  # Retained slot plus its bounded digest/index entry.
                    seen.setdefault(fingerprint, []).append(len(result))
                    result.append(child)
            self.spaces()
            byte = self.peek()
            if byte == 93:
                self.pos += 1
                return result if result else DROP
            self.expect(44)
            self.spaces()
            if self.peek() == 93:
                raise ValueError("trailing comma in JSON array")


def identity_projection_stream(
    stream, *, deadline, maximum_bytes=MAXIMUM_BYTES, chunk_bytes=CHUNK_BYTES, statistics=None
):
    reader = Reader(stream, deadline=deadline, maximum_bytes=maximum_bytes, chunk_bytes=chunk_bytes)
    try:
        reader.check()
        projected = reader.value()
        reader.spaces()
        if reader.peek() is not None:
            raise ValueError("trailing JSON payload")
        projected = None if projected is DROP else projected
        identities = {name: [] for name in IDENTITIES} | {"projected_records": 0}
        legacy.collect(projected, identities, Path("/metadata-stream"))
        reader.check()
        reader.stats.update(
            projected_metadata_bytes=reader.metadata_bytes,
            projected_metadata_nodes=reader.metadata_nodes,
            work_units=reader.work,
        )
        return dict(
            identities=identities,
            whole_file_sha256=reader.hash.hexdigest(),
            bytes_read=reader.stats["bytes_read"],
            projected_metadata_sha256=digest(projected),
            statistics=dict(reader.stats),
            hash_passes=1,
            independent_second_pass_verified=False,
            forbidden_value_scalars_JSON_decoded=False,
            unknown_null_trees_materialized=False,
        )
    finally:
        reader.stats.update(
            projected_metadata_bytes=reader.metadata_bytes,
            projected_metadata_nodes=reader.metadata_nodes,
            work_units=reader.work,
        )
        if statistics is not None:
            statistics.update(reader.stats)


def _signature(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def project_file(path, expected_signature, deadline, maximum_bytes=MAXIMUM_BYTES):
    path = Path(path)
    if (
        not isinstance(expected_signature, (tuple, list))
        or len(expected_signature) != 4
        or any(type(value) is not int or value < 0 for value in expected_signature)
        or type(maximum_bytes) is not int
        or not 0 < maximum_bytes <= MAXIMUM_BYTES
    ):
        raise ValueError("strict registered signature and byte bound required")
    expected_signature = list(expected_signature)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or ".codex" in path.parts
        or path.name.lower() in {"auth.json", "credentials.json"}
    ):
        raise ValueError("absolute noncredential registered path required")
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or _signature(before) != expected_signature
        or before.st_size > maximum_bytes
    ):
        raise ValueError("registered regular-file signature or size differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        first = os.fstat(stream.fileno())
        if not stat.S_ISREG(first.st_mode) or _signature(first) != expected_signature:
            raise ValueError("opened file differs from registered source")
        result = identity_projection_stream(stream, deadline=deadline, maximum_bytes=maximum_bytes)
        after = os.fstat(stream.fileno())
        final = os.lstat(path)
        if (
            _signature(after) != expected_signature
            or _signature(final) != expected_signature
            or not stat.S_ISREG(final.st_mode)
            or result["bytes_read"] != first.st_size
        ):
            raise ValueError("source changed during complete streaming projection")
    return dict(
        result,
        path=str(path),
        signature_before=_signature(before),
        signature_after=_signature(final),
        fd_signature_before=_signature(first),
        fd_signature_after=_signature(after),
        parser_schema="evicastkv_strict_streaming_JSON_v41",
        original_errors_preserved=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
    )
