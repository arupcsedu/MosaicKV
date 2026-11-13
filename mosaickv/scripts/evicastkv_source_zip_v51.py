#!/usr/bin/env python3
"""Hash a bounded, registered source ZIP without extracting or executing code."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import struct
import zipfile
import zlib
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_BYTES = 8 * 2**20
MAX_UNCOMPRESSED_BYTES = 8 * 2**20
MAX_MEMBER_BYTES = 2**20
MAX_METADATA_BYTES = 8 * 2**20
MAX_NODES = 100000
CHUNK_BYTES = 64 * 2**10
HASH = re.compile(r"[a-f0-9]{64}\Z")
PROTECTED_PARTS = frozenset({".codex", ".ssh"})
PROTECTED_NAMES = frozenset({"auth.json", "credentials.json"})


def _protected(path):
    return any(part.casefold() in PROTECTED_PARTS for part in path.parts) or (
        path.name.casefold() in PROTECTED_NAMES
    )


def _canonical(path, allowed_root):
    path, allowed_root = Path(path), Path(allowed_root)
    if not path.is_absolute() or _protected(path):
        raise ValueError("absolute noncredential source ZIP path required")
    root = allowed_root.resolve(strict=True)
    target = path.resolve(strict=True)
    if (
        not root.is_dir()
        or not target.is_relative_to(root)
        or _protected(target)
        or target.suffix.casefold() != ".zip"
    ):
        raise ValueError("source ZIP escapes its allowed root or protected scope")
    return target


def _state(info):
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("source ZIP must remain a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _unchanged(path, stream, expected):
    if _state(os.fstat(stream.fileno())) != expected or _state(path.lstat()) != expected:
        raise ValueError("source ZIP changed during its bounded transaction")


class _CountedReader:
    """Charge every physical read, including ZipFile's repeated metadata reads."""

    def __init__(self, stream, size, budget):
        self.stream, self.size, self.budget = stream, size, budget

    def read(self, size=-1):
        self.budget.check()
        if size == -1:
            size = max(0, self.size - self.stream.tell())
        if type(size) is not int or not 0 <= size <= MAX_ARCHIVE_BYTES:
            raise ValueError("source ZIP read request exceeds the registered bound")
        self.budget.authorize(size)
        raw = self.stream.read(size)
        self.budget.add(len(raw))
        return raw

    def seek(self, offset, whence=0):
        self.budget.check()
        return self.stream.seek(offset, whence)

    def tell(self):
        return self.stream.tell()

    def seekable(self):
        return self.stream.seekable()


def _hash_pass(reader, size):
    reader.seek(0)
    result, count = hashlib.sha256(), 0
    while count < size:
        raw = reader.read(min(CHUNK_BYTES, size - count))
        if not raw:
            raise ValueError("truncated source ZIP during whole-file hash pass")
        count += len(raw)
        result.update(raw)
    return result.hexdigest()


def _directory_bound(reader, size):
    """Bound central-directory allocation before ZipFile constructs ZipInfo rows."""
    amount = min(size, 65535 + 22)
    reader.seek(size - amount)
    tail = reader.read(amount)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("complete bounded source ZIP directory required")
    _, disk, directory_disk, on_disk, entries, width, start, comment = struct.unpack_from(
        "<4s4H2LH", tail, offset
    )
    # ZIP64, multi-volume sources, comments and trailing payload are deliberately
    # outside this seven-file historical source-capsule format.
    if (
        disk
        or directory_disk
        or on_disk != entries
        or not 0 < entries < 65535
        or comment
        or offset + 22 != len(tail)
        or start + width != size - amount + offset
    ):
        raise ValueError("unsupported source ZIP directory geometry")
    if 1 + 4 * entries > MAX_NODES or width + 512 * entries > MAX_METADATA_BYTES:
        raise ValueError("source ZIP directory exceeds unchanged metadata/node limits")
    return entries


def _member_name(item):
    name = item.filename
    value = PurePosixPath(name)
    mode = item.external_attr >> 16
    if (
        item.orig_filename != name
        or not name
        or "\\" in name
        or value.is_absolute()
        or ".." in value.parts
        or str(value) != name
        or _protected(value)
        or value.suffix != ".py"
        or item.is_dir()
        or item.external_attr & 0x10
        or item.flag_bits & 1
        or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
    ):
        raise ValueError("unreviewed, aliased, encrypted, or protected source ZIP member")
    if not 0 <= item.file_size <= MAX_MEMBER_BYTES:
        raise ValueError("source ZIP member exceeds unchanged scalar bound")
    return name


def _members(reader, expected_entries, budget):
    with zipfile.ZipFile(reader, "r") as archive:
        items = archive.infolist()
        if len(items) != expected_entries:
            raise ValueError("source ZIP directory entry count differs")
        names, total, metadata = set(), 0, 0
        for item in items:
            budget.check()
            name = _member_name(item)
            if name in names:
                raise ValueError("duplicate source ZIP member")
            names.add(name)
            total += item.file_size
            metadata += len(name.encode("utf-8")) + len(item.extra) + len(item.comment) + 512
        if total > MAX_UNCOMPRESSED_BYTES or metadata > MAX_METADATA_BYTES:
            raise ValueError("source ZIP exceeds unchanged uncompressed/metadata bound")
        result = []
        for item in items:
            result_hash, count = hashlib.sha256(), 0
            # No source body is retained, decoded, executed or written to disk.
            # Physical compressed reads are charged by _CountedReader; expansion
            # is bounded separately by each validated central-directory size.
            with archive.open(item, "r") as stream:
                while True:
                    budget.check()
                    raw = stream.read(min(CHUNK_BYTES, item.file_size - count + 1))
                    if not raw:
                        break
                    count += len(raw)
                    if count > item.file_size or count > MAX_MEMBER_BYTES:
                        raise ValueError("source ZIP expanded beyond its registered member size")
                    result_hash.update(raw)
            if count != item.file_size:
                raise ValueError("source ZIP member ended before its registered size")
            result.append(dict(name=item.filename, bytes=count, sha256=result_hash.hexdigest()))
        return result


def inspect_zip(path, expected_sha256, budget, *, allowed_root):
    """Return byte witnesses only after opening and closing whole-file hashes agree."""
    if not isinstance(expected_sha256, str) or HASH.fullmatch(expected_sha256) is None:
        raise ValueError("original metadata must register the complete source ZIP SHA256")
    budget.check()
    path = _canonical(path, allowed_root)
    before = _state(path.lstat())
    if not 0 < before[2] <= MAX_ARCHIVE_BYTES:
        raise ValueError("source ZIP exceeds unchanged 8-MiB archive bound")
    # Authorization precedes every descriptor/body open, then _CountedReader
    # separately charges all real reads against the shared 210s/2GiB budget.
    budget.authorize(before[2])
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            _unchanged(path, stream, before)
            reader = _CountedReader(stream, before[2], budget)
            if _hash_pass(reader, before[2]) != expected_sha256:
                raise ValueError("source ZIP differs from the original registered hash")
            _unchanged(path, stream, before)
            entries = _directory_bound(reader, before[2])
            members = _members(reader, entries, budget)
            _unchanged(path, stream, before)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, zlib.error) as error:
        raise ValueError("source ZIP integrity or supported-code format failed") from error
    budget.authorize(before[2])
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb", buffering=0) as stream:
        _unchanged(path, stream, before)
        closing = _hash_pass(_CountedReader(stream, before[2], budget), before[2])
        _unchanged(path, stream, before)
    if closing != expected_sha256:
        raise ValueError("independent closing source ZIP hash differs")
    budget.check()
    return dict(path=str(path), sha256=closing, bytes=before[2], members=members)
