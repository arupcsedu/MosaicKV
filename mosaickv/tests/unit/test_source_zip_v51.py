"""Synthetic source capsules only: no historical code or credential reads."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import zipfile

import evicastkv_recover_bindings_v49 as recovery
import evicastkv_source_zip_v51 as source_zip
import pytest


def zipped(entries=None):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in entries or [("00/writer.py", b"print('a')\n")]:
            archive.writestr(zipfile.ZipInfo(name), content)
    return raw.getvalue()


def capsule(tmp_path, entries=None, raw=None):
    path = tmp_path / "analysis_source.zip"
    raw = zipped(entries) if raw is None else raw
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def inspect(path, expected, tmp_path, budget=None):
    return source_zip.inspect_zip(
        path, expected, budget or recovery.Budget(), allowed_root=tmp_path
    )


def test_exact_code_witnesses_no_code_bodies_and_two_complete_hash_reads(tmp_path, monkeypatch):
    path, expected = capsule(tmp_path, [("00/writer.py", b"writer"), ("01/helper.py", b"helper")])
    budget, opens = recovery.Budget(), []
    original = os.open

    def opened(path, flags, *args, **kwargs):
        opens.append(flags)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(source_zip.os, "open", opened)
    result = inspect(path, expected, tmp_path, budget)
    assert set(result) == {"path", "sha256", "bytes", "members"}
    assert result["path"] == str(path.resolve()) and result["sha256"] == expected
    assert result["bytes"] == path.stat().st_size
    assert result["members"] == [
        dict(name="00/writer.py", bytes=6, sha256=hashlib.sha256(b"writer").hexdigest()),
        dict(name="01/helper.py", bytes=6, sha256=hashlib.sha256(b"helper").hexdigest()),
    ]
    assert len(opens) == 2 and all(flags & os.O_NOFOLLOW for flags in opens)
    assert budget.input_bytes > 2 * path.stat().st_size


def test_internal_alias_uses_canonical_source_and_never_extracts(tmp_path):
    path, expected = capsule(tmp_path)
    alias = tmp_path / "alias.zip"
    alias.symlink_to(path)
    result = inspect(alias, expected, tmp_path)
    assert result["path"] == str(path.resolve())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["alias.zip", "analysis_source.zip"]


@pytest.mark.parametrize("kind", ["outside", "broken", "auth", "codex", "ssh"])
def test_outside_and_protected_aliases_fail_before_any_body_open(tmp_path, monkeypatch, kind):
    root = tmp_path / "allowed"
    root.mkdir()
    path, expected = capsule(tmp_path)
    alias = root / "alias.zip"
    if kind == "outside":
        alias.symlink_to(path)
    elif kind == "broken":
        alias.symlink_to(root / "missing.zip")
    elif kind == "auth":
        alias = root / "auth.json"
        alias.symlink_to(path)
    else:
        folder = root / (".codex" if kind == "codex" else ".ssh")
        folder.mkdir()
        alias = folder / "source.zip"
        alias.symlink_to(path)
    monkeypatch.setattr(source_zip.os, "open", lambda *a: pytest.fail("forbidden body opened"))
    with pytest.raises((ValueError, OSError)):
        inspect(alias, expected, root)


@pytest.mark.parametrize("expected", [None, 7, "a" * 40, "not-a-hash"])
def test_unregistered_hash_fails_before_open(tmp_path, monkeypatch, expected):
    path, _ = capsule(tmp_path)
    monkeypatch.setattr(source_zip.os, "open", lambda *a: pytest.fail("body opened"))
    with pytest.raises(ValueError, match="original metadata"):
        inspect(path, expected, tmp_path)


def test_hash_mismatch_never_opens_a_zip_member(tmp_path, monkeypatch):
    path, _ = capsule(tmp_path)
    monkeypatch.setattr(source_zip, "_members", lambda *a: pytest.fail("unbound code decoded"))
    with pytest.raises(ValueError, match="original registered hash"):
        inspect(path, "f" * 64, tmp_path)


def test_exhausted_budget_and_oversize_source_fail_before_open(tmp_path, monkeypatch):
    path, expected = capsule(tmp_path)
    budget = recovery.Budget()
    budget.input_bytes = recovery.MAX_BYTES
    monkeypatch.setattr(source_zip.os, "open", lambda *a: pytest.fail("over-budget body opened"))
    with pytest.raises(ValueError, match="combined-read"):
        inspect(path, expected, tmp_path, budget)
    monkeypatch.setattr(source_zip, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="8-MiB"):
        inspect(path, expected, tmp_path)


def test_expired_budget_does_not_touch_source_path(tmp_path, monkeypatch):
    budget = recovery.Budget()
    budget.started -= 211
    monkeypatch.setattr(source_zip, "_canonical", lambda *a: pytest.fail("path inspected"))
    with pytest.raises(TimeoutError, match="210-second"):
        inspect(tmp_path / "missing.zip", "a" * 64, tmp_path, budget)


@pytest.mark.parametrize(
    "name",
    [
        "../writer.py",
        "/writer.py",
        "./writer.py",
        "a//writer.py",
        "a\\writer.py",
        "auth.json",
        ".codex/writer.py",
        ".ssh/writer.py",
        "manifest.json",
        "writer.py/",
    ],
)
def test_unknown_traversal_and_credential_members_never_decompress(tmp_path, monkeypatch, name):
    path, expected = capsule(tmp_path, [(name, b"not-read")])
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *a, **k: pytest.fail("member opened"))
    with pytest.raises(ValueError, match="source ZIP member"):
        inspect(path, expected, tmp_path)


def test_duplicate_members_are_rejected_before_body_decode(tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="Duplicate"):
        path, expected = capsule(tmp_path, [("writer.py", b"first"), ("writer.py", b"second")])
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *a, **k: pytest.fail("member opened"))
    with pytest.raises(ValueError, match="duplicate"):
        inspect(path, expected, tmp_path)


@pytest.mark.parametrize("kind", ["symlink", "encrypted", "directory", "nul"])
def test_member_alias_and_encryption_flags_fail_closed(kind):
    member = zipfile.ZipInfo("writer.py")
    if kind == "symlink":
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
    elif kind == "encrypted":
        member.flag_bits = 1
    elif kind == "directory":
        member.external_attr = (stat.S_IFDIR | 0o755) << 16
    else:
        member.orig_filename = "writer.py\x00hidden"
    with pytest.raises(ValueError, match="source ZIP member"):
        source_zip._member_name(member)


def test_uncompressed_bomb_is_refused_before_member_open(tmp_path, monkeypatch):
    path, expected = capsule(tmp_path, [("writer.py", b"a" * 20), ("helper.py", b"b" * 20)])
    monkeypatch.setattr(source_zip, "MAX_UNCOMPRESSED_BYTES", 30)
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *a, **k: pytest.fail("bomb expanded"))
    with pytest.raises(ValueError, match="uncompressed/metadata"):
        inspect(path, expected, tmp_path)


def test_high_ratio_compressed_bomb_never_expands(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bomb.py", b"x" * (source_zip.MAX_MEMBER_BYTES + 1))
    assert len(buffer.getvalue()) < 4096
    path, expected = capsule(tmp_path, raw=buffer.getvalue())
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *a, **k: pytest.fail("bomb expanded"))
    with pytest.raises(ValueError, match="scalar bound"):
        inspect(path, expected, tmp_path)


def test_member_scalar_and_directory_node_caps_precede_decode(tmp_path, monkeypatch):
    path, expected = capsule(tmp_path, [("writer.py", b"four")])
    monkeypatch.setattr(source_zip, "MAX_MEMBER_BYTES", 3)
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *a, **k: pytest.fail("oversize member"))
    with pytest.raises(ValueError, match="scalar bound"):
        inspect(path, expected, tmp_path)
    monkeypatch.setattr(source_zip, "MAX_NODES", 1)
    monkeypatch.setattr(zipfile, "ZipFile", lambda *a, **k: pytest.fail("directory allocated"))
    with pytest.raises(ValueError, match="metadata/node"):
        inspect(path, expected, tmp_path)


def test_crc_failure_preserves_registered_hash_guard(tmp_path):
    raw = zipped([("writer.py", b"UNIQUE-CODE-BYTES")])
    raw = raw.replace(b"UNIQUE-CODE-BYTES", b"BROKEN-CODE-BYTES")
    path, expected = capsule(tmp_path, raw=raw)
    with pytest.raises(ValueError, match="integrity"):
        inspect(path, expected, tmp_path)


def test_postvalidation_alias_cannot_open_a_credential_body(tmp_path, monkeypatch):
    path, expected = capsule(tmp_path)
    protected = tmp_path / "credentials.json"
    protected.write_bytes(b"NEVER-READ-SYNTHETIC-CREDENTIAL")
    original = os.open

    def replaced(name, flags, *args, **kwargs):
        assert flags & os.O_NOFOLLOW
        path.unlink()
        path.symlink_to(protected)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(source_zip.os, "open", replaced)
    with pytest.raises(OSError):
        inspect(path, expected, tmp_path)


def test_independent_closing_hash_detects_restored_geometry_mutation(tmp_path, monkeypatch):
    path, expected = capsule(tmp_path)
    before = path.stat()
    replacement = zipped([("00/writer.py", b"print('b')\n")])
    assert len(replacement) == before.st_size
    original = source_zip._members

    def modified(*args):
        result = original(*args)
        path.write_bytes(replacement)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(source_zip, "_members", modified)
    with pytest.raises(ValueError, match="independent closing"):
        inspect(path, expected, tmp_path)


def test_comment_or_trailing_payload_is_outside_closed_capsule_format(tmp_path):
    path, expected = capsule(tmp_path, raw=zipped() + b"trailing")
    with pytest.raises(ValueError, match="directory geometry"):
        inspect(path, expected, tmp_path)


def test_shared_budget_counts_closing_pass_before_its_descriptor_open(tmp_path, monkeypatch):
    path, expected = capsule(tmp_path)
    budget = recovery.Budget()
    original = source_zip._members

    def exhausted(*args):
        result = original(*args)
        budget.input_bytes = recovery.MAX_BYTES
        return result

    monkeypatch.setattr(source_zip, "_members", exhausted)
    with pytest.raises(ValueError, match="combined-read"):
        inspect(path, expected, tmp_path, budget)


def test_existing_limits_are_not_widened():
    assert source_zip.MAX_ARCHIVE_BYTES == 8 * 2**20
    assert source_zip.MAX_UNCOMPRESSED_BYTES == 8 * 2**20
    assert source_zip.MAX_MEMBER_BYTES == 2**20
    assert source_zip.MAX_METADATA_BYTES == 8 * 2**20
    assert source_zip.MAX_NODES == 100000
    assert recovery.SECONDS == 210 and recovery.MAX_BYTES == 2 * 2**30
