"""Saved-proof reader confinement; only synthetic scratch fixtures are used."""

import hashlib
import os
from types import SimpleNamespace

import evicastkv_freeze_writer_recovery_v54 as runner
import pytest


@pytest.fixture
def saved(tmp_path, monkeypatch):
    owner = tmp_path / "owner"
    owner.mkdir()
    alias = tmp_path / "scratch-alias"
    alias.symlink_to(owner, target_is_directory=True)
    path = owner / "receipt.json"
    raw = b'{"source_sha":"synthetic","seconds":0.2}'
    path.write_bytes(raw)
    monkeypatch.setattr(runner, "SAVED_OWNER_ROOT", owner)
    monkeypatch.setattr(runner, "SAVED_ALIAS_ROOT", alias)
    monkeypatch.setattr(runner, "SAVED_INPUTS", frozenset({path}))
    return owner, alias, path, raw, hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("aliased", [False, True])
def test_single_protected_read_charges_exact_bytes_and_retains_digest(saved, monkeypatch, aliased):
    owner, alias, path, raw, expected = saved
    supplied = alias / path.relative_to(owner) if aliased else path
    opened = []
    original = runner.os.open

    def observed(target, flags, *args, **kwargs):
        opened.append(target)
        assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
        return original(target, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", observed)
    budget = runner.prior.Budget()
    assert runner.saved_json(supplied, expected, budget)["seconds"] == 0.2
    assert budget.input_bytes == len(raw) and opened == [path]
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("digest reopened metadata"))
    assert runner.saved_sha(supplied, budget) == expected


@pytest.mark.parametrize("target_name", ["ordinary.json", "credentials.json", ".codex/auth.json"])
def test_leaf_alias_cannot_open_protected_or_unregistered_target(saved, monkeypatch, target_name):
    owner, _, path, _, expected = saved
    target = owner / target_name
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_bytes(b"SYNTHETIC_PRIVATE_VALUE_NEVER_READ")
    path.unlink()
    path.symlink_to(target)
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("symlink target opened"))
    with pytest.raises(ValueError):
        runner.saved_json(path, expected, runner.prior.Budget())


def test_hardlinked_private_content_is_rejected_before_open(saved, monkeypatch):
    owner, _, path, _, expected = saved
    private = owner / "credentials.json"
    private.write_bytes(b"SYNTHETIC_NEVER_READ")
    path.unlink()
    os.link(private, path)
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("hardlink body opened"))
    with pytest.raises(ValueError, match="single-owner"):
        runner.saved_json(path, expected, runner.prior.Budget())


def test_redirected_ancestor_is_rejected_before_open(saved, monkeypatch):
    owner, _, _, raw, expected = saved
    container = owner / "container"
    container.mkdir()
    path = container / "receipt.json"
    path.write_bytes(raw)
    moved = owner / "moved"
    container.rename(moved)
    container.symlink_to(moved, target_is_directory=True)
    monkeypatch.setattr(runner, "SAVED_INPUTS", frozenset({path}))
    monkeypatch.setattr(
        runner.os, "open", lambda *a, **k: pytest.fail("ancestor redirected body opened")
    )
    with pytest.raises(ValueError, match="ancestor"):
        runner.saved_json(path, expected, runner.prior.Budget())


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_nonregular_metadata_is_rejected_without_blocking(saved, monkeypatch, kind):
    _, _, path, _, expected = saved
    path.unlink()
    os.mkfifo(path) if kind == "fifo" else path.mkdir()
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("nonregular object opened"))
    with pytest.raises(ValueError, match="regular"):
        runner.saved_json(path, expected, runner.prior.Budget())


@pytest.mark.parametrize("limit", ["time", "combined", "live"])
def test_limits_are_checked_before_open(saved, monkeypatch, limit):
    _, _, path, _, expected = saved
    budget = runner.prior.Budget()
    if limit == "time":
        budget.started -= 211
    elif limit == "combined":
        budget.input_bytes = runner.prior.MAX_BYTES
    else:
        monkeypatch.setattr(runner.prior, "LIVE_BYTES", 1)
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("limit checked after open"))
    with pytest.raises((ValueError, TimeoutError)):
        runner.saved_json(path, expected, budget)


def test_stat_to_open_growth_is_rejected_before_decode(saved, monkeypatch):
    _, _, path, raw, expected = saved
    original = runner.os.open

    def raced(target, flags, *args, **kwargs):
        path.write_bytes(raw + b"changed")
        return original(target, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", raced)
    budget = runner.prior.Budget()
    with pytest.raises(ValueError, match="before read"):
        runner.saved_json(path, expected, budget)
    assert budget.input_bytes == 0


def test_fifo_substitution_cannot_block_descriptor_acquisition(saved, monkeypatch):
    _, _, path, _, expected = saved
    original = runner.os.open

    def raced(target, flags, *args, **kwargs):
        path.unlink()
        os.mkfifo(path)
        assert flags & os.O_NONBLOCK
        return original(target, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", raced)
    with pytest.raises(ValueError, match="before read"):
        runner.saved_json(path, expected, runner.prior.Budget())


def test_descriptor_mutation_is_rejected(saved, monkeypatch):
    _, _, path, _, expected = saved
    original = runner.os.fstat
    calls = 0

    def changed(fd):
        nonlocal calls
        calls += 1
        info = original(fd)
        if calls == 1:
            return info
        return SimpleNamespace(
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_size=info.st_size,
            st_mtime_ns=info.st_mtime_ns + 1,
            st_ctime_ns=info.st_ctime_ns,
        )

    monkeypatch.setattr(runner.os, "fstat", changed)
    with pytest.raises(ValueError, match="during read"):
        runner.saved_json(path, expected, runner.prior.Budget())


@pytest.mark.parametrize(
    "raw",
    [
        b'{"duplicate":1,"duplicate":2}',
        b'{"nonfinite":NaN}',
        b'{"nonfinite":1e309}',
        b"[]",
        b"{} {}",
        b'{"bad":"\\q"}',
    ],
)
def test_malformed_saved_packets_never_become_proofs(saved, raw):
    _, _, path, _, _ = saved
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        runner.saved_json(path, None, runner.prior.Budget())


def test_digest_failure_precedes_any_json_decoding(saved, monkeypatch):
    _, _, path, _, _ = saved
    monkeypatch.setattr(
        runner.json, "loads", lambda *a, **k: pytest.fail("unverified metadata decoded")
    )
    with pytest.raises(ValueError, match="digest"):
        runner.saved_json(path, "0" * 64, runner.prior.Budget())


def test_unknown_paths_and_scope_names_are_rejected_before_open(saved, monkeypatch):
    owner, _, _, _, expected = saved
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("out-of-scope body opened"))
    for path in [owner / "other.json", owner / ".codex/auth.json", owner / "credentials.json"]:
        with pytest.raises(ValueError):
            runner.saved_json(path, expected, runner.prior.Budget())


@pytest.mark.parametrize("location", ["key", "value"])
def test_saved_json_scalar_width_applies_to_keys_and_values(saved, location):
    import json

    _, _, path, _, _ = saved
    large = "x" * (2**20 + 1)
    path.write_text(json.dumps({large: 0} if location == "key" else {"value": large}))
    with pytest.raises(ValueError, match="scalar"):
        runner.saved_json(path, None, runner.prior.Budget())


def test_saved_json_depth_and_node_caps_are_unchanged(saved, monkeypatch):
    _, _, path, _, _ = saved
    path.write_bytes(b'{"root":' + b"[" * 65 + b"0" + b"]" * 65 + b"}")
    with pytest.raises(ValueError, match="node/depth"):
        runner.saved_json(path, None, runner.prior.Budget())
    path.write_bytes(b'{"root":[0,1,2,3]}')
    monkeypatch.setattr(runner.prior, "LIVE_NODES", 3)
    with pytest.raises(ValueError, match="node/depth"):
        runner.saved_json(path, None, runner.prior.Budget())
