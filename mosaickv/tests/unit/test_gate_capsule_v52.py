"""Gate I/O alias and pre-open confinement regressions on synthetic files only."""

import hashlib
import os
from pathlib import Path

import evicastkv_manifest_recovery_v52 as runner
import pytest

from mosaickv.evaluation import review_engineering_matrix_v52 as matrix


@pytest.fixture
def gates(tmp_path, monkeypatch):
    owner = tmp_path / "owner"
    owner.mkdir()
    (owner / "runs").mkdir()
    alias = tmp_path / "scratch-alias"
    alias.symlink_to(owner, target_is_directory=True)
    monkeypatch.setattr(runner.prior, "HISTORY", owner / "runs")
    raw = b'{"synthetic_control":true}'
    expected = hashlib.sha256(raw).hexdigest()
    bindings = {}
    for name in ("native", "source", "whole", "coco"):
        path = owner / name / "analysis" / "gate.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        bindings[name] = dict(path=str(path), sha256=expected)
    plan = owner / "coco" / "plan.json"
    plan.write_bytes(raw)
    monkeypatch.setattr(matrix, "KNOWN_BINDINGS", bindings)
    return owner, alias, raw, expected


@pytest.mark.parametrize("which", ["native", "source", "whole", "coco", "plan"])
@pytest.mark.parametrize("aliased", [False, True])
def test_all_gate_and_plan_paths_accept_regular_canonical_files(gates, which, aliased):
    owner, alias, raw, expected = gates
    path = owner / ("coco/plan.json" if which == "plan" else f"{which}/analysis/gate.json")
    # Any unregistered ancestor alias remains prohibited. Only the real reviewed
    # /scratch root spelling (handled by normalized()) and canonical paths qualify.
    if aliased:
        path = alias / path.relative_to(owner)
        with pytest.raises(ValueError):
            runner.gate_capsule(path, expected, runner.prior.Budget())
    else:
        assert runner.gate_capsule(path, expected, runner.prior.Budget()) == raw


def test_actual_reviewed_scratch_root_alias_is_normalized_for_plan(gates, monkeypatch):
    owner, _, raw, expected = gates
    virtual = Path("/scratch/djy8hg/coco/plan.json")
    canonical = owner / "coco/plan.json"
    resolve = Path.resolve
    is_symlink = Path.is_symlink

    def resolved(path, **kwargs):
        return canonical if path == virtual else resolve(path, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolved)
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: False if path == virtual else is_symlink(path)
    )
    assert runner.gate_capsule(virtual, expected, runner.prior.Budget()) == raw


@pytest.mark.parametrize("target", ["ordinary.json", "credentials.json", ".codex/auth.json"])
def test_registered_leaf_symlinks_are_rejected_before_open(gates, monkeypatch, target):
    owner, _, _, expected = gates
    leaf = owner / "native/analysis/gate.json"
    target_path = owner / target
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"SYNTHETIC_SECRET_NEVER_READ")
    leaf.unlink()
    leaf.symlink_to(target_path)
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("symlink target opened"))
    with pytest.raises(ValueError, match="symlinks"):
        runner.gate_capsule(leaf, expected, runner.prior.Budget())


def test_replaced_ancestor_cannot_redirect_a_registered_gate(gates, monkeypatch):
    owner, _, _, expected = gates
    leaf = owner / "native/analysis/gate.json"
    original = owner / "native/analysis"
    moved = owner / "moved"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("redirected target opened"))
    with pytest.raises(ValueError, match="canonical"):
        runner.gate_capsule(leaf, expected, runner.prior.Budget())


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_nonregular_gate_is_rejected_before_open(gates, monkeypatch, kind):
    owner, _, _, expected = gates
    leaf = owner / "native/analysis/gate.json"
    leaf.unlink()
    os.mkfifo(leaf) if kind == "fifo" else leaf.mkdir()
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("nonregular gate opened"))
    with pytest.raises(ValueError, match="regular"):
        runner.gate_capsule(leaf, expected, runner.prior.Budget())


def test_hash_mismatch_preserves_registered_evidence(gates):
    owner, _, _, _ = gates
    with pytest.raises(ValueError, match="hash"):
        runner.gate_capsule(owner / "coco/plan.json", "0" * 64, runner.prior.Budget())


@pytest.mark.parametrize("limit", ["time", "bytes", "live"])
def test_gate_reader_retains_all_original_limits(gates, monkeypatch, limit):
    owner, _, _, expected = gates
    budget = runner.prior.Budget()
    if limit == "time":
        budget.started -= 211
    elif limit == "bytes":
        budget.input_bytes = runner.prior.MAX_BYTES
    else:
        monkeypatch.setattr(runner.prior, "LIVE_BYTES", 1)
    monkeypatch.setattr(runner.os, "open", lambda *a, **k: pytest.fail("limit checked after open"))
    with pytest.raises((ValueError, TimeoutError)):
        runner.gate_capsule(owner / "coco/plan.json", expected, budget)
