"""Synthetic source-recovery contracts; no historical roots or real Git reads."""

from __future__ import annotations

import hashlib
import json
import os
import time
from types import SimpleNamespace

import evicastkv_source_recovery_v41 as helper
import pytest


@pytest.fixture
def profile_tree(tmp_path, monkeypatch):
    repo = tmp_path / "upstream"
    configs = repo / "confs"
    configs.mkdir(parents=True)
    entries = []
    for model in ("7b", "13b"):
        for ratio in range(10):
            name = f"confs/prefixkv_llava-v1.5-{model}_0.{ratio}.json"
            raw = json.dumps([ratio / 10] * 32).encode()
            (repo / name).write_bytes(raw)
            oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            entries.append(["100644", "blob", oid, name])
    history = tmp_path / "history"
    history.mkdir()
    alias = history / "confs"
    alias.symlink_to(configs, target_is_directory=True)
    calls = []

    def git(repo_, args, deadline, **kwargs):
        assert repo_ == repo
        assert deadline > time.monotonic()
        calls.append(args)
        if args == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=(helper.PREFIX_COMMIT + "\n").encode())
        assert args == ["ls-tree", "-rz", helper.PREFIX_COMMIT, "--", "confs"]
        raw = b"".join(
            f"{mode} {kind} {oid}\t{name}\0".encode() for mode, kind, oid, name in entries
        )
        return SimpleNamespace(returncode=0, stdout=raw)

    monkeypatch.setattr(helper, "git", git)
    row = dict(path=str(alias), signature=helper.signature(alias))
    return SimpleNamespace(
        repo=repo,
        configs=configs,
        alias=alias,
        history=history,
        row=row,
        entries=entries,
        calls=calls,
    )


def prove(case):
    return helper.prove_alias(
        case.row,
        [str(case.history)],
        time.monotonic() + 5,
        aliases={str(case.alias)},
        repo=case.repo,
    )


def test_exact_twenty_pinned_numeric_profiles_produce_link_not_directory_hash(profile_tree):
    case = profile_tree
    proof = prove(case)
    actual = hashlib.sha256(os.readlink(case.alias).encode()).hexdigest()
    assert proof["configuration_files_verified"] == 20
    assert proof["whole_file_sha256_before"] == proof["whole_file_sha256_after"] == actual
    assert proof["source_hash_kind"] == "symlink_target_utf8"
    assert len(proof["bindings"]) == 20
    assert proof["source_revision"] == helper.PREFIX_COMMIT
    assert proof["old_unbound_historical_version_byte_identity_claimed"] is False
    assert proof["historical_media_exposure_resolved"] is False
    assert proof["fresh_confirmation_authorized"] is False


def test_unregistered_alias_rejected_before_any_git_access(profile_tree):
    case = profile_tree
    with pytest.raises(ValueError, match="explicitly registered"):
        helper.prove_alias(
            case.row, [str(case.history)], time.monotonic() + 5, aliases=set(), repo=case.repo
        )
    assert case.calls == []


def test_alias_outside_registered_parent_scope_rejected(profile_tree):
    case = profile_tree
    with pytest.raises(ValueError, match="explicitly registered"):
        helper.prove_alias(
            case.row,
            [str(case.repo)],
            time.monotonic() + 5,
            aliases={str(case.alias)},
            repo=case.repo,
        )
    assert case.calls == []


def test_wrong_alias_signature_rejected(profile_tree):
    case = profile_tree
    case.row["signature"][2] += 1
    with pytest.raises(ValueError, match="signature changed"):
        prove(case)


def test_changed_profile_bytes_rejected_even_when_filename_is_valid(profile_tree):
    case = profile_tree
    (case.repo / case.entries[0][3]).write_text("[0.123]")
    with pytest.raises(ValueError, match="pinned source blob"):
        prove(case)


def test_profile_symlink_rejected_before_body_open(profile_tree):
    case = profile_tree
    path = case.repo / case.entries[0][3]
    path.unlink()
    path.symlink_to(case.repo / case.entries[1][3])
    with pytest.raises(ValueError, match="regular bounded"):
        prove(case)


def test_untracked_source_configuration_is_not_ignored(profile_tree):
    case = profile_tree
    (case.configs / "unexpected.json").write_text("[0.0]")
    with pytest.raises(ValueError, match="unexpected or untracked"):
        prove(case)


@pytest.mark.parametrize("field,value", [(0, "120000"), (1, "tree"), (3, "../private.json")])
def test_unregistered_tree_geometry_rejected(profile_tree, field, value):
    profile_tree.entries[0][field] = value
    with pytest.raises(ValueError, match="unregistered configuration-tree"):
        prove(profile_tree)


def test_profile_bool_is_not_a_numeric_budget(profile_tree):
    case = profile_tree
    raw = b"[true]"
    (case.repo / case.entries[0][3]).write_bytes(raw)
    case.entries[0][2] = hashlib.sha1(b"blob 6\0" + raw).hexdigest()
    with pytest.raises(ValueError, match="numeric layer-budget"):
        prove(case)


def test_alias_target_change_during_verification_rejected(profile_tree, monkeypatch):
    case = profile_tree
    original = helper.os.readlink

    def readlink(path):
        result = original(path)
        return result + "-changed" if len(case.calls) == 2 else result

    monkeypatch.setattr(helper.os, "readlink", readlink)
    with pytest.raises(ValueError, match="changed during verification"):
        prove(case)


def test_expired_git_deadline_never_starts_process(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("expired deadline must not start Git")

    monkeypatch.setattr(helper.subprocess, "run", forbidden)
    with pytest.raises(TimeoutError, match="deadline"):
        helper.git(tmp_path, ["status"], time.monotonic() - 1)


def catalog(tmp_path, monkeypatch, *, candidates=(), roots=()):
    path = tmp_path / "catalog.json"
    raw = json.dumps(
        dict(
            bounded_project_backup_catalog=dict(
                candidates=list(candidates),
                root_paths=list(roots),
            )
        )
    ).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(helper, "CATALOG", path)
    monkeypatch.setattr(helper, "CATALOG_SHA", hashlib.sha256(raw).hexdigest())
    return path


def test_forged_saved_catalog_is_rejected_before_git(tmp_path, monkeypatch):
    path = catalog(tmp_path, monkeypatch)
    path.write_text("{}")
    monkeypatch.setattr(helper, "git", lambda *a, **k: pytest.fail("unbound Git query"))
    with pytest.raises(ValueError, match="catalog changed"):
        helper.recover_sources(tmp_path, deadline=time.monotonic() + 5)


def test_oversized_catalog_rejected_before_hashing_its_body(tmp_path, monkeypatch):
    path = tmp_path / "oversized.json"
    with path.open("wb") as stream:
        stream.truncate(2**20 + 1)
    monkeypatch.setattr(helper, "CATALOG", path)
    monkeypatch.setattr(
        helper.hashlib, "sha256", lambda *a: pytest.fail("oversized body was hashed")
    )
    with pytest.raises(ValueError, match="exceeds bound"):
        helper.recover_sources(tmp_path, deadline=time.monotonic() + 5)


def test_changed_candidate_alias_cannot_trigger_git_outside_roots(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    outside = tmp_path / "external"
    owned.mkdir()
    outside.mkdir()
    source = owned / "source"
    source.mkdir()
    link = owned / "changed-repo"
    link.symlink_to(outside, target_is_directory=True)
    catalog(
        tmp_path,
        monkeypatch,
        candidates=[dict(canonical_path=str(link / ".git"))],
        roots=[str(owned)],
    )
    calls = []

    def git(repo, args, deadline, **kwargs):
        calls.append(str(repo))
        assert str(repo) == str(source), "outside-root alias reached Git before confinement"
        return SimpleNamespace(returncode=1, stdout=b"")

    monkeypatch.setattr(helper, "git", git)
    helper.recover_sources(source, deadline=time.monotonic() + 5)
    assert str(link) not in calls


def test_source_objects_are_deduplicated_but_never_certify_exposure(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    first, second, common = (owned / name for name in ("repo1", "repo2", "common.git"))
    for path in (first, second, common):
        path.mkdir(parents=True)
    catalog(
        tmp_path,
        monkeypatch,
        candidates=[dict(canonical_path=str(second / ".git"))],
        roots=[str(owned)],
    )
    batches = []

    def git(repo, args, deadline, **kwargs):
        if args[0] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=(str(common) + "\n").encode())
        assert args == ["cat-file", "--batch-check=%(objectname) %(objecttype)"]
        assert kwargs["stdin"] == "".join(r + "\n" for r in helper.REVISIONS).encode()
        batches.append(str(repo))
        return SimpleNamespace(
            returncode=0,
            stdout=(helper.REVISIONS[0] + " commit\n" + helper.REVISIONS[1] + " commit\n").encode(),
        )

    monkeypatch.setattr(helper, "git", git)
    report = helper.recover_sources(first, deadline=time.monotonic() + 5)
    found = report.get(
        "available_exact_source_commits", report.get("recovered_exact_source_commits")
    )
    assert len(batches) == 1
    assert set(found) == set(helper.REVISIONS[:2])
    assert report["missing_exact_source_commits"] == list(helper.REVISIONS[2:])
    for field in (
        "original_execution_journals_recovered",
        "original_media_bindings_recovered",
        "original_dirty_snapshot_recovered",
        "globally_fresh",
        "historical_media_exposure_resolved",
        "source_group_disjointness_verified",
        "fresh_confirmation_authorized",
        "materialization_authorized",
        "paper_campaign_authorized",
    ):
        assert report[field] is False


def test_source_recovery_deadline_retains_unvisited_failure(tmp_path, monkeypatch):
    catalog(tmp_path, monkeypatch)
    monkeypatch.setattr(helper, "git", lambda *a, **k: pytest.fail("expired source probe"))
    report = helper.recover_sources(tmp_path, deadline=time.monotonic() - 1)
    assert report["errors"] == [dict(reason="source_probe_deadline", unvisited=True)]
    assert report["fresh_confirmation_authorized"] is False
