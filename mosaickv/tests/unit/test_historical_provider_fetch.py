"""Mocked public metadata/downloads; no network or benchmark-column reads."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import evicastkv_historical_provider_fetch as fetch
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return fetch.sha(path)


def packet():
    return dict(
        repository=fetch.REPO,
        revision=fetch.REVISION,
        private=False,
        gated=False,
        files=copy.deepcopy(fetch.PINS),
        total_bytes=sum(r["size"] for r in fetch.PINS),
    )


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    hub, root = tmp_path / "hub", tmp_path / "download"
    hub.mkdir()
    card = hub / "datasets--lmms-lab--DocVQA/snapshots" / fetch.REVISION / "README.md"
    card.parent.mkdir(parents=True)
    card.write_bytes(b"Pinned metadata fixture")
    monkeypatch.setattr(fetch, "HUB", hub)
    monkeypatch.setattr(fetch, "clean_source", lambda *a: "a" * 40)
    monkeypatch.setattr(fetch, "cpu_gate", lambda *a: dict(status="passed"))
    monkeypatch.setattr(fetch.providers, "card", lambda *a: ({}, fetch.sha(card)))
    monkeypatch.setattr(
        fetch.providers, "pool_description", lambda *a: (["InfographicVQA/validation-*"], 2801)
    )
    monkeypatch.setattr(fetch, "call_child", lambda *a, **k: packet())
    return dict(root=root, hub=hub)


def test_prepare_is_metadata_only_and_pins_exact_four_files(fixture, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    result = fetch.prepare(fixture["root"], cpu_gate_root=fixture["root"].parent)
    plan = fetch.load(fixture["root"])
    assert result["files"] == 4 and result["maximum_download_bytes"] == 266457648
    assert plan["files"] == fetch.PINS
    assert plan["benchmark_columns_decoded"] is False
    assert not plan["video_exposure_resolved"]


def test_reload_passes_path_not_JSON_string_to_real_gate_contract(fixture, monkeypatch):
    seen = []

    def gate(root, revision):
        assert isinstance(root, Path)
        seen.append(root)
        return {"status": "passed"}

    monkeypatch.setattr(fetch, "cpu_gate", gate)
    fetch.prepare(fixture["root"], cpu_gate_root=str(fixture["root"].parent))
    fetch.load(fixture["root"])
    assert seen == [fixture["root"].parent] * 2


def test_public_IO_scope_removes_tokens_and_only_local_child_disables_offline(fixture, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "SECRET_TOKEN")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "OTHER_SECRET")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    env = fetch.io_environment()
    assert env["HF_HUB_OFFLINE"] == "0" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"


def api(monkeypatch, *, wrong=False, private=False):
    class API:
        def __init__(self, **kwargs):
            assert kwargs == dict(endpoint=fetch.ENDPOINT, token=False)

        def repo_info(self, *args, **kwargs):
            assert kwargs["token"] is False
            return SimpleNamespace(sha=fetch.REVISION, private=private, gated=False)

        def list_repo_tree(self, *args, **kwargs):
            assert kwargs["token"] is False and kwargs["revision"] == fetch.REVISION
            for pin in fetch.PINS:
                yield SimpleNamespace(
                    path=pin["path"],
                    size=pin["size"],
                    blob_id=pin["git_blob_id"],
                    lfs=SimpleNamespace(
                        size=pin["size"], sha256="bad" if wrong else pin["lfs_sha256"]
                    ),
                )

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(__version__=fetch.HUB_VERSION, HfApi=API)
    )


def test_metadata_requires_exact_public_revision_Git_and_LFS_objects(monkeypatch):
    api(monkeypatch)
    assert fetch.remote_metadata() == packet()
    api(monkeypatch, wrong=True)
    with pytest.raises(ValueError, match="OIDs differ"):
        fetch.remote_metadata()
    api(monkeypatch, private=True)
    with pytest.raises(ValueError, match="public ungated"):
        fetch.remote_metadata()


def synthetic_files(fixture, monkeypatch):
    pins, bodies = copy.deepcopy(fetch.PINS), []
    for index, pin in enumerate(pins):
        raw = f"synthetic binary file {index}".encode()
        bodies.append(raw)
        pin.update(size=len(raw), lfs_sha256=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(fetch, "PINS", pins)
    fetch.prepare(fixture["root"], cpu_gate_root=fixture["root"].parent)
    calls = []

    def download(repo, name, **kwargs):
        assert repo == fetch.REPO and kwargs["token"] is False
        assert kwargs["revision"] == fetch.REVISION and not kwargs["force_download"]
        assert kwargs["endpoint"] == fetch.ENDPOINT
        index = next(i for i, pin in enumerate(pins) if pin["path"] == name)
        base = fixture["hub"] / "datasets--lmms-lab--DocVQA"
        alias = base / "snapshots" / fetch.REVISION / name
        blob = base / "blobs" / pins[index]["lfs_sha256"]
        blob.parent.mkdir(exist_ok=True)
        if not blob.exists():
            blob.write_bytes(bodies[index])
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists():
            alias.symlink_to(blob)
        calls.append(name)
        return str(alias)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(__version__=fetch.HUB_VERSION, hf_hub_download=download),
    )
    return pins, bodies, calls


def test_download_verifies_and_reuses_complete_LFS_body_without_payload_decoding(
    fixture, monkeypatch
):
    _pins, _bodies, calls = synthetic_files(fixture, monkeypatch)
    first = fetch.fetch_one(fixture["root"], 0)
    again = fetch.fetch_one(fixture["root"], 0)
    assert not first["reused_complete_blob"] and again["reused_complete_blob"]
    assert first["artifact"] == again["artifact"]
    assert len(calls) == 2
    assert not first["benchmark_columns_decoded"]


def test_corrupt_complete_blob_is_preserved_not_redownloaded(fixture, monkeypatch):
    pins, _bodies, calls = synthetic_files(fixture, monkeypatch)
    blob = fixture["hub"] / "datasets--lmms-lab--DocVQA/blobs" / pins[0]["lfs_sha256"]
    blob.parent.mkdir()
    blob.write_bytes(b"old corrupt complete bytes")
    before = blob.read_bytes()
    with pytest.raises(ValueError, match="preserved without overwrite"):
        fetch.fetch_one(fixture["root"], 0)
    assert blob.read_bytes() == before and calls == []


def test_downloaded_bad_content_cannot_pass(fixture, monkeypatch):
    _pins, _bodies, _calls = synthetic_files(fixture, monkeypatch)
    fetch.fetch_one(fixture["root"], 0)
    record = fetch.fetch_one(fixture["root"], 0)
    Path(record["artifact"]["path"]).write_bytes(b"bad download")
    with pytest.raises(ValueError, match="preserved without overwrite"):
        fetch.fetch_one(fixture["root"], 0)


def test_redirected_cache_directory_is_rejected_before_download(fixture, monkeypatch, tmp_path):
    _pins, _bodies, calls = synthetic_files(fixture, monkeypatch)
    external = tmp_path / "outside-blobs"
    external.mkdir()
    repository = fixture["hub"] / "datasets--lmms-lab--DocVQA"
    (repository / "blobs").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="redirects outside"):
        fetch.fetch_one(fixture["root"], 0)
    assert calls == [] and list(external.iterdir()) == []


def test_worker_failure_stops_without_retry_and_preserves_all_denominators(fixture, monkeypatch):
    fetch.prepare(fixture["root"], cpu_gate_root=fixture["root"].parent)
    calls = []

    def failed(*args, **kwargs):
        calls.append(kwargs["index"])
        raise subprocess.TimeoutExpired("public IO child", kwargs["timeout"])

    monkeypatch.setattr(fetch, "call_child", failed)
    assert fetch.worker(fixture["root"]) == dict(completed=0, unresolved=4)
    assert calls == [0]
    gate = fetch.analyze(fixture["root"])
    assert gate["status"] == "pinned_download_incomplete" and gate["unresolved_files"] == 4
    assert not gate["fresh_confirmation_authorized"]


def test_complete_downloads_still_do_not_certify_provider_media_or_video(fixture, monkeypatch):
    synthetic_files(fixture, monkeypatch)
    monkeypatch.setattr(
        fetch, "call_child", lambda mode, **k: fetch.fetch_one(k["root"], k["index"])
    )
    assert fetch.worker(fixture["root"]) == dict(completed=4, unresolved=0)
    gate = fetch.analyze(fixture["root"])
    assert gate["status"] == "pinned_download_complete"
    assert not gate["provider_pixel_coverage_certified"] and not gate["video_exposure_resolved"]
    assert not gate["fresh_confirmation_authorized"]


def test_missing_worker_and_changed_download_provenance_fail_closed(fixture, monkeypatch):
    synthetic_files(fixture, monkeypatch)
    gate = fetch.analyze(fixture["root"])
    assert gate["missing_worker"] and gate["status"] == "pinned_download_incomplete"
    with pytest.raises(FileExistsError):
        fetch.analyze(fixture["root"])


def test_post_worker_body_mutation_prevents_complete_gate(fixture, monkeypatch):
    synthetic_files(fixture, monkeypatch)
    monkeypatch.setattr(
        fetch, "call_child", lambda mode, **k: fetch.fetch_one(k["root"], k["index"])
    )
    fetch.worker(fixture["root"])
    record = fetch.read(fixture["root"] / "worker.json")["completed"][0]
    Path(record["artifact"]["path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="preserved without overwrite"):
        fetch.analyze(fixture["root"])
    assert not (fixture["root"] / "GATE_historical_provider_fetch.json").exists()
