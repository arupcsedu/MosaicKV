"""Synthetic operational fixtures only; no model/cache/dataset access."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evicastkv_capture_confirmation_processor.py"


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("processor_capture_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def synthetic_hub(tmp_path, module):
    root = tmp_path / "synthetic-hub"
    snapshot = root / "snapshots" / module.MODEL_REVISION
    blobs = root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    for name in (*sorted(module.REQUIRED_NAMES), "tokenizer.json"):
        raw = json.dumps({"synthetic_fixture": name}).encode()
        blob_id = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        (blobs / blob_id).write_bytes(raw)
        (snapshot / name).symlink_to("../../blobs/" + blob_id)
    return snapshot


def test_staging_hashes_only_allowlisted_processor_assets(tmp_path, module):
    snapshot = synthetic_hub(tmp_path, module)
    (snapshot / "model.safetensors").write_bytes(b"must-not-be-opened")
    result = module.stage_assets(snapshot, tmp_path / "staged")
    assert {r["name"] for r in result["assets"]} == {*module.REQUIRED_NAMES, "tokenizer.json"}
    assert result["source_plus_readback_bytes"] == 2 * result["staged_bytes"]
    assert not (tmp_path / "staged/model.safetensors").exists()


def test_escaping_link_is_rejected_without_resolving_its_target(tmp_path, module):
    snapshot = synthetic_hub(tmp_path, module)
    asset = snapshot / "config.json"
    asset.unlink()
    asset.symlink_to("/home/forbidden/.codex/auth.json")
    with pytest.raises(ValueError, match="escapes"):
        module.stage_assets(snapshot, tmp_path / "staged")


def test_content_address_mismatch_rejected(tmp_path, module):
    snapshot = synthetic_hub(tmp_path, module)
    blob = snapshot.parent.parent / "blobs" / os.readlink(snapshot / "config.json").split("/")[-1]
    blob.write_bytes(b"modified")
    with pytest.raises(ValueError, match="content-address"):
        module.stage_assets(snapshot, tmp_path / "staged")


def test_nonblob_regular_snapshot_asset_rejected(tmp_path, module):
    snapshot = synthetic_hub(tmp_path, module)
    asset = snapshot / "config.json"
    asset.unlink()
    asset.write_text("{}")
    with pytest.raises(ValueError, match="content-addressed link"):
        module.stage_assets(snapshot, tmp_path / "staged")


def test_missing_required_asset_and_byte_bounds_fail(tmp_path, module, monkeypatch):
    snapshot = synthetic_hub(tmp_path, module)
    monkeypatch.setattr(module, "MAX_STAGED", 1)
    with pytest.raises(ValueError, match="limits"):
        module.stage_assets(snapshot, tmp_path / "staged")
    monkeypatch.setattr(module, "MAX_STAGED", 64 * 1024**2)
    (snapshot / "config.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        module.stage_assets(snapshot, tmp_path / "staged-missing")


def test_bwrap_mounts_no_original_cache_weights_home_or_data(tmp_path, module):
    command = module.sandbox_command(
        tmp_path, source=Path("/synthetic-code"), environment=Path("/synthetic-env")
    )
    assert "--unshare-net" in command
    assert "--clearenv" not in command  # installed cluster bubblewrap predates this option
    assert "--dev" in command
    assert "/runtime/bin/python" in command
    mounts = []
    for i, value in enumerate(command):
        if value in {"--ro-bind", "--bind"}:
            mounts.append((command[i + 1], command[i + 2]))
    assert (str(tmp_path / "assets"), str(module.SANDBOX_SNAPSHOT)) in mounts
    assert not any(
        any(part in source for part in ("/home/", "/huggingface/", "/datasets/", "/blobs/"))
        for source, _ in mounts
    )
    assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" in command
    assert "CUDA_VISIBLE_DEVICES" in command


def test_subprocess_environment_drops_inherited_credentials_and_preloads(module, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "synthetic-unused-token-fixture")
    monkeypatch.setenv("LD_PRELOAD", "/synthetic/unused-preload.so")
    monkeypatch.setenv("HOME", "/synthetic/unused-home")
    assert module.sandbox_environment() == {"PATH": "/usr/bin", "LANG": "C.UTF-8"}


def test_child_rejects_wrong_environment_before_import_or_loader(module, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = []
    with pytest.raises(RuntimeError, match="sandbox environment"):
        module.capture_child(loader=lambda *a, **kw: calls.append((a, kw)))
    assert calls == []


def test_injected_safe_loader_receives_offline_flags_and_only_fingerprints(module, monkeypatch):
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HOME": "/nonexistent",
        "OMP_NUM_THREADS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    limits = []
    monkeypatch.setattr(module.resource, "setrlimit", lambda *args: limits.append(args))
    monkeypatch.setattr(module.signal, "alarm", lambda *args: None)

    class SyntheticProcessorFixture:
        chat_template = "{{ messages }}"
        image_processor = SimpleNamespace(to_dict=lambda: {"size": 2})
        tokenizer = SimpleNamespace(
            chat_template=None,
            special_tokens_map={"bos_token": "<s>"},
            get_vocab=lambda: {"<s>": 0, "cat": 1},
        )

    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return SyntheticProcessorFixture()

    result = module.capture_child(loader=loader)
    assert calls == [
        (
            (module.MODEL_ID,),
            {
                "revision": module.MODEL_REVISION,
                "local_files_only": True,
                "token": False,
                "trust_remote_code": False,
            },
        )
    ]
    assert result["status"] == "captured_unreviewed"
    assert "vocabulary" not in str(result)
    assert (
        result["model_weights_loaded"] is result["dataset_accessed"] is result["GPU_used"] is False
    )
    assert (module.resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3)) in limits


def test_capture_descriptor_never_implies_review_or_execution(module):
    child = dict(
        processor_class="Synthetic",
        **{
            k: "b" * 64
            for k in (
                "processor_signature_sha256",
                "preprocessing_sha256",
                "tokenization_sha256",
                "chat_template_sha256",
            )
        },
    )
    result = module.processor_descriptor(child, "c" * 64, "d" * 40)
    assert result["reviewed"] is False
    assert result["provenance"]["capture_manifest_sha256"] == "c" * 64
    assert result["processor_revision"] == module.MODEL_REVISION


def test_promotion_requires_matching_derived_hashes_and_explicit_codex_review(tmp_path, module):
    child = dict(
        status="captured_unreviewed",
        processor_class="Synthetic",
        model_weights_loaded=False,
        dataset_accessed=False,
        GPU_used=False,
        **{
            k: "b" * 64
            for k in (
                "processor_signature_sha256",
                "preprocessing_sha256",
                "tokenization_sha256",
                "chat_template_sha256",
            )
        },
    )
    (tmp_path / "child").mkdir()
    child_raw = json.dumps(child).encode()
    (tmp_path / "child/processor_fingerprints.json").write_bytes(child_raw)
    manifest = dict(
        status="captured_unreviewed",
        model_id=module.MODEL_ID,
        model_revision=module.MODEL_REVISION,
        child_sha256=hashlib.sha256(child_raw).hexdigest(),
        source={"source_sha": "d" * 40},
    )
    raw = json.dumps(manifest).encode()
    (tmp_path / "capture_manifest.json").write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="explicit Codex"):
        module.promote_reviewed_capture(
            tmp_path,
            expected_manifest_sha256=expected,
            reviewer="human",
            output=tmp_path / "invalid.json",
        )
    with pytest.raises(ValueError, match="changed"):
        module.promote_reviewed_capture(
            tmp_path,
            expected_manifest_sha256="e" * 64,
            reviewer="codex",
            output=tmp_path / "invalid.json",
        )
    result = module.promote_reviewed_capture(
        tmp_path,
        expected_manifest_sha256=expected,
        reviewer="codex",
        output=tmp_path / "reviewed.json",
    )
    assert result["reviewed"] is True
    assert result["review"]["actor"] == "codex"
    with pytest.raises(FileExistsError):
        module.promote_reviewed_capture(
            tmp_path,
            expected_manifest_sha256=expected,
            reviewer="codex",
            output=tmp_path / "reviewed.json",
        )


@pytest.mark.parametrize(
    "path",
    [
        "/home/example/file",
        "/scratch/djy8hg/workdir/../bad",
        "/scratch/djy8hg/workdir/.codex/auth.json",
    ],
)
def test_home_auth_and_traversal_paths_rejected_lexically(module, path):
    with pytest.raises(ValueError, match="non-home scratch"):
        module.canonical_scratch(path)
