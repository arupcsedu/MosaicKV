from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

import mosaickv.manifest as manifest_module
from mosaickv.config import synthetic_smoke_config
from mosaickv.manifest import (
    ArtifactProvenance,
    EvaluationProvenance,
    InputProvenance,
    ManifestError,
    RunManifestWriter,
    sha256_text,
)
from mosaickv.types import Backend, JsonObject, MeasurementType, ResultCategory


def _inputs() -> InputProvenance:
    return InputProvenance(*(sha256_text(f"input-{index}") for index in range(4)))


def _artifacts() -> ArtifactProvenance:
    return ArtifactProvenance(
        raw_output_sha=sha256_text("raw"),
        metrics_sha="not_applicable",
        log_sha=sha256_text("log"),
    )


def _paper_evaluation() -> EvaluationProvenance:
    return EvaluationProvenance(
        evaluator_name="fixture",
        evaluator_version="1",
        evaluator_commit="a" * 40,
        reference_output_sha256="b" * 64,
        baseline_name="fixture_reimpl",
        baseline_commit="c" * 40,
    )


def test_manifest_contains_all_agent_required_provenance() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    manifest = RunManifestWriter(repo_root).build(
        synthetic_smoke_config(),
        MeasurementType.VALIDATION_SMOKE,
        _inputs(),
        _artifacts(),
        run_id="unit-test-run",
        started_at_utc="2026-07-19T00:00:00+00:00",
    )

    source = cast("JsonObject", manifest["source"])
    model = cast("JsonObject", manifest["model"])
    dataset = cast("JsonObject", manifest["dataset"])
    software = cast("JsonObject", manifest["software"])
    hardware = cast("JsonObject", manifest["hardware"])
    execution = cast("JsonObject", manifest["execution"])
    assert len(cast("str", source["git_sha"])) == 40
    assert len(cast("str", source["config_sha"])) == 64
    assert model["id"] == "synthetic/smoke"
    assert dataset["revision"] == "schema-v1"
    assert set(software) >= {
        "cuda",
        "driver",
        "pytorch",
        "transformers",
        "vllm",
        "sglang",
    }
    assert set(hardware) == {"gpu_type", "gpu_count"}
    assert execution["backend"] == "synthetic"
    assert execution["attention_implementation"] == "numpy"
    assert execution["seed"] == 0
    assert manifest["measurement_type"] == "validation_smoke"
    assert manifest["result_category"] == "unified_comparison"
    assert manifest["result_origin"] == "locally_measured"


def test_manifest_explicitly_labels_paper_reproduction() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    manifest = RunManifestWriter(repo_root).build(
        synthetic_smoke_config(),
        MeasurementType.METHOD,
        _inputs(),
        _artifacts(),
        result_category=ResultCategory.PAPER_REPRODUCTION,
        evaluation_provenance=_paper_evaluation(),
    )
    assert manifest["result_category"] == "paper_reproduction"
    assert manifest["result_origin"] == "locally_reproduced"
    assert manifest["evaluation"]["evaluator_version"] == "1"
    assert manifest["evaluation"]["reference_output_sha256"] == "b" * 64
    assert manifest["evaluation"]["baseline_commit"] == "c" * 40


def test_paper_manifest_rejects_missing_baseline_commit() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(ManifestError, match="immutable baseline commit"):
        RunManifestWriter(repo_root).build(
            synthetic_smoke_config(),
            MeasurementType.METHOD,
            _inputs(),
            _artifacts(),
            result_category=ResultCategory.PAPER_REPRODUCTION,
            evaluation_provenance=EvaluationProvenance(
                evaluator_name="fixture",
                evaluator_version="1",
                evaluator_commit="a" * 40,
                reference_output_sha256="b" * 64,
                baseline_name="fixture_reimpl",
                baseline_commit="not_applicable",
            ),
        )


def test_paper_manifest_rejects_missing_evaluation_provenance() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(ManifestError, match="require evaluator"):
        RunManifestWriter(repo_root).build(
            synthetic_smoke_config(),
            MeasurementType.METHOD,
            _inputs(),
            _artifacts(),
            result_category=ResultCategory.PAPER_REPRODUCTION,
        )


def test_manifest_writer_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    destination = tmp_path / "run_manifest.json"
    writer = RunManifestWriter(repo_root)
    writer.write(
        destination,
        synthetic_smoke_config(),
        MeasurementType.VALIDATION_SMOKE,
        _inputs(),
        _artifacts(),
    )
    parsed = json.loads(destination.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        writer.write(
            destination,
            synthetic_smoke_config(),
            MeasurementType.VALIDATION_SMOKE,
            _inputs(),
            _artifacts(),
        )


def test_manifest_rejects_non_sha_input() -> None:
    with pytest.raises(ManifestError, match=r"inputs\.prompt_set_sha"):
        InputProvenance("not-a-sha", *(sha256_text(str(index)) for index in range(3)))


def test_manifest_records_engine_execution_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    manifest = RunManifestWriter(repo_root).build(
        synthetic_smoke_config(),
        MeasurementType.VALIDATION_SMOKE,
        _inputs(),
        _artifacts(),
        execution_metadata={
            "engine_execution_mode": "eager",
            "attention_backend_configuration": "auto",
        },
        attention_implementation_override="vllm_auto",
    )
    execution = cast("JsonObject", manifest["execution"])
    assert execution["engine_execution_mode"] == "eager"
    assert execution["attention_backend_configuration"] == "auto"
    assert execution["attention_implementation"] == "vllm_auto"
    with pytest.raises(ManifestError, match="cannot replace required field"):
        RunManifestWriter(repo_root).build(
            synthetic_smoke_config(),
            MeasurementType.VALIDATION_SMOKE,
            _inputs(),
            _artifacts(),
            execution_metadata={"seed": 999},
        )


def test_vllm_manifest_records_transformers_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        manifest_module,
        "_version",
        lambda name: {"torch": "2.9.0", "transformers": "4.57.6"}.get(name, "not_installed"),
    )
    software = manifest_module._software_provenance(Backend.VLLM, 1, "595.71.05")
    assert software["transformers"] == "4.57.6"


def test_software_provenance_records_all_installed_backend_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "torch": "2.8.0",
        "transformers": "4.55.2",
        "vllm": "0.10.2",
        "sglang": "0.5.1.post3",
        "numpy": "1.26.4",
    }
    monkeypatch.setattr(manifest_module, "_version", versions.__getitem__)
    monkeypatch.setattr(manifest_module, "_torch_cuda_version", lambda: "12.8")

    software = manifest_module._software_provenance(Backend.HUGGINGFACE, 1, "595.71.05")

    assert software["vllm"] == "0.10.2"
    assert software["sglang"] == "0.5.1.post3"


def test_git_status_timeout_retries_identical_complete_command(monkeypatch, tmp_path):
    command = ["git", "status", "--porcelain=v1", "-z"]
    calls = []

    def fake_run(actual, **kwargs):
        calls.append((actual.copy(), kwargs["timeout"]))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(actual, kwargs["timeout"])
        return subprocess.CompletedProcess(actual, 0, stdout="", stderr="")

    monkeypatch.setattr(manifest_module.subprocess, "run", fake_run)
    assert manifest_module._run(command, tmp_path) == ""
    assert calls == [(command, 30), (command, 120)]


@pytest.mark.parametrize(
    "dirty", [" M source.py\n", "?? untracked.py\n", " m third_party/PrefixKV\n"]
)
def test_retried_status_still_detects_dirty_worktree(monkeypatch, tmp_path, dirty):
    statuses = 0

    def fake_run(command, **kwargs):
        nonlocal statuses
        if command[:2] == ["git", "status"]:
            statuses += 1
            if statuses == 1:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            value = dirty
        else:
            value = "a" * 40
        return subprocess.CompletedProcess(command, 0, stdout=value, stderr="")

    monkeypatch.setattr(manifest_module.subprocess, "run", fake_run)
    monkeypatch.setattr(manifest_module, "_worktree_fingerprint", lambda path: "b" * 64)
    source = manifest_module._source_provenance(tmp_path)
    assert source["git_dirty"] is True
    assert source["canonical_eligible"] is False
    assert source["patch_sha"] == "b" * 64


def test_exhausted_status_timeout_is_never_clean(monkeypatch, tmp_path):
    calls = []

    def timeout(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(manifest_module.subprocess, "run", timeout)
    with pytest.raises(ManifestError, match="cleanliness remains unknown"):
        manifest_module._run(["git", "status", "--porcelain=v1"], tmp_path)
    assert calls == [30, 120]


def test_non_status_timeout_does_not_gain_unbounded_retries(monkeypatch, tmp_path):
    calls = []

    def timeout(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(manifest_module.subprocess, "run", timeout)
    with pytest.raises(ManifestError):
        manifest_module._run(["git", "diff", "--binary", "HEAD"], tmp_path)
    assert calls == [30]


def test_failed_status_command_is_not_treated_as_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(
        manifest_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 128, stdout="", stderr="fatal"
        ),
    )
    with pytest.raises(ManifestError, match="fatal"):
        manifest_module._run(["git", "status", "--porcelain=v1"], tmp_path)
