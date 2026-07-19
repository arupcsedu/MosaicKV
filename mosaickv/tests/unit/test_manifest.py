from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from mosaickv.config import synthetic_smoke_config
from mosaickv.manifest import (
    ArtifactProvenance,
    InputProvenance,
    ManifestError,
    RunManifestWriter,
    sha256_text,
)
from mosaickv.types import JsonObject, MeasurementType


def _inputs() -> InputProvenance:
    return InputProvenance(*(sha256_text(f"input-{index}") for index in range(4)))


def _artifacts() -> ArtifactProvenance:
    return ArtifactProvenance(
        raw_output_sha=sha256_text("raw"),
        metrics_sha="not_applicable",
        log_sha=sha256_text("log"),
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
