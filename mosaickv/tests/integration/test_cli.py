from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "mosaickv.cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )


@pytest.mark.integration
def test_doctor_runs_without_gpu_or_optional_backends() -> None:
    completed = _run_cli("doctor", "--json")
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] in {"ok", "cpu_only"}
    assert report["backends"]["huggingface"]["model_weights_loaded"] is False
    assert "cuda" in report


@pytest.mark.integration
def test_smoke_cli_passes_without_model_downloads() -> None:
    completed = _run_cli("smoke", "--json", "--seed", "13")
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["synthetic"] is True
    assert report["exact_equivalence"] is True


@pytest.mark.integration
def test_inspect_model_does_not_load_weights() -> None:
    completed = _run_cli("inspect-model", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "--json")
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["loads_model_weights"] is False
    assert report["capabilities"]["sglang_source"] is False


@pytest.mark.integration
@pytest.mark.parametrize("command", ["evaluate", "benchmark"])
def test_measurement_commands_are_truthful_preflights(command: str) -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = _run_cli(
        command, "--config", str(project_root / "configs" / "smoke.toml"), "--json"
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "not_run"
    assert report["config_valid"] is True


@pytest.mark.integration
def test_invalid_cli_configuration_has_useful_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version": 1}', encoding="utf-8")
    completed = _run_cli("evaluate", "--config", str(invalid), "--json")
    assert completed.returncode == 2
    assert "missing required section" in completed.stderr


@pytest.mark.integration
def test_synthetic_evaluation_cli_writes_raw_rows_and_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    manifest = tmp_path / "manifest.json"
    completed = _run_cli(
        "evaluate",
        "--task",
        "synthetic_ci",
        "--run-id",
        "cli-synthetic-ci",
        "--raw-output",
        str(raw),
        "--manifest",
        str(manifest),
        "--subset-size",
        "2",
        "--seed",
        "31",
        "--json",
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["completed_samples"] == 2
    assert report["failed_samples"] == 0
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["manifest_path"] == str(manifest.resolve()) for row in rows)
    written_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert written_manifest["run_id"] == "cli-synthetic-ci"
    assert written_manifest["measurement_type"] == "validation_smoke"
    assert written_manifest["model"]["id"] == "mosaickv/synthetic-color-model"
    assert written_manifest["dataset"]["id"] == "mosaickv/synthetic-ci"
