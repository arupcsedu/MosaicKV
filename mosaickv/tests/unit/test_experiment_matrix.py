from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml

from mosaickv.experiment_matrix import (
    REQUIRED_BACKENDS,
    REQUIRED_METHODS,
    REQUIRED_MODELS,
    REQUIRED_OUTPUT_LENGTHS,
    REQUIRED_RETENTION_RATIOS,
    REQUIRED_TASKS,
    ExperimentMatrix,
    ExperimentMatrixError,
    expand_experiment_matrix,
    load_experiment_matrix,
    materialize_experiment_matrix,
    verify_expanded_index,
)

_MATRIX_NAMES = (
    "pilot.yaml",
    "main_quality.yaml",
    "main_performance.yaml",
    "ablations.yaml",
    "repair.yaml",
    "backend_generalization.yaml",
)


def _matrix_root() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "experiments"


def _payload(name: str = "pilot.yaml") -> dict[str, object]:
    payload = yaml.safe_load((_matrix_root() / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize("name", _MATRIX_NAMES)
def test_versioned_matrices_cover_required_vocabularies(name: str) -> None:
    matrix = load_experiment_matrix(_matrix_root() / name)

    assert set(matrix.axes.models) == REQUIRED_MODELS
    assert set(matrix.axes.backends) == REQUIRED_BACKENDS
    assert set(matrix.axes.methods) == REQUIRED_METHODS
    assert set(matrix.axes.tasks) == REQUIRED_TASKS
    assert set(matrix.axes.retention_ratios) == REQUIRED_RETENTION_RATIOS
    assert set(matrix.axes.output_lengths) == REQUIRED_OUTPUT_LENGTHS
    assert len(matrix.axes.selection_seeds) == 3
    assert matrix.performance.warmups >= 5
    assert matrix.performance.timed_repetitions >= 20
    assert matrix.run.resume
    assert matrix.comparison.primary_budget == "active_kv_bytes"
    expand_experiment_matrix(matrix)


def test_vllm_fullkv_is_admitted_for_checkpoint_gated_qwen() -> None:
    payload = _payload()
    sweeps = payload["sweeps"]
    assert isinstance(sweeps, list)
    assert isinstance(sweeps[0], dict)
    sweeps[0]["backends"] = ["vllm"]
    del sweeps[1:]
    matrix = ExperimentMatrix.from_mapping(payload)

    runs = expand_experiment_matrix(matrix)

    assert runs
    assert {run.config.execution.backend.value for run in runs} == {"vllm"}


@pytest.mark.parametrize("backend", ("hf_sdpa", "hf_flash_attention_2"))
def test_fused_hf_backends_admit_fullkv_only(backend: str) -> None:
    payload = _payload()
    sweeps = payload["sweeps"]
    assert isinstance(sweeps, list)
    assert isinstance(sweeps[0], dict)
    sweeps[0]["backends"] = [backend]
    sweeps[0]["methods"] = ["full_kv"]
    sweeps[0]["retention_ratios"] = [1.0]
    del sweeps[1:]
    matrix = ExperimentMatrix.from_mapping(payload)

    runs = expand_experiment_matrix(matrix)

    assert runs
    assert {run.config.method.value for run in runs} == {"full_kv"}
    assert {run.config.execution.attention_implementation for run in runs} == {
        "sdpa" if backend == "hf_sdpa" else "flash_attention_2"
    }


def test_video_task_is_rejected_for_image_only_model() -> None:
    payload = _payload()
    sweeps = payload["sweeps"]
    assert isinstance(sweeps, list)
    assert isinstance(sweeps[0], dict)
    sweeps[0]["models"] = ["llava_1_5_7b"]
    sweeps[0]["tasks"] = ["videomme"]
    matrix = ExperimentMatrix.from_mapping(payload)

    with pytest.raises(ExperimentMatrixError, match="does not support video input"):
        expand_experiment_matrix(matrix)


def test_prototype_method_is_admitted_after_adapter_phase_transform_gate() -> None:
    payload = _payload()
    sweeps = payload["sweeps"]
    assert isinstance(sweeps, list)
    assert isinstance(sweeps[0], dict)
    sweeps[0]["methods"] = ["mosaickv_proto"]
    del sweeps[1:]
    matrix = ExperimentMatrix.from_mapping(payload)

    runs = expand_experiment_matrix(matrix)

    assert runs
    assert {run.config.method.value for run in runs} == {"mosaickv_proto"}


@pytest.mark.parametrize(
    ("method", "message"),
    (
        ("prefixkv_reimpl", "offline layer profile"),
        ("vl_cache_reimpl", "disjoint calibration"),
    ),
)
def test_calibrated_baselines_are_rejected_without_versioned_artifacts(
    method: str, message: str
) -> None:
    payload = _payload()
    sweeps = payload["sweeps"]
    assert isinstance(sweeps, list)
    assert isinstance(sweeps[0], dict)
    sweeps[0]["models"] = ["llava_1_5_7b"]
    sweeps[0]["methods"] = [method]
    matrix = ExperimentMatrix.from_mapping(payload)

    with pytest.raises(ExperimentMatrixError, match=message):
        expand_experiment_matrix(matrix)


def test_performance_minimum_and_seed_cardinality_are_strict() -> None:
    payload = _payload()
    performance = payload["performance"]
    assert isinstance(performance, dict)
    performance["warmups"] = 4
    with pytest.raises(ExperimentMatrixError, match=r"performance\.warmups"):
        ExperimentMatrix.from_mapping(payload)

    payload = _payload()
    axes = payload["axes"]
    assert isinstance(axes, dict)
    axes["selection_seeds"] = [17, 29]
    with pytest.raises(ExperimentMatrixError, match="exactly three"):
        ExperimentMatrix.from_mapping(payload)


def test_materialization_is_immutable_unique_and_resumable(tmp_path: Path) -> None:
    destination = tmp_path / "expanded"
    index_path = materialize_experiment_matrix(
        _matrix_root() / "pilot.yaml", output_directory=destination
    )

    assert verify_expanded_index(index_path) == 60
    assert verify_expanded_index(index_path, array_index=17) == 60
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert len({row["run_id"] for row in rows}) == len(rows)
    assert len({row["config_path"] for row in rows}) == len(rows)
    for row in rows:
        mode = stat.S_IMODE(Path(row["config_path"]).stat().st_mode)
        assert mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
        assert row["warmups"] == 5
        assert row["timed_repetitions"] == 20
    with pytest.raises(FileExistsError, match="pass --resume"):
        materialize_experiment_matrix(_matrix_root() / "pilot.yaml", output_directory=destination)
    resumed = materialize_experiment_matrix(
        _matrix_root() / "pilot.yaml", output_directory=destination, resume=True
    )
    assert resumed == index_path


def test_disabled_matrices_expand_to_no_jobs() -> None:
    for name in ("main_performance.yaml", "repair.yaml"):
        matrix = load_experiment_matrix(_matrix_root() / name)
        assert not matrix.enabled
        assert expand_experiment_matrix(matrix) == ()


def test_main_quality_uses_only_supported_deterministic_strata() -> None:
    matrix = load_experiment_matrix(_matrix_root() / "main_quality.yaml")
    runs = expand_experiment_matrix(matrix)

    assert len(runs) == 532
    assert {run.config.generation.max_new_tokens for run in runs} == {128}
    assert "mmvet" not in {run.task_key for run in runs}
    assert {run.config.method.value for run in runs} == {
        "full_kv",
        "random_kv",
        "prompt_attention_topk",
        "lookm_reimpl",
        "mosaickv_exact",
    }
    assert {run.selection_seed for run in runs if run.config.method.value == "random_kv"} == {
        17,
        29,
        43,
    }
    assert {run.selection_seed for run in runs if run.config.method.value != "random_kv"} == {17}
    for run in runs:
        expected_block_size = 1 if run.config.method.value == "lookm_reimpl" else 16
        assert run.config.cache.block_size == expected_block_size
        if run.config.method.value == "mosaickv_exact":
            assert run.config.selection.stop_on_nonpositive_gain is False
