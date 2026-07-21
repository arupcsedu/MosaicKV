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


def test_unsupported_backend_is_rejected_before_expansion() -> None:
    payload = _payload()
    sweeps = payload["sweeps"]
    assert isinstance(sweeps, list)
    assert isinstance(sweeps[0], dict)
    sweeps[0]["backends"] = ["vllm"]
    matrix = ExperimentMatrix.from_mapping(payload)

    with pytest.raises(
        ExperimentMatrixError, match=r"unsupported combination.*model loading failed"
    ):
        expand_experiment_matrix(matrix)


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


def test_prototype_method_is_rejected_until_adapter_gate_passes() -> None:
    payload = _payload()
    sweeps = payload["sweeps"]
    assert isinstance(sweeps, list)
    assert isinstance(sweeps[0], dict)
    sweeps[0]["methods"] = ["mosaickv_proto"]
    matrix = ExperimentMatrix.from_mapping(payload)

    with pytest.raises(ExperimentMatrixError, match="disable prototype merge"):
        expand_experiment_matrix(matrix)


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
