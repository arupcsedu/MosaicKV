from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _script():
    path = Path(__file__).resolve().parents[2] / "scripts/materialize_milebench_representative.py"
    spec = importlib.util.spec_from_file_location("milebench_materializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(root: Path, count: int = 5) -> None:
    module = _script()
    for task, question_type in module.TASKS.items():
        directory = root / task
        media = directory / "combined_1_images"
        media.mkdir(parents=True)
        rows = []
        for index in range(count):
            filename = f"{index}.jpg"
            (media / filename).write_bytes(f"{task}:{index}".encode())
            instance = {
                "context": "{image#1} What changed?",
                "combined_1_images": [filename],
            }
            if question_type == "multi-choice":
                instance["choice_list"] = ["yes", "no"]
            rows.append(
                {
                    "sample_id": index,
                    "task_instruction_id": 0,
                    "task_instance": instance,
                    "response": "yes",
                }
            )
        (directory / f"{task}.json").write_text(
            json.dumps(
                {
                    "meta_data": {
                        "question_type": question_type,
                        "task_instruction": ["Answer the question."],
                    },
                    "data": rows,
                }
            )
        )


def test_materializer_balances_routes_binds_media_and_is_deterministic(tmp_path, monkeypatch):
    module = _script()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(module, "_under_scratch", lambda path, label: path.resolve())
    dataset = scratch / "dataset"
    _fixture(dataset)
    exclusion = frozenset({"ActionSequence-0000"})
    first = module.materialize(dataset, scratch / "first", count_per_task=2, excluded_ids=exclusion)
    second = module.materialize(
        dataset, scratch / "second", count_per_task=2, excluded_ids=exclusion
    )
    assert first["dataset_revision"] == module.SOURCE_REVISION
    assert first["materialization_content_sha256"] == second["materialization_content_sha256"]
    assert first["total_samples"] == 8
    assert first["official_lookm_execution"] is False
    for _task, path in first["ledgers"].items():
        payload = json.loads(Path(path).read_text())
        assert len(payload["samples"]) == 2
        assert payload["dataset_revision"] == module.SOURCE_REVISION
        assert all(
            sample["media_representation"] == "official_combined_1_images"
            for sample in payload["samples"]
        )
        assert all(Path(sample["media"][0]["path"]).is_file() for sample in payload["samples"])
        assert all("{image#" not in sample["prompt"] for sample in payload["samples"])
        assert "ActionSequence-0000" not in {sample["sample_id"] for sample in payload["samples"]}


def test_materializer_rejects_ragged_combined_media(tmp_path, monkeypatch):
    module = _script()
    monkeypatch.setattr(module, "_under_scratch", lambda path, label: path.resolve())
    dataset = tmp_path / "dataset"
    _fixture(dataset)
    annotation = dataset / "CLEVR-Change/CLEVR-Change.json"
    payload = json.loads(annotation.read_text())
    payload["data"][0]["task_instance"]["combined_1_images"] = ["0.jpg", "1.jpg"]
    annotation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly one"):
        module.materialize(dataset, tmp_path / "output", count_per_task=5)
