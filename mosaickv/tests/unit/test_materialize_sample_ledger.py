from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_materializer() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "materialize_sample_ledger.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_sample_ledger", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MATERIALIZER = _load_materializer()


def _stub_task(monkeypatch: pytest.MonkeyPatch) -> None:
    task = SimpleNamespace(
        development_lmms_task="mmstar",
        scoring_deterministic=True,
        dataset_id="Lin-Chen/MMStar",
        split="val",
    )
    registry = SimpleNamespace(resolve=lambda _name: task)
    monkeypatch.setattr(MATERIALIZER, "default_task_registry", lambda: registry)
    monkeypatch.setattr(
        MATERIALIZER,
        "prepare_seeded_lmms_tasks",
        lambda *_args, **_kwargs: (None, None, ("mmstar:1", "mmstar:2"), None),
    )


def test_ledger_binds_source_at_both_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_task(monkeypatch)
    sha = "a" * 40
    monkeypatch.setattr(MATERIALIZER, "_source_provenance", lambda: (sha, False))
    output = tmp_path / "ledger.json"

    result = MATERIALIZER.materialize(
        task_name="mmstar",
        dataset_revision="b" * 40,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        seed=17,
        output=output,
        resume=False,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "materialized"
    assert payload["source_git_sha"] == sha
    assert payload["source_git_dirty"] is False
    assert payload["materialization_model_name"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert output.stat().st_mode & 0o222 == 0


def test_ledger_rejects_source_change_during_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_task(monkeypatch)
    provenance = iter((("a" * 40, False), ("b" * 40, False)))
    monkeypatch.setattr(MATERIALIZER, "_source_provenance", lambda: next(provenance))
    output = tmp_path / "ledger.json"

    with pytest.raises(RuntimeError, match="changed during"):
        MATERIALIZER.materialize(
            task_name="mmstar",
            dataset_revision="c" * 40,
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            seed=17,
            output=output,
            resume=False,
        )

    assert not output.exists()


def test_ledger_excludes_prior_samples_before_fixed_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_task(monkeypatch)
    sha = "d" * 40
    monkeypatch.setattr(MATERIALIZER, "_source_provenance", lambda: (sha, True))
    exclusion_directory = tmp_path / "prior"
    exclusion_directory.mkdir()
    (exclusion_directory / "old.json").write_text(
        json.dumps(
            {
                "task": "mmstar",
                "dataset_id": "Lin-Chen/MMStar",
                "dataset_revision": "e" * 40,
                "split": "val",
                "sample_ids": ["mmstar:1"],
            }
        )
    )
    output = tmp_path / "fresh.json"

    MATERIALIZER.materialize(
        task_name="mmstar",
        dataset_revision="e" * 40,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        seed=17,
        output=output,
        resume=False,
        subset_size=1,
        exclude_ledger_directories=(exclusion_directory,),
        allow_dirty_exploratory=True,
    )

    payload = json.loads(output.read_text())
    assert payload["sample_ids"] == ["mmstar:2"]
    assert payload["canonical_eligible"] is False
    assert payload["excluded_unique_sample_count"] == 1


def test_ledger_excludes_only_samples_present_in_observed_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_task(monkeypatch)
    sha = "d" * 40
    revision = "e" * 40
    monkeypatch.setattr(MATERIALIZER, "_source_provenance", lambda: (sha, True))
    result_directory = tmp_path / "prior_results"
    run_directory = result_directory / "run-1"
    run_directory.mkdir(parents=True)
    (run_directory / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "id": "Lin-Chen/MMStar",
                    "revision": revision,
                    "split": "val",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "results.jsonl").write_text(
        json.dumps(
            {
                "task": "mmstar",
                "sample_id": "mmstar:1",
                "status": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "fresh.json"

    MATERIALIZER.materialize(
        task_name="mmstar",
        dataset_revision=revision,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        seed=17,
        output=output,
        resume=False,
        subset_size=1,
        exclude_result_directories=(result_directory,),
        allow_dirty_exploratory=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample_ids"] == ["mmstar:2"]
    assert payload["excluded_unique_sample_count"] == 1
    assert payload["exclusion_ledgers"][0]["kind"] == "observed_results"


def test_heldout_ledger_binds_preexisting_selection_freeze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_task(monkeypatch)
    sha = "f" * 40
    monkeypatch.setattr(MATERIALIZER, "_source_provenance", lambda: (sha, False))
    freeze = tmp_path / "frozen.yaml"
    freeze.write_text("selected_candidate: spatial_evidence\n", encoding="utf-8")
    output = tmp_path / "fresh.json"

    MATERIALIZER.materialize(
        task_name="mmstar",
        dataset_revision="e" * 40,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        seed=19,
        output=output,
        resume=False,
        subset_size=1,
        selection_freeze=freeze,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classification"] == "fresh_disjoint_heldout_quality_validation"
    assert payload["selection_freeze"]["path"] == str(freeze.resolve())
    assert payload["selection_freeze_sha256"] == MATERIALIZER.hashlib.sha256(
        freeze.read_bytes()
    ).hexdigest()


def test_calibration_ledger_has_explicit_label_blind_classification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_task(monkeypatch)
    monkeypatch.setattr(
        MATERIALIZER, "_source_provenance", lambda: ("1" * 40, False)
    )
    output = tmp_path / "calibration.json"

    MATERIALIZER.materialize(
        task_name="mmstar",
        dataset_revision="2" * 40,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        seed=23,
        output=output,
        resume=False,
        subset_size=1,
        classification="fresh_disjoint_label_blind_calibration",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classification"] == "fresh_disjoint_label_blind_calibration"
    assert payload["selection_freeze"] is None


def test_fixed_paper_ledger_discloses_its_non_untouched_classification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_task(monkeypatch)
    monkeypatch.setattr(MATERIALIZER, "_source_provenance", lambda: ("3" * 40, False))
    output = tmp_path / "paper.json"

    MATERIALIZER.materialize(
        task_name="mmstar",
        dataset_revision="4" * 40,
        model_name="llava-hf/llava-1.5-7b-hf",
        seed=20270901,
        output=output,
        resume=False,
        subset_size=2,
        classification="fixed_preregistered_paper_quality_memory",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classification"] == "fixed_preregistered_paper_quality_memory"
    assert payload["sample_count"] == 2
