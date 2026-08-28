from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import yaml


def _materializer() -> ModuleType:
    package = Path(__file__).parents[2]
    path = package / "scripts/prepare_paper_candidate_v1_retrospective_preflight.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_retrospective", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_ledger_collection_is_task_scoped_and_ignores_results(tmp_path: Path) -> None:
    materializer = _materializer()
    repository = tmp_path / "repository"
    run_root = tmp_path / "runs"
    ignored = run_root / "current"
    config_root = repository / "mosaickv/configs/ledgers"
    input_root = run_root / "prior/inputs"
    current_root = ignored / "inputs"
    for directory in (config_root, input_root, current_root):
        directory.mkdir(parents=True)
    (config_root / "mmstar.json").write_text(
        json.dumps({"task": "mmstar", "sample_ids": ["mmstar:1", "mmstar:2"]})
    )
    (input_root / "textvqa.yaml").write_text(
        yaml.safe_dump({"task": "textvqa", "sample_ids": ["textvqa:4"]})
    )
    (input_root / "results.jsonl").write_text(
        json.dumps({"task": "docvqa", "sample_id": "docvqa:not_a_ledger"}) + "\n"
    )
    (current_root / "docvqa.json").write_text(
        json.dumps({"task": "docvqa", "sample_ids": ["docvqa:ignored"]})
    )
    identities, sources = materializer.collect_explicit_ledger_identities(
        repository=repository,
        run_root=run_root,
        ignored_root=ignored,
    )
    assert identities == {
        "mmstar": {"mmstar:1", "mmstar:2"},
        "textvqa": {"textvqa:4"},
        "docvqa": set(),
    }
    assert len(sources) == 2


def test_retrospective_protocol_is_nonheldout_and_cannot_authorize_paper() -> None:
    materializer = _materializer()
    package = Path(__file__).parents[2]
    protocol = yaml.safe_load(
        (
            package
            / "configs/experiments/paper_quality_memory_v1_retrospective_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    assert protocol["classification"] == materializer.CLASSIFICATION
    assert protocol["sample_boundary"]["held_out"] is False
    assert protocol["sample_boundary"]["fresh"] is False
    assert protocol["sample_boundary"]["paper_claim_eligible"] is False
    assert protocol["execution_authorization"]["retrospective_gpu_preflight_authorized"]
    assert protocol["execution_authorization"]["paper_experiment_authorized"] is False
    assert protocol["execution_authorization"]["automatic_follow_on_authorized"] is False
