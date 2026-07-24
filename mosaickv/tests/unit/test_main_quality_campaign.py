from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mosaickv.experiment_matrix import TASK_CATALOG, verify_expanded_index


def _load_preparer() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "prepare_main_quality_campaign.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_main_campaign", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARER = _load_preparer()


def test_campaign_materializes_immutable_paired_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = TASK_CATALOG["mmstar"]
    ledger = tmp_path / "mmstar.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": task.task_name,
                "dataset_id": task.dataset_id,
                "dataset_revision": task.revision,
                "split": task.split,
                "selection_seed": 17,
                "source_git_sha": "a" * 40,
                "source_git_dirty": False,
                "sample_count": 3,
                "sample_ids": ["mmstar:1", "mmstar:2", "mmstar:3"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(ledger, 0o444)
    monkeypatch.setattr(PREPARER, "_clean_source_sha", lambda: "a" * 40)
    matrix = Path(__file__).parents[2] / "configs" / "experiments" / "main_quality.yaml"
    destination = tmp_path / "campaign"
    output_root = tmp_path / "runs"

    index = PREPARER.materialize(
        matrix_path=matrix,
        ledger_paths={"mmstar": ledger},
        destination=destination,
        shard_size=1,
        diagnostic_size=1,
        resume=False,
        sample_limit=2,
        model_keys=frozenset({"qwen2_5_vl_3b"}),
        task_keys=frozenset({"mmstar"}),
        methods=frozenset({"full_kv"}),
        retention_ratios=frozenset({1.0}),
        output_root=output_root,
    )

    assert verify_expanded_index(index) == 2
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    assert {row["shard_index"] for row in rows} == {0, 1}
    assert len({row["run_id"] for row in rows}) == 2
    assert all(Path(row["sample_id_file"]).stat().st_mode & 0o222 == 0 for row in rows)
    assert all(Path(row["config_path"]).stat().st_mode & 0o222 == 0 for row in rows)
    assert {row["output_root"] for row in rows} == {str(output_root.resolve())}
    manifest = json.loads((destination / "matrix_manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_root"] == str(output_root.resolve())
