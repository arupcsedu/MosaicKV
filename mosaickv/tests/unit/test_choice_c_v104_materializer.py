from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from mosaickv.evaluation.disjoint_samples import (
    canonical_bytes,
    validate_exclusion_manifest,
    verify_exclusion_source_files,
)


def _materializer(package: Path) -> ModuleType:
    scripts = package / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "prepare_choice_c_v104_preflight.py"
        spec = importlib.util.spec_from_file_location("mosaickv_test_v104_materializer", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _with_content_sha(payload: dict[str, object], field: str) -> dict[str, object]:
    payload[field] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def test_v104_augments_prior_exclusions_with_all_planned_v103_ids(tmp_path: Path) -> None:
    package = Path(__file__).parents[2]
    materializer = _materializer(package)
    bound_source = tmp_path / "bound.json"
    bound_source.write_text("{}\n", encoding="utf-8")
    identity = {
        "dataset_id": "dataset",
        "dataset_config": "default",
        "dataset_revision": "revision",
        "split": "train",
        "sample_id": "task:prior",
    }
    prior = _with_content_sha(
        {
            "schema_version": 2,
            "classification": "label_blind_sample_exclusion_manifest",
            "source_git_sha": "a" * 40,
            "source_git_dirty": False,
            "scan_roots": [str(tmp_path)],
            "dataset_contracts": [],
            "source_count": 1,
            "sources": [
                {
                    "kind": "sample_ledger",
                    "path": str(bound_source),
                    "sha256": hashlib.sha256(bound_source.read_bytes()).hexdigest(),
                }
            ],
            "excluded_unique_sample_count": 1,
            "excluded_identities": [identity],
        },
        "manifest_content_sha256",
    )
    prior_path = tmp_path / "prior.json"
    prior_path.write_bytes(canonical_bytes(prior))
    ledger = _with_content_sha(
        {
            "schema_version": 2,
            "classification": "fresh_disjoint_label_blind_calibration",
            "task": "task",
            "dataset_id": "dataset",
            "dataset_config": "default",
            "dataset_revision": "revision",
            "split": "train",
            "sample_count": 1,
            "sample_ids": ["task:v103"],
        },
        "ledger_content_sha256",
    )
    ledger_path = tmp_path / "v103_ledger.json"
    ledger_path.write_bytes(canonical_bytes(ledger))

    result = materializer.build_augmented_exclusion_manifest(
        prior_manifest=prior,
        prior_manifest_path=prior_path,
        additional_ledger_paths=[ledger_path],
        source_git_sha="b" * 40,
    )

    identities = validate_exclusion_manifest(result)
    verify_exclusion_source_files(result)
    assert [item.sample_id for item in identities] == ["task:prior", "task:v103"]
    assert result["excluded_unique_sample_count"] == 2
    assert result["source_count"] == 2


def test_v104_slurm_launcher_is_exactly_two_one_a100_workers() -> None:
    package = Path(__file__).parents[2]
    launcher = (package / "slurm/choice_c_v104_preflight_a100.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --array=0-1%2" in launcher
    assert "#SBATCH --gres=gpu:a100:1" in launcher
    assert "choice_c_v104_joint_reallocation_preflight" in launcher
    assert "--validate-only" in launcher
