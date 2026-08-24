from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mosaickv.config import synthetic_smoke_config
from mosaickv.hf_cli import _load_sample_id_file, _model_kwargs


def test_common_transformers_lock_uses_torch_dtype_keyword() -> None:
    kwargs = _model_kwargs(synthetic_smoke_config(), local_files_only=True)

    assert "torch_dtype" in kwargs
    assert "dtype" not in kwargs
    assert kwargs["local_files_only"] is True


def test_sample_id_file_requires_exact_content_and_sha(tmp_path: Path) -> None:
    config = synthetic_smoke_config()
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "task": "synthetic_ci",
                "dataset_id": config.dataset.id,
                "dataset_revision": config.dataset.revision,
                "split": config.dataset.split,
                "sample_ids": ["synthetic:1", "synthetic:2"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    path = tmp_path / "shard.json"
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    assert _load_sample_id_file(str(path), sha, task_name="synthetic_ci", config=config) == (
        "synthetic:1",
        "synthetic:2",
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        _load_sample_id_file(str(path), "0" * 64, task_name="synthetic_ci", config=config)


def test_sample_id_file_accepts_reviewed_schema_version_two_ledger(tmp_path: Path) -> None:
    config = synthetic_smoke_config()
    record = {
        "schema_version": 2,
        "classification": "fresh_disjoint_label_blind_calibration",
        "task": "synthetic_ci",
        "dataset_id": config.dataset.id,
        "dataset_revision": config.dataset.revision,
        "split": config.dataset.split,
        "sample_count": 2,
        "sample_ids": ["synthetic:1", "synthetic:2"],
    }
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = tmp_path / "reviewed-ledger.json"
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    assert _load_sample_id_file(str(path), sha, task_name="synthetic_ci", config=config) == (
        "synthetic:1",
        "synthetic:2",
    )

    record["sample_count"] = 1
    invalid = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(invalid)
    with pytest.raises(ValueError, match="inconsistent sample_count"):
        _load_sample_id_file(
            str(path),
            hashlib.sha256(invalid).hexdigest(),
            task_name="synthetic_ci",
            config=config,
        )
