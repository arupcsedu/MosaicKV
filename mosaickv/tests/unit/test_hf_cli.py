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
