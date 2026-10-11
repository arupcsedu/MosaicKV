from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _module() -> object:
    path = Path(__file__).parents[2] / "scripts" / "validate_shared_byte_curve.py"
    spec = importlib.util.spec_from_file_location("shared_byte_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_valid_artifact(root: Path) -> None:
    methods = ("full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
    raw_rows = []
    for method in methods:
        raw_rows.append(
            {
                "task": "docvqa",
                "sample_id": "item-1",
                "method": method,
                "configured_byte_cap": 10,
                "active_kv_bytes": 100 if method == "full_kv" else 10,
                "status": "completed",
                "error": None,
                "source_git_sha": "abc",
                "model_revision": "def",
                "result_category": "unified_comparison",
                "result_origin": "locally_measured",
            }
        )
    pd.DataFrame(raw_rows).to_parquet(root / "raw_results.parquet")
    pd.DataFrame(
        [
            {
                "task": "docvqa",
                "sample_id": "item-1",
                "configured_byte_cap": 10,
                "prompt_attention_topk_bytes": 10,
                "vl_cache_reimpl_bytes": 10,
                "mosaickv_exact_bytes": 10,
                "exact_byte_match": True,
                "within_hard_cap": True,
            }
        ]
    ).to_parquet(root / "byte_validation.parquet")
    pd.DataFrame(
        [{"task": "docvqa", "metric_name": "anls", "method": "mosaickv_exact", "sample_count": 1}]
    ).to_parquet(root / "summary.parquet")
    pd.DataFrame(
        [
            {
                "task": "docvqa",
                "metric_name": "anls",
                "reference_method": "prompt_attention_topk",
                "candidate_method": "mosaickv_exact",
                "sample_count": 1,
                "bootstrap_resamples": 10,
            }
        ]
    ).to_parquet(root / "paired_comparisons.parquet")


def test_shared_byte_validator_accepts_exact_matched_rows(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path)
    result = _module().validate_artifact(  # type: ignore[attr-defined]
        tmp_path, expected_source_sha="abc", expected_model_revision="def"
    )
    assert result["status"] == "passed"
    assert result["byte_validated_samples"] == 1


def test_shared_byte_validator_rejects_nonmatching_physical_bytes(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path)
    checks = pd.read_parquet(tmp_path / "byte_validation.parquet")
    checks.loc[0, "exact_byte_match"] = False
    checks.to_parquet(tmp_path / "byte_validation.parquet")
    with pytest.raises(ValueError, match="exact physical-byte"):
        _module().validate_artifact(tmp_path)  # type: ignore[attr-defined]
