from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaickv.config import (
    CacheConfig,
    ConfigurationError,
    RunConfig,
    canonical_config_json,
    config_sha256,
    load_config,
    synthetic_smoke_config,
)
from mosaickv.types import BudgetUnit


def test_smoke_config_is_stable_and_valid() -> None:
    config = synthetic_smoke_config(seed=7)
    assert config.execution.seed == 7
    assert len(config_sha256(config)) == 64
    assert canonical_config_json(config) == canonical_config_json(config)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_invalid_retention_ratio_has_useful_path(ratio: float) -> None:
    with pytest.raises(ConfigurationError, match=r"cache\.retention_ratio"):
        CacheConfig(
            budget_value=8,
            budget_unit=BudgetUnit.RETAINED_SLOTS,
            retention_ratio=ratio,
        )


def test_unknown_configuration_field_is_rejected() -> None:
    payload = json.loads(canonical_config_json(synthetic_smoke_config()))
    payload["model"]["typo_revision"] = "bad"
    with pytest.raises(ConfigurationError, match=r"model contains unknown field.*typo_revision"):
        RunConfig.from_mapping(payload)


def test_invalid_enum_lists_allowed_values() -> None:
    payload = json.loads(canonical_config_json(synthetic_smoke_config()))
    payload["execution"]["backend"] = "unknown-backend"
    with pytest.raises(ConfigurationError, match=r"execution\.backend must be one of"):
        RunConfig.from_mapping(payload)


def test_load_toml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "smoke.toml")
    assert config.cache.retention_ratio == 1.0
    assert config.model.id == "synthetic/smoke"


def test_unsupported_config_format_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"use \.json or \.toml"):
        load_config(path)
