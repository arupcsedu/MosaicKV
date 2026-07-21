from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaickv.config import (
    CacheConfig,
    ConfigurationError,
    ForecastingConfig,
    LookMConfig,
    PrefixKVConfig,
    PrototypeConfig,
    ResidualConfig,
    RunConfig,
    VLCacheConfig,
    canonical_config_json,
    config_sha256,
    load_config,
    synthetic_smoke_config,
)
from mosaickv.types import (
    BudgetUnit,
    ForecastMode,
    LookMMergeStrategy,
    MosaicKVMethod,
    PrefixKVProfileMode,
    ResidualStorageDType,
)


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
    assert config.forecasting.mode is ForecastMode.HYBRID
    assert config.forecasting.prompt_window == 16


def test_load_hf_yaml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_mosaickv.yaml")
    assert config.method is MosaicKVMethod.MOSAICKV_FULL
    assert config.model.id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert len(config.model.revision) == 40
    assert config.execution.attention_implementation == "eager"


def test_load_simple_baseline_yaml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_simple_baseline.yaml")
    assert config.method is MosaicKVMethod.RANDOM_KV
    assert config.cache.retention_ratio == 0.5
    assert not config.forecasting.enabled
    assert not config.graph.enabled
    assert not config.selection.enabled
    assert not config.prototypes.enabled
    assert not config.residual.enabled
    assert not config.repair.enabled


def test_load_prefixkv_yaml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_prefixkv_reimpl.yaml")
    assert config.method is MosaicKVMethod.PREFIXKV_REIMPL
    assert config.cache.block_size == 1
    assert config.prefixkv.enabled
    assert config.prefixkv.profile_mode is PrefixKVProfileMode.FIXED_GLOBAL


def test_load_vl_cache_yaml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_vl_cache_reimpl.yaml")
    assert config.method is MosaicKVMethod.VL_CACHE_REIMPL
    assert config.cache.block_size == 1
    assert config.vl_cache.enabled
    assert config.vl_cache.sparsity_threshold == pytest.approx(0.01)


def test_vl_cache_calibration_provenance_is_strict() -> None:
    with pytest.raises(ConfigurationError, match="require dataset ID, revision, and split"):
        VLCacheConfig(enabled=True, calibration_sample_ids=("cal-0",))
    with pytest.raises(ConfigurationError, match="cannot contain duplicates"):
        VLCacheConfig(
            enabled=True,
            calibration_dataset_id="dataset",
            calibration_dataset_revision="revision",
            calibration_split="train",
            calibration_sample_ids=("cal-0", "cal-0"),
        )


def test_simple_baseline_rejects_mosaickv_stages() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_simple_baseline.yaml")
    payload = json.loads(canonical_config_json(config))
    payload["prototypes"]["enabled"] = True
    with pytest.raises(
        ConfigurationError,
        match=r"method='random_kv' requires disabled MosaicKV stages: prototypes",
    ):
        RunConfig.from_mapping(payload)


def test_full_kv_requires_retention_one() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_simple_baseline.yaml")
    payload = json.loads(canonical_config_json(config))
    payload["method"] = "full_kv"
    with pytest.raises(ConfigurationError, match=r"full_kv.*retention_ratio=1\.0"):
        RunConfig.from_mapping(payload)


def test_lookm_configuration_is_strict_and_explicit() -> None:
    config = LookMConfig(
        enabled=True,
        recent_ratio=0.1,
        important_ratio=0.1,
        merge_strategy=LookMMergeStrategy.PIVOTAL,
    )
    assert config.recent_ratio + config.important_ratio == pytest.approx(0.2)
    with pytest.raises(ConfigurationError, match="cannot exceed 1"):
        LookMConfig(enabled=True, recent_ratio=0.6, important_ratio=0.5)


def test_prefixkv_offline_profile_is_explicit_and_fixed_global_needs_no_file() -> None:
    with pytest.raises(ConfigurationError, match=r"prefixkv\.profile_path is required"):
        PrefixKVConfig(enabled=True)
    config = PrefixKVConfig(
        enabled=True,
        profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
    )
    assert config.profile_path is None


def test_forecasting_configuration_rejects_ambiguous_unused_sources() -> None:
    with pytest.raises(ConfigurationError, match="prompt_window mode"):
        ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=1,
        )


def test_prototype_configuration_rejects_invalid_modality_pairs() -> None:
    with pytest.raises(ConfigurationError, match=r"prototypes\.allowed_modality_pairs"):
        PrototypeConfig(allowed_modality_pairs=("image:audio",))


def test_residual_configuration_parses_storage_dtype_strictly() -> None:
    config = ResidualConfig.from_mapping(
        {
            "enabled": True,
            "rank": 8,
            "storage_dtype": "int8",
            "require_pinned_memory": False,
        }
    )
    assert config.storage_dtype is ResidualStorageDType.INT8
    with pytest.raises(ConfigurationError, match=r"residual\.storage_dtype must be one of"):
        ResidualConfig.from_mapping({"storage_dtype": "int4"})


def test_unsupported_config_format_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.ini"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"use \.json, \.toml, \.yaml, or \.yml"):
        load_config(path)
