"""Ablation controls and CPU-only configuration generation (no model/dataset access)."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_prefixkv_evidence import captures, make_profile

from mosaickv.config import (
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    GraphConfig,
    ModelConfig,
    PrefixKVConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    canonical_config,
)
from mosaickv.evaluation.prefixkv_evidence_ablation import (
    EVIDENCE_ABLATIONS,
    evidence_ablation_configs,
)
from mosaickv.types import Backend, BudgetUnit, MosaicKVMethod, Precision, RepairPolicy


def controls():
    topk = RunConfig(
        model=ModelConfig("llava-hf/llava-1.5-7b-hf", "b" * 40, Precision.FP32),
        dataset=DatasetConfig("synthetic", "c" * 40, "test"),
        execution=ExecutionConfig(Backend.HUGGINGFACE, "eager", 17, True),
        generation=GenerationConfig(max_new_tokens=64),
        cache=CacheConfig(
            768, BudgetUnit.BYTES, 0.99, 1, physical_budget_mandatory_policy="first_and_last"
        ),
        method=MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        forecasting=ForecastingConfig(enabled=False),
        graph=GraphConfig(enabled=False),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE, max_blocks_per_step=0),
    )
    full = replace(
        topk, method=MosaicKVMethod.FULL_KV, cache=replace(topk.cache, retention_ratio=1.0)
    )
    mosaic = replace(
        topk,
        method=MosaicKVMethod.MOSAICKV_EXACT,
        forecasting=replace(topk.forecasting, enabled=True),
        graph=GraphConfig(),
        selection=SelectionConfig(),
    )
    return {c.method.value: c for c in (full, topk, mosaic)}


def profiles(tmp_path):
    return {
        a.name: make_profile(a.sink_neutral, a.question_protected, a.question_weight).write(
            tmp_path / (a.name + ".json")
        )
        for a in EVIDENCE_ABLATIONS
    }


def test_factorial_configs_match_controls_and_roundtrip(tmp_path):
    groups = evidence_ablation_configs(controls(), profiles(tmp_path))
    assert {p: len(arms) for p, arms in groups.items()} == {
        "first_and_last": 7,
        "question_and_boundaries": 7,
    }
    for policy, configs in groups.items():
        for cfg in configs.values():
            assert cfg.cache.physical_budget_mandatory_policy == policy
            assert cfg.cache.budget_value == 768
            assert cfg.execution == controls()["full_kv"].execution
            assert RunConfig.from_mapping(canonical_config(cfg)) == cfg
    mismatched = controls()
    c = mismatched["mosaickv_exact"]
    mismatched["mosaickv_exact"] = replace(c, generation=replace(c.generation, max_new_tokens=7))
    with pytest.raises(ValueError, match="generation differs"):
        evidence_ablation_configs(mismatched, profiles(tmp_path / "second"))


def test_reject_mismatched_calibration_and_backend(tmp_path):
    paths = profiles(tmp_path)
    p = make_profile(True)
    changed = replace(p, calibration_sample_ids=("other-cal",))
    paths["a_sink"] = changed.write(tmp_path / "changed.json")
    with pytest.raises(ValueError, match="different calibration"):
        evidence_ablation_configs(controls(), paths)
    topk = controls()["prompt_attention_topk"]
    with pytest.raises(ValueError, match="HF eager"):
        replace(
            topk,
            execution=replace(topk.execution, attention_implementation="sdpa"),
            cache=replace(topk.cache, physical_budget_mandatory_policy="question_and_boundaries"),
        )


def test_bundle_cli_is_write_once_and_never_authorizes_execution(tmp_path):
    obs = tmp_path / "scores.jsonl"
    obs.write_text("\n".join(json.dumps(asdict(o)) for o in captures()[0]))
    ctrl = tmp_path / "controls.json"
    ctrl.write_text(json.dumps({name: canonical_config(cfg) for name, cfg in controls().items()}))
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["eval-1"]))
    output = tmp_path / "bundle"
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_prefixkv_evidence_ablations.py"
    cmd = [
        sys.executable,
        str(script),
        "--attention-jsonl",
        str(obs),
        "--controls-json",
        str(ctrl),
        "--evaluation-sample-ids",
        str(ids),
        "--bytes-per-layer-position",
        "16",
        "--calibration-dataset",
        "synthetic",
        "--calibration-revision",
        "c" * 40,
        "--calibration-split",
        "train",
        "--output",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["configs"] == 14
    original = (output / "bundle.json").read_bytes()
    manifest = json.loads(original)
    assert not manifest["gpu_execution_authorized"] and not manifest["paper_claim_authorized"]
    assert len(manifest["profiles"]) == 8
    assert len(manifest["stages"]) == 3
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert (output / "bundle.json").read_bytes() == original


def test_legacy_defaults_remain_disabled():
    assert not PrefixKVConfig().exclude_protected_from_calibration
    assert PrefixKVConfig().question_score_weight == 0
    assert CacheConfig(1024, BudgetUnit.BYTES, 0.99).physical_budget_mandatory_policy == "method"
