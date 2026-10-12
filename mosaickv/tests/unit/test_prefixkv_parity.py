from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from mosaickv.baselines import (
    PrefixKVParityArtifact,
    PrefixKVParityControls,
    PrefixKVSampleObservation,
    compare_prefixkv_artifacts,
)
from mosaickv.types import JsonObject


def _runner_module() -> object:
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_prefixkv_llava_parity.py"
    spec = importlib.util.spec_from_file_location("prefixkv_official_runner", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"cannot load PrefixKV runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_profile_aggregator_module() -> object:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "aggregate_prefixkv_official_profile.py"
    )
    spec = importlib.util.spec_from_file_location("prefixkv_official_aggregator", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"cannot load PrefixKV aggregator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controls() -> PrefixKVParityControls:
    return PrefixKVParityControls(
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
        tokenizer_revision="a" * 40,
        dataset_id="small-llava-parity",
        dataset_revision="b" * 40,
        calibration_sample_set_sha256="1" * 64,
        evaluation_sample_set_sha256="2" * 64,
        prompt_payload_sha256="3" * 64,
        media_payload_sha256="4" * 64,
        profile_sha256="5" * 64,
        environment_sha256="6" * 64,
        hardware_sha256="7" * 64,
        measurement_protocol_sha256="8" * 64,
        cache_budget_value=10_000,
        cache_budget_unit="blocks",
        block_size=1,
        retention_ratio=0.5,
        official_forget_ratio=0.5,
        generation_parameters={"do_sample": False, "temperature": 0.0},
        output_length_policy="fixed_max_new_tokens",
        model_precision="bf16",
        backend="huggingface",
        backend_configuration={"cache_type": "legacy"},
        attention_implementation="eager",
        seed=0,
    )


def _artifact(implementation: str) -> PrefixKVParityArtifact:
    return PrefixKVParityArtifact(
        implementation=implementation,
        official_repository_sha="597f1ab032704951550f93bcc8a23f1454b80aa4",
        executable_git_sha="a" * 40,
        config_sha256="b" * 64,
        manifest_path="manifest.json",
        measurement_type=(
            "baseline_official_measured"
            if implementation == "official_prefixkv"
            else "baseline_reimpl_measured"
        ),
        controls=_controls(),
        samples=(
            PrefixKVSampleObservation(
                sample_id="sample-0",
                per_layer_cache_sizes=(20, 12, 20),
                total_retained_bytes=4096,
                actual_active_kv_bytes=4096,
                generated_answer="answer",
                generated_token_ids=(1, 2, 3),
                latency_seconds=0.25,
                perplexity=3.5,
                rouge_l_f1=0.75,
            ),
        ),
    )


def test_parity_report_compares_all_requested_observations() -> None:
    report = compare_prefixkv_artifacts(
        _artifact("official_prefixkv"), _artifact("prefixkv_reimpl")
    )
    assert report["status"] == "comparable"
    rows = cast("list[JsonObject]", report["samples"])
    row = rows[0]
    assert row["per_layer_cache_sizes_match"] is True
    assert row["retained_byte_delta"] == 0
    assert row["actual_active_kv_byte_delta"] == 0
    assert row["perplexity_delta"] == 0
    assert row["rouge_l_f1_delta"] == 0
    assert row["answers_exact_match"] is True
    assert row["token_agreement"] == 1.0


def test_parity_report_refuses_different_controls() -> None:
    official = _artifact("official_prefixkv")
    reimplementation = replace(
        _artifact("prefixkv_reimpl"),
        controls=replace(_controls(), seed=1),
    )
    report = compare_prefixkv_artifacts(official, reimplementation)
    assert report["status"] == "not_comparable"
    assert report["control_mismatches"] == ["seed"]


def test_parity_report_aggregates_nll_before_exponentiating() -> None:
    first = PrefixKVSampleObservation(
        sample_id="sample-0",
        per_layer_cache_sizes=(20,),
        total_retained_bytes=4096,
        actual_active_kv_bytes=4096,
        generated_answer="first",
        generated_token_ids=(1,),
        latency_seconds=0.1,
        perplexity=2.0,
        nll_sum=2.0 * math.log(2.0),
        evaluated_token_count=2,
        rouge_l_f1=0.5,
    )
    second = PrefixKVSampleObservation(
        sample_id="sample-1",
        per_layer_cache_sizes=(20,),
        total_retained_bytes=4096,
        actual_active_kv_bytes=4096,
        generated_answer="second",
        generated_token_ids=(2,),
        latency_seconds=0.1,
        perplexity=4.0,
        nll_sum=4.0 * math.log(4.0),
        evaluated_token_count=4,
        rouge_l_f1=1.0,
    )
    official = replace(_artifact("official_prefixkv"), samples=(first, second))
    reimplementation = replace(_artifact("prefixkv_reimpl"), samples=(first, second))

    report = compare_prefixkv_artifacts(official, reimplementation)
    corpus = cast(JsonObject, report["corpus"])
    assert corpus["sample_count"] == 2
    # exp((2 log 2 + 4 log 4) / 6), not mean(2, 4).
    assert float(corpus["official_teacher_forced_ppl"]) == pytest.approx(3.1748021)
    assert float(corpus["official_mean_rouge_l_f1"]) == 0.75


def test_paper_parity_gate_rejects_a_token_or_ppl_regression() -> None:
    runner = _runner_module()
    observed = replace(
        _artifact("official_prefixkv").samples[0],
        nll_sum=3.0 * math.log(3.5),
        evaluated_token_count=3,
    )
    comparable = compare_prefixkv_artifacts(
        replace(_artifact("official_prefixkv"), samples=(observed,)),
        replace(_artifact("prefixkv_reimpl"), samples=(observed,)),
    )
    passed = runner._parity_gate(
        comparable,
        minimum_token_agreement=0.99,
        maximum_relative_ppl_delta=0.05,
        maximum_rouge_l_delta=0.05,
        maximum_retained_byte_delta=16_384,
    )
    assert passed["passed"] is True

    failed = dict(comparable)
    failed["samples"] = [dict(comparable["samples"][0], token_agreement=0.5)]
    rejected = runner._parity_gate(
        failed,
        minimum_token_agreement=0.99,
        maximum_relative_ppl_delta=0.05,
        maximum_rouge_l_delta=0.05,
        maximum_retained_byte_delta=16_384,
    )
    assert rejected["passed"] is False
    assert "token_agreement:sample-0" in rejected["failures"]


def test_official_profile_loader_preserves_and_validates_raw_forget_ratios(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    profile = tmp_path / "prefixkv_llava-v1.5-7b_0.5.json"
    profile.write_text("[0.0, 0.25, 1.0]\n", encoding="utf-8")
    assert runner._load_official_profile(profile, expected_layers=3) == (0.0, 0.25, 1.0)

    profile.write_text("[0.0, 1.1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        runner._load_official_profile(profile)


def test_official_profile_aggregator_requires_complete_unique_shard_union(tmp_path: Path) -> None:
    aggregate = _official_profile_aggregator_module()
    aggregate.EXPECTED_SAMPLE_COUNT = 2
    data_path = tmp_path / "detail_1k.json"
    data_path.write_text('[{"id": "a"}, {"id": "b"}]', encoding="utf-8")
    profile = tmp_path / "prefixkv_llava-v1.5-7b_0.5.json"
    profile.write_text("[0.5]\n", encoding="utf-8")
    profile_sha = aggregate._sha256_file(profile)
    root = tmp_path / "shards"
    for sample_id in ("a", "b"):
        directory = root / f"worker-{sample_id}"
        directory.mkdir(parents=True)
        manifest = {
            "run_id": f"worker-{sample_id}",
            "git_dirty": False,
            "protocol": {"official_profile_reproduction": True},
            "dataset": {
                "id": aggregate.DATASET_ID,
                "revision": aggregate.DATASET_REVISION,
                "evaluation_sample_ids": [sample_id],
            },
            "profile": {"source_sha256": profile_sha},
            "paper_eligible": False,
            "paper_ineligibility_reason": "incomplete_official_profile_shard",
        }
        artifact = {
            "implementation": "official_prefixkv",
            "samples": [
                {
                    "sample_id": sample_id,
                    "nll_sum": 2.0,
                    "evaluated_token_count": 2,
                    "rouge_l_f1": 0.8,
                }
            ],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "official.json").write_text(json.dumps(artifact), encoding="utf-8")

    result = aggregate.aggregate_official_profile_shards(
        shard_root=root,
        data_path=data_path,
        official_profile_path=profile,
        published_ppl_target=math.e,
        published_ppl_tolerance=0.01,
    )
    assert result["paper_eligible"] is True
    assert result["metrics"]["teacher_forced_ppl"]["value"] == pytest.approx(math.e)
    assert result["metrics"]["rouge_l_f1"]["value"] == pytest.approx(0.8)
