from __future__ import annotations

from dataclasses import replace
from typing import cast

from mosaickv.baselines import (
    PrefixKVParityArtifact,
    PrefixKVParityControls,
    PrefixKVSampleObservation,
    compare_prefixkv_artifacts,
)
from mosaickv.types import JsonObject


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
