from __future__ import annotations

from dataclasses import replace

import pytest

from mosaickv.baselines import (
    LookMParityArtifact,
    LookMParityControls,
    LookMParityError,
    LookMSampleObservation,
    LookMSelectedPositions,
    build_lookm_parity_report,
    compare_lookm_artifacts,
)


def _controls() -> LookMParityControls:
    return LookMParityControls(
        model_id="liuhaotian/llava-v1.5-7b",
        model_revision="a" * 40,
        tokenizer_id="liuhaotian/llava-v1.5-7b",
        tokenizer_revision="a" * 40,
        dataset_id="controlled/lookm-parity",
        dataset_revision="b" * 40,
        sample_set_sha256="c" * 64,
        prompt_payload_sha256="d" * 64,
        media_payload_sha256="e" * 64,
        environment_sha256="2" * 64,
        hardware_sha256="3" * 64,
        measurement_protocol_sha256="4" * 64,
        cache_budget_value=4096,
        cache_budget_unit="retained_slots",
        block_size=1,
        retention_ratio=0.2,
        recent_ratio=0.1,
        important_ratio=0.1,
        merge_strategy="pivotal",
        generation_parameters={"do_sample": False, "max_new_tokens": 4, "temperature": 0.0},
        output_length_policy="fixed_max_new_tokens",
        model_precision="bf16",
        backend="legacy_llava_eager",
        backend_configuration={"transformers": "4.37.0"},
        attention_implementation="eager",
        seed=0,
    )


def _artifact(implementation: str) -> LookMParityArtifact:
    official = implementation == "official_lookm"
    return LookMParityArtifact(
        implementation=implementation,
        official_repository_sha="ecf0f51a9c416c2d85e47faf2638502f01a6d748",
        executable_git_sha="f" * 40,
        config_sha256="1" * 64,
        manifest_path="/artifact/manifest.json",
        measurement_type=(
            "baseline_official_measured" if official else "baseline_reimpl_measured"
        ),
        controls=_controls(),
        samples=(
            LookMSampleObservation(
                sample_id="sample-0001",
                selected_positions=(LookMSelectedPositions(0, 0, (0, 3)),),
                active_kv_bytes=128,
                generated_token_ids=(1, 2, 3, 4),
                task_score=1.0,
                latency_seconds=0.5,
            ),
        ),
    )


def test_controlled_parity_compares_all_requested_observations() -> None:
    report = compare_lookm_artifacts(
        _artifact("official_lookm"),
        _artifact("lookm_reimpl"),
    )

    assert report["status"] == "comparable"
    sample = report["samples"][0]
    assert sample["selected_positions_exact_match"] is True
    assert sample["active_kv_bytes_delta"] == 0
    assert sample["generated_tokens_exact_match"] is True
    assert sample["task_score_delta"] == 0
    assert sample["latency_seconds_delta"] == 0


def test_mismatched_backend_produces_no_numerical_comparison() -> None:
    reimplementation = _artifact("lookm_reimpl")
    reimplementation = replace(
        reimplementation,
        controls=replace(reimplementation.controls, backend="hf_transformers"),
    )

    report = build_lookm_parity_report(_artifact("official_lookm"), reimplementation)

    assert report["status"] == "not_comparable"
    assert report["samples"] == []
    assert report["control_mismatches"] == ["controls.backend"]
    with pytest.raises(LookMParityError, match=r"controls\.backend"):
        compare_lookm_artifacts(_artifact("official_lookm"), reimplementation)
