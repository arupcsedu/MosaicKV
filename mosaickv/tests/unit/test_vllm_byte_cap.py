"""Serving budgets use physical padded storage, never nominal retention."""

import asyncio
import sys
import types
from dataclasses import replace

import pytest

from mosaickv.backends.native_selector_plan import (
    NativeSelectorPlanEntry,
    NativeSelectorPlanError,
    RequestWideSelectorProjection,
    fit_projection_to_byte_cap,
)
from mosaickv.backends.vllm_runtime import AsyncVLLMTrialRunner, prepare_vllm_prompt
from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages


def projection(method="mosaickv_exact"):
    return RequestWideSelectorProjection(
        method=method,
        logical_positions=65,
        block_size=1,
        retention_ratio=1.0,
        selected_blocks=tuple(range(65)),
        mandatory_blocks=(0, 64),
        selected_position_votes=tuple(range(65)),
        selected_score_sums=(1.0,) * 65,
        source_selected_positions=65,
        source_total_positions=65,
    )


@pytest.mark.parametrize("cap,blocks", [(32, 2), (47, 2), (48, 3), (79, 4), (80, 5), (999, 5)])
def test_largest_whole_block_capacity_and_padding(cap, blocks):
    plan = fit_projection_to_byte_cap(
        projection(), target_block_size=16, physical_byte_cap=cap, bytes_per_position=1
    )
    assert len(plan.selected_blocks) == blocks
    assert {0, 4}.issubset(plan.selected_blocks)
    assert len(plan.selected_blocks) * 16 <= cap
    if blocks < 5:
        assert cap - blocks * 16 < 16
    assert plan == fit_projection_to_byte_cap(
        projection(), target_block_size=16, physical_byte_cap=cap, bytes_per_position=1
    )


def test_infeasible_mandatory_cap_is_rejected_not_rounded_up():
    with pytest.raises(NativeSelectorPlanError, match="mandatory"):
        fit_projection_to_byte_cap(
            projection(), target_block_size=16, physical_byte_cap=31, bytes_per_position=1
        )


@pytest.mark.parametrize(
    "method",
    [
        "prompt_attention_topk",
        "prefixkv_reimpl",
        "vl_cache_reimpl",
        "lookm_reimpl",
        "mosaickv_exact",
    ],
)
def test_same_byte_target_for_every_compressed_method(method):
    plan = fit_projection_to_byte_cap(
        projection(method), target_block_size=16, physical_byte_cap=50, bytes_per_position=1
    )
    entry = NativeSelectorPlanEntry("sample", plan, 0.1, "trace", "a" * 64, 50, 1)
    assert len(entry.projection.selected_blocks) * 16 == 48
    with pytest.raises(NativeSelectorPlanError, match="underfilled"):
        replace(entry, physical_byte_cap=64)


@pytest.mark.parametrize("cap", [True, 0, -1, 3.5])
def test_invalid_cap(cap):
    with pytest.raises(NativeSelectorPlanError, match="positive integer"):
        fit_projection_to_byte_cap(
            projection(), target_block_size=16, physical_byte_cap=cap, bytes_per_position=1
        )


def test_llava_next_fallback_matches_hf_and_does_not_call_missing_template():
    class Processor:
        chat_template = None
        tokenizer = None

        def apply_chat_template(self, *_args, **_kwargs):
            raise AssertionError("no template available")

    messages = build_multimodal_messages("Read this.", (MediaItem(MediaKind.IMAGE, b"pixels"),))
    prepared = prepare_vllm_prompt(Processor(), "llava-hf/llava-v1.6-mistral-7b-hf", messages)
    assert prepared.rendered_text == "[INST] <image>\nRead this. [/INST]"
    assert prepared.engine_prompt["multi_modal_data"] == {"image": b"pixels"}


def test_request_sampler_uses_catalog_ratio_not_nominal_hf_placeholder(monkeypatch):
    """Capture SamplingParams before any CUDA or engine access."""
    captured = {}

    class Captured(Exception):
        pass

    def sampling(**kwargs):
        captured.update(kwargs)
        raise Captured

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = sampling
    params = types.ModuleType("vllm.sampling_params")
    params.RequestOutputKind = types.SimpleNamespace(DELTA="delta")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", params)
    runner = object.__new__(AsyncVLLMTrialRunner)
    runner.options = types.SimpleNamespace(enable_mosaickv=True)
    runner.config = types.SimpleNamespace(
        cache=types.SimpleNamespace(retention_ratio=0.99, block_size=16)
    )
    generation = {
        "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64, "seed": 3,
        "native_selected_blocks": [0, 4], "native_mandatory_blocks": [0, 4],
        "native_retention_ratio": 0.4, "native_block_size": 16,
        "native_logical_positions": 65, "native_policy": "byte_cap",
        "native_method": "mosaickv_exact", "native_catalog_sha256": "a" * 64,
        "native_selector_planning_seconds": 0.0, "native_source_trace_sha256": "b" * 64,
    }
    with pytest.raises(Captured):
        asyncio.run(runner._run_async({}, generation, "test"))
    native = captured["extra_args"]["mosaickv_native"]
    assert native["retention_ratio"] == 0.4
    assert native["selected_blocks"] == [0, 4]
    assert native["selected_block_count"] == 2
