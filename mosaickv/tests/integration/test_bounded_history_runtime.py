"""Random-weight CPU architecture tests, NOT real-checkpoint answer quality."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_hf_tiny_models import (  # noqa: E402
    _baseline_runtime_config,
    _llava,
    _runtime_config,
    _runtime_request,
)

from mosaickv.adapters.huggingface.bounded_history_cache import (  # noqa: E402
    install_bounded_history,
)
from mosaickv.adapters.huggingface.types import DecodeState  # noqa: E402
from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel  # noqa: E402
from mosaickv.config import (  # noqa: E402
    BoundedDecodeConfig,
    CacheConfig,
    ConfigurationError,
    GenerationConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    canonical_config,
    config_sha256,
    load_config,
)
from mosaickv.measurements.cache_lifecycle import CacheLifecycle  # noqa: E402
from mosaickv.types import BudgetUnit, MosaicKVMethod, RepairPolicy  # noqa: E402


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(204)
    yield
    torch.set_num_threads(previous)


def _config(method=MosaicKVMethod.PROMPT_ATTENTION_TOPK, *, ratio=0.5):
    base = (
        _baseline_runtime_config(method, ratio)
        if method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
        else _runtime_config(method, ratio)
    )
    return replace(
        base,
        generation=GenerationConfig(max_new_tokens=20),
        cache=CacheConfig(
            512,
            BudgetUnit.BYTES,
            ratio,
            1,
            physical_budget_block_size=1,
            physical_budget_mandatory_policy="last_token",
        ),
        bounded_decode=BoundedDecodeConfig(generated_capacity=2, sink_tokens=1),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False, require_pinned_memory=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE),
    )


@pytest.mark.parametrize(
    "method", [MosaicKVMethod.MOSAICKV_EXACT, MosaicKVMethod.PROMPT_ATTENTION_TOPK]
)
def test_full_runtime_caps_initial_final_peak_and_writes_variant_trace(tmp_path, method):
    runtime = HuggingFaceMosaicKVModel(_llava(), _config(method), trace_directory=tmp_path)
    result = runtime.generate(_runtime_request(f"bounded-{method.value}"))
    assert result.metrics.generated_tokens == 20
    assert result.metrics.active_kv_bytes == 512
    assert result.metrics.initial_active_kv_storage_bytes == 512
    assert result.metrics.final_active_kv_storage_bytes == 512
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert result.effective_method.endswith("__bounded_generated_history_v1")
    trace_path = next(tmp_path.rglob("*.json"))
    trace = json.loads(trace_path.read_text())
    record = trace["bounded_decode"]
    assert record["protected_prompt_positions_per_head"] == 2
    assert record["evicted_generated_positions_per_head"] == 17
    assert record["next_decode_position"] == 23
    assert set(record["retained_generated_logical_positions_physical_order"]) == {4, 22}
    assert trace["cache_lifecycle"]["peak_active_kv_storage_bytes"] == 512
    assert record["budget_reservation"]["reserved_generated_bytes"] == 256


def test_retention_one_bypasses_ring_and_matches_untouched_fullkv(tmp_path):
    adapter = _llava()
    bounded = HuggingFaceMosaicKVModel(
        adapter,
        _config(MosaicKVMethod.MOSAICKV_EXACT, ratio=1.0),
        trace_directory=tmp_path / "bounded",
    )
    reference_config = replace(
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        generation=GenerationConfig(max_new_tokens=20),
    )
    reference = HuggingFaceMosaicKVModel(
        adapter, reference_config, trace_directory=tmp_path / "reference"
    )
    assert (
        bounded.generate(_runtime_request("identity")).answer
        == reference.generate(_runtime_request("reference")).answer
    )
    trace = json.loads(next((tmp_path / "bounded").rglob("*.json")).read_text())
    ref_trace = json.loads(next((tmp_path / "reference").rglob("*.json")).read_text())
    assert trace["generated_token_ids"] == ref_trace["generated_token_ids"]
    assert trace["bounded_decode"]["status"] == "bypassed_retention_one_identity"
    assert trace["effective_method"].endswith("__retention_one_exact")


def test_ring_logits_equal_explicit_chronological_eviction_reference():
    """Checks actual HF RoPE/attention before/after many ring wraps, not mocks."""
    adapter = _llava(kv_heads=2)
    prepared = adapter.prepare_inputs(_runtime_request("rope").messages)
    prefill = adapter.prefill(prepared)
    prompt = adapter._legacy_layers(prefill.state.past_key_values)[0]
    next_position = prefill.state.next_decode_position
    state = DecodeState(
        adapter.inject_past_key_values(
            adapter.extract_past_key_values(prefill.state.past_key_values)
        ),
        torch.ones(1, 4, dtype=torch.long),
        4,
        4,
        4,
        (),
        {"mosaickv_packed_all_valid": True, "mosaickv_cache_all_valid": True},
    )
    install_bounded_history(adapter, state, generated_capacity=3, sink_tokens=1, cap_bytes=7 * 256)
    reference_layers = tuple((k.clone(), v.clone()) for k, v in prompt)
    generated_positions = []
    token = prefill.next_token_id
    for step in range(16):
        # Before an append, remove only the oldest non-sink generated slot.
        if len(generated_positions) == 3:
            keep = torch.tensor([0, 1, 2, 3, 4, 6])
            reference_layers = tuple(
                (k.index_select(-2, keep), v.index_select(-2, keep)) for k, v in reference_layers
            )
            generated_positions.pop(1)
        from transformers import DynamicCache

        reference_cache = DynamicCache.from_legacy_cache(reference_layers)
        position = torch.tensor([next_position + step])
        with torch.inference_mode():
            output = adapter.model(
                input_ids=token,
                past_key_values=reference_cache,
                position_ids=position.reshape(1, 1),
                cache_position=position,
                use_cache=True,
                return_dict=True,
            )
        result = adapter.decode_one_token(token, state, capture_queries=False)
        torch.testing.assert_close(result.logits, output.logits[:, -1, :], rtol=1e-5, atol=1e-6)
        assert torch.equal(result.next_token_id, output.logits[:, -1, :].argmax(-1, keepdim=True))
        reference_layers = adapter._legacy_layers(output.past_key_values)[0]
        generated_positions.append(next_position + step)
        token = result.next_token_id
        assert state.next_decode_position == next_position + step + 1
        assert CacheLifecycle.count(adapter._legacy_layers(state.past_key_values)[0])[1] == 7 * 256


def test_teacher_forcing_uses_bounded_path_and_complete_targets(tmp_path):
    adapter = _llava()
    target_ids = tuple(range(3, 23))
    adapter.processor.tokenizer = lambda *_args, **_kwargs: {
        "input_ids": torch.tensor([target_ids])
    }
    runtime = HuggingFaceMosaicKVModel(adapter, _config(), trace_directory=tmp_path)
    request = _runtime_request("teacher-bounded")
    result = runtime.teacher_forced_nll(request, "a synthetic target")
    assert result.evaluated_tokens == 20
    assert result.negative_log_likelihood_sum >= 0
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["generated_token_ids"] == list(target_ids)
    assert trace["teacher_forced_evaluation"]["evaluated_tokens"] == 20
    assert trace["teacher_forced_evaluation"]["target_was_not_truncated"]
    assert trace["bounded_decode"]["evicted_generated_positions_per_head"] == 17


def test_unsupported_attention_and_unsafe_state_apis_fail_explicitly():
    adapter = _llava(attention="sdpa")
    prepared = adapter.prepare_inputs(_runtime_request("unsupported").messages)
    state = adapter.prefill(prepared).state
    with pytest.raises(RuntimeError, match=r"LLaVA-1\.5 eager"):
        install_bounded_history(adapter, state, generated_capacity=2, sink_tokens=0, cap_bytes=768)
    adapter.set_attention_implementation("eager")
    state.model_state["mosaickv_packed_all_valid"] = True
    install_bounded_history(adapter, state, generated_capacity=2, sink_tokens=0, cap_bytes=768)
    with pytest.raises(NotImplementedError, match="generic snapshot/reinjection"):
        adapter.extract_past_key_values(state.past_key_values)
    with pytest.raises(RuntimeError, match="erase bounded-history"):
        adapter.preallocate_decode_cache(state, 8)
    with pytest.raises(NotImplementedError, match="candidate-batch"):
        adapter.decode_token_batch(torch.tensor([[3]]), state)


def test_mandatory_underflow_preserves_failure_trace(tmp_path):
    config = _config()
    config = replace(
        config,
        cache=replace(
            config.cache, budget_value=384, physical_budget_mandatory_policy="first_and_last"
        ),
    )
    runtime = HuggingFaceMosaicKVModel(_llava(), config, trace_directory=tmp_path)
    with pytest.raises(RuntimeError, match="mandatory"):
        runtime.generate(_runtime_request("mandatory-underflow"))
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["status"] == "failed"
    assert trace["bounded_decode"]["budget_reservation"]["reserved_generated_bytes"] == 256


def test_development_preset_has_reproducible_strict_configuration():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs/development/llava_bounded_history_v1.yaml"
    config = load_config(path)
    assert config.bounded_decode == BoundedDecodeConfig(32, 4)
    assert config_sha256(RunConfig.from_mapping(canonical_config(config))) == config_sha256(config)
    with pytest.raises(ConfigurationError, match="bounded_decode requires"):
        replace(config, cache=replace(config.cache, total_storage_budget_bytes=1))
