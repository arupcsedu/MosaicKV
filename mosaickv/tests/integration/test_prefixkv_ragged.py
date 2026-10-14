"""Random tiny LLaVA CPU correctness tests, not benchmark measurements."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_hf_tiny_models import _baseline_runtime_config, _llava, _runtime_request  # noqa: E402

from mosaickv.adapters.huggingface import compare_with_generate  # noqa: E402
from mosaickv.adapters.huggingface.prefixkv_cache import assert_ragged_support  # noqa: E402
from mosaickv.backends.hf_runtime import (  # noqa: E402
    HuggingFaceMosaicKVModel,
    _clone_decode_state,
    _pack_prefixkv_ragged,
    _state_from_packed,
    maintain_prefixkv_decode_cache,
)
from mosaickv.baselines.prefixkv import (  # noqa: E402
    PrefixKVOfflineProfile,
    build_prefixkv_reimpl_plan,
)
from mosaickv.cache_state import FullKVState  # noqa: E402
from mosaickv.config import CacheConfig, GenerationConfig, PrefixKVConfig  # noqa: E402
from mosaickv.types import BudgetUnit, MosaicKVMethod, PrefixKVProfileMode  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _compressed(adapter, ratio=0.75):
    from transformers.cache_utils import DynamicCache

    prepared = adapter.prepare_inputs([{"role": "user", "content": "test"}])
    prefill = adapter.prefill(prepared, capture_attentions=True)
    full = FullKVState.from_cache_snapshot(
        adapter.extract_past_key_values(prefill.state.past_key_values),
        block_size=1,
        mandatory_logical_positions=(3,),
        modality_spans=prepared.modality_map,
        token_ids=prepared.model_inputs["input_ids"],
    )
    profile = PrefixKVOfflineProfile(
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
        target_retention_ratio=0.75,
        layer_forget_ratios=(0.0, 0.5),
        calibration_dataset_id="synthetic",
        calibration_dataset_revision="b" * 40,
        calibration_split="train",
        calibration_sample_ids=("cal-only",),
        calibration_seed=1,
        start_size=0,
        protect_size=1,
    )
    cfg = PrefixKVConfig(
        enabled=True,
        profile_path="unused-reviewed-profile.json",
        start_size=0,
        hf_cache_transport="ragged_llama_eager_v1",
    )
    plan = build_prefixkv_reimpl_plan(
        full,
        prefill.attention_weights,
        cfg,
        CacheConfig(int(full.active_bytes * ratio), BudgetUnit.BYTES, ratio, 1),
        model_id=profile.model_id,
        model_revision=profile.model_revision,
        profile=profile,
        physical_prompt_capacity=3 if ratio < 1 else 4,
    )
    packed = _pack_prefixkv_ragged(adapter, plan, prefill.state.past_key_values)
    state = _state_from_packed(adapter, prefill.state, packed)
    reference_cache = DynamicCache.from_legacy_cache(
        tuple((layer.key.clone(), layer.value.clone()) for layer in packed.snapshot.layers)
    )
    return prefill, plan, packed, state, reference_cache


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_ragged_decode_matches_independent_layer_oracle_and_isolates_requests(kv_heads):
    """16 tokens, GQA/MHA, unequal lengths, original RoPE positions, evictions."""
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(17)
    adapter = _llava(kv_heads=kv_heads)
    prefill, plan, packed, state, oracle_cache = _compressed(adapter)
    assert plan.layer_cache_sizes == (4, 2)
    assert plan.retained_bytes == sum(
        x.untyped_storage().nbytes()
        for layer in packed.snapshot.layers
        for x in (layer.key, layer.value)
    )
    assert plan.retained_bytes == 3 * 2 * kv_heads * 8 * 4 * 2
    assert packed.validity_masks == ()
    layout = adapter.get_cache_layout(state.past_key_values)
    assert layout.active_sequence_length == 4
    assert tuple(layer.key_shape[-2] for layer in layout.layers) == (4, 2)
    saved = adapter.extract_past_key_values(prefill.state.past_key_values)
    isolated = _clone_decode_state(adapter, state)
    token = prefill.next_token_id
    language = adapter.model.model.language_model
    for step in range(1, 17):
        logical = state.next_decode_position
        before_lengths = [key.shape[-2] for key, _ in oracle_cache.to_legacy_cache()]
        with torch.inference_mode():
            hidden = language.embed_tokens(token)
            positions = torch.tensor([[logical]])
            rotary = language.rotary_emb(hidden, positions)
            for layer in language.layers:
                hidden = layer(
                    hidden,
                    attention_mask=None,
                    position_ids=positions,
                    past_key_value=oracle_cache,
                    cache_position=positions[0],
                    position_embeddings=rotary,
                )
            expected = adapter.model.lm_head(language.norm(hidden))[:, -1, :]
        output = adapter.decode_one_token(token, state, capture_queries=False)
        torch.testing.assert_close(output.logits, expected, rtol=0, atol=1e-6)
        assert torch.equal(output.next_token_id, expected.argmax(-1, keepdim=True))
        state, events = maintain_prefixkv_decode_cache(adapter, output.state, plan, step_index=step)
        oracle_layers = []
        for (key, value), layer, event in zip(
            oracle_cache.to_legacy_cache(), plan.layers, events, strict=True
        ):
            if (
                math.trunc(key.shape[-2] - (logical + 1) * layer.retention_ratio) > 0
                and key.shape[-2] > 1
            ):
                offset = min(layer.eviction_offset, key.shape[-2] - 1)
                key = torch.cat((key[..., :offset, :], key[..., offset + 1 :, :]), dim=-2)
                value = torch.cat((value[..., :offset, :], value[..., offset + 1 :, :]), dim=-2)
            oracle_layers.append((key, value))
            assert event["active_positions_after"] == key.shape[-2]
        oracle_cache = DynamicCache.from_legacy_cache(tuple(oracle_layers))
        for (key, value), (okey, oval), before in zip(
            adapter._legacy_layers(state.past_key_values)[0],
            oracle_layers,
            before_lengths,
            strict=True,
        ):
            torch.testing.assert_close(key, okey, rtol=0, atol=1e-6)
            torch.testing.assert_close(value, oval, rtol=0, atol=1e-6)
            assert key.shape[-2] in (before, before + 1)
            assert key.untyped_storage().nbytes() == key.numel() * key.element_size()
        assert state.next_decode_position == logical + 1
        token = output.next_token_id
    assert isolated.model_state["prefixkv_logical_positions"] == packed.layerwise_logical_positions
    for source, original in zip(
        adapter.extract_past_key_values(prefill.state.past_key_values).layers,
        saved.layers,
        strict=True,
    ):
        assert torch.equal(source.key, original.key)
        assert torch.equal(source.value, original.value)
    # No request-scoped mask override remains active after compressed decoding.
    parity = compare_with_generate(
        adapter, adapter.prepare_inputs([{"role": "user", "content": "test"}]), max_new_tokens=16
    )
    assert parity.token_agreement == 1.0
    assert parity.maximum_logit_difference <= 1e-6


def test_ragged_retention_one_reinjection_matches_untouched_cache():
    torch.manual_seed(19)
    adapter = _llava()
    prefill, plan, _packed, state, _ = _compressed(adapter, ratio=1.0)
    untouched = _clone_decode_state(adapter, prefill.state)
    assert plan.layer_cache_sizes == (4, 4)
    token = prefill.next_token_id
    for _ in range(16):
        original = adapter.decode_one_token(token, untouched, capture_queries=False)
        candidate = adapter.decode_one_token(token, state, capture_queries=False)
        torch.testing.assert_close(candidate.logits, original.logits, rtol=0, atol=1e-6)
        token = original.next_token_id


def test_ragged_full_runtime_writes_trace_and_exact_bytes(tmp_path: Path):
    torch.manual_seed(23)
    adapter = _llava()
    config = replace(
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        method=MosaicKVMethod.PREFIXKV_REIMPL,
        generation=GenerationConfig(max_new_tokens=16),
        cache=CacheConfig(256, BudgetUnit.BYTES, 0.5, 1),
        prefixkv=PrefixKVConfig(
            enabled=True,
            profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
            hf_cache_transport="ragged_llama_eager_v1",
        ),
    )
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path)
    output = runtime.generate(_runtime_request("ragged-runtime"))
    # Random weights may emit EOS first; that is not a transport failure.
    assert isinstance(output.answer, str)
    trace = json.loads(next((tmp_path / "ragged-runtime").glob("*.json")).read_text())
    assert trace["prefixkv"]["hf_cache_transport"] == "ragged_llama_eager_v1"
    assert trace["status"] == "completed"
    assert len(trace["generated_token_ids"]) == 16
    assert trace["prefixkv"]["retained_bytes"] == 256
    assert trace["prefixkv_decode_events"]


def test_ragged_rejects_unaudited_attention():
    with pytest.raises(RuntimeError, match="eager attention only"):
        assert_ragged_support(_llava(attention="sdpa"))


def test_ragged_config_roundtrip_and_invalid_transport():
    cfg = PrefixKVConfig.from_mapping({"hf_cache_transport": "ragged_llama_eager_v1"})
    assert cfg.hf_cache_transport == "ragged_llama_eager_v1"
    assert PrefixKVConfig().hf_cache_transport == "common_guarded"
    with pytest.raises(ValueError, match="supported transport"):
        PrefixKVConfig(hf_cache_transport="silently_pad")


@pytest.mark.parametrize(
    "policy", ["profile_ratio", "effective_profile_ratio", "byte_calibrated", "capacity_floor"]
)
def test_actual_upstream_selector_and_evictor_64_steps(policy):
    from mosaickv.baselines.prefixkv import (
        generate_prefixkv_byte_profile,
        prefixkv_calibration_observation,
    )
    from mosaickv.evaluation.prefixkv_differential import compare_upstream_decode

    torch.manual_seed(177)
    adapter = _llava()
    prepared = adapter.prepare_inputs([{"role": "user", "content": "test"}])
    # A longer random tiny-model prompt exposes fractional rounding schedules.
    prepared = replace(
        prepared,
        model_inputs={
            "input_ids": torch.arange(3, 44).reshape(1, -1),
            "attention_mask": torch.ones((1, 41), dtype=torch.long),
        },
        logical_sequence_length=41,
    )
    prefill = adapter.prefill(prepared, capture_attentions=True)
    full = FullKVState.from_cache_snapshot(
        adapter.extract_past_key_values(prefill.state.past_key_values),
        block_size=1,
        mandatory_logical_positions=(0, 40),
        modality_spans=(),
        token_ids=prepared.model_inputs["input_ids"],
    )
    profile = PrefixKVOfflineProfile(
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
        target_retention_ratio=0.5,
        layer_forget_ratios=(0.411, 0.589),
        calibration_dataset_id="synthetic",
        calibration_dataset_revision="b" * 40,
        calibration_split="train",
        calibration_sample_ids=("cal",),
        calibration_seed=0,
        start_size=1,
        protect_size=1,
    )
    config = PrefixKVConfig(
        enabled=True,
        profile_path="unused.json",
        hf_cache_transport="ragged_llama_eager_v1",
        decode_boundary_policy="protected",
        decode_policy=policy if policy == "profile_ratio" else "effective_profile_ratio",
        allocation_policy="upstream_round" if policy == "profile_ratio" else "global_apportion",
    )
    cache = CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.5, 1)
    if policy in {"byte_calibrated", "capacity_floor"}:
        width = full.active_bytes // (2 * 41)
        profile = generate_prefixkv_byte_profile(
            (prefixkv_calibration_observation("cal-fixture", prefill.attention_weights),),
            byte_cap=28 * width,
            bytes_per_layer_position=width,
            model_id=profile.model_id,
            model_revision=profile.model_revision,
            dataset_id="synthetic",
            dataset_revision="b" * 40,
            calibration_split="train",
            evaluation_sample_ids=("eval-fixture",),
            seed=0,
        )
        config = replace(
            config,
            rescale_offline_profile_to_byte_budget=True,
            profile_reference_retention_ratio=profile.target_retention_ratio,
            layer_budget_floor_fraction=0.5 if policy == "capacity_floor" else 0,
        )
        cache = CacheConfig(28 * width, BudgetUnit.BYTES, 0.99, 1)
    plan = build_prefixkv_reimpl_plan(
        full,
        prefill.attention_weights,
        config,
        cache,
        model_id=profile.model_id,
        model_revision=profile.model_revision,
        profile=profile,
    )
    result = compare_upstream_decode(
        adapter, prefill, plan, Path(__file__).resolve().parents[3], steps=64, atol=1e-6
    )
    assert result["passed"] and result["source_isolated"]
    assert len(result["steps"]) == 64
    assert result["token_agreement"] == 1


def test_append_only_never_evicts_and_is_labeled_in_raw_runtime(tmp_path):
    torch.manual_seed(83)
    adapter = _llava()
    config = replace(
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        method=MosaicKVMethod.PREFIXKV_REIMPL,
        generation=GenerationConfig(max_new_tokens=16),
        cache=CacheConfig(256, BudgetUnit.BYTES, 0.5, 1),
        prefixkv=PrefixKVConfig(
            enabled=True,
            profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
            hf_cache_transport="ragged_llama_eager_v1",
            decode_policy="append_only",
        ),
    )
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path)
    output = runtime.generate(_runtime_request("append-only"))
    assert output.effective_method == "prefixkv_selection_only_reimpl"
    trace = json.loads(next((tmp_path / "append-only").glob("*.json")).read_text())
    events = trace["prefixkv_decode_events"]
    assert len(events) == 30
    assert all(event["removed_physical_position"] is None for event in events)
    assert all(event["decode_policy"] == "append_only" for event in events)
    assert trace["final_active_kv_bytes"] == 256 + 15 * 128


def test_native_round_schedule_works_through_complete_runtime(tmp_path):
    torch.manual_seed(85)
    adapter = _llava()

    class LongerProcessor(type(adapter.processor)):
        def __call__(self, **kwargs):
            return {
                "input_ids": torch.arange(3, 44).reshape(1, -1),
                "attention_mask": torch.ones((1, 41), dtype=torch.long),
            }

    adapter.processor = LongerProcessor()
    path = tmp_path / "native_profile.json"
    path.write_text(json.dumps([0.411, 0.589]))
    config = replace(
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        method=MosaicKVMethod.PREFIXKV_REIMPL,
        generation=GenerationConfig(max_new_tokens=16),
        cache=CacheConfig(2**63 - 1, BudgetUnit.BLOCKS, 0.5, 1),
        prefixkv=PrefixKVConfig(
            enabled=True,
            profile_path=str(path),
            hf_cache_transport="ragged_llama_eager_v1",
            decode_boundary_policy="protected",
            allocation_policy="upstream_round",
            decode_policy="profile_ratio",
        ),
    )
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path)
    runtime.generate(_runtime_request("native-schedule"))
    trace = json.loads(next((tmp_path / "native-schedule").glob("*.json")).read_text())
    assert trace["status"] == "completed"
    assert trace["prefixkv"]["allocation_policy"] == "upstream_round"
    assert len(trace["prefixkv_decode_events"]) == 30


def test_ragged_rejects_future_position_before_forward():
    adapter = _llava()
    _prefill, _plan, _packed, state, _oracle = _compressed(adapter)
    state.model_state["prefixkv_logical_positions"] = ((0, 1, 2, 99), (0, 1))
    with pytest.raises(RuntimeError, match="future token"):
        adapter.decode_one_token(torch.tensor([[3]]), state, capture_queries=False)


def test_protected_three_position_layer_preserves_bos_through_decode():
    torch.manual_seed(31)
    adapter = _llava()
    prepared = adapter.prepare_inputs([{"role": "user", "content": "test"}])
    prefill = adapter.prefill(prepared, capture_attentions=True)
    full = FullKVState.from_cache_snapshot(
        adapter.extract_past_key_values(prefill.state.past_key_values),
        block_size=1,
        mandatory_logical_positions=(0, 3),
        modality_spans=prepared.modality_map,
        token_ids=prepared.model_inputs["input_ids"],
    )
    plan = build_prefixkv_reimpl_plan(
        full,
        prefill.attention_weights,
        PrefixKVConfig(
            enabled=True,
            profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
            hf_cache_transport="ragged_llama_eager_v1",
            decode_boundary_policy="protected",
        ),
        CacheConfig(
            full.active_bytes * 3 // 4,
            BudgetUnit.BYTES,
            0.75,
            1,
            physical_budget_mandatory_policy="first_and_last",
        ),
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
    )
    assert plan.layer_cache_sizes == (3, 3)
    packed = _pack_prefixkv_ragged(adapter, plan, prefill.state.past_key_values)
    state = _state_from_packed(adapter, prefill.state, packed)
    token = prefill.next_token_id
    evicted = 0
    for step in range(16):
        output = adapter.decode_one_token(token, state, capture_queries=False)
        state, events = maintain_prefixkv_decode_cache(adapter, output.state, plan, step_index=step)
        assert all(event["protected_start_present"] for event in events)
        assert all(event["removed_logical_position"] != 0 for event in events)
        assert all(p[0] == 0 for p in state.model_state["prefixkv_logical_positions"])
        evicted += sum(e["removed_logical_position"] is not None for e in events)
        token = output.next_token_id
    assert evicted > 0


def test_protected_decode_rejects_tampered_eviction_offset_before_deleting_bos():
    adapter = _llava()
    _prefill, plan, _packed, state, _oracle = _compressed(adapter)
    plan = replace(
        plan,
        config=replace(plan.config, start_size=1, decode_boundary_policy="protected"),
        layers=tuple(replace(layer, eviction_offset=0) for layer in plan.layers),
    )
    output = adapter.decode_one_token(torch.tensor([[3]]), state, capture_queries=False)
    # Fractional excess truncates to zero on step one; step two must evict.
    output = adapter.decode_one_token(output.next_token_id, output.state, capture_queries=False)
    saved = tuple(adapter._legacy_layers(output.state.past_key_values)[0])
    with pytest.raises(RuntimeError, match="protected start/tail"):
        maintain_prefixkv_decode_cache(adapter, output.state, plan, step_index=0)
    for (key, value), (old_key, old_value) in zip(
        adapter._legacy_layers(output.state.past_key_values)[0], saved, strict=True
    ):
        assert torch.equal(key, old_key) and torch.equal(value, old_value)
