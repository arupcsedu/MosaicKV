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
    _VLCacheProcessor,
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
    LookMConfig,
    PrefixKVConfig,
    PrototypeConfig,
    PublishedKVConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    canonical_config,
    config_sha256,
    load_config,
)
from mosaickv.measurements.cache_lifecycle import CacheLifecycle  # noqa: E402
from mosaickv.types import BudgetUnit, MosaicKVMethod, RepairPolicy  # noqa: E402


@pytest.mark.parametrize(
    "method",
    [
        MosaicKVMethod.RANDOM_KV,
        MosaicKVMethod.UNIFORM_KV,
        MosaicKVMethod.VALUE_TOPK,
        MosaicKVMethod.LOOKM_REIMPL,
        MosaicKVMethod.VL_CACHE_REIMPL,
        MosaicKVMethod.H2O_REIMPL,
        MosaicKVMethod.STREAMINGLLM_REIMPL,
        MosaicKVMethod.PYRAMIDKV_REIMPL,
        MosaicKVMethod.SNAPKV_REIMPL,
        MosaicKVMethod.ROCO_REIMPL,
    ],
)
def test_baselines_share_exact_lifetime_storage_without_native_relabeling(tmp_path, method):
    cfg = _config()
    kwargs = {"method": method}
    if method is MosaicKVMethod.LOOKM_REIMPL:
        kwargs["lookm"] = LookMConfig(
            enabled=True, recent_ratio=0.25, important_ratio=cfg.cache.retention_ratio - 0.25
        )
    elif method is MosaicKVMethod.VL_CACHE_REIMPL:
        from mosaickv.config import VLCacheConfig

        kwargs["vl_cache"] = VLCacheConfig(enabled=True)
    elif method.is_published_eviction_reimplementation:
        kwargs["published_kv"] = PublishedKVConfig(
            enabled=True, sink_tokens=1, observation_window=1
        )
    cfg = replace(cfg, **kwargs)
    adapter = _llava(kv_heads=2 if method is MosaicKVMethod.LOOKM_REIMPL else 1)
    expected_bytes = 1024 if method is MosaicKVMethod.LOOKM_REIMPL else 512
    cfg = replace(cfg, cache=replace(cfg.cache, budget_value=expected_bytes))
    if method is MosaicKVMethod.VL_CACHE_REIMPL:
        adapter.processor = _VLCacheProcessor()
    result = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(
        _runtime_request(method.value)
    )
    assert result.metrics.active_kv_bytes == expected_bytes
    assert result.metrics.initial_active_kv_storage_bytes == expected_bytes
    assert result.metrics.final_active_kv_storage_bytes == expected_bytes
    assert result.metrics.peak_active_kv_storage_bytes == expected_bytes
    assert "__bounded_generated_history_v1" in result.effective_method
    if method.is_published_reimplementation:
        assert result.effective_method.endswith("_reimpl")


def test_prefixkv_profile_remains_layer_ragged_without_padding_or_double_eviction(tmp_path):
    from mosaickv.baselines.prefixkv import PrefixKVOfflineProfile

    cfg = _config()
    profile = PrefixKVOfflineProfile(
        model_id=cfg.model.id,
        model_revision=cfg.model.revision,
        target_retention_ratio=0.625,
        layer_forget_ratios=(0.0, 0.75),
        calibration_dataset_id="synthetic",
        calibration_dataset_revision="b" * 40,
        calibration_split="train",
        calibration_sample_ids=("cal-only",),
        calibration_seed=0,
        start_size=0,
        protect_size=1,
    )
    path = profile.write(tmp_path / "profile.json")
    cfg = replace(
        cfg,
        method=MosaicKVMethod.PREFIXKV_REIMPL,
        prefixkv=PrefixKVConfig(
            enabled=True,
            profile_path=str(path),
            start_size=0,
            hf_cache_transport="ragged_llama_eager_v1",
            rescale_offline_profile_to_byte_budget=True,
            profile_reference_retention_ratio=0.625,
            decode_policy="append_only",
        ),
    )
    result = HuggingFaceMosaicKVModel(_llava(), cfg, trace_directory=tmp_path / "traces").generate(
        _runtime_request("bounded-prefix")
    )
    trace = json.loads(next((tmp_path / "traces").rglob("*.json")).read_text())
    lengths = trace["bounded_decode"]["protected_prompt_positions_per_layer"]
    assert len(set(lengths)) == 2
    assert sum(lengths) == 4
    assert trace["bounded_decode"]["cross_layer_padding_bytes"] == 0
    assert trace["prefixkv_decode_events"] == []
    assert trace["bounded_decode"]["evicted_generated_positions_per_head"] == 17
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert result.effective_method.endswith("__bounded_generated_history_v1_reimpl")


def test_vl_cache_adaptive_layer_lengths_use_no_padding_and_keep_exact_lifetime_cap(
    tmp_path, monkeypatch
):
    import mosaickv.baselines.vl_cache as vl
    from mosaickv.config import VLCacheConfig

    monkeypatch.setattr(
        vl, "paper_layer_retention_ratios", lambda *_a, **_k: ((0.75, 0.25), (0.75, 0.25))
    )
    adapter = _llava()
    adapter.processor = _VLCacheProcessor()
    cfg = replace(
        _config(), method=MosaicKVMethod.VL_CACHE_REIMPL, vl_cache=VLCacheConfig(enabled=True)
    )
    result = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=tmp_path).generate(
        _runtime_request("vl-ragged-regression")
    )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["bounded_decode"]["protected_prompt_positions_per_layer"] == [3, 1]
    assert trace["bounded_decode"]["cross_layer_padding_bytes"] == 0
    assert trace["packed_cache_all_slots_valid"] is True
    assert result.metrics.initial_active_kv_storage_bytes == 512
    assert result.metrics.final_active_kv_storage_bytes == 512
    assert result.metrics.peak_active_kv_storage_bytes == 512


def test_evidence_allocation_is_a_labeled_exact_byte_runtime_variant(tmp_path):
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(
            generated_capacity=2,
            sink_tokens=1,
            history_fraction=0.2,
            layer_allocation="output_impact_v1",
        ),
    )
    result = HuggingFaceMosaicKVModel(_llava(), cfg, trace_directory=tmp_path).generate(
        _runtime_request("layer-impact")
    )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["evidence_layer_allocation"]["padding_bytes"] == 0
    assert trace["evidence_layer_allocation"]["benchmark_labels_accessed"] is False
    assert sum(trace["evidence_layer_allocation"]["layer_capacities"]) == 4
    assert trace["hard_prompt_capacity"] is None
    assert (
        trace["selected_layer_prompt_capacities"]
        == trace["evidence_layer_allocation"]["layer_capacities"]
    )
    assert trace["byte_budget_materialization"]["dense_quantization_underfill_bytes"] == 0
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert "output_impact_layers" in result.effective_method
    assert result.effective_method.endswith("bounded_generated_history_v2")


def test_guarded_allocation_reports_final_errors_and_keeps_lifetime_cap(tmp_path):
    cfg = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT),
        bounded_decode=BoundedDecodeConfig(2, 1, 0.2, "output_impact_guarded_v1"),
    )
    result = HuggingFaceMosaicKVModel(_llava(), cfg, trace_directory=tmp_path).generate(
        _runtime_request("guarded-layer-impact")
    )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    allocation = trace["evidence_layer_allocation"]
    guard = allocation["reconstruction_guard"]
    assert allocation["padding_bytes"] == 0
    assert allocation["benchmark_labels_accessed"] is False
    assert len(guard["final_mse_by_layer_head"]) == 2
    assert guard["used_fullkv_fallback"] is False
    assert guard["host_wall_seconds_including_device_transfers"] >= 0
    for before, after in zip(
        guard["anchor_mse_by_layer_head"], guard["final_mse_by_layer_head"], strict=True
    ):
        assert all(
            b <= a + guard["absolute_tolerance"] + a * guard["relative_tolerance"]
            for a, b in zip(before, after, strict=True)
        )
    assert sum(allocation["layer_capacities"]) == 4
    assert result.metrics.initial_active_kv_storage_bytes == 512
    assert result.metrics.final_active_kv_storage_bytes == 512
    assert result.metrics.peak_active_kv_storage_bytes == 512
    assert "__guarded_output_impact_layers" in result.effective_method


def test_rejected_reallocation_reuses_packed_anchor_without_gathering(monkeypatch):
    from types import SimpleNamespace

    import mosaickv.adapters.huggingface.layer_local_cache as gathering
    import mosaickv.selection.evidence_layer_allocation as selection
    from mosaickv.backends.hf_runtime import _reallocate_evidence_layers

    anchor = SimpleNamespace(
        prompt_capacity=2,
        slot_records=(),
        snapshot=SimpleNamespace(
            layers=(SimpleNamespace(key=torch.zeros(1, 1, 2, 2), value=torch.zeros(1, 1, 2, 2)),)
        ),
    )
    plan = SimpleNamespace(
        prompt_bytes=32,
        reconstruction={"accepted": False},
        trace=lambda: {"reconstruction_guard": {"accepted": False}},
    )
    monkeypatch.setattr(selection, "build_evidence_layer_plan", lambda *_a, **_k: plan)

    def forbidden_gather(*_a, **_k):
        pytest.fail("rollback must reuse the existing same-byte anchor")

    monkeypatch.setattr(gathering, "gather_layer_local_snapshot", forbidden_gather)
    returned, trace = _reallocate_evidence_layers(
        None, None, None, anchor, {}, 32, guard_reconstruction=True
    )
    assert returned is anchor
    assert trace["reconstruction_guard"]["accepted"] is False


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(204)
    yield
    torch.set_num_threads(previous)


def _config(method=MosaicKVMethod.PROMPT_ATTENTION_TOPK, *, ratio=0.9999999999999999):
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


@pytest.mark.parametrize("allocation", ["uniform", "output_impact_guarded_v1"])
def test_retention_one_bypasses_ring_and_matches_untouched_fullkv(tmp_path, allocation):
    adapter = _llava()
    bounded = HuggingFaceMosaicKVModel(
        adapter,
        replace(
            _config(MosaicKVMethod.MOSAICKV_EXACT, ratio=1.0),
            bounded_decode=BoundedDecodeConfig(2, 1, layer_allocation=allocation),
        ),
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


def test_bounded_native_metrics_share_a_policy_bound_full_reference(tmp_path):
    import hashlib

    from PIL import Image

    from mosaickv.config import DatasetConfig
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import record_hf_run

    adapter = _llava()
    config = _config()
    spec = default_task_registry().resolve("llava_description")
    config = replace(config, dataset=DatasetConfig(spec.dataset_id, "b" * 40, spec.split))
    full = replace(
        config,
        method=MosaicKVMethod.FULL_KV,
        bounded_decode=None,
        cache=replace(config.cache, retention_ratio=1.0, budget_value=2**63 - 1),
    )
    media = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(media)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            dict(
                schema_version=1,
                task=spec.name,
                dataset_id=spec.dataset_id,
                dataset_revision="b" * 40,
                split=spec.split,
                samples=[
                    dict(
                        sample_id="native-fixture",
                        prompt="Describe.",
                        references=["target"],
                        teacher_forcing_target="human target",
                        media=[
                            dict(
                                kind="image",
                                path=str(media),
                                sha256=hashlib.sha256(media.read_bytes()).hexdigest(),
                            )
                        ],
                    )
                ],
            )
        )
    )
    reference = tmp_path / "reference.json"
    policy = dict(
        name="common_bounded_history_v1_fullkv_reference_exempt",
        generated_capacity=2,
        sink_tokens=1,
    )
    common = dict(
        adapter=adapter,
        task=spec.name,
        output_root=tmp_path / "output",
        ledger_path=ledger,
        ledger_sha=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        sample_ids=("native-fixture",),
        phase="synthetic_cpu_not_quality_evidence",
        repetition=0,
        order=0,
        export_metrics=True,
        materialized_prefixkv=True,
        comparison_decode_policy=policy,
    )
    first = record_hf_run(full, **common, run_id="full", fullkv_reference_output=reference)
    second = record_hf_run(
        config,
        **common,
        run_id="topk",
        fullkv_reference_input=reference,
        fullkv_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
    )
    assert first["rows"][0]["status"] == second["rows"][0]["status"] == "completed"
    assert second["rows"][0]["active_kv_bytes"] == 512
