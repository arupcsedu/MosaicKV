from __future__ import annotations

import inspect
from copy import deepcopy
from types import SimpleNamespace

import evicastkv_review_common_bridge as bridge
import pytest


def template():
    return bridge.read(
        bridge.SOURCE / "mosaickv/configs/experiments/evicastkv_fox/milebench/mosaickv_exact.json"
    )


def source_config():
    result = template()
    result["model"]["precision"] = "bf16"
    return result


@pytest.mark.parametrize("method", ("full_kv", "prompt_attention_topk", "mosaickv_exact"))
def test_common_bf16_model_config_is_strict_and_preserves_candidate(method):
    source, candidate = source_config(), template()
    before = deepcopy(candidate)
    config = bridge.exact_config(source, candidate, method)
    assert candidate == before
    assert config.model.id == bridge.MODEL[0]
    assert config.model.revision == bridge.MODEL[1]
    assert config.model.precision.value == "bf16"
    assert config.generation.max_new_tokens == 128
    assert config.execution.attention_implementation == "eager"
    assert config.cache.budget_value == 64 * 2**20
    assert config.bounded_decode is None
    assert config.cache.physical_budget_mandatory_policy == "method"
    assert not config.prototypes.enabled and not config.residual.enabled
    assert not config.repair.enabled
    assert config.graph.enabled == (method == "mosaickv_exact")
    assert config.forecasting.draft_steps == 0


def test_bridge_refuses_mistral_instead_of_silently_porting_source_codec():
    source = source_config()
    source["model"]["id"] = "llava-hf/llava-v1.6-mistral-7b-hf"
    with pytest.raises(ValueError, match="source checkpoint"):
        bridge.exact_config(source, template(), "mosaickv_exact")


def test_no_unregistered_method():
    with pytest.raises(ValueError, match="registered"):
        bridge.exact_config(source_config(), template(), "lookm_reimpl")


def test_resolved_hashes_bind_actual_method_and_execution_policy():
    from dataclasses import replace

    from mosaickv.config import RunConfig, config_sha256

    topk = bridge.arm_config(source_config(), template(), "prompt_attention_topk")
    binding = bridge.resolved_binding(topk, "prompt_attention_topk")
    restored = RunConfig.from_mapping(binding["config"])
    assert binding["config_sha256"] == config_sha256(restored)
    assert restored.execution.seed == bridge.POLICY["seed"]
    changed = replace(topk, execution=replace(topk.execution, seed=14))
    assert (
        bridge.resolved_binding(changed, "prompt_attention_topk")["config_sha256"]
        != binding["config_sha256"]
    )
    assert (
        bridge.resolved_binding(topk, "mosaickv_exact")["resolved_execution_sha256"]
        != binding["resolved_execution_sha256"]
    )


def test_source_arm_records_actual_seed_and_128_token_driver():
    config = bridge.arm_config(source_config(), template(), "source3_attention_optional")
    assert config.execution.seed == bridge.POLICY["seed"]
    assert config.generation.max_new_tokens == 128
    assert config.model.precision.value == "bf16"
    assert bridge.resolved_binding(config, "source3_attention_optional")["config_sha256"]


def test_common_mandatory_positions_match_source_formula():
    spans = [
        SimpleNamespace(start=0, end=5, modality="text"),
        SimpleNamespace(start=5, end=25, modality="image"),
        SimpleNamespace(start=25, end=29, modality="text"),
    ]
    assert bridge.mandatory_positions(spans, 29) == tuple((*range(5), *range(21, 29)))


@pytest.mark.parametrize("length", [1, 2, 4, 8, 12])
def test_short_prompt_mandatory_is_deduplicated_and_monotonic(length):
    assert bridge.mandatory_positions([], length) == tuple(range(length))


def test_invalid_modality_span_fails():
    with pytest.raises(ValueError, match="outside"):
        bridge.mandatory_positions([SimpleNamespace(start=0, end=4, modality="text")], 3)


def rows(sizes=(60, 60, 50)):
    return [
        dict(
            arm=arm,
            mandatory=dict(mandatory_sha256="a" * 64),
            storage=dict(
                initial_active_kv_storage_bytes=size,
                final_active_kv_storage_bytes=size + 30,
                peak_active_kv_storage_bytes=size + 40,
            ),
        )
        for arm, size in zip(bridge.ARMS[2:], sizes, strict=True)
    ]


def test_shared_ceiling_is_not_matched_bytes_and_no_fictitious_fill():
    result = bridge.budget_relation(rows(), 64)
    assert result["within_initial_ceiling"]
    assert result["initial_ceiling_shortfalls"]["source3_attention_optional"] == 14
    assert result["label"] == "common_ceiling_not_matched"
    assert not result["direct_byte_matched_comparison"]
    assert not result["all_boundaries_exactly_matched"]


def test_equal_observed_bytes_do_not_hide_different_history_transport():
    result = bridge.budget_relation(rows((60, 60, 60)), 64)
    assert result["all_boundaries_exactly_matched"]
    assert result["label"] == "equal_observed_bytes_but_transport_scope_differs"
    assert not result["direct_byte_matched_comparison"]


def test_mandatory_mismatch_blocks_direct_comparison():
    records = rows()
    records[0]["mandatory"]["mandatory_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identical mandatory"):
        bridge.budget_relation(records, 64)


def test_over_cap_is_preserved_as_failure_not_clamped():
    result = bridge.budget_relation(rows((65, 60, 50)), 64)
    assert not result["within_initial_ceiling"]
    assert result["initial_ceiling_shortfalls"]["prompt_attention_topk"] == -1


def test_forecast_and_selection_receive_no_reference_prefix():
    source = inspect.getsource(bridge.exact_request)
    assert "messages.load_messages(payload),\n        {}," in source
    assert "forced = torch.full_like(token, prefix[index])" in source
    assert "_mosaickv_teacher_forced_token_ids" not in source
    assert bridge.POLICY["selection_reference_access"] is False
    assert bridge.POLICY["labels_access"] is False
    assert bridge.POLICY["warmups"] == 0 and bridge.POLICY["no_headline_latency"]


def test_immutable_policy_has_only_two_consumed_worker_scope():
    assert len(bridge.ARMS) == 5
    assert bridge.POLICY["consumed_only"]
    assert not bridge.POLICY["fresh_confirmation_authorized"]
    assert bridge.POLICY["no_superiority_claim"]


def test_scoped_factory_and_pack_audit_preserve_all_common_mandatory(monkeypatch):
    import torch

    from mosaickv.adapters.huggingface.types import (
        CachedKeyState,
        CacheLayerSnapshot,
        CacheSnapshot,
    )
    from mosaickv.backends import hf_runtime
    from mosaickv.cache_state import FullKVState, Modality, ModalitySpan

    tensor = torch.zeros((1, 2, 16, 4))
    snapshot = CacheSnapshot(
        (CacheLayerSnapshot(tensor, tensor.clone(), 2),),
        object,
        "fixture",
        16,
        CachedKeyState.POST_ROPE,
    )
    spans = (
        ModalitySpan(0, 2, Modality.TEXT),
        ModalitySpan(2, 14, Modality.IMAGE),
        ModalitySpan(14, 16, Modality.TEXT),
    )
    original_factory = FullKVState.from_cache_snapshot.__func__
    monkeypatch.setattr(hf_runtime, "_pack_exact_runtime_blocks", lambda *args, **kwargs: "packed")
    facts = {}
    with bridge.aligned_mandatory(facts):
        state = FullKVState.from_cache_snapshot(
            snapshot,
            modality_spans=spans,
            token_ids=None,
            block_size=1,
            original_logical_sequence_length=16,
            next_decode_position=16,
            mandatory_logical_positions=(15,),
        )
        assert state.mandatory_logical_positions == frozenset((*range(4), *range(8, 16)))
        assert hf_runtime._pack_exact_runtime_blocks(None, state, state.blocks) == "packed"
        with pytest.raises(ValueError, match="mandatory"):
            hf_runtime._pack_exact_runtime_blocks(None, state, tuple(state.blocks)[1:])
    assert FullKVState.from_cache_snapshot.__func__ is original_factory
    assert facts["all_routes_preserve_common_mandatory"]


def test_forcing_isolated_to_decode_and_wrappers_rollback(tmp_path, monkeypatch):
    import evicastkv_request_validation as messages
    import torch

    from mosaickv.backends import hf_runtime
    from mosaickv.evaluation.model import GenerationMetrics, ModelGeneration

    seen = []

    def prefill(*args, **kwargs):
        return SimpleNamespace(logits=torch.arange(10).reshape(1, 10).float())

    def decode(token, state, *args, **kwargs):
        seen.append(int(token.item()))
        return SimpleNamespace(logits=torch.arange(10).reshape(1, 10).float())

    adapter = SimpleNamespace(prefill=prefill, decode_one_token=decode)

    class Runtime:
        def __init__(self, adapter, config, trace_directory):
            self.adapter, self.path = adapter, trace_directory

        def generate(self, request):
            assert request.generation_kwargs == {}
            self.adapter.prefill(None)
            for _ in range(127):
                self.adapter.decode_one_token(torch.tensor([[999]]), None)
            self.path.mkdir()
            (self.path / "trace.json").write_text("{}")
            return ModelGeneration(
                "not_a_benchmark_answer",
                metrics=GenerationMetrics(
                    initial_active_kv_storage_bytes=10,
                    final_active_kv_storage_bytes=20,
                    peak_active_kv_storage_bytes=20,
                ),
            )

    monkeypatch.setattr(hf_runtime, "HuggingFaceMosaicKVModel", Runtime)
    monkeypatch.setattr(messages, "load_messages", lambda payload: ())
    config = bridge.exact_config(source_config(), template(), "full_kv")
    prefix = tuple(range(128))
    logits, record = bridge.exact_request(
        adapter,
        dict(sample_id="consumed_fixture", task="fixture"),
        config,
        prefix,
        tmp_path / "traces",
    )
    assert seen == list(range(127))
    assert logits.shape == (128, 10)
    assert adapter.prefill is prefill and adapter.decode_one_token is decode
    assert record["selector_prefix_access"] is False
    assert record["free_greedy_sequence"] is False
