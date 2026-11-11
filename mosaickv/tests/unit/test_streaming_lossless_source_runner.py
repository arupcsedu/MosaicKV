"""CPU-only runner boundaries; synthetic tiny HF weights, never quality evidence."""

import copy
import gc
import inspect
import json
import weakref

import evicastkv_streaming_lossless_source as study
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mosaickv.adapters.huggingface.types import (
    DecodeState,
    PrefillOutput,
    PreparedInputs,
    QueryVectors,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class TinyAdapter:
    def __init__(self):
        torch.manual_seed(782)
        self.model = (
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=32,
                    hidden_size=512,
                    intermediate_size=768,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    head_dim=128,
                    max_position_embeddings=32,
                    attn_implementation="eager",
                )
            )
            .bfloat16()
            .eval()
        )
        self.tokens, self.calls, self.raw_refs = torch.arange(8)[None], [], []

    def _language_layers(self):
        return self.model.model.layers

    @staticmethod
    def _legacy_layers(cache):
        return tuple((layer.keys, layer.values) for layer in cache.layers), "cache_layers"

    def prepare_inputs(self, payload):
        return PreparedInputs(dict(input_ids=self.tokens.clone()), (), self.tokens.shape[1])

    def prefill(self, prepared, *, capture_queries, capture_attentions):
        self.calls.append((capture_queries, capture_attentions))
        output = self.model(**prepared.model_inputs, use_cache=True)
        for layer in output.past_key_values.layers:
            if layer.keys is not None:
                self.raw_refs.extend(weakref.ref(t) for t in (layer.keys, layer.values))
        logits = output.logits[:, -1, :]
        length = prepared.logical_sequence_length
        return PrefillOutput(
            logits,
            logits.argmax(-1)[:, None],
            DecodeState(output.past_key_values, torch.ones(1, length), length, length, length, ()),
            QueryVectors(()),
        )


def test_native_control_does_not_pin_last_loop_pair_or_original_cache(monkeypatch):
    import evicastkv_request_validation as runtime

    monkeypatch.setattr(runtime, "load_messages", lambda payload: payload)
    adapter = TinyAdapter()
    with torch.inference_mode():
        result = study.native_prefill_control(adapter, {"prompt": "synthetic"})
    gc.collect()
    assert result["retired"]
    assert len(result["pairs"]) == 2
    assert all(ref() is None for ref in adapter.raw_refs)
    assert result["raw_bytes"] == result["CPU_verification_KV_bytes"] > 0
    assert result["logical_length"] == 8
    assert result["logits"].dtype == torch.bfloat16


def test_stream_request_has_no_global_capture_and_prefix_enters_only_reader(monkeypatch):
    import evicastkv_request_validation as runtime

    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture

    monkeypatch.setattr(runtime, "load_messages", lambda payload: payload)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 12345)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 23456)
    monkeypatch.setattr(
        SourceCapture, "__init__", lambda *a, **k: pytest.fail("global source capture used")
    )
    monkeypatch.setattr(
        PromptQueryCapture, "__init__", lambda *a, **k: pytest.fail("global query capture used")
    )
    adapter, prefix, saved = TinyAdapter(), tuple(range(128)), []

    def decode(actual_adapter, prefill, packets, **kwargs):
        assert actual_adapter is adapter
        assert prefill.state.past_key_values is None
        assert all(ref() is None for ref in adapter.raw_refs)
        assert kwargs["forced"] is prefix
        assert kwargs["source_history"] and kwargs["shared_workspace"]
        assert kwargs["audit"] and not kwargs["workspace_graph"]
        raw = sum(packet.raw_kv_bytes for packet in packets)
        assert kwargs["initial_physical_byte_cap"] == raw
        assert len(packets) == 2 and all(packet.shape[-2] == 8 for packet in packets)
        return torch.zeros(128, 32), dict(
            used_fullkv_fallback=False,
            output_tokens=128,
            storage=dict(initial_active_kv_storage_bytes=sum(p.component_bytes for p in packets)),
        )

    monkeypatch.setattr(storage_runs, "run", decode)
    with torch.inference_mode():
        _, row = study.request(
            adapter, {"prompt": "synthetic"}, object(), prefix=prefix, save_prefill=saved.append
        )
    assert adapter.calls == [(False, False), (False, False)]
    assert study.parent.bitwise(saved[0]["native"], saved[0]["streaming"])
    assert row["global_source_or_query_capture"] is False
    assert row["independent_prefill_control"]["CPU_control_retired_before_decode"]
    assert row["inactive_original_source_retired"]
    assert row["streamed_source"]["prefill_storage"]["original_native_layers_remaining"] == 0
    assert not row["whole_request_KV_peak_certified"]
    assert row["no_latency_or_quality_claim"]
    assert len(row["phase_wall_seconds"]) == 5
    assert all(value >= 0 for value in row["phase_wall_seconds"].values())


def prerequisite(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    names = ("raw.jsonl", "logits.pt", "manifest.json", "GATE_cuda_operator.json")
    for name in names:
        (worker / name).write_bytes(b"synthetic immutable output")
    outputs = {name: study.sha(worker / name) for name in names}
    gate = dict(
        source_sha=study.PARENT_SOURCE,
        status="completed_lossless_source_diagnostic",
        completed=2,
        failures=[],
        all_original_prompt_KV_bitwise=True,
        all_unchanged_32_token_guards_pass=True,
        all_128_source_tokens_agree=True,
        both_initial_caches_smaller_than_raw=True,
    )
    result = dict(
        source_sha=study.PARENT_SOURCE,
        status="completed",
        completed_samples=2,
        failures=[],
        outputs=outputs,
    )
    for name, value in (
        ("GATE_lossless_source_control.json", gate),
        ("protocol.json", {"source_sha": study.PARENT_SOURCE, "jobs": []}),
        ("worker/result.json", result),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    return gate


def test_prerequisite_requires_passed_original_allword_32_128_and_bound_outputs(tmp_path):
    prerequisite(tmp_path)
    protocol, bindings = study.parent_gate(tmp_path)
    assert protocol["source_sha"] == study.PARENT_SOURCE
    assert len(bindings) == 7
    (tmp_path / "worker/logits.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="output changed"):
        study.parent_gate(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "all_original_prompt_KV_bitwise",
        "all_unchanged_32_token_guards_pass",
        "all_128_source_tokens_agree",
        "both_initial_caches_smaller_than_raw",
    ],
)
def test_failed_prerequisite_cannot_be_promoted(tmp_path, field):
    gate = prerequisite(tmp_path)
    gate[field] = False
    (tmp_path / "GATE_lossless_source_control.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="immutable V12"):
        study.parent_gate(tmp_path)


def valid_row():
    records = [
        dict(
            layer=index,
            persistent_bytes=60,
            raw_kv_bytes=100,
            reference="original_prefill_KV_not_lossy_parent_decoder",
            all_prompt_positions_retained=True,
        )
        for index in range(32)
    ]
    job = dict(index=0, task="llava_description", sample_id="synthetic-consumed")
    row = dict(
        **job,
        method=study.LABEL,
        used_fullkv_fallback=False,
        source_fallback=False,
        inactive_original_source_retired=True,
        global_source_or_query_capture=False,
        original_prompt_KV_bitwise=True,
        output_tokens=128,
        original_logical_length=8,
        raw_source_kv_bytes=3200,
        representation_storage_bytes=1920,
        storage=dict(
            history_updates_per_layer=[127] * 32,
            initial_active_kv_storage_bytes=2000,
            final_active_kv_storage_bytes=2500,
            peak_active_kv_storage_bytes=2700,
        ),
        streamed_source=dict(
            original_raw_KV_bytes=3200,
            persistent_bytes=1920,
            global_source_or_query_capture=False,
            original_KV_all_word_bitwise_validation=True,
            whole_request_gpu_peak_certified=False,
            prefill_storage=dict(
                original_KV_tensor_objects_alive=0,
                original_native_layers_remaining=0,
                logical_lengths=[8] * 32,
            ),
            layer_records=records,
        ),
        independent_prefill_control=dict(
            every_layer_original_KV_bitwise=True,
            last_position_logits_bitwise=True,
            raw_GPU_control_retired_before_streamed_prefill=True,
            CPU_control_retired_before_decode=True,
        ),
        phase_wall_seconds={
            name: 0.1
            for name in (
                "media_and_processor",
                "prefill_including_layer_local_construction_and_audit",
                "post_prefill_independent_verification_not_a_decode_source",
                "retire_prefill_handle_and_CPU_verification_control",
                "decode_including_reader_construction_and_retirement",
            )
        },
    )
    return row, job


def test_row_rejects_global_capture_unretired_raw_missing_layers_bad_bytes_and_timing():
    row, job = valid_row()
    assert study.verify_row(row, job)
    changes = (
        lambda x: x.update(global_source_or_query_capture=True),
        lambda x: x.update(used_fullkv_fallback=True),
        lambda x: x["streamed_source"]["prefill_storage"].update(
            original_native_layers_remaining=1
        ),
        lambda x: x["independent_prefill_control"].update(last_position_logits_bitwise=False),
        lambda x: x["streamed_source"]["layer_records"].pop(),
        lambda x: x["streamed_source"].update(persistent_bytes=1921),
        lambda x: x["storage"].update(initial_active_kv_storage_bytes=3200),
        lambda x: x["phase_wall_seconds"].update(media_and_processor=float("nan")),
    )
    for change in changes:
        altered = copy.deepcopy(row)
        change(altered)
        with pytest.raises(ValueError):
            study.verify_row(altered, job)


def test_control_constructor_has_no_prefix_or_label_parameter_and_gate_has_no_promotion():
    assert set(inspect.signature(study.native_prefill_control).parameters) == {"adapter", "payload"}
    assert study.POLICY["guard"] == study.parent.POLICY["guard"]
    assert study.POLICY["predictions"] == 128
    assert study.POLICY["independent_quality_N"] == 0
    assert study.POLICY["global_source_and_query_capture"] is False
    assert study.POLICY["no_whole_request_peak_claim"]
    assert study.LABEL != study.parent.LABEL


def test_worker_scope_reuses_bounded_parent_and_restores_every_override(monkeypatch, tmp_path):
    original = {
        name: getattr(study.parent, name)
        for name in (
            "LABEL",
            "POLICY",
            "validate",
            "verify_row",
            "request",
            "cuda_operator_control",
        )
    }
    monkeypatch.setattr(study, "validate", lambda root: {"jobs": []})

    def parent_worker(root, index):
        assert root == tmp_path and index == 0
        assert study.parent.LABEL == study.LABEL
        assert study.parent.POLICY == study.POLICY
        assert study.parent.verify_row is study.verify_row
        assert study.parent.cuda_operator_control is study.cuda_operator_control
        assert study.parent.request is not original["request"]
        return {"synthetic": True}

    monkeypatch.setattr(study.parent, "worker", parent_worker)
    assert study.worker(tmp_path) == {"synthetic": True}
    assert all(getattr(study.parent, name) is value for name, value in original.items())


@pytest.mark.parametrize("last_guard", [True, False])
def test_final_gate_requires_saved_prefill_bits_and_complete_128_guard(
    monkeypatch, tmp_path, last_guard
):
    worker = tmp_path / "worker"
    worker.mkdir()
    numerical = dict(
        status="completed_lossless_source_diagnostic",
        source_sha="synthetic-current-clean-sha",
        all_original_prompt_KV_bitwise=True,
        all_unchanged_32_token_guards_pass=True,
        all_128_source_tokens_agree=True,
        both_initial_caches_smaller_than_raw=True,
        failures=[],
        rows=2,
        completed=2,
    )

    def numerical_analysis(root):
        (root / "GATE_lossless_source_control.json").write_text(json.dumps(numerical))
        return numerical

    monkeypatch.setattr(study.parent, "analyze", numerical_analysis)
    (worker / "GATE_cuda_operator.json").write_text(
        json.dumps({"actual_tiny_CUDA_HF_streamed_prefill": True})
    )
    rows = []
    for index in range(2):
        raw = torch.ones(1, 32, dtype=torch.bfloat16)
        torch.save({"native": raw, "streaming": raw.clone()}, worker / f"prefill_{index}.pt")
        rows.append(
            dict(
                index=index,
                status="completed",
                diagnostics={
                    name: {"registered_guard_pass": last_guard if name == "prefix_128" else True}
                    for name in (
                        "registered_guard_32",
                        "cache_dependent_guard_32",
                        "prefix_64",
                        "prefix_128",
                    )
                },
            )
        )
    (worker / "raw.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    gate = study.analyze(tmp_path)
    assert gate["passed"] is last_guard
    assert gate["all_32_and_128_guards_pass"] is last_guard
    assert not gate["superiority_established"]
    assert not gate["whole_request_KV_peak_certified"]
    assert gate["independent_quality_N"] == 0
