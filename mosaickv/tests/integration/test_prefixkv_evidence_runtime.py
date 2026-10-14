"""CPU random-weight architecture checks, not pretrained quality evidence."""

import json
import re
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_hf_tiny_models import _baseline_runtime_config, _llava, _runtime_request  # noqa: E402

from mosaickv.adapters.huggingface.prompt_evidence import map_prompt_evidence  # noqa: E402
from mosaickv.backends.hf_runtime import (  # noqa: E402
    HuggingFaceMosaicKVModel,
    _pack_prefixkv_ragged,
    _state_from_packed,
    maintain_prefixkv_decode_cache,
)
from mosaickv.baselines.prefixkv import (  # noqa: E402
    build_prefixkv_reimpl_plan,
    generate_prefixkv_byte_profile,
    prefixkv_calibration_observation,
)
from mosaickv.cache_state import FullKVState  # noqa: E402
from mosaickv.config import CacheConfig, GenerationConfig, PrefixKVConfig  # noqa: E402
from mosaickv.evaluation.messages import build_multimodal_messages  # noqa: E402
from mosaickv.types import BudgetUnit, MosaicKVMethod  # noqa: E402


class WordProcessor:
    """Synthetic offset-preserving tokenizer; real mapping and runtime are exercised."""

    chat_template = "synthetic"
    is_fast = True

    @property
    def tokenizer(self):
        return self

    def apply_chat_template(self, chat, **kwargs):
        return (
            " ".join(m["role"] + ": " + " ".join(p["text"] for p in m["content"]) for m in chat)
            + " assistant:"
        )

    def __call__(self, text, **kwargs):
        text = text[0] if isinstance(text, list) else text
        offsets = [(0, 0)] + [m.span() for m in re.finditer(r"\S+", text)]
        ids = [1, *range(3, len(offsets) + 2)]
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}
        return {"input_ids": ids, "offset_mapping": offsets}

    def batch_decode(self, ids, **kwargs):
        return [" ".join(str(int(x)) for x in ids.reshape(-1))]


@pytest.fixture
def fixture(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    adapter = _llava()
    adapter.processor = WordProcessor()
    request = replace(
        _runtime_request("evidence"),
        messages=build_multimodal_messages(
            "Which word? Answer briefly.", system_prompt="Context: a b c d e f g h"
        ),
    )
    prepared = adapter.prepare_inputs(request.messages)
    mapping = map_prompt_evidence(adapter, prepared, request.messages)
    with torch.inference_mode():
        prefill = adapter.prefill(prepared, capture_attentions=True)
    observation = prefixkv_calibration_observation(
        "cal-only",
        prefill.attention_weights,
        question_positions=mapping.positions,
        question_mapping_sha256=mapping.trace()["mapping_sha256"],
    )
    profile = generate_prefixkv_byte_profile(
        (observation,),
        byte_cap=1024,
        bytes_per_layer_position=64,
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
        dataset_id="synthetic",
        dataset_revision="b" * 40,
        calibration_split="train",
        evaluation_sample_ids=(request.sample_id,),
        seed=71,
        exclude_protected_from_calibration=True,
        prompt_reservation_policy="question_and_boundaries",
        question_score_weight=0.5,
    )
    path = profile.write(tmp_path / "profile.json")
    base = _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0)
    config = replace(
        base,
        model=replace(base.model, id=profile.model_id),
        method=MosaicKVMethod.PREFIXKV_REIMPL,
        generation=GenerationConfig(max_new_tokens=16),
        cache=CacheConfig(
            1024,
            BudgetUnit.BYTES,
            0.99,
            1,
            physical_budget_mandatory_policy="question_and_boundaries",
        ),
        prefixkv=PrefixKVConfig(
            enabled=True,
            profile_path=str(path),
            rescale_offline_profile_to_byte_budget=True,
            profile_reference_retention_ratio=profile.target_retention_ratio,
            hf_cache_transport="ragged_llama_eager_v1",
            decode_boundary_policy="protected",
            decode_policy="effective_profile_ratio",
            exclude_protected_from_calibration=True,
            question_score_weight=0.5,
        ),
    )
    yield adapter, request, prepared, prefill, mapping, profile, config
    torch.set_num_threads(previous)


def test_question_reservation_survives_64_decode_evictions_and_cache_isolation(fixture):
    adapter, _, prepared, prefill, mapping, profile, cfg = fixture
    snapshot = adapter.extract_past_key_values(prefill.state.past_key_values)
    saved = tuple((x.key.clone(), x.value.clone()) for x in snapshot.layers)
    mandatory = tuple(sorted({0, prepared.logical_sequence_length - 1, *mapping.positions}))
    full = FullKVState.from_cache_snapshot(
        snapshot,
        block_size=1,
        mandatory_logical_positions=mandatory,
        modality_spans=prepared.modality_map,
        token_ids=prepared.model_inputs["input_ids"],
    )
    plan = build_prefixkv_reimpl_plan(
        full,
        prefill.attention_weights,
        cfg.prefixkv,
        cfg.cache,
        model_id=cfg.model.id,
        model_revision=cfg.model.revision,
        profile=profile,
        question_positions=mapping.positions,
        physical_prompt_capacity=8,
    )
    packed = _pack_prefixkv_ragged(adapter, plan, prefill.state.past_key_values)
    assert plan.retained_bytes == 1024
    assert (
        sum(t.untyped_storage().nbytes() for x in packed.snapshot.layers for t in (x.key, x.value))
        == 1024
    )
    state = _state_from_packed(adapter, prefill.state, packed)
    token = prefill.next_token_id
    evictions = 0
    with torch.inference_mode():
        for step in range(1, 65):
            out = adapter.decode_one_token(token, state, capture_queries=False)
            state, events = maintain_prefixkv_decode_cache(
                adapter, out.state, plan, step_index=step
            )
            evictions += sum(e["removed_logical_position"] is not None for e in events)
            for positions in state.model_state["prefixkv_logical_positions"]:
                assert set(mandatory) <= set(positions)
            assert state.next_decode_position == prepared.logical_sequence_length + step
            assert torch.isfinite(out.logits).all()
            token = out.next_token_id
    assert evictions > 0
    for source, (key, value) in zip(snapshot.layers, saved, strict=True):
        assert torch.equal(source.key, key) and torch.equal(source.value, value)


def test_evidence_runtime_trace_and_retention_one_fullkv_identity(fixture, tmp_path):
    adapter, request, _, _, mapping, _, config = fixture
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path / "compressed")
    output = runtime.generate(request)
    assert isinstance(output.answer, str)
    trace = json.loads(next((tmp_path / "compressed" / request.run_id).glob("*.json")).read_text())
    assert trace["status"] == "completed"
    assert trace["prompt_evidence"]["positions"] == list(mapping.positions)
    assert trace["prefixkv"]["retained_bytes"] == 1024
    assert trace["selection_provenance"] == "prefixkv_evidence_ablation_v1"
    assert "question_protected" in trace["effective_method"]
    full = replace(
        config,
        method=MosaicKVMethod.FULL_KV,
        prefixkv=PrefixKVConfig(),
        cache=replace(config.cache, retention_ratio=1.0),
    )
    identity = replace(config, cache=replace(config.cache, retention_ratio=1.0))
    baseline = HuggingFaceMosaicKVModel(adapter, full, trace_directory=tmp_path / "full")
    candidate = HuggingFaceMosaicKVModel(adapter, identity, trace_directory=tmp_path / "identity")
    assert baseline.generate(request).answer == candidate.generate(request).answer
    traces = [
        json.loads(next((tmp_path / name / request.run_id).glob("*.json")).read_text())
        for name in ("full", "identity")
    ]
    assert traces[0]["generated_token_ids"] == traces[1]["generated_token_ids"]
    assert len(traces[0]["generated_token_ids"]) == 16


def test_question_scored_append_only_64_steps_preserve_growth_and_label(fixture):
    adapter, _, prepared, prefill, mapping, profile, cfg = fixture
    cfg = replace(cfg, prefixkv=replace(cfg.prefixkv, decode_policy="append_only"))
    snapshot = adapter.extract_past_key_values(prefill.state.past_key_values)
    mandatory = tuple(sorted({0, prepared.logical_sequence_length - 1, *mapping.positions}))
    full = FullKVState.from_cache_snapshot(
        snapshot,
        block_size=1,
        mandatory_logical_positions=mandatory,
        modality_spans=prepared.modality_map,
        token_ids=prepared.model_inputs["input_ids"],
    )
    plan = build_prefixkv_reimpl_plan(
        full,
        prefill.attention_weights,
        cfg.prefixkv,
        cfg.cache,
        model_id=cfg.model.id,
        model_revision=cfg.model.revision,
        profile=profile,
        question_positions=mapping.positions,
        physical_prompt_capacity=8,
    )
    assert plan.implementation_label.endswith("question_scored_selection_only_reimpl")
    packed = _pack_prefixkv_ragged(adapter, plan, prefill.state.past_key_values)
    state = _state_from_packed(adapter, prefill.state, packed)
    width = sum(
        2 * x.key.shape[0] * x.key.shape[1] * x.key.shape[3] * x.key.element_size()
        for x in snapshot.layers
    )
    token = prefill.next_token_id
    with torch.inference_mode():
        for step in range(1, 65):
            out = adapter.decode_one_token(token, state, capture_queries=False)
            state, events = maintain_prefixkv_decode_cache(
                adapter, out.state, plan, step_index=step
            )
            assert not any(e.get("removed_logical_position") is not None for e in events)
            cache = adapter.extract_past_key_values(state.past_key_values)
            actual = sum(
                t.untyped_storage().nbytes() for x in cache.layers for t in (x.key, x.value)
            )
            assert actual == 1024 + step * width
            assert state.next_decode_position == prepared.logical_sequence_length + step
            assert torch.isfinite(out.logits).all()
            token = out.next_token_id


@pytest.mark.parametrize(
    "method", [MosaicKVMethod.PROMPT_ATTENTION_TOPK, MosaicKVMethod.MOSAICKV_EXACT]
)
def test_same_question_policy_reaches_topk_and_mosaic_cache(fixture, monkeypatch, tmp_path, method):
    from mosaickv.backends import hf_runtime

    adapter, request, prepared, _, mapping, _, config = fixture
    config = replace(
        config,
        method=method,
        prefixkv=PrefixKVConfig(),
        forecasting=replace(config.forecasting, enabled=method.is_mosaickv),
        graph=replace(config.graph, enabled=method.is_mosaickv),
        selection=replace(config.selection, enabled=method.is_mosaickv),
    )
    seen = []
    original = hf_runtime._state_from_packed
    mandatory = {0, prepared.logical_sequence_length - 1, *mapping.positions}

    def inspect(adapter, state, packed):
        by_head = {}
        for slot in packed.slot_records:
            by_head.setdefault((slot["layer"], slot["kv_head"]), set()).add(
                slot["logical_position"]
            )
        assert by_head
        assert all(mandatory <= positions for positions in by_head.values())
        assert (
            sum(
                t.untyped_storage().nbytes()
                for layer in packed.snapshot.layers
                for t in (layer.key, layer.value)
            )
            == 1024
        )
        seen.append(True)
        return original(adapter, state, packed)

    monkeypatch.setattr(hf_runtime, "_state_from_packed", inspect)
    output = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path).generate(request)
    assert isinstance(output.answer, str) and seen
