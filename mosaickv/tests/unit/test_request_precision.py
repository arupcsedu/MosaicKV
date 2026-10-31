from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.request_precision import online_frozen, ranked_positions


def test_deterministic_capacity_mandatory_and_ties():
    scores = torch.tensor([[1.0, 2.0, 2.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    before = scores.clone()
    assert ranked_positions(scores, 2, (3,)) == ((1, 3), (2, 3))
    assert torch.equal(scores, before)
    assert ranked_positions(scores, 4, (3,)) == ((0, 1, 2, 3),) * 2


@pytest.mark.parametrize("count", [0, 5])
def test_infeasible_capacity_rejected(count):
    with pytest.raises(ValueError):
        ranked_positions(torch.ones(2, 4), count, (3,))


def test_online_selection_uses_inputs_not_saved_membership_or_labels():
    torch.manual_seed(7)
    probabilities = tuple(torch.rand(1, 2, 40, 40).softmax(-1) for _ in range(4))
    p = SimpleNamespace(attention_weights=probabilities)
    a, proof = online_frozen(p, (0.1, 0.2, 0.3, 0.4), 0.5)
    b, _ = online_frozen(p, (0.1, 0.2, 0.3, 0.4), 0.5)
    assert a == b
    for variant in a.values():
        for count, heads in zip(
            proof["per_layer_capacity"], variant["anchor"]["positions"], strict=True
        ):
            assert all(len(h) == count and 39 in h and tuple(sorted(set(h))) == h for h in heads)
    assert proof["labels_or_reference_outputs_used"] is False
    assert proof["encoded_byte_matching"] is False


def test_no_false_encoded_byte_or_official_method_claims():
    p = SimpleNamespace(attention_weights=(torch.ones(1, 2, 40, 40) / 40,) * 4)
    _, proof = online_frozen(p, (0.1,) * 4, 0.5)
    assert "not_frozen_graph_paper_method" in proof["variant"]
    assert "not_official" in proof["profile_source"]


def test_complete_request_retires_original_owner_and_measures_complete_boundary(monkeypatch):
    import gc
    import weakref

    import mosaickv.evaluation.request_precision as module
    from mosaickv.adapters.huggingface.types import DecodeState, PrefillOutput, QueryVectors
    from mosaickv.evaluation import storage_runs

    events = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            events.append("cuda_event")

        def elapsed_time(self, other):
            return 10.0

    for name, fn in dict(
        synchronize=lambda: events.append("sync"),
        empty_cache=lambda: events.append("empty_pool"),
        memory_allocated=lambda: 100,
        reset_peak_memory_stats=lambda: events.append("reset_peaks"),
        max_memory_allocated=lambda: 200,
        max_memory_reserved=lambda: 256,
        Event=Event,
    ).items():
        monkeypatch.setattr(torch.cuda, name, fn)

    class Owner:
        pass

    references = []

    def prefill(prepared, **kwargs):
        owner = Owner()
        references.append(weakref.ref(owner))
        state = DecodeState(owner, torch.ones(1, 4), 4, 4, 4, ())
        return PrefillOutput(
            torch.zeros(1, 1, 8),
            torch.zeros(1, 1, dtype=torch.long),
            state,
            QueryVectors(()),
            (torch.ones(1, 2, 4, 4),),
        )

    def prepared(messages):
        assert "reset_peaks" in events and "media_loaded" in events
        events.append("processor")
        return SimpleNamespace(model_inputs={"pixels": torch.ones(2)}, modality_map=())

    def formats(*args):
        events.append("selection_and_construction")
        return ("owned_format",), {"no_labels": True}

    def decode(adapter, before, formats, **kwargs):
        assert before.state.past_key_values is None
        assert before.attention_weights == ()
        gc.collect()
        assert references[0]() is None
        events.append("decode")
        return torch.zeros(128, 8), dict(
            storage={"boundary": "decode_owned_only"}, output_tokens=128, used_fullkv_fallback=False
        )

    monkeypatch.setattr(module, "construct_request_formats", formats)
    monkeypatch.setattr(storage_runs, "run", decode)
    adapter = SimpleNamespace(
        prepare_inputs=prepared,
        prefill=prefill,
        _legacy_layers=lambda cache: (((torch.zeros(1, 2, 4, 128),) * 2,), None),
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2)),
        processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: "answer")),
    )

    def messages():
        events.append("media_loaded")
        return ["request"]

    logits, result = module.complete_request(
        adapter,
        messages,
        None,
        method="prefixkv_reimpl",
        precision="lossless",
        profile_path="pinned",
        ratio=0.5,
        audit=True,
    )
    assert logits.shape == (128, 8)
    assert events.index("reset_peaks") < events.index("media_loaded") < events.index("processor")
    assert events.index("selection_and_construction") < events.index("decode")
    assert result["source_KV_owner_retired_before_decode"] is True
    assert result["whole_request_peak_allocated_bytes"] == 200
    assert result["whole_request_incremental_peak_allocated_bytes"] == 100
    assert result["whole_request_KV_only_peak_certified"] is False
    assert result["whole_request_allocator_peak_certified"] is True
    assert result["scored_answer"]["answer"] == "answer"
    assert result["phase_wall_seconds"]["detokenize"] >= 0
    adapter._legacy_layers = lambda cache: (((torch.zeros(1, 2, 4, 128),) * 2,), None)
    adapter.decode_one_token = lambda token, state, **kwargs: SimpleNamespace(
        logits=torch.zeros(1, 1, 8), next_token_id=token, state=state
    )
    identity, control = module.complete_request(
        adapter,
        messages,
        None,
        method="mosaickv_exact",
        precision="query3",
        profile_path="not_accessed",
        ratio=1.0,
        audit=True,
    )
    assert identity.shape == (128, 8)
    assert control["retention_one_identity"] and control["quantization_disabled"]
    assert control["online_variant"]["no_compression"]


def test_bounded_summaries_reproduce_full_attention_selection_without_retaining_full_maps():
    from mosaickv.evaluation.request_precision import PromptQueryCapture

    torch.manual_seed(17)
    probabilities = torch.rand(1, 2, 40, 40).softmax(-1).bfloat16()

    class Attention(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings):
            return hidden_states, probabilities

    attention = Attention()
    adapter = SimpleNamespace(_language_layers=lambda: (SimpleNamespace(self_attn=attention),))
    hidden = torch.randn(1, 40, 256).bfloat16()
    phase = torch.ones(1, 40, 128).bfloat16()
    before = hidden.clone()
    with PromptQueryCapture(adapter, summaries=True) as capture:
        output = attention(hidden_states=hidden, position_embeddings=(phase, phase))
    assert torch.equal(output[0], before)
    assert capture.windows[0].untyped_storage().nbytes() < probabilities.untyped_storage().nbytes()
    assert capture.records[0]["normalized"].shape == (1, 16, 256)
    full, _ = online_frozen(SimpleNamespace(attention_weights=(probabilities,)), (0.2,), 0.5)
    small, _ = online_frozen(
        SimpleNamespace(attention_weights=(capture.windows[0],)),
        (0.2,),
        0.5,
        method="mosaickv_exact",
        priors=capture.priors,
    )
    assert full["uniform_0.5_control"] == small["uniform_0.5_control"]
    topk, _ = online_frozen(
        SimpleNamespace(attention_weights=(capture.windows[0],)),
        (0.2,),
        0.5,
        method="prompt_attention_topk",
        priors=None,
    )
    assert full["window_topk_control"] == topk["window_topk_control"]
    assert len(attention._forward_hooks) == len(attention._forward_pre_hooks) == 0


@pytest.mark.parametrize("ratio", [True, 0, -1, float("nan"), float("inf"), 1.1])
def test_invalid_request_ratio_rejected_before_any_cuda_call(ratio):
    from mosaickv.evaluation.request_precision import complete_request

    with pytest.raises(ValueError, match="source ratio"):
        complete_request(
            None,
            None,
            None,
            method="mosaickv_exact",
            precision="query3",
            profile_path="not_opened",
            ratio=ratio,
            audit=False,
        )
