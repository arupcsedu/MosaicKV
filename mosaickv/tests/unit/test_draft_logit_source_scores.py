"""Random-architecture CPU feasibility, never pretrained quality evidence."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import (
    CLIPVisionConfig,
    DynamicCache,
    LlamaConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
)

from mosaickv.adapters.huggingface.llava import Llava15Adapter
from mosaickv.adapters.huggingface.types import DecodeState, PreparedInputs
from mosaickv.evaluation.draft_logit_source_scores import _loss, draft_row_scores
from mosaickv.evaluation.native_quantized_source_control import native_quantized_fill
from mosaickv.evaluation.phase_capacity import PhaseCapture
from mosaickv.evaluation.shared_source_format import SourceCapture, construct


def fixture():
    torch.manual_seed(22311)
    text = LlamaConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=8,
        patch_size=4,
    )
    config = LlavaConfig(
        text_config=text.to_dict(), vision_config=vision.to_dict(), image_token_index=127
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    model = LlavaForConditionalGeneration(config).bfloat16().eval()
    adapter = Llava15Adapter(model, SimpleNamespace(), device=torch.device("cpu"))
    n = 17
    prepared = PreparedInputs(
        {
            "input_ids": torch.randint(0, 126, (1, n)),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
        },
        (),
        n,
    )
    with SourceCapture(adapter) as source, PhaseCapture(adapter) as phase:
        prefill = adapter.prefill(prepared, capture_queries=False)
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    pairs = tuple(tuple(pair) for pair in pairs)
    identity = (
        torch.arange(n, dtype=torch.int32),
        torch.arange(n, dtype=torch.int16)[None].expand(2, n),
    )
    packets = tuple(
        construct(
            source.records[layer],
            pair,
            (tuple(range(n)),) * 2,
            (0, 1, 15, 16),
            phase.phases,
            identity,
        )
        for layer, pair in enumerate(pairs)
    )
    return adapter, prefill, packets, pairs


def ordinary_logits(adapter, prefill, pairs, tokens):
    cache = DynamicCache()
    with torch.inference_mode(False):
        for layer, pair in enumerate(pairs):
            cache.update(*(t.detach().clone() for t in pair), layer)
        n = pairs[0][0].shape[2]
        state = DecodeState(
            cache,
            torch.ones(1, n, dtype=torch.long),
            n,
            n,
            n,
            (),
            {"mosaickv_cache_all_valid": True},
        )
        values = []
        for token in tokens:
            output = adapter.decode_one_token(torch.tensor([[token]]), state, capture_queries=False)
            values.append(output.logits.detach().float())
    return torch.cat(values, 0)


def digest(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def test_native_grad_logits_equal_ordinary_inference_and_all_original_state_is_preserved():
    adapter, prefill, packets, pairs = fixture()
    snapshots = tuple(t.clone() for pair in pairs for t in pair)
    weights = tuple(adapter.model.parameters())
    weight_snapshots = tuple(p.detach().clone() for p in weights)
    flags = tuple(p.requires_grad for p in weights)
    weights[0].grad = torch.ones_like(weights[0])
    grad_before = weights[0].grad.clone()
    cache_identity = id(prefill.state.past_key_values)
    metadata = dict(prefill.state.model_state)
    # Must work even when invoked by an inference-mode constructor.
    with (
        patch.object(
            adapter, "prefill", side_effect=AssertionError("A second prefill is forbidden")
        ),
        torch.inference_mode(),
    ):
        scores, report = draft_row_scores(adapter, prefill, packets, pairs)
    assert report["horizon"] == 4
    assert report["no_second_prefill"] and report["original_cache_unchanged"]
    assert report["model_parameter_gradients_unchanged"]
    assert report["positive_row_gains"] > 0
    assert report["nonzero_gradient_scalars"] > 0
    assert report["autograd_saved_nonweight_peak_backing_bytes"] > 0
    assert set(scores) == set(range(2))
    assert all(
        s.shape == (17,) and s.device.type == "cpu" and not s.requires_grad and s.grad_fn is None
        for s in scores.values()
    )
    json.dumps(report, allow_nan=False)
    assert id(prefill.state.past_key_values) == cache_identity
    assert prefill.state.past_key_values.get_seq_length() == 17
    assert prefill.state.model_state == metadata
    assert all(
        torch.equal(a, b)
        for a, b in zip(snapshots, (t for pair in pairs for t in pair), strict=True)
    )
    assert all(torch.equal(a, b) for a, b in zip(weight_snapshots, weights, strict=True))
    assert tuple(p.requires_grad for p in weights) == flags
    assert torch.equal(weights[0].grad, grad_before)
    assert all(p.grad is None for p in weights[1:])
    expected_ref = ordinary_logits(adapter, prefill, pairs, report["draft_input_token_ids"])
    assert digest(expected_ref) == report["reference_logits_sha256"]
    assert len(report["reference_logits_fp32"]) == len(report["parent_logits_fp32"]) == 4
    assert digest(torch.tensor(report["reference_logits_fp32"], dtype=torch.float32)) == report[
        "reference_logits_sha256"
    ]
    assert digest(torch.tensor(report["parent_logits_fp32"], dtype=torch.float32)) == report[
        "parent_logits_sha256"
    ]
    parent = []
    for packet, block in zip(packets, adapter._language_layers(), strict=True):
        targets = tuple(torch.empty(packet.shape, dtype=torch.bfloat16) for _ in range(2))
        native_quantized_fill(packet, block.self_attn, targets, direct=True)
        parent.append(targets)
    expected_parent = ordinary_logits(adapter, prefill, parent, report["draft_input_token_ids"])
    assert digest(expected_parent) == report["parent_logits_sha256"]


def test_decision_margin_and_KL_direction_have_correct_first_order_sign():
    reference = torch.tensor([[1.5, 0.2, -0.3], [0.0, 1.1, 0.2]], dtype=torch.float64)
    candidate = torch.tensor(
        [[0.8, 0.4, -0.1], [0.2, 0.7, 0.3]], dtype=torch.float64, requires_grad=True
    )
    loss, *_ = _loss(reference, candidate)
    (gradient,) = torch.autograd.grad(loss, (candidate,))
    delta = reference - candidate.detach()
    predicted_gain = -(gradient * delta).sum()
    assert predicted_gain > 0
    epsilon = 1e-5
    actual_delta = _loss(reference, candidate.detach() + epsilon * delta)[0] - loss.detach()
    assert torch.allclose(actual_delta / epsilon, (gradient * delta).sum(), rtol=1e-4, atol=1e-4)


def test_runtime_failure_rolls_back_without_changing_prefill_or_parameter_flags():
    adapter, prefill, packets, pairs = fixture()
    old_flags = tuple(p.requires_grad for p in adapter.model.parameters())
    saved = tuple(t.clone() for pair in pairs for t in pair)
    ordinary = adapter._decode_forward
    calls = 0

    def fail_in_grad(*args, **kwargs):
        nonlocal calls
        calls += 1
        if torch.is_grad_enabled():
            raise RuntimeError("Injected source-gradient failure")
        return ordinary(*args, **kwargs)

    with (
        patch.object(adapter, "_decode_forward", fail_in_grad),
        pytest.raises(RuntimeError, match="Injected"),
    ):
        draft_row_scores(adapter, prefill, packets, pairs)
    assert calls == 5
    assert prefill.state.past_key_values.get_seq_length() == 17
    assert tuple(p.requires_grad for p in adapter.model.parameters()) == old_flags
    assert all(p.grad is None for p in adapter.model.parameters())
    assert all(
        torch.equal(a, b) for a, b in zip(saved, (t for pair in pairs for t in pair), strict=True)
    )


@pytest.mark.parametrize("horizon", [0, 1, 3, 5, 128, True])
def test_horizon_is_fixed_and_cannot_be_changed_into_an_oracle(horizon):
    adapter, prefill, packets, pairs = fixture()
    with pytest.raises(ValueError, match="exactly four"):
        draft_row_scores(adapter, prefill, packets, pairs, horizon=horizon)


def test_active_hooks_training_logical_mismatch_and_inference_weights_fail_explicitly():
    adapter, prefill, packets, pairs = fixture()
    hook = adapter.model.register_forward_hook(lambda *_args: None)
    try:
        with pytest.raises(ValueError, match="hooks"):
            draft_row_scores(adapter, prefill, packets, pairs)
    finally:
        hook.remove()
    adapter.model.train()
    with pytest.raises(ValueError, match="Eval-only"):
        draft_row_scores(adapter, prefill, packets, pairs)
    adapter.model.eval()
    prefill.state.next_decode_position += 1
    with pytest.raises(ValueError, match="untouched"):
        draft_row_scores(adapter, prefill, packets, pairs)
    prefill.state.next_decode_position -= 1
    target = adapter.model.lm_head
    original_weight = target.weight
    with torch.inference_mode():
        target.weight = torch.nn.Parameter(original_weight.detach().clone(), requires_grad=False)
    try:
        with pytest.raises(NotImplementedError, match="Inference model weights"):
            draft_row_scores(adapter, prefill, packets, pairs)
    finally:
        target.weight = original_weight


def test_no_answer_or_long_horizon_reference_input_is_supported():
    adapter, prefill, packets, pairs = fixture()
    with pytest.raises(TypeError, match="unexpected keyword"):
        draft_row_scores(adapter, prefill, packets, pairs, oracle_logits=torch.zeros(128, 128))
