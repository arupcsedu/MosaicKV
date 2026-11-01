from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mosaickv.evaluation.exact_head_bypass import ExactHeadTransport, construct
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.last_layer_tail import TailScout, order, rerank, statistics, tail_logits
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.storage_runs import cache_from_formats


@torch.inference_mode()
def test_actual_nonlinear_tail_and_scout_leave_original_replay_unchanged():
    torch.manual_seed(781)
    cfg = LlamaConfig(
        hidden_size=256,
        intermediate_size=384,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        num_hidden_layers=1,
        vocab_size=31,
    )
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).bfloat16().eval()
    adapter = SimpleNamespace(_language_layers=lambda: model.model.layers)
    block = model.model.layers[0]
    source = tuple(torch.randn(1, 2, 4, 128).bfloat16() for _ in range(2))
    selected = tuple(t[:, :, :2].contiguous() for t in source)
    fmt = ExactFormat(
        "kv", selected[0].shape, pair=tuple(PackedBF16(t.shape, True, (t,)) for t in selected)
    )
    cache = cache_from_formats(adapter, (fmt,), updates=128, audit=False, sealed=False)
    phase = torch.ones(1, 1, 128).bfloat16(), torch.zeros(1, 1, 128).bfloat16()
    logits = []
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    with TailScout(adapter, source) as scout:
        for _ in range(32):
            hidden = torch.randn(1, 1, 256).bfloat16()
            output = block(hidden, position_embeddings=phase, past_key_value=cache)
            logits.append(model.lm_head(model.model.norm(output)).reshape(1, -1).float())
            cache.layers[0].release()
    values = torch.cat(logits)
    assert len(scout.records) == 32
    assert not block._forward_hooks and not block._forward_pre_hooks
    assert not block.self_attn.q_proj._forward_hooks
    for r, value in zip(scout.records, values, strict=True):
        assert torch.equal(
            tail_logits(block, model.model.norm, model.lm_head, r["residual"], r["vector"])[0],
            value,
        )
    proof, _ = rerank(scout, model.model.norm, model.lm_head, values, values)
    assert proof["zero_tail_bitwise_identical"]
    assert proof["chosen_head"] is None and proof["fail_closed_source_anchor"]
    assert proof["no_label_access"]
    # Last-layer interventions leave all K/V unchanged at the same prefix.
    # Exact nonlinear-tail prediction must match an ACTUAL private replay.
    replacement = cache_from_formats(
        adapter,
        construct((fmt,), (source,), ((0, 1),)),
        updates=128,
        audit=False,
        sealed=False,
    )
    predicted_values = []
    for r in scout.records:
        vector = r["vector"].clone()
        vector[..., 128:] = r["full_heads"][..., 128:]
        predicted_values.append(
            tail_logits(block, model.model.norm, model.lm_head, r["residual"], vector)
        )
    with ExactHeadTransport(adapter, replacement):
        for r, predicted in zip(scout.records, predicted_values, strict=True):
            output = block(r["residual"], position_embeddings=phase, past_key_value=replacement)
            actual = model.lm_head(model.model.norm(output)).reshape(1, -1).float()
            assert torch.equal(actual, predicted)
            replacement.layers[0].release()
    assert all(torch.equal(p, weights[n]) for n, p in model.named_parameters())
    with pytest.raises(ValueError):
        rerank(scout, model.model.norm, model.lm_head, values, values, shortlist=5)


def test_registered_guard_and_deterministic_order():
    x = torch.tensor([[0.0, 2.0, 1.0], [2.0, 0.0, 1.0]])
    zero = statistics(x, x)
    assert zero["guard_pass"] and zero["mean_KL_FP32"] == 0
    changed = statistics(x, x.flip(-1))
    assert not changed["guard_pass"]
    assert order(dict(head=0, statistics=zero)) < order(dict(head=1, statistics=zero))
    with pytest.raises(ValueError):
        statistics(x, torch.zeros(1, 3))
    with pytest.raises(ValueError):
        statistics(x, x * float("nan"))
