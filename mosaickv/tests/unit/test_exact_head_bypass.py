from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from mosaickv.evaluation.exact_head_bypass import ExactHeadTransport, construct, rank_routes
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.storage_runs import cache_from_formats


def fixture():
    torch.manual_seed(414)
    module = (
        LlamaAttention(
            LlamaConfig(
                hidden_size=256, num_attention_heads=2, num_key_value_heads=2, head_dim=128
            ),
            0,
        )
        .bfloat16()
        .eval()
    )
    module.config._attn_implementation = "eager"
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=module)])
    source = tuple(torch.randn(1, 2, 4, 128).bfloat16() for _ in range(2))
    kept = tuple(t[:, :, :2].contiguous().clone() for t in source)
    base = ExactFormat(
        "kv", kept[0].shape, pair=tuple(PackedBF16(t.shape, True, (t,)) for t in kept)
    )
    return module, adapter, source, base


def test_private_decode_forks_and_history_reservation_are_counted():
    _, adapter, source, base = fixture()
    (seed,) = construct((base,), (source,), ((0, 1),))
    first = cache_from_formats(adapter, (seed,), updates=33, audit=False, sealed=False)
    second = cache_from_formats(adapter, (seed,), updates=128, audit=False, sealed=False)
    a, b = (c.layers[0].representation for c in (first, second))
    assert a.sidecar[0].data_ptr() != b.sidecar[0].data_ptr()
    assert a.sidecar[0].data_ptr() != seed.prompt[0].data_ptr()
    assert a.sidecar[0].shape == (1, 1, 132, 128)
    assert torch.equal(a.sidecar[0][:, :, :4], source[0][:, 1:2])
    assert first.initial_bytes == second.initial_bytes == base.component_bytes + 2 * 132 * 128 * 2
    a.sidecar[0][:, :, 4:] = 1
    assert not b.sidecar[0][:, :, 4:].any()
    assert seed.prompt[0].shape[-2] == 4
    expanded = a.sidecar[0][:, :, :5].expand(1, 2, -1, -1)
    assert expanded.stride(1) == 0
    assert expanded.untyped_storage().data_ptr() == a.sidecar[0].untyped_storage().data_ptr()
    assert expanded.untyped_storage().nbytes() == a.sidecar[0].untyped_storage().nbytes()


@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_actual_query_full_head_context_and_other_head_isolation(enabled):
    module, adapter, source, base = fixture()
    original = tuple(t.clone() for t in source)
    original_weights = {name: p.detach().clone() for name, p in module.named_parameters()}
    seeds = construct((base,), (source,), ((0, 1),), enabled=enabled)
    cache = cache_from_formats(adapter, seeds, updates=128, audit=False, sealed=False)
    phase = (torch.ones(1, 1, 128).bfloat16(), torch.zeros(1, 1, 128).bfloat16())
    with ExactHeadTransport(adapter, cache) as transport:
        for step in range(2):
            hidden = torch.randn(1, 1, 256).bfloat16()
            actual = module(
                hidden, position_embeddings=phase, attention_mask=None, past_key_value=cache
            )[0]
            query = (
                torch.nn.functional.linear(hidden, module.q_proj.weight, module.q_proj.bias)
                .view(1, 1, 2, 128)
                .transpose(1, 2)
            )
            query, _ = apply_rotary_pos_emb(query, query, *phase)
            key, value = cache.layers[0].leased
            output, _ = eager_attention_forward(
                module, query, key, value, None, scaling=module.scaling
            )
            vector = output.reshape(1, 1, 256)
            if enabled:
                sk, sv = cache.layers[0].representation.sidecar
                result, _ = eager_attention_forward(
                    module,
                    query[:, 1:2],
                    sk[:, :, : 5 + step],
                    sv[:, :, : 5 + step],
                    None,
                    scaling=module.scaling,
                )
                vector[..., 128:] = result.reshape(1, 1, 128)
            expected = torch.nn.functional.linear(vector, module.o_proj.weight, module.o_proj.bias)
            assert torch.equal(actual, expected)
            cache.layers[0].release()
            cache.storage_record()
    assert transport.recomputed_heads == (2 if enabled else 0)
    assert all(torch.equal(t, r) for t, r in zip(source, original, strict=True))
    assert all(torch.equal(p, original_weights[name]) for name, p in module.named_parameters())
    assert not module._forward_hooks and not module._forward_pre_hooks
    assert not module.q_proj._forward_hooks and not module.o_proj._forward_pre_hooks


def test_foreign_request_rejected_and_hooks_cleaned_after_exception():
    module, adapter, source, base = fixture()
    cache = cache_from_formats(
        adapter, construct((base,), (source,), ((0, 1),)), updates=128, audit=False, sealed=False
    )
    with pytest.raises(RuntimeError, match="foreign"), ExactHeadTransport(adapter, cache):
        module(
            torch.zeros(1, 1, 256).bfloat16(),
            position_embeddings=(torch.ones(1, 1, 128),) * 2,
            attention_mask=None,
            past_key_value=object(),
        )
    assert not module._forward_hooks and not module._forward_pre_hooks
    assert not module.q_proj._forward_hooks and not module.o_proj._forward_pre_hooks


def test_route_ranking_is_deterministic_magnitude_preserving_and_label_blind():
    module, _, source, base = fixture()
    p = torch.tensor([0.1, 0.1, 0.4, 0.4]).repeat(2, 1)[None, :, None]
    steps = tuple((torch.cat((p, torch.zeros(1, 2, 1, s + 1)), dim=-1),) for s in range(4))
    args = (steps, (source,), (base,), (((0, 1), (0, 1)),), (module,))
    a, b = rank_routes(*args), rank_routes(*args)
    assert a == b and len(a) == 2
    assert a[0]["score"] > 0 and a[0]["score"] >= a[1]["score"]
    with pytest.raises(ValueError):
        construct((base,), (source,), ((0, 1), (0, 1)))
