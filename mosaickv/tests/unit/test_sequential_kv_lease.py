from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama import modeling_llama as llama

from mosaickv.evaluation.exact_storage import ExactFormat, encode
from mosaickv.evaluation.head_local_precision import localize
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import construct as precision
from mosaickv.evaluation.sequential_kv_lease import (
    SequentialAttentionTransport,
    SequentialLayer,
    fill_component,
)
from mosaickv.evaluation.shared_source_format import construct as source
from mosaickv.evaluation.storage_runs import cache_from_formats
from mosaickv.evaluation.text_source_storage import optimize


def example():
    torch.manual_seed(20411)
    h, n, d = 4, 64, 128
    cfg = LlamaConfig(
        hidden_size=h * d,
        intermediate_size=1024,
        num_attention_heads=h,
        num_key_value_heads=h,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    attention = llama.LlamaAttention(cfg, layer_idx=0).bfloat16().eval()
    x = torch.randn(1, n, h * d).bfloat16()
    angle = torch.randn(1, n, d // 2).repeat(1, 1, 2)
    phases = angle.cos().bfloat16(), angle.sin().bfloat16()
    with torch.inference_mode():
        k = attention.k_proj(x).reshape(1, n, h, d).transpose(1, 2)
        v = attention.v_proj(x).reshape(1, n, h, d).transpose(1, 2)
        k = k * phases[0][:, None] + torch.cat((-k[..., 64:], k[..., :64]), -1) * phases[1][:, None]
    pair = k.contiguous(), v.contiguous()
    heads = (tuple(range(n)),) * h
    text = tuple((*range(3), *range(48, 64)))
    fmt = localize(
        precision(pair, bits=3, encoded_exact=True),
        pair,
        heads,
        set(text),
        policy="head3_boundary8",
    )
    record = dict(
        normalized=x[:, text].contiguous(),
        cos=phases[0][:, text].contiguous(),
        sin=phases[1][:, text].contiguous(),
    )
    textfmt, proof = optimize(fmt, record, heads, text, attention)
    assert proof["source_used"]
    src = source(
        x,
        pair,
        heads,
        text,
        phases,
        (torch.arange(n, dtype=torch.int32), torch.arange(n, dtype=torch.int16)[None].expand(h, n)),
    )
    raw = ExactFormat("kv", tuple(k.shape), pair=tuple(encode(t) for t in pair))
    return attention, (raw, fmt, textfmt, src)


@pytest.mark.parametrize("index", range(4))
def test_each_component_preserves_bits_and_history_for_common_formats(index):
    attention, packets = example()
    fmt = packets[index]
    original = fmt.materialize(attention)
    for component in (0, 1):
        lease = torch.full((1, 4, 69, 128), -71).bfloat16()
        fill_component(fmt, attention, component, lease)
        assert torch.equal(
            lease[..., :64, :].view(torch.int16), original[component].view(torch.int16)
        )
        assert bool((lease[..., 64:, :] == -71).all())


def test_single_backing_native_math_and_private_state():
    attention, packets = example()
    fmt = packets[0]

    def observe(_extras=()):
        return None

    layer = SequentialLayer(fmt, attention, observe=observe, direct=True, updates=128, audit=False)
    k, v = (torch.randn(1, 4, 1, 128).bfloat16() for _ in range(2))
    raw = fmt.materialize(attention)
    q = torch.randn(1, 4, 1, 128).bfloat16()
    expected = llama.eager_attention_forward(
        attention, q, torch.cat((raw[0], k), 2), torch.cat((raw[1], v), 2), None, attention.scaling
    )
    key, value = layer.update(k, v)
    assert key.data_ptr() == value.data_ptr()
    assert layer.lease_bytes == key.numel() * key.element_size()
    assert backing_bytes((key, value)) == layer.lease_bytes
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    cache = SimpleNamespace(layers=[layer])
    original = llama.eager_attention_forward
    with torch.inference_mode(), SequentialAttentionTransport(adapter, cache):
        actual = llama.eager_attention_forward(attention, q, key, value, None, attention.scaling)
    assert llama.eager_attention_forward is original
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert layer.current_component == 1 and layer.component_fills == 2
    assert torch.equal(key, torch.cat((raw[1], v), 2))
    layer.release()
    assert not layer.leased and layer.history_length == 1


def test_rejects_unsupported_history_and_foreign_cache_and_restores_operator():
    attention, packets = example()
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    with pytest.raises(ValueError, match="raw append-only"):
        cache_from_formats(adapter, (packets[0],), updates=32, audit=False, sequential_lease=True)
    cache = cache_from_formats(
        adapter, (packets[0],), updates=32, audit=False, sealed=False, sequential_lease=True
    )
    original = llama.eager_attention_forward
    q = torch.randn(1, 4, 1, 128).bfloat16()
    with (
        pytest.raises(RuntimeError, match="isolated"),
        torch.inference_mode(),
        SequentialAttentionTransport(adapter, cache),
    ):
        llama.eager_attention_forward(attention, q, q, q, None, attention.scaling)
    assert llama.eager_attention_forward is original


def test_component_kernel_specialization_is_explicit_and_default_is_unchanged():
    from mosaickv.evaluation.encoded_projection_kernel import _project as text_project
    from mosaickv.evaluation.shared_source_kernel import _project as source_project

    for kernel in (text_project, source_project):
        assert next(p for p in kernel.params if p.name == "COMPONENT").is_constexpr
        assert kernel.signature.parameters["COMPONENT"].default == -1
