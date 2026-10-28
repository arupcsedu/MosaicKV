"""Arithmetic diagnosis preserves stored entries, scope, GQA and error cleanup."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.fox_attention_precision import attention_precision, fp32_eager_attention


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fp32_operation_matches_manual_reference_without_mutating_cache(groups, dtype):
    torch.manual_seed(3)
    q = torch.randn(1, 2 * groups, 2, 4).to(dtype)
    k = torch.randn(1, 2, 6, 4).to(dtype)
    v = torch.randn_like(k)
    saved = k.clone(), v.clone()
    mask = torch.zeros(1, 2 * groups, 2, 7, dtype=dtype)
    mask[..., 2] = -torch.inf
    owner = SimpleNamespace(training=False, num_key_value_groups=groups)
    out, p = fp32_eager_attention(owner, q, k, v, mask, scaling=0.5)
    expanded_k = k.float().repeat_interleave(groups, dim=1)
    expanded_v = v.float().repeat_interleave(groups, dim=1)
    expected = torch.softmax(q.float() @ expanded_k.transpose(2, 3) * 0.5 + mask[..., :6], -1)
    expected_out = (expected @ expanded_v).transpose(1, 2).contiguous().to(dtype)
    assert torch.equal(out, expected_out) and torch.equal(p, expected.to(dtype))
    assert torch.equal(k, saved[0]) and torch.equal(v, saved[1])
    assert out.dtype == dtype and p.dtype == dtype and bool((p[..., 2] == 0).all())


def test_unsupported_geometry_and_training_fail_explicitly():
    x = torch.ones(1, 1, 2, 4)
    owner = SimpleNamespace(training=False, num_key_value_groups=1)
    with pytest.raises(ValueError, match="mask"):
        fp32_eager_attention(owner, x, x, x, torch.ones(1, 2, dtype=torch.bool), 0.5)
    with pytest.raises(ValueError, match="dtype"):
        fp32_eager_attention(owner, x.double(), x.double(), x.double(), None, 0.5)
    with pytest.raises(ValueError, match="dropout"):
        fp32_eager_attention(owner, x, x, x, None, 0.5, dropout=0.1)
    owner.num_key_value_groups = 2
    with pytest.raises(ValueError, match="geometry"):
        fp32_eager_attention(owner, x, x, x, None, 0.5)


@pytest.mark.parametrize("family", ["llama", "mistral"])
def test_context_is_owner_scoped_and_restored_after_exception(family):
    import importlib

    from transformers import LlamaConfig, MistralConfig

    config_class = LlamaConfig if family == "llama" else MistralConfig
    attention_class = "LlamaAttention" if family == "llama" else "MistralAttention"
    modeling_llama = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    config = config_class(hidden_size=8, num_attention_heads=2, num_key_value_heads=1)
    config._attn_implementation = "eager"
    owner = getattr(modeling_llama, attention_class)(config, layer_idx=0).half().eval()
    other = getattr(modeling_llama, attention_class)(config, layer_idx=0).half().eval()
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=owner)])
    original = modeling_llama.eager_attention_forward
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    weights = {name: p.detach().clone() for name, p in owner.named_parameters()}
    q = torch.ones(1, 2, 1, 4, dtype=torch.float16)
    kv = torch.ones(1, 1, 2, 4, dtype=torch.float16)
    with (
        pytest.raises(RuntimeError, match="intentional"),
        attention_precision(adapter, "fp32_attention") as metadata,
    ):
        assert modeling_llama.eager_attention_forward is not original
        assert metadata["policy"] == "fp32_attention"
        assert metadata["source_functions"]
        assert torch.equal(
            modeling_llama.eager_attention_forward(other, q, kv, kv, None, scaling=0.5)[0],
            original(other, q, kv, kv, None, scaling=0.5)[0],
        )
        with (
            pytest.raises(RuntimeError, match="nested"),
            attention_precision(adapter, "native_fp16"),
        ):
            pass
        raise RuntimeError("intentional")
    assert modeling_llama.eager_attention_forward is original
    assert torch.backends.cuda.matmul.allow_tf32 == previous_tf32
    assert all(torch.equal(p, weights[name]) for name, p in owner.named_parameters())
    with attention_precision(adapter, "native_fp16"):
        assert modeling_llama.eager_attention_forward is original
