from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import fused_model_guard as guard
from mosaickv.evaluation.fused_lossless_attention import bits_equal, reference


def fixture():
    modules = [SimpleNamespace(self_attn=object()) for _ in range(32)]
    adapter = SimpleNamespace(_language_layers=lambda: modules)
    torch.manual_seed(5)
    q = torch.randn(1, 32, 1, 128).bfloat16()
    k, v = (torch.randn(1, 32, 9, 128).bfloat16() for _ in range(2))

    def parent(module, query, key, value, mask, scale, dropout=0.0, **kw):
        weights = torch.softmax(
            (query @ key.transpose(2, 3)) * scale, dim=-1, dtype=torch.float32
        ).bfloat16()
        value.copy_(v)
        return (weights @ value).transpose(1, 2).contiguous(), weights

    return adapter, modules, q, k, v, parent


@pytest.mark.parametrize("mode", ["clone_repack", "fused"])
def test_all_32_real_prefixes_required_and_candidate_output_is_substituted(monkeypatch, mode):
    adapter, modules, q, k, _v, parent = fixture()
    monkeypatch.setattr(guard, "run", reference)  # CPU orchestration test, not GPU evidence.
    diagnostic = guard.Guard(adapter, mode=mode)
    operation = diagnostic.wrap(parent)
    assert not diagnostic.report()["complete_32_prefix_guard"]
    for _ in range(32):
        for layer in (0, 16, 31):
            lease = k.clone()
            out, weights = operation(modules[layer].self_attn, q, lease, lease, None, 128**-0.5)
            expected_lease = k.clone()
            expected, expected_w = parent(None, q, expected_lease, expected_lease, None, 128**-0.5)
            assert bits_equal(out, expected) and bits_equal(weights, expected_w)
    report = diagnostic.report()
    assert report["complete_32_prefix_guard"] and len(report["records"]) == 96
    assert report["functional_substitution_not_native_or_production_backend"]


def test_numerical_failure_is_not_a_parent_fallback_and_saves_first_failure(monkeypatch):
    adapter, modules, q, k, _v, parent = fixture()

    def wrong(*args, **kwargs):
        out, weights = reference(*args, **kwargs)
        return out + 1, weights

    monkeypatch.setattr(guard, "run", wrong)
    diagnostic = guard.Guard(adapter, mode="fused")
    lease = k.clone()
    with pytest.raises(guard.GuardFailure, match="layer=0, prefix=0"):
        diagnostic.wrap(parent)(modules[0].self_attn, q, lease, lease, None, 128**-0.5)
    assert diagnostic.failed_tensors is not None
    assert not diagnostic.report()["complete_32_prefix_guard"]
    assert not diagnostic.records[0]["attention_output_bits_exact"]


def test_word_comparator_distinguishes_signed_zero_and_preserves_nan_payload():
    assert torch.equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    special = torch.tensor([32705, -32768], dtype=torch.int16).view(torch.bfloat16)
    assert bits_equal(special, special.clone())
    assert not bits_equal(special, special.float())


def test_no_label_api_or_unregistered_horizon():
    adapter, *_ = fixture()
    with pytest.raises(ValueError):
        guard.Guard(adapter, mode="fused", horizon=4)
    with pytest.raises(TypeError):
        guard.Guard(adapter, mode="fused", reference="not permitted")


def test_nested_guard_rejected_and_attention_entry_restored():
    from mosaickv.evaluation.sequential_kv_lease import SequentialAttentionTransport

    adapter, *_ = fixture()
    original = SequentialAttentionTransport.__enter__
    with (
        guard.Guard(adapter, mode="clone_repack").use(),
        pytest.raises(ValueError, match="nested"),
        guard.Guard(adapter, mode="fused").use(),
    ):
        pass
    assert SequentialAttentionTransport.__enter__ is original
