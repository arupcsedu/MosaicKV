from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.full_layer_causal import (
    ProjectionInputCapture,
    compositions,
    ordered,
    restore,
)
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def packet(n):
    pair = tuple(torch.randn(1, 2, n, 128).bfloat16() for _ in range(2))
    return ExactFormat(
        "kv", tuple(pair[0].shape), pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in pair)
    )


def test_restore_changes_only_declared_layers_counts_real_higher_memory_and_does_not_mutate():
    base, raw = tuple(packet(17) for _ in range(4)), tuple(packet(31) for _ in range(4))
    result = restore(base, raw, (1, 3))
    assert result == (base[0], raw[1], base[2], raw[3])
    assert [p.shape[-2] for p in base] == [17] * 4
    assert backing_bytes(tuple(t for p in result for t in p.tensors())) > backing_bytes(
        tuple(t for p in base for t in p.tensors())
    )
    with pytest.raises(ValueError):
        restore(base, raw, (3, 1))
    with pytest.raises(ValueError):
        restore(base, raw, (4,))


def test_deterministic_compositions_are_model_evidence_only():
    stats = dict(
        guard_pass=False,
        token_mismatches=1,
        maximum_KL_FP32=0.01,
        maximum_logit_difference=2.0,
        mean_KL_FP32=0.001,
    )
    records = [dict(layer=i, statistics=stats) for i in reversed(range(8))]
    assert [r["layer"] for r in ordered(records)] == list(range(8))
    assert compositions(records) == ((0, 1), (0, 1, 2, 3), tuple(range(8)))
    with pytest.raises(ValueError):
        ordered([dict(r, answer="unavailable") for r in records])
    with pytest.raises(ValueError):
        ordered([*records, records[0]])


def test_full_capture_is_cpu_private_immutable_and_hooks_removed():
    class Attention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    module = Attention()
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=module)])
    x = torch.randn(1, 17, 256).bfloat16()
    original = x.clone()
    with ProjectionInputCapture(adapter) as capture:
        module(hidden_states=x)
    assert torch.equal(capture.records[0], original)
    assert capture.records[0].data_ptr() != x.data_ptr()
    assert not module._forward_pre_hooks
    assert torch.equal(x, original)
