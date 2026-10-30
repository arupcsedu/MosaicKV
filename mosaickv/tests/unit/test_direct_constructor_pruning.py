from types import SimpleNamespace

import pytest
import torch

import mosaickv.evaluation.direct_constructor as module


@pytest.mark.parametrize("zero_cost", [True, False])
def test_other_policy_is_pruned_only_at_proven_zero_correction_lower_bound(monkeypatch, zero_cost):
    key, value = tuple(torch.randn(1, 2, 4, 4).bfloat16() for _ in range(2))
    record = dict(
        normalized=torch.randn(1, 4, 8).bfloat16(),
        cos=torch.ones(1, 4, 4, dtype=torch.bfloat16),
        sin=torch.zeros(1, 4, 4, dtype=torch.bfloat16),
    )
    adapter = SimpleNamespace(
        _legacy_layers=lambda _: (((key, value),), None),
        _language_layers=lambda: [SimpleNamespace(self_attn=None)],
    )
    prefill = SimpleNamespace(state=SimpleNamespace(past_key_values=None))
    capture = SimpleNamespace(records={0: record})
    policies = []

    def compact(base, attention, reference, *, reduced_precision):
        policies.append(reduced_precision)
        correction = 0 if zero_cost or reduced_precision else 8
        return SimpleNamespace(component_bytes=1 + correction, kind="compact"), dict(
            correction_storage_bytes=correction
        )

    monkeypatch.setattr(module, "build_compact", compact)
    monkeypatch.setattr(
        module, "build_dual", lambda *a: (SimpleNamespace(component_bytes=2, kind="dual"), {})
    )
    result, proof = module.construct(adapter, prefill, capture, (((0, 2, 3), (0, 1, 3)),))
    assert policies == ([False, False] if zero_cost else [False, True, False, True])
    assert result[0].kind == "compact"
    assert proof[0]["unchanged_memberships"]
