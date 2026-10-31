from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.request_precision import online_frozen


def test_balanced_allocation_has_same_global_source_capacity_and_mandatory_positions():
    generator = torch.Generator().manual_seed(83)
    probabilities = tuple(
        torch.rand(1, 2, 40, 40, generator=generator).softmax(-1) for _ in range(4)
    )
    original = [p.clone() for p in probabilities]
    prefill = SimpleNamespace(attention_weights=probabilities)
    profile = (0.01, 0.7, 0.5, 0.2)
    _, old = online_frozen(prefill, profile, 0.5)
    uniform, proof = online_frozen(prefill, profile, 0.5, allocation_mode="balanced")
    repeated, _ = online_frozen(prefill, profile, 0.5, allocation_mode="balanced")
    assert repeated == uniform
    counts = proof["per_layer_capacity"]
    assert max(counts) - min(counts) <= 1
    assert len(set(old["per_layer_capacity"])) > 1
    assert sum(counts) == sum(old["per_layer_capacity"])
    assert proof["actual_source_slot_cap"] == old["actual_source_slot_cap"]
    for variant in uniform.values():
        for layer, heads in enumerate(variant["anchor"]["positions"]):
            assert all(
                len(h) == counts[layer] and 39 in h and tuple(sorted(set(h))) == h for h in heads
            )
    assert all(torch.equal(a, b) for a, b in zip(original, probabilities, strict=True))


def test_unknown_allocation_and_heterogeneous_balancing_fail_closed():
    p = SimpleNamespace(attention_weights=(torch.ones(1, 2, 40, 40) / 40,) * 4)
    with pytest.raises(ValueError, match="registered"):
        online_frozen(p, (0.1,) * 4, 0.5, allocation_mode="arbitrary")
    p.attention_weights = (*p.attention_weights[:-1], torch.ones(1, 2, 39, 39) / 39)
    with pytest.raises(ValueError, match="equal layer geometry"):
        online_frozen(p, (0.1,) * 4, 0.5, allocation_mode="balanced")


def test_explicit_uniform_arm_maps_codec_without_changing_ordinary_controls(monkeypatch):
    import evicastkv_text_row_study as driver
    from mosaickv.evaluation import head_local_precision

    calls = []

    def bundle(*args, **kwargs):
        calls.append(kwargs)
        return {kwargs["policies"][0]: "packet"}, {}

    monkeypatch.setattr(head_local_precision, "bundle", bundle)
    _, proof = driver.construct(*range(8), "query3", policy="balanced_boundary8")
    assert proof["precision"] == "balanced_boundary8" and proof["allocation_mode"] == "balanced"
    assert calls == [dict(policies=("head3_boundary8",), allocation_mode="balanced")]
    calls.clear()
    driver.construct(*range(8), "query3", policy="union3")
    assert calls == [dict(policies=("union3",))]
