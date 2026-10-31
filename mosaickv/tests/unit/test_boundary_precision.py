from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import head_local_precision as heads
from mosaickv.evaluation.precision_tier import construct


def test_eight_bit_override_requires_explicit_variant_and_preserves_text():
    pair = tuple(torch.randn(1, 2, 141, 128).bfloat16() for _ in range(2))
    selected = (tuple(range(141)),) * 2
    packet = construct(pair, bits=8, encoded_exact=True)
    with pytest.raises(ValueError, match="explicit policy"):
        heads.localize(packet, pair, selected, {67}, policy="head3", boundary_override=True)
    candidate = heads.localize(
        packet, pair, selected, {67}, policy="head3_boundary8", boundary_override=True
    )
    assert [p.bits for p in candidate.base.pair] == [8, 8]
    for x, y in zip(candidate.materialize(), pair, strict=True):
        assert torch.equal(x[..., 67, :], y[..., 67, :])
        assert torch.equal(x[..., :4, :], y[..., :4, :])
        assert torch.equal(x[..., -8:, :], y[..., -8:, :])


def test_boundary_override_changes_only_explicit_arm_and_records_real_costs(monkeypatch):
    pair = tuple(torch.randn(1, 2, 141, 128).bfloat16() for _ in range(2))
    selected = ((tuple(range(80)),) * 2,) * 5
    monkeypatch.setattr(heads.request_precision, "online_frozen", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(heads, "baseline_payload", lambda *a, **k: (selected, None, {}))
    monkeypatch.setattr(heads.Path, "read_text", lambda *a, **k: "[]")
    monkeypatch.setattr(heads, "prompt_weights", lambda *a, **k: (None, None))
    attention = SimpleNamespace(self_attn=None)
    adapter = SimpleNamespace(
        _legacy_layers=lambda _: ((pair,) * 5, None), _language_layers=lambda: [attention] * 5
    )
    prefill = SimpleNamespace(
        state=SimpleNamespace(past_key_values=None), attention_weights={i: None for i in range(5)}
    )
    formats, proof = heads.bundle(
        adapter,
        SimpleNamespace(modality_map=[]),
        prefill,
        SimpleNamespace(summaries=False, records={i: None for i in range(5)}),
        None,
        "mosaickv_exact",
        "fixture",
        0.5,
        policies=("union3", "head3_boundary8"),
    )
    for i, packet in enumerate(formats["head3_boundary8"]):
        assert tuple(b.bits for b in packet.base.pair) == ((3, 3) if i == 2 else (8, 8))
        if i != 2:
            assert formats["union3"][i].kind == "kv"
            assert packet.component_bytes < formats["union3"][i].component_bytes
    boundary = [r for r in proof["layers"] if r.get("boundary_override_bits")]
    assert {r["layer"] for r in boundary} == {0, 1, 3, 4}
    assert all(r["boundary_not_exact_for_explicit_variant_only"] for r in boundary)
