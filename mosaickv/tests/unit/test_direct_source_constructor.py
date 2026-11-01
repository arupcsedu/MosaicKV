from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import direct_source_constructor as direct
from mosaickv.evaluation.request_precision import online_frozen


@pytest.mark.parametrize(
    "kwargs", [dict(ratio=0.4), dict(head_group=True), dict(lossless_minimum=1)]
)
def test_policy_rejects_unregistered_changes(kwargs):
    with pytest.raises(ValueError):
        direct.DirectSourcePolicy(**kwargs)


def test_online_memberships_match_old_selection_without_shadow_quantization(monkeypatch):
    torch.manual_seed(127)
    h, n = 4, 32
    probs = torch.softmax(torch.randn(1, h, n, n), -1).bfloat16()
    original = ((torch.randn(1, h, n, 128).bfloat16(), torch.randn(1, h, n, 128).bfloat16()),)
    prefill = SimpleNamespace(
        attention_weights=(probs,),
        state=SimpleNamespace(past_key_values=object(), logical_sequence_length=n),
    )
    adapter = SimpleNamespace(
        device="cpu",
        _legacy_layers=lambda value: (original, None),
        _language_layers=lambda: [SimpleNamespace(self_attn=object())],
    )
    capture = SimpleNamespace(records={0: torch.randn(1, n, h * 128).bfloat16()})
    text = SimpleNamespace(records={0: object()}, positions=(0, 31))
    phase = SimpleNamespace(
        phases=(torch.ones(1, n, 128).bfloat16(), torch.zeros(1, n, 128).bfloat16())
    )
    monkeypatch.setattr(direct, "fit_transport", lambda packet, attention: (packet, {}))
    monkeypatch.setattr(direct, "optimize", lambda packet, *args: (packet, {}))
    packets, proof = direct.construct(
        adapter,
        SimpleNamespace(logical_sequence_length=n),
        prefill,
        capture,
        text,
        phase,
        (0.5,),
        2**24,
    )
    old, _ = online_frozen(
        prefill, (0.5,), 0.5, method="mosaickv_exact", allocation_mode="balanced"
    )
    assert proof["source_memberships"] == old["uniform_0.5_control"]["anchor"]["positions"]
    assert (
        proof["physical_bytes"] <= proof["cap_bytes"]
        and proof["no_discarded_shadow_codec_constructed"]
    )
    assert packets[0].source.minimum.dtype == torch.bfloat16
    assert packets[0].source.scale.dtype == torch.float32
    with pytest.raises(RuntimeError, match="hard cap"):
        direct.construct(
            adapter,
            SimpleNamespace(logical_sequence_length=n),
            prefill,
            capture,
            text,
            phase,
            (0.5,),
            1,
        )
