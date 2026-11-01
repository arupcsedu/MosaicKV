from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.shared_source_format import (
    apply_transport_patch,
    construct,
    transport_corrections,
)
from mosaickv.evaluation.text_source_storage import optimize


def test_only_frozen_envelope_violations_patched_and_charged():
    original = tuple(torch.ones(1, 2, 33, 128).bfloat16() for _ in range(2))
    actual = tuple(t.clone() for t in original)
    actual[0][0, 1, 12, 99] = 1.125
    actual[1][0, 1, 12, 99] = 1.0078125  # within envelope: not an exact-value patch
    patches = transport_corrections(original, actual, torch.tensor([0], dtype=torch.int32))
    assert [p.indices.numel() for p in patches] == [1, 0]
    assert backing_bytes(tuple(t for p in patches for t in p.tensors)) == 6
    target = torch.full((1, 2, 39, 128), -71).bfloat16()
    target[..., :33, :].copy_(actual[0])
    apply_transport_patch(patches[0], target, prompt_length=33)
    assert torch.equal(target[..., :33, :], original[0])
    assert (target[..., 33:, :] == -71).all()
    with pytest.raises(RuntimeError, match="protected"):
        transport_corrections(original, actual, torch.tensor([33 + 12], dtype=torch.int32))


def test_shared_source_common_text_option_preserves_actual_values_and_protections():
    torch.manual_seed(5114)
    h, n, d = 2, 33, 128
    x = torch.randn(1, n, h * d).bfloat16()
    attention = SimpleNamespace(
        head_dim=d,
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    angles = torch.randn(1, n, 64).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    with torch.inference_mode():
        k = attention.k_proj(x).reshape(1, n, h, d).transpose(1, 2)
        v = attention.v_proj(x).reshape(1, n, h, d).transpose(1, 2)
        k = k * phases[0][:, None] + torch.cat((-k[..., 64:], k[..., :64]), -1) * phases[1][:, None]
    heads = (tuple(range(n)),) * h
    text = tuple((*range(3), *range(20, 33)))
    packet = construct(
        x,
        (k, v),
        heads,
        text,
        phases,
        (torch.arange(n, dtype=torch.int32), torch.arange(n, dtype=torch.int16)[None].expand(h, n)),
    )
    values = packet.materialize(attention)
    # Explicit numerical correction survives wrapper construction too.
    patches = transport_corrections(values, values, packet.row_positions)
    packet = replace(packet, transport_patches=patches)
    record = dict(
        normalized=x[:, text].contiguous(),
        cos=phases[0][:, text].contiguous(),
        sin=phases[1][:, text].contiguous(),
    )
    candidate, proof = optimize(packet, record, heads, text, attention)
    assert proof["changed_selected_positions"] == proof["changed_decoded_KV_scalars"] == 0
    assert candidate.component_bytes <= packet.component_bytes
    for a, b in zip(values, candidate.materialize(attention), strict=True):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    with pytest.raises(ValueError, match="identity"):
        optimize(packet, record, (tuple(reversed(range(n))),) * h, text, attention)
