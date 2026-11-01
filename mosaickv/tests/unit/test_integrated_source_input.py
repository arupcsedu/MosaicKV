from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import encode
from mosaickv.evaluation.integrated_source_input import ExactInput, fuse, overlay_input
from mosaickv.evaluation.precision_tier import unpack_cpu
from mosaickv.evaluation.shared_source_format import construct
from mosaickv.evaluation.text_source_storage import optimize


def test_chunk_overlay_uses_original_positions_and_keeps_other_rows_bitwise():
    values = torch.randn(1, 1, 4, 256).bfloat16()
    overlay = ExactInput(encode(values), torch.tensor([1, 129, 260, 601], dtype=torch.int32))
    for first in (0, 128, 256, 512):
        target = torch.full((1, 128, 256), -71).bfloat16()
        overlay_input(overlay, target, first)
        valid = (overlay.positions >= first) & (overlay.positions < first + 128)
        expected = torch.full_like(target, -71)
        expected[0, overlay.positions[valid].long() - first] = values[0, 0, valid]
        assert torch.equal(target.view(torch.int16), expected.view(torch.int16))


def test_cpu_fusion_is_lossless_or_explicitly_retains_existing_packet():
    torch.manual_seed(42041)
    h, n, d = 8, 64, 128
    x = torch.randn(1, n, h * d).bfloat16()
    attention = SimpleNamespace(
        head_dim=d,
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    phase = torch.randn(1, n, 64).repeat(1, 1, 2)
    cos, sin = phase.cos().bfloat16(), phase.sin().bfloat16()
    with torch.inference_mode():
        k = attention.k_proj(x).reshape(1, n, h, d).transpose(1, 2)
        v = attention.v_proj(x).reshape(1, n, h, d).transpose(1, 2)
        k = k * cos[:, None] + torch.cat((-k[..., 64:], k[..., :64]), -1) * sin[:, None]
    heads = (tuple(range(n)),) * h
    text = tuple((*range(3), *range(40, 64)))
    packet = construct(
        x,
        (k, v),
        heads,
        text,
        (cos, sin),
        (torch.arange(n, dtype=torch.int32), torch.arange(n, dtype=torch.int16)[None].expand(h, n)),
    )
    packet = replace(packet, projection_backend="native_bmm_256")
    optimized, _ = optimize(
        packet,
        dict(
            normalized=x[:, text].contiguous(),
            cos=cos[:, text].contiguous(),
            sin=sin[:, text].contiguous(),
        ),
        heads,
        text,
        attention,
    )
    fused, proof = fuse(optimized, attention)
    assert fused.component_bytes <= optimized.component_bytes
    for a, b in zip(optimized.materialize(attention), fused.materialize(attention), strict=True):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert (
        proof.get("three_materializations_bitwise", False)
        or proof["reason"] == "no_shared_source_text_packet"
    )
    if proof["used"]:
        with pytest.raises(ValueError):
            from mosaickv.evaluation.shared_source_format import fit_transport

            fit_transport(fused, attention)
        # Quantized source itself remains untouched by exact overlay execution.
        assert torch.isfinite(unpack_cpu(fused.source)).all()
