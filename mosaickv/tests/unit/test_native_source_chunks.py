from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.native_source_chunks import new_workspace, project_component, source_packet
from mosaickv.evaluation.precision_tier import unpack_cpu
from mosaickv.evaluation.shared_source_format import construct


def test_bounded_chunks_into_padded_lease_preserve_positions_and_sentinels():
    torch.manual_seed(2408)
    h, n, d = 2, 521, 128
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    x = torch.randn(1, n, h * d).bfloat16()
    angle = torch.randn(1, n, 64).repeat(1, 1, 2)
    phases = angle.cos().bfloat16(), angle.sin().bfloat16()
    pair = tuple(torch.randn(1, h, n, d).bfloat16() for _ in range(2))
    heads = (tuple(range(n)),) * h
    packet = construct(
        x,
        pair,
        heads,
        (0, n - 1),
        phases,
        (torch.arange(n, dtype=torch.int32), torch.arange(n, dtype=torch.int16)[None].expand(h, n)),
    )
    packet = replace(packet, projection_backend="native_bmm_256")
    workspace = new_workspace(packet)
    assert workspace.shape == (1, 256, h * d) and source_packet(packet) is packet
    decoded = unpack_cpu(packet.source).squeeze(1)
    with torch.inference_mode():
        k = attention.k_proj(decoded).reshape(1, n, h, d).transpose(1, 2)
        v = attention.v_proj(decoded).reshape(1, n, h, d).transpose(1, 2)
        k = k * phases[0][:, None] + torch.cat((-k[..., 64:], k[..., :64]), -1) * phases[1][:, None]
    for component, reference in enumerate((k, v)):
        destination = torch.full((1, h, n + 7, d), -71).bfloat16()
        project_component(packet, attention, component, destination, workspace)
        torch.testing.assert_close(destination[..., :n, :], reference, rtol=0.01, atol=2**-6)
        assert (destination[..., n:, :] == -71).all()
    with pytest.raises(ValueError, match="bounded"):
        project_component(packet, attention, 0, destination, workspace[:, :1])


def test_unknown_backend_does_not_silently_use_cpu_oracle():
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    packet = SharedSourceFormat(
        None, (1, 1, 16, 128), None, (), None, None, None, None, projection_backend="unsupported"
    )
    with pytest.raises(NotImplementedError):
        packet.fill(None, (), direct=True)
