from dataclasses import replace

import pytest
import torch
from test_shared_source_format import example

from mosaickv.evaluation.precision_tier import unpack_cpu
from mosaickv.evaluation.shared_source_format import construct, fit_transport


@pytest.mark.parametrize("exclude", [False, True])
def test_component_constructor_preserves_packet_bits_parameters_and_native_inputs(
    exclude, monkeypatch
):
    x, pair, heads, text, phases, identity, attention = example()
    packet = construct(x, pair, heads, text, phases, identity, exclude_global_exact=exclude)
    monkeypatch.setattr(
        "mosaickv.evaluation.shared_source_kernel.decode_source_for_construction",
        lambda p: unpack_cpu(p.source).squeeze(1),
    )
    before = tuple(t.clone() for t in (*pair, x, attention.k_proj.weight, attention.v_proj.weight))
    original, control = fit_transport(packet, attention)
    candidate, proof = fit_transport(packet, attention, componentwise=True)
    assert (
        control["frozen_construction_tolerance_pass"]
        and proof["frozen_construction_tolerance_pass"]
    )
    assert proof["componentwise_reference_and_independent_verification"]
    assert candidate.component_bytes == original.component_bytes
    assert len(candidate.tensors()) == len(original.tensors())
    for a, b in zip(candidate.tensors(), original.tensors(), strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    for a, b in zip(candidate.materialize(attention), original.materialize(attention), strict=True):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    for a, b in zip(
        before, (*pair, x, attention.k_proj.weight, attention.v_proj.weight), strict=True
    ):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    with pytest.raises(ValueError):
        fit_transport(candidate, attention, componentwise=True)
    with pytest.raises(ValueError):
        fit_transport(packet, attention, componentwise=1)
    with pytest.raises(NotImplementedError):
        fit_transport(replace(packet, shape=(1, 2, 33, 64)), attention, componentwise=True)
