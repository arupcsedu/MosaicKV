import torch

from mosaickv.evaluation.head_local_precision import localize, refit_corrections
from mosaickv.evaluation.precision_corrections import fund_corrections
from mosaickv.evaluation.precision_tier import construct


def packets(bits, old_policy, boundary):
    g = torch.Generator().manual_seed(705)
    pair = tuple(torch.randn(1, 2, 20, 128, generator=g).bfloat16() for _ in range(2))
    fmt = construct(pair, bits=bits, protected=(10,), encoded_exact=True)
    fmt, _ = fund_corrections(fmt, pair, fraction=0.005)
    heads = (tuple(range(20)),) * 2
    outputs = []
    for policy in (old_policy, "head3k2v_refit_boundary8"):
        packet = localize(fmt, pair, heads, {10}, policy=policy, boundary_override=boundary)
        packet, _ = refit_corrections(packet, pair)
        outputs.append(packet)
    return outputs


def test_new_boundary_is_identical_to_registered_eight_bit_boundary():
    old, new = packets(8, "head3_boundary8", True)
    assert old.component_bytes == new.component_bytes
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(old.materialize(), new.materialize(), strict=True)
    )


def test_new_middle_is_identical_to_registered_actual_two_bit_refit():
    old, new = packets(3, "head3k2v_refit", False)
    assert old.component_bytes == new.component_bytes
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(old.materialize(), new.materialize(), strict=True)
    )
