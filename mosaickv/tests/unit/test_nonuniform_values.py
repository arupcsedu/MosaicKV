import pytest
import torch

from mosaickv.evaluation.head_local_precision import cpu_packet, localize, refit_corrections
from mosaickv.evaluation.nonuniform_values import pack_nonuniform_values
from mosaickv.evaluation.precision_corrections import fund_corrections
from mosaickv.evaluation.precision_tier import construct, pack, unpack_cpu


@pytest.mark.parametrize("kind", ["gaussian", "constant", "outliers", "skew"])
def test_same_physical_bytes_and_no_row_mse_regression(kind):
    g = torch.Generator().manual_seed(26)
    x = torch.randn(1, 3, 19, 128, generator=g).bfloat16()
    if kind == "constant":
        x.fill_(3)
    elif kind == "outliers":
        x[..., 0] *= 20
    elif kind == "skew":
        x = x.float().abs().square().bfloat16()
    old, new = pack(x, bits=2, axis=3), pack_nonuniform_values(x)
    assert sum(t.untyped_storage().nbytes() for t in old.tensors()) == sum(
        t.untyped_storage().nbytes() for t in new.tensors()
    )
    before = (unpack_cpu(old).float() - x.float()).square().sum(-1)
    after = (unpack_cpu(new).float() - x.float()).square().sum(-1)
    assert (after <= before + 1e-6).all()
    if kind != "constant":
        assert (after < before).any()
    assert new.codebook.dtype == torch.bfloat16
    assert new.codebook.shape == (1, 3, 19, 4)
    assert torch.equal(new.codes, pack_nonuniform_values(x).codes)


def test_empty_clusters_remain_finite_and_exact_for_constant_row():
    x = torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16)
    packet = pack_nonuniform_values(x)
    assert torch.isfinite(packet.codebook).all()
    assert torch.equal(x, unpack_cpu(packet))


def test_guarded_shape_and_nonfinite_input():
    with pytest.raises(ValueError):
        pack_nonuniform_values(torch.zeros(1, 1, 8, 64).bfloat16())
    with pytest.raises(ValueError):
        pack_nonuniform_values(torch.full((1, 1, 8, 128), torch.nan).bfloat16())


def test_head_local_protocol_preserves_keys_text_and_exception_budget():
    g = torch.Generator().manual_seed(84)
    pair = tuple(torch.randn(1, 2, 40, 128, generator=g).bfloat16() for _ in range(2))
    fmt = construct(pair, bits=3, protected=(17,), encoded_exact=True)
    fmt, _ = fund_corrections(fmt, pair, fraction=0.005)
    heads = (tuple(range(40)),) * 2
    old = localize(fmt, pair, heads, {17}, policy="head3k2v_refit_boundary8")
    new = localize(fmt, pair, heads, {17}, policy="head3k2v_nonuniform_boundary8")
    old, _ = refit_corrections(old, pair)
    new, proof = refit_corrections(new, pair)
    assert old.component_bytes == new.component_bytes
    decoded = new.materialize()
    assert torch.equal(old.materialize()[0], decoded[0])
    assert torch.equal(decoded[1][:, :, 17], pair[1][:, :, 17])
    assert proof["no_extra_exception_bytes"]
    copied = cpu_packet(new)
    assert all(torch.equal(a, b) for a, b in zip(copied.materialize(), decoded, strict=True))
    for boundary in (True,):
        a = localize(
            fmt, pair, heads, {17}, policy="head3k2v_refit_boundary8", boundary_override=boundary
        )
        b = localize(
            fmt,
            pair,
            heads,
            {17},
            policy="head3k2v_nonuniform_boundary8",
            boundary_override=boundary,
        )
        assert a.component_bytes == b.component_bytes
        assert all(torch.equal(x, y) for x, y in zip(a.materialize(), b.materialize(), strict=True))
