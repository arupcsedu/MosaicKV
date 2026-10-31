import pytest
import torch

from mosaickv.evaluation.head_local_precision import POLICIES, cpu_packet, localize
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import construct


def fixture():
    generator = torch.Generator().manual_seed(31)
    pair = tuple(torch.randn(1, 3, 141, 128, generator=generator).bfloat16() for _ in range(2))
    heads = tuple(tuple(range(h, 141 + h)) for h in range(3))
    text = {40, 72, 99}
    union = tuple(sorted({j for head in heads for j, p in enumerate(head) if p in text}))
    fmt = construct(pair, bits=3, protected=union, encoded_exact=True)
    return pair, heads, text, fmt


@pytest.mark.parametrize("policy", [p for p in POLICIES if p != "union3"])
def test_head_local_protection_is_exact_without_cross_head_union(policy):
    pair, heads, text, fmt = fixture()
    packet = localize(fmt, pair, heads, text, policy=policy)
    reconstructed = packet.materialize()
    assert tuple(body.bits for body in packet.base.pair) == POLICIES[policy]
    for actual, original in zip(reconstructed, pair, strict=True):
        for h, row in enumerate(heads):
            protected = sorted(
                {j for j, p in enumerate(row) if p in text} | set(range(4)) | set(range(133, 141))
            )
            assert torch.equal(
                actual[0, h, protected].view(torch.int16),
                original[0, h, protected].view(torch.int16),
            )
        assert torch.isfinite(actual).all()
    assert packet.component_bytes == backing_bytes(packet.private())
    assert packet.component_bytes < fmt.component_bytes


def test_no_text_needs_no_flat_payload_and_preserves_reference_bits():
    pair, heads, _, _ = fixture()
    reference = construct(pair, bits=3, protected=(), encoded_exact=True)
    packet = localize(reference, pair, heads, set(), policy="head3")
    assert packet.protected == () and packet.flat_tokens.numel() == 0
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(packet.materialize(), reference.materialize(), strict=True)
    )


def test_common_text_slots_preserve_old_packet_outputs_and_strided_history():
    pair, heads, _, _ = fixture()
    heads = (heads[0],) * 3
    text = {40, 72}
    reference = construct(pair, bits=3, protected=(40, 72), encoded_exact=True)
    packet = localize(reference, pair, heads, text, policy="head3")
    actual = tuple(torch.full((1, 3, 151, 128), -7, dtype=torch.bfloat16) for _ in range(2))
    packet.fill(None, actual, direct=True)
    for a, b in zip(actual, reference.materialize(), strict=True):
        assert torch.equal(a[..., :141, :].view(torch.int16), b.view(torch.int16))
        assert bool((a[..., 141:, :] == -7).all())


def test_invalid_policy_and_membership_fail():
    pair, heads, text, fmt = fixture()
    with pytest.raises(ValueError, match="registered"):
        localize(fmt, pair, heads, text, policy="unknown")
    with pytest.raises(ValueError, match="membership"):
        localize(fmt, pair, heads[:1], text, policy="head3")


def test_geometry_does_not_specialize_per_decode_token():
    from mosaickv.evaluation.head_local_precision_kernel import _scatter_flat

    parameters = {p.name: p for p in _scatter_flat.params}
    assert all(
        not parameters[n].is_constexpr and parameters[n].do_not_specialize
        for n in ("COUNT", "N", "HEAD_OUT")
    )


@pytest.mark.parametrize("policy", [p for p in POLICIES if p != "union3"])
def test_cpu_oracle_copies_identical_encoded_packet_without_requantization(policy):
    pair, heads, text, fmt = fixture()
    packet = localize(fmt, pair, heads, text, policy=policy)
    copied = cpu_packet(packet)
    for original, clone in zip(packet.private(), copied.private(), strict=True):
        assert torch.equal(original.view(torch.uint8), clone.view(torch.uint8))
        if original.numel():
            assert original.data_ptr() != clone.data_ptr()
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(packet.materialize(), copied.materialize(), strict=True)
    )
