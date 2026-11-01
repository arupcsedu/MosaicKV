from dataclasses import replace

import pytest
import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.head_local_precision import localize
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.phase_capacity import inverse_phase, recode, rotate_cpu, selected_phases
from mosaickv.evaluation.precision_tier import construct, unpack_cpu


def example():
    torch.manual_seed(92441)
    h, n, d = 2, 32, 128
    ids = (tuple(range(0, 64, 2)), tuple(range(1, 65, 2)))
    angles = torch.randn(1, 65, 64).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    pair = tuple(torch.randn(1, h, n, d).bfloat16() for _ in range(2))
    packet = localize(
        construct(pair, bits=3, encoded_exact=True), pair, ids, {28, 29}, policy="head3_boundary8"
    )
    return packet, pair, ids, phases


def test_inverse_uses_original_head_local_positions_not_physical_slots():
    _, pair, ids, phases = example()
    positions = torch.tensor(ids, dtype=torch.int32)
    pre = inverse_phase(pair[0], positions, *phases)
    reconstructed = pre.clone()
    rotate_cpu(reconstructed, positions, *phases)
    assert torch.allclose(reconstructed.float(), pair[0].float(), atol=0.03125, rtol=0.02)
    wrong = pre.clone()
    rotate_cpu(wrong, torch.arange(32, dtype=torch.int32).repeat(2, 1), *phases)
    assert not torch.allclose(wrong.float(), pair[0].float(), atol=0.1)


@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("bits", [(3, 3), (2, 3), (2, 2)])
def test_recode_preserves_protected_post_rope_keys_and_values_and_history(pre, bits):
    packet, pair, ids, phases = example()
    before = tuple(t.clone() for t in pair)
    out = recode(packet, pair, ids, phases, key_bits=bits[0], value_bits=bits[1], pre_rotary=pre)
    result = out.materialize(None)
    for head in range(2):
        for slot, position in enumerate(ids[head]):
            if slot < 4 or slot >= 24 or position in {28, 29}:
                for actual, reference in zip(result, pair, strict=True):
                    assert torch.equal(actual[0, head, slot], reference[0, head, slot])
    target = tuple(torch.full((1, 2, 49, 128), -77).bfloat16() for _ in range(2))
    out.fill(None, target, direct=True)
    for a, b in zip(result, target, strict=True):
        assert torch.equal(a, b[..., :32, :])
        assert (b[..., 32:, :] == -77).all()
    for old, original in zip(before, pair, strict=True):
        assert torch.equal(old, original)
    assert out.component_bytes == backing_bytes(out.tensors())


def test_scalar_post_rope_patch_does_not_modify_its_unpatched_partner():
    packet, pair, ids, phases = example()
    index = torch.tensor([10 * 128 + 7], dtype=torch.int32)
    patch = SparseBitPatch(index, pair[0].flatten()[index.long()].clone())
    empty = SparseBitPatch(torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.bfloat16))
    packet = replace(packet, base=replace(packet.base, corrections=(patch, empty)))
    out = recode(packet, pair, ids, phases, key_bits=2, value_bits=3, pre_rotary=True)
    actual = out.materialize(None)[0]
    expected = unpack_cpu(out.base.base.pair[0])
    rotate_cpu(expected, out.positions, *phases)
    assert actual[0, 0, 10, 7] == pair[0][0, 0, 10, 7]
    assert actual[0, 0, 10, 71] == expected[0, 0, 10, 71]


def test_phase_metadata_is_shared_once_and_positions_are_charged():
    packet, pair, ids, phases = example()
    a = recode(packet, pair, ids, phases, key_bits=2, value_bits=2, pre_rotary=True)
    b = recode(packet, pair, ids, phases, key_bits=2, value_bits=2, pre_rotary=True)
    assert backing_bytes(
        (*a.tensors(), *b.tensors())
    ) == a.component_bytes + b.positions.numel() * 4 + sum(
        t.untyped_storage().nbytes() for body in b.base.base.pair for t in body.tensors()
    )


def test_invalid_phase_geometry_positions_and_bits_fail_closed():
    packet, pair, ids, phases = example()
    bad = phases[0].clone()
    bad[..., 64:] = 0
    with pytest.raises(ValueError):
        recode(packet, pair, ids, (bad, phases[1]), key_bits=2, value_bits=3, pre_rotary=True)
    with pytest.raises(ValueError):
        recode(packet, pair, ids, phases, key_bits=1, value_bits=3, pre_rotary=True)
    with pytest.raises(ValueError):
        selected_phases(torch.tensor(ids, dtype=torch.int32).flip(1), *phases)


def test_identity_phase_matches_unchanged_three_bit_source_bitwise():
    packet, pair, ids, phases = example()
    identity = torch.ones_like(phases[0]), torch.zeros_like(phases[1])
    out = recode(packet, pair, ids, identity, key_bits=3, value_bits=3, pre_rotary=True)
    for actual, expected in zip(out.materialize(None), packet.materialize(), strict=True):
        assert torch.equal(actual, expected)


def test_shared_capture_removes_hooks_and_rejects_foreign_phases():
    from types import SimpleNamespace

    from mosaickv.evaluation.phase_capacity import PhaseCapture

    class Attention(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings):
            return hidden_states

    modules = [Attention(), Attention()]
    adapter = SimpleNamespace(
        _language_layers=lambda: [SimpleNamespace(self_attn=m) for m in modules]
    )
    x = torch.zeros(1, 65, 256).bfloat16()
    _, _, _, phases = example()
    with PhaseCapture(adapter) as capture:
        for m in modules:
            m(hidden_states=x, position_embeddings=phases)
    assert capture.seen == {0, 1}
    assert not any(m._forward_pre_hooks for m in modules)
    assert all(
        torch.equal(a, b) and a.data_ptr() != b.data_ptr()
        for a, b in zip(capture.phases, phases, strict=True)
    )
    with pytest.raises(ValueError), PhaseCapture(adapter):
        modules[0](hidden_states=x, position_embeddings=phases)
        modules[1](hidden_states=x, position_embeddings=tuple(t.clone() for t in phases))
    assert not any(m._forward_pre_hooks for m in modules)
