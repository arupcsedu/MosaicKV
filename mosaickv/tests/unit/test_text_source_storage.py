from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.head_local_precision import localize
from mosaickv.evaluation.precision_tier import construct
from mosaickv.evaluation.text_source_storage import (
    TextSourceCapture,
    optimize,
    strip_text_overlays,
    text_slots,
)


def setup(merged=False):
    torch.manual_seed(924)
    h, n, d = 4, 64, 128
    hidden = torch.randn(1, n, h * d).bfloat16()
    attention = SimpleNamespace(
        head_dim=d,
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    angles = torch.randn(1, n, d)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    with torch.inference_mode():
        k = attention.k_proj(hidden).reshape(1, n, h, d).transpose(1, 2)
        v = attention.v_proj(hidden).reshape(1, n, h, d).transpose(1, 2)
        rotated = torch.cat((-k[..., d // 2 :], k[..., : d // 2]), -1)
        k = k * cos[:, None] + rotated * sin[:, None]
    text = tuple((*range(3), *range(48, 64)))
    head = tuple((*range(32), *range(48, 64)))
    positions = (head,) * h
    pair = (k[:, :, head].contiguous(), v[:, :, head].contiguous())
    if merged:
        # A baseline's merged text is NOT replaced with original projection K/V.
        pair = tuple(t + 0.25 for t in pair)
    base = construct(pair, bits=3, encoded_exact=True)
    fmt = localize(base, pair, positions, set(text), policy="head3_boundary8")
    record = dict(
        normalized=hidden[:, text].contiguous(),
        cos=cos[:, text].contiguous(),
        sin=sin[:, text].contiguous(),
    )
    return fmt, positions, text, record, attention


def test_head_slot_map_uses_original_positions_not_text_rank():
    assert text_slots(((0, 2, 8, 9), (0, 1, 8, 9)), (0, 1, 9)) == ((0, -1, 3), (0, 1, 3))
    with pytest.raises(ValueError):
        text_slots(((0, 2, 2),), (0, 2))


def test_immutable_alias_reuse_preserves_distinct_and_mutable_heads():
    head = (0, 2, 8, 9)
    slots = text_slots((head, head, [0, 1, 8, 9], tuple([0, 2, 8, 9])), (0, 1, 9))
    assert slots == ((0, -1, 3), (0, -1, 3), (0, 1, 3), (0, -1, 3))
    assert slots[0] is slots[1]
    # No cross-call reuse or reuse of a mutable row's previously valid contents.
    mutable = [0, 2, 8, 9]
    text_slots((mutable, mutable), (0, 9))
    mutable[-1] = 8
    with pytest.raises(ValueError):
        text_slots((mutable, mutable), (0, 9))


@pytest.mark.parametrize("bad", [(0, 1, 1), (0, -1, 2), (0, True, 2), (0, 1.0, 2)])
def test_alias_reuse_never_hides_bad_distinct_head(bad):
    head = (0, 1, 2)
    with pytest.raises(ValueError):
        text_slots((head, head, bad), (0, 2))


def test_alias_slots_match_reference_for_random_unique_head_shapes():
    import random

    rng = random.Random(20261104)
    for _ in range(100):
        rows = [tuple(rng.sample(range(40), 20)) for _ in range(4)]
        positions = tuple(rng.choice(rows) for _ in range(32))
        text = tuple(sorted(rng.sample(range(45), 12)))
        expected = tuple(
            tuple(({p: j for j, p in enumerate(head)}).get(t, -1) for t in text)
            for head in positions
        )
        assert text_slots(positions, text) == expected


@pytest.mark.parametrize("text", [(), (0, 1, 2, 3), (1, 3)])
def test_shared_source_overlay_cpu_plan_matches_original_order_and_words(text):
    from mosaickv.evaluation.exact_storage import decoded, encode
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    shape = (1, 2, 4, 8)
    ids = torch.tensor([7, 1, 4, 2], dtype=torch.int32)
    references = tuple(torch.arange(64).reshape(shape).bfloat16() + offset for offset in (0, 10))
    fmt = SharedSourceFormat(None, shape, ids, (), None, None, None, None)
    result = strip_text_overlays(fmt, references, (tuple(range(4)),) * 2, set(text))
    kept = [p for p in ids.tolist() if p % 4 not in text] or ids[:1].tolist()
    assert result.row_positions.tolist() == kept
    expected_ids = torch.tensor(kept)
    for original, body in zip(references, result.exact, strict=True):
        expected = original[0, expected_ids // 4, expected_ids % 4].reshape(1, 1, -1, 8)
        assert torch.equal(decoded(body).view(torch.uint8), expected.view(torch.uint8))
        assert all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(body.buffers, encode(expected.contiguous()).buffers, strict=True)
        )


def test_strip_changes_only_text_overlays_and_keeps_visual_protection():
    fmt, positions, text, _, attention = setup()
    old = fmt.materialize(attention)
    stripped = strip_text_overlays(fmt, old, positions, set(text))
    new = stripped.materialize(attention)
    slots = [i for i, p in enumerate(positions[0]) if p not in text]
    for a, b in zip(old, new, strict=True):
        assert torch.equal(a[:, :, slots].view(torch.int16), b[:, :, slots].view(torch.int16))
    assert stripped.protected == () and stripped.flat_tokens.numel() == 0
    assert stripped.component_bytes < fmt.component_bytes


@pytest.mark.parametrize("merged", [False, True])
def test_actual_baseline_kv_is_preserved_or_retained_with_raw_fallback(merged):
    fmt, positions, text, record, attention = setup(merged)
    optimized, proof = optimize(fmt, record, positions, text, attention)
    assert optimized.component_bytes <= fmt.component_bytes
    assert proof["changed_decoded_KV_scalars"] == 0
    assert proof["changed_selected_positions"] == 0
    assert proof["merged_baseline_values_preserved"]
    for a, b in zip(fmt.materialize(attention), optimized.materialize(attention), strict=True):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    if not merged:
        assert proof["source_used"]
    else:
        assert not proof["source_used"]
        assert proof["fallback_reason"] == "exact_residuals_and_headers_not_smaller"


def test_strided_destination_history_remains_untouched():
    fmt, positions, text, record, attention = setup()
    optimized, _ = optimize(fmt, record, positions, text, attention)
    destination = tuple(torch.full((1, 4, 57, 128), -71).bfloat16() for _ in range(2))
    optimized.fill(attention, destination, direct=True)
    for ref, target in zip(fmt.materialize(attention), destination, strict=True):
        assert torch.equal(ref.view(torch.int16), target[..., :48, :].view(torch.int16))
        assert (target[..., 48:, :] == -71).all()


def test_capture_is_text_only_and_removes_hooks():
    class Attention(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings):
            return hidden_states

    module = Attention()
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=module)])
    prepared = SimpleNamespace(
        modality_map=(
            SimpleNamespace(modality="text", start=0, end=2),
            SimpleNamespace(modality="image", start=2, end=6),
            SimpleNamespace(modality="text", start=6, end=8),
        ),
        logical_sequence_length=8,
    )
    x = torch.randn(1, 8, 256).bfloat16()
    phases = tuple(torch.randn(1, 8, 128).bfloat16() for _ in range(2))
    with TextSourceCapture(adapter, prepared) as capture:
        module(hidden_states=x, position_embeddings=phases)
    assert capture.positions == (0, 1, 6, 7)
    assert capture.records[0]["normalized"].shape == (1, 4, 256)
    assert capture.records[0]["normalized"].untyped_storage().nbytes() == 4 * 256 * 2
    assert not module._forward_pre_hooks
