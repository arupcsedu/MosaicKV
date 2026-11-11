"""Synthetic native-wide exact-input accounting, fidelity and isolation tests."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_input_replacement import (
    FeasibilityDeclined,
    _choose_rows,
    build_replacement_formats,
)
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.native_quantized_source_control import native_quantized_fill
from mosaickv.evaluation.shared_source_format import construct


def example(layer_count=2, *, perturb_text=False):
    torch.manual_seed(22123)
    heads, length, dim = 4, 33, 128
    angles = torch.randn(1, length, dim // 2).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    identity = (
        torch.arange(length, dtype=torch.int32),
        torch.arange(length, dtype=torch.int16)[None].expand(heads, length),
    )
    blocks, packets, originals, pairs, windows = [], [], {}, [], {}
    text = tuple(range(4, 16))
    for layer in range(layer_count):
        attention = SimpleNamespace(
            k_proj=torch.nn.Linear(heads * dim, heads * dim, bias=False).bfloat16().eval(),
            v_proj=torch.nn.Linear(heads * dim, heads * dim, bias=False).bfloat16().eval(),
            num_key_value_groups=1,
        )
        originals[layer] = torch.randn(1, length, heads * dim).bfloat16()
        with torch.inference_mode():
            k = attention.k_proj(originals[layer]).reshape(1, length, heads, dim).transpose(1, 2)
            v = attention.v_proj(originals[layer]).reshape(1, length, heads, dim).transpose(1, 2)
            k = (
                k * phases[0][:, None]
                + torch.cat((-k[..., 64:], k[..., :64]), -1) * phases[1][:, None]
            )
            if perturb_text:
                k = k.clone()
                k[0, 0, text[0], 0] += 0.5
        pair = k, v
        packet = construct(
            originals[layer], pair, (tuple(range(length)),) * heads, text, phases, identity
        )
        blocks.append(SimpleNamespace(self_attn=attention))
        packets.append(packet)
        pairs.append(pair)
        windows[layer] = torch.ones(1, heads, 3, length, dtype=torch.float64) / length
    adapter = SimpleNamespace(_language_layers=lambda: blocks)
    return adapter, tuple(packets), originals, tuple(pairs), windows, text


def build(fixture, cap=None, **options):
    adapter, packets, originals, pairs, windows, text = fixture
    if cap is None:
        cap = backing_bytes(tuple(t for f in packets for t in f.tensors()))
    return build_replacement_formats(
        adapter, packets, originals, pairs, windows, text, cap_bytes=cap, **options
    )


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def test_text_dedup_is_native_parent_bitwise_and_only_actual_savings_fund_new_rows():
    fixture = example()
    adapter, packets, originals, pairs, _windows, text = fixture
    before = tuple(t.clone() for f in packets for t in f.tensors())
    original_snapshots = tuple(
        t.clone() for t in (*originals.values(), *(t for p in pairs for t in p))
    )
    weights = tuple(
        p.weight
        for b in adapter._language_layers()
        for p in (b.self_attn.k_proj, b.self_attn.v_proj)
    )
    weight_snapshots = tuple(t.clone() for t in weights)
    text_formats, corrected, report = build(fixture)
    parent_bytes = backing_bytes(tuple(t for f in packets for t in f.tensors()))
    assert report["parent_initial_bytes"] == parent_bytes
    assert report["text_initial_bytes"] < parent_bytes
    assert (
        report["text_initial_bytes"] + report["exact_text_retirement_funding_bytes"] == parent_bytes
    )
    assert report["corrected_initial_bytes"] <= parent_bytes
    assert report["selected_original_nontext_row_count"] > 0
    assert (
        report["corrected_initial_bytes"] - report["text_initial_bytes"]
        == report["restored_input_row_bytes"]
    )
    assert report["cap_shortfall_bytes"] == parent_bytes - report["corrected_initial_bytes"]
    assert report["CPU_tensor_metadata_bytes"] == 0
    assert not report["quality_or_latency_improvement_guaranteed"]
    changed = False
    for block, parent, replacement, improved, original in zip(
        adapter._language_layers(), packets, text_formats, corrected, pairs, strict=True
    ):
        reference = tuple(torch.empty(parent.shape, dtype=torch.bfloat16) for _ in range(2))
        native_quantized_fill(parent, block.self_attn, reference, direct=True)
        actual = replacement.materialize(block.self_attn)
        assert all(same(a, b) for a, b in zip(actual, reference, strict=True))
        actual = improved.materialize(block.self_attn)
        changed |= any(not same(a, b) for a, b in zip(actual, reference, strict=True))
        ids, n = parent.row_positions.long(), parent.shape[2]
        for value, old in zip(actual, original, strict=True):
            assert same(value[0, ids // n, ids % n], old[0, ids // n, ids % n])
        assert set(text).issubset(improved.input_positions.tolist())
        assert not hasattr(improved, "original_pair")
    assert changed, "Additional input rows must not all be protected no-ops"
    assert all(
        same(a, b) for a, b in zip(before, (t for f in packets for t in f.tensors()), strict=True)
    )
    assert all(
        same(a, b)
        for a, b in zip(
            original_snapshots, (*originals.values(), *(t for p in pairs for t in p)), strict=True
        )
    )
    assert all(same(a, b) for a, b in zip(weight_snapshots, weights, strict=True))


def test_strided_destinations_surroundings_and_private_transient_workspace():
    fixture = example(layer_count=1)
    text_formats, corrected, _ = build(fixture)
    attention = fixture[0]._language_layers()[0].self_attn
    for fmt in (*text_formats, *corrected):
        length = fmt.shape[2]
        buffers = tuple(torch.full((1, 4, length + 8, 128), -71).bfloat16() for _ in range(2))
        targets = tuple(t[..., 3 : length + 6, :] for t in buffers)
        observed = []
        fmt.fill(
            attention,
            targets,
            direct=True,
            workspace_observer=lambda ts, observed=observed: observed.append(backing_bytes(ts)),
        )
        expected = fmt.materialize(attention)
        for want, target, backing in zip(expected, targets, buffers, strict=True):
            assert same(want, target[..., :length, :])
            assert (backing[..., :3, :] == -71).all()
            assert (backing[..., length + 3 :, :] == -71).all()
        assert observed and max(observed) > 0
        owned = {t.untyped_storage()._cdata for t in fmt.tensors()}
        assert all(t.untyped_storage()._cdata not in owned for t in buffers)


def test_hard_cap_failure_and_no_unfunded_spending_or_answer_inputs():
    fixture = example(layer_count=1)
    adapter, packets, originals, pairs, windows, text = fixture
    parent_bytes = backing_bytes(tuple(t for f in packets for t in f.tensors()))
    with pytest.raises(FeasibilityDeclined, match="already exceeds"):
        build(fixture, cap=parent_bytes - 1)
    _text, _corrected, report = build(fixture, cap=parent_bytes + 2**20)
    assert report["restored_input_row_bytes"] <= report["exact_text_retirement_funding_bytes"]
    assert (
        report["exact_text_retirement_funding_bytes"] == parent_bytes - report["text_initial_bytes"]
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        build_replacement_formats(
            adapter,
            packets,
            originals,
            pairs,
            windows,
            text,
            cap_bytes=parent_bytes,
            answers=["forbidden"],
        )


def test_equal_scores_use_layer_then_original_position_and_exact_cost():
    chosen, rows, remaining = _choose_rows(
        [(2.0, 1, 9, 1028), (2.0, 0, 8, 1028), (2.0, 0, 7, 1028)], 2056, 2
    )
    assert chosen == [[7, 8], []]
    assert [(r["layer"], r["position"]) for r in rows] == [(0, 7), (0, 8)]
    assert remaining == 0
    assert _choose_rows([(1.0, 0, 5, 1028)], 1027, 1) == ([[]], [], 1027)


def test_frozen_weights_and_inputs_cannot_change_after_construction():
    fixture = example(layer_count=1)
    text_formats, _corrected, _ = build(fixture)
    attention = fixture[0]._language_layers()[0].self_attn
    with torch.no_grad():
        attention.k_proj.weight.add_(1)
    with pytest.raises(RuntimeError, match="Frozen original"):
        text_formats[0].materialize(attention)


def test_non_exact_text_membership_is_rejected_and_zero_demand_spends_nothing():
    fixture = example(layer_count=1)
    adapter, packets, originals, pairs, windows, _text = fixture
    cap = backing_bytes(tuple(t for f in packets for t in f.tensors()))
    with pytest.raises(ValueError, match="already be exact"):
        build_replacement_formats(adapter, packets, originals, pairs, windows, (20,), cap_bytes=cap)
    windows[0].zero_()
    _text, _corrected, report = build(fixture)
    assert report["selected_original_nontext_row_count"] == 0
    assert report["corrected_initial_bytes"] == report["text_initial_bytes"]


def test_text_only_original_word_patches_are_charged_and_remain_valid_after_visual_rows():
    fixture = example(layer_count=1, perturb_text=True)
    text_formats, corrected, report = build(fixture)
    assert report["layers"][0]["text_numeric_patch_scalars"][0] > 0
    assert all(p.indices.numel() == p.values.numel() for p in text_formats[0].text_patches)
    adapter, packets, _originals, pairs, _windows, _text = fixture
    attention = adapter._language_layers()[0].self_attn
    for fmt in (*text_formats, *corrected):
        actual = fmt.materialize(attention)
        n = packets[0].shape[2]
        ids = packets[0].row_positions.long()
        for value, original in zip(actual, pairs[0], strict=True):
            assert same(value[0, ids // n, ids % n], original[0, ids // n, ids % n])


@pytest.fixture(scope="module")
def prospective_fixture():
    return example()


def eligible_rows(fixture):
    _adapter, packets, _originals, _pairs, _windows, text = fixture
    result = []
    for layer, packet in enumerate(packets):
        old = set(packet.row_positions.tolist())
        heads, length = packet.shape[1:3]
        result.extend(
            (layer, position)
            for position in range(length)
            if position not in text
            and not all(head * length + position in old for head in range(heads))
        )
    return result


def negative_scores(fixture):
    return {
        layer: -torch.ones(packet.shape[2], dtype=torch.float32)
        for layer, packet in enumerate(fixture[1])
    }


def test_explicit_default_ranking_preserves_positions_funding_and_constructed_bits(
    prospective_fixture,
):
    implicit = build(prospective_fixture)
    explicit = build(prospective_fixture, ranking_mode="prompt_mse")
    assert implicit[2] == explicit[2]
    assert implicit[2]["ranking_mode"] == "prompt_mse"
    assert not implicit[2]["model_draft_derived_scoring"]
    assert implicit[2]["benchmark_answers_or_logits_accessed"] is False
    for left_group, right_group in zip(implicit[:2], explicit[:2], strict=True):
        for left, right in zip(left_group, right_group, strict=True):
            assert all(same(a, b) for a, b in zip(left.tensors(), right.tensors(), strict=True))


def test_signed_prospective_scores_control_order_without_changing_funding(prospective_fixture):
    eligible = eligible_rows(prospective_fixture)
    assert len(eligible) >= 2
    first, second = eligible[-1], eligible[0]
    scores = negative_scores(prospective_fixture)
    scores[first[0]][first[1]] = 4.0
    scores[second[0]][second[1]] = 2.0
    _text, _corrected, report = build(
        prospective_fixture,
        prospective_row_scores=scores,
        ranking_mode="draft_kl_margin_gradient",
    )
    selected = report["selected_original_nontext_rows"]
    assert [(r["layer"], r["position"]) for r in selected] == [first, second]
    assert [r["score"] for r in selected] == [4.0, 2.0]
    assert report["model_draft_derived_scoring"]
    assert not report["constructor_raw_logits_or_labels_accessed"]
    assert not report["benchmark_answers_accessed"]
    assert "benchmark_answers_or_logits_accessed" not in report
    assert report["restored_input_row_bytes"] == sum(r["raw_row_bytes"] for r in selected)
    assert report["corrected_initial_bytes"] <= report["parent_initial_bytes"]
    scores[first[0]][first[1]], scores[second[0]][second[1]] = 2.0, 4.0
    _, _, reversed_report = build(
        prospective_fixture,
        prospective_row_scores=scores,
        ranking_mode="draft_kl_margin_gradient",
    )
    assert [
        (r["layer"], r["position"]) for r in reversed_report["selected_original_nontext_rows"]
    ] == [second, first]
    assert (
        reversed_report["exact_text_retirement_funding_bytes"]
        == report["exact_text_retirement_funding_bytes"]
    )


def test_all_nonpositive_scores_never_fall_back_to_prompt_mse(prospective_fixture):
    scores = negative_scores(prospective_fixture)
    scores[0].zero_()
    text, corrected, report = build(
        prospective_fixture,
        prospective_row_scores=scores,
        ranking_mode="draft_kl_margin_gradient",
    )
    assert report["additional_input_rows"] == 0
    assert report["selected_original_nontext_rows"] == []
    assert report["corrected_initial_bytes"] == report["text_initial_bytes"]
    assert (
        report["unused_retirement_funding_bytes"] == report["exact_text_retirement_funding_bytes"]
    )
    assert all(left is right for left, right in zip(text, corrected, strict=True))


def test_prospective_scores_cannot_select_text_or_fully_protected_rows(prospective_fixture):
    scores = negative_scores(prospective_fixture)
    eligible = set(eligible_rows(prospective_fixture))
    for layer, values in scores.items():
        for position in range(values.numel()):
            if (layer, position) not in eligible:
                values[position] = 1e20
    _, _, report = build(
        prospective_fixture,
        prospective_row_scores=scores,
        ranking_mode="draft_kl_margin_gradient",
    )
    assert report["additional_input_rows"] == 0
    assert report["fully_protected_all_head_rows_excluded"]


@pytest.mark.parametrize(
    "case",
    ["none", "list", "missing", "extra", "bool_key", "integer", "matrix", "short", "nan", "inf"],
)
def test_invalid_prospective_score_inputs_fail_closed_before_construction(
    prospective_fixture, case, monkeypatch
):
    import mosaickv.evaluation.exact_input_replacement as module

    scores = negative_scores(prospective_fixture)
    if case == "none":
        scores = None
    elif case == "list":
        scores = list(scores.values())
    elif case == "missing":
        scores.pop(0)
    elif case == "extra":
        scores[2] = scores[0].clone()
    elif case == "bool_key":
        scores[False] = scores.pop(0)
    elif case == "integer":
        scores[0] = scores[0].long()
    elif case == "matrix":
        scores[0] = scores[0][None]
    elif case == "short":
        scores[0] = scores[0][:-1]
    elif case in ("nan", "inf"):
        scores[0][0] = float(case)
    monkeypatch.setattr(module, "_validate", lambda *_: pytest.fail("constructed invalid input"))
    with pytest.raises(ValueError, match="CPU finite floating"):
        build(
            prospective_fixture,
            prospective_row_scores=scores,
            ranking_mode="draft_kl_margin_gradient",
        )


def test_ranking_mode_and_optional_score_mismatches_fail_closed(prospective_fixture):
    with pytest.raises(ValueError, match="explicit draft"):
        build(prospective_fixture, prospective_row_scores=negative_scores(prospective_fixture))
    with pytest.raises(ValueError, match="Explicit prompt_mse"):
        build(prospective_fixture, ranking_mode="silent_fallback")
