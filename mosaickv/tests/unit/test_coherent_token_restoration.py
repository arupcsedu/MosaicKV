from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.coherent_token_restoration import rank_tokens, restore_token
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16


def fixture():
    torch.manual_seed(244)
    pair = tuple(torch.randn(1, 2, 6, 128).bfloat16() for _ in range(2))
    positions = ((0, 2, 5), (1, 3, 5))
    selected = tuple(
        torch.stack([t[0, h, list(ids)] for h, ids in enumerate(positions)])[None] for t in pair
    )
    fmt = ExactFormat(
        "kv",
        tuple(selected[0].shape),
        pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in selected),
    )
    return pair, positions, fmt


def test_restore_is_a_superset_sorted_exact_and_already_present_head_is_masked():
    pair, positions, fmt = fixture()
    (candidate,), proof = restore_token((fmt,), (pair,), (positions,), (None,), 2)
    result = candidate.materialize(None)
    assert candidate.valid_prompt_counts.tolist() == [3, 4]
    assert proof[0]["added_heads"] == (1,) and proof[0]["physical_padding_slots"] == 1
    for t, original in zip(result, pair, strict=True):
        assert torch.equal(t[0, 0, :3], original[0, 0, [0, 2, 5]])
        assert not t[0, 0, 3].any()
        assert torch.equal(t[0, 1], original[0, 1, [1, 2, 3, 5]])
    assert proof[0]["no_donor_removal"] and proof[0]["original_selected_KV_bitwise_preserved"]
    # Retention-1/no-addition is the original format, no transformed cache.
    (unchanged,), records = restore_token((fmt,), (pair,), (positions,), (None,), 5)
    assert unchanged is fmt and records[0]["added_KV_items"] == 0
    with pytest.raises(ValueError):
        restore_token((fmt,), (pair,), (positions,), (None,), 6)


def test_forecast_uses_individual_queries_and_no_labels_deterministic_visual_only():
    pair, positions, fmt = fixture()
    probabilities = torch.arange(1, 7).float().repeat(2, 1)
    steps = tuple(
        (torch.cat((probabilities.roll(s, dims=1), torch.zeros(2, s + 1)), dim=1)[None, :, None],)
        for s in range(4)
    )
    attention = SimpleNamespace()
    args = (steps, (pair,), (fmt,), (positions,), (attention,), (1, 2, 3, 4))
    a, b = rank_tokens(*args), rank_tokens(*args)
    assert a == b and {r["position"] for r in a} == {1, 2, 3, 4}
    assert a == tuple(sorted(a, key=lambda r: (-r["forecast_gain"], r["position"])))
    for row in a:
        gains = []
        position = row["position"]
        for step in steps:
            for head, ids in enumerate(positions):
                p = step[0][0, head, 0, :6].float()
                p = p / p.sum()
                mass = p[list(ids)].sum()
                kept = (p[list(ids), None] * pair[1][0, head, list(ids)].float()).sum(0) / mass
                full = (p[:, None] * pair[1][0, head].float()).sum(0)
                new = (
                    kept
                    if position in ids
                    else (kept * mass + p[position] * pair[1][0, head, position].float())
                    / (mass + p[position])
                )
                gains.append((kept - full).square().sum() - (new - full).square().sum())
        assert row["forecast_gain"] == pytest.approx(float(torch.stack(gains).mean()), abs=2e-5)
