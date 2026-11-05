"""Synthetic timing fixtures only, never performance evidence."""

import evicastkv_deferred_exact_constructor as control
import pytest


def test_offered_to_both_baseline_readers_without_mutating_parent_records():
    parent = {
        "mosaickv_exact": {"graph": {"old": 1}},
        "prefixkv_reimpl": {"graph": {"old": 2}, "graph-original": {"old": 3}},
    }
    choices = control.variants(parent)
    assert len(choices) == 3
    assert all(r["defer_exact_encoding"] is True for _, _, r in choices)
    assert choices[0][2]["selective_text_rows"] is True
    assert all("defer_exact_encoding" not in r for item in parent.values() for r in item.values())


def test_three_repetitions_cannot_accept_a_formal_latency_gate():
    rows = [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(
                total_wall_seconds=0.90 if (m, a) == control.CANDIDATE else 1.0,
                logits_bitwise=True,
                no_FullKV_fallback=True,
            ),
        )
        for m, a in control.ARMS
        for t in range(-5, 3)
    ]
    proof = control.summary(rows)
    assert proof["four_percent_POINT_only"]
    assert not proof["formal_latency_acceptance"]
    assert len(proof["mean_ratios_against_original_and_optimized_PrefixKV"]) == 4
    with pytest.raises(ValueError):
        control.summary(rows[:-1])


def test_deferred_branch_engagement_required(monkeypatch):
    value = dict(
        deferred_exact_encoding=dict(enabled=True, created=4, encoded_at_barriers=2)
    )
    monkeypatch.setattr(control, "ORIGINAL_EXECUTE", lambda *a, **k: value)
    args = (None, None, None, None, None, "mosaickv_exact", dict(defer_exact_encoding=True))
    assert control.execute(*args) is value
    value["deferred_exact_encoding"]["encoded_at_barriers"] = 0
    with pytest.raises(RuntimeError, match="actual deferred"):
        control.execute(*args)
