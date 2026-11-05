import evicastkv_graph_accounting as control
import pytest


def test_each_baseline_reader_receives_optimization_and_originals_are_retained():
    records = {
        "mosaickv_exact": {"graph": {"old": 1}},
        "prefixkv_reimpl": {"graph": {"old": 2}, "graph-original": {"old": 3}},
    }
    result = control.variants(records)
    assert len(result) == 4
    assert len(control.ARMS) == 7
    assert sum(r["graph_accounting"] == "event_sum" for _, _, r in result) == 3
    assert all("graph_accounting" not in r for item in records.values() for r in item.values())
    assert ("mosaickv_exact", "graph") in control.ARMS
    assert ("prefixkv_reimpl", "graph") in control.ARMS
    assert ("prefixkv_reimpl", "graph-original") in control.ARMS


def test_three_repeat_screen_cannot_accept_latency_and_checks_optimized_controls():
    rows = [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(
                total_wall_seconds=0.9 if (m, a) == control.CANDIDATE else 1.0,
                logits_bitwise=True,
                no_FullKV_fallback=True,
            ),
        )
        for m, a in control.ARMS
        for t in range(-5, 3)
    ]
    result = control.summary(rows)
    assert result["four_percent_POINT_only"]
    assert not result["formal_latency_acceptance"]
    assert len(result["mean_ratios_against_original_and_optimized_PrefixKV"]) == 4
    with pytest.raises(ValueError):
        control.summary(rows[:-1])


def test_event_sum_requires_actual_notifications_and_audit_checks(monkeypatch):
    proof = dict(
        policy="event_sum",
        notifications=1,
        audited_independent_comparisons=1,
        final_independent_tensor_enumeration_preserved=True,
    )
    value = dict(graph_cache_accounting=proof)
    monkeypatch.setattr(control, "ORIGINAL_EXECUTE", lambda *a, **k: value)
    args = (None, None, None, None, None, "mosaickv_exact", dict(graph_accounting="event_sum"))
    assert control.execute(*args, audit=True) is value
    proof["audited_independent_comparisons"] = 0
    with pytest.raises(RuntimeError):
        control.execute(*args, audit=True)
