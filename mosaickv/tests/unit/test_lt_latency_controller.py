"""Synthetic controller fixtures, not measured latency or quality evidence."""

import copy

import evicastkv_lt_latency as control
import pytest


def rows(final=False, ratio=0.94):
    return [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(
                total_wall_seconds=ratio if (m, a) == control.CANDIDATE else 1.0,
                logits_bitwise=True,
                no_FullKV_fallback=True,
            ),
        )
        for m, a in control.required(final)
        for t in range(-5, 20)
    ]


@pytest.mark.parametrize("final", [False, True])
def test_all_original_and_new_baseline_controls_retained(final):
    values = rows(final)
    assert control.summarize(values, 4, final=final)["four_percent_latency_gate"]
    for m, a in control.required(final):
        if m == "mosaickv_exact":
            continue
        bad = copy.deepcopy(values)
        for r in bad:
            if (r["method"], r["arm"]) == (m, a):
                r["measurement"]["total_wall_seconds"] = 0.90
        assert not control.summarize(bad, 4, final=final)["four_percent_latency_gate"]


def test_family_bound_covers_both_stages_and_inputs():
    assert 2 * (len(control.required(False)) - 1 + len(control.required(True)) - 1) <= 64
    assert len(control.required(False)) == 12
    assert len(control.required(True)) == 21


@pytest.mark.parametrize(
    "ratio,passed", [(0.95, True), (0.961, False), (1.0, False), (1.04, False)]
)
def test_strict_four_percent_gate(ratio, passed):
    assert control.summarize(rows(ratio=ratio), 3)["four_percent_latency_gate"] is passed


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "nan", "boolean", "parity"])
def test_fail_closed_incomplete_or_invalid_trials(fault):
    values = rows()
    if fault == "missing":
        values.pop()
    elif fault == "duplicate":
        values.append(values[0])
    elif fault == "unknown":
        values[0]["arm"] = "unsupported"
    elif fault == "parity":
        values[0]["measurement"]["logits_bitwise"] = False
    else:
        values[0]["measurement"]["total_wall_seconds"] = (
            True if fault == "boolean" else float("nan")
        )
    with pytest.raises(ValueError):
        control.summarize(values, 3)


def test_common_option_does_not_mutate_parent_or_remove_old_arms():
    parent = dict(
        records={m: dict(direct={}, **{"whole-6291456": {}}) for m in control.previous.METHODS}
    )
    original = copy.deepcopy(parent)
    result = control.make_records(parent, control.previous.METHODS)
    assert original == parent
    for arms in result.values():
        assert arms["graph-lt"]["lt_source_reader"] is True
        assert "lt_source_reader" not in arms["graph"]
        assert arms["graph-lt"]["prompt_graph_policy"] == control.arena.POLICY
    assert result["prefixkv_reimpl"]["whole-graph-lt"]["lt_source_reader"] is True
