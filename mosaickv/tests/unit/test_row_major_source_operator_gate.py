import copy

import pytest

from mosaickv.evaluation.row_major_source_operator import summarize


def fixture(ratio=0.7):
    return [
        dict(trial=t, row4=r, bitwise=True, wall_seconds=ratio if r else 1.0)
        for t in range(-5, 20)
        for r in (False, True)
    ]


def test_registered_operator_gate_is_stricter_than_small_speedup_and_not_request_evidence():
    d = summarize(fixture())
    assert d["operator_gate"] and d["registered_operator_upper_ratio"] == 0.8
    assert d["not_request_latency_evidence"] and d["independent_quality_N"] == 0
    assert not summarize(fixture(0.81))["operator_gate"]
    assert d == summarize(fixture())


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "bool", "nan", "zero", "words", "mode", "trial"]
)
def test_incomplete_or_numerically_invalid_operator_cannot_pass(mutation):
    rows = copy.deepcopy(fixture())
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "words":
        rows[0]["bitwise"] = False
    elif mutation == "mode":
        rows[0]["row4"] = 0
    elif mutation == "trial":
        rows[0]["trial"] = False
    else:
        rows[0]["wall_seconds"] = {"bool": True, "nan": float("nan"), "zero": 0}[mutation]
    with pytest.raises(ValueError):
        summarize(rows)
