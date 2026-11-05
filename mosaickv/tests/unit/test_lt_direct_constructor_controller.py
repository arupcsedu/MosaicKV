"""Synthetic cost-screen fixtures, not measured performance evidence."""

import evicastkv_lt_direct_constructor as control
import pytest


def rows(ratio=0.90):
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
        for m, a in control.SCREEN_ARMS
        for t in range(-5, 3)
    ]


@pytest.mark.parametrize(
    "ratio,passed", [(0.90, True), (0.94, True), (0.941, False), (0.96, False), (1.0, False)]
)
def test_screen_is_conservative_and_never_acceptance(ratio, passed):
    decision = control.screen_summary(rows(ratio))
    assert decision["screen_pass"] is passed
    assert decision["preliminary_only_NOT_latency_acceptance"] is True
    assert decision["independent_quality_N"] == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "bool", "parity", "warmup"])
def test_screen_rejects_incomplete_or_bad_requests(fault):
    values = rows()
    if fault == "missing":
        values.pop()
    elif fault == "duplicate":
        values.append(values[0])
    elif fault == "parity":
        values[0]["measurement"]["logits_bitwise"] = False
    elif fault == "warmup":
        values[0]["warmup"] = False
    else:
        values[0]["measurement"]["total_wall_seconds"] = True if fault == "bool" else float("nan")
    with pytest.raises(ValueError):
        control.screen_summary(values)


def test_formal_followup_retains_every_old_baseline_control_and_family_bound():
    assert len(control.required(False)) == 13
    assert len(control.required(True)) == 21
    assert 2 * (len(control.required(False)) - 1 + len(control.required(True)) - 1) == 64
    for final in (False, True):
        assert {
            x for x in control.ORIGINAL_REQUIRED(final) if x[0] != "mosaickv_exact"
        } <= control.required(final)


@pytest.mark.parametrize("audit", [False, True])
def test_actual_constructor_path_and_audited_stride_proof_required(monkeypatch, audit):
    value = dict(
        lt_source_conversion=dict(
            constructor_policy="direct",
            full_parent_KV_word_validation=True,
            exhaustive_stride_audit=audit,
        )
    )
    monkeypatch.setattr(control, "ORIGINAL_EXECUTE", lambda *a, **k: value)
    record = dict(lt_fit_policy="direct")
    args = (None, None, None, None, {}, "mosaickv_exact", record)
    assert control.execute(*args, audit=audit) is value
    value["lt_source_conversion"]["full_parent_KV_word_validation"] = False
    with pytest.raises(RuntimeError):
        control.execute(*args, audit=audit)
