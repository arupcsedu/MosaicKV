"""Synthetic controller fixtures, never measured latency evidence."""

import evicastkv_selective_text_constructor as control
import pytest


def rows(ratio=0.95):
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
        for m, a in control.ARMS
        for t in range(-5, 3)
    ]


def test_point_progress_does_not_establish_any_latency_acceptance():
    result = control.summary(rows())
    assert result["four_percent_POINT_only"] is True
    assert result["construction_progress_vs_original"] is True
    assert result["formal_latency_acceptance"] is False
    assert result["full_twenty_repeat_adjusted_ratio_gate_still_required"] is True


def test_negative_point_is_preserved_and_cannot_release_larger_jobs():
    result = control.summary(rows(1.05))
    assert not result["four_percent_POINT_only"]
    assert not result["construction_progress_vs_original"]
    assert not result["formal_latency_acceptance"]


def test_incomplete_data_fail_closed():
    with pytest.raises(ValueError):
        control.summary(rows()[:-1])


@pytest.mark.parametrize("fault", [None, "zero", "full", "metadata"])
def test_actual_selective_path_must_engage(monkeypatch, fault):
    proof = dict(
        enabled=True,
        source_layers=26,
        projected_rows=1040,
        original_full_rows=15600,
        cpu_metadata_hits=26,
        cpu_metadata_misses=0,
    )
    if fault == "zero":
        proof["source_layers"] = 0
    if fault == "full":
        proof["projected_rows"] = 15600
    if fault == "metadata":
        proof["cpu_metadata_misses"] = 1
    monkeypatch.setattr(
        control.base, "ORIGINAL_EXECUTE", lambda *a, **k: dict(selective_text_construction=proof)
    )
    args = (None, None, None, None, {}, "mosaickv_exact", dict(selective_text_rows=True))
    if fault is None:
        assert control.execute(*args)["selective_text_construction"] is proof
    else:
        with pytest.raises(RuntimeError):
            control.execute(*args)
