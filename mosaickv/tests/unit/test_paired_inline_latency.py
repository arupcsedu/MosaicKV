import numpy as np
import pytest

from mosaickv.evaluation.paired_inline_latency import summarize, validate_rows


def test_paired_bounds_and_reproducibility_do_not_claim_cross_baseline_superiority():
    a = np.linspace(6.0, 7.0, 20)
    r = summarize(a, a * 0.95, seed=9)
    assert r == summarize(a, a * 0.95, seed=9)
    assert r["paired_latency_engineering_gate"]
    assert r["adjusted_one_sided_upper_ratio"] == pytest.approx(0.95)
    assert not r["superiority_over_other_baselines_established"]
    assert not summarize(a, a * 1.02, seed=9)["paired_latency_engineering_gate"]
    with pytest.raises(ValueError):
        summarize(a[:2], a[:2], seed=9)


def test_completeness_and_all_storage_boundaries_are_rechecked_from_raw_rows():
    rows = [
        dict(
            phase=phase,
            trial=i,
            arm=a,
            sample_id="fixture",
            status="completed",
            all_128_logits_bitwise=True,
            method_order=["original_probe", "inline"],
            detail=dict(
                initial_encoded_bytes=1024,
                final_encoded_bytes=2048,
                reported_counter_peak_bytes=3072,
            ),
        )
        for phase, n in (("warmup", 5), ("timed", 20))
        for i in range(n)
        for a in ("original_probe", "inline")
    ]
    assert len(validate_rows(rows)["inline"]) == 20
    with pytest.raises(ValueError):
        validate_rows([*rows[:-1], rows[0]])
    rows[0]["detail"]["final_encoded_bytes"] += 1
    with pytest.raises(ValueError, match="storage"):
        validate_rows(rows)
