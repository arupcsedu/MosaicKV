"""Synthetic timing fixtures are schema tests, never experimental evidence."""

import pytest

from mosaickv.evaluation.lossless_storage_analysis import summarize_trials
from mosaickv.evaluation.lossless_storage_benchmark import VARIANTS, randomized_orders


def fixtures():
    return [
        dict(
            block=block,
            warmup=block < 0,
            encoded=encoded,
            fused=fused,
            bitwise_reference_parity=True,
            construction_seconds=0.01,
            decode_wall_seconds=1.0,
            decode_cuda_ms=1000.0,
            storage={"initial": 100 if encoded else 200},
        )
        for block in range(-5, 20)
        for encoded, fused in VARIANTS
    ]


def test_randomized_orders_are_deterministic_complete_and_nonconstant():
    orders = randomized_orders(seed=7)
    assert orders == randomized_orders(seed=7)
    assert len(orders) == 25
    assert len({tuple(order) for _, order in orders}) > 1
    assert all(set(order) == set(VARIANTS) for _, order in orders)
    with pytest.raises(ValueError, match="five warmups"):
        randomized_orders(seed=7, warmups=4)


def test_timing_analysis_requires_paired_complete_finite_records_and_equal_storage():
    rows = summarize_trials(fixtures())
    assert len(rows) == 4
    assert all(r["decode_ratio_ci95"] == [1.0, 1.0] for r in rows)
    with pytest.raises(ValueError, match="complete"):
        summarize_trials(fixtures()[:-1])
    records = fixtures()
    records[0]["decode_cuda_ms"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        summarize_trials(records)
    records = fixtures()
    for row in records:
        if row["fused"]:
            row["storage"] = {"initial": 42}
    with pytest.raises(ValueError, match="fusion changed"):
        summarize_trials(records)
