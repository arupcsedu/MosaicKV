from copy import deepcopy

import pytest

from mosaickv.evaluation.source_hybrid_validation import (
    ARMS,
    BOUNDARIES,
    ordered_trials,
    paired_summary,
)


def fixture():
    return [
        dict(
            sample_id="consumed",
            phase=phase,
            trial=trial,
            arm=arm,
            method_order=list(ARMS),
            status="completed",
            request_wall_seconds=2.0,
            all_128_expected_logits_bitwise=True,
            detail={k: 1024 for k in BOUNDARIES},
        )
        for phase, count in (("warmup", 5), ("timed", 20))
        for trial in range(count)
        for arm in ARMS
    ]


def test_validation_requires_all_trials_and_checks_bits_and_real_storage():
    rows = fixture()
    assert all(len(r) == 20 for r in ordered_trials(rows).values())
    for invalid in (rows[:-1], [*rows, rows[0]]):
        with pytest.raises(ValueError, match="complete unique"):
            ordered_trials(invalid)
    for key, value in (("status", "failed"), ("all_128_expected_logits_bitwise", False)):
        invalid = deepcopy(rows)
        invalid[0][key] = value
        with pytest.raises(ValueError, match="failures"):
            ordered_trials(invalid)
    invalid = deepcopy(rows)
    invalid[2]["detail"][BOUNDARIES[0]] += 1
    with pytest.raises(ValueError, match="changed storage"):
        ordered_trials(invalid)
    invalid = deepcopy(rows)
    invalid[0]["request_wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ordered_trials(invalid)


def test_strict_latency_uses_registered_family_not_old_five_percent_tradeoff():
    good = paired_summary([2.0] * 20, [1.8] * 20, seed=1)
    bad = paired_summary([2.0] * 20, [2.01] * 20, seed=1)
    assert good["paired_latency_engineering_gate"]
    assert not bad["paired_latency_engineering_gate"]
    assert "10 registered" in good["adjustment"]


def test_compact_followup_cannot_omit_either_storage_control_or_hybrid():
    arms = ("storage_only", "codec_fast", "hybrid_fast")
    rows = [r for r in fixture() if r["arm"] in arms]
    for r in rows:
        r["method_order"] = list(arms)
    assert set(ordered_trials(rows, arms=arms)) == set(arms)
    with pytest.raises(ValueError, match="storage-controlled"):
        ordered_trials(rows, arms=("hybrid_fast",))
