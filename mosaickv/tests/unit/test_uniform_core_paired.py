import copy

import evicastkv_uniform_core_paired as control
import pytest


def rows():
    return [
        dict(
            method=m,
            arm=a,
            trial=t,
            warmup=t < 0,
            measurement=dict(total_wall_seconds=6.5 if a == "joint-async-uniform-core" else 7.5),
        )
        for m, a in control.prior.ARMS
        for t in range(-5, 20)
    ]


def test_paired_bound_synthetic_fixture_not_benchmark_results():
    actual = control.paired_bounds(rows(), 3)
    assert actual["formal_latency_acceptance_against_both_Prefix_controls"]
    assert len(actual["comparisons"]) == 3
    assert actual["does_NOT_certify_all_baselines_or_benchmark_quality"]


def test_missing_or_duplicate_trial_fails_closed():
    data = rows()
    with pytest.raises(ValueError, match="paired trials"):
        control.paired_bounds(data[:-1], 3)
    with pytest.raises(ValueError, match="paired trials"):
        control.paired_bounds([*data, copy.deepcopy(data[0])], 3)


def test_nonfinite_time_fails_closed():
    data = rows()
    data[-1]["measurement"]["total_wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        control.paired_bounds(data, 3)


def test_semantic_gate_accepts_only_import_reordering_not_runtime_changes():
    assert control.semantic("import a\nimport b\nx=1") == control.semantic(
        "import b\nimport a\nx=1"
    )
    assert control.semantic("import a\nx=1") != control.semantic("import b\nx=1")
    assert control.semantic("import a\nx=1") != control.semantic("import a\nx=2")


def test_common_worker_repetition_registration_reaches_real_summary(monkeypatch):
    original = control.prior.base.ORIGINAL_SUMMARY
    shared = control.prior.base.control
    monkeypatch.setattr(shared, "ARMS", control.prior.ARMS)
    monkeypatch.setattr(shared, "CANDIDATE_ARM", "joint-async-uniform-core")
    monkeypatch.setattr(shared, "SCREEN_REPETITIONS", 20)
    data = rows()
    for row in data:
        row["measurement"].update(
            logits_bitwise=True,
            no_FullKV_fallback=True,
            preprocessing_wall_seconds=0.1,
            prefill_construct_retire_wall_seconds=0.5,
            decode_wall_seconds=6.0,
        )
    audited = {
        (m, a): {k: (1 if m == "mosaickv_exact" else 2) for k in control.driver.STORAGE}
        for m, a in control.prior.ARMS
    }
    assert original(data, audited)["four_percent_POINT_only"]
