from types import SimpleNamespace

import dispatch_affinity_diagnostic as study
import pytest

from mosaickv.evaluation.dispatch_affinity_audit import enqueue, pinned_thread


def test_cpu_affinity_is_restricted_to_allowed_mask_and_restored_after_failure(monkeypatch):
    state = {3, 5}
    changes = []

    def set_mask(pid, value):
        nonlocal state
        assert pid == 0
        state = set(value)
        changes.append(state)

    monkeypatch.setattr("os.sched_getaffinity", lambda pid: state)
    monkeypatch.setattr("os.sched_setaffinity", set_mask)
    with pytest.raises(RuntimeError, match="sentinel"), pinned_thread() as report:
        assert state == {3} and report["original_allowed_cpus"] == [3, 5]
        raise RuntimeError("sentinel")
    assert state == {3, 5} and changes == [{3}, {3, 5}]


def test_wall_thread_and_context_switch_counters_are_kept_without_clipping():
    walls = iter((100, 110))
    threads = iter((50, 65))
    counters = iter(
        (
            SimpleNamespace(ru_nvcsw=3, ru_nivcsw=2, ru_minflt=5, ru_majflt=0),
            SimpleNamespace(ru_nvcsw=4, ru_nivcsw=4, ru_minflt=8, ru_majflt=0),
        )
    )
    called = []
    result = enqueue(
        lambda: called.append(True),
        2,
        wall=lambda: next(walls),
        thread=lambda: next(threads),
        usage=lambda _: next(counters),
    )
    assert len(called) == 2
    assert result["wall_minus_thread_seconds_per_launch"] < 0
    assert result["voluntary_context_switches"] == 1
    assert result["involuntary_context_switches"] == 2
    assert result["minor_page_faults"] == 3


def test_exact_twenty_timed_pairs_plus_warmups_and_no_removed_outlier():
    rows = [
        dict(repetition=i, raw_addresses=f, warmup=i < 0)
        for i in range(-5, 20)
        for f in (False, True)
    ]
    study.validate_trials(rows)
    with pytest.raises(ValueError, match="every unique"):
        study.validate_trials(rows[:-1])
    with pytest.raises(ValueError, match="every unique"):
        study.validate_trials([*rows, rows[-1]])


def test_cpu_timing_cannot_replace_wall_gate_or_authorize_promotion():
    p = study.protocol()
    assert p["strict_adjusted_upper_wall_ratio"] == 1.0
    assert p["thread_time_is_diagnostic_only_not_a_replacement_gate"] is True
    for key in (
        "model_execution_authorized",
        "candidate_combination_authorized",
        "candidate_promotion_authorized",
        "outlier_removal_authorized",
        "repeat_until_pass_authorized",
    ):
        assert p[key] is False


def test_operator_review_has_no_model_submission_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study.base.common, "checked", lambda *args: {})
    monkeypatch.setattr(study.base.common, "submit", study.base.common.submit)
    monkeypatch.setattr(study.base.common, "launch", lambda *args: pytest.fail("model launch"))
    monkeypatch.setattr(study, "analyze", lambda root, p: calls.append(root))
    monkeypatch.setattr("sys.argv", ["diagnostic", "after_operator", "--root", str(tmp_path)])
    study.main()
    assert calls == [tmp_path]
