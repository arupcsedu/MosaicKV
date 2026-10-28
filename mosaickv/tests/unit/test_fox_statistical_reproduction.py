from pathlib import Path

import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_statistics_recheck

    return evicastkv_statistics_recheck


def test_numeric_reproduction_preserves_labels_denominators_and_missing_values(monkeypatch):
    m = module(monkeypatch)
    a = dict(value=100.0, ci=[-1.0, 3.0], n=32, status="development", missing=None)
    b = dict(a, value=100.0 + 1e-12, implementation="sparse")
    assert m.compare_numeric(a, b) < 1e-10
    for change in (dict(n=31), dict(status="confirmed"), dict(missing=0), dict(value=float("nan"))):
        with pytest.raises(ValueError):
            m.compare_numeric(a, dict(b, **change))
    with pytest.raises(ValueError):
        m.compare_numeric(a, {})


def test_follow_does_not_poll_or_submit_gpu_work(monkeypatch):
    m = module(monkeypatch)
    import inspect

    source = inspect.getsource(m.follow)
    assert "squeue" not in source
    assert "sleep" not in source
    assert "submit(" not in source


def test_only_explicit_cpu_provenance_is_separate_from_numerical_results(monkeypatch):
    m = module(monkeypatch)
    a = [
        dict(
            scores={"x": 100.0},
            unique_inputs=64,
            reducer_verification=dict(
                requested_cpus=4,
                used_cpus=4,
                maximum_absolute_CIDEr_difference=1e-14,
            ),
        )
    ]
    b = [
        dict(
            a[0],
            reducer_verification=dict(a[0]["reducer_verification"], requested_cpus=1, used_cpus=1),
        )
    ]
    provenance = []
    assert m.compare_numeric(a, b, provenance=provenance) == 0
    assert [(p["reference"], p["candidate"]) for p in provenance] == [(4, 1), (4, 1)]
    for change in (dict(unique_inputs=63), dict(scores={"x": 99.0})):
        with pytest.raises(ValueError):
            m.compare_numeric(a, [dict(b[0], **change)])
    for change in (dict(used_cpus=0), dict(maximum_absolute_CIDEr_difference=0.1)):
        with pytest.raises(ValueError):
            m.compare_numeric(
                a, [dict(b[0], reducer_verification=dict(b[0]["reducer_verification"], **change))]
            )
    with pytest.raises(ValueError):
        m.compare_numeric(dict(requested_cpus=4), dict(requested_cpus=1))
    with pytest.raises(ValueError):
        m.compare_numeric(a, [dict(b[0], reducer_verification={})])


def test_mechanism_shards_partition_complete_cohort_without_changing_arms(monkeypatch):
    module(monkeypatch)
    from evicastkv_campaign import jobs_for

    from mosaickv.evaluation.fox_campaign import registered_arms

    lock = dict(
        budgets_mib={"coco": [20, 40]},
        cohorts={"coco": {"development": {"sample_ids": [str(i) for i in range(64)]}}},
    )
    jobs = jobs_for(lock, "mechanisms")
    assert len(jobs) == 32
    assert [i for job in jobs for i in job["indices"]] == list(range(64))
    assert all(len(job["indices"]) == 2 for job in jobs)
    assert all(job["arms"] == registered_arms("coco", "mechanisms") for job in jobs)


def test_analysis_can_follow_an_explicitly_pinned_different_finalizer_source(tmp_path, monkeypatch):
    m = module(monkeypatch)
    from mosaickv.evaluation.fox_campaign import write_once

    finalizer = "1" * 40
    analyst = "2" * 40
    monkeypatch.setattr(m, "clean_source", lambda *a: analyst)
    root = tmp_path / "primary"
    write_once(
        root / "analysis/mechanisms/completion_decision.json",
        dict(action="submitted_registered_successor", stage="full_graph_pilot"),
    )
    write_once(
        root / "jobs" / f"full_graph_pilot-final-audit-{finalizer[:12]}.json",
        dict(job="12345", source_sha=finalizer),
    )
    calls = []
    monkeypatch.setattr(m, "schedule", lambda *args: calls.append(args))
    monkeypatch.setattr(m, "reproduce", lambda *a, **k: pytest.fail("producer still pending"))
    m.follow(root, root, "mechanisms", finalizer)
    assert calls == [(root, root, "full_graph_pilot", "12345", finalizer)]
