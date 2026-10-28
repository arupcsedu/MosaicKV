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
