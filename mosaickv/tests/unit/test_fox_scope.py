from pathlib import Path


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_campaign

    return evicastkv_campaign


def test_scoped_execution_never_creates_a_blocked_workload_job(monkeypatch):
    m = module(monkeypatch)
    lock = dict(
        budgets_mib={"coco": [20, 40]},
        cohorts=dict(
            coco=dict(
                validation=dict(sample_ids=list(range(32))),
                development=dict(sample_ids=list(range(64))),
                profiler=dict(sample_ids=list(range(8))),
                timing=dict(sample_ids=list(range(16))),
            )
        ),
    )
    for stage in (
        "smoke",
        "representation_canary",
        "correctness",
        "scoring",
        "profiling",
        "measurement",
        "mechanisms",
        "full_graph_pilot",
    ):
        jobs = m.jobs_for(lock, stage)
        assert jobs
        assert {job["workload"] for job in jobs} == {"coco"}
    assert len(m.jobs_for(lock, "correctness")) == 4


def test_scope_change_requires_explicit_pre_quality_registration(tmp_path, monkeypatch):
    import pytest

    m = module(monkeypatch)
    monkeypatch.setattr(m, "clean_source", lambda *a: "fixture")
    with pytest.raises(ValueError, match="pre-quality scope registration"):
        m.prepare(tmp_path / "no-write", workloads=("coco",))
    assert not (tmp_path / "no-write").exists()


def test_scoped_groups_do_not_require_unsupported_dataset_files(tmp_path, monkeypatch):
    module(monkeypatch)
    import evicastkv_evidence as evidence

    from mosaickv.evaluation.fox_campaign import sha, write_once

    data = tmp_path / "coco.json"
    write_once(data, dict(samples=[dict(sample_id="x", media=[dict(sha256="frame")])]))
    write_once(
        tmp_path / "protocol_lock.json",
        dict(cohorts=dict(coco=dict(development=dict(path=str(data), sha256=sha(data))))),
    )
    monkeypatch.setattr(evidence, "MATERIALIZATION", tmp_path / "must-not-be-read.json")
    assert evidence.group_manifest(tmp_path)["groups"] == dict(coco=dict(x=["input-media:frame"]))
