from pathlib import Path


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_finalize

    return evicastkv_finalize


def test_confirmation_blocks_exhausted_pinned_tasks_without_replacement(tmp_path, monkeypatch):
    m = module(monkeypatch)
    from mosaickv.evaluation.fox_campaign import TASKS, sha, write_once

    dependencies = {}
    used = []
    for task in TASKS:
        path = tmp_path / f"{task}.json"
        write_once(path, dict(data=[dict(sample_id=i) for i in range(3)]))
        dependencies[task] = dict(annotation=str(path), annotation_sha256=sha(path))
        used += [f"{task}-{i:04d}" for i in range(3)]
    source = tmp_path / "materialization.json"
    write_once(source, dict(dependencies=dependencies))
    monkeypatch.setattr(m, "MATERIALIZATION", source)
    root = tmp_path / "run"
    write_once(
        root / "history-inventory.json",
        dict(MileBench_consumed_ids=used, COCO_consumed_ids=[], complete_scan=False),
    )
    output = tmp_path / "readiness.json"
    m.confirmation_feasibility(root, output)
    result = m.read(output)
    assert result["status"] == "BLOCKED"
    assert result["actual_confirmation_requests"] == 0
    assert result["no_datasets_replaced"]
    assert all(
        v["maximum_potentially_unused_ids_before_media_checks"] == 0
        for v in result["milebench_availability_upper_bounds"].values()
    )


def test_finite_callback_rejects_cycle_without_polling(tmp_path, monkeypatch):
    import pytest

    m = module(monkeypatch)
    from mosaickv.evaluation.fox_campaign import write_once

    monkeypatch.setattr(m, "clean_source", lambda *a: "fixture")
    write_once(
        tmp_path / "analysis/scoring/completion_decision.json",
        dict(action="submitted_registered_successor", stage="smoke"),
    )
    with pytest.raises(ValueError, match="may not loop"):
        m.follow(tmp_path, tmp_path, "scoring")


def test_legacy_job_spec_does_not_invent_new_arm_counts(tmp_path, monkeypatch):
    module(monkeypatch)
    import evicastkv_packet

    from mosaickv.evaluation.fox_campaign import write_once

    monkeypatch.setattr(evicastkv_packet, "clean_source", lambda *a: "fixture")
    root = tmp_path / "run"
    write_once(
        root / "protocol_lock.json", dict(source_root=str(root), source_sha="fixture", repeats=3)
    )
    write_once(root / "jobs/correctness.json", dict(jobs=[dict(indices=[0, 1])]))
    output = tmp_path / "derived"
    evicastkv_packet.packet(root, output)
    assert "| correctness | NOT_RUN | NOT_RUN |" in (output / "final_writing_packet.md").read_text()
