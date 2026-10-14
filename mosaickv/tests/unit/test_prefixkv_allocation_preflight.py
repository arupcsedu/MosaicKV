import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def worker(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "prefix_allocation_worker", root / "scripts/run_prefixkv_allocation_preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ledgers(worker):
    consumed, calibration, evaluation = worker.split_ids([str(i) for i in range(64)])
    metadata = dict(
        task="textvqa", dataset_id="test", dataset_revision="a" * 40, split="validation"
    )
    return (
        dict(metadata, sample_ids=list(calibration)),
        dict(metadata, sample_ids=list(evaluation)),
        consumed,
    )


def test_id_only_split_excludes_all_prior_debug_and_has_no_leakage(worker):
    cal, ev, consumed = ledgers(worker)
    assert cal["sample_ids"] == [str(i) for i in range(32, 42)]
    assert ev["sample_ids"] == [str(i) for i in range(42, 50)]
    worker.validate_ledgers(cal, ev, consumed)
    with pytest.raises(ValueError, match="consumed"):
        worker.validate_ledgers(cal, ev, [*consumed, "49"])
    with pytest.raises(ValueError, match="disjoint"):
        worker.validate_ledgers(cal, dict(ev, sample_ids=[*ev["sample_ids"][:-1], "32"]), consumed)
    with pytest.raises(ValueError, match="provenance"):
        worker.validate_ledgers(cal, dict(ev, dataset_revision="b" * 40), consumed)


def test_summary_never_hides_missing_failed_or_duplicate_samples(worker):
    ids = [str(i) for i in range(8)]
    rows = [
        dict(
            sample_id=s,
            status="completed",
            task_score=0.5,
            active_kv_bytes=worker.CAP,
            end_to_end_time=1,
            answer="synthetic",
        )
        for s in ids
    ]
    records = [dict(variant=name, rows=[dict(r) for r in rows]) for name in worker.ARMS]
    result = worker.summarize_task(dict(status="completed", failures=[], records=records), ids)
    assert result["arms"]["prefixkv_capacity_floor_reimpl"]["quality_gate"]
    assert "mean_end_to_end_seconds" not in result["arms"]["full_kv"]
    assert result["arms"]["mosaickv_exact"]["paired_score_delta_ci95"] == [0.0, 0.0]
    records[2]["rows"][-1]["status"] = "failed"
    result = worker.summarize_task(
        dict(status="failed", failures=["synthetic"], records=records), ids
    )
    failed = result["arms"]["prefixkv_upstream_shape_reimpl"]
    assert failed["attempted"] == 8 and failed["completed"] == 7
    assert failed["failed_or_unavailable"] == 1 and not failed["quality_gate"]
    records[2]["rows"] = [rows[0]] * 8
    result = worker.summarize_task(dict(status="failed", failures=[], records=records), ids)
    assert not result["arms"]["prefixkv_upstream_shape_reimpl"]["complete"]


def test_allocation_gate_rejects_quality_loss_or_different_bytes(worker):
    ids = ["a", "b"]
    top = [
        dict(
            sample_id=s,
            status="completed",
            task_score=1.0,
            active_kv_bytes=worker.CAP,
            end_to_end_time=1,
            answer="control",
        )
        for s in ids
    ]
    candidate = [dict(r, task_score=0.0) for r in top]
    summary = dict(
        status="completed",
        failures=[],
        records=[
            dict(variant="prompt_attention_topk", rows=top),
            dict(variant="prefixkv_capacity_floor_reimpl", rows=candidate),
        ],
    )
    report = worker.summarize_task(summary, ids)
    assert not report["arms"]["prefixkv_capacity_floor_reimpl"]["quality_gate"]
    for r in candidate:
        r.update(task_score=1.0, active_kv_bytes=worker.CAP + 1)
    report = worker.summarize_task(summary, ids)
    assert not report["arms"]["prefixkv_capacity_floor_reimpl"]["quality_gate"]
