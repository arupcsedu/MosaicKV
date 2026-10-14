"""Synthetic statistics/scope tests; values are not experimental results."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "evidence_worker", SCRIPTS / "run_prefixkv_evidence_preflight.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
finally:
    sys.path.remove(str(SCRIPTS))


def completion(zero=False):
    ids = ["a", "b"]
    records = []
    for policy, name in worker.ORDER:
        rows = [
            dict(
                sample_id=s,
                status="completed",
                task_score=0.0 if zero else 1.0,
                answer=s,
                active_kv_bytes=worker.CAP,
                end_to_end_time=1.0,
                compression_time=0.1,
            )
            for s in ids
        ]
        records.append(dict(variant=name, mandatory_policy=policy, rows=rows))
    return dict(status="completed", records=records, failures=[]), ids


def test_id_only_split_excludes_prior_and_is_disjoint():
    cal, ev = worker.select_ids([f"x{i}" for i in range(80)], [f"x{i}" for i in range(50)])
    assert cal == tuple(f"x{i}" for i in range(50, 60))
    assert ev == tuple(f"x{i}" for i in range(60, 76))
    assert not set(cal) & set(ev)
    with pytest.raises(ValueError, match="duplicate"):
        worker.select_ids(["x", "x"], [])


def test_zero_score_cohort_never_passes_quality_and_policies_stay_separate():
    data, ids = completion(zero=True)
    summary = worker.summarize(data, ids)
    assert not summary["groups"][worker.QUESTION]["abc_combined"]["quality_gate"]
    assert not summary["ablation_deltas"]["ABC_vs_control_policy_change"]["same_mandatory_policy"]
    assert summary["ablation_deltas"]["ABC_vs_AB"]["same_mandatory_policy"]


def test_missing_rows_block_gate_and_unmatched_bytes_do_not_pass():
    data, ids = completion()
    data["records"][-1]["rows"].pop()
    data["records"][4]["rows"][0]["active_kv_bytes"] -= 1
    summary = worker.summarize(data, ids)
    assert not summary["groups"][worker.FIRST]["a_sink"]["quality_gate"]
    assert summary["groups"][worker.QUESTION]["abc_combined"]["failed_or_unavailable"] == 1
    assert not summary["groups"][worker.QUESTION]["abc_combined"]["quality_gate"]


def test_paired_intervals_and_repeatability():
    data, ids = completion()
    data["records"][3]["rows"][0]["task_score"] = 0
    summary = worker.summarize(data, ids)
    assert summary == worker.summarize(data, ids)
    assert summary["ablation_deltas"]["A_vs_control"]["paired_score_delta"] == 0.5
    assert summary["ablation_deltas"]["A_vs_control"]["ci95"] == [0.0, 1.0]
