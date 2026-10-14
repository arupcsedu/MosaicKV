"""Synthetic recovery/paired-curve guards, not measured results."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/analyze_hf_baseline_recovery.py"
spec = importlib.util.spec_from_file_location("hf_recovery", PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def frame(cap=40):
    return pd.DataFrame(
        [
            dict(
                sample_id=str(i),
                task_score=float(i % 2),
                active_kv_bytes=cap * 2**20,
                end_to_end_time=float(i + 1),
            )
            for i in range(8)
        ]
    )


def test_exact_byte_latency_comparison_and_deterministic_intervals():
    p = worker.pair(frame(), frame())
    assert p == worker.pair(frame(), frame())
    assert p["quality_delta"] == 0 and p["latency_ratio"] == 1


def test_cross_cap_curve_never_reports_direct_latency():
    with pytest.raises(ValueError, match="shared realized bytes"):
        worker.pair(frame(20), frame(40))
    p = worker.pair(frame(20), frame(40), latency=False)
    assert p["memory_reduction"] == 0.5
    assert not any("latency" in k for k in p)


def test_missing_or_duplicate_sample_cannot_enter_paired_analysis():
    with pytest.raises(ValueError, match="paired identity"):
        worker.pair(frame(), frame().iloc[1:])
    f = frame()
    f.loc[1, "sample_id"] = "0"
    with pytest.raises(ValueError, match="paired identity"):
        worker.pair(f, f)


def test_missing_campaign_cannot_be_treated_as_complete(tmp_path):
    (tmp_path / "matrix").mkdir()
    (tmp_path / "matrix/jobs.jsonl").write_text("")
    with pytest.raises(ValueError, match="128 preregistered"):
        worker.collect(tmp_path, tmp_path / "retry")
