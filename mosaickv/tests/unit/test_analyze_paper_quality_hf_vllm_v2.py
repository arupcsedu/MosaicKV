from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest


def _module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "analyze_paper_quality_hf_vllm_v2.py"
    spec = importlib.util.spec_from_file_location("analyze_paper_hf_vllm_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(*, mosaic_scores: tuple[float, float] = (1.0, 0.0)) -> pd.DataFrame:
    rows = []
    for method, scores, active in (
        ("full_kv", (1.0, 1.0), 200),
        ("prompt_attention_topk", (1.0, 0.0), 100),
        ("mosaickv_full", mosaic_scores, 69),
    ):
        for index, score in enumerate(scores):
            rows.append(
                {
                    "backend_scope": "hf_eager",
                    "task": "textvqa",
                    "protocol_method": method,
                    "sample_id": f"sample-{index}",
                    "task_score": score,
                    "active_kv_bytes": active,
                    "residual_kv_bytes": 0,
                    "answer": f"answer-{index}" if method != "full_kv" else "full",
                    "end_to_end_time": 1.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_paired_summary_passes_exact_thirty_one_percent_reduction() -> None:
    module = _module()
    summary, failures = module.paired_task_summary(
        _frame(),
        backend="hf_eager",
        task="textvqa",
        minimum_reduction=0.30,
        bootstrap_resamples=1000,
        bootstrap_seed=17,
    )
    assert failures == []
    assert summary["minimum_active_memory_reduction_fraction"] == pytest.approx(0.31)
    assert summary["paired_mean_score_delta"] == 0.0
    assert summary["memory_gate_passed"] is True
    assert summary["quality_gate_passed"] is True


def test_paired_summary_reports_quality_regression_without_hiding_memory_win() -> None:
    module = _module()
    summary, failures = module.paired_task_summary(
        _frame(mosaic_scores=(0.0, 0.0)),
        backend="hf_eager",
        task="textvqa",
        minimum_reduction=0.30,
        bootstrap_resamples=1000,
        bootstrap_seed=17,
    )
    assert summary["memory_gate_passed"] is True
    assert summary["quality_gate_passed"] is False
    assert any("paired official-score delta" in failure for failure in failures)
