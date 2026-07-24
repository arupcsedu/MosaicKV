from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_analysis() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "analyze_quality_memory.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_quality_analysis", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_analysis()


def test_paired_bootstrap_preserves_constant_score_delta() -> None:
    reference = np.asarray([0.0, 1.0, 0.5, 0.25])
    candidate = reference - 0.1

    result = ANALYSIS._bootstrap_quality(candidate, reference, seed=7, iterations=1000)

    assert result["score_delta_ci_low"] == pytest.approx(-0.1)
    assert result["score_delta_ci_high"] == pytest.approx(-0.1)


def test_pareto_marks_dominated_points_for_every_resource_projection() -> None:
    common = {
        "classification": "derived_from_measured_main_quality",
        "model": "model",
        "task": "task",
        "metric": "accuracy",
        "output_length": 128,
    }
    summaries = [
        {
            **common,
            "method": "mosaickv_exact",
            "retention_ratio": 0.5,
            "task_score": 0.8,
            "active_kv_bytes": 100.0,
            "total_latency": 2.0,
            "decode_latency": 1.0,
        },
        {
            **common,
            "method": "prompt_attention_topk",
            "retention_ratio": 0.5,
            "task_score": 0.9,
            "active_kv_bytes": 90.0,
            "total_latency": 1.5,
            "decode_latency": 0.8,
        },
    ]

    rows = ANALYSIS._pareto(summaries)

    mosaic = [row for row in rows if row["method"] == "mosaickv_exact"]
    assert len(mosaic) == 3
    assert all(not row["pareto_optimal"] for row in mosaic)
    assert all("prompt_attention_topk@0.5" in row["dominated_by"] for row in mosaic)
