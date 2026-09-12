from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


def _module(name: str, filename: str) -> Any:
    package_root = Path(__file__).parents[2]
    sys.path.insert(0, str(package_root))
    path = package_root / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_locks_equal_byte_controls() -> None:
    module = _module(
        "mosaickv_test_prepare_hf_byte_curves",
        "prepare_hf_byte_matched_curves.py",
    )
    path = Path(__file__).parents[2] / "configs/experiments/hf_byte_matched_curves_v1.yaml"

    protocol = module.load_protocol(path, require_authorized=True)

    evaluation = protocol["evaluation"]
    assert tuple(evaluation["methods"]) == module.METHODS
    assert tuple(evaluation["budgets_mib"]) == (16, 22, 30, 40, 60)
    assert evaluation["samples_per_shard"] == 16
    assert evaluation["block_size"] == evaluation["physical_budget_block_size"] == 4
    assert evaluation["physical_budget_mandatory_policy"] == "last_token"
    assert protocol["paired_controls"]["require_exact_per_sample_active_byte_match"] is True


def test_sample_selection_is_deterministic_and_reads_only_identity_columns() -> None:
    module = _module(
        "mosaickv_test_prepare_hf_byte_curves_selection",
        "prepare_hf_byte_matched_curves.py",
    )
    rows = []
    for sample_index in range(12):
        for method in module.SOURCE_METHODS:
            rows.append(
                {
                    "task": "mmstar",
                    "sample_id": f"mmstar:{sample_index}",
                    "backend_scope": "hf_eager",
                    "protocol_method": method,
                    "status": "completed",
                }
            )
    frame = pd.DataFrame(rows)

    first = module.select_paired_sample_ids(
        frame,
        task="mmstar",
        sample_count=5,
        seed=17,
    )
    second = module.select_paired_sample_ids(
        frame.sample(frac=1.0, random_state=3),
        task="mmstar",
        sample_count=5,
        seed=17,
    )

    assert first == second
    assert len(first) == len(set(first)) == 5


def test_curve_configs_share_exact_physical_budget() -> None:
    module = _module(
        "mosaickv_test_run_hf_byte_curves",
        "run_hf_byte_matched_curves_worker.py",
    )
    package_root = Path(__file__).parents[2]
    template = module.load_config(package_root / "configs/paper_candidate_llava_v1.yaml")

    configs = module.curve_configs(
        template,
        task="textvqa",
        seed=20271001,
        output_length=64,
        budget_bytes=22 * 2**20,
        retention_ratio_upper_bound=0.99,
    )

    prompt = configs["prompt_attention_topk"]
    mosaic = configs["mosaickv_exact"]
    assert prompt.cache == mosaic.cache
    assert prompt.cache.budget_value == 22 * 2**20
    assert prompt.cache.budget_unit.value == "bytes"
    assert prompt.cache.block_size == 4
    assert prompt.cache.physical_budget_mandatory_policy == "last_token"
    assert prompt.model == mosaic.model
    assert prompt.dataset == mosaic.dataset
    assert prompt.generation == mosaic.generation
    assert mosaic.method.value == "mosaickv_exact"
    assert mosaic.prototypes.enabled is False
    assert mosaic.residual.enabled is False
    assert mosaic.repair.enabled is False


def _synthetic_rows(*, mismatch: bool = False) -> pd.DataFrame:
    rows = []
    for task in ("mmstar",):
        for budget_mib in (16, 22):
            for sample_index in range(2):
                for method in ("prompt_attention_topk", "mosaickv_exact"):
                    active = budget_mib * 2**20
                    if (
                        mismatch
                        and budget_mib == 22
                        and sample_index == 1
                        and method == "mosaickv_exact"
                    ):
                        active -= 4
                    rows.append(
                        {
                            "task": task,
                            "budget_mib": budget_mib,
                            "protocol_method": method,
                            "sample_id": f"{task}:{sample_index}",
                            "task_score": 1.0 if method == "mosaickv_exact" else 0.5,
                            "answer": "mosaic" if method == "mosaickv_exact" else "prompt",
                            "reference": "reference",
                            "active_kv_bytes": active,
                            "compression_time": 1.0,
                            "decode_time": 2.0,
                            "end_to_end_time": 3.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_analysis_reports_paired_equal_budget_curve(monkeypatch: Any) -> None:
    module = _module(
        "mosaickv_test_analyze_hf_byte_curves",
        "analyze_hf_byte_matched_curves.py",
    )
    monkeypatch.setattr(module, "TASKS", ("mmstar",))
    monkeypatch.setattr(module, "BUDGETS_MIB", (16, 22))

    summary, comparisons = module.summarize(
        _synthetic_rows(),
        bootstrap_seed=1,
        bootstrap_resamples=100,
    )

    assert len(summary) == 4
    assert len(comparisons) == 2
    assert comparisons["exact_per_sample_byte_match"].all()
    assert (comparisons["paired_score_delta"] == 0.5).all()


def test_analysis_rejects_realized_byte_mismatch(monkeypatch: Any) -> None:
    module = _module(
        "mosaickv_test_analyze_hf_byte_curves_mismatch",
        "analyze_hf_byte_matched_curves.py",
    )
    monkeypatch.setattr(module, "TASKS", ("mmstar",))
    monkeypatch.setattr(module, "BUDGETS_MIB", (16, 22))

    with pytest.raises(RuntimeError, match="realized active bytes differ"):
        module.summarize(
            _synthetic_rows(mismatch=True),
            bootstrap_seed=1,
            bootstrap_resamples=10,
        )
