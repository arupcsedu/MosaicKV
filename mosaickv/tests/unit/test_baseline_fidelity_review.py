from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]


def _module(name: str):  # type: ignore[no-untyped-def]
    path = Path(__file__).parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_fail_closed_and_has_two_families() -> None:
    module = _module("prepare_baseline_fidelity_review")
    root = Path(__file__).parents[2]
    protocol = module.load_protocol(root / "configs/experiments/baseline_fidelity_review_v1.yaml")
    assert tuple(protocol["families"]) == ("prefixkv", "vl_cache")
    assert protocol["evaluation"]["budget_unit"] == "bytes"
    assert protocol["controls"]["primary_budget_reported_as_active_kv_bytes"] is True


def test_identity_selection_reads_only_completed_fullkv_rows() -> None:
    module = _module("prepare_baseline_fidelity_review")
    frame = pd.DataFrame(
        [
            {
                "task": "textvqa",
                "sample_id": f"sample-{index}",
                "backend_scope": "hf_eager",
                "protocol_method": "full_kv",
                "status": "completed",
            }
            for index in range(5)
        ]
        + [
            {
                "task": "textvqa",
                "sample_id": "bad",
                "backend_scope": "vllm",
                "protocol_method": "full_kv",
                "status": "completed",
            }
        ]
    )
    first = module.select_ids(frame, task="textvqa", count=3, seed=7)
    second = module.select_ids(frame, task="textvqa", count=3, seed=7)
    assert first == second
    assert "bad" not in first


def test_sample_ledger_identity_matches_hf_evaluator_schema() -> None:
    module = _module("prepare_baseline_fidelity_review")
    identity = module.sample_ledger_identity(
        "textvqa",
        {
            "dataset_id": "lmms-lab/textvqa",
            "dataset_revision": "revision-sha",
            "split": "validation",
        },
    )
    assert identity == {
        "task": "textvqa",
        "dataset_id": "lmms-lab/textvqa",
        "dataset_revision": "revision-sha",
        "split": "validation",
    }
