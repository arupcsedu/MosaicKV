from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd  # type: ignore[import-untyped]


def _load_script() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "analyze_choice_c_v73_label_blind_failures.py"
    )
    spec = importlib.util.spec_from_file_location("mosaickv_test_v73_blind_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_script()


def test_audit_projects_away_labels_and_accepts_empty_prototypes(
    monkeypatch, tmp_path: Path
) -> None:
    rows = []
    for method, answer in (
        ("full_kv", "alpha"),
        ("prompt_attention_topk", "beta"),
        ("mosaickv_full", "alpha"),
    ):
        rows.append(
            {
                "task": "docvqa",
                "method": method,
                "repetition": 0,
                "sample_id": "docvqa:1",
                "answer": answer,
                "active_kv_bytes": 10,
                "residual_kv_bytes": 0,
                "compression_time": 0.1,
                "end_to_end_time": 1.0,
                "status": "completed",
                "reference": "forbidden-label",
                "task_score": 1.0,
            }
        )
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame(rows).to_parquet(analysis / "heldout_trials.parquet", index=False)
    trace_dir = tmp_path / "traces/v73-docvqa-trial-00-mosaickv_full"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.json").write_text(
        json.dumps(
            {
                "sample_id": "docvqa:1",
                "graph_diagnostics": {"node_count": 4, "edge_counts": []},
                "candidate_graph_routing": {
                    "planner_node_count": 4,
                    "external_stable_capacity": {
                        "route_count": 1,
                        "routes_with_shortfall": [],
                    },
                },
                "prototype_diagnostics_by_kv_head": [],
                "repair_events": [],
            }
        )
    )
    original = AUDIT.pd.read_parquet

    def checked_read(path, *, columns):
        assert "reference" not in columns
        assert "task_score" not in columns
        return original(path, columns=columns)

    monkeypatch.setattr(AUDIT.pd, "read_parquet", checked_read)
    result = AUDIT.audit(tmp_path, tasks=("docvqa",), run_prefix="v73")

    assert result["benchmark_labels_accessed"] is False
    task = result["tasks"]["docvqa"]
    assert task["paired_output_agreement"]["mosaic_equals_fullkv"] == 1
    assert task["trace_mechanisms"]["samples_without_prototypes"] == 1
