from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"could not load test script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry_analyzer_reads_canonical_root_level_layout_trace() -> None:
    analyzer = _load_script("analyze_paper_native_retry_v2.py")
    traces = (
        {
            "packed_padding_kv_bytes": 0,
            "tier_active_cache_lengths": [37],
            "candidate_graph_routing": {
                "planner_node_count": 2_500,
                "source_node_count": 600_000,
            },
        },
        {
            "packed_padding_kv_bytes": 0,
            "tier_active_cache_lengths": [41],
            "candidate_graph_routing": {
                "planner_node_count": 3_000,
                "source_node_count": 600_000,
            },
        },
    )

    summary = analyzer._layout_summary(traces)

    assert summary["maximum_packed_padding_kv_bytes"] == 0
    assert summary["uniform_active_positions_per_layer_head"] is True
    assert summary["maximum_planner_node_count"] == 3_000
    assert summary["maximum_planner_node_fraction"] == pytest.approx(0.005)


def test_retry_analyzer_rejects_nonuniform_physical_capacity() -> None:
    analyzer = _load_script("analyze_paper_native_retry_v2.py")
    traces = (
        {
            "packed_padding_kv_bytes": 128,
            "tier_active_cache_lengths": [37, 52],
            "candidate_graph_routing": {
                "planner_node_count": 600_000,
                "source_node_count": 600_000,
            },
        },
    )

    summary = analyzer._layout_summary(traces)

    assert summary["maximum_packed_padding_kv_bytes"] == 128
    assert summary["uniform_active_positions_per_layer_head"] is False
    assert summary["maximum_planner_node_fraction"] == pytest.approx(1.0)


def test_retry_analyzer_allows_metric_only_prefix_rows() -> None:
    analyzer = _load_script("analyze_paper_native_retry_v2.py")
    from mosaickv.evaluation.results import EvaluationResult

    row = EvaluationResult.from_json_object(
        {
            "run_id": "prefix",
            "sample_id": "sample-1",
            "task": "llava_description",
            "model": "llava-hf/llava-1.5-7b-hf",
            "backend": "huggingface",
            "method": "full_kv",
            "retention_ratio": 1.0,
            "answer": "description",
            "reference": "reference",
            "task_score": None,
            "ttft": 1.0,
            "prefill_time": 1.0,
            "compression_time": 0.0,
            "decode_time": 1.0,
            "end_to_end_time": 2.0,
            "generated_tokens": 1,
            "active_kv_bytes": 1024,
            "residual_kv_bytes": 0,
            "peak_gpu_memory": 2048,
            "repair_count": 0,
            "repaired_bytes": 0,
            "token_agreement": None,
            "logit_kl": None,
            "attention_output_error": None,
            "manifest_path": "manifest.json",
            "status": "completed",
            "error": None,
        }
    )

    summary = analyzer._completed_summary((row,))

    assert summary["mean_task_score"] is None
    with pytest.raises(ValueError, match="requires sample-level task scores"):
        analyzer._required_task_score(summary, "PrefixKV")
