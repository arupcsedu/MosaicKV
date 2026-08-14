from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_analyzer() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "analyze_choice_c_v67_matched_sdpa.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_quality_memory_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_analyzer()


def _load_v72_analyzer() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "analyze_choice_c_v72_heldout_quality_memory.py"
    )
    spec = importlib.util.spec_from_file_location("mosaickv_test_v72_quality_memory", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V72_ANALYZER = _load_v72_analyzer()


def _rows(
    method: str, *, active_bytes: int, score: float, latency: float
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample_id,
            "answer": answer,
            "active_kv_bytes": active_bytes,
            "residual_kv_bytes": 0,
            "compression_time": 0.2,
            "decode_time": latency - 0.2,
            "end_to_end_time": latency,
            "task_score": score,
            "method": method,
        }
        for sample_id, answer in (("docvqa:1", "alpha"), ("docvqa:2", "beta"))
    ]


def test_quality_memory_gate_does_not_reject_slower_mosaickv(tmp_path: Path) -> None:
    telemetry = {
        "status": "completed",
        "classification": "unit_test",
        "source_snapshot": {"snapshot_sha256": "a" * 64},
        "sample_ids": ["docvqa:1", "docvqa:2"],
        "warmups_per_method": 1,
        "measured_repetitions_per_method": 2,
        "trials": [],
    }
    for repetition in range(2):
        for order, method in enumerate(("prompt_attention_topk", "mosaickv_full")):
            telemetry["trials"].append(
                {
                    "method": method,
                    "repetition": repetition,
                    "order": order,
                    "rows": _rows(
                        method,
                        active_bytes=70 if method == "mosaickv_full" else 100,
                        score=1.0,
                        latency=1.2 if method == "mosaickv_full" else 1.0,
                    ),
                }
            )
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    (telemetry_dir / "unit.json").write_text(json.dumps(telemetry))

    result = ANALYZER.analyze(tmp_path)

    assert result["gates"]["quality_memory_gate_pass"] is True
    assert "paper_gate_pass" not in result["gates"]
    assert "mosaickv_at_least_5_percent_faster" not in result["gates"]
    assert result["latency_report"]["policy"] == "informational_not_a_rejection_criterion"
    assert result["latency_report"]["paired_end_to_end_ratios"] == [1.2, 1.2]


def test_v72_missing_error_normalization_accepts_legacy_nan() -> None:
    assert V72_ANALYZER._is_missing_error(None)
    assert V72_ANALYZER._is_missing_error(float("nan"))
    assert V72_ANALYZER._is_missing_error("nan")
    assert not V72_ANALYZER._is_missing_error("CUDA out of memory")
