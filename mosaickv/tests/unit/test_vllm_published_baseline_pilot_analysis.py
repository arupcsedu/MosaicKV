from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "analyze_vllm_published_baseline_pilot.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_vllm_baseline_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rows(root: Path, task: str, method: str) -> None:
    run_root = root / task / method / "vllm" / "run"
    run_root.mkdir(parents=True)
    if method == "full_kv":
        active_bytes = 160
    elif method == "mosaickv_exact":
        active_bytes = 64
    else:
        active_bytes = 96
    rows = [
        {
            "status": "completed",
            "sample_id": f"{task}:{index}",
            "task_score": 1.0,
            "active_kv_bytes": active_bytes,
            "end_to_end_time": 1.0,
            "answer": "same",
            "method": (
                method
                if method == "full_kv"
                else f"{method}__vllm_requestwide_projection"
            ),
        }
        for index in range(2)
    ]
    (run_root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_pilot_analyzer_requires_paired_memory_and_quality_gates(tmp_path: Path) -> None:
    module = _module()
    for task in module.TASKS:
        for method in module.METHODS:
            _write_rows(tmp_path, task, method)

    report = module.analyze(tmp_path)

    assert report["gate_passed"] is True
    assert len(report["summary"]) == len(module.TASKS) * len(module.METHODS)
    prompt = [
        row
        for row in report["comparisons"]
        if row["competitor"] == "prompt_attention_topk"
    ]
    assert len(prompt) == len(module.TASKS)
    assert all(row["mosaickv_memory_reduction_fraction"] > 0.30 for row in prompt)
    assert (tmp_path / "analysis" / "report.md").is_file()
