from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_analysis() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "analyze_vllm_paper_quality_memory.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_vllm_paper_analysis", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_analysis()


def _materialize_method(
    root: Path,
    *,
    task: str,
    method: str,
    score: float,
    active_bytes: int,
    answer: str,
) -> None:
    method_root = root / task / method / "vllm"
    run_id = f"run-{task}-{method}"
    run_root = method_root / run_id
    manifest_path = run_root / "manifest.json"
    rows: list[dict[str, Any]] = []
    for index in range(2):
        sample_id = f"{task}:{index}"
        rows.append(
            {
                "active_kv_bytes": active_bytes,
                "answer": answer,
                "backend": "vllm",
                "compression_time": 0.1,
                "decode_time": 0.2,
                "end_to_end_time": 0.4,
                "error": None,
                "manifest_path": str(manifest_path),
                "method": (
                    method
                    if method == "full_kv"
                    else f"{method}__vllm_requestwide_projection"
                ),
                "model": ANALYSIS.MODEL_ID,
                "reference": "reference",
                "repair_count": 0,
                "residual_kv_bytes": 0,
                "run_id": run_id,
                "sample_id": sample_id,
                "status": "completed",
                "task": task,
                "task_score": score,
                "ttft": 0.2,
            }
        )
        trace_root = method_root / "traces" / run_id
        trace_root.mkdir(parents=True, exist_ok=True)
        trace = {
            "run_id": run_id,
            "sample_id": sample_id,
            "prompt_sha256": f"prompt-{task}-{index}",
            "media_sha256": f"media-{task}-{index}",
            "effective_sampling_parameters": {
                "max_new_tokens": 64,
                "seed": 17,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "native_block_compaction": method != "full_kv",
            "native_scope": (
                "not_used"
                if method == "full_kv"
                else "hash_bound_common_selector_request_wide_projection"
            ),
            "selector_plan": None if method == "full_kv" else {"catalog_sha256": "a" * 64},
        }
        (trace_root / f"{index}.json").write_text(json.dumps(trace), encoding="utf-8")
    run_root.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    (run_root / "results.jsonl").write_bytes(encoded)
    manifest = {
        "source": {
            "git_sha": "1" * 40,
            "git_dirty": False,
            "canonical_eligible": True,
        },
        "model": {"id": ANALYSIS.MODEL_ID, "revision": "revision", "precision": "fp16"},
        "dataset": {"id": task, "revision": f"{task}-revision", "split": "test"},
        "generation": {"parameters_sha": "2" * 64},
        "hardware": {"gpu_count": 1, "gpu_type": "A100"},
        "software": {"vllm": "0.10.2", "torch": "2.8.0"},
        "environment": {"lock_sha256": "3" * 64},
        "execution": {
            "backend": "vllm",
            "attention_implementation": "vllm_auto",
            "cuda_graph": False,
            "seed": 17,
            "request_isolation_verified": method == "mosaickv_exact",
        },
        "artifacts": {"raw_output_sha": hashlib.sha256(encoded).hexdigest()},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _root(tmp_path: Path, *, candidate_score: float) -> Path:
    for task in ANALYSIS.TASKS:
        _materialize_method(
            tmp_path,
            task=task,
            method="full_kv",
            score=1.0,
            active_bytes=200,
            answer="full",
        )
        _materialize_method(
            tmp_path,
            task=task,
            method="prompt_attention_topk",
            score=0.5,
            active_bytes=100,
            answer="prompt",
        )
        _materialize_method(
            tmp_path,
            task=task,
            method="mosaickv_exact",
            score=candidate_score,
            active_bytes=69,
            answer="different-but-equally-scored",
        )
    return tmp_path


def test_paper_gate_uses_official_score_and_keeps_exact_answers_secondary(
    tmp_path: Path,
) -> None:
    report = ANALYSIS.analyze(
        _root(tmp_path, candidate_score=0.5),
        expected_samples=2,
        minimum_reduction=0.30,
        bootstrap_resamples=1000,
        bootstrap_seed=7,
        require_clean_source=True,
    )

    assert report["passed"] is True
    assert report["canonical_eligible"] is True
    assert all(task["answer_agreement_with_prompt_topk"] == 0 for task in report["task_results"])
    assert all(
        task["minimum_active_memory_reduction_fraction"] > 0.30
        for task in report["task_results"]
    )


def test_paper_gate_rejects_negative_official_score_delta(tmp_path: Path) -> None:
    report = ANALYSIS.analyze(
        _root(tmp_path, candidate_score=0.0),
        expected_samples=2,
        minimum_reduction=0.30,
        bootstrap_resamples=1000,
        bootstrap_seed=7,
        require_clean_source=True,
    )

    assert report["passed"] is False
    assert any("mean official-score difference" in value for value in report["failures"])
