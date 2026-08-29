from __future__ import annotations

import json
from pathlib import Path

from mosaickv.backends.vllm_verification import MODEL_ID, MODEL_REVISION, verify_campaign
from mosaickv.evaluation.model import GenerationMetrics
from mosaickv.evaluation.results import EvaluationResult
from mosaickv.evaluation.storage import JsonlResultStore


def _manifest(task: str, backend: str, run_id: str) -> dict[str, object]:
    dataset = {
        "mmstar": (
            "Lin-Chen/MMStar",
            "bc98d668301da7b14f648724866e57302778ab27",
            "val",
        )
    }[task]
    execution: dict[str, object] = {
        "backend": "huggingface" if backend == "hf" else "vllm",
        "attention_implementation": "eager",
        "seed": 17,
    }
    if backend == "vllm":
        execution.update(
            {
                "engine_execution_mode": "v1_eager",
                "cuda_graph": False,
                "prefix_caching": False,
                "request_isolation_verified": True,
            }
        )
    return {
        "run_id": run_id,
        "measurement_type": "reference_measured",
        "source": {"git_dirty": False, "canonical_eligible": True},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "precision": "bf16"},
        "dataset": {"id": dataset[0], "revision": dataset[1], "split": dataset[2]},
        "software": {"vllm": "0.10.2"},
        "execution": execution,
    }


def _row(run_id: str, sample_id: str, backend: str, manifest: Path) -> EvaluationResult:
    return EvaluationResult.from_generation(
        run_id=run_id,
        sample_id=sample_id,
        task="mmstar",
        model=MODEL_ID,
        backend="huggingface" if backend == "hf" else "vllm",
        method="full_kv",
        retention_ratio=1.0,
        answer="A",
        reference="A",
        task_score=1.0,
        metrics=GenerationMetrics(
            ttft=0.1,
            prefill_time=0.08,
            compression_time=0.0,
            decode_time=0.1,
            end_to_end_time=0.2,
            generated_tokens=2,
            active_kv_bytes=1152,
            residual_kv_bytes=0,
            peak_gpu_memory=2048,
            repair_count=0,
            repaired_bytes=0,
        ),
        manifest_path=str(manifest),
    )


def _vllm_trial(tokens: list[int]) -> dict[str, object]:
    return {
        "token_ids": tokens,
        "ttft_seconds": 0.1,
        "request_latency_seconds": 0.2,
        "throughput_tokens_per_second": 10.0,
        "engine_prefill_seconds": 0.08,
        "engine_decode_seconds": 0.1,
        "engine_ttft_seconds": 0.09,
        "gpu_memory_baseline_bytes": 1024,
        "gpu_memory_peak_bytes": 2048,
        "num_cached_tokens": 0,
        "prefix_cache_hit_rate": 0.0,
        "mm_cache_queries": 1,
        "mm_cache_hits": 0,
        "mm_cache_hit_rate": 0.0,
    }


def _materialize_pair(
    root: Path,
    *,
    mismatch_quality_token: bool = False,
    mismatch_post_terminal_token: bool = False,
) -> None:
    for backend in ("hf", "vllm"):
        run_id = f"{backend}-run"
        base = root / "mmstar" / backend
        run_root = base / run_id
        manifest_path = run_root / "manifest.json"
        run_root.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(_manifest("mmstar", backend, run_id)), encoding="utf-8"
        )
        trace_root = base / "traces" / run_id
        trace_root.mkdir(parents=True)
        for index, sample_id in enumerate(("mmstar:1", "mmstar:2")):
            JsonlResultStore(run_root / "results.jsonl").append(
                _row(run_id, sample_id, backend, manifest_path)
            )
            tokens = [7, 2, 9]
            if backend == "vllm" and mismatch_quality_token and index == 1:
                tokens = [8, 2, 9]
            if backend == "vllm" and mismatch_post_terminal_token and index == 1:
                tokens = [7, 2, 10]
            quality_decoding = {
                "terminal_token_index": 1,
                "quality_completion_tokens": 2,
            }
            if backend == "hf":
                trace: dict[str, object] = {
                    "sample_id": sample_id,
                    "status": "completed",
                    "backend": "huggingface",
                    "method": "full_kv",
                    "retention_ratio": 1.0,
                    "generated_token_ids": tokens,
                    "quality_decoding": quality_decoding,
                }
            else:
                trace = {
                    "sample_id": sample_id,
                    "status": "completed",
                    "measurement_type": "vllm_fullkv",
                    "backend": "vllm",
                    "method": "full_kv",
                    "native_mosaickv": False,
                    "quality_decoding": quality_decoding,
                    "trials": [_vllm_trial(tokens), _vllm_trial(tokens)],
                    "kv_cache_accounting": {
                        "boundary": "packed_prefill_active_kv_tensor_storage",
                        "active_kv_bytes": 1152,
                        "includes_live_block_padding": True,
                        "excludes_free_block_pool": True,
                    },
                }
                if index == 0:
                    trace["request_isolation"] = {"token_ids_match_anchor": True}
            (trace_root / f"{index}.json").write_text(json.dumps(trace), encoding="utf-8")


def test_campaign_verifier_accepts_complete_paired_fullkv_artifacts(tmp_path: Path) -> None:
    _materialize_pair(tmp_path)

    payload = verify_campaign(tmp_path, tasks=("mmstar",), expected_samples=2)

    assert payload["status"] == "passed"
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    task = tasks[0]
    assert isinstance(task, dict)
    assert task["quality_token_parity_count"] == 2
    assert task["timed_token_parity_count"] == 2
    assert task["request_isolation_verified"] is True


def test_campaign_verifier_reports_quality_token_parity_failure(tmp_path: Path) -> None:
    _materialize_pair(tmp_path, mismatch_quality_token=True)

    payload = verify_campaign(tmp_path, tasks=("mmstar",), expected_samples=2)

    assert payload["status"] == "failed"
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    task = tasks[0]
    assert isinstance(task, dict)
    assert task["quality_token_parity_count"] == 1


def test_campaign_verifier_reports_but_does_not_gate_post_terminal_tokens(
    tmp_path: Path,
) -> None:
    _materialize_pair(tmp_path, mismatch_post_terminal_token=True)

    payload = verify_campaign(tmp_path, tasks=("mmstar",), expected_samples=2)

    assert payload["status"] == "passed"
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    task = tasks[0]
    assert isinstance(task, dict)
    assert task["quality_token_parity_count"] == 2
    assert task["timed_token_parity_count"] == 1
    assert task["post_terminal_timed_token_parity_is_gate"] is False
