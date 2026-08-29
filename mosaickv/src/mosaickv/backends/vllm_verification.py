"""Fail-closed LLaVA-1.5 HF/vLLM verification for the paper task subset."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mosaickv.evaluation.results import EvaluationResult, ResultStatus
from mosaickv.evaluation.storage import load_jsonl
from mosaickv.types import JsonObject, JsonValue

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
VLLM_VERSION = "0.10.2"
TASK_DATASETS: dict[str, tuple[str, str, str]] = {
    "mmstar": ("Lin-Chen/MMStar", "bc98d668301da7b14f648724866e57302778ab27", "val"),
    "textvqa": (
        "lmms-lab/textvqa",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "validation",
    ),
    "docvqa": (
        "lmms-lab/DocVQA",
        "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "validation",
    ),
}


class VLLMVerificationError(RuntimeError):
    """Raised when a purported verification artifact violates its contract."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VLLMVerificationError(f"{label} must be a JSON object")
    return cast("dict[str, Any]", value)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise VLLMVerificationError(f"cannot read JSON artifact {path}: {error}") from error


def _one_manifest(run_base: Path) -> tuple[Path, dict[str, Any]]:
    manifests = sorted(run_base.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise VLLMVerificationError(
            f"expected exactly one immutable manifest below {run_base}, found {len(manifests)}"
        )
    return manifests[0], _load_object(manifests[0])


def _validate_manifest(
    manifest: Mapping[str, Any], *, task: str, backend: str, expected_run_id: str
) -> None:
    dataset_id, dataset_revision, split = TASK_DATASETS[task]
    expected_backend = "huggingface" if backend == "hf" else "vllm"
    expected = {
        "run_id": expected_run_id,
        "measurement_type": "reference_measured",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise VLLMVerificationError(
                f"{backend}/{task} manifest {key}={manifest.get(key)!r}, expected {value!r}"
            )
    model = _object(manifest.get("model"), f"{backend}/{task}.model")
    if (model.get("id"), model.get("revision"), model.get("precision")) != (
        MODEL_ID,
        MODEL_REVISION,
        "bf16",
    ):
        raise VLLMVerificationError(f"{backend}/{task} model provenance is not pinned")
    dataset = _object(manifest.get("dataset"), f"{backend}/{task}.dataset")
    if (dataset.get("id"), dataset.get("revision"), dataset.get("split")) != (
        dataset_id,
        dataset_revision,
        split,
    ):
        raise VLLMVerificationError(f"{backend}/{task} dataset provenance is not pinned")
    source = _object(manifest.get("source"), f"{backend}/{task}.source")
    if source.get("git_dirty") is not False or source.get("canonical_eligible") is not True:
        raise VLLMVerificationError(f"{backend}/{task} was not executed from a clean worktree")
    execution = _object(manifest.get("execution"), f"{backend}/{task}.execution")
    if execution.get("backend") != expected_backend or execution.get("seed") != 17:
        raise VLLMVerificationError(f"{backend}/{task} execution controls differ")
    if backend == "hf":
        if execution.get("attention_implementation") != "eager":
            raise VLLMVerificationError("HF reference must use eager attention")
    else:
        if (
            execution.get("engine_execution_mode") != "v1_eager"
            or execution.get("cuda_graph") is not False
            or execution.get("prefix_caching") is not False
            or execution.get("request_isolation_verified") is not True
        ):
            raise VLLMVerificationError(
                f"vLLM/{task} eager, uncached, isolated execution was not verified"
            )
        software = _object(manifest.get("software"), f"{backend}/{task}.software")
        if software.get("vllm") != VLLM_VERSION:
            raise VLLMVerificationError(
                f"vLLM/{task} used {software.get('vllm')!r}, expected {VLLM_VERSION!r}"
            )


def _validate_rows(
    rows: Sequence[EvaluationResult], *, task: str, backend: str, expected_samples: int
) -> dict[str, EvaluationResult]:
    if len(rows) != expected_samples:
        raise VLLMVerificationError(
            f"{backend}/{task} produced {len(rows)} rows, expected {expected_samples}"
        )
    by_id: dict[str, EvaluationResult] = {}
    expected_backend = "huggingface" if backend == "hf" else "vllm"
    for row in rows:
        if row.sample_id in by_id:
            raise VLLMVerificationError(f"duplicate {backend}/{task} sample {row.sample_id}")
        if (
            row.status is not ResultStatus.COMPLETED
            or row.error is not None
            or row.task != task
            or row.model != MODEL_ID
            or row.backend != expected_backend
            or row.method != "full_kv"
            or row.retention_ratio != 1.0
        ):
            raise VLLMVerificationError(
                f"invalid terminal row for {backend}/{task}/{row.sample_id}"
            )
        for field in ("ttft", "prefill_time", "decode_time", "end_to_end_time"):
            value = getattr(row, field)
            if value is None or not math.isfinite(value) or value < 0:
                raise VLLMVerificationError(
                    f"{backend}/{task}/{row.sample_id} lacks finite {field}"
                )
        if (
            row.task_score is None
            or not math.isfinite(row.task_score)
            or row.generated_tokens is None
            or row.generated_tokens < 1
            or row.active_kv_bytes is None
            or row.active_kv_bytes < 1
            or row.residual_kv_bytes != 0
            or row.compression_time != 0.0
            or row.repair_count != 0
            or row.repaired_bytes != 0
        ):
            raise VLLMVerificationError(
                f"{backend}/{task}/{row.sample_id} has incomplete FullKV metrics"
            )
        if backend == "vllm" and (row.peak_gpu_memory is None or row.peak_gpu_memory < 1):
            raise VLLMVerificationError(
                f"vLLM/{task}/{row.sample_id} lacks GPU memory measurement"
            )
        by_id[row.sample_id] = row
    return by_id


def _trace_map(trace_directory: Path, *, expected_run_id: str) -> dict[str, dict[str, Any]]:
    trace_root = trace_directory / expected_run_id
    traces: dict[str, dict[str, Any]] = {}
    for path in sorted(trace_root.glob("*.json")):
        payload = _load_object(path)
        sample_id = payload.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise VLLMVerificationError(f"trace lacks sample_id: {path}")
        if sample_id in traces:
            raise VLLMVerificationError(f"duplicate trace for sample {sample_id}")
        traces[sample_id] = payload
    return traces


def _validate_vllm_trace(
    trace: Mapping[str, Any], *, row: EvaluationResult
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        trace.get("status") != "completed"
        or trace.get("measurement_type") != "vllm_fullkv"
        or trace.get("backend") != "vllm"
        or trace.get("method") != "full_kv"
        or trace.get("native_mosaickv") is not False
    ):
        raise VLLMVerificationError(f"invalid vLLM trace for {row.sample_id}")
    trials = trace.get("trials")
    if not isinstance(trials, list) or len(trials) < 2:
        raise VLLMVerificationError(f"vLLM trace {row.sample_id} lacks repeated trials")
    token_sequences: list[tuple[int, ...]] = []
    for trial in trials:
        trial_object = _object(trial, f"vLLM trial {row.sample_id}")
        token_ids = trial_object.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise VLLMVerificationError(f"vLLM trial {row.sample_id} lacks token IDs")
        token_sequences.append(tuple(int(token) for token in token_ids))
        for field in (
            "ttft_seconds",
            "request_latency_seconds",
            "throughput_tokens_per_second",
            "engine_prefill_seconds",
            "engine_decode_seconds",
            "engine_ttft_seconds",
            "gpu_memory_baseline_bytes",
            "gpu_memory_peak_bytes",
            "num_cached_tokens",
            "prefix_cache_hit_rate",
            "mm_cache_queries",
            "mm_cache_hits",
            "mm_cache_hit_rate",
        ):
            if trial_object.get(field) is None:
                raise VLLMVerificationError(
                    f"vLLM trial {row.sample_id} lacks measured field {field}"
                )
    if any(tokens != token_sequences[0] for tokens in token_sequences[1:]):
        raise VLLMVerificationError(f"vLLM repeated tokens differ for {row.sample_id}")
    accounting = _object(trace.get("kv_cache_accounting"), f"vLLM accounting {row.sample_id}")
    if (
        accounting.get("boundary") != "packed_prefill_active_kv_tensor_storage"
        or accounting.get("active_kv_bytes") != row.active_kv_bytes
        or accounting.get("includes_live_block_padding") is not True
        or accounting.get("excludes_free_block_pool") is not True
    ):
        raise VLLMVerificationError(f"vLLM active-KV accounting differs for {row.sample_id}")
    tokens = token_sequences[0]
    return tokens, _quality_token_prefix(trace, tokens, label=f"vLLM/{row.sample_id}")


def _quality_token_prefix(
    trace: Mapping[str, Any], tokens: tuple[int, ...], *, label: str
) -> tuple[int, ...]:
    """Return the decoded completion through the first terminal token.

    Fixed-length timing runs intentionally continue after EOS. Tokens sampled
    after that boundary are timing filler and are not part of the decoded
    answer. Cross-backend parity therefore gates this prefix while preserving
    the complete timed-sequence comparison as a separate diagnostic.
    """

    decoding = _object(trace.get("quality_decoding"), f"{label}.quality_decoding")
    terminal_index = decoding.get("terminal_token_index")
    if terminal_index is None:
        quality_tokens = tokens
    else:
        if (
            isinstance(terminal_index, bool)
            or not isinstance(terminal_index, int)
            or terminal_index < 0
            or terminal_index >= len(tokens)
        ):
            raise VLLMVerificationError(f"{label} has an invalid terminal-token index")
        quality_tokens = tokens[: terminal_index + 1]
    recorded_count = decoding.get("quality_completion_tokens")
    if recorded_count != len(quality_tokens):
        raise VLLMVerificationError(
            f"{label} quality-token count {recorded_count!r} differs from {len(quality_tokens)}"
        )
    return quality_tokens


def _validate_hf_trace(
    trace: Mapping[str, Any], *, row: EvaluationResult
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        trace.get("status") != "completed"
        or trace.get("backend") != "huggingface"
        or trace.get("method") != "full_kv"
        or trace.get("retention_ratio") != 1.0
    ):
        raise VLLMVerificationError(f"invalid HF trace for {row.sample_id}")
    raw_tokens = trace.get("generated_token_ids")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise VLLMVerificationError(f"HF trace {row.sample_id} lacks generated token IDs")
    tokens = tuple(int(token) for token in raw_tokens)
    return tokens, _quality_token_prefix(trace, tokens, label=f"HF/{row.sample_id}")


def verify_task_pair(task_root: Path, *, task: str, expected_samples: int) -> JsonObject:
    """Verify one task's paired HF/vLLM FullKV artifacts."""

    if task not in TASK_DATASETS:
        raise VLLMVerificationError(f"unsupported verification task {task!r}")
    backends: dict[str, tuple[dict[str, Any], dict[str, EvaluationResult], dict[str, Any]]] = {}
    for backend in ("hf", "vllm"):
        run_base = task_root / backend
        manifest_path, manifest = _one_manifest(run_base)
        run_id = str(manifest.get("run_id", ""))
        _validate_manifest(manifest, task=task, backend=backend, expected_run_id=run_id)
        rows = _validate_rows(
            load_jsonl(manifest_path.parent / "results.jsonl"),
            task=task,
            backend=backend,
            expected_samples=expected_samples,
        )
        traces = _trace_map(run_base / "traces", expected_run_id=run_id)
        if set(traces) != set(rows):
            raise VLLMVerificationError(
                f"{backend}/{task} trace IDs do not match result IDs: "
                f"{sorted(set(rows) ^ set(traces))}"
            )
        backends[backend] = (manifest, rows, traces)

    _, hf_rows, hf_traces = backends["hf"]
    _, vllm_rows, vllm_traces = backends["vllm"]
    if set(hf_rows) != set(vllm_rows):
        raise VLLMVerificationError(f"HF/vLLM sample IDs differ for {task}")
    sample_records: list[JsonValue] = []
    quality_token_matches = 0
    timed_token_matches = 0
    answer_matches = 0
    score_matches = 0
    for sample_id in sorted(hf_rows):
        hf_row = hf_rows[sample_id]
        vllm_row = vllm_rows[sample_id]
        hf_tokens, hf_quality_tokens = _validate_hf_trace(
            hf_traces[sample_id], row=hf_row
        )
        vllm_tokens, vllm_quality_tokens = _validate_vllm_trace(
            vllm_traces[sample_id], row=vllm_row
        )
        quality_token_match = hf_quality_tokens == vllm_quality_tokens
        timed_token_match = hf_tokens == vllm_tokens
        answer_match = hf_row.answer == vllm_row.answer
        score_match = hf_row.task_score == vllm_row.task_score
        quality_token_matches += int(quality_token_match)
        timed_token_matches += int(timed_token_match)
        answer_matches += int(answer_match)
        score_matches += int(score_match)
        sample_records.append(
            {
                "sample_id": sample_id,
                "quality_token_ids_match": quality_token_match,
                "timed_token_ids_match": timed_token_match,
                "hf_quality_token_count": len(hf_quality_tokens),
                "vllm_quality_token_count": len(vllm_quality_tokens),
                "answer_match": answer_match,
                "task_score_match": score_match,
                "hf_task_score": hf_row.task_score,
                "vllm_task_score": vllm_row.task_score,
                "vllm_active_kv_bytes": vllm_row.active_kv_bytes,
            }
        )
    isolation_verified = any(
        _object(trace.get("request_isolation"), "request isolation").get(
            "token_ids_match_anchor"
        )
        is True
        for trace in vllm_traces.values()
        if isinstance(trace.get("request_isolation"), dict)
    )
    parity_passed = (
        quality_token_matches == expected_samples
        and answer_matches == expected_samples
        and score_matches == expected_samples
    )
    return {
        "task": task,
        "expected_samples": expected_samples,
        "completed_samples": len(sample_records),
        "request_isolation_verified": isolation_verified,
        "quality_token_parity_count": quality_token_matches,
        "timed_token_parity_count": timed_token_matches,
        "post_terminal_timed_token_parity_is_gate": False,
        "answer_parity_count": answer_matches,
        "score_parity_count": score_matches,
        "parity_passed": parity_passed,
        "status": "passed" if parity_passed and isolation_verified else "failed",
        "samples": sample_records,
    }


def verify_campaign(
    root: str | Path,
    *,
    tasks: Sequence[str] = tuple(TASK_DATASETS),
    expected_samples: int = 2,
) -> JsonObject:
    """Verify one or more task pairs and return a machine-readable verdict."""

    if expected_samples < 2:
        raise ValueError("expected_samples must be >= 2 to verify request isolation")
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be a non-empty unique sequence")
    root_path = Path(root).resolve()
    task_payloads = [
        verify_task_pair(root_path / task, task=task, expected_samples=expected_samples)
        for task in tasks
    ]
    passed = all(payload["status"] == "passed" for payload in task_payloads)
    return {
        "schema_version": 2,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backends": ["huggingface", "vllm"],
        "method": "full_kv",
        "tasks": cast("list[JsonValue]", task_payloads),
        "status": "passed" if passed else "failed",
        "native_mosaickv_claimed": False,
        "sglang_used": False,
    }


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "TASK_DATASETS",
    "VLLMVerificationError",
    "verify_campaign",
    "verify_task_pair",
]
