#!/usr/bin/env python3
"""Run one task of the narrow FullKV/Prompt-Top-K/MosaicKV GPU preflight."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import (
    ForecastingConfig,
    GraphConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    SelectionConfig,
    canonical_config_json,
    config_sha256,
    load_config,
)
from mosaickv.evaluation.paper_candidate import (
    PAPER_CANDIDATE_MODEL_ID,
    PAPER_CANDIDATE_STORAGE_QUANTUM_BYTES,
    PAPER_CANDIDATE_TASK_ROUTES,
    PAPER_CANDIDATE_TASKS,
    PaperCandidateContract,
)
from mosaickv.hf_cli import _model_kwargs, run_hf_evaluation
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import ForecastMode, MosaicKVMethod, RepairPolicy

METHOD_ORDER = ("full_kv", "prompt_attention_topk", "mosaickv_full")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("sample_ids"), list):
        raise ValueError(f"invalid sample ledger: {path}")
    sample_ids = cast("list[object]", value["sample_ids"])
    if not sample_ids or len({str(item) for item in sample_ids}) != len(sample_ids):
        raise ValueError(f"sample ledger IDs must be nonempty and unique: {path}")
    return cast("dict[str, Any]", value), _sha(path)


def _source_snapshot(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "git_dirty": bool(status.strip())}


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"refusing to mutate immutable preflight input: {path}")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


def _write_telemetry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def _require_clean(repository: Path) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("paper-candidate preflight requires a clean worktree")


def _task(index: int) -> str:
    if not 0 <= index < len(PAPER_CANDIDATE_TASKS):
        raise ValueError(f"task index must be in [0, {len(PAPER_CANDIDATE_TASKS) - 1}]")
    return PAPER_CANDIDATE_TASKS[index]


def _control_configs(template: Any, *, task: str, seed: int, output_length: int) -> dict[str, Any]:
    route = PAPER_CANDIDATE_TASK_ROUTES[task]
    common: dict[str, Any] = {
        "dataset": route,
        "execution": replace(template.execution, seed=seed),
        "generation": replace(template.generation, max_new_tokens=output_length),
        "forecasting": ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=16,
            draft_steps=0,
        ),
        "graph": GraphConfig(enabled=False),
        "selection": SelectionConfig(enabled=False),
        "prototypes": PrototypeConfig(enabled=False),
        "residual": ResidualConfig(enabled=False),
        "repair": RepairConfig(
            enabled=False,
            policy=RepairPolicy.NONE,
            max_blocks_per_step=0,
        ),
    }
    prompt = replace(
        template,
        method=MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        cache=replace(
            template.cache,
            budget_value=2**63 - 1,
            total_storage_budget_bytes=None,
        ),
        **common,
    )
    full = replace(
        template,
        method=MosaicKVMethod.FULL_KV,
        cache=replace(
            template.cache,
            budget_value=2**63 - 1,
            retention_ratio=1.0,
            total_storage_budget_bytes=None,
        ),
        **common,
    )
    return {"full_kv": full, "prompt_attention_topk": prompt}


def _storage_quantum(adapter: Any) -> tuple[int, dict[str, int | str]]:
    language_model = getattr(adapter.model, "language_model", None)
    config = getattr(language_model, "config", None)
    if config is None:
        config = getattr(adapter.model.config, "text_config", None)
    if config is None:
        raise RuntimeError("cannot resolve the LLaVA language-model configuration")
    layers = int(config.num_hidden_layers)
    attention_heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(config, "head_dim", int(config.hidden_size) // attention_heads))
    parameter = next(adapter.model.parameters())
    element_bytes = int(parameter.element_size())
    quantum = layers * kv_heads * head_dim * 2 * element_bytes
    metadata: dict[str, int | str] = {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "key_value_tensors": 2,
        "element_bytes": element_bytes,
        "dtype": str(parameter.dtype),
        "dense_position_storage_quantum_bytes": quantum,
    }
    if quantum != PAPER_CANDIDATE_STORAGE_QUANTUM_BYTES:
        raise RuntimeError(
            "pinned LLaVA layout differs from the preregistered storage quantum: "
            f"{quantum} != {PAPER_CANDIDATE_STORAGE_QUANTUM_BYTES}"
        )
    return quantum, metadata


def _single_ledger(
    ledger: dict[str, Any], *, sample_id: str, path: Path
) -> tuple[Path, str]:
    single = dict(ledger)
    single.update({"sample_count": 1, "sample_ids": [sample_id]})
    _write_immutable(path, single)
    return path, _sha(path)


def _only_row(record: dict[str, Any], *, sample_id: str) -> dict[str, Any]:
    rows = cast("list[dict[str, Any]]", record["rows"])
    if len(rows) != 1 or rows[0].get("sample_id") != sample_id:
        raise RuntimeError(f"run did not return the expected sample: {sample_id}")
    row = rows[0]
    if row.get("status") != "completed":
        raise RuntimeError(f"run did not complete: {record['run_id']}")
    return row


def _write_config(config: Any, path: Path) -> None:
    payload = yaml.safe_dump(json.loads(canonical_config_json(config)), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to mutate resolved run configuration: {path}")
    if not path.exists():
        path.write_text(payload, encoding="utf-8")


def _record(
    config: Any,
    adapter: Any,
    *,
    task: str,
    run_id: str,
    output_root: Path,
    ledger_path: Path,
    ledger_sha: str,
    sample_ids: tuple[str, ...],
    phase: str,
    repetition: int,
    order: int,
) -> dict[str, Any]:
    if config.diagnostics.enabled:
        raise RuntimeError("paper-candidate preflight requires online-only diagnostics")
    config_path = output_root / "configs" / f"{run_id}-{config_sha256(config)}.yaml"
    _write_config(config, config_path)
    arguments = argparse.Namespace(
        task=task,
        run_id=run_id,
        output_dir=str(output_root),
        raw_output=None,
        parquet_output=None,
        manifest=None,
        trace_directory=str(output_root / "traces"),
        subset_size=None,
        full_dataset=False,
        sample_offset=0,
        sample_id_file=str(ledger_path),
        sample_id_file_sha=ledger_sha,
        local_files_only=True,
    )
    status, summary = run_hf_evaluation(arguments, config, adapter=adapter)
    failed_samples = summary.get("failed_samples", 0)
    if status or failed_samples:
        raise RuntimeError(f"run failed or contains failed samples: {run_id}: {summary}")
    frame = pd.read_parquet(cast("str", summary["parquet_output"]))
    if (
        len(frame) != len(sample_ids)
        or set(str(value) for value in frame["sample_id"]) != set(sample_ids)
        or set(str(value) for value in frame["status"]) != {"completed"}
    ):
        raise RuntimeError(f"run is incomplete: {run_id}")
    rows = [
        {
            "sample_id": str(row.sample_id),
            "answer": str(row.answer),
            "reference": str(row.reference),
            "task_score": float(row.task_score),
            "active_kv_bytes": int(row.active_kv_bytes),
            "residual_kv_bytes": int(row.residual_kv_bytes),
            "compression_time": float(row.compression_time),
            "decode_time": float(row.decode_time),
            "end_to_end_time": float(row.end_to_end_time),
            "generated_tokens": int(row.generated_tokens),
            "repair_count": int(row.repair_count),
            "status": str(row.status),
            "error": None if pd.isna(row.error) else str(row.error),
            "manifest_path": str(row.manifest_path),
        }
        for row in frame.itertuples()
    ]
    return {
        "phase": phase,
        "repetition": repetition,
        "order": order,
        "method": config.method.value,
        "run_id": run_id,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    package_root = repository / "mosaickv"
    output_root = args.output_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = cast("dict[str, Any]", yaml.safe_load(protocol_path.read_text()))
    if protocol.get("experiment_id") != "paper_quality_memory_v1":
        raise RuntimeError("worker requires paper_quality_memory_v1")
    authorization = cast("dict[str, Any]", protocol["execution_authorization"])
    if authorization.get("gpu_preflight_authorized") is not True:
        raise PermissionError("GPU preflight is not authorized")
    if authorization.get("paper_experiment_authorized") is not False:
        raise RuntimeError("worker refuses a protocol that authorizes the paper campaign")
    preflight = cast("dict[str, Any]", protocol["preflight"])
    if tuple(cast("list[str]", preflight["methods"])) != METHOD_ORDER:
        raise RuntimeError("preflight method order changed")
    _require_clean(repository)
    source_snapshot = _source_snapshot(repository)
    expected_sha = os.environ.get("MOSAICKV_PAPER_CANDIDATE_EXPECTED_GIT_SHA")
    if not expected_sha or source_snapshot["commit"] != expected_sha:
        raise RuntimeError("worker source SHA does not match the immutable submission SHA")

    task = _task(args.task_index)
    route = PAPER_CANDIDATE_TASK_ROUTES[task]
    ledger_path = output_root / "inputs" / f"{task}.json"
    ledger, ledger_sha = _load_ledger(ledger_path)
    expected_ledger = {
        "task": task,
        "dataset_id": route.id,
        "dataset_revision": route.revision,
        "split": route.split,
        "selection_seed": int(preflight["selection_seed"]),
        "sample_count": int(preflight["sample_count_per_task"]),
        "source_git_sha": expected_sha,
        "source_git_dirty": False,
        "canonical_eligible": True,
        "classification": "fresh_disjoint_heldout_quality_validation",
        "selection_freeze_sha256": _sha(protocol_path),
    }
    if any(ledger.get(key) != value for key, value in expected_ledger.items()):
        raise RuntimeError(f"{task} ledger differs from the immutable preflight protocol")
    sample_ids = tuple(str(value) for value in cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != int(preflight["sample_count_per_task"]):
        raise RuntimeError(f"{task} sample count changed")

    template_path = package_root / "configs/paper_candidate_llava_v1.yaml"
    template = load_config(template_path)
    if template.model.id != PAPER_CANDIDATE_MODEL_ID:
        raise RuntimeError("candidate template changed model")
    controls = _control_configs(
        template,
        task=task,
        seed=int(template.execution.seed),
        output_length=int(preflight["output_length"]),
    )
    worker_id = f"paper-v1-{task}-{expected_sha[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    if telemetry_path.exists():
        recorded = json.loads(telemetry_path.read_text(encoding="utf-8"))
        if recorded.get("status") == "completed":
            print(f"paper-candidate preflight already completed: {worker_id}", flush=True)
            return 0
        raise RuntimeError(f"immutable worker telemetry already exists: {telemetry_path}")

    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "fresh_disjoint_heldout_quality_validation",
        "canonical_eligible": True,
        "worker_id": worker_id,
        "task": task,
        "task_index": args.task_index,
        "source_snapshot": source_snapshot,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "candidate_config": str(template_path),
        "candidate_config_sha256": _sha(template_path),
        "ledger": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "dataset_id": route.id,
        "dataset_revision": route.revision,
        "dataset_split": route.split,
        "sample_ids": list(sample_ids),
        "methods": list(METHOD_ORDER),
        "primary_gates": ["active_kv_bytes_reduction_gt_30_percent", "task_score_noninferiority"],
        "paper_experiment_authorized": False,
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
        "runs": [],
        "paired_samples": [],
        "status": "started",
    }
    try:
        adapter = load_hf_adapter(
            template.model.id,
            revision=template.model.revision,
            model_kwargs=_model_kwargs(template, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        quantum, storage_layout = _storage_quantum(adapter)
        telemetry["storage_layout"] = storage_layout

        run_records = cast("list[dict[str, Any]]", telemetry["runs"])
        for method in ("full_kv", "prompt_attention_topk"):
            record = _record(
                controls[method],
                adapter,
                task=task,
                run_id=f"{worker_id}-{method}",
                output_root=output_root,
                ledger_path=ledger_path,
                ledger_sha=ledger_sha,
                sample_ids=sample_ids,
                phase="preflight",
                repetition=0,
                order=METHOD_ORDER.index(method),
            )
            run_records.append(record)
            _settle(adapter)

        prompt_rows = {
            str(row["sample_id"]): row
            for row in cast("list[dict[str, Any]]", run_records[1]["rows"])
        }
        full_rows = {
            str(row["sample_id"]): row
            for row in cast("list[dict[str, Any]]", run_records[0]["rows"])
        }
        contract = PaperCandidateContract()
        paired_samples = cast("list[dict[str, Any]]", telemetry["paired_samples"])
        for sample_index, sample_id in enumerate(sample_ids):
            prompt_row = prompt_rows[sample_id]
            baseline_active_bytes = int(prompt_row["active_kv_bytes"])
            candidate = contract.resolve_run_config(
                replace(
                    template,
                    execution=replace(template.execution, seed=int(template.execution.seed)),
                    generation=replace(
                        template.generation,
                        max_new_tokens=int(preflight["output_length"]),
                    ),
                ),
                task=task,
                baseline_active_bytes=baseline_active_bytes,
                storage_quantum_bytes=quantum,
            )
            single_path, single_sha = _single_ledger(
                ledger,
                sample_id=sample_id,
                path=output_root / "sample_ledgers" / task / f"{sample_index:03d}.json",
            )
            record = _record(
                candidate,
                adapter,
                task=task,
                run_id=f"{worker_id}-sample-{sample_index:03d}-mosaickv-full",
                output_root=output_root,
                ledger_path=single_path,
                ledger_sha=single_sha,
                sample_ids=(sample_id,),
                phase="preflight",
                repetition=0,
                order=2,
            )
            run_records.append(record)
            candidate_row = _only_row(record, sample_id=sample_id)
            candidate_active_bytes = int(candidate_row["active_kv_bytes"])
            reduction = 1 - candidate_active_bytes / baseline_active_bytes
            paired_samples.append(
                {
                    "sample_id": sample_id,
                    "full_kv": full_rows[sample_id],
                    "prompt_attention_topk": prompt_row,
                    "mosaickv_full": candidate_row,
                    "resolved_candidate_active_cap_bytes": int(candidate.cache.budget_value),
                    "active_reduction_fraction": reduction,
                    "memory_gate_passed": reduction > 0.30,
                    "score_delta": float(candidate_row["task_score"])
                    - float(prompt_row["task_score"]),
                }
            )
            _settle(adapter)
        telemetry["status"] = "completed"
    except Exception as error:
        telemetry["status"] = "failed"
        telemetry["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        telemetry["after_measurement"] = asdict(capture_gpu_environment(torch))
        _write_telemetry(telemetry_path, telemetry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
