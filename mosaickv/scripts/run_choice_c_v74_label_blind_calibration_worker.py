#!/usr/bin/env python3
"""Run one clean, label-blind versioned calibration task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from choice_c_v74_protocol import calibration_configs
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _write_telemetry
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot
from run_choice_c_v63_anchor_routing_worker import _settle, _write_immutable
from run_choice_c_v72_heldout_quality_memory_worker import _record, _resolve, _task_entry

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import load_config
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment


def _require_clean_worktree(project_root: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("v74 calibration requires a completely clean worktree")


BYTE_BUDGET_FIELDS = (
    "configured_active_byte_cap",
    "retention_requested_bytes",
    "requested_hard_budget_bytes",
    "dense_attainable_budget_bytes",
    "dense_quantization_underfill_bytes",
    "effective_active_budget_bytes",
    "capacity_fraction_reduction_bytes",
    "planner_budget_after_external_exact_bytes",
    "external_exact_selected_bytes",
    "planner_selected_bytes",
    "exact_logical_active_bytes",
    "prototype_logical_active_bytes",
    "logical_active_bytes",
    "logical_underfill_vs_effective_budget_bytes",
    "logical_underfill_vs_dense_budget_bytes",
    "final_packed_active_bytes",
    "packed_padding_bytes",
    "final_underfill_vs_dense_budget_bytes",
    "final_dense_budget_utilization",
)


def _trace_records(output_root: Path, run_id: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((output_root / "traces" / run_id).glob("*.json")):
        trace = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        sample_id = str(trace["sample_id"])
        if sample_id in result:
            raise RuntimeError(f"duplicate trace for {run_id}/{sample_id}")
        result[sample_id] = trace
    return result


def _label_blind_record(*args: Any, arm: str, output_root: Path, **kwargs: Any) -> dict[str, Any]:
    record = _record(*args, output_root=output_root, **kwargs)
    traces = _trace_records(output_root, str(record["run_id"]))
    rows = cast("list[dict[str, Any]]", record["rows"])
    if set(traces) != {str(row["sample_id"]) for row in rows}:
        raise RuntimeError(f"{arm} traces do not cover every calibration sample")
    for row in rows:
        trace = traces[str(row["sample_id"])]
        # The underlying benchmark adapter may need references to prepare its task,
        # but calibration telemetry and candidate selection never retain them.
        row.pop("reference", None)
        row.pop("task_score", None)
        row["generated_token_ids"] = [int(value) for value in trace["generated_token_ids"]]
        accounting = trace.get("byte_budget_materialization")
        if accounting is not None and not isinstance(accounting, dict):
            raise RuntimeError(f"{arm} byte-budget trace must be a mapping or null")
        for field in BYTE_BUDGET_FIELDS:
            row[f"byte_budget_{field}"] = None if accounting is None else accounting.get(field)
        routing = trace.get("candidate_graph_routing")
        if routing is not None and not isinstance(routing, dict):
            raise RuntimeError(f"{arm} candidate graph routing trace must be a mapping or null")
        external_capacity = None if routing is None else routing.get("external_stable_capacity")
        if external_capacity is not None and not isinstance(external_capacity, dict):
            raise RuntimeError(f"{arm} external stable capacity trace must be a mapping or null")
        for field in (
            "route_count",
            "requested_positions",
            "selected_positions",
            "shortfall_positions",
            "selected_bytes",
            "selected_visual_positions",
            "target_visual_positions",
            "routes_missing_visual_target",
            "adaptive_swap_count",
            "baseline_text_prompt_mass",
            "selected_text_prompt_mass",
            "normalized_visual_evidence_gain",
            "normalized_text_prompt_mass_loss",
            "normalized_forecast_visual_gain",
            "normalized_forecast_text_loss",
            "adaptive_swap_candidate_count",
            "adaptive_swap_risk_rejected_count",
            "future_rerank_count",
            "baseline_selected_prompt_mass",
            "selected_prompt_mass_after_future_rerank",
            "normalized_future_query_gain",
            "normalized_future_prompt_mass_loss",
            "future_rerank_candidate_count",
            "future_rerank_rejected_count",
            "future_rerank_risk_rejected_count",
            "forecast_divergence_eligible_route_count",
            "prompt_forecast_total_variation_min",
            "prompt_forecast_total_variation_mean",
            "prompt_forecast_total_variation_max",
            "query_coverage_rerank_count",
            "baseline_query_coverage",
            "selected_query_coverage",
            "normalized_query_coverage_gain",
            "query_coverage_baseline_prompt_mass",
            "query_coverage_selected_prompt_mass",
            "query_coverage_normalized_prompt_loss",
            "query_coverage_candidate_count",
            "query_coverage_rejected_count",
            "query_coverage_initial_reserved_count",
            "query_coverage_initial_changed_count",
            "query_coverage_initial_baseline",
            "query_coverage_initial_selected",
            "query_coverage_initial_normalized_gain",
            "query_coverage_initial_baseline_prompt_mass",
            "query_coverage_initial_selected_prompt_mass",
            "query_coverage_initial_normalized_prompt_loss",
            "query_coverage_initial_candidate_count",
        ):
            row[f"external_stable_{field}"] = (
                None if external_capacity is None else external_capacity.get(field)
            )
        shortfall_routes = (
            None if external_capacity is None else external_capacity.get("routes_with_shortfall")
        )
        row["external_stable_routes_with_shortfall_count"] = (
            None if shortfall_routes is None else len(cast("list[Any]", shortfall_routes))
        )
        adaptive_routes = (
            None
            if external_capacity is None
            else external_capacity.get("routes_with_adaptive_swaps")
        )
        risk_rejected_routes = (
            None
            if external_capacity is None
            else external_capacity.get("routes_with_risk_rejections")
        )
        row["external_stable_routes_with_adaptive_swaps_count"] = (
            None if adaptive_routes is None else len(cast("list[Any]", adaptive_routes))
        )
        row["external_stable_routes_with_risk_rejections_count"] = (
            None if risk_rejected_routes is None else len(cast("list[Any]", risk_rejected_routes))
        )
        future_rerank_routes = (
            None
            if external_capacity is None
            else external_capacity.get("routes_with_future_reranking")
        )
        future_rejected_routes = (
            None
            if external_capacity is None
            else external_capacity.get("routes_with_future_rerank_rejections")
        )
        row["external_stable_routes_with_future_reranking_count"] = (
            None if future_rerank_routes is None else len(cast("list[Any]", future_rerank_routes))
        )
        row["external_stable_routes_with_future_rerank_rejections_count"] = (
            None
            if future_rejected_routes is None
            else len(cast("list[Any]", future_rejected_routes))
        )
        future_risk_rejected_routes = (
            None
            if external_capacity is None
            else external_capacity.get("routes_with_future_rerank_risk_rejections")
        )
        row["external_stable_routes_with_future_rerank_risk_rejections_count"] = (
            None
            if future_risk_rejected_routes is None
            else len(cast("list[Any]", future_risk_rejected_routes))
        )
        query_coverage_routes = (
            None
            if external_capacity is None
            else external_capacity.get("routes_with_query_coverage_reranking")
        )
        query_coverage_rejected_routes = (
            None
            if external_capacity is None
            else external_capacity.get("routes_with_query_coverage_rejections")
        )
        row["external_stable_routes_with_query_coverage_reranking_count"] = (
            None if query_coverage_routes is None else len(cast("list[Any]", query_coverage_routes))
        )
        row["external_stable_routes_with_query_coverage_rejections_count"] = (
            None
            if query_coverage_rejected_routes is None
            else len(cast("list[Any]", query_coverage_rejected_routes))
        )
        initial_query_coverage_routes = (
            None
            if external_capacity is None
            else external_capacity.get("routes_with_query_coverage_initial_selection")
        )
        row["external_stable_routes_with_query_coverage_initial_selection_count"] = (
            None
            if initial_query_coverage_routes is None
            else len(cast("list[Any]", initial_query_coverage_routes))
        )
        active_lengths = trace.get("tier_active_cache_lengths")
        if active_lengths is not None and not isinstance(active_lengths, list):
            raise RuntimeError(f"{arm} tier active cache lengths must be a list or null")
        row["tier_active_cache_length_count"] = (
            None if active_lengths is None else len(active_lengths)
        )
        row["tier_active_cache_length_min"] = (
            None if not active_lengths else min(int(value) for value in active_lengths)
        )
        row["tier_active_cache_length_max"] = (
            None if not active_lengths else max(int(value) for value in active_lengths)
        )
        prompt_mass = trace.get("prompt_mass_audit")
        if prompt_mass is not None and not isinstance(prompt_mass, dict):
            raise RuntimeError(f"{arm} prompt-mass audit must be a mapping or null")
        for field in (
            "source_block_count",
            "exact_block_count",
            "prototype_source_block_count",
            "dropped_block_count",
            "total_prompt_mass",
            "exact_prompt_mass",
            "prototype_source_prompt_mass",
            "exact_prompt_mass_fraction",
            "represented_prompt_mass_fraction",
        ):
            row[f"prompt_mass_{field}"] = None if prompt_mass is None else prompt_mass.get(field)
        prototype_diagnostics = trace.get("prototype_diagnostics_by_kv_head")
        if prototype_diagnostics is not None and not isinstance(prototype_diagnostics, list):
            raise RuntimeError(f"{arm} prototype diagnostics must be a list or null")
        row["prototype_count"] = (
            None
            if prototype_diagnostics is None
            else sum(
                int(cast("dict[str, Any]", item)["prototype_count"])
                for item in prototype_diagnostics
            )
        )
        final_packed = row["byte_budget_final_packed_active_bytes"]
        if final_packed is not None and int(final_packed) != int(row["active_kv_bytes"]):
            raise RuntimeError(
                f"{arm} measured active bytes disagree with the packed byte-budget trace"
            )
    record["arm"] = arm
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-prefix", default="v74")
    parser.add_argument(
        "--expected-experiment-id",
        default="choice_c_v74f_bounded_prototype_prompt_core_calibration",
    )
    parser.add_argument(
        "--expected-ledger-classification",
        default="fresh_disjoint_label_blind_calibration",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "mosaickv"
    _require_clean_worktree(project_root)
    source_snapshot = _source_snapshot(project_root)
    spec_path = args.spec.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    run_prefix = str(args.run_prefix)
    if not run_prefix or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in run_prefix
    ):
        raise ValueError("run-prefix must contain only lowercase letters, digits, '_' or '-'")
    if spec.get("experiment_id") != args.expected_experiment_id:
        raise RuntimeError(
            "worker experiment differs from the explicitly requested calibration protocol"
        )
    if spec.get("selection_rule_frozen_before_calibration") is not True:
        raise RuntimeError("v74 selection rule was not frozen before calibration")
    objective = cast("dict[str, Any]", spec["calibration_objective"])
    if (
        objective.get("labels_accessed") is not False
        or objective.get("official_task_scores_accessed") is not False
    ):
        raise RuntimeError("v74 calibration objective is not label-blind")

    entry = _task_entry(spec, args.task_index)
    task = str(entry["task"])
    ledger_path = _resolve(package_root, entry["ledger"])
    ledger, ledger_sha = _load_ledger(ledger_path)
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    expected = {
        "task": task,
        "dataset_id": entry["dataset_id"],
        "dataset_revision": entry["dataset_revision"],
        "split": entry["split"],
        "selection_seed": entry["selection_seed"],
        "sample_count": entry["sample_count"],
        "classification": args.expected_ledger_classification,
        "canonical_eligible": True,
        "source_git_dirty": False,
    }
    if any(ledger.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{task} ledger differs from the frozen calibration protocol")
    if len(sample_ids) != int(entry["sample_count"]) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"{task} calibration ledger has the wrong unique sample count")

    base_path = _resolve(package_root, spec["candidate_config"])
    candidates = cast("list[dict[str, Any]]", spec["candidates"])
    common_cache_policy = cast("dict[str, Any]", spec["common_cache_policy"])
    configs = calibration_configs(load_config(base_path), candidates, common_cache_policy)
    common_block_sizes = {config.cache.block_size for config in configs.values()}
    common_physical_block_sizes = {
        config.cache.physical_budget_block_size for config in configs.values()
    }
    if common_block_sizes != {4} or common_physical_block_sizes != {4}:
        raise RuntimeError("calibration arms do not share the frozen 4-token block policy")
    frozen_cap = int(
        cast("dict[str, Any]", spec["protocol_amendment"])[
            "configured_active_byte_cap_frozen_bytes"
        ]
    )
    candidate_caps = {
        config.cache.budget_value
        for arm, config in configs.items()
        if arm not in {"full_kv", "prompt_attention_topk"}
    }
    if candidate_caps != {frozen_cap}:
        raise RuntimeError("calibration candidates do not share the preregistered active byte cap")
    for arm, config in tuple(configs.items()):
        configs[arm] = replace(
            config,
            dataset=replace(
                config.dataset,
                id=str(entry["dataset_id"]),
                revision=str(entry["dataset_revision"]),
                split=str(entry["split"]),
            ),
            execution=replace(config.execution, seed=int(spec["seed"])),
            generation=replace(config.generation, max_new_tokens=int(spec["output_length"])),
        )
    arm_order = tuple(configs)
    if arm_order[:2] != ("full_kv", "prompt_attention_topk"):
        raise RuntimeError("v74 controls must precede the candidate arms")

    worker_id = f"{run_prefix}-{task}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    if telemetry_path.exists():
        prior = json.loads(telemetry_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed":
            print(f"calibration already completed: {worker_id}", flush=True)
            return 0
        raise RuntimeError(f"v74 has non-complete immutable telemetry: {telemetry_path}")

    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": spec["classification"],
        "canonical_eligible": True,
        "labels_accessed_for_selection": False,
        "official_task_scores_accessed_for_selection": False,
        "worker_id": worker_id,
        "task": task,
        "task_index": args.task_index,
        "spec": str(spec_path),
        "spec_sha256": _sha(spec_path),
        "base_config": str(base_path),
        "base_config_sha256": _sha(base_path),
        "ledger": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "dataset_id": entry["dataset_id"],
        "dataset_revision": entry["dataset_revision"],
        "sample_ids": list(sample_ids),
        "arm_order": list(arm_order),
        "candidates": candidates,
        "common_cache_policy": common_cache_policy,
        "source_snapshot": source_snapshot,
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
        "trials": [],
        "status": "started",
    }
    try:
        adapter = load_hf_adapter(
            configs[arm_order[-1]].model.id,
            revision=configs[arm_order[-1]].model.revision,
            model_kwargs=_model_kwargs(configs[arm_order[-1]], local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup = dict(ledger)
        warmup.update({"sample_count": 1, "sample_ids": [sample_ids[0]]})
        warmup_path = output_root / "warmup_ledgers" / f"{task}.json"
        _write_immutable(warmup_path, warmup)
        warmup_sha = _sha(warmup_path)
        for index, arm in enumerate(arm_order):
            _label_blind_record(
                configs[arm],
                adapter,
                task=task,
                run_id=f"{run_prefix}-{task}-warmup-{arm}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                sample_ids=(sample_ids[0],),
                phase="warmup",
                repetition=0,
                order=index,
                arm=arm,
            )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        repetitions = int(spec["measured_repetitions_per_arm"])
        for repetition in range(repetitions):
            order = arm_order if repetition % 2 == 0 else tuple(reversed(arm_order))
            for order_index, arm in enumerate(order):
                _require_clean_worktree(project_root)
                if _source_snapshot(project_root) != source_snapshot:
                    raise RuntimeError("repository source changed during calibration")
                record = _label_blind_record(
                    configs[arm],
                    adapter,
                    task=task,
                    run_id=f"{run_prefix}-{task}-trial-{repetition:02d}-{arm}",
                    output_root=output_root,
                    ledger_path=ledger_path,
                    ledger_sha=ledger_sha,
                    sample_ids=sample_ids,
                    phase="measured",
                    repetition=repetition,
                    order=order_index,
                    arm=arm,
                )
                cast("list[dict[str, Any]]", telemetry["trials"]).append(record)
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
