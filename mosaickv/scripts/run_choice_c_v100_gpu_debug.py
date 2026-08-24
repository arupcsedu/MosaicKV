#!/usr/bin/env python3
"""Run one explicitly authorized V100 compressed synthetic sample on one A100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from choice_c_v74_protocol import apply_candidate, apply_common_cache_policy

from mosaickv.config import DatasetConfig, load_config
from mosaickv.hf_cli import run_hf_evaluation


def _git_sha(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_existing_debug_artifacts(
    output_root: Path,
    *,
    runtime_git_sha: str,
    validation_git_sha: str,
    source_slurm_job_id: str = "not_recorded",
) -> dict[str, Any]:
    """Validate and summarize one preserved V100 runtime without GPU execution."""

    run_id = f"v100-single-sample-{runtime_git_sha[:12]}"
    run_root = output_root / run_id
    raw_path = run_root / "results.jsonl"
    parquet_path = run_root / "aggregate.parquet"
    manifest_path = run_root / "manifest.json"
    trace_paths = sorted((output_root / "traces" / run_id).glob("*.json"))
    if len(trace_paths) != 1:
        raise RuntimeError(f"expected exactly one V100 trace, found {len(trace_paths)}")
    raw_lines = [line for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
    if len(raw_lines) != 1:
        raise RuntimeError(f"expected exactly one V100 result row, found {len(raw_lines)}")
    row = cast("dict[str, Any]", json.loads(raw_lines[0]))
    trace = cast("dict[str, Any]", json.loads(trace_paths[0].read_text(encoding="utf-8")))
    manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8")))
    source = cast("dict[str, Any]", manifest["source"])
    if source.get("git_sha") != runtime_git_sha or source.get("git_dirty") is not False:
        raise RuntimeError("V100 artifacts do not come from the expected clean runtime SHA")
    artifacts = cast("dict[str, Any]", manifest["artifacts"])
    if artifacts.get("raw_output_sha") != _sha256(raw_path):
        raise RuntimeError("V100 raw-result hash differs from the manifest")
    if artifacts.get("metrics_sha") != _sha256(parquet_path):
        raise RuntimeError("V100 Parquet hash differs from the manifest")
    if artifacts.get("log_sha") != _trace_sha(trace_paths):
        raise RuntimeError("V100 trace hash differs from the manifest")
    if row.get("status") != "completed" or row.get("error") is not None:
        raise RuntimeError("V100 preserved result row is not successful")
    required_trace_fields = {
        "answer",
        "byte_budget_materialization",
        "forecast_statistics",
        "generated_token_ids",
        "graph_edges",
        "hierarchical_logit_safe_eviction",
        "online_uncertainty_routing",
        "prototypes",
        "repair_events",
        "selected_blocks",
        "timing_breakdown",
    }
    missing = sorted(required_trace_fields - set(trace))
    if trace.get("status") != "completed" or missing:
        raise RuntimeError(f"V100 trace is incomplete; missing={missing}")
    routing = cast("dict[str, Any]", trace["online_uncertainty_routing"])
    if routing.get("hierarchical_gpu_debug_authorized") is not True:
        raise RuntimeError("V100 trace omitted explicit GPU debug authorization")
    decision = cast("dict[str, Any]", trace["hierarchical_logit_safe_eviction"])
    if decision.get("status") != "completed_label_blind":
        raise RuntimeError("V100 hierarchical trace did not complete")
    if decision.get("probe_state_discarded_before_measured_decode") is not True:
        raise RuntimeError("V100 trace does not prove probe-state disposal")
    resolved = cast("dict[str, Any]", manifest["resolved_config"])
    selection = cast("dict[str, Any]", resolved["selection"])
    evaluations = cast("list[dict[str, Any]]", decision["evaluations"])
    if not evaluations:
        raise RuntimeError("V100 trace contains no hierarchical probe evaluations")
    maximum_evaluations = int(selection["stable_head_hierarchical_maximum_candidate_evaluations"])
    maximum_rounds = int(selection["stable_head_hierarchical_maximum_sequential_rounds"])
    maximum_kl = float(selection["stable_head_hierarchical_maximum_kl"])
    maximum_logit_difference = float(
        selection["stable_head_hierarchical_maximum_logit_difference"]
    )
    horizon = int(selection["stable_head_hierarchical_horizon"])
    if (
        int(decision.get("evaluations_used", maximum_evaluations + 1)) > maximum_evaluations
        or int(decision.get("rounds_used", maximum_rounds + 1)) > maximum_rounds
        or int(decision.get("evaluations_used", -1)) != len(evaluations)
    ):
        raise RuntimeError("V100 exceeded or misreported its probe/round budget")
    for evaluation in evaluations:
        result = cast("dict[str, Any]", evaluation["result"])
        if (
            evaluation.get("passed") is not True
            or result.get("greedy_token_match") is not True
            or int(result.get("evaluated_steps", -1)) != horizon
            or float(result.get("maximum_forward_kl", float("inf"))) > maximum_kl
            or float(result.get("maximum_absolute_logit_difference", float("inf")))
            > maximum_logit_difference
        ):
            raise RuntimeError("V100 contains a probe admitted outside the frozen logit gate")
    if (
        decision.get("selected_candidate_passed") is not True
        or decision.get("final_joint_probe_passed") is not True
        or decision.get("target_reached") is not True
        or decision.get("used_fullkv_fallback") is not False
        or decision.get("unresolved_exact_routes") != []
    ):
        raise RuntimeError("V100 did not finish with a verified target cache")
    cache = cast("dict[str, Any]", resolved["cache"])
    accounting = cast("dict[str, Any]", trace["byte_budget_materialization"])
    logical = int(accounting["logical_active_bytes"])
    padding = int(accounting["packed_padding_bytes"])
    packed = int(accounting["final_packed_active_bytes"])
    configured_cap = int(accounting["configured_active_byte_cap"])
    if (
        logical + padding != packed
        or int(row["active_kv_bytes"]) != packed
        or int(trace["active_kv_bytes"]) != packed
        or int(row["residual_kv_bytes"]) != 0
        or packed > configured_cap
        or int(trace["final_active_kv_bytes"]) > configured_cap
        or cache.get("block_size") != 4
        or cache.get("physical_budget_block_size") != 4
        or cache.get("physical_budget_mandatory_policy") != "last_token"
    ):
        raise RuntimeError("V100 cache policy or active-byte accounting failed")
    return {
        "schema_version": 1,
        "status": "passed",
        "measurement_type": "validation_smoke",
        "canonical_experiment_row": False,
        "calibration_or_heldout": False,
        "runtime_git_sha": runtime_git_sha,
        "validation_git_sha": validation_git_sha,
        "source_slurm_job_id": source_slurm_job_id,
        "source_artifacts_reused_without_gpu_rerun": runtime_git_sha != validation_git_sha,
        "gpu_name": cast("dict[str, Any]", manifest["hardware"])["gpu_type"],
        "model_id": cast("dict[str, Any]", manifest["model"])["id"],
        "model_revision": cast("dict[str, Any]", manifest["model"])["revision"],
        "task": "synthetic_smoke",
        "sample_count": 1,
        "answer": row["answer"],
        "task_score": row["task_score"],
        "hierarchical_decision": decision,
        "probe_gate_summary": {
            "evaluations_used": decision["evaluations_used"],
            "maximum_evaluations": maximum_evaluations,
            "rounds_used": decision["rounds_used"],
            "maximum_rounds": maximum_rounds,
            "maximum_observed_kl": max(
                float(cast("dict[str, Any]", item["result"])["maximum_forward_kl"])
                for item in evaluations
            ),
            "maximum_allowed_kl": maximum_kl,
            "maximum_observed_logit_difference": max(
                float(
                    cast("dict[str, Any]", item["result"])[
                        "maximum_absolute_logit_difference"
                    ]
                )
                for item in evaluations
            ),
            "maximum_allowed_logit_difference": maximum_logit_difference,
            "probe_state_discarded_before_measured_decode": True,
            "target_reached": True,
            "final_joint_probe_passed": True,
            "used_fullkv_fallback": False,
        },
        "active_kv_bytes": packed,
        "logical_active_kv_bytes": logical,
        "packed_padding_bytes": padding,
        "final_decode_active_kv_bytes": int(trace["final_active_kv_bytes"]),
        "configured_active_byte_cap": configured_cap,
        "residual_kv_bytes": 0,
        "trace_path": str(trace_paths[0].resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "artifact_hashes_verified": True,
        "trace_required_fields_verified": True,
    }


def _write_report(output_root: Path, report: dict[str, Any]) -> None:
    report_path = output_root / "v100_single_sample_debug.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite debug report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _v100_config(package_root: Path, *, max_new_tokens: int) -> Any:
    v99_path = (
        package_root
        / "configs/experiments/choice_c_v99_progressive_logit_safe_eviction_calibration.yaml"
    )
    v100_path = (
        package_root
        / "configs/experiments/choice_c_v100_hierarchical_head_local_logit_safe_design.yaml"
    )
    v99 = cast("dict[str, Any]", yaml.safe_load(v99_path.read_text(encoding="utf-8")))
    v100 = cast("dict[str, Any]", yaml.safe_load(v100_path.read_text(encoding="utf-8")))
    candidates = cast("list[dict[str, Any]]", v99["candidates"])
    if len(candidates) != 1:
        raise RuntimeError("V100 debug requires the single frozen V99 exact-cache parent")
    common_cache_policy = cast("dict[str, Any]", v99["common_cache_policy"])
    base = apply_common_cache_policy(
        load_config(package_root / str(v99["candidate_config"])),
        common_cache_policy,
    )
    parent = apply_candidate(base, candidates[0])
    controls = cast("dict[str, Any]", v100["strict_runtime_configuration"])
    selection = replace(
        parent.selection,
        stable_head_progressive_logit_safe_eviction=False,
        stable_head_progressive_eviction_initial_batch_blocks=0,
        stable_head_progressive_eviction_maximum_probes=0,
        stable_head_progressive_eviction_horizon=0,
        stable_head_progressive_eviction_maximum_kl=0.0,
        stable_head_progressive_eviction_maximum_logit_difference=0.0,
        stable_head_hierarchical_logit_safe_eviction=bool(
            controls["stable_head_hierarchical_logit_safe_eviction"]
        ),
        stable_head_hierarchical_partition_count=int(
            controls["stable_head_hierarchical_partition_count"]
        ),
        stable_head_hierarchical_maximum_refinement_depth=int(
            controls["stable_head_hierarchical_maximum_refinement_depth"]
        ),
        stable_head_hierarchical_maximum_candidate_evaluations=int(
            controls["stable_head_hierarchical_maximum_candidate_evaluations"]
        ),
        stable_head_hierarchical_maximum_sequential_rounds=int(
            controls["stable_head_hierarchical_maximum_sequential_rounds"]
        ),
        stable_head_hierarchical_maximum_rollback_rounds=int(
            controls["stable_head_hierarchical_maximum_rollback_rounds"]
        ),
        stable_head_hierarchical_horizon=int(controls["stable_head_hierarchical_horizon"]),
        stable_head_hierarchical_maximum_kl=float(
            controls["stable_head_hierarchical_maximum_kl"]
        ),
        stable_head_hierarchical_maximum_logit_difference=float(
            controls["stable_head_hierarchical_maximum_logit_difference"]
        ),
    )
    return replace(
        parent,
        dataset=DatasetConfig("mosaickv/synthetic-ci", "schema-v1", "test"),
        generation=replace(parent.generation, max_new_tokens=max_new_tokens),
        selection=selection,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--artifacts-only", action="store_true")
    parser.add_argument("--source-slurm-job-id", default="not_recorded")
    args = parser.parse_args()
    if args.max_new_tokens <= 2:
        parser.error("--max-new-tokens must exceed the frozen two-token probe horizon")
    package_root = Path(__file__).resolve().parents[1]
    repository = package_root.parent
    observed_sha = _git_sha(repository)
    output_root = args.output_root.resolve()
    if args.artifacts_only:
        report = validate_existing_debug_artifacts(
            output_root,
            runtime_git_sha=str(args.expected_git_sha),
            validation_git_sha=observed_sha,
            source_slurm_job_id=str(args.source_slurm_job_id),
        )
        _write_report(output_root, report)
        return 0
    if os.environ.get("MOSAICKV_ENABLE_V100_GPU_DEBUG") != "1":
        raise RuntimeError("V100 debug requires MOSAICKV_ENABLE_V100_GPU_DEBUG=1")
    if not torch.cuda.is_available():
        raise RuntimeError("V100 GPU debug requires CUDA")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(
            f"V100 GPU debug requires one A100, observed {torch.cuda.get_device_name(0)!r}"
        )
    if observed_sha != args.expected_git_sha:
        raise RuntimeError(
            f"source SHA mismatch: expected {args.expected_git_sha}, observed {observed_sha}"
        )
    config = _v100_config(package_root, max_new_tokens=args.max_new_tokens)
    run_id = f"v100-single-sample-{observed_sha[:12]}"
    run_args = Namespace(
        task="synthetic_smoke",
        run_id=run_id,
        output_dir=str(output_root),
        raw_output=None,
        parquet_output=None,
        manifest=None,
        trace_directory=str(output_root / "traces"),
        subset_size=1,
        local_files_only=True,
    )
    code, summary = run_hf_evaluation(run_args, config)
    if code != 0 or summary.get("failed_samples") != 0:
        raise RuntimeError(f"V100 one-sample evaluation failed: {summary}")
    report = validate_existing_debug_artifacts(
        output_root,
        runtime_git_sha=observed_sha,
        validation_git_sha=observed_sha,
        source_slurm_job_id=os.environ.get("SLURM_JOB_ID", "not_used"),
    )
    _write_report(output_root, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
