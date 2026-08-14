#!/usr/bin/env python3
"""Verify the candidate-first MosaicKV runtime on two exploratory A100 workers."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry
from run_choice_c_v36_recommended_worker import mode_config as v36_mode_config
from run_choice_c_v38_bottleneck_worker import mode_config as v38_mode_config
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import RepairCandidateRanking, RepairPolicy

RANK_MODES = {
    0: ("prompt_attention_topk", "candidate_fast"),
    1: ("full_kv", "retention_one", "candidate_repair"),
}


def candidate_config(
    mode: str,
    parameters: dict[str, Any],
    ledger: dict[str, Any],
    ledger_sha: str,
    *,
    output_length: int,
    seed: int,
) -> Any:
    """Build one immutable verification cell from the frozen v36 profile."""

    if mode in {"prompt_attention_topk", "full_kv"}:
        return v36_mode_config(
            mode,
            parameters,
            ledger,
            ledger_sha,
            output_length=output_length,
            seed=seed,
        )
    base = v38_mode_config(
        "implementation_only",
        parameters,
        ledger,
        ledger_sha,
        output_length=output_length,
        seed=seed,
    )
    if mode == "retention_one":
        return replace(base, cache=replace(base.cache, retention_ratio=1.0))

    candidate = replace(
        base,
        graph=replace(
            base.graph,
            candidate_graph_mode="shortlist_rerank",
            candidate_neighbor_hops=1,
            similarity_shortlist_multiplier=3.0,
            similarity_shortlist_min_nodes=64,
            candidate_nonmandatory_fraction=0.10,
            candidate_descriptor_pooling=True,
            screening_group_blocks=4,
            uncertain_head_only=True,
            uncertain_head_fraction=0.25,
            uncertain_head_threshold=0.2,
        ),
        selection=replace(
            base.selection,
            algorithm="vectorized_coverage",
            coverage_correction_rounds=1,
        ),
        prototypes=replace(
            base.prototypes,
            pooling_mode="exact_medoid_key_forecast_value",
            attention_bias_basis="forecast_attention_entropy",
            anchor_position_strategy="forecast_weighted_member_medoid",
            source_gather_mode="batched_layer",
            group_size=31,
            max_position_span=2048,
            # The frozen fidelity profile contains a few legacy per-head
            # group/span overrides. They would silently defeat this bounded
            # two-prototype candidate by requesting up to five anchors in one
            # routed head, so the system candidate deliberately uses its
            # declared uniform geometry.
            head_group_sizes=(),
            head_position_spans=(),
            uncertain_head_only=True,
            uncertain_head_fraction=0.25,
            uncertain_head_threshold=0.2,
            source_modalities=("image", "video"),
            maximum_prototypes_per_head=2,
        ),
    )
    if mode == "candidate_fast":
        return candidate
    if mode == "candidate_repair":
        return replace(
            candidate,
            repair=replace(
                candidate.repair,
                enabled=True,
                policy=RepairPolicy.DECISION_AWARE,
                decision_margin_threshold=0.05,
                decision_high_risk_threshold=0.1,
                max_blocks_per_step=1,
                candidate_ranking=RepairCandidateRanking.DECISION_INFLUENCE,
            ),
            residual=replace(
                candidate.residual,
                enabled=True,
                gpu_rescue_max_blocks=2,
                gpu_rescue_max_bytes=64 * 1024,
            ),
        )
    raise ValueError(f"unknown v61 verification mode: {mode}")


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError(f"refusing to mutate immutable verification input: {path}")
    if not path.exists():
        path.write_text(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", choices=tuple(RANK_MODES), required=True, type=int)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    source_snapshot = _source_snapshot(project_root)
    spec_path = args.spec.resolve()
    profile_path = args.profile.resolve()
    ledger_path = args.ledger.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    profile = cast("dict[str, Any]", yaml.safe_load(profile_path.read_text()))
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen fidelity profile has no parameter mapping")
    ledger, ledger_sha = _load_ledger(ledger_path)
    if ledger.get("classification") != "retrospective_non_claim_system_validation":
        raise RuntimeError("v61 requires the retrospective non-claim ledger")
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("v61 requires exactly two fixed verification samples")
    modes = RANK_MODES[args.rank]
    configs = {
        mode: candidate_config(
            mode,
            parameters,
            ledger,
            ledger_sha,
            output_length=int(spec["output_length"]),
            seed=int(spec["seed"]),
        )
        for mode in modes
    }
    worker_id = f"v61-rank{args.rank}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_candidate_runtime_verification",
        "worker_id": worker_id,
        "rank": args.rank,
        "modes": list(modes),
        "sample_ids": list(sample_ids),
        "source_snapshot": source_snapshot,
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
        "status": "started",
    }
    try:
        adapter = load_hf_adapter(
            configs[modes[0]].model.id,
            revision=configs[modes[0]].model.revision,
            model_kwargs=_model_kwargs(configs[modes[0]], local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup = dict(ledger)
        warmup["sample_count"] = 1
        warmup["sample_ids"] = [sample_ids[0]]
        warmup_path = output_root / "warmup_ledgers" / f"rank{args.rank}.json"
        _write_immutable(warmup_path, warmup)
        warmup_sha = _sha(warmup_path)
        for mode in modes:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v61-warmup-rank{args.rank}-{mode}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for mode in modes:
            if _source_snapshot(project_root) != source_snapshot:
                raise RuntimeError("repository source changed during v61 verification")
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v61-validation-rank{args.rank}-{mode}-{_sha(spec_path)[:10]}",
                output_root=output_root,
                ledger_path=ledger_path,
                ledger_sha=ledger_sha,
                expected_sample_ids=sample_ids,
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
