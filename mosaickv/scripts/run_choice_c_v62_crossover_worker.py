#!/usr/bin/env python3
"""Measure the fused candidate at decode crossover lengths on two A100s."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot
from run_choice_c_v61_candidate_runtime_worker import candidate_config

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment

OUTPUT_LENGTHS = (16, 64, 128)
PRIMARY_MODES = ("full_kv", "retention_one", "prompt_attention_topk", "candidate_fast")


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError(f"refusing to mutate immutable v62 input: {path}")
    if not path.exists():
        path.write_text(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=range(len(OUTPUT_LENGTHS)), required=True, type=int)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--candidate-preflight", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    source_snapshot = _source_snapshot(project_root)
    output_length = OUTPUT_LENGTHS[args.index]
    spec = cast("dict[str, Any]", yaml.safe_load(args.spec.resolve().read_text()))
    profile = cast("dict[str, Any]", yaml.safe_load(args.profile.resolve().read_text()))
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen fidelity profile has no parameter mapping")
    ledger, ledger_sha = _load_ledger(args.ledger.resolve())
    if ledger.get("classification") != "retrospective_non_claim_system_validation":
        raise RuntimeError("v62 requires the retrospective non-claim ledger")
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("v62 requires exactly two fixed verification samples")
    configs = {
        mode: candidate_config(
            mode,
            parameters,
            ledger,
            ledger_sha,
            output_length=output_length,
            seed=int(spec["seed"]),
        )
        for mode in (*PRIMARY_MODES, "candidate_repair")
    }
    worker_id = f"v62-length{output_length}-{_sha(args.spec.resolve())[:10]}"
    telemetry_path = args.output_root.resolve() / "telemetry" / f"{worker_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_crossover_and_divergent_repair",
        "worker_id": worker_id,
        "output_length": output_length,
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
    output_root = args.output_root.resolve()
    try:
        adapter = load_hf_adapter(
            configs[PRIMARY_MODES[0]].model.id,
            revision=configs[PRIMARY_MODES[0]].model.revision,
            model_kwargs=_model_kwargs(configs[PRIMARY_MODES[0]], local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup = dict(ledger)
        warmup["sample_count"] = 1
        warmup["sample_ids"] = [sample_ids[0]]
        warmup_path = output_root / "warmup_ledgers" / f"length{output_length}.json"
        _write_immutable(warmup_path, warmup)
        warmup_sha = _sha(warmup_path)
        frames: dict[str, pd.DataFrame] = {}
        if args.candidate_preflight:
            candidate_frame = _run_online(
                configs["candidate_fast"],
                adapter,
                run_id=f"v62-preflight-length{output_length}-candidate_fast",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            observed_methods = set(candidate_frame["method"].astype(str))
            expected_method = configs["candidate_fast"].method.value
            if observed_methods != {expected_method}:
                raise RuntimeError(
                    "candidate preflight did not materialize the requested three-tier method: "
                    f"expected={expected_method}, observed={sorted(observed_methods)}"
                )
            telemetry["candidate_preflight"] = "three_tier_completed"
            telemetry["status"] = "completed"
            return 0
        for mode in PRIMARY_MODES:
            warmup_frame = _run_online(
                configs[mode],
                adapter,
                run_id=f"v62-warmup-length{output_length}-{mode}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            if mode == "candidate_fast":
                observed_methods = set(warmup_frame["method"].astype(str))
                expected_method = configs[mode].method.value
                if observed_methods != {expected_method}:
                    raise RuntimeError(
                        "candidate warmup did not materialize the requested three-tier "
                        f"method: expected={expected_method}, "
                        f"observed={sorted(observed_methods)}"
                    )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for mode in PRIMARY_MODES:
            if _source_snapshot(project_root) != source_snapshot:
                raise RuntimeError("repository source changed during v62 verification")
            frames[mode] = _run_online(
                configs[mode],
                adapter,
                run_id=f"v62-validation-length{output_length}-{mode}",
                output_root=output_root,
                ledger_path=args.ledger.resolve(),
                ledger_sha=ledger_sha,
                expected_sample_ids=sample_ids,
            )
            if mode == "candidate_fast":
                observed_methods = set(frames[mode]["method"].astype(str))
                expected_method = configs[mode].method.value
                if observed_methods != {expected_method}:
                    raise RuntimeError(
                        "candidate validation did not materialize the requested three-tier "
                        f"method: expected={expected_method}, "
                        f"observed={sorted(observed_methods)}"
                    )
            _settle(adapter)

        full_answers = frames["full_kv"].set_index("sample_id")["answer"]
        candidate_answers = frames["candidate_fast"].set_index("sample_id")["answer"]
        divergent = tuple(
            sample_id
            for sample_id in sample_ids
            if str(full_answers.loc[sample_id]) != str(candidate_answers.loc[sample_id])
        )
        telemetry["divergent_sample_ids"] = list(divergent)
        if divergent:
            repair_ledger = dict(ledger)
            repair_ledger.update(
                {
                    "classification": "derived_divergent_sample_repair_validation",
                    "canonical_eligible": False,
                    "parent_ledger": str(args.ledger.resolve()),
                    "parent_ledger_sha256": ledger_sha,
                    "selection_rule": "candidate_answer_differs_from_same_run_fullkv_answer",
                    "sample_count": len(divergent),
                    "sample_ids": list(divergent),
                }
            )
            repair_path = output_root / "divergent_ledgers" / f"length{output_length}.json"
            _write_immutable(repair_path, repair_ledger)
            _run_online(
                configs["candidate_repair"],
                adapter,
                run_id=f"v62-validation-length{output_length}-candidate_repair",
                output_root=output_root,
                ledger_path=repair_path,
                ledger_sha=_sha(repair_path),
                expected_sample_ids=divergent,
            )
            telemetry["repair_validation"] = "completed_on_divergent_samples"
        else:
            telemetry["repair_validation"] = "skipped_no_divergent_samples"
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
