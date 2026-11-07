#!/usr/bin/env python3
"""Prepare a one-identity timing diagnostic for the exact V6 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_output_preserving_route_logit_validation as shared
from prepare_output_preserving_sweep import source_digest
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import canonical_config, config_sha256, load_config


def prepare(args: argparse.Namespace) -> None:
    source, root = args.source.resolve(), args.root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("a new external scratch output root is required")
    candidate_plan_path = args.candidate_plan.resolve()
    if digest(candidate_plan_path) != args.candidate_plan_sha256:
        raise ValueError("candidate plan hash mismatch")
    candidate_plan = json.loads(candidate_plan_path.read_text())
    candidate_protocol = candidate_plan.get("protocol_version")
    candidate_labels = {
        "output_preserving_surrogate_validation_v6": (
            "output_preserving_surrogate_25p5",
            "output_preserving_probe_microbenchmark_v6",
            "output_preserving_surrogate_validation_v6.yaml",
        ),
        "output_preserving_vectorized_surrogate_validation_v7": (
            "output_preserving_vectorized_surrogate_25p5",
            "output_preserving_probe_microbenchmark_v7",
            "output_preserving_vectorized_surrogate_validation_v7.yaml",
        ),
        "output_preserving_directional_single_validation_v8": (
            "output_preserving_directional_single_25p5",
            "output_preserving_probe_microbenchmark_v8",
            "output_preserving_directional_single_validation_v8.yaml",
        ),
    }
    if candidate_protocol not in candidate_labels:
        raise ValueError("probe diagnostic requires a registered surrogate plan")
    candidate_label, diagnostic_protocol, protocol_filename = candidate_labels[
        candidate_protocol
    ]

    timing_plan_path = args.timing_identity_plan.resolve()
    if digest(timing_plan_path) != args.timing_identity_plan_sha256:
        raise ValueError("timing-identity plan hash mismatch")
    timing_plan = json.loads(timing_plan_path.read_text())
    if timing_plan.get("protocol_version") != "output_preserving_layer_capacity_validation_v5":
        raise ValueError("timing identity must come from the consumed V5 plan")

    shared.ALLOWED_OVERLAY_PATHS = shared.ALLOWED_OVERLAY_PATHS | {
        "mosaickv/src/mosaickv/adapters/huggingface/layerwise_exact_cache.py",
        "mosaickv/scripts/prepare_output_preserving_layer_capacity_validation.py",
        "mosaickv/scripts/prepare_output_preserving_surrogate_validation.py",
        "mosaickv/scripts/prepare_output_preserving_vectorized_surrogate_validation.py",
        "mosaickv/scripts/prepare_output_preserving_directional_single_validation.py",
        "mosaickv/scripts/prepare_output_preserving_probe_microbenchmark.py",
        "mosaickv/scripts/analyze_output_preserving_probe_microbenchmark.py",
        "mosaickv/scripts/submit_output_preserving_probe_microbenchmark.py",
        "mosaickv/configs/experiments/output_preserving_layer_capacity_validation_v5.yaml",
        "mosaickv/configs/experiments/output_preserving_surrogate_validation_v6.yaml",
        "mosaickv/configs/experiments/output_preserving_vectorized_surrogate_validation_v7.yaml",
        "mosaickv/configs/experiments/output_preserving_directional_single_validation_v8.yaml",
        "mosaickv/docs/output_preserving_route_logit_v4_16_development_seal.md",
        "mosaickv/docs/output_preserving_layer_capacity_v5_16_development_seal.md",
        "mosaickv/docs/output_preserving_surrogate_v6_timing_seal.md",
    }
    overlay = shared._source_snapshot(source, args.source_sha)

    source_ledger = json.loads(Path(timing_plan["ledger"]).read_text())
    samples = source_ledger.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("sealed V5 ledger has no timing identity")
    ledger = {**source_ledger, "samples": [samples[0]]}
    ledger_path = root / "ledger.json"
    write_once(ledger_path, ledger)

    candidate_job = next(
        job for job in candidate_plan["jobs"] if job["label"] == candidate_label
    )
    config = load_config(candidate_job["config"])
    options = config.output_preserving
    if (
        options is None
        or options.logit_candidate_strategy
        not in {
            "layer_capacity_surrogate_beam",
            "layer_capacity_vectorized_surrogate_beam",
            "layer_capacity_directional_single_beam",
        }
        or options.logit_layer_transfer_limit != 8
        or options.logit_probe_batch_size != 2
        or options.logit_stability_horizon != 2
        or options.logit_extended_horizon != 8
        or options.logit_extended_candidate_limit != 1
    ):
        raise ValueError("V6 candidate does not implement the registered probe schedule")
    label = "output_preserving_surrogate_probe"
    config_path = root / "configs" / f"{label}.json"
    write_once(config_path, canonical_config(config))
    jobs: list[dict[str, object]] = [
        {
            "index": 0,
            "label": label,
            "analysis_label": label,
            "method": config.method.value,
            "effective_method": options.effective_method,
            "candidate_strategy": options.logit_candidate_strategy,
            "surrogate_candidate_count": options.logit_layer_transfer_limit,
            "model_probed_candidate_count": 2,
            "short_probe_horizon": options.logit_stability_horizon,
            "extended_probe_horizon": options.logit_extended_horizon,
            "extended_candidate_limit": options.logit_extended_candidate_limit,
            "cap_mib": 25.5,
            "cap_bytes": round(25.5 * 2**20),
            "config": str(config_path),
            "config_sha256": config_sha256(config),
            "output": str(root / f"outputs/000-{label}"),
        }
    ]

    protocol_path = source / "mosaickv/configs/experiments" / protocol_filename
    plan = {
        "schema_version": 1,
        "protocol_version": diagnostic_protocol,
        "protocol_path": str(protocol_path),
        "protocol_sha256": digest(protocol_path),
        "classification": "preregistered_disjoint_development_validation_not_paper_evidence",
        "heldout_claim": False,
        "paper_claim_authorized": False,
        "paper_campaign_submission_authorized": False,
        "timing_diagnostic_only": True,
        "quality_evidence": False,
        "source_root": str(source),
        "source_base_git_sha": args.source_sha,
        "source_worktree_dirty": True,
        "source_content_sha256": source_digest(source),
        "source_overlay_files": overlay,
        "parent_plan": str(candidate_plan_path),
        "parent_plan_sha256": args.candidate_plan_sha256,
        "timing_identity_plan": str(timing_plan_path),
        "timing_identity_plan_sha256": args.timing_identity_plan_sha256,
        "processor_signature": candidate_plan["processor_signature"],
        "task": candidate_plan["task"],
        "model_id": candidate_plan["model_id"],
        "model_revision": candidate_plan["model_revision"],
        "dataset_id": candidate_plan["dataset_id"],
        "dataset_revision": candidate_plan["dataset_revision"],
        "split": candidate_plan["split"],
        "sample_count": 1,
        "sample_ids": [samples[0]["sample_id"]],
        "ledger": str(ledger_path),
        "ledger_sha256": digest(ledger_path),
        "warmups": 0,
        "output_tokens": 64,
        "max_global_a100s": 4,
        "cpus_per_worker": 12,
        "time_limit": "02:00:00",
        "retention_one_config": candidate_plan["retention_one_config"],
        "retention_one_config_sha256": candidate_plan[
            "retention_one_config_sha256"
        ],
        "maximum_surrogate_generation_seconds": 5.0,
        "maximum_worker_wall_seconds": 360.0,
        "jobs": jobs,
    }
    write_once(root / "plan.json", plan)
    print(
        json.dumps(
            {
                "root": str(root),
                "jobs": len(jobs),
                "candidate_strategy": options.logit_candidate_strategy,
                "sample_id": samples[0]["sample_id"],
                "ledger_sha256": digest(ledger_path),
                "plan_sha256": digest(root / "plan.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--candidate-plan", "--v6-plan", type=Path, required=True)
    parser.add_argument(
        "--candidate-plan-sha256", "--v6-plan-sha256", required=True
    )
    parser.add_argument("--timing-identity-plan", type=Path, required=True)
    parser.add_argument("--timing-identity-plan-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    prepare(parser.parse_args())
