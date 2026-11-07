#!/usr/bin/env python3
"""Prepare the preregistered 128-sample joint draft-aware validation."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from prepare_output_preserving_draft_sweep import DRAFT_STEPS, _draft_candidate
from prepare_output_preserving_frozen_validation import merge_reviewed_ledgers
from prepare_output_preserving_sweep import (
    MODEL_ID,
    MODEL_REVISION,
    SEED,
    configurations,
    source_digest,
)
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import DatasetConfig, canonical_config, config_sha256

CLASSIFICATION = "preregistered_fresh_development_validation_not_paper_evidence"
SAMPLE_COUNT = 128
CAPS_MIB = (25.5, 40.0)
ALLOWED_OVERLAY_PATHS = frozenset(
    {
        "mosaickv/src/mosaickv/config.py",
        "mosaickv/src/mosaickv/selection/output_preserving.py",
        "mosaickv/src/mosaickv/backends/output_preserving.py",
        "mosaickv/scripts/prepare_output_preserving_sweep.py",
        "mosaickv/scripts/prepare_output_preserving_joint_validation.py",
        "mosaickv/scripts/submit_output_preserving_joint_validation.py",
        "mosaickv/scripts/run_output_preserving_sweep.py",
        "mosaickv/scripts/analyze_output_preserving_frozen_validation.py",
        "mosaickv/configs/experiments/output_preserving_draft_joint_validation_v2.yaml",
    }
)


def _source_snapshot(source: Path, expected_sha: str) -> tuple[list[dict[str, str]], bool]:
    observed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected_sha:
        raise ValueError(f"source SHA mismatch: expected {expected_sha}, observed {observed}")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    paths = [line[3:] for line in status if line]
    unexpected = sorted(set(paths) - ALLOWED_OVERLAY_PATHS)
    if unexpected:
        raise ValueError(f"source overlay contains out-of-scope paths: {unexpected}")
    if not paths:
        raise ValueError("joint candidate must use an explicit hash-bound source overlay")
    records = [
        {"path": path, "sha256": digest(source / path)} for path in sorted(set(paths))
    ]
    return records, True


def _joint_candidate(base):
    candidate = _draft_candidate(base)
    assert candidate.output_preserving is not None
    return replace(
        candidate,
        output_preserving=replace(
            candidate.output_preserving,
            draft_ranking_mode="joint_normalized",
            prompt_objective_weight=0.5,
            draft_objective_weight=0.5,
            minimum_joint_improvement=1e-7,
        ),
    )


def prepare(args: argparse.Namespace) -> None:
    source, root = args.source.resolve(), args.root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("a new external scratch output root is required")
    overlay, dirty = _source_snapshot(source, args.source_sha)
    protocol_path = (
        source
        / "mosaickv/configs/experiments/output_preserving_draft_joint_validation_v2.yaml"
    )
    if not protocol_path.is_file():
        raise ValueError("registered joint validation protocol is missing")
    parent_path = args.parent_plan.resolve()
    parent = json.loads(parent_path.read_text())
    if (
        parent.get("model_id") != MODEL_ID
        or parent.get("model_revision") != MODEL_REVISION
        or not isinstance(parent.get("processor_signature"), dict)
    ):
        raise ValueError("reviewed parent plan has incompatible controls")

    excluded: set[str] = set()
    exclusion_bindings = []
    for path in args.excluded_plan:
        resolved = path.resolve()
        payload = json.loads(resolved.read_text())
        identities = payload.get("sample_ids")
        if not isinstance(identities, list) or not identities:
            raise ValueError(f"excluded plan has no bound sample identities: {resolved}")
        excluded.update(str(item) for item in identities)
        exclusion_bindings.append({"path": str(resolved), "sha256": digest(resolved)})
    ledger, ledger_bindings = merge_reviewed_ledgers(
        args.ledger_shard,
        parent_plan=parent,
        excluded_ids=excluded,
        expected_samples=args.samples,
    )
    ledger_path = root / "ledger.json"
    write_once(ledger_path, ledger)
    dataset = DatasetConfig(
        str(ledger["dataset_id"]), str(ledger["dataset_revision"]), str(ledger["split"])
    )
    plan: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "output_preserving_draft_joint_validation_v2",
        "protocol_path": str(protocol_path),
        "protocol_sha256": digest(protocol_path),
        "classification": CLASSIFICATION,
        "cohort_relation": "disjoint_from_all_inspected_output_preserving_cohorts",
        "heldout_claim": False,
        "paper_claim_authorized": False,
        "paper_campaign_submission_authorized": False,
        "candidate_frozen_before_sample_execution": True,
        "candidate_retuning_authorized": False,
        "source_root": str(source),
        "source_base_git_sha": args.source_sha,
        "source_worktree_dirty": dirty,
        "source_content_sha256": source_digest(source),
        "source_overlay_files": overlay,
        "task": "coco_caption",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": dataset.id,
        "dataset_revision": dataset.revision,
        "split": dataset.split,
        "sample_count": args.samples,
        "ledger": str(ledger_path),
        "ledger_sha256": digest(ledger_path),
        "ledger_shards": ledger_bindings,
        "parent_plan": str(parent_path),
        "parent_plan_sha256": digest(parent_path),
        "excluded_plans": exclusion_bindings,
        "excluded_identity_count": len(excluded),
        "sample_ids": [sample["sample_id"] for sample in ledger["samples"]],
        "processor_signature": parent["processor_signature"],
        "caps_mib": list(CAPS_MIB),
        "draft_steps": DRAFT_STEPS,
        "warmups": 5,
        "output_tokens": 64,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": SEED,
        "max_global_a100s": 4,
        "cpus_per_worker": 12,
        "analysis_cpus": 12,
        "time_limit": "02:00:00",
        "comparison": {
            "physical_byte_cap": True,
            "largest_feasible_below_cap": True,
            "realized_byte_tolerance": 0,
            "fullkv_quality_reference_only": True,
            "minimum_initial_final_peak_memory_reduction": 0.30,
            "quality_noninferiority_margin": 0.0,
            "paired_cider_ci_lower_bound_required": 0.0,
        },
        "candidate_analysis_label": "output_preserving_draft_joint",
        "fixed_candidate": {
            "effective_method": "mosaickv_exact__output_preserving_v1__draft_joint",
            "cap_mib": 25.5,
            "query_evidence": "individual",
            "max_routed_heads": 16,
            "max_swaps_per_head": 2,
            "shortlist_size": 16,
            "pair_chunk_size": 64,
            "draft_steps": DRAFT_STEPS,
            "draft_ranking_mode": "joint_normalized",
            "prompt_objective_weight": 0.5,
            "draft_objective_weight": 0.5,
            "minimum_joint_improvement": 1e-7,
            "selection": "equal_byte_joint_normalized_prompt_and_draft_swaps",
        },
        "jobs": [],
    }

    candidates_25 = configurations(source, dataset, 25.5)
    candidates_40 = configurations(source, dataset, 40.0)
    joint = _joint_candidate(candidates_25["output_preserving_individual"])
    identity = replace(joint, cache=replace(joint.cache, retention_ratio=1.0))
    parity_path = root / "configs" / "retention-one.json"
    write_once(parity_path, canonical_config(identity))
    plan["retention_one_config"] = str(parity_path)
    plan["retention_one_config_sha256"] = config_sha256(identity)

    scheduled = (
        ("full_kv", "full_kv", None, candidates_25["full_kv"]),
        (
            "prompt_attention_topk_25p5",
            "prompt_attention_topk_25p5",
            25.5,
            candidates_25["prompt_attention_topk"],
        ),
        (
            "prompt_attention_topk_40p0",
            "prompt_attention_topk_40p0",
            40.0,
            candidates_40["prompt_attention_topk"],
        ),
        (
            "output_preserving_draft_joint_25p5",
            "output_preserving_draft_joint",
            25.5,
            joint,
        ),
    )
    jobs = plan["jobs"]
    assert isinstance(jobs, list)
    for label, analysis_label, cap_mib, config in scheduled:
        path = root / "configs" / f"{label}.json"
        write_once(path, canonical_config(config))
        expected = (
            config.output_preserving.effective_method
            if config.output_preserving is not None
            else config.method.value
        )
        index = len(jobs)
        jobs.append(
            {
                "index": index,
                "label": label,
                "analysis_label": analysis_label,
                "method": config.method.value,
                "effective_method": expected,
                "cap_mib": cap_mib,
                "cap_bytes": None if cap_mib is None else round(cap_mib * 2**20),
                "config": str(path),
                "config_sha256": config_sha256(config),
                "output": str(root / "outputs" / f"{index:03d}-{label}"),
            }
        )
    write_once(root / "plan.json", plan)
    print(
        json.dumps(
            {
                "root": str(root),
                "jobs": len(jobs),
                "samples": args.samples,
                "excluded_identity_count": len(excluded),
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
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--ledger-shard", type=Path, action="append", required=True)
    parser.add_argument("--excluded-plan", type=Path, action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    prepare(parser.parse_args())
