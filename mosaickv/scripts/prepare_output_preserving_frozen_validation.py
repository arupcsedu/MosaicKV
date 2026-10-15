#!/usr/bin/env python3
"""Prepare the frozen 128-sample draft-gated development validation."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from prepare_output_preserving_draft_sweep import DRAFT_STEPS, _draft_candidate
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


def _git_snapshot(source: Path, expected_sha: str) -> None:
    observed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected_sha:
        raise ValueError(f"source SHA mismatch: expected {expected_sha}, observed {observed}")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("fresh validation requires a clean source worktree")


def merge_reviewed_ledgers(
    paths: list[Path],
    *,
    parent_plan: dict[str, object],
    excluded_ids: set[str],
    expected_samples: int = SAMPLE_COUNT,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Merge reviewed, unexecuted shards and fail closed on identity reuse."""

    if not paths:
        raise ValueError("at least one reviewed ledger shard is required")
    jobs = parent_plan.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("parent plan has no job bindings")
    samples: list[dict[str, object]] = []
    bindings: list[dict[str, str]] = []
    route: tuple[object, ...] | None = None
    for raw_path in paths:
        path = raw_path.resolve()
        ledger = json.loads(path.read_text())
        current_route = (
            ledger.get("task"),
            ledger.get("dataset_id"),
            ledger.get("dataset_revision"),
            ledger.get("split"),
        )
        if route is None:
            route = current_route
        elif current_route != route:
            raise ValueError("ledger shards do not share one dataset route")
        sha = digest(path)
        matched = [
            job
            for job in jobs
            if Path(str(job.get("ledger", ""))).resolve() == path
            and job.get("ledger_sha") == sha
        ]
        if not matched:
            raise ValueError(f"ledger is not hash-bound by the parent plan: {path}")
        for job in matched:
            output = Path(str(job.get("output", "")))
            if output.exists() and any(output.rglob("*")):
                raise ValueError(f"reviewed ledger already has execution output: {output}")
        shard_samples = ledger.get("samples")
        if not isinstance(shard_samples, list) or not shard_samples:
            raise ValueError(f"ledger has no samples: {path}")
        samples.extend(shard_samples)
        bindings.append({"path": str(path), "sha256": sha})
    if route != (
        "coco_caption",
        "lmms-lab/COCO-Caption",
        "c3213b54e173ae86517d481de17f31439d6f8c72",
        "val",
    ):
        raise ValueError("wrong COCO route")
    ids = [str(sample.get("sample_id")) for sample in samples]
    if len(samples) != expected_samples or len(set(ids)) != expected_samples:
        raise ValueError("merged ledger must contain exactly 128 unique samples")
    overlap = sorted(set(ids) & excluded_ids)
    if overlap:
        raise ValueError(f"validation identities overlap inspected cohorts: {overlap}")
    return (
        {
            "schema_version": 1,
            "task": route[0],
            "dataset_id": route[1],
            "dataset_revision": route[2],
            "split": route[3],
            "samples": samples,
        },
        bindings,
    )


def prepare(args: argparse.Namespace) -> None:
    source, root = args.source.resolve(), args.root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("a new external scratch output root is required")
    _git_snapshot(source, args.source_sha)
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
        payload = json.loads(path.read_text())
        excluded.update(str(item) for item in payload.get("sample_ids", ()))
        exclusion_bindings.append({"path": str(path.resolve()), "sha256": digest(path)})
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
        "protocol_version": "output_preserving_draft_frozen_validation_v1",
        "classification": CLASSIFICATION,
        "cohort_relation": "disjoint_from_both_inspected_output_preserving_cohorts",
        "heldout_claim": False,
        "paper_claim_authorized": False,
        "paper_campaign_submission_authorized": False,
        "candidate_frozen_before_sample_execution": True,
        "candidate_retuning_authorized": False,
        "source_root": str(source),
        "source_base_git_sha": args.source_sha,
        "source_worktree_dirty": False,
        "source_content_sha256": source_digest(source),
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
        "fixed_candidate": {
            "effective_method": "mosaickv_exact__output_preserving_v1__draft_gate",
            "cap_mib": 25.5,
            "query_evidence": "individual",
            "max_routed_heads": 16,
            "max_swaps_per_head": 2,
            "shortlist_size": 16,
            "pair_chunk_size": 64,
            "draft_steps": DRAFT_STEPS,
            "selection": "equal_byte_prompt_best_swaps_with_draft_nonregression_veto",
        },
        "jobs": [],
    }

    candidates_25 = configurations(source, dataset, 25.5)
    candidates_40 = configurations(source, dataset, 40.0)
    identity = replace(
        _draft_candidate(candidates_25["output_preserving_individual"]),
        cache=replace(
            candidates_25["output_preserving_individual"].cache, retention_ratio=1.0
        ),
    )
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
            "output_preserving_draft4_25p5",
            "output_preserving_draft_gate",
            25.5,
            _draft_candidate(candidates_25["output_preserving_individual"]),
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
