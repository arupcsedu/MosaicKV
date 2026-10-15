#!/usr/bin/env python3
"""Prepare a fresh-cohort deterministic draft-gated output-selection ablation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from prepare_output_preserving_capacity_sweep import AGGRESSIVE_CAPS_MIB, PROMPT_CAPS_MIB
from prepare_output_preserving_sweep import (
    MODEL_ID,
    MODEL_REVISION,
    SEED,
    configurations,
    source_digest,
)
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import DatasetConfig, canonical_config, config_sha256
from mosaickv.types import ForecastMode

DRAFT_STEPS = 4


def _draft_candidate(base):
    if base.output_preserving is None:
        raise ValueError("output-preserving base configuration is missing")
    return replace(
        base,
        forecasting=replace(
            base.forecasting,
            mode=ForecastMode.HYBRID,
            draft_steps=DRAFT_STEPS,
        ),
        output_preserving=replace(
            base.output_preserving,
            max_swaps_per_head=2,
            draft_validation=True,
        ),
    )


def prepare(args: argparse.Namespace) -> None:
    source, root = args.source.resolve(), args.root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("a new external scratch output root is required")
    ledger = json.loads(args.ledger.read_text())
    samples = ledger.get("samples", [])
    if len(samples) != args.samples or len({s["sample_id"] for s in samples}) != args.samples:
        raise ValueError("ledger must contain the registered unique sample count")
    if (
        ledger.get("task") != "coco_caption"
        or ledger.get("dataset_id") != "lmms-lab/COCO-Caption"
        or ledger.get("dataset_revision") != "c3213b54e173ae86517d481de17f31439d6f8c72"
        or ledger.get("split") != "val"
    ):
        raise ValueError("wrong COCO route")
    excluded = json.loads(args.excluded_plan.read_text())
    overlap = set(excluded.get("sample_ids", ())) & {s["sample_id"] for s in samples}
    if overlap:
        raise ValueError(
            f"fresh development cohort overlaps the inspected sweep: {sorted(overlap)}"
        )
    parent_plan = json.loads(args.parent_plan.read_text())
    if (
        parent_plan.get("model_id") != MODEL_ID
        or parent_plan.get("model_revision") != MODEL_REVISION
        or parent_plan.get("dataset_id") != ledger["dataset_id"]
        or parent_plan.get("dataset_revision") != ledger["dataset_revision"]
        or not isinstance(parent_plan.get("processor_signature"), dict)
    ):
        raise ValueError("reviewed parent plan has incompatible controls")
    ledger_path = args.ledger.resolve()
    if not any(
        Path(job.get("ledger", "")).resolve() == ledger_path
        and job.get("ledger_sha") == digest(args.ledger)
        for job in parent_plan.get("jobs", [])
    ):
        raise ValueError("sample ledger is not hash-bound by the reviewed parent plan")

    dataset = DatasetConfig(ledger["dataset_id"], ledger["dataset_revision"], ledger["split"])
    plan = {
        "schema_version": 1,
        "classification": "retrospective_development_only_not_paper_evidence",
        "cohort_relation": "disjoint_from_inspected_prompt_only_capacity_sweep",
        "heldout_claim": False,
        "paper_claim_authorized": False,
        "source_root": str(source),
        "source_base_git_sha": args.base_sha,
        "source_worktree_dirty": True,
        "source_content_sha256": source_digest(source),
        "task": "coco_caption",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": dataset.id,
        "dataset_revision": dataset.revision,
        "split": dataset.split,
        "sample_count": args.samples,
        "ledger": str(ledger_path),
        "ledger_sha256": digest(args.ledger),
        "parent_plan": str(args.parent_plan.resolve()),
        "parent_plan_sha256": digest(args.parent_plan),
        "excluded_plan": str(args.excluded_plan.resolve()),
        "excluded_plan_sha256": digest(args.excluded_plan),
        "sample_ids": [s["sample_id"] for s in samples],
        "processor_signature": parent_plan["processor_signature"],
        "caps_mib": list(PROMPT_CAPS_MIB),
        "aggressive_caps_mib": list(AGGRESSIVE_CAPS_MIB),
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
            "fullkv_latency_comparison": False,
            "minimum_memory_reduction": 0.30,
            "quality_noninferiority_margin": 0.0,
        },
        "fixed_controls": {
            "query_evidence": "individual",
            "max_routed_heads": 16,
            "max_swaps_per_head": 2,
            "shortlist_size": 16,
            "pair_chunk_size": 64,
            "selection": "equal_byte_corrective_swaps",
        },
        "jobs": [],
    }

    first = configurations(source, dataset, AGGRESSIVE_CAPS_MIB[0])
    identity = replace(
        _draft_candidate(first["output_preserving_individual"]),
        cache=replace(first["output_preserving_individual"].cache, retention_ratio=1.0),
    )
    parity_path = root / "configs" / "retention-one.json"
    write_once(parity_path, canonical_config(identity))
    plan["retention_one_config"] = str(parity_path)
    plan["retention_one_config_sha256"] = config_sha256(identity)

    scheduled = []
    for cap_mib in PROMPT_CAPS_MIB:
        slug = str(cap_mib).replace(".", "p")
        prompt = configurations(source, dataset, cap_mib)["prompt_attention_topk"]
        scheduled.append(
            (f"prompt_attention_topk_{slug}", "prompt_attention_topk", cap_mib, prompt)
        )
    for cap_mib in AGGRESSIVE_CAPS_MIB:
        slug = str(cap_mib).replace(".", "p")
        prompt_candidate = configurations(source, dataset, cap_mib)[
            "output_preserving_individual"
        ]
        scheduled.extend(
            (
                (
                    f"output_preserving_prompt_{slug}",
                    "output_preserving_prompt",
                    cap_mib,
                    prompt_candidate,
                ),
                (
                    f"output_preserving_draft{DRAFT_STEPS}_{slug}",
                    "output_preserving_draft_gate",
                    cap_mib,
                    _draft_candidate(prompt_candidate),
                ),
            )
        )

    for label, analysis_label, cap_mib, config in scheduled:
        path = root / "configs" / f"{label}.json"
        write_once(path, canonical_config(config))
        expected = (
            config.output_preserving.effective_method
            if config.output_preserving is not None
            else config.method.value
        )
        index = len(plan["jobs"])
        plan["jobs"].append(
            {
                "index": index,
                "label": label,
                "analysis_label": analysis_label,
                "method": config.method.value,
                "effective_method": expected,
                "cap_mib": cap_mib,
                "cap_bytes": round(cap_mib * 2**20),
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
                "jobs": len(plan["jobs"]),
                "plan_sha256": digest(root / "plan.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--excluded-plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    prepare(parser.parse_args())
