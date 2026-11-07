#!/usr/bin/env python3
"""Prepare an immutable retrospective COCO sweep for the development candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from run_baseline_fidelity_review_worker import byte_capped_configs
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import (
    DatasetConfig,
    ForecastingConfig,
    GraphConfig,
    LookMConfig,
    OutputPreservingConfig,
    PrefixKVConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    SelectionConfig,
    VLCacheConfig,
    canonical_config,
    config_sha256,
    load_config,
)
from mosaickv.types import ForecastMode, RepairPolicy

CAPS_MIB = (20, 25.5, 28, 40)
SEED = 20271014
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
MODEL_REVISION = "2424fdd47412fccc66d91719126b420e9fbd7065"


def source_digest(source: Path) -> str:
    paths = sorted((source / "mosaickv/src/mosaickv").rglob("*.py"))
    paths += [
        source / "mosaickv/scripts" / name
        for name in (
            "prepare_output_preserving_sweep.py",
            "prepare_output_preserving_capacity_sweep.py",
            "prepare_output_preserving_draft_sweep.py",
            "prepare_output_preserving_frozen_validation.py",
            "prepare_output_preserving_joint_validation.py",
            "prepare_output_preserving_short_logit_validation.py",
            "submit_output_preserving_joint_validation.py",
            "submit_output_preserving_frozen_validation.py",
            "promote_output_preserving_sweep.py",
            "run_output_preserving_sweep.py",
            "analyze_output_preserving_sweep.py",
            "analyze_output_preserving_capacity_sweep.py",
            "analyze_output_preserving_draft_sweep.py",
            "analyze_output_preserving_frozen_validation.py",
            "run_baseline_fidelity_review_worker.py",
            "run_coco_shared_bytes.py",
            "run_prefixkv_decode_preflight.py",
        )
    ]
    paths += [
        source / "mosaickv/slurm" / name
        for name in (
            "output_preserving_canary_promotion.sbatch",
            "output_preserving_sweep.sbatch",
            "output_preserving_sweep_analysis.sbatch",
            "output_preserving_capacity_analysis.sbatch",
            "output_preserving_draft_analysis.sbatch",
            "output_preserving_frozen_validation_analysis.sbatch",
        )
    ]
    payload = {
        str(path.relative_to(source)): digest(path)
        for path in paths
        if path.is_file()
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def configurations(source: Path, dataset: DatasetConfig, cap_mib: float):
    template = load_config(source / "mosaickv/configs/paper_candidate_llava_v1.yaml")
    common = byte_capped_configs(
        template,
        repository=source,
        family="vl_cache",
        task="coco_caption",
        dataset=dataset,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        seed=SEED,
        output_length=64,
        byte_cap=round(cap_mib * 2**20),
    )
    for name, config in tuple(common.items()):
        if name != "full_kv":
            common[name] = replace(
                config,
                cache=replace(config.cache, retention_ratio=math.nextafter(1.0, 0.0)),
            )
    base = common["mosaickv_exact"]

    def candidate(query_evidence: str, swaps: int):
        return replace(
            base,
            forecasting=ForecastingConfig(
                enabled=True,
                mode=ForecastMode.PROMPT_WINDOW,
                prompt_window=16,
                draft_steps=0,
            ),
            graph=GraphConfig(enabled=False),
            selection=SelectionConfig(enabled=True),
            prototypes=PrototypeConfig(enabled=False),
            residual=ResidualConfig(enabled=False),
            repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE, max_blocks_per_step=0),
            lookm=LookMConfig(enabled=False),
            prefixkv=PrefixKVConfig(enabled=False),
            vl_cache=VLCacheConfig(enabled=False),
            output_preserving=OutputPreservingConfig(
                query_evidence=query_evidence,
                max_routed_heads=16,
                max_swaps_per_head=swaps,
                shortlist_size=16,
                pair_chunk_size=64,
            ),
        )

    return {
        "full_kv": common["full_kv"],
        "prompt_attention_topk": common["prompt_attention_topk"],
        "vl_cache_reimpl": common["vl_cache_reimpl"],
        "frozen_mosaickv_exact": common["mosaickv_exact"],
        "output_preserving_anchor": candidate("individual", 0),
        "output_preserving_pooled": candidate("pooled", 2),
        "output_preserving_individual": candidate("individual", 2),
    }


def prepare(args: argparse.Namespace) -> None:
    source, root = args.source.resolve(), args.root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("a new external scratch output root is required")
    ledger = json.loads(args.ledger.read_text())
    samples = ledger.get("samples", [])
    if len(samples) != args.samples or len({s["sample_id"] for s in samples}) != args.samples:
        raise ValueError("ledger must contain the exact registered sample count without duplicates")
    if (
        ledger.get("task") != "coco_caption"
        or ledger.get("dataset_id") != "lmms-lab/COCO-Caption"
        or ledger.get("dataset_revision")
        != "c3213b54e173ae86517d481de17f31439d6f8c72"
        or ledger.get("split") != "val"
    ):
        raise ValueError("wrong COCO route")
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
    bindings = [
        job
        for job in parent_plan.get("jobs", [])
        if Path(job.get("ledger", "")).resolve() == ledger_path
        and job.get("ledger_sha") == digest(args.ledger)
    ]
    if not bindings:
        raise ValueError("sample ledger is not hash-bound by the reviewed parent plan")
    dataset = DatasetConfig(ledger["dataset_id"], ledger["dataset_revision"], ledger["split"])
    plan = {
        "schema_version": 1,
        "classification": "retrospective_development_only_not_paper_evidence",
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
        "ledger": str(args.ledger.resolve()),
        "ledger_sha256": digest(args.ledger),
        "parent_plan": str(args.parent_plan.resolve()),
        "parent_plan_sha256": digest(args.parent_plan),
        "sample_ids": [s["sample_id"] for s in samples],
        "processor_signature": parent_plan["processor_signature"],
        "caps_mib": list(CAPS_MIB),
        "warmups": 5,
        "output_tokens": 64,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": SEED,
        "max_global_a100s": 4,
        "cpus_per_worker": 12,
        "analysis_cpus": 12,
        "time_limit": "02:00:00",
        "canary_indices": [0, 12],
        "remaining_indices": [*range(1, 12), *range(13, 25)],
        "comparison": {
            "physical_byte_cap": True,
            "largest_feasible_below_cap": True,
            "realized_byte_tolerance": 0,
            "fullkv_latency_comparison": False,
            "minimum_memory_reduction": 0.30,
            "quality_noninferiority_margin": 0.0,
        },
        "jobs": [],
    }
    parity_path = root / "configs" / "retention-one.json"
    first_candidates = configurations(source, dataset, CAPS_MIB[0])
    identity = replace(
        first_candidates["output_preserving_individual"],
        cache=replace(first_candidates["output_preserving_individual"].cache, retention_ratio=1.0),
    )
    write_once(parity_path, canonical_config(identity))
    plan["retention_one_config"] = str(parity_path)
    plan["retention_one_config_sha256"] = config_sha256(identity)
    for cap_mib in CAPS_MIB:
        for label, config in configurations(source, dataset, cap_mib).items():
            if label == "full_kv" and cap_mib != CAPS_MIB[0]:
                continue
            slug = str(cap_mib).replace(".", "p")
            path = root / "configs" / f"{label}-{slug}.json"
            write_once(path, canonical_config(config))
            expected = (
                config.output_preserving.effective_method
                if config.output_preserving is not None
                else config.method.value
            )
            plan["jobs"].append(
                {
                    "index": len(plan["jobs"]),
                    "label": label,
                    "method": config.method.value,
                    "effective_method": expected,
                    "cap_mib": None if label == "full_kv" else cap_mib,
                    "cap_bytes": None if label == "full_kv" else round(cap_mib * 2**20),
                    "config": str(path),
                    "config_sha256": config_sha256(config),
                    "output": str(root / "outputs" / f"{len(plan['jobs']):03d}-{label}-{slug}"),
                }
            )
    write_once(root / "plan.json", plan)
    print(json.dumps({"root": str(root), "jobs": len(plan["jobs"]),
                      "plan_sha256": digest(root / "plan.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    prepare(parser.parse_args())
