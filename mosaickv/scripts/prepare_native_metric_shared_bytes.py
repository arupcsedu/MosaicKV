#!/usr/bin/env python3
"""Prepare, but never submit, new COCO/MileBench shared-byte lane inputs.

Existing nominal-ratio artifacts are read-only historical evidence, not an
input to score-based selection. Dataset/media materialization is separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from run_baseline_fidelity_review_worker import byte_capped_configs
from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import DatasetConfig, canonical_config, config_sha256, load_config
from mosaickv.evaluation.hf_matched_validation import matched_configs
from mosaickv.evaluation.paper_native_lanes import load_paper_native_ledger
from mosaickv.evaluation.shared_byte_contract import (
    candidate_content_digest,
    load_contract,
    validate_configuration,
    validate_ledger,
)
from mosaickv.evaluation.tasks import default_task_registry

CAPS = (10, 20, 40, 80)
MILEBENCH_SUBTASKS = ("ActionPrediction", "ActionSequence", "CLEVR-Change", "Spot-the-Diff")
MODELS = {
    "coco_caption": (
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "2424fdd47412fccc66d91719126b420e9fbd7065",
    ),
    "milebench": ("llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"),
}
SEED = 20271014


def select_stages(groups, *, excluded=(), validation_per_group=8, larger_per_group=64):
    """Hash-ranked IDs only; no access to answers, predictions or scores."""
    if validation_per_group < 1 or larger_per_group < 1:
        raise ValueError("positive stage sizes required")
    all_ids = [s for group in groups.values() for s in group]
    if not groups or len(all_ids) != len(set(all_ids)):
        raise ValueError("empty groups or duplicate input identities")
    exclusions = set(excluded)
    stages = {"validation": [], "larger": []}
    ranked = {}
    for name, ids in sorted(groups.items()):
        ranked[name] = sorted(
            (sid for sid in ids if sid not in exclusions),
            key=lambda sid: hashlib.sha256(f"{SEED}:{name}:{sid}".encode()).digest(),
        )
        if len(ranked[name]) < validation_per_group + larger_per_group:
            raise ValueError(f"insufficient distinct inputs in subtask {name}")
    # Round-robin groups so each full shard represents every registered subtask.
    for stage, start, count in (
        ("validation", 0, validation_per_group),
        ("larger", validation_per_group, larger_per_group),
    ):
        stages[stage] = [
            ranked[name][index] for index in range(start, start + count) for name in sorted(ranked)
        ]
    return stages


def configurations(source, task, dataset, cap_mib):
    if task not in MODELS or cap_mib not in CAPS:
        raise ValueError("unregistered task or byte cap")
    model_id, revision = MODELS[task]
    cfgs = byte_capped_configs(
        load_config(source / "mosaickv/configs/paper_candidate_llava_v1.yaml"),
        repository=source,
        family="vl_cache",
        task=task,
        dataset=dataset,
        model_id=model_id,
        model_revision=revision,
        seed=SEED,
        output_length=64,
        byte_cap=cap_mib * 2**20,
    )
    # This is a byte-only lane. The historical 0.99 sentinel otherwise prunes
    # one percent even when an input's full cache is already below the cap.
    # Stay below exactly 1 (the identity fast path) while ceil(source_bytes*r)
    # remains source_bytes for every supported cache smaller than 2**52 bytes.
    for method, cfg in cfgs.items():
        if task == "coco_caption" and method != "full_kv":
            cfgs[method] = replace(
                cfg, cache=replace(cfg.cache, retention_ratio=math.nextafter(1.0, 0.0))
            )
    # FullKV keeps every token and has no finite cap, but declare the lane's
    # cache policy consistently in its reference manifest. No pruning or
    # packing is invoked on its retention-one path.
    shared_cache = cfgs["prompt_attention_topk"].cache
    cfgs["full_kv"] = replace(
        cfgs["full_kv"],
        cache=replace(
            cfgs["full_kv"].cache,
            physical_budget_mandatory_policy=shared_cache.physical_budget_mandatory_policy,
            packing_mode=shared_cache.packing_mode,
        ),
    )
    if task == "milebench":
        controls = {
            name: replace(
                cfgs[name],
                cache=replace(
                    cfgs[name].cache,
                    block_size=1,
                    physical_budget_block_size=1,
                    physical_budget_mandatory_policy="first_and_last",
                ),
            )
            for name in ("full_kv", "prompt_attention_topk", "mosaickv_exact")
        }
        return matched_configs(controls, lane="lookm")
    return cfgs


def prepare(args):
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, args.source_sha)
    contract_path = source / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml"
    contract = load_contract(contract_path)
    if candidate_content_digest(source) != contract["candidate"]["content_sha256"]:
        raise ValueError("frozen candidate content changed; register a new version")
    root = args.root.resolve()
    scratch = Path("/scratch/djy8hg").resolve()
    if scratch not in root.parents or source == root or source in root.parents:
        raise ValueError("output root must be external scratch")
    if root.exists():
        raise FileExistsError("new immutable root required")
    task = default_task_registry().resolve(args.task)
    available_methods = tuple(configurations(
        source,
        task.name,
        DatasetConfig(
            contract["lanes"]["lookm" if task.name == "milebench" else "vl_cache"]["tasks"][
                task.name
            ]["dataset_id"],
            contract["lanes"]["lookm" if task.name == "milebench" else "vl_cache"]["tasks"][
                task.name
            ]["dataset_revision"],
            contract["lanes"]["lookm" if task.name == "milebench" else "vl_cache"]["tasks"][
                task.name
            ]["split"],
        ),
        40,
    ))
    requested_methods = tuple(getattr(args, "methods", None) or available_methods)
    if (
        len(requested_methods) != len(set(requested_methods))
        or any(method not in available_methods for method in requested_methods)
        or "full_kv" not in requested_methods
        or "mosaickv_exact" not in requested_methods
    ):
        raise ValueError(
            "methods must be distinct registered methods including FullKV and MosaicKV"
        )
    baseline = "lookm_reimpl" if task.name == "milebench" else "vl_cache_reimpl"
    if baseline not in requested_methods:
        raise ValueError(f"minimum lane requires {baseline}")
    larger_caps = tuple(getattr(args, "larger_caps", None) or CAPS)
    if (
        len(larger_caps) != len(set(larger_caps))
        or any(cap not in CAPS for cap in larger_caps)
    ):
        raise ValueError("larger caps must be distinct registered byte caps")
    inputs = [json.loads(path.read_text()) for path in args.ledger]
    meta = {k: inputs[0][k] for k in ("task", "dataset_id", "dataset_revision", "split")}
    if meta["task"] != task.name or any(
        any(item[k] != value for k, value in meta.items()) for item in inputs
    ):
        raise ValueError("input ledger provenance mismatch")
    if (meta["dataset_id"], meta["split"]) != (task.dataset_id, task.split):
        raise ValueError("input ledger route differs from registry")
    dataset = DatasetConfig(meta["dataset_id"], meta["dataset_revision"], meta["split"])
    lane = "lookm" if task.name == "milebench" else "vl_cache"
    # Resolve every planned configuration before writing any output. In
    # particular, an unresolved composite MileBench revision fails closed.
    for cap in CAPS:
        for cfg in configurations(source, task.name, dataset, cap).values():
            validate_configuration(contract, lane, task.name, cfg)
    for path in args.ledger:
        validate_ledger(path, digest(path), contract["lanes"][lane]["tasks"][task.name], task.name)
    bound = {str(path.resolve()): digest(path) for path in args.ledger}
    bound[str(contract_path)] = digest(contract_path)
    if task.name == "coco_caption" and any(
        p.get("processor_signature") != inputs[0].get("processor_signature") for p in inputs
    ):
        raise ValueError("COCO parent ledgers disagree on processor controls")
    excluded = []
    for path in args.exclude_ledger:
        payload = json.loads(path.read_text())
        if payload["task"] != task.name:
            raise ValueError("exclusion ledger task mismatch")
        excluded.extend(
            payload.get("sample_ids", [s["sample_id"] for s in payload.get("samples", [])])
        )
        bound[str(path.resolve())] = digest(path)
    materialized = {}
    if task.name == "milebench":
        groups = {key: [] for key in MILEBENCH_SUBTASKS}
        for path, payload in zip(args.ledger, inputs, strict=True):
            checked = load_paper_native_ledger(
                path, expected_sha256=digest(path), task=task, dataset_revision=dataset.revision
            )
            for sample, validated in zip(payload["samples"], checked.samples, strict=True):
                sample = dict(sample)
                sample["media"] = [
                    dict(kind=media.kind.value, path=str(media.path), sha256=media.sha256)
                    for media in validated.media
                ]
                sid = sample["sample_id"]
                if sid in materialized:
                    raise ValueError("duplicate materialized identity")
                materialized[sid] = sample
                if sample["milebench_dataset"] in groups:
                    groups[sample["milebench_dataset"]].append(sid)
        stages = select_stages(groups, excluded=excluded)
    else:
        if dataset.revision != "c3213b54e173ae86517d481de17f31439d6f8c72":
            raise ValueError("COCO requires the pinned caption dataset revision")
        for path, payload in zip(args.ledger, inputs, strict=True):
            checked = load_paper_native_ledger(
                path, expected_sha256=digest(path), task=task, dataset_revision=dataset.revision
            )
            if (
                not payload.get("bound_files")
                or not payload.get("historical_prior_use_disclosure")
                or not payload.get("processor_signature")
            ):
                raise ValueError(
                    "COCO requires frozen preprocessing assets and prior-use disclosure"
                )
            for name, sha in payload["bound_files"].items():
                if digest(Path(name)) != sha:
                    raise ValueError("materialized COCO dependency changed")
                bound[name] = sha
            for sample, validated in zip(payload["samples"], checked.samples, strict=True):
                sample = dict(sample)
                sample["media"] = [
                    dict(kind=m.kind.value, path=str(m.path), sha256=m.sha256)
                    for m in validated.media
                ]
                if sample["sample_id"] in materialized:
                    raise ValueError("duplicate materialized COCO identity")
                materialized[sample["sample_id"]] = sample
        groups = {task.name: list(materialized)}
        stages = select_stages(
            groups, excluded=excluded, validation_per_group=32, larger_per_group=256
        )
    plan = dict(
        schema_version=1,
        source_sha=args.source_sha,
        comparison_contract=str(contract_path),
        comparison_contract_sha256=digest(contract_path),
        candidate_content_sha256=contract["candidate"]["content_sha256"],
        source_root=str(source),
        classification="unified_shared_byte_native_metric_validation",
        task=task.name,
        model_id=MODELS[task.name][0],
        model_revision=MODELS[task.name][1],
        dataset_id=dataset.id,
        dataset_revision=dataset.revision,
        result_category="unified_comparison",
        heldout_claim=False,
        historical_nominal_ratio_rerun=False,
        paper_claim_authorized=False,
        larger_execution_authorized=False,
        final_execution_authorized=False,
        prior_use_disclosures=[
            p.get("historical_prior_use_disclosure", "ledger-defined") for p in inputs
        ],
        full_pool_size=len(materialized),
        processor_signature=inputs[0].get("processor_signature"),
        validation_execution_authorized=True,
        bound_files=bound,
        jobs=[],
        methods=list(requested_methods),
        max_global_a100s=4,
        a100s_per_worker=1,
        cpus_per_worker=12,
        time_limit="02:00:00",
        partition="bii-gpu",
        reservation="bi_fox_dgx",
        warmups=5,
        timed_quality_trials=1,
        output_tokens=64,
        fullkv_latency_comparison=False,
        caps_mib=list(larger_caps),
        comparison_boundary="initial physical KV cap plus equal append-only decode growth",
        require_equal_measured_initial_final_peak_storage=True,
        require_independent_largest_feasible_capacity_certificate=True,
        parity_tokens=16,
        parity_atol=1e-4,
        bootstrap_resamples=contract["statistics"]["paired_bootstrap_replicates"],
        bootstrap_seed=SEED,
        required_gate=(
            "complete canonical rows; same controls/IDs and initial/final bytes; finite metrics"
        ),
        metric_reduction=(
            "pooled corpus CIDEr, rerun reducer in each paired bootstrap replicate"
            if task.name == "coco_caption"
            else "official MileBench hierarchy; separate Accuracy/ROUGE-L; selected subtasks only"
        ),
        media_claim="reviewed ledger representation only; no native multi-image claim implied",
    )
    for stage, ids in stages.items():
        for shard, offset in enumerate(range(0, len(ids), 16)):
            shard_ids = ids[offset : offset + 16]
            payload = dict(schema_version=1, **meta)
            payload.update(
                {"samples": [materialized[sid] for sid in shard_ids]}
                if materialized
                else {"sample_ids": shard_ids}
            )
            ledger = root / "ledgers" / stage / f"{shard:03d}.json"
            write_once(ledger, payload)
            bound[str(ledger)] = digest(ledger)
            for cap in (40,) if stage == "validation" else larger_caps:
                for method, cfg in configurations(source, task.name, dataset, cap).items():
                    if method not in requested_methods:
                        continue
                    # Same FullKV controls at every cap: measure it once per shard.
                    if method == "full_kv" and stage == "larger" and cap != larger_caps[0]:
                        continue
                    config_path = root / "configs" / stage / f"{method}-{cap}.json"
                    if not config_path.exists():
                        write_once(config_path, canonical_config(cfg))
                        bound[str(config_path)] = digest(config_path)
                    plan["jobs"].append(
                        dict(
                            stage=stage,
                            shard=shard,
                            method=method,
                            cap_mib=None if method == "full_kv" else cap,
                            sample_ids=shard_ids,
                            ledger=str(ledger),
                            ledger_sha=digest(ledger),
                            config=str(config_path),
                            config_sha=config_sha256(cfg),
                            output=str(root / "outputs" / stage / f"{shard:03d}-{method}-{cap}"),
                        )
                    )
    write_once(root / "plan.json", plan)
    print(
        json.dumps(
            {
                "plan": str(root / "plan.json"),
                "sha256": digest(root / "plan.json"),
                "submitted": False,
                "larger_execution_authorized": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(MODELS), required=True)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--exclude-ledger", type=Path, action="append", default=[])
    parser.add_argument(
        "--methods",
        nargs="+",
        help="registered methods to plan; defaults to the complete shared-byte lane",
    )
    parser.add_argument(
        "--larger-caps",
        type=int,
        nargs="+",
        help="registered MiB caps for the larger stage; defaults to 10/20/40/80",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
