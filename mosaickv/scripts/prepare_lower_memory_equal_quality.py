#!/usr/bin/env python3
"""Prepare, but never submit, a lower-memory/equal-quality native-metric plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from isoquality_gate import validate_isoquality_protocol
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

SEED = 20271014
MILEBENCH_SUBTASKS = ("ActionPrediction", "ActionSequence", "CLEVR-Change", "Spot-the-Diff")


def _rank(group: str, sample_ids: list[str]) -> list[str]:
    return sorted(
        sample_ids,
        key=lambda sample_id: hashlib.sha256(
            f"{SEED}:{group}:{sample_id}".encode()
        ).digest(),
    )


def select_samples(
    groups: dict[str, list[str]], *, excluded: set[str], validation_per_group: int,
    paper_per_group: int,
) -> dict[str, list[str]]:
    """Select by identity only; answers and predictions are never accepted."""

    if not groups or validation_per_group < 1 or paper_per_group < 1:
        raise ValueError("positive nonempty strata required")
    all_ids = [sample_id for values in groups.values() for sample_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate candidate identity")
    ranked = {}
    for group, values in sorted(groups.items()):
        ranked[group] = _rank(group, [value for value in values if value not in excluded])
        if len(ranked[group]) < validation_per_group + paper_per_group:
            raise ValueError(f"insufficient unconsumed samples: {group}")
    result: dict[str, list[str]] = {"validation": [], "larger": []}
    for stage, start, count in (
        ("validation", 0, validation_per_group),
        ("larger", validation_per_group, paper_per_group),
    ):
        result[stage] = [
            ranked[group][index]
            for index in range(start, start + count)
            for group in sorted(ranked)
        ]
    return result


def configurations(
    source: Path,
    *,
    task_name: str,
    dataset: DatasetConfig,
    cap_mib: int,
    contract: dict,
) -> dict:
    lane = "lookm" if task_name == "milebench" else "vl_cache"
    rule = contract["lanes"][lane]
    if cap_mib not in rule["caps_mib"]:
        raise ValueError("unregistered physical cap")
    configs = byte_capped_configs(
        load_config(source / "mosaickv/configs/paper_candidate_llava_v1.yaml"),
        repository=source,
        family="vl_cache",
        task=task_name,
        dataset=dataset,
        model_id=rule["model_id"],
        model_revision=rule["model_revision"],
        seed=SEED,
        output_length=rule["tasks"][task_name]["output_tokens"],
        byte_cap=cap_mib * 2**20,
    )
    if task_name == "coco_caption":
        for method, config in tuple(configs.items()):
            if method != "full_kv":
                configs[method] = replace(
                    config,
                    cache=replace(config.cache, retention_ratio=math.nextafter(1.0, 0.0)),
                )
    shared = configs["prompt_attention_topk"].cache
    configs["full_kv"] = replace(
        configs["full_kv"],
        cache=replace(
            configs["full_kv"].cache,
            physical_budget_mandatory_policy=shared.physical_budget_mandatory_policy,
            packing_mode=shared.packing_mode,
        ),
    )
    if task_name == "milebench":
        controls = {
            name: replace(
                configs[name],
                cache=replace(
                    configs[name].cache,
                    block_size=1,
                    physical_budget_block_size=1,
                    physical_budget_mandatory_policy="first_and_last",
                ),
            )
            for name in ("full_kv", "prompt_attention_topk", "mosaickv_exact")
        }
        configs = matched_configs(controls, lane="lookm")
    for config in configs.values():
        validate_configuration(contract, lane, task_name, config)
    return configs


def _sample_payload(sample) -> dict:
    payload = {
        "sample_id": sample.sample_id,
        "prompt": sample.prompt,
        "references": list(sample.references),
        "media": [
            {"kind": media.kind.value, "path": str(media.path), "sha256": media.sha256}
            for media in sample.media
        ],
    }
    if sample.milebench_dataset is not None:
        payload.update(
            milebench_dataset=sample.milebench_dataset,
            milebench_question_type=sample.milebench_question_type.value,
            choices=list(sample.choices),
        )
    return payload


def prepare(args: argparse.Namespace) -> dict:
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, args.source_sha)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    validate_isoquality_protocol(protocol)
    if protocol["source_sha"] != args.source_sha or not protocol["execution_authorized"]:
        raise ValueError("protocol must be sealed and explicitly execution-authorized")
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path)
    if (
        contract["candidate"]["base_commit"] != args.source_sha
        or contract["candidate"]["content_sha256"] != candidate_content_digest(source)
    ):
        raise ValueError("candidate source/content is not sealed")
    task = default_task_registry().resolve(args.task)
    lane = "lookm" if task.name == "milebench" else "vl_cache"
    rule = contract["lanes"][lane]
    route = rule["tasks"][task.name]
    root = args.root.resolve()
    scratch = Path("/scratch/djy8hg").resolve()
    if root.exists() or scratch not in root.parents or source in root.parents:
        raise ValueError("new external scratch output root required")
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in args.ledger]
    meta = {key: inputs[0][key] for key in ("task", "dataset_id", "dataset_revision", "split")}
    if any(any(item[key] != value for key, value in meta.items()) for item in inputs):
        raise ValueError("input ledger provenance mismatch")
    dataset = DatasetConfig(meta["dataset_id"], meta["dataset_revision"], meta["split"])
    if (dataset.id, dataset.revision, dataset.split) != (
        route["dataset_id"], route["dataset_revision"], route["split"]
    ):
        raise ValueError("ledger route differs from preregistration")
    exclusions: set[str] = set()
    if not args.exclude_ledger:
        raise ValueError("paper plan requires an explicit prior-use exclusion ledger")
    bound_files = {
        str(args.protocol.resolve()): digest(args.protocol),
        str(contract_path): digest(contract_path),
    }
    for path in args.exclude_ledger:
        payload = json.loads(path.read_text(encoding="utf-8"))
        exclusions.update(payload.get("sample_ids", []))
        exclusions.update(sample["sample_id"] for sample in payload.get("samples", []))
        bound_files[str(path.resolve())] = digest(path)
    samples: dict[str, dict] = {}
    groups: dict[str, list[str]] = (
        {name: [] for name in MILEBENCH_SUBTASKS}
        if task.name == "milebench"
        else {task.name: []}
    )
    for path in args.ledger:
        validate_ledger(path, digest(path), route, task.name)
        checked = load_paper_native_ledger(
            path,
            expected_sha256=digest(path),
            task=task,
            dataset_revision=dataset.revision,
        )
        for sample in checked.samples:
            if sample.sample_id in samples:
                raise ValueError("duplicate materialized identity")
            samples[sample.sample_id] = _sample_payload(sample)
            group = sample.milebench_dataset if task.name == "milebench" else task.name
            if group in groups:
                groups[group].append(sample.sample_id)
        bound_files[str(path.resolve())] = digest(path)
    if task.name == "milebench":
        lane_protocol = protocol["lanes"]["lookm_milebench"]
        groups_per_stage = len(MILEBENCH_SUBTASKS)
        validation_count = lane_protocol["validation_sample_count"]
        paper_count = lane_protocol["sample_count"]
        if validation_count % groups_per_stage or paper_count % groups_per_stage:
            raise ValueError("MileBench stage counts must be balanced across subtasks")
        stages = select_samples(
            groups,
            excluded=exclusions,
            validation_per_group=validation_count // groups_per_stage,
            paper_per_group=paper_count // groups_per_stage,
        )
    else:
        lane_protocol = protocol["lanes"]["vl_cache_coco"]
        stages = select_samples(
            groups,
            excluded=exclusions,
            validation_per_group=lane_protocol["validation_sample_count"],
            paper_per_group=lane_protocol["sample_count"],
        )
    caps = tuple(rule["caps_mib"])
    plan = {
        "schema_version": 1,
        "classification": "preregistered_lower_memory_equal_quality",
        "source_sha": args.source_sha,
        "source_root": str(source),
        "comparison_contract": str(contract_path),
        "comparison_contract_sha256": digest(contract_path),
        "candidate_content_sha256": contract["candidate"]["content_sha256"],
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": digest(args.protocol),
        "task": task.name,
        "model_id": rule["model_id"],
        "model_revision": rule["model_revision"],
        "dataset_id": dataset.id,
        "dataset_revision": dataset.revision,
        "result_category": "unified_comparison",
        "heldout_claim": True,
        "historical_nominal_ratio_rerun": False,
        "paper_claim_authorized": False,
        "validation_execution_authorized": True,
        "larger_execution_authorized": False,
        "final_execution_authorized": False,
        "prior_use_disclosures": ["all explicit exclusion ledgers are hash bound"],
        "full_pool_size": len(samples),
        "processor_signature": inputs[0].get("processor_signature"),
        "bound_files": bound_files,
        "jobs": [],
        "methods": list(rule["methods"]),
        "caps_mib": list(caps),
        "validation_caps_mib": list(caps),
        "stage_sample_counts": {name: len(values) for name, values in stages.items()},
        "max_global_a100s": 6,
        "a100s_per_worker": 1,
        "cpus_per_worker": 12,
        "time_limit": "02:00:00",
        "partition": "bii-gpu",
        "reservation": "bi_fox_dgx",
        "warmups": 5,
        "timed_quality_trials": 1,
        "output_tokens": route["output_tokens"],
        "fullkv_latency_comparison": False,
        "isoquality_minimum_memory_reduction": 0.30,
        "isoquality_quality_margin": 0.0,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": SEED,
        "parity_tokens": 16,
        "parity_atol": 1e-4,
    }
    for stage, stage_ids in stages.items():
        for shard, offset in enumerate(range(0, len(stage_ids), 16)):
            shard_ids = stage_ids[offset : offset + 16]
            ledger_path = root / "ledgers" / stage / f"{shard:03d}.json"
            write_once(
                ledger_path,
                {"schema_version": 1, **meta, "samples": [samples[sid] for sid in shard_ids]},
            )
            bound_files[str(ledger_path)] = digest(ledger_path)
            for cap in caps:
                for method, config in configurations(
                    source,
                    task_name=task.name,
                    dataset=dataset,
                    cap_mib=cap,
                    contract=contract,
                ).items():
                    if method == "full_kv" and cap != caps[0]:
                        continue
                    config_path = root / "configs" / stage / f"{method}-{cap}.json"
                    if not config_path.exists():
                        write_once(config_path, canonical_config(config))
                        bound_files[str(config_path)] = digest(config_path)
                    plan["jobs"].append(
                        {
                            "stage": stage,
                            "shard": shard,
                            "method": method,
                            "cap_mib": None if method == "full_kv" else cap,
                            "sample_ids": shard_ids,
                            "ledger": str(ledger_path),
                            "ledger_sha": digest(ledger_path),
                            "config": str(config_path),
                            "config_sha": config_sha256(config),
                            "output": str(root / "outputs" / stage / f"{shard:03d}-{method}-{cap}"),
                        }
                    )
    write_once(root / "plan.json", plan)
    result = {
        "plan": str(root / "plan.json"),
        "plan_sha256": digest(root / "plan.json"),
        "submitted": False,
        "validation_jobs": sum(job["stage"] == "validation" for job in plan["jobs"]),
        "paper_jobs": sum(job["stage"] == "larger" for job in plan["jobs"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("coco_caption", "milebench"), required=True)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--exclude-ledger", type=Path, action="append", default=[])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("mosaickv/configs/experiments/lower_memory_equal_quality_v1.yaml"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("mosaickv/configs/experiments/shared_byte_isoquality_v1.yaml"),
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
