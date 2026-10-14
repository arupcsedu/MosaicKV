#!/usr/bin/env python3
"""CPU-only, write-once ablation configs from label-free captures; never submits jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from generate_prefixkv_profile import _ids, _observations

from mosaickv.baselines.prefixkv import generate_prefixkv_byte_profile
from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.prefixkv_evidence_ablation import (
    EVIDENCE_ABLATIONS,
    evidence_ablation_configs,
)


def prepare(args):
    controls = {
        name: RunConfig.from_mapping(value)
        for name, value in json.loads(args.controls_json.read_text()).items()
    }
    template = controls["prompt_attention_topk"]
    observations = _observations(args.attention_jsonl)
    evaluation_ids = _ids(args.evaluation_sample_ids)
    if not evaluation_ids or len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("require nonempty unique evaluation identities")
    profiles = {
        arm.name: generate_prefixkv_byte_profile(
            observations,
            byte_cap=template.cache.budget_value,
            bytes_per_layer_position=args.bytes_per_layer_position,
            model_id=template.model.id,
            model_revision=template.model.revision,
            dataset_id=args.calibration_dataset,
            dataset_revision=args.calibration_revision,
            calibration_split=args.calibration_split,
            evaluation_sample_ids=evaluation_ids,
            seed=template.execution.seed,
            exclude_protected_from_calibration=arm.sink_neutral,
            prompt_reservation_policy=arm.profile_policy,
            question_score_weight=arm.question_weight,
        )
        for arm in EVIDENCE_ABLATIONS
    }
    # Never overwrite a historical capture/profile/run. A partial failure has no bundle seal.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, profile in profiles.items():
        paths[name] = profile.write(output / "profiles" / (name + ".json"))
    groups = evidence_ablation_configs(controls, paths)
    configs = {}
    for policy, arms in groups.items():
        for name, cfg in arms.items():
            path = output / "configs" / policy / (name + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                json.dump(canonical_config(cfg), handle, indent=2, sort_keys=True)
                handle.write("\n")
            configs[f"{policy}/{name}"] = {
                "path": str(path),
                "config_sha256": config_sha256(cfg),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest = {
        "schema_version": 1,
        "classification": "unexecuted_local_reimpl_ablation_plan",
        "gpu_execution_authorized": False,
        "paper_claim_authorized": False,
        "calibration_sample_ids": sorted(o.sample_id for o in observations),
        "evaluation_sample_ids": evaluation_ids,
        "inputs": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.controls_json, args.attention_jsonl, args.evaluation_sample_ids)
        },
        "profiles": {
            name: {
                "path": str(paths[name]),
                "profile_sha256": p.profile_sha256,
                "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            }
            for name, p in profiles.items()
        },
        "configs": configs,
        "stages": [
            {"stage": 1, "policy": "first_and_last", "compare": ["control", "a_sink"]},
            {
                "stage": 2,
                "policy": "question_and_boundaries",
                "compare": ["b_protect", "ab_sink_protect"],
            },
            {
                "stage": 3,
                "policy": "question_and_boundaries",
                "compare": ["bc_protect_rank", "abc_combined"],
            },
        ],
        "standalone_ranking": ["first_and_last/c_rank", "first_and_last/ac_sink_rank"],
        "warning": "Compare methods only within mandatory-policy groups. Initial byte cap "
        "is not a constant decode-time memory budget; report growth/eviction separately.",
    }
    with (output / "bundle.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-jsonl", type=Path, required=True)
    parser.add_argument("--evaluation-sample-ids", type=Path, required=True)
    parser.add_argument(
        "--controls-json",
        type=Path,
        required=True,
        help="JSON mapping FullKV/Prompt Top-K/MosaicKV exact to RunConfig objects",
    )
    parser.add_argument("--bytes-per-layer-position", type=int, required=True)
    parser.add_argument("--calibration-dataset", required=True)
    parser.add_argument("--calibration-revision", required=True)
    parser.add_argument("--calibration-split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args)
    print(
        json.dumps(
            {
                "status": "prepared_not_executed",
                "configs": len(manifest["configs"]),
                "bundle": str(args.output.resolve() / "bundle.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
