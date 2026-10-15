#!/usr/bin/env python3
"""Prepare write-once native-metric fidelity inputs; never submit a GPU job.

The 32-example cohorts are deterministic development subsets of the already
evaluated official pools, not fresh held-out data. No answers select IDs.
Large execution and shared-byte execution stay blocked pending fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from run_baseline_fidelity_review_worker import byte_capped_configs
from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import DatasetConfig, canonical_config, config_sha256, load_config
from mosaickv.evaluation.paper_native_lanes import load_paper_native_ledger
from mosaickv.evaluation.shared_byte_contract import (
    candidate_content_digest,
    load_contract,
    validate_configuration,
    validate_ledger,
)
from mosaickv.evaluation.tasks import default_task_registry


def configurations(source, task, contract, cap):
    lane = contract["lanes"]["prefixkv"]
    route = lane["tasks"][task]
    configs = byte_capped_configs(
        load_config(source / "mosaickv/configs/paper_candidate_llava_v1.yaml"),
        repository=source,
        family="prefixkv",
        task=task,
        dataset=DatasetConfig(route["dataset_id"], route["dataset_revision"], route["split"]),
        model_id=lane["model_id"],
        model_revision=lane["model_revision"],
        seed=contract["comparison"]["seed"],
        output_length=route["output_tokens"],
        byte_cap=cap * 2**20,
        profile_forget_ratio="0.5",
        profile_reference_retention_ratio=0.5,
        prefixkv_transport="ragged_llama_eager_v1",
        mandatory_policy="first_and_last",
    )
    prefix = configs["prefixkv_reimpl"]
    configs["prefixkv_reimpl"] = replace(
        prefix, prefixkv=replace(prefix.prefixkv, decode_policy="append_only")
    )
    full = configs["full_kv"]
    configs["full_kv"] = replace(
        full,
        cache=replace(
            full.cache,
            physical_budget_block_size=1,
            physical_budget_mandatory_policy="first_and_last",
            packing_mode=configs["prompt_attention_topk"].cache.packing_mode,
        ),
    )
    for cfg in configs.values():
        validate_configuration(contract, "prefixkv", task, cfg)
    return configs


def cohort_ids(ids, seed, task, size=32):
    if len(ids) != len(set(ids)) or size < 1 or len(ids) < size:
        raise ValueError("need enough distinct identities")
    return sorted(ids, key=lambda sid: hashlib.sha256(f"{seed}:{task}:{sid}".encode()).digest())[
        :size
    ]


def prepare(args):
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, args.source_sha)
    contract_path = source / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml"
    contract = load_contract(contract_path)
    if candidate_content_digest(source) != contract["candidate"]["content_sha256"]:
        raise ValueError("candidate content fingerprint changed")
    root = args.root.resolve()
    if (
        root.exists()
        or Path("/scratch/djy8hg").resolve() not in root.parents
        or source in root.parents
    ):
        raise ValueError("require new output root outside repository under scratch")
    lane = contract["lanes"]["prefixkv"]
    # Preflight all input/config contracts before creating any artifact.
    checked = {}
    for task, route in lane["tasks"].items():
        path = Path(route["parent_ledger"])
        ids = validate_ledger(path, route["parent_ledger_sha256"], route, task)
        ledger = load_paper_native_ledger(
            path,
            expected_sha256=route["parent_ledger_sha256"],
            task=default_task_registry().resolve(task),
            dataset_revision=route["dataset_revision"],
        )
        for cap in lane["caps_mib"]:
            configurations(source, task, contract, cap)
        checked[task] = (ledger, cohort_ids(ids, contract["comparison"]["seed"], task))
    plan = dict(
        schema_version=1,
        source_sha=args.source_sha,
        classification="native_metric_development",
        heldout_claim=False,
        full_campaign_authorized=False,
        byte_validation_authorized=False,
        fidelity_execution_authorized=args.authorize_fidelity,
        max_global_a100s=4,
        gpus_per_worker=1,
        cpus_per_worker=12,
        time_limit_seconds=7200,
        caps_mib=lane["caps_mib"],
        fullkv_latency_comparison=False,
        candidate_policy_sha256=contract["candidate"]["policy_sha256"],
        comparison_contract=str(contract_path),
        comparison_contract_sha256=digest(contract_path),
        profile_sha256=digest(
            source / "third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.5.json"
        ),
        fidelity_steps=64,
        fidelity_logit_atol=1e-4,
        fidelity_token_agreement=1.0,
        nll_tolerance="(2*logit_atol+1e-6)*evaluated_token_count",
        corpus_ppl_reducer="exp(sum(nll)/sum(target_tokens))",
        rouge_reference="one immutable FullKV output per shard, shared across caps/methods",
        caps_status="registered; feasibility must pass independently per sample before large run",
        bound_files={str(contract_path): digest(contract_path)},
        jobs=[],
    )
    for task, (ledger, ids) in checked.items():
        route = lane["tasks"][task]
        plan["bound_files"][route["parent_ledger"]] = route["parent_ledger_sha256"]
        samples = {sample.sample_id: sample for sample in ledger.samples}
        for shard, offset in enumerate(range(0, len(ids), 2)):
            chosen = [samples[sid] for sid in ids[offset : offset + 2]]
            payload = dict(
                schema_version=1,
                task=task,
                dataset_id=route["dataset_id"],
                dataset_revision=route["dataset_revision"],
                split=route["split"],
                parent_ledger_sha256=route["parent_ledger_sha256"],
                samples=[
                    dict(
                        sample_id=s.sample_id,
                        prompt=s.prompt,
                        references=list(s.references),
                        teacher_forcing_target=s.teacher_forcing_target,
                        media=[
                            dict(kind=m.kind.value, path=str(m.path), sha256=m.sha256)
                            for m in s.media
                        ],
                    )
                    for s in chosen
                ],
            )
            path = root / "ledgers" / task / f"{shard:02d}.json"
            write_once(path, payload)
            plan["bound_files"][str(path)] = digest(path)
            reference = root / "references" / task / f"{shard:02d}.json"
            for cap in lane["caps_mib"]:
                for method, cfg in configurations(source, task, contract, cap).items():
                    if method == "full_kv" and cap != lane["caps_mib"][0]:
                        continue
                    config_path = root / "configs" / task / f"{method}-{cap}.json"
                    if not config_path.exists():
                        write_once(config_path, canonical_config(cfg))
                        plan["bound_files"][str(config_path)] = digest(config_path)
                    plan["jobs"].append(
                        dict(
                            task=task,
                            shard=shard,
                            method=method,
                            cap_mib=None if method == "full_kv" else cap,
                            config=str(config_path),
                            config_sha=config_sha256(cfg),
                            ledger=str(path),
                            ledger_sha=digest(path),
                            reference=str(reference),
                            sample_ids=[s.sample_id for s in chosen],
                            output=str(root / "outputs" / task / f"{shard:02d}-{method}-{cap}"),
                        )
                    )
    write_once(root / "plan.json", plan)
    print(
        json.dumps(
            dict(plan=str(root / "plan.json"), sha256=digest(root / "plan.json"), submitted=False)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--authorize-fidelity",
        action="store_true",
        help="enable only native-path fidelity, never shared-byte or large runs",
    )
    prepare(parser.parse_args())
