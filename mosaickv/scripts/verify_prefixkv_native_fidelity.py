#!/usr/bin/env python3
"""Native-route differential GPU gate; never launch or authorize large work.

Executes pinned upstream selection/eviction and local implementations on the
same HF checkpoint, inputs, profile, precision and device. This is not a claim
of parity with the different historical BF16 legacy-checkpoint execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import digest, write_once
from verify_hf_adapters import provenance, validate_cache_parity

from mosaickv.config import config_sha256, load_config


def check_inputs(root, plan_sha, index):
    source = Path(__file__).resolve().parents[2]
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("immutable plan hash changed")
    plan = json.loads((root / "plan.json").read_text())
    _require_clean_source(source, plan["source_sha"])
    if (
        not plan["fidelity_execution_authorized"]
        or plan["full_campaign_authorized"]
        or plan["max_global_a100s"] != 4
        or plan["gpus_per_worker"] != 1
        or plan["cpus_per_worker"] != 12
        or plan["time_limit_seconds"] != 7200
    ):
        raise ValueError("unexpected authorization or resources")
    for path, sha in plan["bound_files"].items():
        if digest(Path(path)) != sha:
            raise ValueError(f"immutable input changed: {path}")
    if index < 0 or index >= len(plan["jobs"]):
        raise ValueError("invalid job index")
    job = plan["jobs"][index]
    if job["method"] != "full_kv":
        raise ValueError("select a FullKV reference shard for the native fidelity gate")
    config = load_config(job["config"])
    if config_sha256(config) != job["config_sha"]:
        raise ValueError("resolved config changed")
    return source, plan, job, config


def run(args):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel, _decode_text
    from mosaickv.baselines.prefixkv import build_prefixkv_reimpl_plan
    from mosaickv.cache_state import FullKVState
    from mosaickv.config import CacheConfig, PrefixKVConfig
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.prefixkv_differential import compare_upstream_decode
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.types import BudgetUnit

    source, plan, job, config = check_inputs(args.root, args.plan_sha, args.index)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires exactly one allocated GPU")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("requires registered A100 hardware")
    output = args.root / "fidelity" / job["task"] / f"{job['shard']:02d}.json"
    if output.exists():
        raise FileExistsError(output)
    ledger = load_paper_native_ledger(
        job["ledger"],
        expected_sha256=job["ledger_sha"],
        task=default_task_registry().resolve(job["task"]),
        dataset_revision=config.dataset.revision,
    )
    controls = dict(
        model=asdict(config.model),
        dataset=asdict(config.dataset),
        execution=asdict(config.execution),
        ledger_sha=ledger.sha256,
        generation_steps=plan["fidelity_steps"],
        logit_atol=plan["fidelity_logit_atol"],
        profile_sha256=plan["profile_sha256"],
        source_sha=plan["source_sha"],
    )
    report = dict(
        status="running",
        configuration=controls,
        source=provenance(controls),
        planned_ids=job["sample_ids"],
        plan_sha256=args.plan_sha,
        results=[],
        failures=[],
        claim_scope="same-checkpoint HF upstream algorithm parity; not legacy checkpoint parity",
        legacy_official_reproduction_reused=False,
        full_campaign_authorized=False,
    )
    report["source"]["seed"] = config.execution.seed
    try:
        torch.manual_seed(config.execution.seed)
        torch.use_deterministic_algorithms(True)
        adapter = load_hf_adapter(
            config.model.id,
            revision=config.model.revision,
            model_kwargs=_model_kwargs(config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        policy = PrefixKVConfig(
            enabled=True,
            profile_path=str(source / "third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.5.json"),
            hf_cache_transport="ragged_llama_eager_v1",
            decode_boundary_policy="protected",
            allocation_policy="upstream_round",
            decode_policy="profile_ratio",
        )
        if digest(Path(policy.profile_path)) != plan["profile_sha256"]:
            raise RuntimeError("profile changed")
        runtime = HuggingFaceMosaicKVModel(
            adapter, config, trace_directory=output.parent / "traces"
        )
        for sample in ledger.samples:
            try:
                messages = build_multimodal_messages(
                    sample.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media)
                )
                prepared = adapter.prepare_inputs(messages)
                parity = validate_cache_parity(
                    adapter, prepared, tolerance=plan["fidelity_logit_atol"]
                )
                reference = adapter.greedy_decode(
                    prepared, max_new_tokens=plan["fidelity_steps"] + 1
                )
                answer = _decode_text(adapter.processor, reference.token_ids)
                # Use an ID hash as the filename; dataset IDs are not safe path components.
                reference_file = output.parent / (
                    hashlib.sha256(sample.sample_id.encode()).hexdigest() + "-reference.json"
                )
                write_once(
                    reference_file,
                    dict(
                        sample_id=sample.sample_id,
                        answer=answer,
                        controls=controls,
                        generated_token_ids=reference.token_ids.cpu().tolist(),
                    ),
                )
                del reference
                prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
                full = FullKVState.from_cache_snapshot(
                    adapter.extract_past_key_values(prefill.state.past_key_values),
                    block_size=1,
                    modality_spans=prepared.modality_map,
                    token_ids=prepared.model_inputs["input_ids"],
                    mandatory_logical_positions=(0, prepared.logical_sequence_length - 1),
                )
                compression = build_prefixkv_reimpl_plan(
                    full,
                    prefill.attention_weights,
                    policy,
                    CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.5, 1),
                    model_id=config.model.id,
                    model_revision=config.model.revision,
                )
                targets = runtime._teacher_forced_target_ids(sample.teacher_forcing_target)
                greedy = compare_upstream_decode(
                    adapter,
                    prefill,
                    compression,
                    source,
                    steps=plan["fidelity_steps"],
                    atol=plan["fidelity_logit_atol"],
                    fullkv_reference_answer=answer,
                )
                forced = compare_upstream_decode(
                    adapter,
                    prefill,
                    compression,
                    source,
                    steps=len(targets) - 1,
                    atol=plan["fidelity_logit_atol"],
                    teacher_forced_ids=targets,
                )
                report["results"].append(
                    dict(
                        sample_id=sample.sample_id,
                        retention_one=parity,
                        greedy=greedy,
                        teacher_forced=forced,
                        reference_artifact=str(reference_file),
                        reference_artifact_sha256=digest(reference_file),
                    )
                )
                del full, prefill, compression
            except Exception as error:
                report["failures"].append(
                    dict(sample_id=sample.sample_id, error=f"{type(error).__name__}: {error}")
                )
        report["status"] = "passed" if not report["failures"] else "failed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        write_once(output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--index", type=int, required=True)
    raise SystemExit(run(parser.parse_args()))
