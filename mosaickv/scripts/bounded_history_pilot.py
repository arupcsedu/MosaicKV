#!/usr/bin/env python3
"""Prepare/run/analyze the small, explicitly adapted common-ring comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from mosaickv.config import (
    BoundedDecodeConfig,
    DatasetConfig,
    DraftAllocationGuardConfig,
    DraftExchangeSearchConfig,
    ForecastingConfig,
    GraphConfig,
    LookMConfig,
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
from mosaickv.types import ForecastMode, MosaicKVMethod, RepairPolicy

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "mosaickv/configs/experiments/bounded_history_baselines_v1.yaml"
GUARDED_ARMS = {"mosaickv_output_impact_guarded_v3", "mosaickv_direct_error_v4"}
DRAFT_GUARDED_ARM = "mosaickv_draft_logit_guard_v5"
SINGLE_EXCHANGE_ARM = "mosaickv_draft_single_exchange_v6"
DRAFT_GUARDED_ARMS = {DRAFT_GUARDED_ARM, SINGLE_EXCHANGE_ARM}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(event, **details):
    print(json.dumps(dict(event=event, **details), allow_nan=False), flush=True)


def write_once(path, payload):
    # Validate/normalize before opening: a serialization failure must not leave
    # an incomplete write-once artifact that cannot be safely overwritten.
    def scalar(value):
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")

    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=scalar)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def clean_source(expected_sha=None):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    subprocess.run(["bash", "mosaickv/scripts/assert_clean_worktree.sh"], cwd=SOURCE, check=True)
    if expected_sha is not None and sha != expected_sha:
        raise ValueError("source SHA mismatch")
    return sha


def configurations(task, cap, route, profile):
    candidate = load_config(SOURCE / "mosaickv/configs/development/llava_bounded_history_v1.yaml")
    candidate = replace(
        candidate,
        dataset=DatasetConfig(route["dataset_id"], route["dataset_revision"], route["split"]),
        cache=replace(candidate.cache, budget_value=cap * 2**20),
    )
    disabled = dict(
        forecasting=ForecastingConfig(
            enabled=False, mode=ForecastMode.PROMPT_WINDOW, prompt_window=16, draft_steps=0
        ),
        graph=GraphConfig(enabled=False),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE, max_blocks_per_step=0),
    )
    baseline = replace(candidate, method=MosaicKVMethod.PROMPT_ATTENTION_TOPK, **disabled)
    return {
        "mosaickv_exact": candidate,
        "prompt_attention_topk": baseline,
        "full_kv": replace(
            baseline,
            method=MosaicKVMethod.FULL_KV,
            bounded_decode=None,
            cache=replace(baseline.cache, budget_value=2**63 - 1, retention_ratio=1.0),
        ),
        "prefixkv_reimpl": replace(
            baseline,
            method=MosaicKVMethod.PREFIXKV_REIMPL,
            prefixkv=PrefixKVConfig(
                enabled=True,
                profile_path=str(profile),
                rescale_offline_profile_to_byte_budget=True,
                profile_reference_retention_ratio=0.5,
                hf_cache_transport="ragged_llama_eager_v1",
                decode_boundary_policy="protected",
                allocation_policy="global_apportion",
                decode_policy="append_only",
            ),
        ),
        "lookm_reimpl": replace(
            baseline,
            method=MosaicKVMethod.LOOKM_REIMPL,
            lookm=LookMConfig(
                enabled=True,
                recent_ratio=math.nextafter(1.0, 0.0) / 2,
                important_ratio=math.nextafter(1.0, 0.0) / 2,
            ),
        ),
        "vl_cache_reimpl": replace(
            baseline, method=MosaicKVMethod.VL_CACHE_REIMPL, vl_cache=VLCacheConfig(enabled=True)
        ),
    }


def development_exclusions(protocol, task, chosen_ids):
    """Bind prior identity ledgers, not predictions, scores or failure identities."""
    records = protocol.get("development_exclusions", {}).get(task, [])
    if protocol.get("candidate_arm") in GUARDED_ARMS | DRAFT_GUARDED_ARMS and not records:
        raise ValueError("guarded development validation requires explicit exclusion ledgers")
    if len(chosen_ids) != len(set(chosen_ids)):
        raise ValueError("development cohort contains duplicate identities")
    bound, excluded = {}, set()
    for record in records:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("development exclusion ledger hash changed")
        payload = json.loads(path.read_text())
        if payload["task"] != task:
            raise ValueError("development exclusion task mismatch")
        excluded.update(str(sample["sample_id"]) for sample in payload["samples"])
        bound[str(path)] = record["sha256"]
    if set(chosen_ids) & excluded:
        raise ValueError("development cohort overlaps a consumed candidate-tuning ledger")
    return bound, dict(
        task=task,
        selected_count=len(chosen_ids),
        excluded_count=len(excluded),
        overlap_count=0,
        selection_uses_answers_or_scores=False,
        scope=(
            "disjoint_from_explicit_v1_v2_v3_v4_v5_development_not_globally_untouched"
            if protocol.get("candidate_arm") == SINGLE_EXCHANGE_ARM
            else "disjoint_from_explicit_v1_v2_v3_v4_development_not_globally_untouched"
            if protocol.get("candidate_arm") == DRAFT_GUARDED_ARM
            else "disjoint_from_explicit_v1_v2_v3_development_ledgers_not_globally_untouched"
            if protocol.get("candidate_arm") == "mosaickv_direct_error_v4"
            else "disjoint_from_explicit_v1_v2_development_ledgers_not_globally_untouched"
        ),
    )


def prepare(root, protocol_path=PROTOCOL):
    from mosaickv.evaluation.paper_native_lanes import load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry

    sha = clean_source()
    root = root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("require a new immutable output root under /scratch/djy8hg")
    protocol_path = Path(protocol_path).resolve()
    if SOURCE.resolve() not in protocol_path.parents:
        raise ValueError("protocol must be versioned inside the execution worktree")
    protocol = yaml.safe_load(protocol_path.read_text())
    routes = yaml.safe_load(
        (SOURCE / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml").read_text()
    )["lanes"]["prefixkv"]["tasks"]
    profile = SOURCE / "third_party/PrefixKV" / protocol["prefixkv_profile"]
    upstream = subprocess.check_output(
        ["git", "-C", str(profile.parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream != protocol["prefixkv_commit"]:
        raise ValueError("PrefixKV upstream revision mismatch")
    bound = {str(protocol_path): digest(protocol_path), str(profile): digest(profile)}
    for record in protocol.get("prior_negative_evidence", []):
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("prior negative-result seal changed")
        bound[record["path"]] = record["sha256"]
    feasibility_summary = None
    if protocol.get("candidate_arm") in {"mosaickv_direct_error_v4"} | DRAFT_GUARDED_ARMS:
        from mosaickv.selection.direct_error_exchange import (
            cpu_feasibility,
            single_exchange_cpu_feasibility,
        )

        check = (
            single_exchange_cpu_feasibility
            if protocol.get("candidate_arm") == SINGLE_EXCHANGE_ARM
            else cpu_feasibility
        )
        feasibility = dict(source_sha=sha, **check())
        if not feasibility["passed"]:
            raise ValueError("CPU exchange feasibility failed; materialization/submission blocked")
        feasibility_path = root / "cpu-feasibility.json"
        write_once(feasibility_path, feasibility)
        bound[str(feasibility_path)] = digest(feasibility_path)
        feasibility_summary = dict(
            path=str(feasibility_path),
            sha256=digest(feasibility_path),
            source_sha=sha,
            passed=feasibility["passed"],
            classification=feasibility["classification"],
            **(
                {k: feasibility[k] for k in ("policy", "exact_bytes_and_mandatory")}
                if protocol.get("candidate_arm") == SINGLE_EXCHANGE_ARM
                else {}
            ),
        )
    jobs, disjointness = [], []
    for task in protocol["tasks"]:
        route = routes[task]
        parent = Path(route["parent_ledger"])
        if digest(parent) != route["parent_ledger_sha256"]:
            raise ValueError("reviewed parent ledger changed")
        # Bind all parent metadata, but verify media only for requested IDs.
        original = json.loads(parent.read_text())
        if any(original[key] != route[key] for key in ("dataset_id", "dataset_revision", "split")):
            raise ValueError("parent dataset provenance mismatch")
        offset = protocol.get("sample_offset", 0)
        chosen = original["samples"][offset : offset + protocol["samples_per_task"]]
        if len(chosen) != protocol["samples_per_task"]:
            raise ValueError("parent ledger lacks the complete preregistered sample cohort")
        excluded_files, identity_record = development_exclusions(
            protocol, task, [str(sample["sample_id"]) for sample in chosen]
        )
        bound.update(excluded_files)
        disjointness.append(identity_record)
        for sample in chosen:
            for media in sample["media"]:
                if not Path(media["path"]).is_absolute():
                    media["path"] = str((parent.parent / media["path"]).resolve())
        payload = dict(
            schema_version=1,
            task=task,
            dataset_id=original["dataset_id"],
            dataset_revision=original["dataset_revision"],
            split=original["split"],
            parent_ledger_sha256=digest(parent),
            samples=chosen,
        )
        derived = root / "ledgers" / f"{task}.json"
        write_once(derived, payload)
        ledger = load_paper_native_ledger(
            derived,
            expected_sha256=digest(derived),
            task=default_task_registry().resolve(task),
            dataset_revision=route["dataset_revision"],
        )
        bound[str(parent)] = digest(parent)
        bound[str(derived)] = digest(derived)
        reference = root / "references" / f"{task}.json"
        for cap in protocol["caps_mib"]:
            configs = configurations(task, cap, route, profile)
            if "history_fraction" in protocol:
                configs = {
                    name: replace(
                        cfg,
                        bounded_decode=(
                            None
                            if name == "full_kv"
                            else BoundedDecodeConfig(
                                protocol["generated_capacity"],
                                protocol["sink_tokens"],
                                protocol["history_fraction"],
                            )
                        ),
                    )
                    for name, cfg in configs.items()
                }
                candidate = configs["mosaickv_exact"]
                configs["mosaickv_output_impact_v2"] = replace(
                    candidate,
                    bounded_decode=replace(
                        candidate.bounded_decode, layer_allocation="output_impact_v1"
                    ),
                )
                configs["mosaickv_output_impact_guarded_v3"] = replace(
                    candidate,
                    bounded_decode=replace(
                        candidate.bounded_decode, layer_allocation="output_impact_guarded_v1"
                    ),
                )
                configs["mosaickv_direct_error_v4"] = replace(
                    candidate,
                    bounded_decode=replace(
                        candidate.bounded_decode, layer_allocation="direct_error_exchange_v1"
                    ),
                )
                if protocol.get("candidate_arm") in DRAFT_GUARDED_ARMS:
                    single = protocol["candidate_arm"] == SINGLE_EXCHANGE_ARM
                    configs[protocol["candidate_arm"]] = replace(
                        candidate,
                        bounded_decode=replace(
                            candidate.bounded_decode,
                            layer_allocation="draft_single_exchange_v1"
                            if single
                            else "draft_logit_guard_v1",
                            draft_guard=DraftAllocationGuardConfig.from_mapping(
                                protocol["draft_guard"]
                            ),
                            draft_exchange=DraftExchangeSearchConfig.from_mapping(
                                protocol["draft_exchange"]
                            )
                            if single
                            else None,
                        ),
                    )
            methods = (
                ["full_kv", *protocol["methods"]]
                if cap == protocol["caps_mib"][0]
                else protocol["methods"]
            )
            for method in methods:
                index = len(jobs)
                cfg = configs[method]
                path = root / "configs" / f"{index:02d}.json"
                write_once(path, canonical_config(cfg))
                bound[str(path)] = digest(path)
                jobs.append(
                    dict(
                        index=index,
                        task=task,
                        method=method,
                        runtime_method=cfg.method.value,
                        cap_mib=None if method == "full_kv" else cap,
                        config=str(path),
                        config_sha=config_sha256(cfg),
                        ledger=str(derived),
                        ledger_sha=digest(derived),
                        sample_ids=[s.sample_id for s in ledger.samples],
                        output=str(root / "outputs" / f"{index:02d}"),
                        reference=str(reference),
                    )
                )
    plan = dict(
        schema_version=1,
        classification=protocol["classification"],
        source_sha=sha,
        source_root=str(SOURCE),
        protocol=protocol,
        bound_files=bound,
        jobs=jobs,
        heldout_claim=False,
        paper_campaign_authorized=False,
        development_disjointness=disjointness,
        **({"cpu_feasibility": feasibility_summary} if feasibility_summary is not None else {}),
        comparison_decode_policy=dict(
            name=(
                "common_bounded_history_v2_fullkv_reference_exempt"
                if "history_fraction" in protocol
                else "common_bounded_history_v1_fullkv_reference_exempt"
            ),
            generated_capacity=protocol["generated_capacity"],
            sink_tokens=protocol["sink_tokens"],
            **(
                {"history_fraction": protocol["history_fraction"]}
                if "history_fraction" in protocol
                else {}
            ),
        ),
    )
    write_once(root / "plan.json", plan)
    print(
        json.dumps(
            dict(
                root=str(root),
                plan_sha256=digest(root / "plan.json"),
                jobs=len(jobs),
                source_sha=sha,
            )
        )
    )


def inputs(root, plan_sha, index=None, check_clean=True):
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("plan hash mismatch")
    plan = json.loads((root / "plan.json").read_text())
    if check_clean:
        clean_source(plan["source_sha"])
    if Path(plan["source_root"]).resolve() != SOURCE.resolve():
        raise ValueError("wrong source worktree")
    for path, sha in plan["bound_files"].items():
        if digest(path) != sha:
            raise ValueError(f"bound input changed: {path}")
    if index is None:
        return plan
    if type(index) is not int or not 0 <= index < len(plan["jobs"]):
        raise ValueError("unregistered worker index")
    job = plan["jobs"][index]
    cfg = load_config(job["config"])
    if config_sha256(cfg) != job["config_sha"] or cfg.method.value != job.get(
        "runtime_method", job["method"]
    ):
        raise ValueError("resolved configuration mismatch")
    return plan, job, cfg


def prepare_retry(root, parent, parent_sha):
    """Reuse hash-verified successes; execute ONLY the four missing V1 arms."""
    source_sha = clean_source()
    original = inputs(parent, parent_sha, check_clean=False)
    if root.exists() or root.resolve() == parent.resolve():
        raise ValueError("retry requires a new immutable output directory")
    plan = copy.deepcopy(original)
    plan.update(source_sha=source_sha, source_root=str(SOURCE))
    plan["reuse_parent"] = dict(
        root=str(parent), plan_sha256=parent_sha, source_sha=original["source_sha"]
    )
    plan["bound_files"][str(parent / "plan.json")] = parent_sha
    plan["protocol"]["max_global_a100s"] = 6
    records, missing = {}, []
    for job in plan["jobs"]:
        index = job["index"]
        old_path = parent / "workers" / f"{index:02d}.json"
        if old_path.is_file():
            result = json.loads(old_path.read_text())
            if result["plan_sha256"] != parent_sha or result["index"] != index:
                raise ValueError("parent worker identity mismatch")
            for path, sha in result["artifacts"].items():
                if digest(path) != sha:
                    raise ValueError("parent result changed; cannot reuse")
            result["reused_from"] = dict(
                path=str(old_path),
                sha256=digest(old_path),
                plan_sha256=parent_sha,
                source_sha=result["source_sha"],
            )
            plan["bound_files"][str(old_path)] = digest(old_path)
            records[index] = result
        else:
            if job["method"] != "vl_cache_reimpl":
                raise ValueError("selective retry scope permits failed VL-Cache arms only")
            missing.append(index)
            job["output"] = str(root.resolve() / "outputs" / f"{index:02d}")
    if missing != [4, 9, 15, 20]:
        raise ValueError("retry scope must be exactly the four original VL-Cache workers")
    plan["retry_indices"] = missing
    write_once(root / "plan.json", plan)
    sha = digest(root / "plan.json")
    for index, result in records.items():
        result["plan_sha256"] = sha
        write_once(root / "workers" / f"{index:02d}.json", result)
    print(
        json.dumps(
            dict(
                root=str(root),
                plan_sha256=sha,
                retry_indices=missing,
                reused_workers=len(records),
                source_sha=source_sha,
            )
        )
    )


def reconstruction_trace_checks(trace_root, sample_ids, cap_bytes):
    from mosaickv.selection.evidence_layer_allocation import validate_reconstruction_trace

    phases = {}
    for path in Path(trace_root).rglob("*.json"):
        trace = json.loads(path.read_text())
        allocation = trace["evidence_layer_allocation"]
        if allocation["padding_bytes"] or allocation["benchmark_labels_accessed"]:
            raise RuntimeError("guarded allocation violates padding/label contract")
        budget = trace["bounded_decode"]["budget_reservation"]
        if allocation["prompt_bytes"] + budget["reserved_generated_bytes"] != cap_bytes:
            raise RuntimeError("guarded allocation prompt/history accounting mismatch")
        check = validate_reconstruction_trace(allocation)
        if "exchange_search" in allocation["reconstruction_guard"]:
            from mosaickv.selection.direct_error_exchange import validate_exchange_trace

            check = validate_exchange_trace(allocation)
        phase = "teacher_forced" if trace["teacher_forced_evaluation"]["enabled"] else "generation"
        key = (trace["sample_id"], phase)
        if key in phases:
            raise RuntimeError("duplicate reconstruction phase trace")
        phases[key] = (check, allocation["selected_positions_sha256"], allocation["prompt_bytes"])
    expected = {
        (sample, phase) for sample in sample_ids for phase in ("generation", "teacher_forced")
    }
    if set(phases) != expected:
        raise RuntimeError("missing reconstruction sample/phase trace")
    checks = {}
    for sample in sample_ids:
        generated, generation_hash, generation_bytes = phases[sample, "generation"]
        forced, forced_hash, forced_bytes = phases[sample, "teacher_forced"]
        same_budget = generation_bytes == forced_bytes
        same_selection = generation_hash == forced_hash
        if same_budget and (not same_selection or generated["accepted"] != forced["accepted"]):
            raise RuntimeError("selection must not change when teacher-forcing labels are attached")
        if forced_bytes < generation_bytes:
            raise RuntimeError("shorter teacher-forcing run cannot increase its history reserve")
        checks[sample] = dict(
            **generated,
            teacher_forced_accepted=forced["accepted"],
            teacher_forced_budget_identical=same_budget,
            teacher_forced_selection_identical=same_selection,
            generation_prompt_bytes=generation_bytes,
            teacher_forced_prompt_bytes=forced_bytes,
            validated_trace_count=2,
        )
    return checks


def draft_allocation_trace_checks(trace_root, sample_ids, cap_bytes):
    from mosaickv.backends.hf_allocation_guard import validate_draft_guard_trace
    from mosaickv.backends.hf_single_exchange_guard import validate_single_exchange_trace
    from mosaickv.selection.direct_error_exchange import validate_exchange_trace

    phases = {}
    for path in Path(trace_root).rglob("*.json"):
        trace = json.loads(path.read_text())
        proposal = trace["evidence_layer_allocation"]
        guard = trace["draft_logit_allocation_guard"]
        if guard["policy"] == "bounded_single_equal_byte_draft_logit_v6":
            if proposal.get("role") != "shortlist_only_final_draft_guard_is_authoritative":
                raise RuntimeError("V6 shortlist must not masquerade as final selection")
            check = validate_single_exchange_trace(guard)
        else:
            validate_exchange_trace(proposal)
            if proposal.get("role") != "proposal_only_before_draft_guard_not_final_selection":
                raise RuntimeError("V5 proposal must not masquerade as its final selection")
            check = validate_draft_guard_trace(guard)
        if guard["active_cap_bytes"] != cap_bytes:
            raise RuntimeError("draft guard cap changed")
        phase = "teacher_forced" if trace["teacher_forced_evaluation"]["enabled"] else "generation"
        key = trace["sample_id"], phase
        if key in phases:
            raise RuntimeError("duplicate draft guard phase trace")
        phases[key] = (check, guard["final_positions_sha256"], guard["prompt_bytes"])
    expected = {(s, p) for s in sample_ids for p in ("generation", "teacher_forced")}
    if set(phases) != expected:
        raise RuntimeError("missing draft guard sample/phase trace")
    result = {}
    for sample in sample_ids:
        generated, gh, gb = phases[sample, "generation"]
        forced, fh, fb = phases[sample, "teacher_forced"]
        if gb == fb and gh != fh:
            raise RuntimeError("draft selection changed when target labels were attached")
        result[sample] = dict(
            **generated,
            teacher_forced_accepted=forced["accepted"],
            target_tokens_not_used_by_guard=True,
            same_prompt_budget=gb == fb,
            same_selection=gh == fh,
            validated_trace_count=2,
        )
    return result


def worker(root, plan_sha, index):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import record_hf_run, settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan, job, cfg = inputs(root, plan_sha, index)
    progress(
        "validated_inputs",
        index=index,
        task=job["task"],
        method=job["method"],
        cap_mib=job["cap_mib"],
        source_sha=plan["source_sha"],
    )
    completed = root / "workers" / f"{index:02d}.json"
    if completed.exists():
        record = json.loads(completed.read_text())
        execution_sha = plan["source_sha"]
        if record.get("reused_from") is not None:
            reuse = record["reused_from"]
            if (
                plan.get("reuse_parent", {}).get("plan_sha256") != reuse["plan_sha256"]
                or digest(reuse["path"]) != reuse["sha256"]
            ):
                raise ValueError("reused worker provenance changed")
            execution_sha = reuse["source_sha"]
        if (record.get("source_sha"), record.get("plan_sha256"), record.get("index")) != (
            execution_sha,
            plan_sha,
            index,
        ):
            raise ValueError("completed worker belongs to a different source/plan/index")
        for path, sha in record["artifacts"].items():
            if digest(path) != sha:
                raise ValueError("completed artifact changed")
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before Python")
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("one allocated A100 required")
    started = time.monotonic()
    attempt = root / "attempts" / f"{index:02d}-{time.time_ns()}"
    try:
        reference = Path(job["reference"])
        is_full = job["method"] == "full_kv"
        if not is_full and not reference.exists():
            raise RuntimeError("FullKV reference not completed; compressed inference is blocked")
        torch.set_num_threads(12)
        torch.manual_seed(cfg.execution.seed)
        torch.use_deterministic_algorithms(True)
        ledger = load_paper_native_ledger(
            job["ledger"],
            expected_sha256=job["ledger_sha"],
            task=default_task_registry().resolve(job["task"]),
            dataset_revision=cfg.dataset.revision,
        )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        progress("model_loaded", index=index, elapsed_seconds=time.monotonic() - started)
        gpu_before = asdict(capture_gpu_environment(torch))
        first = ledger.samples[0]
        messages = build_multimodal_messages(
            first.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in first.media)
        )
        prepared = adapter.prepare_inputs(messages)
        parity = {}
        for name, check in [
            ("generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ]:
            result = check(adapter, prepared, max_new_tokens=16)
            parity[name] = asdict(result)
            if result.token_agreement != 1.0 or result.maximum_logit_difference > 0.001:
                raise RuntimeError(f"FullKV parity failed: {name}")
        write_once(attempt / "parity.json", parity)
        progress("fullkv_parity_passed", index=index)
        runtime = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=attempt / "warmup-traces")
        for repeat in range(plan["protocol"]["warmups"]):
            settle(adapter)
            result = runtime.generate(
                EvaluationRequest(f"warmup-{repeat}", first.sample_id, job["task"], messages, {})
            )
            write_once(attempt / f"warmup-{repeat}.json", dict(metrics=asdict(result.metrics)))
            progress(
                "warmup_completed",
                index=index,
                repetition=repeat,
                active_kv_bytes=result.metrics.active_kv_bytes,
            )
        del runtime
        settle(adapter)
        record = record_hf_run(
            cfg,
            adapter,
            task=job["task"],
            run_id=f"bounded-{index:02d}",
            output_root=Path(job["output"]),
            ledger_path=Path(job["ledger"]),
            ledger_sha=job["ledger_sha"],
            sample_ids=tuple(job["sample_ids"]),
            phase="bounded_history_development_pilot",
            repetition=0,
            order=index,
            export_metrics=True,
            materialized_prefixkv=True,
            fullkv_reference_output=reference if is_full else None,
            fullkv_reference_input=None if is_full else reference,
            fullkv_reference_sha256=None if is_full else digest(reference),
            comparison_decode_policy=plan["comparison_decode_policy"],
        )
        for row in record["rows"]:
            if row.get("error") or row.get("status") != "completed":
                raise RuntimeError("sample failed; raw failure output retained")
            if row["generated_tokens"] != cfg.generation.max_new_tokens:
                raise RuntimeError("incomplete output token trajectory")
            for key in (
                "prefill_time",
                "compression_time",
                "decode_time",
                "ttft",
                "end_to_end_time",
            ):
                if row.get(key) is None or not math.isfinite(row[key]) or row[key] < 0:
                    raise RuntimeError(f"missing/non-finite measured timing: {key}")
            if not is_full:
                for boundary in ("initial", "final", "peak"):
                    if row[f"{boundary}_active_kv_storage_bytes"] != cfg.cache.budget_value:
                        raise RuntimeError("worker failed exact lifetime-storage cap")
        reconstruction_checks = {}
        draft_guard_checks = {}
        from mosaickv.evaluation.cache_phase_audit import audit_phase_directory

        phase_storage_checks = audit_phase_directory(
            Path(job["output"]) / "traces",
            job["sample_ids"],
            None if is_full else cfg.cache.budget_value,
        )
        if job["method"] in GUARDED_ARMS:
            reconstruction_checks = reconstruction_trace_checks(
                Path(job["output"]) / "traces", job["sample_ids"], cfg.cache.budget_value
            )
        if job["method"] in DRAFT_GUARDED_ARMS:
            draft_guard_checks = draft_allocation_trace_checks(
                Path(job["output"]) / "traces", job["sample_ids"], cfg.cache.budget_value
            )
        artifacts = {str(p): digest(p) for p in Path(job["output"]).rglob("*") if p.is_file()}
        artifacts[str(attempt / "parity.json")] = digest(attempt / "parity.json")
        artifacts[str(reference)] = digest(reference)
        write_once(
            completed,
            dict(
                status="passed",
                source_sha=plan["source_sha"],
                plan_sha256=plan_sha,
                index=index,
                task=job["task"],
                method=job["method"],
                cap_mib=job["cap_mib"],
                record=record,
                artifacts=artifacts,
                gpu_before=gpu_before,
                gpu_after=asdict(capture_gpu_environment(torch)),
                wall_seconds=time.monotonic() - started,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                reconstruction_checks=reconstruction_checks,
                phase_storage_checks=phase_storage_checks,
                **({"draft_guard_checks": draft_guard_checks} if draft_guard_checks else {}),
            ),
        )
        progress(
            "worker_completed",
            index=index,
            rows=len(record["rows"]),
            elapsed_seconds=time.monotonic() - started,
        )
    except Exception as error:
        progress("worker_failed", index=index, error=f"{type(error).__name__}: {error}")
        write_once(
            attempt / "failure.json",
            dict(
                status="failed",
                index=index,
                plan_sha256=plan_sha,
                error=f"{type(error).__name__}: {error}",
                sample_ids=job["sample_ids"],
                wall_seconds=time.monotonic() - started,
            ),
        )
        raise


def v5_preliminary_screen(summaries, tasks, candidate_arm=DRAFT_GUARDED_ARM):
    checks = []
    for task in tasks:
        candidate = [r for r in summaries if r["task"] == task and r["method"] == candidate_arm]
        baseline = [
            r for r in summaries if r["task"] == task and r["method"] == "prompt_attention_topk"
        ]
        if len(candidate) != 1 or len(baseline) != 1:
            checks.append(dict(task=task, passed=False, reason="incomplete_or_duplicate_arms"))
            continue
        a, b = candidate[0], baseline[0]
        passed = (
            a["samples"] == b["samples"] == 8
            and a["cap_mib"] == b["cap_mib"] == 40
            and a["rouge_l_f1"] >= b["rouge_l_f1"]
            and a["corpus_ppl"] <= b["corpus_ppl"]
        )
        if candidate_arm == SINGLE_EXCHANGE_ARM:
            anchor = [r for r in summaries if r["task"] == task and r["method"] == "mosaickv_exact"]
            passed = (
                passed
                and len(anchor) == 1
                and a.get("draft_guard_acceptance_fraction", 0) > 0
                and a["rouge_l_f1"] >= anchor[0]["rouge_l_f1"]
                and a["corpus_ppl"] <= anchor[0]["corpus_ppl"]
            )
        checks.append(
            dict(
                task=task,
                passed=bool(passed),
                rouge_delta=a["rouge_l_f1"] - b["rouge_l_f1"],
                ppl_delta=a["corpus_ppl"] - b["corpus_ppl"],
            )
        )
    return dict(
        per_task=checks,
        preliminary_point_estimate_screen_passed=all(r["passed"] for r in checks),
        frozen_candidate_eligible=False,
        paper_campaign_authorized=False,
        reason="eight_examples_per_task_are_not_consistent_independent_quality_evidence",
    )


def analyze(root, plan_sha, *, analysis_root=None):
    import numpy as np
    import pandas as pd

    plan = (
        inputs(root, plan_sha)
        if analysis_root is None
        else inputs(root, plan_sha, check_clean=False)
    )
    destination = root / "analysis" if analysis_root is None else Path(analysis_root).resolve()
    if analysis_root is not None and (
        Path("/scratch/djy8hg").resolve() not in destination.parents
        or destination == root.resolve()
        or root.resolve() in destination.parents
    ):
        raise ValueError("derived analysis requires a separate new scratch directory")
    if destination.exists():
        raise FileExistsError("analysis is write-once")
    rows, missing, failures, failure_records = [], [], [], []
    for job in plan["jobs"]:
        path = root / "workers" / f"{job['index']:02d}.json"
        if not path.exists():
            missing.append(job["index"])
            attempts = sorted((root / "attempts").glob(f"{job['index']:02d}-*/failure.json"))
            error = json.loads(attempts[-1].read_text())["error"] if attempts else "worker_missing"
            failure_records.extend(
                dict(
                    index=job["index"],
                    task=job["task"],
                    method=job["method"],
                    cap_mib=job["cap_mib"],
                    sample_id=sample_id,
                    status="failed_or_not_run",
                    error=error,
                    failure_stage="worker_or_warmup",
                    metric_value=None,
                )
                for sample_id in job.get("sample_ids", [])
            )
            continue
        result = json.loads(path.read_text())
        if result.get("plan_sha256") != plan_sha or result.get("index") != job["index"]:
            raise ValueError("worker result provenance mismatch")
        for name, sha in result["artifacts"].items():
            if digest(name) != sha:
                raise ValueError("result artifact changed")
        observations = pd.read_parquet(result["record"]["metric_observations_parquet_output"])
        for row in result["record"]["rows"]:
            sample = observations[
                (observations.sample_id == row["sample_id"])
                & (observations.metric_scope == "sample")
            ]
            ppl = sample[sample.metric_name == "teacher_forced_ppl"]
            rouge = sample[sample.metric_name == "rouge_l_f1"]
            if len(ppl) != 1 or len(rouge) != 1:
                raise ValueError("duplicate/missing native metric observation")
            row.update(
                task=job["task"],
                method=job["method"],
                cap_mib=job["cap_mib"],
                nll_sum=float(ppl.iloc[0].numerator),
                tokens=int(ppl.iloc[0].denominator),
                rouge_l_f1=float(rouge.iloc[0].metric_value),
                effective_method=str(rouge.iloc[0].method),
                implementation_origin=(
                    "reimpl_adaptation" if job["method"].endswith("_reimpl") else "local_variant"
                ),
            )
            check = result.get("reconstruction_checks", {}).get(row["sample_id"])
            if check is not None:
                row.update({f"reconstruction_{key}": value for key, value in check.items()})
            if job["method"] in GUARDED_ARMS and check is None:
                failures.append(
                    f"missing reconstruction telemetry: {job['index']}/{row['sample_id']}"
                )
            rows.append(row)
            if job["method"] in DRAFT_GUARDED_ARMS:
                check = result.get("draft_guard_checks", {}).get(row["sample_id"])
                if check is None:
                    failures.append(
                        f"missing draft guard telemetry: {job['index']}/{row['sample_id']}"
                    )
                else:
                    row.update({f"draft_guard_{key}": value for key, value in check.items()})
    frame = pd.DataFrame(rows)
    if not rows:
        write_once(
            destination / "report.json",
            dict(
                status="failed_no_complete_workers",
                missing_workers=missing,
                plan_sha256=plan_sha,
                paper_eligible=False,
            ),
        )
        return
    summaries, comparisons = [], []
    for (task, method, cap), group in frame.groupby(["task", "method", "cap_mib"], dropna=False):
        expected = plan["protocol"]["samples_per_task"]
        if len(group) != expected or group.sample_id.nunique() != expected:
            failures.append(f"incomplete {task}/{method}/{cap}")
        if method != "full_kv":
            for boundary in ["initial", "final", "peak"]:
                column = f"{boundary}_active_kv_storage_bytes"
                if not (group[column] == int(cap) * 2**20).all():
                    failures.append(f"byte mismatch {task}/{method}/{cap}/{boundary}")
        summaries.append(
            dict(
                task=task,
                method=method,
                cap_mib=None if pd.isna(cap) else int(cap),
                samples=len(group),
                corpus_ppl=math.exp(group.nll_sum.sum() / group.tokens.sum()),
                rouge_l_f1=group.rouge_l_f1.mean(),
                initial_mib=group.initial_active_kv_storage_bytes.mean() / 2**20,
                final_mib=group.final_active_kv_storage_bytes.mean() / 2**20,
                peak_mib=group.peak_active_kv_storage_bytes.mean() / 2**20,
                e2e_seconds=None if method == "full_kv" else group.end_to_end_time.mean(),
                **(
                    dict(
                        reconstruction_acceptance_fraction=group.reconstruction_accepted.mean(),
                        reconstruction_anchor_mse_mean=group.reconstruction_anchor_mse_mean.mean(),
                        reconstruction_proposed_mse_mean=group.reconstruction_proposed_mse_mean.mean(),
                        reconstruction_final_mse_mean=group.reconstruction_final_mse_mean.mean(),
                        reconstruction_measurement_seconds=group.reconstruction_measurement_seconds.mean(),
                    )
                    if method in GUARDED_ARMS
                    else {}
                ),
                **(
                    dict(
                        draft_guard_acceptance_fraction=group.draft_guard_accepted.mean(),
                        draft_guard_probe_updates_mean=group.draft_guard_probe_updates.mean(),
                        draft_guard_wall_seconds_mean=group.draft_guard_wall_seconds.mean(),
                    )
                    if method in DRAFT_GUARDED_ARMS
                    else {}
                ),
                **(
                    dict(
                        changed_cache_fraction=group.reconstruction_exchange_changed.mean(),
                        committed_exchanges_mean=group.reconstruction_exchange_committed.mean(),
                        initial_safe_donors_mean=group.reconstruction_exchange_initial_donors.mean(),
                    )
                    if method == "mosaickv_direct_error_v4"
                    else {}
                ),
            )
        )
    rng = np.random.default_rng(plan["protocol"]["seed"])
    for task in plan["protocol"]["tasks"]:
        for cap in plan["protocol"]["caps_mib"]:
            candidate_name = plan["protocol"].get("candidate_arm", "mosaickv_exact")
            candidate = (
                frame[
                    (frame.task == task) & (frame.cap_mib == cap) & (frame.method == candidate_name)
                ]
                .set_index("sample_id")
                .sort_index()
            )
            for method in plan["protocol"]["methods"]:
                if method == candidate_name:
                    continue
                baseline = (
                    frame[(frame.task == task) & (frame.cap_mib == cap) & (frame.method == method)]
                    .set_index("sample_id")
                    .sort_index()
                )
                if len(candidate) != plan["protocol"][
                    "samples_per_task"
                ] or not candidate.index.equals(baseline.index):
                    continue
                boundaries = [f"{b}_active_kv_storage_bytes" for b in ("initial", "final", "peak")]
                if any(
                    not np.array_equal(candidate[b].to_numpy(), baseline[b].to_numpy())
                    or not (candidate[b] == cap * 2**20).all()
                    for b in boundaries
                ):
                    failures.append(f"excluded unequal realized budget: {task}/{cap}/{method}")
                    continue
                draws = rng.integers(
                    0,
                    len(candidate),
                    size=(plan["protocol"]["bootstrap_replicates"], len(candidate)),
                )

                def ppl(g, draws=draws):
                    return np.exp(
                        g.nll_sum.to_numpy()[draws].sum(1) / g.tokens.to_numpy()[draws].sum(1)
                    )

                quality = candidate.rouge_l_f1.to_numpy()[draws].mean(
                    1
                ) - baseline.rouge_l_f1.to_numpy()[draws].mean(1)
                ppl_delta = ppl(candidate) - ppl(baseline)
                latency = candidate.end_to_end_time.to_numpy()[draws].mean(
                    1
                ) / baseline.end_to_end_time.to_numpy()[draws].mean(1)
                comparisons.append(
                    dict(
                        task=task,
                        cap_mib=cap,
                        baseline=method,
                        candidate=candidate_name,
                        rouge_delta_ci95=np.quantile(quality, [0.025, 0.975]).tolist(),
                        ppl_delta_ci95=np.quantile(ppl_delta, [0.025, 0.975]).tolist(),
                        e2e_ratio_ci95=np.quantile(latency, [0.025, 0.975]).tolist(),
                        interpretation="same_realized_budget_development_only_unadjusted_intervals_not_paper_gate",
                    )
                )
    destination.mkdir()
    if analysis_root is not None:
        patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=SOURCE)
        (destination / "analysis-source.patch").write_bytes(patch)
        write_once(
            destination / "analysis_provenance.json",
            dict(
                classification="derived_CPU_analysis_no_new_GPU_inference",
                producer_git_sha=plan["source_sha"],
                parent_plan_sha256=plan_sha,
                parent_root=str(root.resolve()),
                original_analysis_preserved=True,
                original_analysis_artifacts={
                    str(p): digest(p) for p in (root / "analysis").iterdir() if p.is_file()
                },
                analyzer_git_sha=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
                ).strip(),
                analyzer_worktree_dirty=bool(
                    subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE)
                ),
                analyzer_patch_sha256=hashlib.sha256(patch).hexdigest(),
                analyzer_script_sha256=digest(__file__),
                gpu_reruns=0,
                producer_artifacts_verified=True,
            ),
        )
    frame.to_parquet(destination / "samples.parquet", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_parquet(destination / "summary.parquet", index=False)
    if plan["protocol"].get("equal_quality_exploratory_screen"):
        from mosaickv.evaluation.equal_quality_memory import exploratory_memory_screen

        screen = exploratory_memory_screen(
            frame,
            candidate=plan["protocol"]["candidate_arm"],
            baselines=tuple(
                m
                for m in plan["protocol"]["methods"]
                if m not in {plan["protocol"]["candidate_arm"], "mosaickv_exact"}
            ),
            replicates=plan["protocol"]["bootstrap_replicates"],
            seed=plan["protocol"]["seed"],
        )
        pd.DataFrame(screen).to_parquet(
            destination / "equal_quality_memory_screen.parquet", index=False
        )
    if failure_records:
        pd.DataFrame(failure_records).to_parquet(
            destination / "failed_samples.parquet", index=False
        )
    write_once(
        destination / "report.json",
        dict(
            status="passed_execution" if not missing and not failures else "incomplete_or_failed",
            plan_sha256=plan_sha,
            missing_workers=missing,
            failures=failures,
            completed_rows=len(rows),
            expected_rows=len(plan["jobs"]) * plan["protocol"]["samples_per_task"],
            failed_or_missing_rows=(
                len(plan["jobs"]) * plan["protocol"]["samples_per_task"] - len(rows)
            ),
            comparisons=comparisons,
            **(
                {
                    "v6_preliminary_quality_screen"
                    if plan["protocol"].get("candidate_arm") == SINGLE_EXCHANGE_ARM
                    else "v5_preliminary_quality_screen": v5_preliminary_screen(
                        summaries, plan["protocol"]["tasks"], plan["protocol"]["candidate_arm"]
                    )
                }
                if plan["protocol"].get("candidate_arm") in DRAFT_GUARDED_ARMS
                else {}
            ),
            paper_eligible=False,
            result_scope="small_retrospective_development_not_statistical_paper_evidence",
            smaller_cache_at_preserved_quality_established=False,
            cross_budget_direct_comparisons=False,
        ),
    )
    with (destination / "report.md").open("x") as stream:
        stream.write(
            "# Common bounded-history pilot\n\nDevelopment-only; "
            "all baseline rows are adaptations. "
            "FullKV is a quality reference, never a compressed latency comparator.\n\n"
        )
        stream.write(summary.to_markdown(index=False))
        stream.write(
            "\n\nInitial/final/peak bytes include generated reserve. No smaller-cache claim can "
            "be inferred from equal-byte points. "
            f"{plan['protocol']['samples_per_task']} examples/task cannot establish "
            "general superiority. Smaller-cache claims require a frozen quality target.\n"
        )
    print(
        json.dumps(
            dict(
                missing=missing,
                failures=failures,
                rows=len(rows),
                report=str(destination / "report.json"),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "prepare-retry", "worker", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-sha")
    parser.add_argument("--analysis-root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.protocol)
    elif args.action == "prepare-retry":
        prepare_retry(args.root, args.parent, args.parent_sha)
    elif args.action == "worker":
        worker(args.root, args.plan_sha, args.index)
    else:
        analyze(args.root, args.plan_sha, analysis_root=args.analysis_root)
