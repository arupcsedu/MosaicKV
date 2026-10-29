#!/usr/bin/env python3
"""Finite PrefixKV native-metric development curves, not a new paper claim.

Reuse input ledgers and fidelity evidence, never incompatible predictions.
The budget sweep is paired at each cap; 80-versus-160 is explicitly a separate
equal-quality hypothesis. No annotation or historical score selects identities.
"""

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

import numpy as np
import pandas as pd
import yaml

from mosaickv.config import RunConfig, canonical_config, config_sha256, load_config
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "mosaickv/configs/experiments/prefixkv_isoquality_development_v1.yaml"
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
STORAGE = tuple(f"{b}_active_kv_storage_bytes" for b in ("initial", "final", "peak"))
METRICS = ("teacher_forced_ppl", "rouge_l_f1")
POLICY_FILES = (
    "mosaickv/src/mosaickv/baselines/prefixkv.py",
    "mosaickv/src/mosaickv/evaluation/prefixkv_differential.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def validate_protocol(p):
    expected = dict(
        tasks=["llava_description", "mmvet_prefixkv"],
        samples_per_task=8,
        initial_caps_mib=[80, 160],
        methods=["prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact"],
        output_tokens=1024,
        warmups=5,
        timed_repetitions=20,
        timing_ids_per_task=2,
        paper_campaign_authorized=False,
        untouched_confirmation_authorized=False,
        maximum_logit_difference=0.001,
        bootstrap_replicates=10000,
        quality_margins={"teacher_forced_ppl": 0.0, "rouge_l_f1": 0.0},
        latency_ratio_upper_bound=1.0,
    )
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("registered bounded development scope changed")
    r = p["resources"]
    if (r["prefixkv_array_concurrency"], r["fox_array_concurrency"], r["maximum_a100s"]) != (
        3,
        3,
        6,
    ):
        raise ValueError("parallel arrays must reserve at most six A100s in total")
    if not r["six_a100s_explicitly_authorized"] or (r["cpus_per_worker"], r["wall_minutes"]) != (
        12,
        120,
    ):
        raise ValueError("one-A100/12-CPU/two-hour worker policy required")


def choose_ids(samples, excluded, *, count, seed):
    ids = [str(s["sample_id"]) for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate parent identities")
    ordered = sorted(
        set(ids) - set(excluded),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    if len(ordered) < count:
        raise ValueError("not enough IDs outside explicit tuning ledgers")
    return ordered[:count]


def controls(cfg):
    """The cache policy differs; model/input/generation/execution cannot."""
    return {name: cfg[name] for name in ("model", "dataset", "execution", "generation")} | {
        "cache_controls": {
            key: cfg["cache"][key]
            for key in (
                "block_size",
                "physical_budget_block_size",
                "physical_budget_mandatory_policy",
                "packing_mode",
                "exact_gather_mode",
            )
        }
    }


def exclusions():
    """Read planned identities only; never select based on prediction outcomes."""
    blocked = {task: set() for task in ("llava_description", "mmvet_prefixkv")}
    bindings = {}
    for path in sorted(Path("/scratch/djy8hg/runs/mosaickv").glob("bounded*/plan.json")):
        plan = read(path)
        bindings[str(path)] = sha(path)
        for job in plan.get("jobs", []):
            task = job.get("task")
            if task not in blocked:
                continue
            blocked[task].update(map(str, job.get("sample_ids", [])))
            if job.get("ledger"):
                ledger = Path(job["ledger"])
                if not ledger.is_file():
                    raise ValueError(f"unresolved tuning ledger: {ledger}")
                bindings[str(ledger)] = sha(ledger)
                blocked[task].update(str(s["sample_id"]) for s in read(ledger)["samples"])
            if job.get("input"):
                item = Path(job["input"])
                bindings[str(item)] = sha(item)
                payload = read(item)
                if "sample_id" in payload:
                    blocked[task].add(str(payload["sample_id"]))
                elif "sample" in payload and "sample_id" in payload["sample"]:
                    blocked[task].add(str(payload["sample"]["sample_id"]))
                else:
                    raise ValueError(f"unclassified tuning input identity: {item}")
    return blocked, bindings


def prepare(root):
    source_sha = clean_source(SOURCE)
    protocol = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(protocol)
    parent = Path(protocol["historical_root"])
    if sha(parent / "plan.json") != protocol["historical_plan_sha256"]:
        raise ValueError("historical parent changed")
    historical = read(parent / "plan.json")
    if root.exists():
        raise FileExistsError("new immutable output root required")
    blocked, bound = exclusions()
    bound[str(PROTOCOL)] = sha(PROTOCOL)
    bound[str(parent / "plan.json")] = sha(parent / "plan.json")
    for record in historical["evidence"].values():
        if sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("historical fidelity evidence changed")
        bound[record["path"]] = record["sha256"]
    comparison = read(historical["evidence"]["parity_comparison"]["path"])
    if comparison["status"] != "comparable" or comparison["control_mismatches"]:
        raise ValueError("baseline parity was not established")
    for sample in comparison["samples"]:
        if not sample["per_layer_cache_sizes_match"] or any(
            sample[field] != target
            for field, target in (
                ("retained_byte_delta", 0),
                ("token_agreement", 1.0),
                ("perplexity_delta", 0.0),
                ("rouge_l_f1_delta", 0.0),
            )
        ):
            raise ValueError("strict historical official-path parity required")
    for name in POLICY_FILES:
        previous = subprocess.check_output(
            ["git", "show", f"{protocol['historical_source_sha']}:{name}"], cwd=SOURCE
        )
        if hashlib.sha256(previous).hexdigest() != sha(SOURCE / name):
            raise ValueError("baseline changed since passed official-path parity")
        bound[str(SOURCE / name)] = sha(SOURCE / name)
    root.mkdir(parents=True)
    jobs = []
    cohorts = {}
    contract = yaml.safe_load(
        (SOURCE / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml").read_text()
    )
    for task in protocol["tasks"]:
        route = contract["lanes"]["prefixkv"]["tasks"][task]
        parent_ledger = Path(route["parent_ledger"])
        if sha(parent_ledger) != route["parent_ledger_sha256"]:
            raise ValueError("native input parent changed")
        bound[str(parent_ledger)] = sha(parent_ledger)
        data = read(parent_ledger)
        ids = choose_ids(data["samples"], blocked[task], count=8, seed=protocol["seed"])
        by_id = {str(s["sample_id"]): s for s in data["samples"]}
        data["samples"] = [by_id[value] for value in ids]
        data.update(parent_ledger=str(parent_ledger), parent_ledger_sha256=sha(parent_ledger))
        ledger = root / f"ledgers/{task}.json"
        write_once(ledger, data)
        bound[str(ledger)] = sha(ledger)
        cohorts[task] = dict(
            sample_ids=ids,
            excluded_ids=sorted(blocked[task]),
            classification=protocol["classification"],
            globally_untouched=False,
        )
        reference = root / f"references/{task}.json"
        common = None
        for method in ["full_kv", *protocol["methods"]]:
            old = next(
                j
                for j in historical["jobs"]
                if j["task"] == task and j["lane"] == "shared_byte_40mib" and j["method"] == method
            )
            base = canonical_config(load_config(old["config"]))
            if common is None:
                common = controls(base)
            if controls(base) != common or base["generation"]["max_new_tokens"] != 1024:
                raise ValueError("input/generation/processor controls differ across methods")
            if (
                base.get("bounded_decode")
                or base.get("evidence_policy")
                or base.get("output_preserving")
            ):
                raise ValueError("frozen append-only candidate required")
            if method == "prefixkv_reimpl":
                profile = Path(base["prefixkv"]["profile_path"])
                bound[str(profile)] = sha(profile)
            for cap in [None] if method == "full_kv" else protocol["initial_caps_mib"]:
                cfg = copy.deepcopy(base)
                if cap is not None:
                    cfg["cache"]["budget_value"] = cap * 2**20
                parsed = RunConfig.from_mapping(cfg)
                path = root / f"configs/{task}-{method}-{cap}.json"
                write_once(path, canonical_config(parsed))
                bound[str(path)] = sha(path)
                index = len(jobs)
                jobs.append(
                    dict(
                        index=index,
                        task=task,
                        method=method,
                        cap_mib=cap,
                        config=str(path),
                        config_sha=config_sha256(parsed),
                        ledger=str(ledger),
                        ledger_sha=sha(ledger),
                        sample_ids=ids,
                        reference=str(reference),
                        output=str(root / f"outputs/{index:02d}"),
                    )
                )
    write_once(
        root / "plan.json",
        dict(
            schema_version=1,
            source_sha=source_sha,
            protocol=protocol,
            bound_files=bound,
            cohorts=cohorts,
            jobs=jobs,
            paper_campaign_authorized=False,
            quality_requests=112,
            generation_warmups=70,
            maximum_conditional_timing_requests=80,
            maximum_conditional_timing_warmups=20,
            classification=protocol["classification"],
        ),
    )
    print(json.dumps(dict(root=str(root), plan_sha=sha(root / "plan.json"), workers=len(jobs))))


def inputs(root, plan_sha):
    if sha(root / "plan.json") != plan_sha:
        raise ValueError("plan hash changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    validate_protocol(plan["protocol"])
    for path, expected in plan["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"bound artifact changed: {path}")
    return plan


def worker(root, plan_sha, index):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.backends.hf_runtime import (
        HuggingFaceMosaicKVModel,
        compare_runtime_retention_one,
    )
    from mosaickv.evaluation.cache_phase_audit import audit_phase_directory
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import record_hf_run, settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = inputs(root, plan_sha)
    if plan.get("retry") and index not in plan["retry"]["retry_indices"]:
        raise ValueError("only the explicitly registered failed worker may be rerun")
    job = plan["jobs"][index]
    completion = root / f"workers/{index:02d}.json"
    if completion.exists():
        previous = read(completion)
        if previous["plan_sha256"] != plan_sha or previous["source_sha"] != plan["source_sha"]:
            raise ValueError("resume provenance changed")
        if previous["status"] != "completed":
            raise RuntimeError("failed attempt retained; retry needs a new immutable attempt root")
        for path, expected in previous["artifacts"].items():
            if sha(Path(path)) != expected:
                raise ValueError("completed artifact changed")
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic CUDA workspace must be set before Python starts")
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("exactly one A100 required")
    attempt = root / f"attempts/{index:02d}-{time.time_ns()}"
    started = time.monotonic()
    result = dict(
        index=index,
        plan_sha256=plan_sha,
        source_sha=plan["source_sha"],
        task=job["task"],
        method=job["method"],
        cap_mib=job["cap_mib"],
    )
    adapter = None
    try:
        cfg = load_config(job["config"])
        if config_sha256(cfg) != job["config_sha"]:
            raise ValueError("resolved configuration changed")
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
        result["gpu_before"] = asdict(capture_gpu_environment(torch))
        first = ledger.samples[0]
        messages = build_multimodal_messages(
            first.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in first.media)
        )
        prepared = adapter.prepare_inputs(messages)
        parity = {}
        for name, fn in (
            ("generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ):
            check = fn(adapter, prepared, max_new_tokens=16)
            parity[name] = asdict(check)
            if check.token_agreement != 1 or check.maximum_logit_difference > 0.001:
                raise RuntimeError(f"registered FullKV parity failed: {name}")
        if job["method"] != "full_kv":
            identity = replace(
                cfg,
                cache=replace(cfg.cache, retention_ratio=1.0, budget_value=2**63 - 1),
                generation=replace(cfg.generation, max_new_tokens=16),
            )
            check = compare_runtime_retention_one(adapter, prepared, identity)
            parity["runtime_retention_one"] = asdict(check)
            if check.token_agreement != 1 or check.maximum_logit_difference > 0.001:
                raise RuntimeError("registered full runtime retention-1 parity failed")
        write_once(attempt / "parity.json", parity)
        runtime = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=attempt / "warmup-traces")
        for repeat in range(5):
            settle(adapter)
            warm = runtime.generate(
                EvaluationRequest(f"warmup-{repeat}", first.sample_id, job["task"], messages, {})
            )
            write_once(attempt / f"warmup-{repeat}.json", dict(metrics=asdict(warm.metrics)))
        del runtime
        settle(adapter)
        reference = Path(job["reference"])
        is_full = job["method"] == "full_kv"
        record = record_hf_run(
            cfg,
            adapter,
            task=job["task"],
            run_id=f"prefix-iso-{index:02d}",
            output_root=Path(job["output"]),
            ledger_path=Path(job["ledger"]),
            ledger_sha=job["ledger_sha"],
            sample_ids=tuple(job["sample_ids"]),
            phase="retrospective_development",
            repetition=0,
            order=index,
            export_metrics=True,
            materialized_prefixkv=True,
            fullkv_reference_output=reference if is_full else None,
            fullkv_reference_input=None if is_full else reference,
            fullkv_reference_sha256=None if is_full else sha(reference),
        )
        phase_checks = audit_phase_directory(
            Path(job["output"]) / "traces", job["sample_ids"], None
        )
        validate_rows(record["rows"], job["sample_ids"], job["cap_mib"])
        if not is_full and any(
            c["storage_bytes"]["initial"] != cfg.cache.budget_value for c in phase_checks
        ):
            raise RuntimeError(
                "teacher-forcing/generation initial bytes differ from the registered cap"
            )
        result.update(
            status="completed",
            record=record,
            phase_checks=phase_checks,
            parity=parity,
            reference_sha256=sha(reference),
            gpu_after=asdict(capture_gpu_environment(torch)),
        )
        artifact_paths = [*Path(job["output"]).rglob("*"), reference]
        artifact_paths += [Path(r["manifest_path"]) for r in record["rows"]]
        result["artifacts"] = {str(p): sha(p) for p in artifact_paths if p.is_file()}
    except Exception as exc:
        result.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            failed_sample_ids=job["sample_ids"],
            failure_denominator=len(job["sample_ids"]),
        )
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "not_used")
        write_once(completion, result)


def validate_rows(rows, ids, cap):
    if len(rows) != len(ids) or sorted(r["sample_id"] for r in rows) != sorted(ids):
        raise ValueError("missing/duplicate sample rows")
    for row in rows:
        if row["status"] != "completed" or row.get("error") or row["generated_tokens"] != 1024:
            raise ValueError("failed or truncated prediction")
        for field in ("prefill_time", "compression_time", "decode_time", "ttft", "end_to_end_time"):
            if row[field] is None or not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError("missing timing")
        if cap is not None and row[STORAGE[0]] != cap * 2**20:
            raise ValueError("initial physical cap not filled exactly")
        if any(type(row[s]) is not int or row[s] < 1 for s in STORAGE):
            raise ValueError("missing physical storage")
        if row[STORAGE[2]] < max(row[STORAGE[0]], row[STORAGE[1]]):
            raise ValueError("invalid peak storage")
        if row["residual_kv_bytes"] != 0 or row["repair_count"] != 0:
            raise ValueError("exact-only comparison contains residuals or repair")


def paired_metrics(candidate, reference, metric, *, replicates=10000, family=4, seed=20271014):
    c = candidate.sort_values("sample_id")
    r = reference.sort_values("sample_id")
    if list(c.sample_id) != list(r.sample_id) or c.sample_id.duplicated().any() or len(c) == 0:
        raise ValueError("paired metrics require identical unique samples")

    def reduce(frame, indices):
        if metric == "rouge_l_f1":
            if not np.isfinite(frame.metric_value.to_numpy(float)).all():
                raise ValueError("non-finite ROUGE-L observations")
            return frame.metric_value.to_numpy(float)[indices].mean(axis=-1)
        if metric != "teacher_forced_ppl":
            raise ValueError("metric families cannot be averaged together")
        nll, tokens = frame.numerator.to_numpy(float), frame.denominator.to_numpy(float)
        if not np.isfinite(nll).all() or not np.isfinite(tokens).all() or (tokens <= 0).any():
            raise ValueError("invalid NLL sufficient statistics")
        return np.exp(nll[indices].sum(axis=-1) / tokens[indices].sum(axis=-1))

    point = float(reduce(c, np.arange(len(c))) - reduce(r, np.arange(len(r))))
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, replicates, 256):
        idx = rng.integers(0, len(c), size=(min(256, replicates - start), len(c)))
        deltas.extend(reduce(c, idx) - reduce(r, idx))
    low, high = map(float, np.quantile(deltas, [0.025 / family, 1 - 0.025 / family]))
    return dict(
        difference=point,
        simultaneous_ci_low=low,
        simultaneous_ci_high=high,
        family_size=family,
        bootstrap_replicates=replicates,
        metric_direction="lower" if metric == "teacher_forced_ppl" else "higher",
        passes=bool(high <= 0 if metric == "teacher_forced_ppl" else low >= 0),
    )


def completed_worker(root, plan, plan_sha, index):
    """Read original producer records without relabeling their Git/plan SHAs."""
    imported = plan.get("imported_workers", {}).get(str(index))
    path = Path(imported["path"]) if imported else root / f"workers/{index:02d}.json"
    if not path.exists():
        if imported:
            raise ValueError("imported completed worker disappeared")
        return None
    if imported and sha(path) != imported["sha256"]:
        raise ValueError("imported worker changed")
    result = read(path)
    expected_plan = imported["plan_sha256"] if imported else plan_sha
    expected_source = imported["source_sha"] if imported else plan["source_sha"]
    if (
        result["plan_sha256"] != expected_plan
        or result["source_sha"] != expected_source
        or result["index"] != index
    ):
        raise ValueError("worker source/plan/index mismatch")
    if imported and result["status"] != "completed":
        raise ValueError("only completed workers can be reused")
    for artifact, expected in result.get("artifacts", {}).items():
        if sha(Path(artifact)) != expected:
            raise ValueError("completed metric/trace/manifest artifact changed")
    return result


def analyze(root, plan_sha):
    plan = inputs(root, plan_sha)
    out = root / "analysis"
    records, failed = [], []
    for job in plan["jobs"]:
        w = completed_worker(root, plan, plan_sha, job["index"])
        if w is None or w.get("status") != "completed":
            failed.append(
                dict(
                    index=job["index"],
                    task=job["task"],
                    method=job["method"],
                    sample_ids=job["sample_ids"],
                    error=w.get("error") if w else "missing_worker",
                )
            )
        else:
            records.append((job, w))
    write_once(
        out / "failures.json",
        dict(
            failures=failed,
            planned_sample_rows=sum(len(j["sample_ids"]) for j in plan["jobs"]),
            failed_or_missing_sample_rows=sum(len(j["sample_ids"]) for j in failed),
        ),
    )
    if failed:
        write_once(
            out / "GATE.json",
            dict(
                status="failed_execution",
                failures=failed,
                calibration_authorized=False,
                paper_campaign_authorized=False,
            ),
        )
        return
    rows, observations, summaries = [], {}, []
    for job, w in records:
        record = w["record"]
        frame = pd.read_parquet(record["metric_observations_parquet_output"])
        frame = frame[frame.metric_scope == "sample"]
        observations[job["task"], job["method"], job["cap_mib"]] = frame
        rows.extend(
            dict(**r, task=job["task"], configured_method=job["method"], cap_mib=job["cap_mib"])
            for r in record["rows"]
        )
        for metric in METRICS:
            f = frame[frame.metric_name == metric]
            if len(f) != len(job["sample_ids"]) or f.sample_id.duplicated().any():
                raise ValueError("missing native metric rows")
            value = (
                float(f.metric_value.mean())
                if metric == "rouge_l_f1"
                else math.exp(math.fsum(map(float, f.numerator)) / sum(map(int, f.denominator)))
            )
            summaries.append(
                dict(
                    task=job["task"],
                    method=job["method"],
                    cap_mib=job["cap_mib"],
                    metric_name=metric,
                    metric_value=value,
                    sample_count=len(f),
                )
            )
    raw = pd.DataFrame(rows)
    exact, phase_exact = True, True
    for task in plan["protocol"]["tasks"]:
        for cap in (80, 160):
            subset = raw[(raw.task == task) & (raw.cap_mib == cap)]
            for _, group in subset.groupby("sample_id"):
                exact &= len(group) == 3 and all(group[f].nunique() == 1 for f in STORAGE)
            phase = [
                dict(**c, method=j["method"])
                for j, w in records
                if j["task"] == task and j["cap_mib"] == cap
                for c in w["phase_checks"]
            ]
            for sample in plan["cohorts"][task]["sample_ids"]:
                for name in ("generation", "teacher_forced"):
                    values = [
                        c["storage_bytes"]
                        for c in phase
                        if c["sample_id"] == sample and c["phase"] == name
                    ]
                    phase_exact &= len(values) == 3 and all(v == values[0] for v in values)
    contrasts = []
    matched_contrasts = []
    task_gates = {}
    for task in plan["protocol"]["tasks"]:
        for cap in (80, 160):
            for baseline in ("prompt_attention_topk", "prefixkv_reimpl"):
                candidate_frame = observations[task, "mosaickv_exact", cap]
                baseline_frame = observations[task, baseline, cap]
                for metric in METRICS:
                    stats = paired_metrics(
                        candidate_frame[candidate_frame.metric_name == metric],
                        baseline_frame[baseline_frame.metric_name == metric],
                        metric,
                        family=16,
                    )
                    matched_contrasts.append(
                        dict(
                            task=task,
                            cap_mib=cap,
                            baseline=baseline,
                            metric_name=metric,
                            label="same_realized_byte_budget",
                            **stats,
                        )
                    )
        c = observations[task, "mosaickv_exact", 80]
        r = observations[task, "prefixkv_reimpl", 160]
        quality = True
        for metric in METRICS:
            stats = paired_metrics(c[c.metric_name == metric], r[r.metric_name == metric], metric)
            contrasts.append(
                dict(
                    task=task,
                    metric_name=metric,
                    label="unequal_budget_equal_quality_hypothesis",
                    **stats,
                )
            )
            quality &= stats["passes"]
        can = raw[
            (raw.task == task) & (raw.configured_method == "mosaickv_exact") & (raw.cap_mib == 80)
        ].sort_values("sample_id")
        ref = raw[
            (raw.task == task) & (raw.configured_method == "prefixkv_reimpl") & (raw.cap_mib == 160)
        ].sort_values("sample_id")
        reductions = {f: float((1 - can[f].to_numpy() / ref[f].to_numpy()).min()) for f in STORAGE}
        task_gates[task] = dict(
            quality=bool(quality),
            memory=all(v > 0 for v in reductions.values()),
            minimum_paired_memory_reductions=reductions,
        )
    raw.to_parquet(out / "sample_rows.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(out / "native_metrics.parquet", index=False)
    pd.DataFrame(contrasts).to_parquet(out / "paired_cross_budget.parquet", index=False)
    pd.DataFrame(matched_contrasts).to_parquet(out / "paired_same_budget.parquet", index=False)
    passed = bool(
        exact and phase_exact and all(g["quality"] and g["memory"] for g in task_gates.values())
    )
    write_once(
        out / "GATE.json",
        dict(
            status="completed_development_only",
            source_sha=plan["source_sha"],
            imported_worker_count=len(plan.get("imported_workers", {})),
            producer_sources=sorted({w["source_sha"] for _, w in records}),
            measured_byte_matching=bool(exact),
            both_metric_phase_storage_matching=bool(phase_exact),
            tasks=task_gates,
            quality_memory_gate=passed,
            latency_gate="not_yet_measured_with_repetitions",
            next_action="bounded_timing_validation" if passed else "stop_negative_or_inconclusive",
            paper_campaign_authorized=False,
            cohort_is_untouched=False,
        ),
    )
    report = [
        "# PrefixKV native-metric development curves",
        "",
        "Retrospective development; not untouched confirmation or a paper superiority claim.",
        "FullKV supplies quality references and is excluded from compressed latency comparisons.",
        "PPL is corpus exp(sum NLL / sum tokens). ROUGE-L uses the hash-bound FP16 FullKV outputs.",
        "",
        "| Task | Method | Initial cap MiB | Metric | Measured value | N |",
        "|---|---|---:|---|---:|---:|",
    ]
    for s in summaries:
        report.append(
            f"| {s['task']} | {s['method']} | {s['cap_mib']} | "
            f"{s['metric_name']} | {s['metric_value']:.6f} | {s['sample_count']} |"
        )
    report += [
        "",
        f"Exact per-input generation storage matching: {bool(exact)}.",
        f"Both metric-phase storage matching: {bool(phase_exact)}.",
        "",
        "The registered 80/160 MiB test is an unequal-budget equal-quality hypothesis,",
        "not byte-matched. See paired_cross_budget.parquet for family-four adjusted intervals.",
        "See paired_same_budget.parquet for all matched-cap contrasts (family sixteen).",
        "No 30%-smaller-final-memory conclusion is inferred from initial caps.",
        "Quality-run latency is descriptive only; twenty-repetition timing is conditional",
        f"on the quality/memory gate: {passed}. No prototype/repair performance is asserted.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n")
    if passed:
        # Finite registered continuation, never a new cohort or parameter search.
        job = sbatch(root, plan_sha, mode="timing", indices=[0, 1])
        callback = sbatch(root, plan_sha, mode="analyze-timing", dependency=job)
        write_once(root / "decisions/timing.json", dict(gpu_array=job, cpu_callback=callback))


def timing(root, plan_sha, index):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = inputs(root, plan_sha)
    if not read(root / "analysis/GATE.json")["quality_memory_gate"]:
        raise ValueError("timing continuation requires both native quality gates")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic CUDA workspace missing")
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("one A100 required")
    task = plan["protocol"]["tasks"][index]
    selected = [
        j
        for j in plan["jobs"]
        if j["task"] == task
        and (j["method"], j["cap_mib"]) in (("mosaickv_exact", 80), ("prefixkv_reimpl", 160))
    ]
    configs = {j["method"]: load_config(j["config"]) for j in selected}
    cfg = configs["mosaickv_exact"]
    ledger = load_paper_native_ledger(
        selected[0]["ledger"],
        expected_sha256=selected[0]["ledger_sha"],
        task=default_task_registry().resolve(task),
        dataset_revision=cfg.dataset.revision,
    )
    samples = ledger.samples[:2]
    rows = []
    output = root / f"timing/{index:02d}"
    result = dict(
        task=task,
        plan_sha256=plan_sha,
        source_sha=plan["source_sha"],
        classification="same_input_generation_only_paired_timing_development",
    )
    started = time.monotonic()
    try:
        torch.set_num_threads(12)
        torch.manual_seed(cfg.execution.seed)
        torch.use_deterministic_algorithms(True)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        result["gpu_before"] = asdict(capture_gpu_environment(torch))
        messages = {
            s.sample_id: build_multimodal_messages(
                s.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in s.media)
            )
            for s in samples
        }

        def run_one(method, sample, label):
            settle(adapter)
            runtime = HuggingFaceMosaicKVModel(
                adapter, configs[method], trace_directory=output / f"traces/{label}"
            )
            value = runtime.generate(
                EvaluationRequest(label, sample.sample_id, task, messages[sample.sample_id], {})
            )
            row = dict(sample_id=sample.sample_id, method=method, **asdict(value.metrics))
            if not math.isfinite(row["end_to_end_time"]) or row["end_to_end_time"] <= 0:
                raise ValueError("invalid paired timing")
            cap = 80 if method == "mosaickv_exact" else 160
            if row[STORAGE[0]] != cap * 2**20 or row["generated_tokens"] != 1024:
                raise ValueError("timing execution changed storage or output length")
            return row

        for method in configs:
            for repetition in range(5):
                value = run_one(method, samples[0], f"warm-{method}-{repetition}")
                write_once(output / f"warmups/{method}-{repetition}.json", value)
        rng = np.random.default_rng(plan["protocol"]["seed"])
        for repetition in range(20):
            for sample in samples:
                order = list(configs)
                rng.shuffle(order)
                for ordinal, method in enumerate(order):
                    if time.monotonic() - started > 6500:
                        raise TimeoutError("bounded timing worker exhausted its safety horizon")
                    label = f"r{repetition:02d}-{sample.sample_id}-{method}"
                    row = run_one(method, sample, label)
                    row.update(repetition=repetition, order=ordinal)
                    write_once(output / f"trials/{label}.json", row)
                    rows.append(row)
        result.update(
            status="completed", rows=rows, gpu_after=asdict(capture_gpu_environment(torch))
        )
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", rows=rows)
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_once(root / f"timing/workers/{index:02d}.json", result)


def latency_interval(candidate, reference, *, seed=20271014, replicates=10000):
    """Paired hierarchical bootstrap: input clusters first, repetitions within."""
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    if c.shape != r.shape or c.ndim != 2 or c.shape[1] != 20 or min(c.shape) < 1:
        raise ValueError("paired timing needs the same inputs and twenty repetitions")
    if not np.isfinite(c).all() or not np.isfinite(r).all() or (r <= 0).any() or (c <= 0).any():
        raise ValueError("non-finite/invalid timing")
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(replicates):
        ids = rng.integers(0, len(c), size=len(c))
        reps = rng.integers(0, 20, size=(len(c), 20))
        ratios.append(float(c[ids[:, None], reps].mean() / r[ids[:, None], reps].mean()))
    low, high = map(float, np.quantile(ratios, [0.0125, 0.9875]))
    return dict(
        mean_latency_ratio=float(c.mean() / r.mean()),
        simultaneous_ci_low=low,
        simultaneous_ci_high=high,
        family_size=2,
        no_slower_gate=high <= 1,
        independent_timing_inputs=len(c),
        repetitions=20,
    )


def analyze_timing(root, plan_sha):
    plan = inputs(root, plan_sha)
    tasks, rows = {}, []
    for index, task in enumerate(plan["protocol"]["tasks"]):
        path = root / f"timing/workers/{index:02d}.json"
        if not path.exists() or read(path).get("status") != "completed":
            tasks[task] = dict(status="failed_or_incomplete")
            continue
        result = read(path)
        if result["plan_sha256"] != plan_sha or result["source_sha"] != plan["source_sha"]:
            raise ValueError("timing provenance changed")
        frame = pd.DataFrame(result["rows"])
        matrices = []
        for method in ("mosaickv_exact", "prefixkv_reimpl"):
            chosen = frame[frame.method == method]
            if len(chosen) != 40 or chosen.duplicated(["sample_id", "repetition"]).any():
                raise ValueError("incomplete/duplicate paired timing rows")
            matrix = chosen.pivot(index="sample_id", columns="repetition", values="end_to_end_time")
            if list(matrix.columns) != list(range(20)) or matrix.isna().any().any():
                raise ValueError("missing paired timing cell")
            matrices.append(matrix.sort_index())
        if list(matrices[0].index) != list(matrices[1].index):
            raise ValueError("unpaired timing identities")
        tasks[task] = dict(status="completed", **latency_interval(*matrices))
        rows.extend(dict(task=task, **r) for r in result["rows"])
    out = root / "timing/analysis"
    out.mkdir(parents=True, exist_ok=False)
    if rows:
        pd.DataFrame(rows).to_parquet(out / "trials.parquet", index=False)
    write_once(
        out / "GATE.json",
        dict(
            tasks=tasks,
            development_only=True,
            paper_campaign_authorized=False,
            no_slower_all_tasks=all(t.get("no_slower_gate", False) for t in tasks.values()),
            inferential_scope="two_fixed_development_inputs_per_task_not_population_latency",
        ),
    )


def sbatch(root, plan_sha, *, mode, indices=None, dependency=None):
    plan = read(root / "plan.json")
    r = plan["protocol"]["resources"]
    ceiling = plan.get("prefixkv_admission_ceiling", 3)
    if type(ceiling) is not int or not 1 <= ceiling <= 3:
        raise ValueError("PrefixKV scheduling amendment may only reduce its admission ceiling")
    is_gpu = mode in {"worker", "timing"}
    command = [
        "sbatch",
        "--parsable",
        f"--account={r['account']}",
        f"--partition={r['partition'] if is_gpu else 'bii'}",
        f"--cpus-per-task={12 if is_gpu else 2}",
        f"--mem={r['memory_gib_per_worker'] if is_gpu else 8}G",
        f"--time={'02:00:00' if is_gpu else '00:10:00'}",
        f"--output={root}/slurm/{mode}-%A_%a.out",
        f"--error={root}/slurm/{mode}-%A_%a.err",
    ]
    if is_gpu:
        command += [
            "--gres=gpu:a100:1",
            f"--reservation={r['reservation']}",
            "--array=" + ",".join(map(str, indices)) + f"%{ceiling}",
        ]
    else:
        command += ["--gres=none"]
    if dependency:
        command += [f"--dependency=afterany:{dependency}"]
    command += [
        str(SOURCE / "mosaickv/slurm/prefixkv_isoquality_development.sbatch"),
        str(SOURCE),
        str(root),
        plan_sha,
        mode,
    ]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    write_once(root / f"submissions/{mode}-{job}.json", dict(job=job, command=command))
    return job


def advance(root, plan_sha, phase):
    plan = inputs(root, plan_sha)
    # Claim the phase before any submission: a repeated callback cannot double
    # the arrays and silently exceed the aggregate six-GPU allocation ceiling.
    write_once(
        root / f"decisions/{phase}-started.json",
        dict(plan_sha256=plan_sha, phase=phase, submission_is_write_once=True),
    )
    if phase == "start":
        indices = [j["index"] for j in plan["jobs"] if j["method"] == "full_kv"]
        job = sbatch(root, plan_sha, mode="worker", indices=indices)
        callback = sbatch(root, plan_sha, mode="after-references", dependency=job)
    else:
        refs = [j for j in plan["jobs"] if j["method"] == "full_kv"]
        if any(
            not (root / f"workers/{j['index']:02d}.json").is_file()
            or read(root / f"workers/{j['index']:02d}.json")["status"] != "completed"
            for j in refs
        ):
            write_once(
                root / "decisions/references-failed.json", dict(status="blocked_reference_failure")
            )
            return
        indices = [j["index"] for j in plan["jobs"] if j["method"] != "full_kv"]
        job = sbatch(root, plan_sha, mode="worker", indices=indices)
        callback = sbatch(root, plan_sha, mode="analyze", dependency=job)
    write_once(
        root / f"decisions/{phase}.json",
        dict(
            gpu_array=job,
            cpu_callback=callback,
            maximum_a100s=6,
            prefixkv_reserved_slots=3,
            fox_reserved_slots=3,
        ),
    )
    # Register with the tested scheduler, but do not let a global throttle
    # rewrite increase this lane's deliberately stricter static three-slot cap.
    from control_a100_schedule import tick

    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=120,
        window="all_times",
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[job],
        per_lane_static_throttle=3,
        other_lane_reserved_slots=3,
    )
    write_once(root / f"scheduler/{phase}-policy.json", policy)
    report = tick(policy, {}, root / f"scheduler/{phase}-state.json", False)
    write_once(root / f"scheduler/{phase}-registration.json", report)
    print(json.dumps(dict(gpu_array=job, cpu_callback=callback, mode=phase)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "start",
            "after-references",
            "worker",
            "analyze",
            "timing",
            "analyze-timing",
        ],
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("scratch-only artifacts")
    if args.action == "prepare":
        prepare(args.root)
    elif not args.plan_sha:
        parser.error("immutable plan SHA required")
    elif args.action == "worker":
        worker(args.root, args.plan_sha, args.index)
    elif args.action == "analyze":
        analyze(args.root, args.plan_sha)
    elif args.action == "timing":
        timing(args.root, args.plan_sha, args.index)
    elif args.action == "analyze-timing":
        analyze_timing(args.root, args.plan_sha)
    else:
        advance(args.root, args.plan_sha, args.action)


if __name__ == "__main__":
    main()
