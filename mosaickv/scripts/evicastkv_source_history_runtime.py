#!/usr/bin/env python3
"""Finite source-history canary -> six-path paired consumed-input diagnostic."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import evicastkv_encoded_baselines as encoded

from mosaickv.config import RunConfig, canonical_config_json, config_sha256
from mosaickv.evaluation.fox_campaign import sha, write_once

base = encoded.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/source-history-runtime-consumed-20261030-v1")
FEASIBILITY = Path("/scratch/djy8hg/runs/evicastkv/history-source-packets-consumed-20261030-v1")
original_analyze = base.analyze


def prepare(root, phase):
    if root.name not in ("canary", "comparison"):
        raise ValueError("only finite canary/comparison stages permitted")
    decision_path = FEASIBILITY / "analysis/history_source_packet_decision.json"
    if not base.common.read(decision_path)["prospective_feasible"]:
        raise ValueError("all six paths must first pass packet feasibility")
    writer = encoded.history.original_writer

    def register(path, value):
        if path.name == "plan.json":
            value["bound_files"][str(decision_path)] = sha(decision_path)
            for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
                source = FEASIBILITY / name
                value["bound_files"][str(source)] = sha(source)
            if root.name == "comparison":
                path_gate = root.parent / "canary/analysis/source_history_decision.json"
                if not base.common.read(path_gate)["runtime_canary_passed"]:
                    raise ValueError("source history numerical/memory canary must pass")
                value["bound_files"][str(path_gate)] = sha(path_gate)
            methods = (
                ("mosaickv_exact",) if root.name == "canary" else encoded.history.paired.METHODS
            )
            jobs = []
            for old in value["jobs"]:
                for method in methods:
                    job = copy.deepcopy(old)
                    job.update(index=len(jobs), method=method)
                    for arm in job["arms"]:
                        config = RunConfig.from_mapping(arm["config"])
                        config = replace(
                            config, generation=replace(config.generation, max_new_tokens=128)
                        )
                        arm["original_parent_config_sha"] = arm["config_sha"]
                        arm.update(
                            config=json.loads(canonical_config_json(config)),
                            config_sha=config_sha256(config),
                        )
                    jobs.append(job)
            value.update(
                protocol="exact_source_sealed_history_runtime_v1",
                stage=root.name,
                jobs=jobs,
                requested_workers=len(jobs),
                output_tokens=128,
                quality_reference_tokens=32,
                ordinary_generate_parity_tokens=16,
                raw_normalized_tail_and_phase_storage_charged=True,
                source_history_default_enabled=False,
                canary_initial_bytes_equal_required=True,
                canary_final_bytes_reduced_required=True,
                canary_observed_peak_reduced_required=True,
                timing_only_after_numerical_memory_canary=True,
                fullkv_excluded_from_latency=True,
                finite_source_history_comparison_authorized=root.name == "canary",
                maximum_a100s=6,
                all_method_formats_resident_diagnostic=False,
                no_new_selection_or_paper_campaign=True,
            )
        writer(path, value)

    encoded.history.original_writer = register
    try:
        return encoded.prepare(root, phase)
    finally:
        encoded.history.original_writer = writer


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import gc
    import random

    import torch

    from mosaickv.evaluation.storage_runs import run

    pending = {}
    old_choices, old_timing = base.choices, base.timing

    def choices(*args, **kwargs):
        variants, proof = encoded.fused_choices(*args, **kwargs)
        pending["formats"] = next(iter(variants.values()))
        return variants, proof

    def no_timing(_adapter, _prefill, variants, reference, method, _plan):
        pending["formats"] = next(iter(variants.values()))
        pending["reference32"] = reference
        return []

    base.choices, base.timing = choices, no_timing
    try:
        folder = out / "baseline"
        folder.mkdir()
        metrics = encoded.history.paired.original_study(
            adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, folder, plan
        )
    finally:
        base.choices, base.timing = old_choices, old_timing
    formats = pending["formats"]
    constructor_comparison = plan.get("source_history_comparison_kind") == "constructor"
    workspace_comparison = plan.get("source_history_comparison_kind") == "workspace"
    reference_policy = (
        dict(source_history=True, source_history_constructor="reference")
        if constructor_comparison
        else dict(batched_history=True)
    )
    candidate_policy = dict(
        source_history=True,
        source_history_constructor=plan.get("candidate_constructor", "reference"),
    )
    if workspace_comparison:
        reference_policy = dict(source_history=True, source_history_constructor="direct")
        candidate_policy.update(shared_workspace=True, workspace_graph=True)
    reference, old = run(adapter, prefill, formats, tokens=128, audit=True, **reference_policy)
    candidate, new = run(adapter, prefill, formats, tokens=128, audit=True, **candidate_policy)
    if not torch.equal(reference, candidate) or not torch.equal(
        reference[:32], pending.get("reference32", full_reference)
    ):
        raise RuntimeError("source history changed original own-greedy logits")
    a, b = old["storage"], new["storage"]
    fields = (
        "initial_active_kv_storage_bytes",
        "final_active_kv_storage_bytes",
        "peak_active_kv_storage_bytes",
    )
    memory_pass = (
        a[fields[0]] == b[fields[0]] and b[fields[1]] < a[fields[1]] and b[fields[2]] < a[fields[2]]
    )
    if constructor_comparison:
        memory_pass = (
            a[fields[0]] == b[fields[0]]
            and a[fields[1]] == b[fields[1]]
            and b[fields[2]] <= a[fields[2]]
        )
    if workspace_comparison:
        memory_pass = (
            a[fields[0]] == b[fields[0]]
            and a[fields[1]] == b[fields[1]]
            and b[fields[2]] <= a[fields[2]] * plan["maximum_workspace_peak_ratio"]
            and b["active_workspace_bytes"] == 0
            and b["graph_private_live_bytes"] == 0
            and b["graph_private_accounting_verified"]
        )
    if b["source_history_seals"] != 7 or b["source_history_normalized_tail_bytes"] != 4 * 2**20:
        raise RuntimeError("missing source chunk or normalized-tail accounting")
    torch.save(dict(reference=reference, candidate=candidate), out / "128-token-logits.pt")
    row = dict(
        method=job["method"],
        bitwise_logits=True,
        original_32_token_prefix_bitwise=True,
        ordinary_history=old,
        source_history=new,
        initial_final_peak_memory_gate=memory_pass,
        whole_request_peak_certified=False,
        no_quality_superiority_claim=True,
        source_history_comparison_kind=(
            "workspace"
            if workspace_comparison
            else "constructor"
            if constructor_comparison
            else "KV_vs_source"
        ),
    )
    write_once(out / "source_history_controls.json", row)
    trials = []
    if plan["stage"] == "comparison" and job["method"] != "full_kv":
        rng = random.Random(plan["timing_seed"])
        variants = {"ordinary": reference_policy, "source": candidate_policy}
        for trial in range(-plan["warmups"], plan["timed_repetitions"]):
            order = list(variants)
            rng.shuffle(order)
            for order_index, name in enumerate(order):
                gc.collect()
                logits, record = run(
                    adapter, prefill, formats, tokens=128, audit=False, **variants[name]
                )
                if not torch.equal(logits, reference):
                    raise RuntimeError("timed source history changed model outputs")
                record.pop("storage")
                trials.append(
                    dict(
                        trial=trial,
                        warmup=trial < 0,
                        order_index=order_index,
                        variant=name,
                        **record,
                    )
                )
    write_once(out / "paired_128_decode_trials.json", trials)
    return [
        dict(
            **row,
            baseline_metrics=metrics,
            trials=trials,
            fullkv_excluded_from_latency=job["method"] == "full_kv",
        )
    ]


def analyze(root, expected):
    import numpy as np

    original_analyze(root, expected)
    plan = base.common.read(root / "plan.json")
    gate = base.common.read(root / "analysis/GATE.json")
    rows = gate["controls"]
    passed = (
        not gate["failures"]
        and len(rows) == len(plan["jobs"])
        and all(row["bitwise_logits"] and row["initial_final_peak_memory_gate"] for row in rows)
    )
    timings = []
    for row in rows:
        trials = [r for r in row["trials"] if not r["warmup"]]
        if row["method"] == "full_kv" or not trials:
            continue
        old = {r["trial"]: r["decode_wall_seconds"] for r in trials if r["variant"] == "ordinary"}
        new = {r["trial"]: r["decode_wall_seconds"] for r in trials if r["variant"] == "source"}
        if set(old) != set(new) or len(old) != 20:
            raise ValueError("twenty complete paired timing blocks required")
        ratio = np.array([new[i] / old[i] for i in sorted(old)])
        boot = np.median(ratio[np.random.default_rng(20261030).integers(0, 20, (10000, 20))], 1)
        timings.append(
            dict(
                task=row["task"],
                method=row["method"],
                paired_ratio=float(np.median(ratio)),
                ratio_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                ordinary_median_seconds=float(np.median(list(old.values()))),
                source_median_seconds=float(np.median(list(new.values()))),
            )
        )
    write_once(
        root / "analysis/source_history_decision.json",
        dict(
            runtime_canary_passed=passed,
            stage=plan["stage"],
            timing=timings,
            no_independent_quality_or_general_superiority_claim=True,
            whole_request_peak_certified=False,
            no_fresh_or_paper_campaign=True,
        ),
    )
    if passed and plan["finite_source_history_comparison_authorized"]:
        child = root.parent / "comparison"
        digest = prepare(child, "baselines")
        base.launch.submit(
            child, digest, script_name=base.SCRIPT_NAME, worker_minutes=30, memory_gb=24
        )
        write_once(root / "analysis/advancement.json", dict(root=str(child), plan_sha256=digest))


if __name__ == "__main__":
    encoded.history.paired.install()
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_source_history_runtime.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, study, analyze
    base.main()
