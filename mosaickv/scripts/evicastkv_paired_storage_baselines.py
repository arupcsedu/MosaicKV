#!/usr/bin/env python3
"""Finite same-GPU baseline decode pairing on exactly two consumed inputs."""

from pathlib import Path

import evicastkv_storage_successor as base

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

BASE = Path("/scratch/djy8hg/runs/evicastkv/paired-storage-baselines-20261030-v1")
METHODS = (
    "full_kv",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
    "mosaickv_exact",
)
original_study, original_choices, original_analyze = base.baseline_study, base.choices, base.analyze


def prepare(root, phase):
    if phase != "baselines" or root.exists():
        raise ValueError("one new immutable consumed baseline phase required")
    parent = base.PARENT
    prior = base.common.read(parent / "plan.json")
    if base.common.read(parent / "analysis/GATE.json")["failures"]:
        raise ValueError("completed exact controls required")
    bound = dict(prior["bound_files"])
    for root_prior in (
        parent,
        Path("/scratch/djy8hg/runs/evicastkv/quality-latency-consumed-20261030-v1/lookm-retry1"),
    ):
        for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
            bound[str(root_prior / name)] = sha(root_prior / name)
    if base.common.read(root_prior / "analysis/GATE.json")["failures"]:
        raise ValueError("LOOK-M selective repair must have passed")
    plan = dict(
        prior,
        source_sha=clean_source(base.SOURCE),
        bound_files=bound,
        phase="baselines",
        protocol="same_GPU_consumed_baseline_randomized_decode_v1",
        jobs=[dict(j, method="all_primary_baselines") for j in prior["jobs"]],
        requested_workers=2,
        warmups=5,
        timed_repetitions=20,
        timing_seed=20261030,
        baseline_methods=list(METHODS),
        output_tokens=32,
        maximum_a100s=6,
        exact_encoded_byte_matching_claim=False,
        same_original_source_slot_capacity=True,
        baseline_selection_budget="same_frozen_source_slots_before_lossless_storage",
        same_storage_choices_offered_to_all_eligible_methods=True,
        common_format_choice="minimum_actual_component_bytes_per_layer",
        all_method_formats_resident_diagnostic=True,
        timings_exclude_prefill_and_original_selection=True,
        whole_request_peak_certified=False,
        no_benchmark_label_access=True,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        finite_next_phase_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def choices(*args, **kwargs):
    variants, proof = original_choices(*args, **kwargs)
    selected = tuple(
        min(layer, key=lambda f: (f.component_bytes, f.kind))
        for layer in zip(*variants.values(), strict=True)
    )
    return {"common_min_storage": selected}, proof


def paired_study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import gc
    import random

    import torch

    from mosaickv.evaluation.storage_runs import run

    pending, controls = {}, []

    def capture_timing(_adapter, _prefill, representations, reference, method, _plan):
        if len(representations) != 1:
            raise ValueError("one preregistered minimum-storage format per method required")
        pending[method] = (next(iter(representations.values())), reference)
        return []

    original_timing = base.timing
    base.timing = capture_timing
    try:
        for method in METHODS:
            folder = out / method
            folder.mkdir()
            controls.extend(
                original_study(
                    adapter,
                    prefill,
                    capture,
                    frozen,
                    dict(job, method=method),
                    cfg,
                    full_reference,
                    prepared,
                    folder,
                    plan,
                )
            )
        if set(pending) != set(METHODS) - {"full_kv"}:
            raise ValueError("all compressed competitors, no FullKV latency required")
        generator = random.Random(plan["timing_seed"])
        trials = []
        for trial in range(-plan["warmups"], plan["timed_repetitions"]):
            order = list(pending)
            generator.shuffle(order)
            for order_index, method in enumerate(order):
                gc.collect()
                formats, reference = pending[method]
                actual, record = run(adapter, prefill, formats, audit=False)
                if not torch.equal(actual, reference):
                    raise RuntimeError(f"paired timing changed {method} logits")
                record.pop("storage")
                row = dict(
                    method=method,
                    trial=trial,
                    warmup=trial < 0,
                    order_index=order_index,
                    bitwise_baseline_logits=True,
                    **record,
                )
                write_once(out / f"trial-{trial}-{method}.json", row)
                trials.append(row)
        return [
            dict(
                method="paired_primary_baselines",
                baseline_controls=controls,
                trials=trials,
                fullkv_excluded_from_latency=True,
                no_paper_or_statistical_quality_claim=True,
            )
        ]
    finally:
        base.timing = original_timing
        write_once(
            out / "nested_evidence.json",
            {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
        )


def analyze(root, expected):
    import numpy as np

    original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    if gate["failures"]:
        return
    plan = base.common.inputs(root, expected)
    evidence, summaries = {}, []
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        index = base.common.read(folder / "nested_evidence.json")
        for name, digest in index.items():
            if sha(folder / name) != digest:
                raise ValueError("nested paired evidence changed")
        evidence[str(folder / "nested_evidence.json")] = sha(folder / "nested_evidence.json")
    for record in gate["controls"]:
        trials = [r for r in record["trials"] if not r["warmup"]]
        evicast = {
            r["trial"]: r["decode_wall_seconds"] for r in trials if r["method"] == "mosaickv_exact"
        }
        for method in METHODS[1:]:
            rows = [r for r in trials if r["method"] == method]
            if len(rows) != 20 or len(evicast) != 20:
                raise ValueError("twenty complete paired trials required")
            times = np.array([r["decode_wall_seconds"] for r in rows])
            ratios = np.array([evicast[r["trial"]] / r["decode_wall_seconds"] for r in rows])
            rng = np.random.default_rng(plan["timing_seed"])
            boot = np.median(ratios[rng.integers(0, 20, size=(10000, 20))], axis=1)
            summaries.append(
                dict(
                    task=record["task"],
                    method=method,
                    repetitions=20,
                    decode_seconds_p50=float(np.median(times)),
                    decode_seconds_p95=float(np.percentile(times, 95)),
                    evicast_over_method_paired_median_ratio=float(np.median(ratios)),
                    evicast_ratio_trial_bootstrap_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                    not_example_bootstrap=True,
                    not_end_to_end=True,
                )
            )
    out = root / "paired_analysis"
    out.mkdir()
    write_once(
        out / "summary.json",
        dict(
            summaries=summaries,
            quality_scope="two_consumed_inputs_model_reference_only",
            exact_byte_matched=False,
            whole_request_peak_certified=False,
            superiority_established=False,
        ),
    )
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            evidence=evidence,
            outputs={"summary.json": sha(out / "summary.json")},
        ),
    )


def install():
    base.BASE = BASE
    base.PHASES = ("baselines",)
    base.SCRIPT_NAME = "evicastkv_paired_storage_baselines.sbatch"
    base.prepare, base.choices, base.baseline_study, base.analyze = (
        prepare,
        choices,
        paired_study,
        analyze,
    )


if __name__ == "__main__":
    install()
    base.main()
