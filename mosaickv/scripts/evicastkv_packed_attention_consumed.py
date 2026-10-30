#!/usr/bin/env python3
"""Finite consumed-input numerical canary, then same-GPU storage/timing study."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import evicastkv_storage_successor as base

from mosaickv.config import RunConfig, canonical_config_json, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

ROOT = Path("/scratch/djy8hg/runs/evicastkv/packed-attention-consumed-20261030-v1")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/source-history-direct-consumed-20261030-v1/canary")
MICRO = Path("/scratch/djy8hg/runs/evicastkv/packed-attention-20261030-v4/micro")
METHODS = (
    "full_kv",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
    "mosaickv_exact",
)


def prepare(root, phase):
    if phase != "baselines" or root.exists() or root.name not in {"canary", "comparison"}:
        raise ValueError("new bounded canary/comparison root required")
    source = clean_source(base.SOURCE)
    old = base.common.read(PARENT / "plan.json")
    micro = base.common.read(MICRO / "GATE.json")
    if not micro["numerical_passed"]:
        raise ValueError("GPU reader and arithmetic controls must pass first")
    bound = dict(old["bound_files"])
    for path in (
        PARENT / "plan.json",
        MICRO / "GATE.json",
        MICRO / "manifest.json",
        MICRO / "config.json",
    ):
        bound[str(path)] = sha(path)
    if root.name == "comparison":
        gate = root.parent / "canary/analysis/packed_attention_decision.json"
        if not base.common.read(gate)["canary_passed"]:
            raise ValueError("model-level canary failed")
        bound[str(gate)] = sha(gate)
    methods = ("mosaickv_exact",) if root.name == "canary" else METHODS
    jobs = []
    for job in old["jobs"]:
        for method in methods:
            j = copy.deepcopy(job)
            j.update(index=len(jobs), method=method)
            for arm in j["arms"]:
                cfg = RunConfig.from_mapping(arm["config"])
                cfg = replace(
                    cfg,
                    generation=replace(
                        cfg.generation, max_new_tokens=32 if root.name == "canary" else 128
                    ),
                )
                arm.update(
                    config=json.loads(canonical_config_json(cfg)), config_sha=config_sha256(cfg)
                )
            jobs.append(j)
    plan = dict(
        old,
        source_sha=source,
        bound_files=bound,
        jobs=jobs,
        protocol="packed_KV_attention_stable_accumulation_consumed_v1",
        stage=root.name,
        phase="baselines",
        requested_workers=len(jobs),
        output_tokens=32 if root.name == "canary" else 128,
        retention_one_tokens=32,
        retention_one_max_logit_difference=0.001,
        selected_token_agreement_required=1.0,
        selected_max_logit_difference=1.0,
        selected_max_KL=0.02,
        warmups=5,
        timed_repetitions=20,
        timing_seed=20261030,
        finite_next_phase_authorized=False,
        no_benchmark_label_access=True,
        unchanged_selected_KV=True,
        common_kernel_offered_to_all_primary_paths=True,
        whole_request_peak_certified=False,
        numerical_micro_passed=True,
        micro_all_lengths_latency_gate_passed=micro["latency_gate_passed"],
        micro_latency_limitations_disclosed=True,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
        calibration_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def differences(reference, actual):
    import torch

    a, b = reference.double(), actual.double()
    kl = (a.softmax(-1) * (a.log_softmax(-1) - b.log_softmax(-1))).sum(-1)
    return dict(
        maximum_logit_difference=float((a - b).abs().max()),
        maximum_logit_KL=max(0.0, float(kl.max())),
        token_agreement=float((a.argmax(-1) == b.argmax(-1)).double().mean()),
        bitwise_logits=torch.equal(reference, actual),
    )


def gathered(adapter, prefill, positions):
    import torch

    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    pairs = []
    for (key, value), heads in zip(source, positions, strict=True):
        if len(heads) != key.shape[1] or len({len(h) for h in heads}) != 1:
            raise ValueError("complete uniform physical head capacity per layer required")
        if any(
            tuple(sorted(set(h))) != tuple(h) or not h or h[0] < 0 or h[-1] >= key.shape[-2]
            for h in heads
        ):
            raise ValueError("original unique sorted in-bounds memberships required")
        index = torch.tensor(heads, dtype=torch.long, device=key.device)[None, :, :, None]
        index = index.expand(1, key.shape[1], len(heads[0]), key.shape[-1])
        pairs.append(tuple(t.gather(2, index) for t in (key, value)))
    return tuple(pairs)


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import gc
    import random

    import torch

    from mosaickv.evaluation.packed_attention_decode import run
    from mosaickv.evaluation.storage_baselines import baseline_payload

    del capture, full_reference
    if plan["stage"] == "canary":
        full_pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        full, a = run(adapter, prefill, full_pairs, variant="ordinary", tokens=32, audit=True)
        identity, b = run(
            adapter, prefill, full_pairs, variant="packed_kernel", tokens=32, audit=True
        )
        parity = differences(full, identity)
        write_once(out / "retention_one.json", dict(**parity, reference=a, candidate=b))
        torch.save(dict(reference=full, candidate=identity), out / "retention_one_logits.pt")
        if parity["token_agreement"] != 1 or parity["maximum_logit_difference"] > 0.001:
            raise RuntimeError("unchanged retention-one model parity gate failed")
    profile = cfg.bounded_decode.incremental_history.layer_profile_path
    positions, pairs, proof = baseline_payload(
        adapter, prepared, prefill, cfg, job["method"], frozen, profile
    )
    if pairs is None:
        pairs = gathered(adapter, prefill, positions)
    write_once(out / "baseline_selection.json", proof)
    snapshots = tuple(t.detach().clone() for pair in pairs for t in pair)
    controls, outputs = {}, {}
    for variant in ["ordinary", "raw_kernel", "packed_kernel"]:
        outputs[variant], controls[variant] = run(
            adapter, prefill, pairs, variant=variant, tokens=plan["output_tokens"], audit=True
        )
    comparisons = {
        name: differences(outputs["ordinary"], outputs[name])
        for name in ["raw_kernel", "packed_kernel"]
    }
    codec_bits = torch.equal(outputs["raw_kernel"], outputs["packed_kernel"])
    unchanged_source = all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(snapshots, (t for pair in pairs for t in pair), strict=True)
    )
    row = dict(
        method=job["method"],
        controls=controls,
        comparisons=comparisons,
        raw_encoded_kernel_logits_bitwise=codec_bits,
        selected_source_unchanged=unchanged_source,
        no_new_selection=True,
        timings_exclude_prefill_and_selection=True,
        quality_scope="same_consumed_inputs_original_trajectory_only",
    )
    write_once(out / "packed_controls.json", row)
    torch.save(outputs, out / "logits.pt")
    if (
        not codec_bits
        or not unchanged_source
        or any(
            c["token_agreement"] != 1
            or c["maximum_logit_difference"] > 1.0
            or c["maximum_logit_KL"] > 0.02
            for c in comparisons.values()
        )
    ):
        raise RuntimeError("unchanged selected-cache numerical/isolation gate failed")
    old, new = (controls[k]["storage"] for k in ["ordinary", "packed_kernel"])
    row["memory_smaller_initial_final_peak"] = all(
        new[k] < old[k]
        for k in [
            "initial_active_kv_storage_bytes",
            "final_active_kv_storage_bytes",
            "peak_active_kv_storage_bytes",
        ]
    )
    trials = []
    if plan["stage"] == "comparison" and job["method"] != "full_kv":
        order = random.Random(plan["timing_seed"])
        for trial in range(-5, 20):
            variants = ["ordinary", "packed_kernel"]
            order.shuffle(variants)
            for order_index, variant in enumerate(variants):
                gc.collect()
                logits, record = run(
                    adapter,
                    prefill,
                    pairs,
                    variant=variant,
                    tokens=plan["output_tokens"],
                    audit=False,
                )
                if not torch.equal(logits, outputs[variant]):
                    raise RuntimeError("repeated execution changed that variant's logits")
                record.pop("storage")
                trials.append(
                    dict(trial=trial, warmup=trial < 0, order_index=order_index, **record)
                )
    write_once(out / "timing_trials.json", trials)
    row.update(trials=trials, fullkv_excluded_from_latency=job["method"] == "full_kv")
    return [row]


original_analyze = base.analyze


def analyze(root, expected):
    import numpy as np

    original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    rows = gate["controls"]
    passed = not gate["failures"] and len(rows) == len(base.common.read(root / "plan.json")["jobs"])
    timing = []
    for row in rows:
        trials = [t for t in row["trials"] if not t["warmup"]]
        if not trials:
            continue
        a = {t["trial"]: t for t in trials if t["variant"] == "ordinary"}
        b = {t["trial"]: t for t in trials if t["variant"] == "packed_kernel"}
        if set(a) != set(range(20)) or set(a) != set(b):
            raise ValueError("twenty paired complete repetitions required")
        for field in ["decode_wall_seconds", "constructor_plus_decode_seconds"]:
            ratios = np.array([b[i][field] / a[i][field] for i in sorted(a)])
            rng = np.random.default_rng(20261030)
            boot = np.median(ratios[rng.integers(0, 20, size=(10000, 20))], axis=1)
            timing.append(
                dict(
                    task=row["task"],
                    method=row["method"],
                    field=field,
                    paired_ratio_p50=float(np.median(ratios)),
                    ratio_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                    reference_p50=float(np.median([a[i][field] for i in a])),
                    candidate_p50=float(np.median([b[i][field] for i in b])),
                )
            )
    write_once(
        root / "analysis/packed_attention_decision.json",
        dict(
            canary_passed=passed,
            complete_workers=len(rows),
            failures=gate["failures"],
            timing=timing,
            whole_request_peak_certified=False,
            superiority_established=False,
            independent_quality_established=False,
            paper_campaign_authorized=False,
        ),
    )


if __name__ == "__main__":
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_packed_attention_consumed.sbatch"
    base.PHASES = ("baselines",)
    base.prepare, base.baseline_study, base.analyze = prepare, study, analyze
    base.main()
