#!/usr/bin/env python3
"""Consumed-input, all-baseline lossless history controls and paired timing."""

from pathlib import Path

import evicastkv_paired_storage_baselines as paired
from evicastkv_dual_projection import choices as dual_choices

from mosaickv.evaluation.fox_campaign import sha, write_once

base = paired.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/batched-history-consumed-20261030-v1")
original_writer = paired.write_once


def prepare(root, phase):
    validated = Path("/scratch/djy8hg/runs/evicastkv/dual-projection-consumed-20261030-v1-retry3")
    for stage in ("compact", "timing"):
        gate = base.common.read(validated / stage / "analysis/GATE.json")
        if gate["status"] != "complete_consumed_engineering_diagnostic" or gate["failures"]:
            raise ValueError("dual projection must first pass complete parity and timing controls")

    def register(path, value):
        if path.name == "plan.json":
            for stage in ("compact", "timing"):
                for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
                    value["bound_files"][str(validated / stage / name)] = sha(
                        validated / stage / name
                    )
            value.update(
                protocol="fused_prompt_bounded_batch_history_v1",
                batched_history_controls=True,
                history_control_horizons=[32, 128],
                maximum_layers_per_history_batch=4,
                timing_history_variants=["per_layer", "batched"],
                common_format_choice="minimum_bytes_among_dual_compact_or_encoded_KV",
                original_slow_full_source_replay_not_a_candidate=True,
                evicast_initial_cap_bytes=62 * 2**20,
                evicast_decode_owned_peak_cap_bytes=96 * 2**20,
                initial_cap_applies_at_both_horizons=True,
                peak_cap_applies_only_at_32_tokens=True,
            )
        original_writer(path, value)

    paired.write_once = register
    try:
        return paired.prepare(root, phase)
    finally:
        paired.write_once = original_writer


def fused_choices(*args, **kwargs):
    variants, proof = dual_choices(*args, **kwargs)
    return {"common_fused": variants["dual"]}, proof


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import gc
    import random

    import torch

    from mosaickv.evaluation.storage_runs import run

    pending, references, metrics = {}, {"full_kv": full_reference}, []
    current_method = ""
    original_choices, original_timing = base.choices, base.timing

    def capture_choices(*args, **kwargs):
        variants, proof = fused_choices(*args, **kwargs)
        pending[current_method] = next(iter(variants.values()))
        return variants, proof

    def capture_timing(_adapter, _prefill, representations, reference, method, _plan):
        pending[method] = next(iter(representations.values()))
        references[method] = reference
        return []

    base.choices, base.timing = capture_choices, capture_timing
    try:
        for method in paired.METHODS:
            current_method = method
            folder = out / method
            folder.mkdir()
            metrics.extend(
                paired.original_study(
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
        if set(pending) != set(paired.METHODS):
            raise ValueError("complete common fused-format baseline set required")
        controls = []
        for method, formats in pending.items():
            for horizon in plan["history_control_horizons"]:
                reference, original = run(adapter, prefill, formats, tokens=horizon, audit=True)
                candidate, batched = run(
                    adapter, prefill, formats, tokens=horizon, audit=True, batched_history=True
                )
                if not torch.equal(reference, candidate) or (
                    horizon == 32 and not torch.equal(reference, references[method])
                ):
                    raise RuntimeError(f"batched history changed {method}/{horizon} logits")
                if batched["batched_history_seals"] != (horizon - 2) // 16:
                    raise RuntimeError("missing batched history seal")
                a, b = original["storage"], batched["storage"]
                if a["initial_active_kv_storage_bytes"] != b["initial_active_kv_storage_bytes"]:
                    raise RuntimeError("history policy changed initial prompt storage")
                if b["final_active_kv_storage_bytes"] > a["final_active_kv_storage_bytes"]:
                    raise RuntimeError("batched history increased final live storage")
                if method == "mosaickv_exact" and (
                    b["initial_active_kv_storage_bytes"] > plan["evicast_initial_cap_bytes"]
                    or (
                        horizon == 32
                        and b["peak_active_kv_storage_bytes"]
                        > plan["evicast_decode_owned_peak_cap_bytes"]
                    )
                ):
                    raise RuntimeError("EviCastKV exceeded unchanged consumed storage cap")
                row = dict(
                    method=method,
                    horizon=horizon,
                    bitwise_logits=True,
                    original=original,
                    batched=batched,
                    observed_peak_not_increased=b["peak_active_kv_storage_bytes"]
                    <= a["peak_active_kv_storage_bytes"],
                )
                torch.save(
                    dict(reference=reference, candidate=candidate),
                    out / f"{method}-{horizon}-logits.pt",
                )
                write_once(out / f"{method}-{horizon}-history-control.json", row)
                controls.append(row)
        generator = random.Random(plan["timing_seed"])
        trials = []
        for trial in range(-plan["warmups"], plan["timed_repetitions"]):
            order = [
                (m, batched) for m in paired.METHODS if m != "full_kv" for batched in (False, True)
            ]
            generator.shuffle(order)
            for order_index, (method, batched) in enumerate(order):
                gc.collect()
                actual, record = run(
                    adapter, prefill, pending[method], audit=False, batched_history=batched
                )
                if not torch.equal(actual, references[method]):
                    raise RuntimeError("paired history timing changed baseline logits")
                record.pop("storage")
                row = dict(
                    method=method,
                    trial=trial,
                    warmup=trial < 0,
                    order_index=order_index,
                    history="batched" if batched else "per_layer",
                    **record,
                )
                write_once(out / f"trial-{trial}-{method}-{row['history']}.json", row)
                trials.append(row)
        return [
            dict(
                method="all_baselines_batched_history",
                baseline_metrics=metrics,
                history_controls=controls,
                trials=trials,
                fullkv_excluded_from_latency=True,
            )
        ]
    finally:
        base.choices, base.timing = original_choices, original_timing
        write_once(
            out / "nested_evidence.json",
            {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
        )


def analyze(root, expected):
    import numpy as np

    paired.original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    if gate["failures"]:
        return
    summaries = []
    for record in gate["controls"]:
        for method in paired.METHODS[1:]:
            rows = [r for r in record["trials"] if r["method"] == method and not r["warmup"]]
            old = {
                r["trial"]: r["decode_wall_seconds"] for r in rows if r["history"] == "per_layer"
            }
            new = [r for r in rows if r["history"] == "batched"]
            if len(old) != 20 or len(new) != 20:
                raise ValueError("complete paired history trials required")
            ratio = np.array([r["decode_wall_seconds"] / old[r["trial"]] for r in new])
            boot = np.median(
                ratio[np.random.default_rng(20261030).integers(0, 20, size=(10000, 20))], axis=1
            )
            summaries.append(
                dict(
                    task=record["task"],
                    method=method,
                    per_layer_p50_seconds=float(np.median(list(old.values()))),
                    batched_p50_seconds=float(np.median([r["decode_wall_seconds"] for r in new])),
                    batched_p95_seconds=float(
                        np.percentile([r["decode_wall_seconds"] for r in new], 95)
                    ),
                    paired_median_ratio=float(np.median(ratio)),
                    trial_bootstrap_ratio_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                )
            )
    evidence = {}
    for job in base.common.inputs(root, expected)["jobs"]:
        folder = root / f"workers/{job['index']}"
        for name, digest in base.common.read(folder / "nested_evidence.json").items():
            if sha(folder / name) != digest:
                raise ValueError("nested evidence changed")
        evidence[str(folder / "nested_evidence.json")] = sha(folder / "nested_evidence.json")
    out = root / "history_analysis"
    out.mkdir()
    write_once(
        out / "summary.json",
        dict(
            summaries=summaries,
            quality_improvement_established=False,
            independent_quality_established=False,
            whole_request_peak_certified=False,
            exact_encoded_budget_matching=False,
            fresh_campaign_authorized=False,
        ),
    )
    write_once(
        out / "manifest.json",
        dict(
            source_sha=base.common.inputs(root, expected)["source_sha"],
            plan_sha256=expected,
            evidence=evidence,
            outputs={"summary.json": sha(out / "summary.json")},
        ),
    )


if __name__ == "__main__":
    paired.install()
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_batched_history.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, study, analyze
    base.main()
