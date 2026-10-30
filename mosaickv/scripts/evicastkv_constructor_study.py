#!/usr/bin/env python3
"""Consumed-input redundant-work elimination with exact-format differential gate."""

from pathlib import Path

import evicastkv_paired_storage_baselines as paired
from evicastkv_dual_projection import choices as reference_choices

from mosaickv.evaluation.fox_campaign import sha, write_once

base = paired.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/direct-constructor-consumed-20261030-v1")
METHODS = ("prompt_attention_topk", "prefixkv_reimpl", "vl_cache_reimpl", "mosaickv_exact")
original_writer = paired.write_once


def prepare(root, phase):
    validated = Path("/scratch/djy8hg/runs/evicastkv/dual-projection-consumed-20261030-v1-retry3")
    for stage in ("compact", "timing"):
        gate = base.common.read(validated / stage / "analysis/GATE.json")
        if gate["status"] != "complete_consumed_engineering_diagnostic" or gate["failures"]:
            raise ValueError("validated dual source transport required")

    def register(path, value):
        if path.name == "plan.json":
            for stage in ("compact", "timing"):
                for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
                    value["bound_files"][str(validated / stage / name)] = sha(
                        validated / stage / name
                    )
            value.update(
                protocol="direct_constructor_same_format_v1",
                constructor_methods=list(METHODS),
                constructor_warmups=5,
                constructor_repetitions=20,
                same_selected_memberships=True,
                exact_initial_final_peak_storage_equality=True,
                no_quality_policy_change=True,
                lookm_not_source_eligible=True,
                common_format_choice="minimum_bytes_among_dual_compact_or_encoded_KV",
            )
        original_writer(path, value)

    paired.write_once = register
    try:
        return paired.prepare(root, phase)
    finally:
        paired.write_once = original_writer


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import gc
    import random
    import time

    import torch

    from mosaickv.evaluation.direct_constructor import construct
    from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference
    from mosaickv.evaluation.storage_baselines import baseline_payload
    from mosaickv.evaluation.storage_runs import run

    controls, trials = [], []
    profile = cfg.bounded_decode.incremental_history.layer_profile_path
    for method in METHODS:
        positions, pairs, selection = baseline_payload(
            adapter, prepared, prefill, cfg, method, frozen, profile
        )
        if pairs is not None:
            raise ValueError("original source memberships required")
        reference, old_proof = reference_choices(adapter, prefill, capture, positions, compact=True)
        reference = reference["dual"]
        candidate, new_proof = construct(adapter, prefill, capture, positions)
        if [f.component_bytes for f in candidate] != [f.component_bytes for f in reference]:
            raise RuntimeError("direct constructor changed attainable format storage")
        original = standard_greedy_reference(adapter, prefill, positions)
        old, old_record = run(adapter, prefill, reference, audit=True)
        new, new_record = run(adapter, prefill, candidate, audit=True)
        if not torch.equal(old, original) or not torch.equal(new, original):
            raise RuntimeError("direct construction changed original greedy logits")
        for field in (
            "initial_active_kv_storage_bytes",
            "final_active_kv_storage_bytes",
            "peak_active_kv_storage_bytes",
        ):
            if old_record["storage"][field] != new_record["storage"][field]:
                raise RuntimeError(f"direct constructor changed {field}")
        control = dict(
            method=method,
            bitwise_original_logits=True,
            old=old_record,
            new=new_record,
            selection=selection,
            reference_proof=old_proof,
            direct_proof=new_proof,
        )
        write_once(out / f"{method}-constructor-control.json", control)
        torch.save(
            dict(original=original, reference=old, candidate=new), out / f"{method}-logits.pt"
        )
        controls.append(control)
        expected_cost = [f.component_bytes for f in reference]
        del candidate, reference
        generator = random.Random(plan["timing_seed"])
        for trial in range(-plan["constructor_warmups"], plan["constructor_repetitions"]):
            order = ["reference", "direct"]
            generator.shuffle(order)
            for order_index, variant in enumerate(order):
                gc.collect()
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                started = time.monotonic()
                start.record()
                if variant == "reference":
                    formats, _ = reference_choices(
                        adapter, prefill, capture, positions, compact=True
                    )
                    formats = formats["dual"]
                else:
                    formats, _ = construct(adapter, prefill, capture, positions)
                end.record()
                torch.cuda.synchronize()
                wall = time.monotonic() - started
                if [f.component_bytes for f in formats] != expected_cost:
                    raise RuntimeError("repeated constructor changed format selection/storage")
                row = dict(
                    method=method,
                    trial=trial,
                    warmup=trial < 0,
                    variant=variant,
                    order_index=order_index,
                    construction_wall_seconds=wall,
                    construction_cuda_ms=start.elapsed_time(end),
                    three_repeated_original_KV_checks_retained=True,
                    excludes_prefill_selection_decode_artifact_IO=True,
                )
                write_once(out / f"trial-{trial}-{method}-{variant}.json", row)
                trials.append(row)
                del formats
    return [
        dict(
            method="direct_constructor_controls",
            constructor_controls=controls,
            trials=trials,
            lookm_excluded_as_source_ineligible=True,
            not_end_to_end=True,
        )
    ]


def analyze(root, expected):
    import numpy as np

    paired.original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    if gate["failures"]:
        return
    summaries = []
    for row in gate["controls"]:
        for method in METHODS:
            trials = [t for t in row["trials"] if t["method"] == method and not t["warmup"]]
            old = {
                t["trial"]: t["construction_wall_seconds"]
                for t in trials
                if t["variant"] == "reference"
            }
            new = [t for t in trials if t["variant"] == "direct"]
            if len(old) != 20 or len(new) != 20:
                raise ValueError("twenty complete paired constructor trials required")
            ratio = np.array([t["construction_wall_seconds"] / old[t["trial"]] for t in new])
            boot = np.median(
                ratio[np.random.default_rng(20261030).integers(0, 20, size=(10000, 20))], axis=1
            )
            summaries.append(
                dict(
                    task=row["task"],
                    method=method,
                    reference_p50_seconds=float(np.median(list(old.values()))),
                    direct_p50_seconds=float(
                        np.median([t["construction_wall_seconds"] for t in new])
                    ),
                    direct_p95_seconds=float(
                        np.percentile([t["construction_wall_seconds"] for t in new], 95)
                    ),
                    paired_median_ratio=float(np.median(ratio)),
                    ratio_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                )
            )
    out = root / "constructor_analysis"
    out.mkdir()
    write_once(
        out / "summary.json",
        dict(
            summaries=summaries,
            independent_quality_established=False,
            paper_campaign_authorized=False,
            whole_request_peak_certified=False,
            not_end_to_end=True,
        ),
    )
    write_once(
        out / "manifest.json",
        dict(
            source_sha=base.common.inputs(root, expected)["source_sha"],
            plan_sha256=expected,
            source_gate_sha256=sha(root / "analysis/GATE.json"),
            outputs={"summary.json": sha(out / "summary.json")},
        ),
    )


if __name__ == "__main__":
    paired.install()
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_constructor_study.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, study, analyze
    base.main()
