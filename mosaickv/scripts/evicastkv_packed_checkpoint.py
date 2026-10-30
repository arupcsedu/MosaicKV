#!/usr/bin/env python3
"""Bounded consumed-input checkpoint decode optimization; no selector change."""

from pathlib import Path

import evicastkv_encoded_projection as encoded

from mosaickv.evaluation.fox_campaign import sha, write_once

base = encoded.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/packed-checkpoint-consumed-20261030-v1")
VALIDATED = Path("/scratch/djy8hg/runs/evicastkv/encoded-projection-consumed-20261030-v1")
original_analyze = base.analyze


def prepare(root, phase):
    writer = base.write_once

    def register(path, value):
        if path.name == "plan.json":
            for stage in ("compact", "timing"):
                for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
                    bound = VALIDATED / stage / name
                    value["bound_files"][str(bound)] = sha(bound)
            value.update(
                protocol="packed_literal_counts_no_extra_bytes_v1",
                retention_one_format="packed",
                encoded_source_controls=True,
                timed_formats=["kv", "encoded", "packed"],
                packed_checkpoint_bytes_per_1024=16,
                evicast_initial_cap_bytes=62 * 2**20,
                evicast_peak_cap_bytes=96 * 2**20,
                exact_storage_equality_required=True,
                acceptance_latency_ratio_upper_bound=1.0,
                no_selector_or_benchmark_change=True,
                automatic_baseline_extension_authorized=False,
            )
        writer(path, value)

    base.write_once = register
    try:
        encoded.dual.original_prepare(root, phase)
    finally:
        base.write_once = writer
    return sha(root / "plan.json")


def choices(*args, **kwargs):
    from mosaickv.evaluation.dual_projection import DualFormat
    from mosaickv.evaluation.encoded_projection import build

    variants, proofs = encoded.choices(*args, **kwargs)
    old_dual = variants.pop("dual")
    packed = []
    for layer, (format, ordinary) in enumerate(zip(old_dual, variants["encoded"], strict=True)):
        if isinstance(format, DualFormat):
            candidate, record = build(
                format, args[0]._language_layers()[layer].self_attn, packed_literal_counts=True
            )
            if candidate.component_bytes != ordinary.component_bytes:
                raise RuntimeError("packed count checkpoint changed exact component bytes")
            proofs[layer]["packed_checkpoint"] = record
            packed.append(candidate)
        else:
            packed.append(ordinary)
    return dict(variants, packed=tuple(packed)), proofs


def analyze(root, expected):
    import numpy as np

    original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    if gate["failures"] or gate["phase"] != "timing":
        return
    compact = base.common.read(root.parent / "compact/analysis/GATE.json")
    storage = []
    timing = []
    for task in sorted({r["task"] for r in compact["controls"]}):
        for role in sorted({r["role"] for r in compact["controls"]}):
            rows = {
                r["format"]: r["storage"]
                for r in compact["controls"]
                if r["task"] == task and r["role"] == role
            }
            fields = (
                "initial_active_kv_storage_bytes",
                "final_active_kv_storage_bytes",
                "peak_active_kv_storage_bytes",
            )
            storage.append(
                dict(
                    task=task,
                    role=role,
                    encoded=rows["encoded"],
                    packed=rows["packed"],
                    exact_boundary_equality=all(
                        rows["encoded"][f] == rows["packed"][f] for f in fields
                    ),
                )
            )
            trials = [
                r
                for r in gate["controls"]
                if r["task"] == task and r["role"] == role and not r["warmup"]
            ]
            old = {r["trial"]: r["decode_wall_seconds"] for r in trials if r["format"] == "encoded"}
            new = {r["trial"]: r["decode_wall_seconds"] for r in trials if r["format"] == "packed"}
            if len(old) != 20 or set(old) != set(new):
                raise ValueError("twenty identical timing blocks required")
            ratios = np.array([new[i] / old[i] for i in sorted(old)])
            boot = np.median(
                ratios[np.random.default_rng(20261030).integers(0, 20, size=(10000, 20))], axis=1
            )
            ci = np.percentile(boot, [2.5, 97.5]).tolist()
            timing.append(
                dict(
                    task=task,
                    role=role,
                    paired_ratio=float(np.median(ratios)),
                    ratio_ci95=ci,
                    no_slower_pass=ci[1] <= 1.0,
                )
            )
    write_once(
        root / "analysis/packed_checkpoint_decision.json",
        dict(
            storage=storage,
            timing=timing,
            engineering_eligible=all(r["exact_boundary_equality"] for r in storage)
            and all(r["no_slower_pass"] for r in timing),
            no_general_quality_or_end_to_end_claim=True,
            no_automatic_followup_or_fresh_campaign=True,
        ),
    )


if __name__ == "__main__":
    encoded.dual.install()
    base.BASE = ROOT
    base.PHASES = ("compact", "timing")
    base.COMPACT_PARENT = VALIDATED / "compact"
    base.SCRIPT_NAME = "evicastkv_packed_checkpoint.sbatch"
    base.prepare, base.choices, base.analyze = prepare, choices, analyze
    base.main()
