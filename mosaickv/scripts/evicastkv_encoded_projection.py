#!/usr/bin/env python3
"""Same consumed inputs, exact encoded-source projection -> paired timing."""

from pathlib import Path

import evicastkv_dual_projection as dual

from mosaickv.evaluation.fox_campaign import sha, write_once

base = dual.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/encoded-projection-consumed-20261030-v1")
original_analyze = base.analyze


def prepare(root, phase):
    previous_writer = base.write_once

    def register(path, value):
        if path.name == "plan.json":
            value.update(
                protocol="encoded_source_projection_no_decoded_buffer_v1",
                retention_one_format="encoded",
                encoded_source_controls=True,
                maximum_initial_metadata_ratio=1.02,
                evicast_initial_cap_bytes=62 * 2**20,
                evicast_peak_cap_bytes=96 * 2**20,
                acceptance_maximum_latency_ratio=1.05,
                timed_formats=["kv", "dual", "encoded"],
                no_selection_change=True,
                no_fresh_campaign=True,
            )
        previous_writer(path, value)

    base.write_once = register
    try:
        dual.original_prepare(root, phase)
    finally:
        base.write_once = previous_writer
    return sha(root / "plan.json")


def choices(*args, **kwargs):
    from mosaickv.evaluation.dual_projection import DualFormat
    from mosaickv.evaluation.encoded_projection import build
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    variants, proof = dual.choices(*args, **kwargs)
    variants.pop("compact")
    result = []
    adapter = args[0]
    for layer, format in enumerate(variants["dual"]):
        if isinstance(format, DualFormat):
            encoded, record = build(format, adapter._language_layers()[layer].self_attn)
            proof[layer]["encoded_source"] = record
            result.append(encoded)
        else:
            proof[layer]["encoded_source"] = dict(
                eligible=False, reason="non-Dual original format retained"
            )
            result.append(format)
    old = backing_bytes(tuple(t for f in variants["dual"] for t in f.tensors()))
    new = backing_bytes(tuple(t for f in result for t in f.tensors()))
    if new > old * 1.02 or (old <= 62 * 2**20 and new > 62 * 2**20):
        raise RuntimeError("checkpoint overhead exceeds registered metadata or initial cap")
    return dict(variants, encoded=tuple(result)), proof


def analyze(root, expected):
    import numpy as np

    original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    if gate["failures"] or gate["phase"] != "timing":
        return
    compact = base.common.read(root.parent / "compact/analysis/GATE.json")
    memory = []
    for task in sorted({r["task"] for r in compact["controls"]}):
        for role in sorted({r["role"] for r in compact["controls"]}):
            rows = {
                r["format"]: r["storage"]
                for r in compact["controls"]
                if r["task"] == task and r["role"] == role
            }
            old, new = rows["dual"], rows["encoded"]
            memory.append(
                dict(
                    task=task,
                    role=role,
                    dual=old,
                    encoded=new,
                    peak_reduced=new["peak_active_kv_storage_bytes"]
                    < old["peak_active_kv_storage_bytes"],
                )
            )
    timing = []
    for task in sorted({r["task"] for r in gate["controls"]}):
        for role in sorted({r["role"] for r in gate["controls"]}):
            rows = [
                r
                for r in gate["controls"]
                if r["task"] == task and r["role"] == role and not r["warmup"]
            ]
            old = {r["trial"]: r["decode_wall_seconds"] for r in rows if r["format"] == "dual"}
            new = [r for r in rows if r["format"] == "encoded"]
            if len(old) != 20 or len(new) != 20:
                raise ValueError("twenty complete paired trials required")
            ratio = np.array([r["decode_wall_seconds"] / old[r["trial"]] for r in new])
            boot = np.median(
                ratio[np.random.default_rng(20261030).integers(0, 20, size=(10000, 20))], axis=1
            )
            ci = np.percentile(boot, [2.5, 97.5]).tolist()
            timing.append(
                dict(
                    task=task,
                    role=role,
                    paired_median_ratio=float(np.median(ratio)),
                    ratio_ci95=ci,
                    latency_gate_passed=ci[1] <= 1.05,
                )
            )
    write_once(
        root / "analysis/encoded_source_decision.json",
        dict(
            memory=memory,
            timing=timing,
            engineering_eligible=all(r["peak_reduced"] for r in memory)
            and all(r["latency_gate_passed"] for r in timing),
            no_general_quality_or_end_to_end_claim=True,
            no_fresh_or_paper_campaign=True,
            whole_request_peak_certified=False,
        ),
    )


if __name__ == "__main__":
    dual.install()
    base.BASE = ROOT
    base.PHASES = ("compact", "timing")
    base.COMPACT_PARENT = Path(
        "/scratch/djy8hg/runs/evicastkv/dual-projection-consumed-20261030-v1-retry3/compact"
    )
    base.SCRIPT_NAME = "evicastkv_encoded_projection.sbatch"
    base.prepare, base.choices, base.analyze = prepare, choices, analyze
    base.main()
