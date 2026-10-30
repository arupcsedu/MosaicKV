#!/usr/bin/env python3
"""Validate eligible encoded-source transport identically across primary baselines."""

from pathlib import Path

import evicastkv_batched_history_study as history

from mosaickv.evaluation.fox_campaign import sha

base = history.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/encoded-baselines-consumed-20261030-v1")
VALIDATED = Path("/scratch/djy8hg/runs/evicastkv/encoded-projection-consumed-20261030-v1")


def prepare(root, phase):
    decision = base.common.read(VALIDATED / "timing/analysis/encoded_source_decision.json")
    if not decision["engineering_eligible"]:
        raise ValueError("encoded projection must first pass its numerical/memory/latency controls")
    original = history.original_writer

    def register(path, value):
        if path.name == "plan.json":
            for stage in ("compact", "timing"):
                for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
                    value["bound_files"][str(VALIDATED / stage / name)] = sha(
                        VALIDATED / stage / name
                    )
            path_decision = VALIDATED / "timing/analysis/encoded_source_decision.json"
            value["bound_files"][str(path_decision)] = sha(path_decision)
            value.update(
                protocol="all_primary_encoded_source_and_history_controls_v1",
                encoded_source_controls=True,
                common_format_choice="validated_dual_minimum_then_encoded_source_where_eligible",
                maximum_source_metadata_ratio=1.02,
                no_selection_or_answer_change=True,
                full_position_rows_are_lossless_controls_not_ordinary_FullKV_memory=True,
            )
        original(path, value)

    history.original_writer = register
    try:
        return history.prepare(root, phase)
    finally:
        history.original_writer = original


def fused_choices(adapter, prefill, capture, positions, *, compact):
    from mosaickv.evaluation.direct_constructor import construct
    from mosaickv.evaluation.dual_projection import DualFormat
    from mosaickv.evaluation.encoded_projection import build
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    del compact
    original, proof = construct(adapter, prefill, capture, positions)
    formats = []
    for layer, format in enumerate(original):
        if isinstance(format, DualFormat):
            encoded, record = build(format, adapter._language_layers()[layer].self_attn)
            proof[layer]["encoded_source"] = record
            formats.append(encoded)
        else:
            proof[layer]["encoded_source"] = dict(
                eligible=False, reason="unchanged non-Dual representation"
            )
            formats.append(format)
    old = backing_bytes(tuple(t for f in original for t in f.tensors()))
    new = backing_bytes(tuple(t for f in formats for t in f.tensors()))
    if new > old * 1.02 or (old <= 62 * 2**20 and new > 62 * 2**20):
        raise RuntimeError("encoded source changed registered metadata/initial cap")
    return {"common_encoded": tuple(formats)}, proof


if __name__ == "__main__":
    history.paired.install()
    history.fused_choices = fused_choices
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_encoded_baselines.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, history.study, history.analyze
    base.main()
