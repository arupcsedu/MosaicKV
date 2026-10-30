#!/usr/bin/env python3
"""Two consumed inputs, lossless dual-projection controls then repeated timing."""

from pathlib import Path

import evicastkv_storage_successor as base

from mosaickv.evaluation.fox_campaign import sha

BASE = Path("/scratch/djy8hg/runs/evicastkv/dual-projection-consumed-20261030-v1-retry2")
original_prepare = base.prepare
original_choices = base.choices


def prepare(root, phase):
    # Additions are made before the first immutable plan write, never afterwards.
    original_writer = base.write_once

    def registered_writer(path, value):
        if path.name == "plan.json":
            value.update(
                protocol="exact_corrected_dual_projection_v1",
                retention_one_format="dual",
                timed_formats=["kv", "compact", "dual"],
                dual_correction="original_BF16_bits_sparse_int32_indices",
                no_selector_change=True,
            )
        original_writer(path, value)

    base.write_once = registered_writer
    try:
        original_prepare(root, phase)
    finally:
        base.write_once = original_writer
    return sha(root / "plan.json")


def choices(adapter, prefill, capture, positions, *, compact):
    import torch

    from mosaickv.evaluation.dual_projection import build_dual
    from mosaickv.evaluation.exact_storage import formats_for_layer

    variants, proofs = original_choices(adapter, prefill, capture, positions, compact=compact)
    variants.pop("token")
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    dual = []
    for layer, ((key, value), heads) in enumerate(zip(pairs, positions, strict=True)):
        attention = adapter._language_layers()[layer].self_attn
        candidates, _ = formats_for_layer(capture.records[layer], attention, key, value, heads)
        index = torch.tensor(heads, device=key.device)[None, :, :, None].expand(
            1, key.shape[1], len(heads[0]), key.shape[-1]
        )
        references = tuple(t.gather(2, index) for t in (key, value))
        options = [variants["compact"][layer]]
        records = []
        for candidate in candidates:
            if candidate.archive is not None:
                built, record = build_dual(candidate, attention, references)
                records.append(record)
                options.append(built)
        # Storage-only choice; never use timing or benchmark answers to choose.
        selected = min(options, key=lambda f: (f.component_bytes, f.kind))
        dual.append(selected)
        proofs[layer]["dual_candidates"] = records
        proofs[layer]["chosen_dual"] = selected.kind
    return dict(variants, dual=tuple(dual)), proofs


def install():
    base.BASE = BASE
    base.PHASES = ("compact", "timing")
    base.SCRIPT_NAME = "evicastkv_dual_projection.sbatch"
    base.COMPACT_PARENT = Path(
        "/scratch/djy8hg/runs/evicastkv/quality-latency-compact-direct-20261030-v2/compact"
    )
    base.choices = choices
    base.prepare = prepare


if __name__ == "__main__":
    install()
    base.main()
