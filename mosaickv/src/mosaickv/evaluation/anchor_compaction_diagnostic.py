"""Frozen V3 constructor: consumed dictionary, first-16 coefficient evidence.

No benchmark labels, logits, adapter, or model weights enter this constructor.
Synthetic values and mass biases are explicitly a reimplementation variant.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from mosaickv.prototypes.attention_compaction import AttentionCompactionPolicy, CompactedHead
from mosaickv.prototypes.constrained_compaction import ConstrainedValuePolicy, fit_component

METHOD = "attention_matching_anchor_preserving_reimpl"
VARIANT = "anchor_preserving_v3_joint"
FIT_PREFIXES = 16
VALUE_POLICY = ConstrainedValuePolicy(
    iterations=256, ridge_multiplier=10000.0, maximum_delta_norm_ratio=0.1
)


def construction_policy() -> dict[str, Any]:
    return dict(
        component="joint",
        coefficient_fit_prefixes=FIT_PREFIXES,
        value_policy=asdict(VALUE_POLICY),
        chunk_routes=64,
        dictionary="frozen_parent_39_positions",
        evidence_prefixes=32,
    )


def construct_anchor_heads(
    source: Any,
    evidence: Any,
    positions: int,
    core: AttentionCompactionPolicy,
    required: tuple[int, ...],
    *,
    fixed_dictionary: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Reuse exact positions; fit only the first 16 of 32 FullKV draft prefixes."""
    import torch

    if core != AttentionCompactionPolicy(enabled=True):
        raise ValueError("frozen V3 requires the unchanged original mass/value safety policy")
    if len(source) != len(evidence) or len(source) != len(fixed_dictionary):
        raise ValueError("V3 requires the complete frozen layer dictionary and evidence")
    records: list[dict[str, Any]] = []
    anchors: list[list[CompactedHead]] = []
    fitted: list[list[CompactedHead]] = []
    flat: list[tuple[int, int, CompactedHead, Any, Any]] = []
    for layer, ((keys, values), data, dictionary) in enumerate(
        zip(source, evidence, fixed_dictionary, strict=True)
    ):
        if (
            keys.device.type != "cpu"
            or values.device.type != "cpu"
            or keys.shape != values.shape
            or keys.ndim != 4
            or keys.shape[0] != 1
        ):
            raise ValueError("V3 construction requires original dense CPU K/V")
        if len(dictionary) != keys.shape[1] or data["prompt_probabilities"].shape != (
            keys.shape[1],
            32,
            keys.shape[-2],
        ):
            raise ValueError("frozen dictionary/head or 32-prefix evidence mismatch")
        anchors.append([])
        fitted.append([])
        for head, chosen in enumerate(dictionary):
            chosen = tuple(chosen)
            if (
                len(chosen) != positions
                or sorted(set(chosen)) != list(chosen)
                or any(type(p) is not int or not 0 <= p < keys.shape[-2] for p in chosen)
                or not set(required).issubset(chosen)
            ):
                raise ValueError(
                    "frozen dictionary violates capacity, positions or mandatory policy"
                )
            idx = list(chosen)
            original = values[0, head, idx].clone().contiguous()
            anchor = CompactedHead(
                keys[0, head, idx].clone().contiguous(),
                original,
                torch.zeros(positions, dtype=torch.float32),
                chosen,
                False,
                {},
            )
            anchors[layer].append(anchor)
            flat.append((layer, head, anchor, data, torch.tensor([p in required for p in chosen])))
    deadline = time.perf_counter() + 600
    for start in range(0, len(flat), 64):
        if time.perf_counter() > deadline:
            raise TimeoutError("frozen 600-second V3 CPU construction allowance exceeded")
        chunk = flat[start : start + 64]
        # Only selected dictionary columns and prefixes 1..16 enter fitting.
        a = torch.stack(
            [
                data["prompt_probabilities"][head][:FIT_PREFIXES, list(anchor.logical_positions)]
                for _, head, anchor, data, _ in chunk
            ]
        )
        v = torch.stack([anchor.values for _, _, anchor, _, _ in chunk])
        y = torch.stack([data["target"][head, :FIT_PREFIXES] for _, head, _, data, _ in chunk])
        h = torch.stack(
            [data["generated_mass"][head, :FIT_PREFIXES] for _, head, _, data, _ in chunk]
        )
        n = torch.stack(
            [data["generated_numerator"][head, :FIT_PREFIXES] for _, head, _, data, _ in chunk]
        )
        protected = torch.stack([p for *_, p in chunk])
        fit = fit_component(a, v, y, h, n, protected, "joint", core, VALUE_POLICY)
        delta = (fit.values.double() - v.double()).norm(dim=-1)
        original_norm = v.double().norm(dim=-1)
        if (
            (delta > 0.1 * original_norm).any()
            or (fit.values.double().norm(dim=-1) > 4 * original_norm).any()
            or not torch.equal(fit.values[protected], v[protected])
            or (fit.log_bias[protected] != 0).any()
        ):
            raise RuntimeError("V3 constructed coefficients violate frozen stored-dtype bounds")
        for j, (layer, head, anchor, _, _) in enumerate(chunk):
            details = dict(
                coefficient_fit_prefixes=FIT_PREFIXES,
                evidence_prefixes=32,
                training_mse_before=float(fit.diagnostics["training_mse_before"][j]),
                training_mse_after=float(fit.diagnostics["returned_training_mse"][j]),
                maximum_value_delta_norm_ratio=float(
                    (delta[j] / original_norm[j].clamp_min(1e-12)).max()
                ),
                maximum_value_norm_ratio=float(
                    (fit.values[j].double().norm(dim=-1) / original_norm[j].clamp_min(1e-12)).max()
                ),
                protected_values_unchanged=True,
                protected_bias_zero=True,
            )
            result = CompactedHead(
                anchor.keys.clone(),
                fit.values[j].clone().contiguous(),
                fit.log_bias[j].clone().contiguous(),
                anchor.logical_positions,
                bool(fit.admitted[j]),
                details,
            )
            fitted[layer].append(result)
            records.append(
                dict(
                    layer=layer,
                    kv_head=head,
                    positions=list(anchor.logical_positions),
                    construction_accepted=result.construction_accepted,
                    **details,
                )
            )
    return tuple(tuple(rows) for rows in anchors), tuple(tuple(rows) for rows in fitted), records
