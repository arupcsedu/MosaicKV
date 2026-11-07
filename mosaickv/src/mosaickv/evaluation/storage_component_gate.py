"""Fail-closed read-only comparisons for a consumed-input component study.

Never select a candidate from benchmark answers. A local logit improvement is
diagnostic evidence only, not independent quality validation or authorization
to launch another cohort. Native FullKV is a quality reference, not a latency
competitor. Every ablation retains the same actual overlay allocations.
"""

from __future__ import annotations

import math

import torch


def logit_diagnostics(reference, actual, horizon=32):
    if (
        reference.ndim != 2
        or reference.shape != actual.shape
        or type(horizon) is not int
        or not 1 <= horizon <= len(actual)
        or not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    ):
        raise ValueError("aligned finite complete logit trajectories required")
    reference, actual = reference.double(), actual.double()
    equal = reference.argmax(-1) == actual.argmax(-1)
    indices = (~equal).nonzero().flatten()
    first = int(indices[0]) if len(indices) else None
    # Prediction at first divergence is still conditioned on a shared prefix;
    # subsequent own-trajectory logits must NOT be called fixed-prefix KL.
    shared = min(horizon, len(actual) if first is None else first + 1)
    r, a = reference[:horizon], actual[:horizon]
    kl = (r.softmax(-1) * (r.log_softmax(-1) - a.log_softmax(-1))).sum(-1)
    return dict(
        maximum_kl=float(kl.max()),
        mean_kl=float(kl.mean()),
        maximum_logit_difference=float((r - a).abs().max()),
        token_agreement=float(equal[:horizon].double().mean()),
        full_trajectory_token_agreement=float(equal.double().mean()),
        first_divergent_prediction=first,
        shared_prefix_prediction_count=shared,
        shared_prefix_mean_kl=float(kl[:shared].mean()),
        shared_prefix_maximum_kl=float(kl[:shared].max()),
        fixed_prefix_for_entire_horizon=shared == horizon,
        evaluated_predictions=horizon,
    )


def storage_match(rows):
    fields = (
        "initial_encoded_bytes",
        "final_encoded_bytes",
        "reported_counter_peak_bytes",
    )
    if not rows or any(r.get("status") != "completed" for r in rows):
        raise ValueError("complete component rows required for storage matching")
    result = {}
    for name in fields:
        values = [r["detail"].get(name) for r in rows]
        if any(type(x) is not int or x < 1 for x in values):
            raise ValueError(f"invalid measured storage boundary: {name}")
        result[name] = dict(
            matched=len(set(values)) == 1,
            minimum=min(values),
            maximum=max(values),
            spread_bytes=max(values) - min(values),
        )
    return dict(
        boundaries=result,
        all_boundaries_matched=all(r["matched"] for r in result.values()),
        peak_kind="decode_owned_counter_not_independent_whole_request_audit",
    )


def stage_effect(full, ablated):
    """Positive means full's readout is closer; never a paper-quality gate."""
    keys = ("shared_prefix_mean_kl", "maximum_logit_difference", "token_agreement")
    if any(not math.isfinite(float(row[k])) for row in (full, ablated) for k in keys):
        raise ValueError("finite paired readout observations required")
    comparable = (
        full["fixed_prefix_for_entire_horizon"]
        and ablated["fixed_prefix_for_entire_horizon"]
        and full["evaluated_predictions"] == ablated["evaluated_predictions"]
    )
    return dict(
        full_minus_ablated_token_agreement=full["token_agreement"] - ablated["token_agreement"],
        ablated_minus_full_max_logit_error=ablated["maximum_logit_difference"]
        - full["maximum_logit_difference"],
        ablated_minus_full_shared_prefix_mean_kl=(
            ablated["shared_prefix_mean_kl"] - full["shared_prefix_mean_kl"] if comparable else None
        ),
        equal_shared_prefix_support=comparable,
        local_diagnostic_only=True,
        authorizes_fresh_evaluation=False,
    )


def precision_evidence(payload):
    """CPU-only, label-blind causal decomposition of one fixed overlay plan.

    This audits the constructor packet, not later decoder activations. All
    logical positions remain. Prefix holdout statistics are diagnostics, not an
    independent sample split, next-token evidence, or a source-retirement proof.
    """
    if not payload.get("evaluation_only") or payload.get("labels_accessed") is not False:
        raise ValueError("explicit label-free evaluation-only capture required")
    original = tuple(t.double() for t in payload["original_pair"])
    parent = tuple(t.double() for t in payload["parent_pair"])
    queries = payload["prompt_queries"].double()
    if (
        queries.device.type != "cpu"
        or queries.ndim != 2
        or len(queries) < 2
        or any(t.device.type != "cpu" or t.shape != original[0].shape for t in (*parent, *original))
        or any(not bool(torch.isfinite(t).all()) for t in (*original, *parent, queries))
    ):
        raise ValueError("aligned finite CPU queries and routed K/V captures required")
    length, dim = original[0].shape
    mask = torch.arange(length)[None] > torch.arange(length - len(queries), length)[:, None]

    def attend(pair):
        weights = ((queries @ pair[0].T) / dim**0.5).masked_fill(mask, float("-inf")).softmax(-1)
        return weights @ pair[1]

    target = attend(original)
    ids = payload["evidence"]["selected_extra"]
    members = payload["evidence"]["prototype_members"]
    if set(ids) & set(members):
        raise ValueError("exact/prototype memberships overlap")
    variants = {}
    for name, exact, proto, restore in (
        ("parent", False, False, False),
        ("extra_exact_only", True, False, False),
        ("prototype_only", False, True, False),
        ("joint", True, True, False),
        ("restored_members", True, False, True),
    ):
        pair = tuple(t.clone() for t in parent)
        if exact:
            for t, source in zip(pair, original, strict=True):
                t[ids] = source[ids]
        if proto:
            pair[0][members] = payload["overlay"]["member_keys"].double()
            pair[1][members] = payload["overlay"]["prototype_value"].double()
        if restore:
            for t, source in zip(pair, original, strict=True):
                t[members] = source[members]
        error = (attend(pair) - target).square().sum(-1)
        scale = target.square().sum(-1).clamp_min(1e-30)
        split = len(error) // 2
        variants[name] = dict(
            mean_squared_output_error=float(error.mean()),
            mean_relative_squared_output_error=float((error / scale).mean()),
            first_prefix_half_mean_error=float(error[:split].mean()),
            later_prefix_half_mean_error=float(error[split:].mean()),
        )
    return dict(
        variants=variants,
        evaluated_prompt_queries=len(queries),
        output_target="FP64_post_RoPE_causal_prompt_attention_on_original_KV",
        prefix_halves_are_not_independent_examples=True,
        generated_answer_quality_test=False,
        prototype_retires_parent_source_storage=False,
        replacement_gpu_authorized=False,
    )
