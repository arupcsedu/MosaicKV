"""CPU-only causal audit of fixed-prototype, label-free exact transactions.

Changes to exact selection must not silently change the prototype cluster.
Fit on early prompt queries; verify on later queries AND future-query proxy.
These are diagnostic prefix holdouts, not independent benchmark examples.
"""

from __future__ import annotations

import torch


def attend(queries, pair, *, causal):
    n, d = pair[0].shape
    score = queries.double() @ pair[0].double().T / d**0.5
    if causal:
        qpos = torch.arange(n - len(queries), n)
        score = score.masked_fill(torch.arange(n)[None] > qpos[:, None], -torch.inf)
    return score.softmax(-1) @ pair[1].double()


def metrics(queries, pair, original):
    result = {}
    for causal in (False, True):
        error = attend(queries, pair, causal=causal) - attend(queries, original, causal=causal)
        error = error.square().sum(-1)
        split = len(error) // 2
        result["causal" if causal else "future_proxy"] = dict(
            early=float(error[:split].mean()),
            late=float(error[split:].mean()),
            mean=float(error.mean()),
            maximum=float(error.max()),
        )
    return result


def weighted_prototype_trial(payload, *, return_payload=False):
    """One closed-form demand-weighted prototype; no alpha sweep or answer fitting."""
    from mosaickv.evaluation.residual_delta_prototype import construct
    from mosaickv.evaluation.storage_six_overlay import LayerOverlay

    original, parent = payload["original_pair"], payload["parent_pair"]
    queries, evidence = payload["prompt_queries"], payload["evidence"]
    members, ids = tuple(evidence["prototype_members"]), tuple(evidence["selected_extra"])
    dim = original[0].shape[-1]
    probability = (
        queries[: len(queries) // 2].double() @ original[0].double().T / dim**0.5
    ).softmax(-1)
    demand = probability[:, list(members)]
    mass = demand.sum(-1)
    weights = (mass[:, None] * demand).sum(0) / mass.square().sum().clamp_min(1e-30)
    weights = weights.clamp_min(1e-12)
    weights /= weights.sum()
    anchor = evidence["prototype_anchor_position"]
    pair, dispersion = construct(original, parent, payload["phases"], members, anchor, weights)
    phases = tuple(t[(*members, anchor), :] for t in payload["phases"])
    overlay = LayerOverlay(
        head=0,
        positions=members,
        exact_ids=ids,
        exact_pair=tuple(t[list(ids)] for t in original),
        prototype_pair=pair,
        phases=phases,
        residual_pair=tuple(t[list(members)] for t in original),
        anchor=anchor,
        dispersion=dispersion,
        evidence={},
    )
    before, after = tuple(t.clone() for t in parent), tuple(t.clone() for t in parent)
    for component, (a, b, full) in enumerate(zip(before, after, original, strict=True)):
        a[list(ids)], b[list(ids)] = full[list(ids)], full[list(ids)]
        old = (
            payload["overlay"]["member_keys"]
            if component == 0
            else payload["overlay"]["prototype_value"]
        )
        new = overlay.member_keys if component == 0 else overlay.prototype_pair[1]
        a[list(members)] += old
        b[list(members)] += new
    baseline, proposal = metrics(queries, before, original), metrics(queries, after, original)
    passed = all(
        proposal[k][h] <= baseline[k][h] + 1e-12
        for k in baseline
        for h in ("early", "late", "maximum")
    )
    strict = all(proposal[k]["mean"] < baseline[k]["mean"] for k in baseline)
    report = dict(
        weights=weights.tolist(),
        baseline=baseline,
        proposal=proposal,
        proxy_gate_passed=passed and strict,
        fixed_exact=list(ids),
        fixed_members=list(members),
        estimator="early_query_mass_weighted_residual_least_squares_closed_form",
        benchmark_answers_used=False,
        benchmark_quality_established=False,
    )
    if return_payload:
        report["replacement"] = dict(
            prototype_pair=pair,
            member_keys=overlay.member_keys,
            prototype_value=overlay.prototype_pair[1],
            dispersion=dispersion,
        )
    return report


def restoration_errors(queries, original, current, candidates):
    """Exact single-row correction formula, O(Q*N*D), finite FP64 control.

    Includes changed key scores and renormalization. Rare extreme scores use
    direct recomputation instead of trusting cancellation or overflow.
    """
    n, d = original[0].shape
    q = queries.double() / d**0.5
    k, v = (t.double() for t in current)
    ok, ov = (t.double() for t in original)
    positions = torch.arange(n - len(queries), n)
    mask = torch.arange(n)[None] > positions[:, None]
    probability = (q @ k.T).masked_fill(mask, -torch.inf).softmax(-1)
    output = probability @ v
    target = (q @ ok.T).masked_fill(mask, -torch.inf).softmax(-1) @ ov
    indices = torch.tensor(candidates, dtype=torch.long)
    p = probability[:, indices]
    delta = q @ (ok[indices] - k[indices]).T
    ratios = delta.clamp(-16, 16).exp()
    denominator = 1 + p * (ratios - 1)
    reconstructed = (
        output[:, None] + p[..., None] * (ratios[..., None] * ov[indices] - v[indices])
    ) / denominator[..., None]
    errors = (reconstructed - target[:, None]).square().sum(-1)
    extreme = ((delta.abs() > 16) | ((p > 1 - 1e-7) & (delta < -8))).any(0)
    for slot in extreme.nonzero().flatten().tolist():
        pair = tuple(t.clone() for t in current)
        for value, source in zip(pair, original, strict=True):
            value[candidates[slot]] = source[candidates[slot]]
        errors[:, slot] = (attend(queries, pair, causal=True) - target).square().sum(-1)
    if not bool(torch.isfinite(errors).all()):
        raise ValueError("nonfinite restoration-error estimate")
    return errors


@torch.inference_mode()
def plan_precision_transaction(payload, *, mode="joint_guarded", skip_discarded_exact=False):
    """Pure training-free online planner; original PREFILL data, not future oracles.

    Strict schema excludes benchmark answers, references and generated logits.
    The evaluation-only wrapper below sanitizes saved captures to this schema.
    """
    if set(payload) != {
        "original_pair",
        "parent_pair",
        "prompt_queries",
        "phases",
        "evidence",
        "overlay",
    }:
        raise ValueError("strict label-free precision input schema required")
    if mode not in ("exact", "prototype", "joint_guarded"):
        raise ValueError("registered precision transaction required")
    if type(skip_discarded_exact) is not bool or (skip_discarded_exact and mode != "prototype"):
        raise ValueError("only a discarded prototype-only exact branch may be skipped")
    if set(payload["evidence"]) != {
        "selected_extra",
        "prototype_members",
        "original_protected_positions",
        "prototype_anchor_position",
        "prototype_representation",
    } or set(payload["overlay"]) != {"member_keys", "prototype_value"}:
        raise ValueError("strict label-free precision metadata required")
    original, parent = payload["original_pair"], payload["parent_pair"]
    queries = payload["prompt_queries"]
    if (
        queries.device.type != "cpu"
        or len(queries) < 4
        or any(t.device.type != "cpu" for t in (*original, *parent))
    ):
        raise ValueError("CPU-only routed-head tensors and prompt query holdout required")
    evidence = payload["evidence"]
    ids = tuple(evidence["selected_extra"])
    members = tuple(evidence["prototype_members"])
    forbidden = set(evidence["original_protected_positions"]) | set(members)
    if set(ids) & forbidden:
        raise ValueError("baseline exact and mandatory/prototype memberships overlap")
    fixed = tuple(t.clone() for t in parent)
    corrections = payload["overlay"]["member_keys"], payload["overlay"]["prototype_value"]
    for value, correction in zip(fixed, corrections, strict=True):
        value[list(members)] = (
            value[list(members)] + correction
            if evidence["prototype_representation"] == "parent_residual_delta"
            else correction
        )
    baseline = tuple(t.clone() for t in fixed)
    for value, source in zip(baseline, original, strict=True):
        value[list(ids)] = source[list(ids)]
    # Keep original query positions for the causal mask by evaluating all
    # queries, but only EARLY scores may choose the next exact row.
    before = metrics(queries, baseline, original)
    if skip_discarded_exact:
        # In prototype-only mode exact_ok is always false. None of these
        # discarded candidate scores feeds the weighted prototype or final
        # joint veto. Keep its absence explicit, not fabricated diagnostics.
        selected, decisions, after, verified, strict = list(ids), [], before, False, False
    else:
        candidate = tuple(t.clone() for t in fixed)
        selected, decisions = [], []
        available = sorted(set(range(len(parent[0]))) - forbidden)
        split = len(queries) // 2
        for _ in ids:
            remaining = tuple(i for i in available if i not in selected)
            errors = restoration_errors(queries, original, candidate, remaining)
            slot = int(errors[:split].mean(0).argmin())
            position = remaining[slot]
            selected.append(position)
            decisions.append(
                dict(position=position, early_candidate_error=float(errors[:split, slot].mean()))
            )
            for value, source in zip(candidate, original, strict=True):
                value[position] = source[position]
        after = metrics(queries, candidate, original)
        verified = all(
            after[k][h] <= before[k][h] + 1e-12
            for k in before
            for h in ("early", "late", "maximum")
        )
        strict = all(after[k]["mean"] < before[k]["mean"] for k in before)
    report = dict(
        baseline_exact=list(ids),
        proposed_exact=selected,
        fixed_prototype_members=list(members),
        unchanged_exact_count=len(ids) == len(selected),
        mandatory_preserved=not bool(set(selected) & forbidden),
        baseline=before,
        proposal=after,
        early_only_decisions=decisions,
        proxy_gate_passed=verified and strict,
        prospective_rule="early_causal_marginal_then_late_and_future_proxy_veto",
        generated_logits_used=False,
        benchmark_answers_used=False,
        benchmark_quality_established=False,
        fresh_evaluation_authorized=False,
        discarded_exact_search_evaluated=not skip_discarded_exact,
    )
    weighted = weighted_prototype_trial(payload, return_payload=True)
    proposed_prototype = weighted.pop("replacement")
    report["query_weighted_prototype"] = weighted
    exact_ok = report["proxy_gate_passed"] and mode in ("exact", "joint_guarded")
    proto_ok = weighted["proxy_gate_passed"] and mode in ("prototype", "joint_guarded")
    chosen_ids = selected if exact_ok else list(ids)
    pair = tuple(t.clone() for t in fixed)
    if proto_ok:
        pair = tuple(t.clone() for t in parent)
        for value, correction in zip(
            pair,
            (proposed_prototype["member_keys"], proposed_prototype["prototype_value"]),
            strict=True,
        ):
            value[list(members)] += correction
    for value, source in zip(pair, original, strict=True):
        value[chosen_ids] = source[chosen_ids]
    joint = metrics(queries, pair, original)
    joint_ok = all(
        joint[k][h] <= before[k][h] + 1e-12 for k in before for h in ("early", "late", "maximum")
    )
    if not joint_ok:
        # Keep both individual vetoes AND the final transaction veto. Never
        # infer safe composition from two separately successful changes.
        exact_ok = proto_ok = False
        chosen_ids, joint = list(ids), before
    report.update(
        accepted_exact=exact_ok,
        accepted_prototype=proto_ok,
        accepted_meaningful_change=exact_ok or proto_ok,
        final_metrics=joint,
        returned_exact=sorted(chosen_ids),
        joint_proposal_rejected=not joint_ok,
        final_joint_guard_passed=True,
        replacement={
            **(proposed_prototype if proto_ok else {}),
            "exact_ids": sorted(chosen_ids),
        },
        fixed_prototype_members_preserved=True,
    )
    return report


def study(payload):
    if payload.get("evaluation_only") is not True or payload.get("labels_accessed") is not False:
        raise ValueError("label-free evaluation-only consumed capture required")
    inputs = {
        name: payload[name]
        for name in (
            "original_pair",
            "parent_pair",
            "prompt_queries",
            "phases",
            "evidence",
            "overlay",
        )
    }
    inputs["evidence"] = {
        name: payload["evidence"][name]
        for name in (
            "selected_extra",
            "prototype_members",
            "original_protected_positions",
            "prototype_anchor_position",
            "prototype_representation",
        )
    }
    inputs["overlay"] = {
        name: payload["overlay"][name] for name in ("member_keys", "prototype_value")
    }
    report = plan_precision_transaction(inputs)
    del report["replacement"]
    return report
