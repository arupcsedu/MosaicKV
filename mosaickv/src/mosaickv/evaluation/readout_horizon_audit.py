"""Evaluation-only horizon feasibility on fixed, consumed-input candidates.

Oracle FullKV future readouts here MUST NOT be imported by online selection.
No answers, captions, task scores, references or benchmark labels are used.
"""


def forced_plan(records, ordinal):
    if type(ordinal) is not int or not 0 <= ordinal < len(records) <= 10:
        raise ValueError("one registered existing candidate ordinal required")
    return dict(records[ordinal], evaluation_only_forced_plan=True)


def audit(
    reference,
    candidates,
    horizons=(1, 2, 3, 7),
    *,
    guard_predictions=32,
    semantic_predictions=128,
    guard_reference=None,
):
    guard_reference = reference if guard_reference is None else guard_reference
    if (
        reference.ndim != 2
        or len(reference) != 128
        or not candidates
        or "identity" not in candidates
        or type(semantic_predictions) is not int
        or not 2 <= semantic_predictions <= 128
        or any(t.shape != reference.shape for t in candidates.values())
        or guard_reference.shape != reference.shape
        or any(type(h) is not int or not 1 <= h < guard_predictions for h in horizons)
        or any(
            not bool(t.isfinite().all()) for t in (reference, guard_reference, *candidates.values())
        )
    ):
        raise ValueError("finite complete 128-prediction reference/identity/plans required")
    target = reference.double().log_softmax(-1)
    probability = target.exp()
    guard_target = guard_reference.double().log_softmax(-1)
    rows = {}
    for name, logits in candidates.items():
        candidate = logits.double()
        candidate_log = candidate.log_softmax(-1)
        guard_kl = (guard_target.exp() * (guard_target - candidate_log)).sum(-1)
        kl = (probability * (target - candidate.log_softmax(-1))).sum(-1)
        matches = candidate.argmax(-1) == reference.argmax(-1)
        divergence = (~matches).nonzero().flatten()
        # The first mismatching prediction still has identical input tokens;
        # later predictions are not at a comparable autoregressive prefix.
        engineering_support = int(divergence[0]) + 1 if len(divergence) else 128
        support = min(engineering_support, semantic_predictions)
        rows[name] = dict(
            shared_prefix_predictions=support,
            engineering_shared_prefix_predictions=engineering_support,
            full_token_agreement=float(matches.double().mean()),
            all_128_tokens_equal=bool(matches.all()),
            kl_by_prediction=kl[:support].tolist() + [None] * (128 - support),
            absolute_32_token_guard_passed=(
                bool(matches[:guard_predictions].all())
                and float(kl[:guard_predictions].max()) <= 0.02
                and float(
                    (candidate[:guard_predictions] - reference[:guard_predictions].double())
                    .abs()
                    .max()
                )
                <= 1.0
            ),
            registered_32_token_guard_passed=(
                bool(
                    (
                        candidate[:guard_predictions].argmax(-1)
                        == guard_reference[:guard_predictions].argmax(-1)
                    ).all()
                )
                and float(guard_kl[:guard_predictions].max()) <= 0.02
                and float(
                    (candidate[:guard_predictions] - guard_reference[:guard_predictions].double())
                    .abs()
                    .max()
                )
                <= 1.0
            ),
        )
    selected = {}
    for h in horizons:
        eligible = [
            name
            for name, r in rows.items()
            if r["shared_prefix_predictions"] > h and r["registered_32_token_guard_passed"]
        ]
        if not eligible:
            selected[str(h)] = dict(eligible=False)
            continue
        # Prediction zero is the unchanged prefill, never count it as evidence.
        name = min(eligible, key=lambda key: (sum(rows[key]["kl_by_prediction"][1 : h + 1]), key))
        r, identity = rows[name], rows["identity"]
        support = min(r["shared_prefix_predictions"], identity["shared_prefix_predictions"])
        selected[str(h)] = dict(
            eligible=True,
            selected=name,
            meaningful=name != "identity",
            oracle_selection_mean_kl=sum(r["kl_by_prediction"][1 : h + 1]) / h,
            later_shared_prefixes=max(0, support - h - 1),
            later_kl_difference=(
                sum(
                    a - b
                    for a, b in zip(
                        r["kl_by_prediction"][h + 1 : support],
                        identity["kl_by_prediction"][h + 1 : support],
                        strict=True,
                    )
                )
                / (support - h - 1)
                if support > h + 1
                else None
            ),
            all_128_tokens_equal=r["all_128_tokens_equal"],
            evaluation_only_not_an_online_policy=True,
        )
    return dict(
        candidate_diagnostics=rows,
        horizons=selected,
        benchmark_labels_accessed=False,
        oracle_future_logits_evaluation_only=True,
        superiority_established=False,
        new_cohort_authorized=False,
        semantic_prediction_limit=semantic_predictions,
    )


def eos_boundary(reference, answer, eos_ids):
    """Validate the recorded first-EOS caption boundary against actual logits."""
    ids = reference.argmax(-1).tolist()
    if len(ids) != 128 or not eos_ids:
        raise ValueError("complete generated prediction IDs and pinned model EOS IDs required")
    stops = [i for i, token in enumerate(ids) if token in eos_ids]
    if answer["stopped_by_eos"]:
        count = answer["answer_tokens"]
        if not stops or stops[0] != count or ids[:count] != answer["token_ids"]:
            raise ValueError("reported caption/EOS boundary disagrees with saved model logits")
        return count + 1
    if stops or answer["answer_tokens"] != 128 or ids != answer["token_ids"]:
        raise ValueError("full-length caption disagrees with saved model logits")
    return 128
