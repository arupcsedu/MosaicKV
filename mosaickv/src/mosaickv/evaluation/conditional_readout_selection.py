"""Bounded conditional-head readout selection, NOT global FullKV or oracle logits.

Uses only original prefill K/V and the first actual disposable draft query.
The final frozen layer is evaluated exactly. No model weights are changed.
"""

import math
from time import perf_counter

import torch

from mosaickv.evaluation.exact_last_layer_readout import (
    _cuda_copy,
    _snapshot,
    attention_output,
    compare_logits,
    suffix,
)


def revised_ids(exact, donor, receiver, protected, members, length):
    exact = tuple(exact)
    if (
        tuple(sorted(set(exact))) != exact
        or donor not in exact
        or type(receiver) is not int
        or not 0 <= receiver < length
        or receiver in set(exact) | set(protected) | set(members)
        or donor in set(protected) | set(members)
    ):
        raise ValueError("equal-capacity substitution of unprotected disjoint positions required")
    return tuple(sorted((set(exact) - {donor}) | {receiver}))


def shortlist(data, original, *, head, exact, receivers, protected, members, maximum=4):
    """CPU-only attention-output surrogate SHORTLIST, never final approval."""
    if type(maximum) is not int or not 1 <= maximum <= 4:
        raise ValueError("at most four equal-capacity pairs before actual suffix ranking")
    if len(receivers) > 16 or len(set(receivers)) != len(receivers):
        raise ValueError("bounded distinct graph/utility receiver pool required")
    q = data["query"][0, head].float() * data["scaling"]
    active = tuple(data[k][0, head].float() for k in ("active_key", "active_value"))
    parent = tuple(data[k][0, 0].float() for k in ("parent_key", "parent_value"))
    full = tuple(t.float() for t in original)
    n = len(full[0])
    mask = data["mask"]
    mask = 0.0 if mask is None else mask.reshape(-1, active[0].shape[0])[-1:].float()

    def output(pair):
        return (q @ pair[0].T + mask).softmax(-1) @ pair[1]

    target_pair = tuple(t.clone() for t in active)
    for t, source in zip(target_pair, full, strict=True):
        t[:n].copy_(source)
    target = output(target_pair)
    choices = []
    for receiver in receivers:
        for donor in exact:
            ids = revised_ids(exact, donor, receiver, protected, members, n)
            pair = tuple(t.clone() for t in active)
            for t, source, old in zip(pair, full, parent, strict=True):
                t[donor] = old[donor]
                t[receiver] = source[receiver]
            error = float((output(pair) - target).square().mean())
            if not math.isfinite(error):
                raise ValueError("finite label-free shortlist error required")
            choices.append((error, ids, donor, receiver))
    return tuple(sorted(choices)[:maximum])


def divergence(target, candidate):
    target_log = target.double().log_softmax(-1)
    return float((target_log.exp() * (target_log - candidate.double().log_softmax(-1))).sum())


def choose(records):
    """No token changes at the observed prefix; identity wins non-improving ties."""
    if not records or records[0]["name"] != "identity":
        raise ValueError("explicit unchanged anchor must be first")
    anchor = records[0]
    if any(not math.isfinite(r["kl"]) for r in records):
        raise ValueError("finite actual frozen-suffix readouts required")
    eligible = [r for r in records if r["greedy_token"] == anchor["greedy_token"]]
    best = min(eligible, key=lambda r: (r["kl"], r["ordinal"]))
    tolerance = 64 * torch.finfo(torch.float64).eps * max(1.0, abs(anchor["kl"]))
    return best if best["kl"] < anchor["kl"] - tolerance else anchor


@torch.inference_mode()
def select(adapter, capture, overlay, original_prompt, reference, *, mode):
    if mode not in ("exact", "repair", "joint"):
        raise ValueError("explicit exact/repair/joint conditional-readout variant required")
    n, head = len(original_prompt[0]), overlay.head
    report = capture.validate(n)
    data = capture.data
    layer = adapter._language_layers()[-1]
    norm = adapter.model.model.language_model.norm
    output_head = adapter.model.get_output_embeddings()
    versions = tuple(p._version for p in adapter.model.parameters())
    if adapter.model.training or layer.self_attn.layer_idx != 31:
        raise NotImplementedError("frozen actual final layer required")
    torch.cuda.synchronize()
    started = perf_counter()
    q = _cuda_copy(data["query"], data["query_stride"])
    key = _cuda_copy(data["active_key"], data["active_key_stride"])
    value = _cuda_copy(data["active_value"], data["active_value_stride"])
    residual = _cuda_copy(data["layer_residual"])
    mask = None if data["mask"] is None else _cuda_copy(data["mask"])
    full = tuple(t.to("cuda") for t in original_prompt)

    def predict():
        return suffix(
            layer,
            norm,
            output_head,
            residual,
            attention_output(q, key, value, mask, data["scaling"]),
        )

    baseline = predict()
    control = compare_logits(baseline, reference)
    if not control["logits_bitwise"]:
        raise RuntimeError("actual last-layer readout identity failed; do not rank candidates")
    for t, source in zip((key, value), full, strict=True):
        t[0, head, :n] = source
    conditional = predict()
    exact = tuple(overlay.exact_ids.cpu().tolist())
    members = tuple(overlay.positions.cpu().tolist())
    protected = tuple(overlay.evidence["original_protected_positions"])
    proposals = (
        shortlist(
            data,
            original_prompt,
            head=head,
            exact=exact,
            receivers=overlay.evidence["readout_receiver_shortlist"],
            protected=protected,
            members=members,
        )
        if mode != "repair"
        else ()
    )
    plans = [("identity", exact, False)]
    if mode != "exact":
        plans.append(("restore_members", exact, True))
    for i, (_, ids, _, _) in enumerate(proposals):
        plans.append((f"swap_{i}", ids, False))
        if mode == "joint":
            plans.append((f"swap_{i}_restore_members", ids, True))
    if len(plans) > 10:
        raise RuntimeError("registered ten-suffix bound exceeded")
    records, predictions = [], []
    for ordinal, (name, ids, restore) in enumerate(plans):
        for t, active, old, source in zip(
            (key, value),
            (data["active_key"], data["active_value"]),
            (data["parent_key"], data["parent_value"]),
            full,
            strict=True,
        ):
            t.copy_(active)
            for removed in set(exact) - set(ids):
                t[0, head, removed] = old[0, 0, removed].to("cuda")
            t[0, head, list(ids)] = source[list(ids)]
            if restore:
                t[0, head, list(members)] = source[list(members)]
        logits = predict()
        predictions.append(_snapshot(logits))
        records.append(
            dict(
                name=name,
                ordinal=ordinal,
                exact_ids=ids,
                restore=restore,
                kl=divergence(conditional, logits),
                greedy_token=int(logits.argmax()),
            )
        )
    selected = choose(records)
    revised = selected["exact_ids"]
    if len(revised) != len(exact) or set(revised) & (set(members) | set(protected)):
        raise RuntimeError("readout selection violated exact capacity or protected memberships")
    before_addresses = tuple(t.data_ptr() for t in overlay.gpu_tensors())
    overlay.exact_ids.copy_(torch.tensor(revised))
    for t, source in zip(overlay.exact_pair, full, strict=True):
        t.copy_(source[list(revised)])
    overlay.evidence.update(readout_seed_extra=list(exact), selected_extra=list(revised))
    if tuple(t.data_ptr() for t in overlay.gpu_tensors()) != before_addresses:
        raise RuntimeError("persistent GPU allocation changed")
    if versions != tuple(p._version for p in adapter.model.parameters()):
        raise RuntimeError("frozen model weights changed")
    torch.cuda.synchronize()
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    report.update(
        mode=mode,
        identity_control=control,
        plans=records,
        selected=selected,
        meaningful_exact_change=revised != exact,
        requested_residual_restoration=selected["restore"],
        benchmark_labels_accessed=False,
        global_fullkv_logits_used=False,
        conditional_target="same_state_routed_head_original_prefill_only",
        prompt_query_surrogate_is_only_a_shortlist=True,
        actual_frozen_suffix_evaluations=len(plans) + 2,
        selection_wall_seconds=perf_counter() - started,
        persistent_storage_unchanged=True,
        temporary_gpu_tensor_backing_bytes=backing_bytes(
            (q, key, value, residual, *full, *(() if mask is None else (mask,)))
        ),
        temporary_workspace_not_omitted_from_peak_claim=True,
        whole_request_peak_certified=False,
    )
    data["conditional_target_logits"] = _snapshot(conditional)
    data["candidate_logits"] = torch.stack(predictions)
    return report, predictions[selected["ordinal"]]
