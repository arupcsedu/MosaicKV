"""Consumed-input HF engineering gate; no benchmark annotations or new cohort."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

from mosaickv.backends.hf_allocation_guard import decide_draft_stability, draft_row_causes
from mosaickv.backends.hf_attention_bias import (
    bias_cache_storage,
    install_attention_bias_cache,
    source_contract,
)
from mosaickv.backends.hf_draft_head_evidence import DraftAttentionRecorder
from mosaickv.backends.hf_head_substitution import shadow_logit_metrics
from mosaickv.backends.hf_runtime import _clone_decode_state
from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.prototypes.attention_compaction import CompactedHead, compact_head
from mosaickv.selection.draft_shortlist import meaningful_improvement


def construct_heads(source, evidence, positions, policy, required):
    """Fixed draft-mean key dictionary; independent protected AM reimplementation.

    This is NOT the frozen MosaicKV evidence selector. Benchmark labels and
    model-probe outcomes are not arguments. Fits run on CPU, original dtype kept.
    """
    import torch

    anchors, fitted, records = [], [], []
    deadline = time.perf_counter() + 600
    for i, ((keys, values), data) in enumerate(zip(source, evidence, strict=True)):
        old, new = [], []
        for head in range(keys.shape[1]):
            if time.perf_counter() > deadline:
                raise TimeoutError("fixed 600-second CPU construction allowance exceeded")
            p = data["prompt_probabilities"][head].double()
            scores = p.mean(0)
            scores[list(required)] = -float("inf")
            optional = torch.argsort(scores, descending=True, stable=True)[
                : positions - len(required)
            ]
            chosen = tuple(sorted((*required, *optional.tolist())))
            if len(chosen) != positions or len(set(chosen)) != positions:
                raise ValueError(
                    "dictionary cannot fit the mandatory entries and requested capacity"
                )
            idx = list(chosen)
            protected = torch.tensor([position in required for position in chosen])
            result = compact_head(
                keys[0, head, idx],
                values[0, head, idx],
                p[:, idx],
                data["target"][head],
                data["generated_mass"][head],
                data["generated_numerator"][head],
                protected,
                chosen,
                source_positions=keys.shape[-2],
                policy=policy,
                evidence_kind="draft_rollout",
            )
            old.append(
                CompactedHead(
                    keys[0, head, idx].clone(),
                    values[0, head, idx].clone(),
                    torch.zeros(positions),
                    chosen,
                    False,
                    {},
                )
            )
            new.append(result)
            records.append(
                dict(
                    layer=i,
                    kv_head=head,
                    positions=list(chosen),
                    construction_accepted=result.construction_accepted,
                    **result.diagnostics,
                )
            )
        anchors.append(tuple(old))
        fitted.append(tuple(new))
    return tuple(anchors), tuple(fitted), records


def logit_metrics(reference, logits, active):
    import torch

    logp, logq = torch.log_softmax(reference.float(), -1), torch.log_softmax(logits.float(), -1)
    return dict(
        logit_kl=float((logp.exp() * (logp - logq)).sum(-1).mean()),
        maximum_logit_difference=float((reference.float() - logits.float()).abs().max()),
        greedy_token_agreement=bool(torch.equal(reference.argmax(-1), logits.argmax(-1))),
        active_bytes=active,
        **shadow_logit_metrics(reference, logits),
    )


def diagnose_bias_transport(
    adapter,
    prepared,
    *,
    prompt_positions,
    cap_bytes,
    policy,
    generated_capacity=16,
    sink_tokens=4,
    event=lambda _row: None,
):
    import torch

    guard = DraftAllocationGuardConfig()  # Never replace/relax the registered 32-prefix guard.
    phases, saved, storage_rows = {}, {}, []
    device = adapter.device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def phase(name, operation, *, cpu_only=False):
        sync()
        start_event = end_event = None
        if device.type == "cuda" and not cpu_only:
            start_event, end_event = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start_event.record()
        started = time.perf_counter()
        event(dict(event="phase_start", phase=name))
        try:
            value = operation()
        except Exception as error:
            sync()
            event(
                dict(
                    event="phase_failure",
                    phase=name,
                    wall_seconds=time.perf_counter() - started,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        if end_event is not None:
            end_event.record()
        sync()
        phases[name] = dict(
            wall_seconds=time.perf_counter() - started,
            cuda_event_ms=start_event.elapsed_time(end_event) if start_event is not None else None,
        )
        event(dict(event="phase_complete", phase=name, **phases[name]))
        return value

    versions = tuple(p._version for p in adapter.model.parameters())
    prefill = phase("prefill", lambda: adapter.prefill(prepared, capture_queries=False))
    raw = adapter._legacy_layers(prefill.state.past_key_values)[0]
    n = prefill.state.active_cache_length
    required = (0, n - 1)  # Registered first_and_last policy, not synthetic stand-ins.
    if not 2 <= prompt_positions < n:
        raise ValueError("diagnostic requires a compressed dictionary fitting first/last tokens")
    original_logical = (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    )
    source = phase(
        "source_d2h",
        lambda: tuple((k.detach().cpu().clone(), v.detach().cpu().clone()) for k, v in raw),
    )
    full_heads = tuple(
        tuple(
            CompactedHead(
                k[0, h].clone(), v[0, h].clone(), torch.zeros(n), tuple(range(n)), False, {}
            )
            for h in range(k.shape[1])
        )
        for k, v in source
    )

    def install(rows, *, identity=False):
        # Full-size identity is a parity control, not a 28 MiB comparison row.
        parity_cap = sum(
            k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in raw
        )
        parity_cap += sum(
            k.shape[1] * guard.horizon * k.shape[-1] * k.element_size() * 2 for k, _ in raw
        )
        parity_cap += sum(
            (n * k.shape[1] + (n + guard.horizon) * layer.self_attn.config.num_attention_heads) * 4
            for (k, _), layer in zip(raw, adapter._language_layers(), strict=True)
        )
        return install_attention_bias_cache(
            adapter,
            prefill.state,
            rows,
            required_logical_positions=required,
            generated_capacity=guard.horizon if identity else generated_capacity,
            sink_tokens=sink_tokens,
            cap_bytes=parity_cap + 65536 if identity else cap_bytes,
            retention_one=identity,
        )

    full = _clone_decode_state(adapter, prefill.state)
    adapter.preallocate_decode_cache(full, guard.horizon)
    identity = phase("retention_one_state_construction", lambda: install(full_heads, identity=True))
    evidence_source = SimpleNamespace(
        layers=tuple(
            SimpleNamespace(sequence_length=n, kv_heads=k.shape[1], value=v) for k, v in raw
        ),
        next_decode_position=n,
    )
    recorder = DraftAttentionRecorder(adapter, evidence_source, horizon=guard.horizon)
    reference_logits, tokens, parity = {}, {}, []

    def draft():
        nonlocal full, identity
        history, token = [], prefill.next_token_id
        for step in range(1, guard.horizon + 1):
            tokens[step] = token.detach().clone()
            out = adapter.decode_one_token(
                token, full, capture_queries=False, capture_attentions=True
            )
            full = out.state
            zero = adapter.decode_one_token(token, identity, capture_queries=False)
            identity = zero.state
            reference_logits[step] = out.logits.detach().clone()
            saved.setdefault("zero_bias_retention_one", {})[step] = (
                zero.logits.detach().cpu().clone()
            )
            parity.append(
                dict(
                    step=step,
                    maximum_logit_difference=float(
                        (out.logits.float() - zero.logits.float()).abs().max()
                    ),
                    bitwise_equal=bool(torch.equal(out.logits, zero.logits)),
                    token_agreement=bool(torch.equal(out.next_token_id, zero.next_token_id)),
                )
            )
            update = step - 1
            if update < generated_capacity:
                history.append(n + update)
            else:
                slot = sink_tokens + (update - sink_tokens) % (generated_capacity - sink_tokens)
                history[slot] = n + update
            # Only the physical ring's deterministic position schedule is used
            # here. Probabilities and values come from the ACTUAL FullKV forward.
            schedule = SimpleNamespace(
                past_key_values=SimpleNamespace(
                    layers=tuple(
                        SimpleNamespace(generated_logical_positions=tuple(history)) for _ in raw
                    )
                )
            )
            recorder.observe(step, out, schedule)
            token = out.next_token_id
        return recorder.result()

    evidence = phase("fullkv_draft_and_zero_bias_retention_one", draft)
    saved["fullkv"] = {step: t.detach().cpu().clone() for step, t in reference_logits.items()}
    saved["draft_input_tokens"] = {step: t.detach().cpu().clone() for step, t in tokens.items()}
    event(dict(event="retention_one_parity", steps=parity))
    if not all(r["bitwise_equal"] and r["token_agreement"] for r in parity):
        raise RuntimeError("zero-bias retention-one parity failed; compaction not attempted")
    # Release full-size private parity states before constructing compact states.
    full, identity, full_heads = None, None, None
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        anchor, fitted, fits = phase(
            "cpu_value_and_mass_construction",
            lambda: construct_heads(source, evidence, prompt_positions, policy, required),
            cpu_only=True,
        )
    finally:
        torch.set_num_threads(old_threads)

    def probe(rows, role, anchor_metrics=None, *, early_reject=False):
        state = phase(f"{role}_state_h2d_construction", lambda: install(rows))
        measured, logits = [], {}
        initial_storage = bias_cache_storage(adapter, state)
        storage_rows.append(dict(role=role, step=0, **initial_storage))
        event(dict(event="probe_initial_storage", role=role, **initial_storage))

        def run():
            nonlocal state
            for step in range(1, guard.horizon + 1):
                out = adapter.decode_one_token(tokens[step], state, capture_queries=False)
                state = out.state
                storage = bias_cache_storage(adapter, state)
                storage_rows.append(dict(role=role, step=step, **storage))
                metrics = logit_metrics(
                    reference_logits[step], out.logits, storage["active_kv_backing_bytes"]
                )
                row = dict(
                    step=step,
                    checkpoint=step in guard.checkpoints,
                    anchor=metrics
                    if anchor_metrics is None
                    else anchor_metrics[step - 1]["candidate"],
                    candidate=metrics,
                )
                measured.append(row)
                event(dict(event="probe_prefix", role=role, storage=storage, **row))
                logits[step] = out.logits.detach().cpu().clone()
                if (
                    state.next_decode_position != n + step
                    or state.logical_sequence_length != n + step
                ):
                    raise RuntimeError("bias transport renumbered logical positions")
                if early_reject and draft_row_causes(row, guard):
                    break

        phase(f"{role}_decode", run)
        saved[role] = logits
        return measured

    anchor_metrics = probe(anchor, "anchor")
    clone = tuple(
        tuple(
            replace(
                row, keys=row.keys.clone(), values=row.values.clone(), log_bias=row.log_bias.clone()
            )
            for row in layer
        )
        for layer in anchor
    )
    clone_rows = probe(clone, "null_clone", anchor_metrics)
    candidate_rows = probe(fitted, "candidate", anchor_metrics, early_reject=True)
    # A-B-A: regather from the pristine source AFTER candidate execution.
    repacked = tuple(
        tuple(
            replace(
                row,
                keys=source[i][0][0, h, list(row.logical_positions)].clone(),
                values=source[i][1][0, h, list(row.logical_positions)].clone(),
            )
            for h, row in enumerate(layer)
        )
        for i, layer in enumerate(anchor)
    )
    repack_rows = probe(repacked, "null_repack_after_candidate", anchor_metrics)
    controls = dict()
    for name, rows in (("null_clone", clone_rows), ("null_repack_after_candidate", repack_rows)):
        controls[name] = dict(
            passed=len(rows) == 32
            and all(torch.equal(saved[name][step], saved["anchor"][step]) for step in range(1, 33)),
            steps=rows,
        )
    decision = decide_draft_stability(candidate_rows, guard, allow_early_rejection=True)
    meaningful = decision["accepted"] and meaningful_improvement(candidate_rows, guard)
    unchanged = all(
        torch.equal(k.cpu(), a) and torch.equal(v.cpu(), b)
        for (k, v), (a, b) in zip(raw, source, strict=True)
    )
    weights_unchanged = versions == tuple(p._version for p in adapter.model.parameters())
    logical_unchanged = original_logical == (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    )
    mask_scope_closed = adapter._packed_mask_dispatcher.active_masks is None
    engineering = (
        all(c["passed"] for c in controls.values())
        and unchanged
        and logical_unchanged
        and weights_unchanged
        and mask_scope_closed
    )
    return dict(
        classification="consumed_input_compaction_engineering_not_benchmark_quality",
        effective_method="attention_matching_protected_reimpl",
        source_contract=source_contract(),
        retention_one_parity=parity,
        zero_bias_retention_one_passed=True,
        controls=controls,
        same_byte_guard=decision,
        guard_steps=candidate_rows,
        original_guard=dict(
            horizon=32, checkpoints=[1, 4, 16, 32], absolute_tolerance=1e-7, relative_tolerance=1e-6
        ),
        meaningful_model_improvement=bool(meaningful),
        engineering_passed=bool(engineering),
        diagnostic_passed=bool(engineering and meaningful),
        construction_accepted_heads=sum(r["construction_accepted"] for r in fits),
        constructed_heads=len(fits),
        construction_records=fits,
        storage=storage_rows,
        prompt_positions=prompt_positions,
        source_prompt_positions=n,
        generated_capacity=generated_capacity,
        source_cache_unchanged=unchanged,
        source_weights_unchanged=weights_unchanged,
        mask_scope_closed=mask_scope_closed,
        cpu_construction_threads=1,
        source_logical_positions_unchanged=logical_unchanged,
        phase_timing=phases,
        fullkv_prefills=1,
        second_prefill_performed=False,
        benchmark_labels_accessed=False,
        new_identities=0,
        used_fullkv_fallback=False,
        fp64_kl_shadow_only=True,
        source_verification_cpu_bytes=sum(
            t.numel() * t.element_size() for pair in source for t in pair
        ),
        evidence_cpu_bytes=recorder.storage_bytes,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        fresh_cohort_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
    ), dict(
        logits=saved,
        evidence=evidence,
        constructed=[
            dict(
                layer=i,
                head=h,
                positions=row.logical_positions,
                keys=row.keys,
                original_keys=anchor[i][h].keys,
                values=row.values,
                log_bias=row.log_bias,
                original_values=anchor[i][h].values,
            )
            for i, layer in enumerate(fitted)
            for h, row in enumerate(layer)
        ],
    )
