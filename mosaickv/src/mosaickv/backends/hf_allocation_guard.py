"""V5: atomic, label-blind logit guard for a SAME-BYTE exact allocation.

This is model-generated online evidence, not the evaluation-only oracle API.
Passing a finite draft is not a guarantee of long-answer or benchmark quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.measurements.cache_lifecycle import CacheLifecycle


def packed_positions_sha(packed: Any) -> str:
    """Hash original logical positions, never infer them from physical slots."""
    grouped: list[list[list[tuple[int, int]]]] = [
        [[] for _ in range(int(layer.key.shape[1]))] for layer in packed.snapshot.layers
    ]
    for record in packed.slot_records:
        grouped[record["layer"]][record["kv_head"]].append(
            (record["slot"], record["logical_position"])
        )
    positions = [
        [[position for _, position in sorted(head)] for head in layer] for layer in grouped
    ]
    return hashlib.sha256(json.dumps(positions, separators=(",", ":")).encode()).hexdigest()


def decide_draft_stability(
    rows: Sequence[dict[str, Any]],
    policy: DraftAllocationGuardConfig,
) -> dict[str, Any]:
    """EVERY prefix must be non-worse; averaging cannot conceal a regression."""
    if len(rows) != policy.horizon or [r["step"] for r in rows] != list(
        range(1, policy.horizon + 1)
    ):
        raise ValueError("draft guard requires the entire registered horizon")
    rejected = []
    for row in rows:
        causes = []
        for metric in ("logit_kl", "maximum_logit_difference"):
            a, b = row["anchor"][metric], row["candidate"][metric]
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (a, b)
            ) or min(a, b) < (-1e-6 if metric == "logit_kl" else 0):
                raise ValueError("draft guard requires finite valid divergence")
            if b > a + policy.absolute_tolerance + policy.relative_tolerance * abs(a):
                causes.append(metric)
        for name in ("anchor", "candidate"):
            if type(row[name]["greedy_token_agreement"]) is not bool:
                raise ValueError("draft token agreement must be boolean")
        if (
            row["anchor"]["greedy_token_agreement"]
            and not row["candidate"]["greedy_token_agreement"]
        ):
            causes.append("new_greedy_token_disagreement")
        if causes:
            rejected.append(dict(step=row["step"], causes=causes))
    return dict(
        accepted=not rejected,
        decision="accept_joint_allocation"
        if not rejected
        else "retain_same_byte_compressed_anchor",
        rejected_steps=rejected,
        all_prefixes_checked=True,
        later_checkpoints_completed=all(p in {r["step"] for r in rows} for p in policy.checkpoints),
    )


def guard_joint_allocation(
    adapter: Any,
    prefill: Any,
    anchor: Any,
    candidate: Any,
    budget: Any,
    policy: DraftAllocationGuardConfig,
    *,
    sink_tokens: int,
    clone_state: Callable[[Any], Any],
    state_from_packed: Callable[[Any, Any], Any],
) -> tuple[Any, dict[str, Any]]:
    """Probe private clones, then publish either the entire proposal or anchor."""
    import torch

    from mosaickv.adapters.huggingface.bounded_history_cache import install_bounded_history

    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)
    started = time.perf_counter()

    def layers(packed: Any) -> tuple[Any, ...]:
        return tuple((x.key, x.value) for x in packed.snapshot.layers)

    anchor_sha, proposed_sha = packed_positions_sha(anchor), packed_positions_sha(candidate)
    if budget is None:
        raise ValueError("draft allocation guard requires a physical prompt/history budget")
    if any(
        CacheLifecycle.count(layers(x))[1] != budget.prompt_budget_bytes
        for x in (anchor, candidate)
    ):
        raise ValueError("draft guard received unequal or over-budget prompt caches")
    metadata: dict[str, Any] = dict(
        policy="joint_same_byte_FullKV_draft_logit_nonregression_v1",
        horizon=policy.horizon,
        checkpoints=list(policy.checkpoints),
        absolute_tolerance=policy.absolute_tolerance,
        relative_tolerance=policy.relative_tolerance,
        comparison="all_draft_prefixes_not_mean_divergence",
        benchmark_labels_accessed=False,
        evidence="deterministic_FullKV_greedy_draft_reusing_original_prefill",
        used_fullkv_fallback=False,
        original_KV_values_and_positions_preserved=True,
        active_cap_bytes=budget.cap_bytes,
        prompt_bytes=budget.prompt_budget_bytes,
        reserved_generated_bytes=budget.reserved_generated_bytes,
        anchor_positions_sha256=anchor_sha,
        position_hash_domain="original_logical_positions",
        proposed_positions_sha256=proposed_sha,
        finite_horizon_is_not_a_quality_guarantee=True,
    )
    if candidate is anchor or anchor_sha == proposed_sha or budget.generated_positions == 0:
        metadata.update(
            accepted=False,
            decision="retain_same_byte_compressed_anchor",
            reason="no_decode_updates" if budget.generated_positions == 0 else "unchanged_proposal",
            probe_updates=0,
            steps=[],
            final_positions_sha256=anchor_sha,
            final_layer_capacities=[int(x.key.shape[-2]) for x in anchor.snapshot.layers],
            measured_source_cache_unchanged=True,
            logical_positions_unchanged=True,
            fullkv_prefills=0,
            wall_seconds=time.perf_counter() - started,
        )
        return anchor, metadata
    source_pairs = (
        tuple(adapter._legacy_layers(prefill.state.past_key_values)[0])
        + layers(anchor)
        + layers(candidate)
    )
    tensors = [tensor for pair in source_pairs for tensor in pair]
    saved = [tensor.detach().clone() for tensor in tensors]
    original_positions = (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    )
    reference = clone_state(prefill.state)
    adapter.preallocate_decode_cache(reference, policy.horizon)
    states = {}
    for name, packed in (("anchor", anchor), ("candidate", candidate)):
        state = clone_state(state_from_packed(prefill.state, packed))
        install_bounded_history(
            adapter,
            state,
            generated_capacity=budget.generated_positions,
            sink_tokens=min(sink_tokens, budget.generated_positions - 1),
            cap_bytes=budget.cap_bytes,
        )
        states[name] = state
    steps = []
    token = prefill.next_token_id.detach().clone()
    for step in range(1, policy.horizon + 1):
        full = adapter.decode_one_token(token, reference, capture_queries=False)
        reference = full.state
        logp = torch.log_softmax(full.logits.float(), dim=-1)
        probability = logp.exp()
        row: dict[str, Any] = dict(step=step, checkpoint=step in policy.checkpoints)
        for name, state in states.items():
            out = adapter.decode_one_token(token, state, capture_queries=False)
            states[name] = out.state
            logq = torch.log_softmax(out.logits.float(), dim=-1)
            row[name] = dict(
                logit_kl=float((probability * (logp - logq)).sum(-1).mean().item()),
                maximum_logit_difference=float(
                    (full.logits.float() - out.logits.float()).abs().max().item()
                ),
                greedy_token_agreement=bool(torch.equal(full.next_token_id, out.next_token_id)),
                active_bytes=CacheLifecycle.count(
                    adapter._legacy_layers(out.state.past_key_values)[0]
                )[1],
            )
            if row[name]["active_bytes"] != budget.cap_bytes:
                raise RuntimeError("private draft state violated the common physical cap")
        steps.append(row)
        token = full.next_token_id
    if not all(torch.equal(before, after) for before, after in zip(tensors, saved, strict=True)):
        raise RuntimeError("draft guard mutated source or publishable cache tensors")
    if original_positions != (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    ):
        raise RuntimeError("draft guard mutated original logical sequence positions")
    decision = decide_draft_stability(steps, policy)
    selected = candidate if decision["accepted"] else anchor
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)
    metadata.update(
        **decision,
        steps=steps,
        probe_updates=3 * policy.horizon,
        final_positions_sha256=proposed_sha if decision["accepted"] else anchor_sha,
        final_layer_capacities=[int(x.key.shape[-2]) for x in selected.snapshot.layers],
        measured_source_cache_unchanged=True,
        logical_positions_unchanged=True,
        fullkv_prefills=0,
        wall_seconds=time.perf_counter() - started,
        temporary_probe_cache_bytes=CacheLifecycle.count(
            adapter._legacy_layers(reference.past_key_values)[0]
        )[1]
        + 2 * budget.cap_bytes,
        source_verification_copy_bytes=sum(int(x.numel()) * int(x.element_size()) for x in saved),
        probe_memory_excluded_from_live_decode_cache_but_included_in_peak_GPU_memory=True,
    )
    return selected, metadata


def validate_draft_guard_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Replay serialized decisions, including the actual post-rollback selection."""
    policy = DraftAllocationGuardConfig(
        trace["horizon"],
        tuple(trace["checkpoints"]),
        trace["absolute_tolerance"],
        trace["relative_tolerance"],
    )
    if trace["probe_updates"]:
        expected = decide_draft_stability(trace["steps"], policy)
        if trace["probe_updates"] != 3 * policy.horizon:
            raise ValueError("draft guard probe count changed")
        for key, value in expected.items():
            if trace.get(key) != value:
                raise ValueError(f"draft guard decision mismatch: {key}")
        if any(
            row[name]["active_bytes"] != trace["active_cap_bytes"]
            for row in trace["steps"]
            for name in ("anchor", "candidate")
        ):
            raise ValueError("draft guard trace violates probe storage cap")
    elif (
        trace["accepted"]
        or trace.get("reason") not in {"no_decode_updates", "unchanged_proposal"}
        or trace["steps"]
    ):
        raise ValueError("skipped guard cannot claim a passing model probe")
    final_sha = (
        trace["proposed_positions_sha256"]
        if trace["accepted"]
        else trace["anchor_positions_sha256"]
    )
    if trace["final_positions_sha256"] != final_sha:
        raise ValueError("draft guard final positions contradict rollback")
    if (
        trace["benchmark_labels_accessed"]
        or trace["used_fullkv_fallback"]
        or not trace["measured_source_cache_unchanged"]
        or not trace["logical_positions_unchanged"]
        or trace["prompt_bytes"] + trace["reserved_generated_bytes"] != trace["active_cap_bytes"]
    ):
        raise ValueError("draft guard label/isolation/physical-budget invariant failed")
    return dict(
        accepted=trace["accepted"],
        probe_updates=trace["probe_updates"],
        wall_seconds=trace["wall_seconds"],
        reason=trace.get("reason", trace["decision"]),
    )
