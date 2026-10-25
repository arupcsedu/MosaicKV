"""V6 bounded single exchanges: shared draft evidence, fail-fast rejection.

Only model-generated logits are accepted as evidence. This module has no
benchmark-label or evaluation/oracle dependency. Passing 32 tokens is not a
guarantee of later answer quality.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from mosaickv.backends.hf_allocation_guard import (
    decide_draft_stability,
    draft_row_causes,
    packed_positions_sha,
)
from mosaickv.config import DraftAllocationGuardConfig, DraftExchangeSearchConfig
from mosaickv.measurements.cache_lifecycle import CacheLifecycle


def validate_single_transaction(parent: Any, candidate: Any, event: dict[str, Any]) -> None:
    """Verify a single physical layer slot moved, not merely equal view sizes."""

    def positions(packed):
        grouped = {}
        for row in packed.slot_records:
            if row["tier"] != "exact":
                raise ValueError("single exchanges support exact K/V only")
            route = row["layer"], row["kv_head"]
            group = grouped.setdefault(route, set())
            if row["logical_position"] in group:
                raise ValueError("duplicate logical position in single exchange")
            group.add(row["logical_position"])
        return grouped

    before, after = positions(parent), positions(candidate)
    d, r = event["donor_layer"], event["receiver_layer"]
    if (
        d == r
        or set(before) != set(after)
        or not any(layer == d for layer, _ in before)
        or not any(layer == r for layer, _ in before)
    ):
        raise ValueError("single exchange has invalid donor/receiver routes")
    for (layer, head), selected in before.items():
        removed, added = selected - after[layer, head], after[layer, head] - selected
        if (len(removed), len(added)) != (
            (1, 0) if layer == d else (0, 1) if layer == r else (0, 0)
        ):
            raise ValueError("proposal is not a single whole-layer slot exchange")
    capacities = [int(layer.key.shape[-2]) for layer in parent.snapshot.layers]
    if list(event["initial_layer_capacities"]) != capacities:
        raise ValueError("single exchange parent capacity metadata mismatch")
    expected = capacities.copy()
    expected[d] -= 1
    expected[r] += 1
    if [int(layer.key.shape[-2]) for layer in candidate.snapshot.layers] != expected:
        raise ValueError("single exchange creates padding/incorrect physical capacities")
    layer = parent.snapshot.layers[d]
    slot_bytes = (
        layer.key.numel() * layer.key.element_size()
        + layer.value.numel() * layer.value.element_size()
    ) // capacities[d]
    if event["transferred_bytes"] != slot_bytes:
        raise ValueError("single exchange transfer metadata changes physical bytes")


def _decision(rows: list[dict[str, Any]], policy: DraftAllocationGuardConfig) -> dict[str, Any]:
    def one(evidence):
        if len(evidence) < policy.horizon and not any(
            draft_row_causes(row, policy) for row in evidence
        ):
            return dict(accepted=False, decision="incomplete_not_a_pass")
        return decide_draft_stability(evidence, policy, allow_early_rejection=True)

    root = one(rows)
    parent = one([dict(row, anchor=row["parent"]) for row in rows])
    if root["decision"] == parent["decision"] == "incomplete_not_a_pass":
        raise ValueError("partial evidence without a demonstrated rejection is invalid")
    return dict(accepted=root["accepted"] and parent["accepted"], root=root, parent=parent)


def _rank(rows: list[dict[str, Any]], sha: str) -> tuple[Any, ...]:
    return (
        sum(not row["candidate"]["greedy_token_agreement"] for row in rows),
        sum(row["candidate"]["logit_kl"] for row in rows),
        sum(row["candidate"]["maximum_logit_difference"] for row in rows),
        sha,
    )


def guard_single_exchanges(
    adapter: Any,
    prefill: Any,
    anchor: Any,
    budget: Any,
    policy: DraftAllocationGuardConfig,
    search: DraftExchangeSearchConfig,
    *,
    sink_tokens: int,
    clone_state: Callable[[Any], Any],
    state_from_packed: Callable[[Any, Any], Any],
    propose: Callable[[Any, int], Any],
    pack: Callable[[Any], Any],
) -> tuple[Any, dict[str, Any]]:
    """Probe short-listed singles, rank passing plans by actual draft logits.

    FullKV and the original anchor advance lazily ONCE per prefix, not once per
    proposal. A rejected prefix terminates only that candidate. A passing plan
    must complete the entire horizon and dominate BOTH original and accepted
    parent divergences. Compositions therefore cannot accumulate tolerances.
    """
    import torch

    from mosaickv.adapters.huggingface.bounded_history_cache import install_bounded_history

    def sync():
        if adapter.device.type == "cuda":
            torch.cuda.synchronize(adapter.device)

    def pairs(packed):
        return tuple((layer.key, layer.value) for layer in packed.snapshot.layers)

    def size(packed):
        return CacheLifecycle.count(pairs(packed))[1]

    def capture_sources(packed):
        tensors = [t for pair in pairs(packed) for t in pair]
        return tensors, [t.detach().clone() for t in tensors]

    def assert_unchanged(captured):
        if not all(torch.equal(a, b) for a, b in zip(*captured, strict=True)):
            raise RuntimeError("single-exchange probe mutated publishable/source tensors")

    def private(packed):
        if size(packed) != budget.prompt_budget_bytes:
            raise ValueError("single exchange violates exact physical prompt budget")
        state = clone_state(state_from_packed(prefill.state, packed))
        install_bounded_history(
            adapter,
            state,
            generated_capacity=budget.generated_positions,
            sink_tokens=min(sink_tokens, budget.generated_positions - 1),
            cap_bytes=budget.cap_bytes,
        )
        return state

    sync()
    started = time.perf_counter()
    anchor_sha = packed_positions_sha(anchor)
    if budget is None or size(anchor) != budget.prompt_budget_bytes:
        raise ValueError("single-exchange guard requires an exact physical prompt/history budget")
    trace = dict(
        policy="bounded_single_equal_byte_draft_logit_v6",
        horizon=policy.horizon,
        checkpoints=list(policy.checkpoints),
        absolute_tolerance=policy.absolute_tolerance,
        relative_tolerance=policy.relative_tolerance,
        candidates_per_round=search.candidates_per_round,
        max_rounds=search.max_rounds,
        benchmark_labels_accessed=False,
        used_fullkv_fallback=False,
        active_cap_bytes=budget.cap_bytes,
        prompt_bytes=budget.prompt_budget_bytes,
        reserved_generated_bytes=budget.reserved_generated_bytes,
        anchor_positions_sha256=anchor_sha,
        position_hash_domain="original_logical_positions",
        original_KV_values_and_positions_preserved=True,
        fullkv_prefills=0,
        comparison="every_prefix_against_original_anchor_and_previously_passed_parent",
        ranking="actual_draft_greedy_disagreements_then_sum_KL_then_sum_maxlogit_then_hash",
        finite_horizon_is_not_a_quality_guarantee=True,
        attempts=[],
        rounds=[],
        reference_updates=0,
        anchor_updates=0,
        candidate_updates=0,
        planning_seconds=0.0,
        packing_seconds=0.0,
    )
    full_tensors = [
        t for pair in adapter._legacy_layers(prefill.state.past_key_values)[0] for t in pair
    ]
    source_copies = (full_tensors, [t.detach().clone() for t in full_tensors])
    anchor_copies = capture_sources(anchor)
    original_positions = (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    )
    reference = anchor_state = None
    evidence: list[tuple[Any, Any, dict[str, Any]]] = []
    selected, selected_sha, parent_rows = anchor, anchor_sha, None
    seen = {anchor_sha}

    def measure(full, out):
        logp = torch.log_softmax(full.logits.float(), dim=-1)
        logq = torch.log_softmax(out.logits.float(), dim=-1)
        active = CacheLifecycle.count(adapter._legacy_layers(out.state.past_key_values)[0])[1]
        if active != budget.cap_bytes:
            raise RuntimeError("private single-exchange draft violated live physical cap")
        return dict(
            logit_kl=float((logp.exp() * (logp - logq)).sum(-1).mean().item()),
            maximum_logit_difference=float(
                (full.logits.float() - out.logits.float()).abs().max().item()
            ),
            greedy_token_agreement=bool(torch.equal(full.next_token_id, out.next_token_id)),
            active_bytes=active,
        )

    def prefix(step):
        nonlocal reference, anchor_state
        if len(evidence) < step:
            if reference is None:
                reference = clone_state(prefill.state)
                adapter.preallocate_decode_cache(reference, policy.horizon)
                anchor_state = private(anchor)
            token = (
                prefill.next_token_id.detach().clone()
                if not evidence
                else evidence[-1][1].next_token_id
            )
            full = adapter.decode_one_token(token, reference, capture_queries=False)
            reference = full.state
            original = adapter.decode_one_token(token, anchor_state, capture_queries=False)
            anchor_state = original.state
            # Retain logits/tokens only, not growing decode-state objects.
            from types import SimpleNamespace

            immutable = SimpleNamespace(
                logits=full.logits.detach().clone(),
                next_token_id=full.next_token_id.detach().clone(),
            )
            evidence.append((token.detach().clone(), immutable, measure(full, original)))
            trace["reference_updates"] += 1
            trace["anchor_updates"] += 1
        return evidence[step - 1]

    if budget.generated_positions:
        for round_index in range(search.max_rounds):
            sync()
            t0 = time.perf_counter()
            plans = tuple(propose(selected, search.candidates_per_round))
            sync()
            trace["planning_seconds"] += time.perf_counter() - t0
            if len(plans) > search.candidates_per_round:
                raise ValueError("single-exchange shortlist exceeds registered limit")
            passed = []
            parent_sha = selected_sha
            for plan in plans:
                sync()
                t0 = time.perf_counter()
                candidate = pack(plan)
                sync()
                trace["packing_seconds"] += time.perf_counter() - t0
                validate_single_transaction(
                    selected, candidate, plan.reconstruction["single_exchange"]
                )
                sha = packed_positions_sha(candidate)
                if sha in seen:
                    continue
                seen.add(sha)
                candidate_copies = capture_sources(candidate)
                state = private(candidate)
                rows = []
                for step in range(1, policy.horizon + 1):
                    token, full, original_metrics = prefix(step)
                    out = adapter.decode_one_token(token, state, capture_queries=False)
                    state = out.state
                    row = dict(
                        step=step,
                        checkpoint=step in policy.checkpoints,
                        anchor=original_metrics,
                        parent=original_metrics
                        if parent_rows is None
                        else parent_rows[step - 1]["candidate"],
                        candidate=measure(full, out),
                    )
                    rows.append(row)
                    trace["candidate_updates"] += 1
                    if draft_row_causes(row, policy) or draft_row_causes(
                        dict(row, anchor=row["parent"]), policy
                    ):
                        break
                decision = _decision(rows, policy)
                assert_unchanged(candidate_copies)
                event = dict(
                    round=round_index,
                    parent_positions_sha256=parent_sha,
                    candidate_positions_sha256=sha,
                    steps=rows,
                    **decision,
                    stopped_early=len(rows) < policy.horizon,
                    single_exchange=plan.reconstruction["single_exchange"],
                    layer_capacities=list(plan.capacities),
                )
                trace["attempts"].append(event)
                if decision["accepted"]:
                    passed.append((_rank(rows, sha), candidate, sha, rows))
                # Discard private decode buffers before constructing another candidate.
                del state, out
            if not passed:
                trace["rounds"].append(
                    dict(round=round_index, parent_positions_sha256=parent_sha, winner=None)
                )
                break
            _, selected, selected_sha, parent_rows = min(passed, key=lambda x: x[0])
            trace["rounds"].append(
                dict(round=round_index, parent_positions_sha256=parent_sha, winner=selected_sha)
            )
    assert_unchanged(source_copies)
    assert_unchanged(anchor_copies)
    if original_positions != (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    ):
        raise RuntimeError("single-exchange guard changed original logical counters")
    sync()
    trace.update(
        accepted=selected_sha != anchor_sha,
        final_positions_sha256=selected_sha,
        final_layer_capacities=[int(x.key.shape[-2]) for x in selected.snapshot.layers],
        reason="no_decode_updates"
        if not budget.generated_positions
        else "bounded_search_completed",
        probe_updates=trace["reference_updates"]
        + trace["anchor_updates"]
        + trace["candidate_updates"],
        measured_source_cache_unchanged=True,
        logical_positions_unchanged=True,
        wall_seconds=time.perf_counter() - started,
        full_horizon_acceptance_required=True,
        fail_fast_rejection=True,
        source_verification_copy_bytes=sum(
            t.numel() * t.element_size() for t in source_copies[1] + anchor_copies[1]
        ),
        peak_candidate_verification_copy_bytes=budget.prompt_budget_bytes
        if trace["attempts"]
        else 0,
        temporary_probe_cache_bytes=(
            0
            if reference is None
            else CacheLifecycle.count(adapter._legacy_layers(reference.past_key_values)[0])[1]
            + 2 * budget.cap_bytes
        ),
        probe_memory_excluded_from_live_decode_cache_but_included_in_peak_GPU_memory=True,
        all_attempts_are_single_equal_byte_transactions=True,
    )
    return selected, trace


def validate_single_exchange_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed replay: partial rejections permitted; partial passes forbidden."""
    policy = DraftAllocationGuardConfig(
        trace["horizon"],
        tuple(trace["checkpoints"]),
        trace["absolute_tolerance"],
        trace["relative_tolerance"],
    )
    search = DraftExchangeSearchConfig(trace["candidates_per_round"], trace["max_rounds"])
    attempts, rounds = trace["attempts"], trace["rounds"]
    if (
        len(rounds) > search.max_rounds
        or len(attempts) > search.max_rounds * search.candidates_per_round
        or any(
            type(event["round"]) is not int or not 0 <= event["round"] < len(rounds)
            for event in attempts
        )
        or len({event["candidate_positions_sha256"] for event in attempts}) != len(attempts)
        or trace["anchor_positions_sha256"]
        in {event["candidate_positions_sha256"] for event in attempts}
    ):
        raise ValueError("single exchange trace exceeds search budget")
    parent_sha = trace["anchor_positions_sha256"]
    parent_rows = None
    root_evidence: dict[int, dict[str, Any]] = {}
    capacities = None
    for i, record in enumerate(rounds):
        choices = [event for event in attempts if event["round"] == i]
        if (
            record["round"] != i
            or record["parent_positions_sha256"] != parent_sha
            or len(choices) > search.candidates_per_round
        ):
            raise ValueError("single exchange round continuity failed")
        passed = []
        for event in choices:
            rows = event["steps"]
            if event["parent_positions_sha256"] != parent_sha or _decision(rows, policy) != {
                k: event[k] for k in ("accepted", "root", "parent")
            }:
                raise ValueError("single exchange decision/parent mismatch")
            for row in rows:
                step = row["step"]
                if root_evidence.setdefault(step, row["anchor"]) != row["anchor"]:
                    raise ValueError("single exchange changed shared anchor evidence")
                expected_parent = (
                    row["anchor"]
                    if parent_rows is None
                    else parent_rows[row["step"] - 1]["candidate"]
                )
                if row["parent"] != expected_parent or any(
                    row[name]["active_bytes"] != trace["active_cap_bytes"]
                    for name in ("anchor", "candidate", "parent")
                ):
                    raise ValueError("single exchange prefix storage/parent evidence mismatch")
            transaction = event["single_exchange"]
            initial = transaction["initial_layer_capacities"]
            if capacities is None:
                capacities = list(initial)
            d, r = transaction["donor_layer"], transaction["receiver_layer"]
            if (
                list(initial) != capacities
                or d == r
                or not 0 <= d < len(initial)
                or not 0 <= r < len(initial)
                or min(initial) < 1
                or transaction["transferred_bytes"] * sum(initial) != trace["prompt_bytes"]
            ):
                raise ValueError("single exchange capacity/physical-byte continuity failed")
            expected = list(initial)
            expected[d] -= 1
            expected[r] += 1
            if min(expected) < 1 or event["layer_capacities"] != expected:
                raise ValueError("single exchange final capacity mismatch")
            failed = [
                row["step"]
                for row in rows
                if draft_row_causes(row, policy)
                or draft_row_causes(dict(row, anchor=row["parent"]), policy)
            ]
            if failed and failed != [len(rows)]:
                raise ValueError("rejected exchange did not stop at first violation")
            if event["stopped_early"] != (len(rows) < policy.horizon):
                raise ValueError("single exchange early-stop trace mismatch")
            if event["accepted"]:
                passed.append(event)
        winner = (
            min(passed, key=lambda e: _rank(e["steps"], e["candidate_positions_sha256"]))
            if passed
            else None
        )
        if record["winner"] != (None if winner is None else winner["candidate_positions_sha256"]):
            raise ValueError("single exchange actual-logit ranking mismatch")
        if winner is None:
            if i != len(rounds) - 1:
                raise ValueError("composition started without a full-horizon accepted parent")
        else:
            parent_sha, parent_rows = winner["candidate_positions_sha256"], winner["steps"]
            capacities = winner["layer_capacities"]
    updates = sum(len(event["steps"]) for event in attempts)
    maximum = max((len(event["steps"]) for event in attempts), default=0)
    if (
        trace["reference_updates"] != maximum
        or trace["anchor_updates"] != maximum
        or trace["candidate_updates"] != updates
        or trace["probe_updates"] != 2 * maximum + updates
        or trace["final_positions_sha256"] != parent_sha
        or trace["accepted"] != (parent_sha != trace["anchor_positions_sha256"])
        or trace["benchmark_labels_accessed"]
        or trace["used_fullkv_fallback"]
        or not trace["measured_source_cache_unchanged"]
        or not trace["logical_positions_unchanged"]
        or not trace["fail_fast_rejection"]
        or not trace["full_horizon_acceptance_required"]
        or not trace["all_attempts_are_single_equal_byte_transactions"]
        or trace["fullkv_prefills"] != 0
        or trace["position_hash_domain"] != "original_logical_positions"
        or (capacities is not None and trace["final_layer_capacities"] != capacities)
        or trace["prompt_bytes"] + trace["reserved_generated_bytes"] != trace["active_cap_bytes"]
    ):
        raise ValueError("single exchange isolation/budget/probe accounting failed")
    if any(
        not math.isfinite(trace[key]) or trace[key] < 0
        for key in ("wall_seconds", "planning_seconds", "packing_seconds")
    ):
        raise ValueError("single exchange missing/invalid phase timing")
    return dict(
        accepted=trace["accepted"],
        probe_updates=trace["probe_updates"],
        wall_seconds=trace["wall_seconds"],
        reason=trace["reason"],
        attempts=len(attempts),
        early_rejections=sum(event["stopped_early"] for event in attempts),
    )
