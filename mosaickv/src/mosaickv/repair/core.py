"""Backend-independent uncertainty triggers and single-pass residual repair."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from mosaickv.cache_state import (
    ExactTier,
    KVBlockDescriptor,
    MosaicKVState,
    PrototypeTier,
    tensor_storage_bytes,
)
from mosaickv.config import RepairConfig
from mosaickv.prototypes.refresh import refresh_prototype_after_promotions
from mosaickv.prototypes.rope import RotaryPrototypeKeyTransform
from mosaickv.prototypes.types import PrototypeRecord
from mosaickv.repair.types import (
    RepairCacheState,
    RepairEvent,
    RepairStepResult,
    RepairStepSignals,
    RepairTriggerReason,
)
from mosaickv.residual.storage import (
    discard_residual_payloads,
    restore_residual_payloads_async,
)
from mosaickv.types import BudgetUnit, RepairCandidateRanking, RepairPolicy

ReDecodeCallback = Callable[[RepairCacheState], Any]


@dataclass(frozen=True, slots=True)
class _RepairPlan:
    ranked_promotions: tuple[tuple[int, int], ...]
    active_prototype_ids: tuple[int, ...]
    superseded_prototype_ids: tuple[int, ...]
    budget_evicted_prototype_ids: tuple[int, ...]


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if value.__class__.__module__.startswith("torch") and hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        raise ValueError("decode distributions must contain only finite values")
    return result


def _logit_vector(logits: Any) -> np.ndarray[Any, Any]:
    values = _to_numpy(logits)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("decode logits must have a non-empty vocabulary dimension")
    return cast("np.ndarray[Any, Any]", values.reshape(-1, values.shape[-1])[-1])


def _softmax(logits: Any) -> np.ndarray[Any, Any]:
    values = _logit_vector(logits)
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return cast("np.ndarray[Any, Any]", weights / float(np.sum(weights)))


def normalized_next_token_entropy(logits: Any) -> float:
    """Return entropy divided by log(vocabulary size), in the interval [0, 1]."""

    probabilities = _softmax(logits)
    if probabilities.size == 1:
        return 0.0
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    normalized = entropy / math.log(probabilities.size)
    return min(1.0, max(0.0, normalized))


def next_token_probability_margin(logits: Any) -> float:
    """Return the softmax probability gap between the top two token decisions."""

    probabilities = _softmax(logits)
    if probabilities.size == 1:
        return 1.0
    top_two = np.partition(probabilities, -2)[-2:]
    return min(1.0, max(0.0, float(top_two.max() - top_two.min())))


def draft_kl_divergence(logits: Any, draft_distribution: Any | None) -> float | None:
    """Return KL(provisional || draft) for an optional cheap draft probability vector."""

    if draft_distribution is None:
        return None
    provisional = _softmax(logits)
    draft = _to_numpy(draft_distribution).reshape(-1)
    if draft.shape != provisional.shape:
        raise ValueError("draft distribution vocabulary does not match provisional logits")
    if bool(np.any(draft < 0)) or float(np.sum(draft)) <= 0:
        raise ValueError("draft distribution must be nonnegative with positive total mass")
    normalized_draft = draft / float(np.sum(draft))
    epsilon = np.finfo(np.float64).tiny
    safe_draft = np.maximum(normalized_draft, epsilon)
    positive = provisional > 0
    return max(
        0.0,
        float(
            np.sum(
                provisional[positive]
                * (np.log(provisional[positive]) - np.log(safe_draft[positive]))
            )
        ),
    )


def calculate_repair_signals(
    state: RepairCacheState,
    provisional_logits: Any,
    prototype_attention_mass: Mapping[int, float],
    *,
    draft_distribution: Any | None = None,
    normalize_prototype_risk: bool = False,
) -> RepairStepSignals:
    """Calculate entropy, total prototype mass, dispersion risk, and optional draft KL."""

    masses = {
        int(prototype_id): float(mass) for prototype_id, mass in prototype_attention_mass.items()
    }
    expected_ids = set(state.active_prototype_ids)
    if set(masses) != expected_ids:
        missing = sorted(expected_ids - set(masses))
        extra = sorted(set(masses) - expected_ids)
        raise ValueError(
            "prototype attention must cover every active prototype exactly once; "
            f"missing={missing}, extra={extra}"
        )
    if any(not math.isfinite(mass) or not 0 <= mass <= 1 for mass in masses.values()):
        raise ValueError("prototype attention masses must be finite and lie in [0, 1]")
    total_mass = sum(masses.values())
    if total_mass > 1 + 1e-9:
        raise ValueError("globally normalized prototype attention masses cannot sum above one")
    raw_risks = tuple(
        (
            prototype_id,
            masses[prototype_id] * state.prototype_record(prototype_id).diagnostics.dispersion,
        )
        for prototype_id in state.active_prototype_ids
    )
    risk_total = sum(risk for _prototype_id, risk in raw_risks)
    risks = (
        tuple(
            (prototype_id, risk / risk_total)
            for prototype_id, risk in raw_risks
        )
        if normalize_prototype_risk and risk_total > 0
        else raw_risks
    )
    return RepairStepSignals(
        normalized_entropy=normalized_next_token_entropy(provisional_logits),
        next_token_probability_margin=next_token_probability_margin(provisional_logits),
        total_prototype_attention_mass=total_mass,
        prototype_attention_masses=tuple(
            (prototype_id, masses[prototype_id]) for prototype_id in state.active_prototype_ids
        ),
        prototype_risks=risks,
        maximum_prototype_risk=max((risk for _prototype_id, risk in risks), default=0.0),
        draft_kl_divergence=draft_kl_divergence(provisional_logits, draft_distribution),
        risk_normalization=(
            "sum_to_one" if normalize_prototype_risk else "raw_attention_mass_times_dispersion"
        ),
    )


def _trigger_reason(
    config: RepairConfig,
    signals: RepairStepSignals,
    *,
    oracle_should_repair: bool | None,
) -> RepairTriggerReason:
    if not config.enabled:
        return RepairTriggerReason.DISABLED
    if config.policy is RepairPolicy.NONE:
        return RepairTriggerReason.POLICY_NONE
    if config.policy is RepairPolicy.ORACLE:
        if oracle_should_repair is None:
            raise ValueError("oracle repair requires an evaluation-only oracle decision")
        return (
            RepairTriggerReason.ORACLE
            if oracle_should_repair
            else RepairTriggerReason.ORACLE_NO_REPAIR
        )
    entropy = signals.normalized_entropy >= config.entropy_threshold
    risk = (
        signals.maximum_prototype_risk >= config.prototype_risk_threshold
        and signals.normalized_entropy >= config.prototype_risk_min_entropy
    )
    if config.policy is RepairPolicy.ENTROPY:
        return RepairTriggerReason.ENTROPY if entropy else RepairTriggerReason.THRESHOLD_NOT_MET
    if config.policy is RepairPolicy.PROTOTYPE_RISK:
        return RepairTriggerReason.PROTOTYPE_RISK if risk else RepairTriggerReason.THRESHOLD_NOT_MET
    if config.policy is RepairPolicy.DECISION_AWARE:
        high_risk = bool(
            config.decision_high_risk_threshold is not None
            and signals.maximum_prototype_risk >= config.decision_high_risk_threshold
        )
        if high_risk:
            return RepairTriggerReason.HIGH_PROTOTYPE_RISK
        decision_risk = risk and (
            signals.next_token_probability_margin <= config.decision_margin_threshold
        )
        return (
            RepairTriggerReason.DECISION_RISK
            if decision_risk
            else RepairTriggerReason.THRESHOLD_NOT_MET
        )
    if config.policy is RepairPolicy.ENTROPY_OR_PROTOTYPE_RISK:
        if entropy and risk:
            return RepairTriggerReason.ENTROPY_AND_PROTOTYPE_RISK
        if entropy:
            return RepairTriggerReason.ENTROPY
        if risk:
            return RepairTriggerReason.PROTOTYPE_RISK
        return RepairTriggerReason.THRESHOLD_NOT_MET
    raise ValueError(f"unsupported repair policy: {config.policy}")


def _reason_triggers(reason: RepairTriggerReason) -> bool:
    return reason in {
        RepairTriggerReason.ENTROPY,
        RepairTriggerReason.PROTOTYPE_RISK,
        RepairTriggerReason.ENTROPY_AND_PROTOTYPE_RISK,
        RepairTriggerReason.DECISION_RISK,
        RepairTriggerReason.HIGH_PROTOTYPE_RISK,
        RepairTriggerReason.ORACLE,
    }


def _active_cost_for_plan(
    state: RepairCacheState,
    promotions: tuple[tuple[int, int], ...],
    active_prototype_ids: tuple[int, ...],
) -> int:
    if state.active_budget.unit is BudgetUnit.BLOCKS:
        return len(state.mosaic_state.exact.blocks) + len(promotions) + len(active_prototype_ids)
    if state.active_budget.unit is BudgetUnit.RETAINED_SLOTS:
        return (
            sum(block.position_count for block in state.mosaic_state.exact.blocks)
            + sum(
                state.full_state.blocks[node_id].position_count for node_id, _parent in promotions
            )
            + len(active_prototype_ids)
        )
    if state.active_budget.unit is BudgetUnit.BYTES:
        prototype_payloads = {
            prototype_id: (
                state.mosaic_state.prototypes.prototype_keys[offset],
                state.mosaic_state.prototypes.prototype_values[offset],
            )
            for offset, prototype_id in enumerate(state.active_prototype_ids)
        }
        return (
            state.mosaic_state.exact.active_bytes
            + sum(state.full_state.blocks[node_id].byte_size for node_id, _parent in promotions)
            + sum(
                tensor_storage_bytes(prototype_payloads[prototype_id][0])
                + tensor_storage_bytes(prototype_payloads[prototype_id][1])
                for prototype_id in active_prototype_ids
            )
        )
    raise ValueError(f"unsupported repair budget unit: {state.active_budget.unit}")


def _rank_promotion_candidates(
    state: RepairCacheState,
    signals: RepairStepSignals,
    ranking: RepairCandidateRanking,
) -> tuple[tuple[int, int], ...]:
    residual_nodes = {payload.source_node_id for payload in state.residual_storage.payloads}
    ordered_prototypes = sorted(
        state.active_prototype_ids,
        key=lambda prototype_id: (-signals.risk_for(prototype_id), prototype_id),
    )
    members_by_prototype: dict[int, tuple[Any, ...]] = {}
    for prototype_id in ordered_prototypes:
        record = state.prototype_record(prototype_id)
        members_by_prototype[prototype_id] = tuple(
            sorted(
                (member for member in record.members if member.node_id in residual_nodes),
                key=lambda member: (
                    -(
                        member.normalized_weight
                        * member.reconstruction_error
                        * (
                            1.0 + member.value_output_norm
                            if ranking is RepairCandidateRanking.DECISION_INFLUENCE
                            else 1.0
                        )
                        if ranking
                        in {
                            RepairCandidateRanking.RISK_WEIGHTED_RECONSTRUCTION_ERROR,
                            RepairCandidateRanking.DECISION_INFLUENCE,
                        }
                        else member.normalized_weight
                    ),
                    -member.reconstruction_error,
                    -member.normalized_weight,
                    member.node_id,
                ),
            )
        )
    if ranking is RepairCandidateRanking.PROTOTYPE_WEIGHT:
        return tuple(
            (member.node_id, prototype_id)
            for prototype_id in ordered_prototypes
            for member in members_by_prototype[prototype_id]
        )
    # Allocate the first R slots across the highest-risk clusters before
    # taking a second member from any one cluster. This gives larger repair
    # budgets evidence coverage instead of repeatedly decomposing one local
    # prototype; member order within a cluster targets its largest weighted
    # reconstruction error.
    maximum_cluster_size = max(
        (len(members) for members in members_by_prototype.values()),
        default=0,
    )
    return tuple(
        (members_by_prototype[prototype_id][member_rank].node_id, prototype_id)
        for member_rank in range(maximum_cluster_size)
        for prototype_id in ordered_prototypes
        if member_rank < len(members_by_prototype[prototype_id])
    )


def _plan_repair(
    state: RepairCacheState,
    signals: RepairStepSignals,
    config: RepairConfig,
) -> _RepairPlan | None:
    candidates = list(
        _rank_promotion_candidates(state, signals, config.candidate_ranking)[
            : config.max_blocks_per_step
        ]
    )
    while candidates:
        promotions = tuple(candidates)
        promoted_by_parent: dict[int, set[int]] = {}
        for node_id, parent_id in promotions:
            promoted_by_parent.setdefault(parent_id, set()).add(node_id)
        superseded = (
            {
                parent_id
                for parent_id, promoted in promoted_by_parent.items()
                if promoted == set(state.prototype_record(parent_id).assigned_node_ids)
            }
            if config.preserve_partial_prototype
            else set(promoted_by_parent)
        )
        protected = set(promoted_by_parent) - superseded
        active = set(state.active_prototype_ids) - superseded
        evicted: list[int] = []
        while (
            _active_cost_for_plan(state, promotions, tuple(sorted(active)))
            > state.active_budget.value
            and active
        ):
            victim = (
                min(
                    active - protected,
                    key=lambda prototype_id: (
                        state.prototype_record(prototype_id).eviction_utility,
                        prototype_id,
                    ),
                )
                if active - protected
                else None
            )
            if victim is None:
                break
            active.remove(victim)
            evicted.append(victim)
        if (
            _active_cost_for_plan(state, promotions, tuple(sorted(active)))
            <= state.active_budget.value
        ):
            return _RepairPlan(
                ranked_promotions=promotions,
                active_prototype_ids=tuple(sorted(active)),
                superseded_prototype_ids=tuple(sorted(superseded)),
                budget_evicted_prototype_ids=tuple(sorted(evicted)),
            )
        candidates.pop()
    return None


def _rebuild_prototype_tier(
    full_state: Any,
    records: tuple[PrototypeRecord, ...],
    payloads: Mapping[int, tuple[Any, Any]],
) -> PrototypeTier:
    blocks: list[KVBlockDescriptor] = []
    assignments: list[int] = []
    keys: list[Any] = []
    values: list[Any] = []
    for local_id, record in enumerate(records):
        blocks.extend(full_state.blocks[node_id] for node_id in record.assigned_node_ids)
        assignments.extend(local_id for _node_id in record.assigned_node_ids)
        key, value = payloads[record.prototype_id]
        keys.append(key)
        values.append(value)
    return PrototypeTier(tuple(blocks), tuple(keys), tuple(values), tuple(assignments))


def _promote(
    state: RepairCacheState,
    plan: _RepairPlan,
    key_transform: RotaryPrototypeKeyTransform | None,
) -> tuple[RepairCacheState, tuple[int, ...], int, float, bool]:
    payload_by_node = {
        payload.source_node_id: payload.payload_index for payload in state.residual_storage.payloads
    }
    prototype_payloads = {
        prototype_id: (
            state.mosaic_state.prototypes.prototype_keys[offset],
            state.mosaic_state.prototypes.prototype_values[offset],
        )
        for offset, prototype_id in enumerate(state.active_prototype_ids)
    }
    transfer_items = sorted(
        (
            payload_by_node[node_id],
            node_id,
            parent_id,
        )
        for node_id, parent_id in plan.ranked_promotions
    )
    payload_indices = tuple(payload_index for payload_index, _node_id, _parent in transfer_items)
    references = tuple(
        prototype_payloads[parent_id] for _payload_index, _node_id, parent_id in transfer_items
    )
    transfer = restore_residual_payloads_async(
        state.residual_storage,
        payload_indices,
        references,
    )
    restored = {
        state.residual_storage.payloads[payload_index].source_node_id: (key, value)
        for payload_index, key, value in zip(
            transfer.payload_indices,
            transfer.key_blocks,
            transfer.value_blocks,
            strict=True,
        )
    }
    exact_payloads = {
        node_id: (key, value)
        for node_id, key, value in zip(
            state.exact_node_ids,
            state.mosaic_state.exact.key_blocks,
            state.mosaic_state.exact.value_blocks,
            strict=True,
        )
    }
    exact_payloads.update(restored)
    exact_ids = tuple(sorted(exact_payloads))
    exact = ExactTier(
        tuple(state.full_state.blocks[node_id] for node_id in exact_ids),
        tuple(exact_payloads[node_id][0] for node_id in exact_ids),
        tuple(exact_payloads[node_id][1] for node_id in exact_ids),
    )
    promoted_by_parent: dict[int, dict[int, tuple[Any, Any]]] = {}
    for node_id, parent_id in plan.ranked_promotions:
        promoted_by_parent.setdefault(parent_id, {})[node_id] = restored[node_id]
    active_records: list[PrototypeRecord] = []
    active_payloads: dict[int, tuple[Any, Any]] = {}
    for prototype_id in plan.active_prototype_ids:
        record = state.prototype_record(prototype_id)
        payload = prototype_payloads[prototype_id]
        selected = promoted_by_parent.get(prototype_id)
        if selected:
            refreshed = refresh_prototype_after_promotions(
                state.full_state,
                record,
                payload[0],
                payload[1],
                selected,
                key_transform=key_transform,
            )
            record = refreshed.record
            payload = (refreshed.key, refreshed.value)
        active_records.append(record)
        active_payloads[prototype_id] = payload
    active_record_tuple = tuple(active_records)
    prototypes = _rebuild_prototype_tier(
        state.full_state,
        active_record_tuple,
        active_payloads,
    )
    residual_storage = discard_residual_payloads(state.residual_storage, payload_indices)
    mosaic_state = MosaicKVState.create(
        state.full_state,
        exact=exact,
        prototypes=prototypes,
        residuals=residual_storage.tier,
    )
    restored_ids = tuple(sorted(restored))
    following = RepairCacheState(
        full_state=state.full_state,
        mosaic_state=mosaic_state,
        residual_storage=residual_storage,
        prototype_catalog=state.prototype_catalog,
        active_prototype_records=active_record_tuple,
        active_prototype_ids=plan.active_prototype_ids,
        exact_node_ids=exact_ids,
        initial_exact_node_ids=state.initial_exact_node_ids,
        promoted_node_ids=tuple(sorted((*state.promoted_node_ids, *restored_ids))),
        superseded_prototype_ids=tuple(
            sorted((*state.superseded_prototype_ids, *plan.superseded_prototype_ids))
        ),
        budget_evicted_prototype_ids=tuple(
            sorted((*state.budget_evicted_prototype_ids, *plan.budget_evicted_prototype_ids))
        ),
        active_budget=state.active_budget,
        events=state.events,
    )
    restored_bytes = sum(state.full_state.blocks[node_id].byte_size for node_id in restored_ids)
    return (
        following,
        restored_ids,
        restored_bytes,
        transfer.transfer_time_ms,
        transfer.asynchronous,
    )


def _timed_redecode(
    callback: ReDecodeCallback,
    state: RepairCacheState,
    provisional_logits: Any,
) -> tuple[Any, float]:
    cuda_tensor = (
        provisional_logits.__class__.__module__.startswith("torch")
        and hasattr(provisional_logits, "device")
        and provisional_logits.device.type == "cuda"
    )
    if cuda_tensor:
        import torch

        device = provisional_logits.device
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = callback(state)
        end.record()
        torch.cuda.synchronize(device)
        return logits, float(start.elapsed_time(end))
    started = time.perf_counter()
    logits = callback(state)
    return logits, (time.perf_counter() - started) * 1000.0


def _token_id(logits: Any) -> int:
    return int(np.argmax(_logit_vector(logits)))


def _maximum_logit_change(first: Any, second: Any) -> float:
    first_values = _logit_vector(first)
    second_values = _logit_vector(second)
    if first_values.shape != second_values.shape:
        raise ValueError("re-decoded logits changed vocabulary shape")
    return float(np.max(np.abs(second_values - first_values)))


def _quality_recovered(
    provisional_token: int,
    final_token: int,
    reference_token_id: int | None,
) -> bool | None:
    if reference_token_id is None:
        return None
    if reference_token_id < 0:
        raise ValueError("reference token ID must be nonnegative")
    return provisional_token != reference_token_id and final_token == reference_token_id


def _event_without_repair(
    state: RepairCacheState,
    config: RepairConfig,
    step_index: int,
    logits: Any,
    reason: RepairTriggerReason,
    signals: RepairStepSignals,
    reference_token_id: int | None,
) -> RepairStepResult:
    token = _token_id(logits)
    event = RepairEvent(
        policy=config.policy,
        trigger_reason=reason,
        trigger_step=step_index,
        triggered=False,
        restored_block_ids=(),
        restored_bytes=0,
        transfer_time_ms=0.0,
        transfer_was_asynchronous=False,
        re_decode_time_ms=0.0,
        re_decode_count=0,
        evicted_prototype_ids=(),
        superseded_prototype_ids=(),
        active_budget_unit=state.active_budget.unit,
        active_budget_value=state.active_budget.value,
        active_cost_before=state.active_cost,
        active_cost_after=state.active_cost,
        maximum_logit_change=0.0,
        token_changed=False,
        provisional_token_id=token,
        final_token_id=token,
        quality_recovered=_quality_recovered(token, token, reference_token_id),
        signals=signals,
    )
    following = replace(state, events=(*state.events, event))
    return RepairStepResult(following, logits, logits, event)


def _repair_decode_step(
    state: RepairCacheState,
    config: RepairConfig,
    *,
    step_index: int,
    provisional_logits: Any,
    prototype_attention_mass: Mapping[int, float],
    re_decode: ReDecodeCallback,
    draft_distribution: Any | None,
    oracle_should_repair: bool | None,
    reference_token_id: int | None,
    key_transform: RotaryPrototypeKeyTransform | None,
) -> RepairStepResult:
    if step_index < 0:
        raise ValueError("repair step index must be nonnegative")
    if state.events and step_index <= state.events[-1].trigger_step:
        raise ValueError("each decode step can be processed for repair at most once")
    signals = calculate_repair_signals(
        state,
        provisional_logits,
        prototype_attention_mass,
        draft_distribution=draft_distribution,
        normalize_prototype_risk=config.normalize_prototype_risk,
    )
    reason = _trigger_reason(config, signals, oracle_should_repair=oracle_should_repair)
    if not _reason_triggers(reason):
        return _event_without_repair(
            state,
            config,
            step_index,
            provisional_logits,
            reason,
            signals,
            reference_token_id,
        )
    if not state.residual_storage.payloads or config.max_blocks_per_step == 0:
        return _event_without_repair(
            state,
            config,
            step_index,
            provisional_logits,
            RepairTriggerReason.NO_ELIGIBLE_RESIDUAL,
            signals,
            reference_token_id,
        )
    plan = _plan_repair(state, signals, config)
    if plan is None:
        return _event_without_repair(
            state,
            config,
            step_index,
            provisional_logits,
            RepairTriggerReason.BUDGET_INFEASIBLE,
            signals,
            reference_token_id,
        )

    active_before = state.active_cost
    promoted, restored_ids, restored_bytes, transfer_ms, asynchronous = _promote(
        state,
        plan,
        key_transform,
    )
    repaired_logits, re_decode_ms = _timed_redecode(
        re_decode,
        promoted,
        provisional_logits,
    )
    provisional_token = _token_id(provisional_logits)
    final_token = _token_id(repaired_logits)
    event = RepairEvent(
        policy=config.policy,
        trigger_reason=reason,
        trigger_step=step_index,
        triggered=True,
        restored_block_ids=restored_ids,
        restored_bytes=restored_bytes,
        transfer_time_ms=transfer_ms,
        transfer_was_asynchronous=asynchronous,
        re_decode_time_ms=re_decode_ms,
        re_decode_count=1,
        evicted_prototype_ids=plan.budget_evicted_prototype_ids,
        superseded_prototype_ids=plan.superseded_prototype_ids,
        active_budget_unit=state.active_budget.unit,
        active_budget_value=state.active_budget.value,
        active_cost_before=active_before,
        active_cost_after=promoted.active_cost,
        maximum_logit_change=_maximum_logit_change(provisional_logits, repaired_logits),
        token_changed=provisional_token != final_token,
        provisional_token_id=provisional_token,
        final_token_id=final_token,
        quality_recovered=_quality_recovered(
            provisional_token,
            final_token,
            reference_token_id,
        ),
        signals=signals,
    )
    following = replace(promoted, events=(*state.events, event))
    return RepairStepResult(following, provisional_logits, repaired_logits, event)


def repair_decode_step(
    state: RepairCacheState,
    config: RepairConfig,
    *,
    step_index: int,
    provisional_logits: Any,
    prototype_attention_mass: Mapping[int, float],
    re_decode: ReDecodeCallback,
    draft_distribution: Any | None = None,
    key_transform: RotaryPrototypeKeyTransform | None = None,
) -> RepairStepResult:
    """Apply an online repair policy; oracle/reference data is not accepted here."""

    if config.policy is RepairPolicy.ORACLE:
        raise ValueError("oracle repair is isolated in mosaickv.repair.oracle")
    return _repair_decode_step(
        state,
        config,
        step_index=step_index,
        provisional_logits=provisional_logits,
        prototype_attention_mass=prototype_attention_mass,
        re_decode=re_decode,
        draft_distribution=draft_distribution,
        oracle_should_repair=None,
        reference_token_id=None,
        key_transform=key_transform,
    )


__all__ = [
    "ReDecodeCallback",
    "calculate_repair_signals",
    "draft_kl_divergence",
    "normalized_next_token_entropy",
    "repair_decode_step",
]
