"""Typed decode-time residual-repair state, diagnostics, and event records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mosaickv.cache_state import FullKVState, MosaicKVState
from mosaickv.prototypes.types import PrototypeRecord, ThreeTierCacheConstruction
from mosaickv.residual.types import ResidualStorageReport
from mosaickv.selection.types import SelectionBudget
from mosaickv.types import BudgetUnit, JsonObject, RepairPolicy


class RepairTriggerReason(StrEnum):
    """Machine-readable outcome of one provisional trigger decision."""

    DISABLED = "repair_disabled"
    POLICY_NONE = "policy_none"
    THRESHOLD_NOT_MET = "threshold_not_met"
    ENTROPY = "entropy_threshold"
    PROTOTYPE_RISK = "prototype_risk_threshold"
    ENTROPY_AND_PROTOTYPE_RISK = "entropy_and_prototype_risk_thresholds"
    ORACLE = "oracle_evaluation_only"
    ORACLE_NO_REPAIR = "oracle_no_repair"
    NO_ELIGIBLE_RESIDUAL = "no_eligible_residual"
    BUDGET_INFEASIBLE = "active_budget_infeasible"


@dataclass(frozen=True, slots=True)
class RepairStepSignals:
    """Uncertainty and prototype-attention signals from one provisional step."""

    normalized_entropy: float
    total_prototype_attention_mass: float
    prototype_attention_masses: tuple[tuple[int, float], ...]
    prototype_risks: tuple[tuple[int, float], ...]
    maximum_prototype_risk: float
    draft_kl_divergence: float | None

    def __post_init__(self) -> None:
        if not 0 <= self.normalized_entropy <= 1:
            raise ValueError("normalized next-token entropy must lie in [0, 1]")
        if not 0 <= self.total_prototype_attention_mass <= 1 + 1e-9:
            raise ValueError("total prototype attention mass must lie in [0, 1]")
        mass_ids = tuple(prototype_id for prototype_id, _mass in self.prototype_attention_masses)
        if mass_ids != tuple(sorted(set(mass_ids))):
            raise ValueError("prototype attention masses must have sorted unique IDs")
        if any(
            not math.isfinite(mass) or not 0 <= mass <= 1
            for _prototype_id, mass in self.prototype_attention_masses
        ):
            raise ValueError("per-prototype attention masses must lie in [0, 1]")
        if not math.isclose(
            sum(mass for _prototype_id, mass in self.prototype_attention_masses),
            self.total_prototype_attention_mass,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("per-prototype attention masses do not match their total")
        ids = tuple(prototype_id for prototype_id, _risk in self.prototype_risks)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("prototype risks must have sorted unique IDs")
        if ids != mass_ids:
            raise ValueError("prototype attention masses and risks must cover the same IDs")
        if any(not math.isfinite(risk) or risk < 0 for _prototype_id, risk in self.prototype_risks):
            raise ValueError("prototype risks must be finite and nonnegative")
        expected_maximum = max((risk for _prototype_id, risk in self.prototype_risks), default=0.0)
        if not math.isclose(
            self.maximum_prototype_risk, expected_maximum, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("maximum prototype risk does not match per-prototype risks")
        if self.draft_kl_divergence is not None and (
            not math.isfinite(self.draft_kl_divergence) or self.draft_kl_divergence < 0
        ):
            raise ValueError("draft KL divergence must be finite and nonnegative")

    def risk_for(self, prototype_id: int) -> float:
        for candidate, risk in self.prototype_risks:
            if candidate == prototype_id:
                return risk
        return 0.0

    def attention_mass_for(self, prototype_id: int) -> float:
        for candidate, mass in self.prototype_attention_masses:
            if candidate == prototype_id:
                return mass
        return 0.0


@dataclass(frozen=True, slots=True)
class RepairEvent:
    """Auditable outcome and cost of one provisional decode-step decision."""

    policy: RepairPolicy
    trigger_reason: RepairTriggerReason
    trigger_step: int
    triggered: bool
    restored_block_ids: tuple[int, ...]
    restored_bytes: int
    transfer_time_ms: float
    transfer_was_asynchronous: bool
    re_decode_time_ms: float
    re_decode_count: int
    evicted_prototype_ids: tuple[int, ...]
    superseded_prototype_ids: tuple[int, ...]
    active_budget_unit: BudgetUnit
    active_budget_value: int
    active_cost_before: int
    active_cost_after: int
    maximum_logit_change: float
    token_changed: bool
    provisional_token_id: int
    final_token_id: int
    quality_recovered: bool | None
    signals: RepairStepSignals

    def __post_init__(self) -> None:
        if self.trigger_step < 0:
            raise ValueError("repair trigger step must be nonnegative")
        if self.restored_block_ids != tuple(sorted(set(self.restored_block_ids))):
            raise ValueError("restored block IDs must be sorted and unique")
        if self.evicted_prototype_ids != tuple(sorted(set(self.evicted_prototype_ids))):
            raise ValueError("evicted prototype IDs must be sorted and unique")
        if self.superseded_prototype_ids != tuple(sorted(set(self.superseded_prototype_ids))):
            raise ValueError("superseded prototype IDs must be sorted and unique")
        nonnegative = (
            self.restored_bytes,
            self.transfer_time_ms,
            self.re_decode_time_ms,
            self.re_decode_count,
            self.active_budget_value,
            self.active_cost_before,
            self.active_cost_after,
            self.maximum_logit_change,
            self.provisional_token_id,
            self.final_token_id,
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in nonnegative):
            raise ValueError("repair event numeric fields must be finite and nonnegative")
        if self.re_decode_count not in {0, 1}:
            raise ValueError("one repair event may recompute the current token at most once")
        if self.triggered != bool(self.restored_block_ids):
            raise ValueError("triggered repair events must restore at least one block")
        if self.triggered != (self.re_decode_count == 1):
            raise ValueError("a triggered repair event must re-decode exactly once")
        if not self.triggered and (
            self.restored_bytes
            or self.transfer_time_ms
            or self.re_decode_time_ms
            or self.evicted_prototype_ids
            or self.superseded_prototype_ids
        ):
            raise ValueError("non-triggered repair events cannot mutate cache state")
        if self.active_cost_after > self.active_budget_value:
            raise ValueError("repair event exceeds the active cache budget")
        if self.token_changed != (self.provisional_token_id != self.final_token_id):
            raise ValueError("token-change flag does not match recorded token IDs")
        if self.maximum_logit_change < 0 or not math.isfinite(self.maximum_logit_change):
            raise ValueError("logit change must be finite and nonnegative")

    def to_record(self) -> JsonObject:
        """Return a JSON-compatible per-step record for raw evaluation output."""

        return {
            "policy": self.policy.value,
            "trigger_reason": self.trigger_reason.value,
            "trigger_step": self.trigger_step,
            "triggered": self.triggered,
            "restored_block_ids": list(self.restored_block_ids),
            "restored_bytes": self.restored_bytes,
            "transfer_time_ms": self.transfer_time_ms,
            "transfer_was_asynchronous": self.transfer_was_asynchronous,
            "re_decode_time_ms": self.re_decode_time_ms,
            "re_decode_count": self.re_decode_count,
            "evicted_prototype_ids": list(self.evicted_prototype_ids),
            "superseded_prototype_ids": list(self.superseded_prototype_ids),
            "active_budget_unit": self.active_budget_unit.value,
            "active_budget_value": self.active_budget_value,
            "active_cost_before": self.active_cost_before,
            "active_cost_after": self.active_cost_after,
            "maximum_logit_change": self.maximum_logit_change,
            "token_changed": self.token_changed,
            "provisional_token_id": self.provisional_token_id,
            "final_token_id": self.final_token_id,
            "quality_recovered": self.quality_recovered,
            "normalized_entropy": self.signals.normalized_entropy,
            "total_prototype_attention_mass": (self.signals.total_prototype_attention_mass),
            "prototype_attention_masses": [
                [prototype_id, mass]
                for prototype_id, mass in self.signals.prototype_attention_masses
            ],
            "prototype_risks": [
                [prototype_id, risk] for prototype_id, risk in self.signals.prototype_risks
            ],
            "maximum_prototype_risk": self.signals.maximum_prototype_risk,
            "draft_kl_divergence": self.signals.draft_kl_divergence,
        }


@dataclass(frozen=True, slots=True)
class RepairCacheState:
    """Persistent active tiers and repair history across decode steps."""

    full_state: FullKVState
    mosaic_state: MosaicKVState
    residual_storage: ResidualStorageReport
    prototype_catalog: tuple[PrototypeRecord, ...]
    active_prototype_ids: tuple[int, ...]
    exact_node_ids: tuple[int, ...]
    initial_exact_node_ids: tuple[int, ...]
    promoted_node_ids: tuple[int, ...]
    superseded_prototype_ids: tuple[int, ...]
    budget_evicted_prototype_ids: tuple[int, ...]
    active_budget: SelectionBudget
    events: tuple[RepairEvent, ...] = ()

    def __post_init__(self) -> None:
        source_count = len(self.full_state.blocks)
        ordered_sets = (
            self.active_prototype_ids,
            self.exact_node_ids,
            self.initial_exact_node_ids,
            self.promoted_node_ids,
            self.superseded_prototype_ids,
            self.budget_evicted_prototype_ids,
        )
        if any(values != tuple(sorted(set(values))) for values in ordered_sets):
            raise ValueError("repair state identifier sets must be sorted and unique")
        if any(node_id < 0 or node_id >= source_count for node_id in self.exact_node_ids):
            raise ValueError("repair exact node ID lies outside the source block table")
        if not set(self.initial_exact_node_ids) <= set(self.exact_node_ids):
            raise ValueError("repair cannot remove initially exact blocks")
        if not set(self.promoted_node_ids) <= set(self.exact_node_ids):
            raise ValueError("promoted blocks must remain exact")
        if set(self.initial_exact_node_ids) & set(self.promoted_node_ids):
            raise ValueError("initial exact and promoted block IDs cannot overlap")
        catalog_ids = {record.prototype_id for record in self.prototype_catalog}
        if catalog_ids != set(range(len(self.prototype_catalog))):
            raise ValueError("repair prototype catalog IDs must be contiguous")
        removed = set(self.superseded_prototype_ids) | set(self.budget_evicted_prototype_ids)
        if set(self.superseded_prototype_ids) & set(self.budget_evicted_prototype_ids):
            raise ValueError("a prototype cannot be both superseded and budget-evicted")
        if set(self.active_prototype_ids) & removed:
            raise ValueError("removed prototypes cannot remain active")
        if (set(self.active_prototype_ids) | removed) != catalog_ids:
            raise ValueError("every prototype must be active, superseded, or budget-evicted")
        if len(self.mosaic_state.prototypes.prototype_keys) != len(self.active_prototype_ids):
            raise ValueError("active prototype IDs and payload counts differ")
        expected_exact = tuple(self.full_state.blocks[node_id] for node_id in self.exact_node_ids)
        if self.mosaic_state.exact.blocks != expected_exact:
            raise ValueError("repair exact IDs do not match exact-tier blocks")
        expected_prototype_blocks = tuple(
            self.full_state.blocks[node_id]
            for prototype_id in self.active_prototype_ids
            for node_id in self.prototype_catalog[prototype_id].assigned_node_ids
        )
        if self.mosaic_state.prototypes.source_blocks != expected_prototype_blocks:
            raise ValueError("active repair prototypes do not match source memberships")
        if self.mosaic_state.residuals is not self.residual_storage.tier:
            raise ValueError("repair state residual tier and storage report must be identical")
        if any(
            payload.prototype_id not in catalog_ids for payload in self.residual_storage.payloads
        ):
            raise ValueError("residual storage references an unknown prototype")
        for node_id in self.promoted_node_ids:
            parents = {
                record.prototype_id
                for record in self.prototype_catalog
                if node_id in record.assigned_node_ids
            }
            if len(parents) != 1 or not parents <= set(self.superseded_prototype_ids):
                raise ValueError("promoted block does not belong to one superseded prototype")
        if self.mosaic_state.logical_positions != self.full_state.logical_positions:
            raise ValueError("repair changed the source logical-position map")
        if self.active_cost > self.active_budget.value:
            raise ValueError("repair cache state exceeds its active budget")
        if any(
            previous.trigger_step >= following.trigger_step
            for previous, following in zip(self.events, self.events[1:], strict=False)
        ):
            raise ValueError("repair event steps must be strictly increasing")

    @classmethod
    def from_construction(
        cls,
        full_state: FullKVState,
        construction: ThreeTierCacheConstruction,
    ) -> RepairCacheState:
        """Initialize persistent repair state from a completed tier construction."""

        if construction.state.source_blocks != full_state.blocks:
            raise ValueError("repair source does not match the constructed cache")
        if construction.prototypes and not construction.adapter_declares_residual_repair:
            raise ValueError("adapter does not declare decode-time residual repair support")
        return cls(
            full_state=full_state,
            mosaic_state=construction.state,
            residual_storage=construction.residual_storage,
            prototype_catalog=construction.prototypes,
            active_prototype_ids=tuple(record.prototype_id for record in construction.prototypes),
            exact_node_ids=construction.exact_node_ids,
            initial_exact_node_ids=construction.exact_node_ids,
            promoted_node_ids=(),
            superseded_prototype_ids=(),
            budget_evicted_prototype_ids=(),
            active_budget=construction.active_budget,
        )

    @property
    def active_cost(self) -> int:
        if self.active_budget.unit is BudgetUnit.BLOCKS:
            return len(self.mosaic_state.exact.blocks) + len(
                self.mosaic_state.prototypes.prototype_keys
            )
        if self.active_budget.unit is BudgetUnit.RETAINED_SLOTS:
            return sum(block.position_count for block in self.mosaic_state.exact.blocks) + len(
                self.mosaic_state.prototypes.prototype_keys
            )
        if self.active_budget.unit is BudgetUnit.BYTES:
            return self.mosaic_state.statistics.active_kv_bytes
        raise ValueError(f"unsupported repair budget unit: {self.active_budget.unit}")

    def prototype_record(self, prototype_id: int) -> PrototypeRecord:
        try:
            return self.prototype_catalog[prototype_id]
        except IndexError as error:
            raise KeyError(f"prototype does not exist: {prototype_id}") from error


@dataclass(frozen=True, slots=True)
class RepairStepResult:
    """Final logits/token and persistent cache state after one repair decision."""

    state: RepairCacheState
    provisional_logits: Any
    final_logits: Any
    event: RepairEvent


__all__ = [
    "RepairCacheState",
    "RepairEvent",
    "RepairStepResult",
    "RepairStepSignals",
    "RepairTriggerReason",
]
