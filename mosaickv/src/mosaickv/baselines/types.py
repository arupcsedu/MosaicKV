"""Typed exact-tier baseline decisions and plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from mosaickv.cache_state import FullKVState, KVBlockDescriptor, Modality, MosaicKVState
from mosaickv.selection import SelectionBudget
from mosaickv.types import BudgetUnit, MosaicKVMethod

BaselineStratum = tuple[int, int, Modality]


class BaselineSelectionReason(StrEnum):
    """Machine-readable reason for every exact baseline block decision."""

    MANDATORY_EXACT = "mandatory_exact"
    RETENTION_ONE_EXACT = "retention_one_exact"
    RANDOM_SEEDED = "random_seeded"
    UNIFORM_STRATUM = "uniform_layer_head_modality"
    PROMPT_ATTENTION_TOPK = "prompt_attention_topk"
    VALUE_NOVELTY_TOPK = "value_novelty_topk"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class BaselineSelectionDecision:
    """One complete baseline selection disposition."""

    node_id: int
    selected: bool
    selection_rank: int | None
    score: float
    cost: int
    reason: BaselineSelectionReason
    stratum: BaselineStratum

    def __post_init__(self) -> None:
        if self.node_id < 0 or self.cost < 1:
            raise ValueError("baseline decision node_id and cost must be positive-domain values")
        if not math.isfinite(self.score):
            raise ValueError("baseline decision score must be finite")
        if self.selected != (self.selection_rank is not None):
            raise ValueError("selected baseline decisions require exactly one rank")
        selected_reasons = {
            BaselineSelectionReason.MANDATORY_EXACT,
            BaselineSelectionReason.RETENTION_ONE_EXACT,
            BaselineSelectionReason.RANDOM_SEEDED,
            BaselineSelectionReason.UNIFORM_STRATUM,
            BaselineSelectionReason.PROMPT_ATTENTION_TOPK,
            BaselineSelectionReason.VALUE_NOVELTY_TOPK,
        }
        if (self.reason in selected_reasons) != self.selected:
            raise ValueError("baseline selection reason contradicts selected state")


@dataclass(frozen=True, slots=True)
class BaselineSelectionResult:
    """Exact-only selected block set with hard-budget and seed provenance."""

    method: MosaicKVMethod
    source_blocks: tuple[KVBlockDescriptor, ...]
    selected_blocks: tuple[KVBlockDescriptor, ...]
    selection_order: tuple[int, ...]
    decisions: tuple[BaselineSelectionDecision, ...]
    budget: SelectionBudget
    budget_spent: int
    active_bytes: int
    seed: int
    score_provenance: str

    def __post_init__(self) -> None:
        if not self.method.is_compressed_baseline:
            raise ValueError("baseline selection requires an exact compressed baseline method")
        if self.seed < 0:
            raise ValueError("baseline seed must be nonnegative")
        if not self.score_provenance.strip():
            raise ValueError("baseline score provenance must be non-empty")
        if tuple(decision.node_id for decision in self.decisions) != tuple(
            range(len(self.decisions))
        ):
            raise ValueError("baseline decisions must cover every ordered source block")
        if len(self.source_blocks) != len(self.decisions):
            raise ValueError("baseline source blocks and decisions must align")
        selected_ids = tuple(decision.node_id for decision in self.decisions if decision.selected)
        if len(set(self.selection_order)) != len(self.selection_order):
            raise ValueError("baseline selection order cannot contain duplicate nodes")
        if tuple(sorted(self.selection_order)) != selected_ids:
            raise ValueError("baseline selection order and selected decisions differ")
        if len(self.selected_blocks) != len(selected_ids):
            raise ValueError("baseline selected blocks do not align with selected decisions")
        if self.selected_blocks != tuple(self.source_blocks[node_id] for node_id in selected_ids):
            raise ValueError("baseline selected blocks differ from their source node IDs")
        ranks = sorted(
            decision.selection_rank
            for decision in self.decisions
            if decision.selection_rank is not None
        )
        if ranks != list(range(len(selected_ids))):
            raise ValueError("baseline selection ranks must be contiguous")
        if self.budget_spent < 0 or self.budget_spent > self.budget.value:
            raise ValueError("baseline selection exceeds its hard budget")
        expected_bytes = sum(block.byte_size for block in self.selected_blocks)
        if self.active_bytes != expected_bytes:
            raise ValueError("baseline active bytes do not match selected block storage")
        if self.budget.unit is BudgetUnit.BYTES and self.active_bytes != self.budget_spent:
            raise ValueError("byte-budget baseline cost must equal exact selected bytes")
        for decision, block in zip(self.decisions, self.source_blocks, strict=True):
            mandatory_reason = decision.reason is BaselineSelectionReason.MANDATORY_EXACT
            if block.mandatory != mandatory_reason:
                raise ValueError("baseline mandatory block reason is inconsistent")

    @property
    def selected_node_ids(self) -> tuple[int, ...]:
        """Selected source node IDs in canonical node order."""

        return tuple(decision.node_id for decision in self.decisions if decision.selected)


@dataclass(frozen=True, slots=True)
class BaselineCompressionPlan:
    """Exact-tier baseline plan consumed by the shared HF cache packer."""

    method: MosaicKVMethod
    full_state: FullKVState
    state: MosaicKVState
    selection: BaselineSelectionResult
    source_budget_value: int
    active_budget_value: int
    selection_seconds: float
    tier_seconds: float

    def __post_init__(self) -> None:
        if self.method is not self.selection.method:
            raise ValueError("baseline plan method and selection method differ")
        if self.state.prototypes.source_blocks or self.state.residuals.source_blocks:
            raise ValueError("simple baselines cannot create prototype or residual tiers")
        if self.state.exact.blocks != self.selection.selected_blocks:
            raise ValueError("baseline exact tier does not match selected source blocks")
        if self.state.statistics.active_kv_bytes != self.selection.active_bytes:
            raise ValueError("baseline plan active byte accounting is inconsistent")
        if self.source_budget_value < 1 or self.active_budget_value < 1:
            raise ValueError("baseline source and active budgets must be positive")
        if self.active_budget_value > self.source_budget_value:
            raise ValueError("baseline active budget cannot exceed source budget")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.selection_seconds, self.tier_seconds)
        ):
            raise ValueError("baseline planning timings must be finite and nonnegative")


__all__ = [
    "BaselineCompressionPlan",
    "BaselineSelectionDecision",
    "BaselineSelectionReason",
    "BaselineSelectionResult",
    "BaselineStratum",
]
