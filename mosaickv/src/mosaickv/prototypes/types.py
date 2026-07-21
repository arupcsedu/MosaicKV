"""Three-tier construction provenance, membership, and diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from mosaickv.adapters.huggingface.types import CachedKeyState
from mosaickv.cache_state import Modality, MosaicKVState
from mosaickv.residual.types import ResidualStorageReport
from mosaickv.selection.types import SelectionBudget
from mosaickv.types import BudgetUnit


class TierConstructionMode(StrEnum):
    """Why a construction produced three tiers or an exact-only fallback."""

    RETENTION_ONE = "retention_one"
    THREE_TIER = "three_tier"
    EXACT_ONLY_DISABLED = "exact_only_prototypes_disabled"
    EXACT_ONLY_UNSAFE = "exact_only_prototype_merge_unsafe"
    EXACT_ONLY_INCOMPATIBLE = "exact_only_incompatible_assignments"


@dataclass(frozen=True, slots=True)
class PrototypeSafetyAssessment:
    """Adapter/source RoPE compatibility decision made before tensor merging."""

    model_family: str
    adapter_cached_key_state: CachedKeyState
    source_cached_key_state: CachedKeyState
    adapter_declares_support: bool
    safe: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.model_family.strip() or not self.reason.strip():
            raise ValueError("prototype safety model family and reason must be non-empty")
        expected_safe = (
            self.adapter_declares_support
            and self.adapter_cached_key_state == self.source_cached_key_state
            and self.source_cached_key_state
            in {CachedKeyState.PRE_ROPE, CachedKeyState.NOT_APPLICABLE}
        )
        if self.safe != expected_safe:
            raise ValueError("prototype safety result contradicts the conservative RoPE policy")


@dataclass(frozen=True, slots=True)
class PrototypeMember:
    """One assigned, unselected source block in a prototype average."""

    node_id: int
    raw_weight: float
    normalized_weight: float

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("prototype member node_id must be nonnegative")
        if not math.isfinite(self.raw_weight) or self.raw_weight <= 0:
            raise ValueError("prototype raw weights must be finite and positive")
        if not math.isfinite(self.normalized_weight) or not 0 < self.normalized_weight <= 1:
            raise ValueError("prototype normalized weights must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class PrototypeDiagnostics:
    """Per-cluster structural and approximation diagnostics."""

    cluster_size: int
    key_dispersion: float
    value_dispersion: float
    modality_composition: tuple[tuple[Modality, int], ...]
    minimum_logical_position: int
    maximum_logical_position: int
    position_span: int
    source_member_bytes: int
    prototype_bytes: int
    active_bytes_saved: int

    def __post_init__(self) -> None:
        if self.cluster_size < 1:
            raise ValueError("a prototype cluster requires at least one assigned member")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.key_dispersion, self.value_dispersion)
        ):
            raise ValueError("prototype dispersions must be finite and nonnegative")
        if sum(count for _modality, count in self.modality_composition) != self.cluster_size:
            raise ValueError("prototype modality composition does not match cluster size")
        if any(count <= 0 for _modality, count in self.modality_composition):
            raise ValueError("prototype modality counts must be positive")
        if len({modality for modality, _count in self.modality_composition}) != len(
            self.modality_composition
        ):
            raise ValueError("prototype modality composition cannot contain duplicates")
        if self.minimum_logical_position < 0:
            raise ValueError("prototype logical positions must be nonnegative")
        if self.maximum_logical_position < self.minimum_logical_position:
            raise ValueError("prototype logical position bounds are inverted")
        if self.position_span != self.maximum_logical_position - self.minimum_logical_position:
            raise ValueError("prototype position span does not match its bounds")
        if self.source_member_bytes <= 0 or self.prototype_bytes <= 0:
            raise ValueError("prototype byte diagnostics are invalid")
        if self.active_bytes_saved != self.source_member_bytes - self.prototype_bytes:
            raise ValueError("prototype active-byte savings do not match payload accounting")

    @property
    def dispersion(self) -> float:
        """Mean K/V dispersion used by the decode-time prototype-risk score."""

        return 0.5 * (self.key_dispersion + self.value_dispersion)


@dataclass(frozen=True, slots=True)
class PrototypeRecord:
    """Anchor, membership weights, and diagnostics for one active prototype."""

    prototype_id: int
    layer: int
    kv_head: int
    anchor_node_id: int
    anchor_logical_position: int
    members: tuple[PrototypeMember, ...]
    assigned_node_ids: tuple[int, ...]
    diagnostics: PrototypeDiagnostics
    eviction_utility: float
    utility_provenance: str

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.prototype_id,
                self.layer,
                self.kv_head,
                self.anchor_node_id,
                self.anchor_logical_position,
            )
        ):
            raise ValueError("prototype identifiers must be nonnegative")
        if self.anchor_node_id in {member.node_id for member in self.members}:
            raise ValueError("the exact anchor cannot also be a prototype source member")
        if len({member.node_id for member in self.members}) != len(self.members):
            raise ValueError("prototype source members must be unique")
        if self.assigned_node_ids != tuple(sorted(member.node_id for member in self.members)):
            raise ValueError("prototype assigned node IDs do not match its members")
        if not math.isclose(
            sum(member.normalized_weight for member in self.members),
            1.0,
            rel_tol=0,
            abs_tol=1e-7,
        ):
            raise ValueError("prototype normalized member weights must sum to one")
        if self.diagnostics.cluster_size != len(self.members):
            raise ValueError("prototype diagnostics cluster size does not match members")
        if not math.isfinite(self.eviction_utility):
            raise ValueError("prototype eviction utility must be finite")
        if not self.utility_provenance.strip():
            raise ValueError("prototype utility provenance must be non-empty")


@dataclass(frozen=True, slots=True)
class ActiveHeadLayout:
    """Physical active length and logical provenance for one layer/KV head."""

    layer: int
    kv_head: int
    exact_logical_positions: tuple[int, ...]
    prototype_ids: tuple[int, ...]
    prototype_anchor_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kv_head < 0:
            raise ValueError("active layout layer/head must be nonnegative")
        if len(self.prototype_ids) != len(self.prototype_anchor_positions):
            raise ValueError("prototype IDs and anchor positions must align")
        if len(set(self.prototype_ids)) != len(self.prototype_ids):
            raise ValueError("active layout prototype IDs must be unique")
        if any(position < 0 for position in self.exact_logical_positions):
            raise ValueError("active exact logical positions must be nonnegative")

    @property
    def active_cache_length(self) -> int:
        return len(self.exact_logical_positions) + len(self.prototype_ids)


@dataclass(frozen=True, slots=True)
class ThreeTierCacheConstruction:
    """Complete active tiers, CPU residuals, safety decision, and layouts."""

    state: MosaicKVState
    mode: TierConstructionMode
    reason: str
    safety: PrototypeSafetyAssessment
    adapter_declares_residual_repair: bool
    active_budget: SelectionBudget
    exact_node_ids: tuple[int, ...]
    prototypes: tuple[PrototypeRecord, ...]
    residual_storage: ResidualStorageReport
    active_layouts: tuple[ActiveHeadLayout, ...]
    original_logical_sequence_length: int
    next_decode_position: int

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("three-tier construction reason must be non-empty")
        if self.exact_node_ids != tuple(sorted(set(self.exact_node_ids))):
            raise ValueError("exact node IDs must be sorted and unique")
        if any(
            node_id < 0 or node_id >= len(self.state.source_blocks)
            for node_id in self.exact_node_ids
        ):
            raise ValueError("exact node ID lies outside the source block table")
        expected_exact = tuple(self.state.source_blocks[node_id] for node_id in self.exact_node_ids)
        if self.state.exact.blocks != expected_exact:
            raise ValueError("exact node IDs do not match exact-tier source blocks")
        if tuple(record.prototype_id for record in self.prototypes) != tuple(
            range(len(self.prototypes))
        ):
            raise ValueError("prototype IDs must be contiguous")
        if self.original_logical_sequence_length <= 0:
            raise ValueError("original logical sequence length must be positive")
        if self.next_decode_position < self.original_logical_sequence_length:
            raise ValueError("next decode position cannot precede the logical sequence")
        if self.state.original_logical_sequence_length != self.original_logical_sequence_length:
            raise ValueError("three-tier state changed original logical sequence length")
        if self.state.next_decode_position != self.next_decode_position:
            raise ValueError("three-tier state changed next decode position")
        if self.state.statistics.residual_kv_bytes != self.residual_storage.cpu_bytes:
            raise ValueError("three-tier residual accounting differs from storage report")
        if self.active_budget.unit is BudgetUnit.BLOCKS:
            active_cost = len(self.state.exact.blocks) + len(self.state.prototypes.prototype_keys)
        elif self.active_budget.unit is BudgetUnit.RETAINED_SLOTS:
            active_cost = sum(block.position_count for block in self.state.exact.blocks) + len(
                self.state.prototypes.prototype_keys
            )
        elif self.active_budget.unit is BudgetUnit.BYTES:
            active_cost = self.state.statistics.active_kv_bytes
        else:  # pragma: no cover - enum guards this
            raise ValueError(f"unsupported active budget unit: {self.active_budget.unit}")
        if active_cost > self.active_budget.value:
            raise ValueError("constructed tiers exceed their active cache budget")
        prototype_node_ids = tuple(
            node_id for record in self.prototypes for node_id in record.assigned_node_ids
        )
        expected_prototype_blocks = tuple(
            self.state.source_blocks[node_id] for node_id in prototype_node_ids
        )
        if self.state.prototypes.source_blocks != expected_prototype_blocks:
            raise ValueError("prototype records do not match prototype-tier source blocks")
        expected_assignments = tuple(
            record.prototype_id
            for record in self.prototypes
            for _node_id in record.assigned_node_ids
        )
        if self.state.prototypes.assignments != expected_assignments:
            raise ValueError("prototype records do not match tier assignments")
        if len(self.state.prototypes.prototype_keys) != len(self.prototypes):
            raise ValueError("prototype records and active payload counts differ")
        residual_blocks = self.residual_storage.tier.source_blocks
        if self.state.residuals is not self.residual_storage.tier:
            raise ValueError("state residual tier and residual storage report must be identical")
        if residual_blocks and residual_blocks != self.state.prototypes.source_blocks:
            raise ValueError("residual storage must preserve every prototyped source block")
        source_layouts = {(block.layer, block.kv_head) for block in self.state.source_blocks}
        active_layout_ids = tuple((layout.layer, layout.kv_head) for layout in self.active_layouts)
        if len(set(active_layout_ids)) != len(active_layout_ids):
            raise ValueError("active layer/head layouts must be unique")
        if set(active_layout_ids) != source_layouts:
            raise ValueError("active layouts must cover every source layer and KV head")
        if self.mode is TierConstructionMode.RETENTION_ONE:
            if self.prototypes or self.residual_storage.tier.source_blocks:
                raise ValueError("retention-one construction cannot transform cache tiers")
            if not self.state.is_retention_one:
                raise ValueError("retention-one construction is not an exact full cache")
        if self.mode is TierConstructionMode.THREE_TIER and not self.safety.safe:
            raise ValueError("three-tier mode requires a safe prototype assessment")
        if self.mode is TierConstructionMode.THREE_TIER and not self.prototypes:
            raise ValueError("three-tier mode requires at least one prototype")
        if self.mode is not TierConstructionMode.THREE_TIER and (
            self.prototypes or self.residual_storage.tier.source_blocks
        ):
            raise ValueError("exact-only construction cannot contain transformed tiers")

    @property
    def active_device_bytes(self) -> int:
        return self.state.statistics.active_kv_bytes

    @property
    def cpu_residual_bytes(self) -> int:
        return self.state.statistics.residual_kv_bytes


__all__ = [
    "ActiveHeadLayout",
    "PrototypeDiagnostics",
    "PrototypeMember",
    "PrototypeRecord",
    "PrototypeSafetyAssessment",
    "ThreeTierCacheConstruction",
    "TierConstructionMode",
]
