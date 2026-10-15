"""Unified eager Hugging Face runtime for FullKV and MosaicKV variants."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from mosaickv.adapters.huggingface import (
    AdapterCapabilities,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
    HuggingFaceMultimodalAdapter,
    ParityReport,
    PrefillOutput,
    PreparedInputs,
)
from mosaickv.backends.cache_isolation import (
    IsolatedDecodeStateHandle,
    construct_isolated_decode_state,
)
from mosaickv.backends.native_selector_plan import (
    project_common_selector_to_request_blocks,
)
from mosaickv.baselines import (
    BaselineCompressionPlan,
    HeadLocalCapacityRecord,
    HeadLocalPromptSelectionAlternative,
    LookMCompressionPlan,
    PrefixKVCompressionPlan,
    PrefixKVOfflineProfile,
    PublishedKVCompressionPlan,
    VLCacheCompressionPlan,
    assert_vl_cache_calibration_disjoint,
    build_exact_baseline_plan,
    build_lookm_reimpl_plan,
    build_prefixkv_reimpl_plan,
    build_published_kv_reimpl_plan,
    build_vl_cache_reimpl_plan,
    coherent_layout_groups,
    load_prefixkv_profile,
    lookm_runtime_payloads,
    prefixkv_runtime_payloads,
    published_kv_runtime_payloads,
    select_prompt_attention_topk_ids,
    select_prompt_attention_topk_ids_by_head,
    vl_cache_runtime_payloads,
)
from mosaickv.cache_state import (
    FullKVState,
    KVBlockDescriptor,
    Modality,
    MosaicKVState,
    tensor_storage_bytes,
)
from mosaickv.config import RunConfig
from mosaickv.evaluation.forecast_diagnostics import (
    AttentionDiagnosticInput,
    AttentionSelectionDiagnostics,
    ForecastQualityDiagnostics,
    evaluate_attention_selection_quality,
    evaluate_forecast_quality,
)
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.evaluation.oracle_queries import (
    EvaluationOnlyOracleQueries,
    collect_evaluation_only_true_future_queries,
)
from mosaickv.evaluation.prefixkv_metrics import TeacherForcedNll
from mosaickv.forecasting import QueryForecast, build_query_forecast, forecast_with_rollout
from mosaickv.forecasting.huggingface import (
    IsolatedQueryRollout,
    run_isolated_greedy_query_rollout,
)
from mosaickv.generation import decode_quality_output, terminal_token_ids
from mosaickv.graph import SparseEvidenceGraph, build_evidence_graph
from mosaickv.measurements.cache_lifecycle import CacheLifecycle
from mosaickv.measurements.timing import ModuleCudaTimer, SynchronizationAudit
from mosaickv.prototypes import (
    PrototypeAssignmentPlan,
    PrototypeRecord,
    ThreeTierCacheConstruction,
    TierConstructionMode,
    construct_three_tier_cache,
    estimate_prototype_active_cost,
    plan_prototype_assignments,
)
from mosaickv.prototypes.rope import RotaryPrototypeKeyTransform
from mosaickv.repair import (
    RepairCacheState,
    RepairEvent,
    evaluate_repair_decode_step,
    repair_decode_step,
    stage_gpu_rescue_payloads,
)
from mosaickv.selection import (
    BlockUtilityTable,
    BoundedHierarchicalProbeDecision,
    BudgetedObjective,
    CapacityLevelKind,
    HierarchicalCapacityProposal,
    HierarchicalProposalKind,
    InfeasibleSelectionBudgetError,
    LogitProbeResult,
    ProbeBudgetTracker,
    RouteBlockCandidate,
    RouteCapacityLadder,
    RouteCapacityViolation,
    SelectionBudget,
    SelectionResult,
    build_route_capacity_ladders,
    compute_block_utilities,
    lazy_greedy_select,
    proposal_route_capacity_violations,
    restrict_route_capacity_ladders_to_position_limits,
    run_bounded_hierarchical_probe_loop,
    select_all_exact,
    select_exact_subset,
    select_verified_capacity_levels,
    vectorized_unit_route_topk_select,
)
from mosaickv.types import (
    AblationVariant,
    BudgetUnit,
    JsonObject,
    JsonValue,
    MosaicKVMethod,
    OutputLengthPolicy,
    RepairPolicy,
)


class HuggingFaceRuntimeError(RuntimeError):
    """Raised when a requested HF runtime state cannot be represented safely."""

    mosaickv_failure_trace_payload: JsonObject | None = None


def _assert_prefixkv_common_hf_representable(
    plan: PrefixKVCompressionPlan, adapter: HuggingFaceMultimodalAdapter | None = None
) -> None:
    """Fail closed for compressed PrefixKV in the common HF transport.

    PrefixKV's published LLaVA implementation owns a per-layer cache
    transaction: it gathers a different number of positions in each layer and
    updates those lengths during decoding.  The generic Transformers adapter
    has one request-level attention-mask/cache extent.  A superficially
    uniform initial plan does not fix that mismatch: PrefixKV's independent
    decode evictions immediately make the layer lengths diverge again.

    Earlier common-runtime runs therefore produced padded or numerically
    invalid states (including NaN logits).  The uncompressed identity path is
    safe. The explicit ragged LLaVA/eager opt-in below admits local byte-matched
    reimplementation runs; paper-native official reproduction still requires
    the isolated upstream transport in ``run_prefixkv_llava_parity.py``.
    """

    if math.isclose(plan.target_retention_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return
    if plan.config.hf_cache_transport == "ragged_llama_eager_v1" and adapter is not None:
        from mosaickv.adapters.huggingface.prefixkv_cache import assert_ragged_support

        assert_ragged_support(adapter)
        return
    capacities = plan.layer_cache_sizes
    raise HuggingFaceRuntimeError(
        "compressed PrefixKV is unsupported in the common Hugging Face cache "
        "transport: PrefixKV requires independent per-layer cache lengths and "
        "decode-time eviction. Use mosaickv/scripts/run_prefixkv_llava_parity.py "
        "with the pinned PrefixKV environment instead; do not use padded common-HF "
        f"execution as PrefixKV evidence (min_capacity={min(capacities)}, "
        f"max_capacity={max(capacities)})."
    )


def _hard_budget_fallback_trace_payload(
    decision_record: JsonObject,
    *,
    attempted_active_kv_bytes: int,
    hard_active_kv_budget_bytes: int,
    attempted_prompt_capacity: int,
    hard_prompt_capacity: int,
    timing_breakdown: JsonObject,
    fullkv_fallback_accounting: ExactCacheStorageAccounting,
    rejected_compressed_proposal_accounting: ExactCacheStorageAccounting,
    rejected_compressed_proposal_id: str,
) -> JsonObject:
    """Build the complete trace emitted when a safe search returns FullKV."""

    if decision_record.get("used_fullkv_fallback") is not True:
        raise ValueError("hard-budget fallback trace requires a FullKV fallback decision")
    if attempted_active_kv_bytes <= hard_active_kv_budget_bytes:
        raise ValueError("fallback attempt must exceed the hard active-byte budget")
    if attempted_prompt_capacity <= hard_prompt_capacity:
        raise ValueError("fallback attempt must exceed the hard prompt capacity")
    if not rejected_compressed_proposal_id:
        raise ValueError("fallback trace requires a rejected compressed proposal ID")
    if fullkv_fallback_accounting.constructed_active_bytes != attempted_active_kv_bytes:
        raise ValueError("FullKV fallback accounting differs from the attempted active bytes")
    fallback_accounting = cast("JsonObject", asdict(fullkv_fallback_accounting))
    rejected_accounting = cast("JsonObject", asdict(rejected_compressed_proposal_accounting))
    return {
        "failure_classification": "hard_budget_fullkv_fallback",
        "effective_method": "full_kv_fallback_rejected_before_packing",
        "used_fullkv_fallback": True,
        "hard_budget_enforced": True,
        "over_budget_cache_packed": False,
        "over_budget_cache_decoded": False,
        "over_budget_cache_scope": "final_selected_fullkv_fallback",
        "isolated_probe_candidates_materialized": True,
        "isolated_probe_states_discarded": True,
        "attempted_active_kv_bytes": attempted_active_kv_bytes,
        "attempted_total_kv_bytes": attempted_active_kv_bytes,
        "attempted_residual_kv_bytes": 0,
        "hard_active_kv_budget_bytes": hard_active_kv_budget_bytes,
        "attempted_prompt_capacity": attempted_prompt_capacity,
        "hard_prompt_capacity": hard_prompt_capacity,
        "knapsack_bytes": fullkv_fallback_accounting.knapsack_bytes,
        "unique_selected_source_bytes": (fullkv_fallback_accounting.unique_selected_source_bytes),
        "planner_tier_bytes": fullkv_fallback_accounting.planner_tier_bytes,
        "external_exact_bytes": fullkv_fallback_accounting.external_exact_bytes,
        "constructed_active_bytes": (fullkv_fallback_accounting.constructed_active_bytes),
        "physical_padding_bytes": fullkv_fallback_accounting.physical_padding_bytes,
        "fullkv_fallback_accounting": fallback_accounting,
        "rejected_compressed_proposal_id": rejected_compressed_proposal_id,
        "rejected_compressed_proposal_accounting": rejected_accounting,
        "target_reached": False,
        "probe_decisions": decision_record.get("evaluations", []),
        "probe_evaluations_used": decision_record.get("evaluations_used"),
        "probe_rounds_used": decision_record.get("rounds_used"),
        "hierarchical_logit_safe_eviction": {
            "status": "failed_closed_fullkv_fallback",
            **decision_record,
        },
        "timing_breakdown": timing_breakdown,
        "byte_budget_materialization": {
            "status": "rejected_before_materialization",
            "attempted_active_kv_bytes": attempted_active_kv_bytes,
            "hard_active_kv_budget_bytes": hard_active_kv_budget_bytes,
            "attempted_over_budget_bytes": (
                attempted_active_kv_bytes - hard_active_kv_budget_bytes
            ),
            "attempted_prompt_capacity": attempted_prompt_capacity,
            "hard_prompt_capacity": hard_prompt_capacity,
            "packed_active_kv_bytes": None,
            "knapsack_bytes": fullkv_fallback_accounting.knapsack_bytes,
            "unique_selected_source_bytes": (
                fullkv_fallback_accounting.unique_selected_source_bytes
            ),
            "planner_tier_bytes": fullkv_fallback_accounting.planner_tier_bytes,
            "external_exact_bytes": fullkv_fallback_accounting.external_exact_bytes,
            "constructed_active_bytes": (fullkv_fallback_accounting.constructed_active_bytes),
            "physical_padding_bytes": fullkv_fallback_accounting.physical_padding_bytes,
            "fullkv_fallback_accounting": fallback_accounting,
            "rejected_compressed_proposal_id": rejected_compressed_proposal_id,
            "rejected_compressed_proposal_accounting": rejected_accounting,
        },
    }


def _route_capacity_failure_trace_payload(
    proposal: HierarchicalCapacityProposal,
    violations: Sequence[RouteCapacityViolation],
    *,
    attempted_active_kv_bytes: int,
    hard_active_kv_budget_bytes: int,
    attempted_prompt_capacity: int,
    hard_prompt_capacity: int,
) -> JsonObject:
    """Build a complete fail-closed trace before an invalid logit probe."""

    if not violations:
        raise ValueError("route-capacity failure requires at least one violation")
    violation_rows = [cast("JsonObject", asdict(item)) for item in violations]
    rejection = cast(
        "JsonObject",
        {
            "candidate_id": proposal.proposal_id,
            "candidate_kind": proposal.kind.value,
            "status": "rejected_before_logit_probe",
            "route_capacity_violations": violation_rows,
        },
    )
    return {
        "failure_classification": "hierarchical_route_capacity_violation",
        "effective_method": "mosaickv_exact_route_infeasible_rejected_before_probe",
        "used_fullkv_fallback": False,
        "hard_budget_enforced": True,
        "over_budget_cache_packed": False,
        "over_budget_cache_decoded": False,
        "over_budget_cache_scope": "route_infeasible_candidate",
        "attempted_active_kv_bytes": attempted_active_kv_bytes,
        "attempted_total_kv_bytes": attempted_active_kv_bytes,
        "attempted_residual_kv_bytes": 0,
        "hard_active_kv_budget_bytes": hard_active_kv_budget_bytes,
        "attempted_prompt_capacity": attempted_prompt_capacity,
        "hard_prompt_capacity": hard_prompt_capacity,
        "target_reached": False,
        "probe_decisions": [rejection],
        "hierarchical_logit_safe_eviction": {
            "status": "rejected_route_capacity_before_logit_probe",
            "used_fullkv_fallback": False,
            "rejected_proposal": rejection,
        },
        "timing_breakdown": {
            "hierarchical_capacity_validation": 0.0,
            "cache_packing": 0.0,
            "decode": 0.0,
        },
        "byte_budget_materialization": {
            "status": "rejected_before_materialization",
            "attempted_active_kv_bytes": attempted_active_kv_bytes,
            "hard_active_kv_budget_bytes": hard_active_kv_budget_bytes,
            "attempted_prompt_capacity": attempted_prompt_capacity,
            "hard_prompt_capacity": hard_prompt_capacity,
            "packed_active_kv_bytes": None,
        },
    }


def _hierarchical_accounting_failure_trace_payload(
    decision_record: JsonObject,
    accounting: ExactCacheStorageAccounting,
    *,
    hard_active_kv_budget_bytes: int,
    elapsed_seconds: float,
) -> JsonObject:
    """Emit the complete preflight trace for a byte-boundary disagreement."""

    if accounting.logical_boundaries_aligned:
        raise ValueError("accounting failure trace requires a logical byte mismatch")
    accounting_record = cast("JsonObject", asdict(accounting))
    accounting_record["constructed_physical_bytes"] = accounting.constructed_physical_bytes
    accounting_record["logical_boundaries_aligned"] = False
    attempted_active_bytes = accounting.constructed_active_bytes
    return {
        "status": "failed",
        "failure_classification": "hierarchical_exact_storage_accounting_mismatch",
        "effective_method": "mosaickv_exact_storage_accounting_rejected_before_final_decode",
        "used_fullkv_fallback": bool(decision_record.get("used_fullkv_fallback", False)),
        "hard_budget_enforced": True,
        "over_budget_cache_packed": False,
        "over_budget_cache_decoded": False,
        "over_budget_cache_scope": "exact_construction_storage_boundary",
        "attempted_active_kv_bytes": attempted_active_bytes,
        "attempted_total_kv_bytes": attempted_active_bytes,
        "attempted_residual_kv_bytes": 0,
        "hard_active_kv_budget_bytes": hard_active_kv_budget_bytes,
        "target_reached": False,
        "probe_decisions": decision_record.get("evaluations", []),
        "knapsack_bytes": accounting.knapsack_bytes,
        "unique_selected_source_bytes": accounting.unique_selected_source_bytes,
        "planner_tier_bytes": accounting.planner_tier_bytes,
        "external_exact_bytes": accounting.external_exact_bytes,
        "constructed_active_bytes": accounting.constructed_active_bytes,
        "physical_padding_bytes": accounting.physical_padding_bytes,
        "hierarchical_logit_safe_eviction": {
            "status": "failed_closed_storage_accounting_mismatch",
            **decision_record,
        },
        "timing_breakdown": {
            "hierarchical_eviction_before_failure": elapsed_seconds,
            "cache_packing": 0.0,
            "decode": 0.0,
        },
        "byte_budget_materialization": {
            "status": "rejected_before_final_decode",
            **accounting_record,
            "hard_active_kv_budget_bytes": hard_active_kv_budget_bytes,
        },
    }


@dataclass(frozen=True, slots=True)
class RuntimePhaseTimings:
    """Single-run phase timings in seconds."""

    prefill: float
    image_video_encoder: float | None
    projector: float | None
    language_model_prefill: float | None
    cache_state_construction: float
    forecast: float
    planner_preparation: float
    graph: float
    utility_and_selection: float
    tier_construction: float
    logit_stability: float
    progressive_eviction: float
    hierarchical_eviction: float
    cache_packing: float
    decode: float
    per_token_decode: tuple[float, ...]
    first_decode_step: float | None
    steady_state_decode_mean: float | None
    repair_transfer: float
    repair_redecode: float
    host_orchestration: float
    total: float

    def __post_init__(self) -> None:
        scalar_values = (
            self.prefill,
            self.cache_state_construction,
            self.forecast,
            self.planner_preparation,
            self.graph,
            self.utility_and_selection,
            self.tier_construction,
            self.logit_stability,
            self.progressive_eviction,
            self.cache_packing,
            self.decode,
            self.repair_transfer,
            self.repair_redecode,
            self.host_orchestration,
            self.total,
        )
        optional_values = (
            self.image_video_encoder,
            self.projector,
            self.language_model_prefill,
            self.first_decode_step,
            self.steady_state_decode_mean,
        )
        if any(not math.isfinite(value) or value < 0 for value in scalar_values):
            raise ValueError("runtime timings must be finite and nonnegative")
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in optional_values
        ):
            raise ValueError("optional runtime timings must be finite and nonnegative")
        if any(not math.isfinite(value) or value < 0 for value in self.per_token_decode):
            raise ValueError("per-token runtime timings must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class FidelityDiagnostic:
    """Evaluation-only comparison against an isolated FullKV decode."""

    token_agreement: float
    sequence_agreement: bool
    logit_kl: float
    attention_output_error: float
    attention_output_error_by_kv_head: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True, slots=True)
class ShortHorizonLogitCandidateEvaluation:
    """Label-blind FullKV-logit evidence for one same-byte cache candidate."""

    candidate_id: str
    passed: bool
    greedy_token_match: bool
    maximum_kl: float
    maximum_logit_difference: float
    evaluated_steps: int


@dataclass(frozen=True, slots=True)
class ShortHorizonLogitStabilityDecision:
    """Fail-closed V98 choice over a bounded deterministic Pareto shortlist."""

    enabled: bool
    shortlist_size: int
    evaluated_candidates: tuple[ShortHorizonLogitCandidateEvaluation, ...]
    selected_candidate_id: str
    selected_candidate_passed: bool
    used_fail_closed_scalar_anchor: bool
    horizon: int
    maximum_kl_threshold: float
    maximum_logit_difference_threshold: float
    fullkv_prefill_reused: bool = True
    second_prefill_performed: bool = False
    probe_states_discarded_before_measured_decode: bool = True
    benchmark_labels_accessed: bool = False


@dataclass(frozen=True, slots=True)
class ProgressiveEvictionCheckpoint:
    """One cumulative FullKV-to-target eviction probe."""

    probe_index: int
    proposed_removed_blocks: int
    accepted: bool
    greedy_token_match: bool
    maximum_kl: float
    maximum_logit_difference: float
    maximum_active_positions: int
    logical_active_bytes: int


@dataclass(frozen=True, slots=True)
class ProgressiveLogitSafeEvictionDecision:
    """Fail-closed V99 progressive eviction result and audit evidence."""

    enabled: bool
    started_from_fullkv: bool
    target_reached: bool
    selected_candidate_passed: bool
    used_target_fallback: bool
    horizon: int
    maximum_kl_threshold: float
    maximum_logit_difference_threshold: float
    maximum_probes: int
    probes_used: int
    accepted_checkpoints: int
    rejected_checkpoints: int
    initial_fullkv_logical_bytes: int
    target_active_bytes: int
    final_logical_active_bytes: int
    protected_group_candidate_count: int
    protected_group_count: int
    protected_group_block_count: int
    protected_group_bytes: int
    protected_group_target_bytes: int
    protected_group_score: float
    checkpoints: tuple[ProgressiveEvictionCheckpoint, ...]
    fullkv_prefill_reused: bool = True
    second_prefill_performed: bool = False
    probe_states_discarded_before_measured_decode: bool = True
    benchmark_labels_accessed: bool = False


@dataclass(frozen=True, slots=True)
class MosaicKVCompressionPlan:
    """All inspectable algorithm outputs produced from one full prefill."""

    method: MosaicKVMethod
    effective_method: str
    full_state: FullKVState
    forecast: QueryForecast | None
    graph: SparseEvidenceGraph
    utilities: BlockUtilityTable
    selection: SelectionResult
    construction: ThreeTierCacheConstruction
    source_budget_value: int
    shared_active_budget_value: int
    active_budget_value: int
    planner_active_budget_value: int
    configured_physical_capacity_fraction: float
    graph_seconds: float
    selection_seconds: float
    utility_seconds: float
    reservation_seconds: float
    selector_seconds: float
    tier_seconds: float
    similarity_candidate_node_ids: tuple[int, ...] | None
    optimization_node_ids: tuple[int, ...] | None
    planner_source_node_ids: tuple[int, ...] | None
    external_exact_node_ids: tuple[int, ...]
    external_exact_capacity_records: tuple[HeadLocalCapacityRecord, ...]
    external_exact_logit_shortlist: tuple[HeadLocalPromptSelectionAlternative, ...]
    prompt_attention_by_node: Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None
    stable_routing_seconds: float
    similarity_shortlist_seconds: float
    planner_mapping_seconds: float
    graph_head_routes: tuple[tuple[int, int], ...]
    graph_routing_provenance: str
    hierarchical_storage_accounting: ExactCacheStorageAccounting | None = None
    hierarchical_rejected_proposal_storage_accounting: ExactCacheStorageAccounting | None = None
    hierarchical_rejected_proposal_id: str | None = None

    @property
    def uses_compact_node_ids(self) -> bool:
        """Whether graph/utility/selection IDs require source-ID translation."""

        return self.planner_source_node_ids is not None

    def source_node_id(self, planner_node_id: int) -> int:
        """Translate one dense planner ID to its immutable FullKV block ID."""

        if planner_node_id < 0 or planner_node_id >= len(self.graph.nodes):
            raise IndexError("planner node ID lies outside the evidence graph")
        return self.graph.nodes[planner_node_id].source_id


@dataclass(frozen=True, slots=True)
class PackedRuntimeCache:
    """Uniform HF tensors plus masks for independently selected KV heads."""

    snapshot: CacheSnapshot
    validity_masks: tuple[Any, ...]
    prompt_capacity: int
    all_slots_valid: bool
    prototype_slots: tuple[tuple[int, int, int, int], ...]
    prototype_attention_biases: tuple[tuple[int, int, int, int, float], ...]
    slot_records: tuple[JsonObject, ...]
    layerwise_logical_positions: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class DensePhysicalCacheBudget:
    """Representable physical prefill budget for the dense HF cache layout."""

    source_bytes: int
    target_bytes: int
    prompt_capacity: int
    position_limits: Mapping[tuple[int, int], int]

    @property
    def bytes_per_position(self) -> int:
        """Physical bytes for one shared sequence slot across all layer/head tensors."""

        if self.prompt_capacity < 1 or self.target_bytes % self.prompt_capacity:
            raise HuggingFaceRuntimeError("dense target bytes are not position-aligned")
        return self.target_bytes // self.prompt_capacity

    def minimum_block_representable_fill(
        self,
        block_size: int,
        *,
        target_bytes: int | None = None,
    ) -> int:
        """Lower bound after unavoidable fixed-block rounding in every identity.

        ``target_bytes`` permits a method to consume less than the shared hard
        cap while retaining the same physical layout.  Validation must compare
        that method against its effective target, not against capacity that it
        deliberately left available to the comparison baseline.
        """

        if block_size < 1:
            raise ValueError("block_size must be positive")
        target = self.target_bytes if target_bytes is None else target_bytes
        if target < 1 or target > self.target_bytes:
            raise ValueError("effective target bytes must lie within the dense hard cap")
        if target % self.bytes_per_position:
            raise ValueError("effective target bytes must be dense-position aligned")
        return max(0, target - (block_size - 1) * self.bytes_per_position)


def _mosaickv_physical_prompt_capacity(
    plan: MosaicKVCompressionPlan,
    dense_budget: DensePhysicalCacheBudget,
    *,
    used_fullkv_fallback: bool,
) -> int:
    """Resolve the physical sequence extent from every active head layout.

    Exact-only and three-tier plans share the same dense HF packer.  Restricting
    capacity shrinking to three-tier mode leaves exact-only plans padded to the
    comparison baseline's larger shared cap, erasing their measured memory
    reduction.  The constructed layouts are authoritative for both modes.
    """

    active_lengths = {layout.active_cache_length for layout in plan.construction.active_layouts}
    if not active_lengths:
        raise HuggingFaceRuntimeError("MosaicKV construction has no active layer/head layouts")
    measured_capacity = max(active_lengths)
    if measured_capacity > dense_budget.prompt_capacity and not used_fullkv_fallback:
        raise HuggingFaceRuntimeError(
            "MosaicKV measured physical capacity exceeds its hard cap: "
            f"active={measured_capacity}, cap={dense_budget.prompt_capacity}"
        )
    return measured_capacity


@dataclass(frozen=True, slots=True)
class ExactCacheStorageAccounting:
    """One exact-cache selection measured at every relevant byte boundary.

    ``knapsack_bytes`` is the controller's predicted cost.  The next three
    fields count unique source descriptors split across the compact planner
    and external-exact namespaces.  ``constructed_active_bytes`` is the
    logical K/V payload actually gathered by :class:`ExactTier`, and
    ``physical_padding_bytes`` is the additional storage required by the
    dense Hugging Face cache layout.
    """

    knapsack_bytes: int
    unique_selected_source_bytes: int
    planner_tier_bytes: int
    external_exact_bytes: int
    constructed_active_bytes: int
    physical_padding_bytes: int
    selected_source_block_count: int
    planner_source_block_count: int
    external_source_block_count: int

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(int(value) < 0 for value in values.values()):
            raise ValueError("exact-cache storage accounting cannot be negative")
        if self.selected_source_block_count < 1:
            raise ValueError("exact-cache storage accounting requires selected source blocks")
        if self.unique_selected_source_bytes != (
            self.planner_tier_bytes + self.external_exact_bytes
        ):
            raise ValueError("planner and external bytes do not partition unique source bytes")
        if self.selected_source_block_count != (
            self.planner_source_block_count + self.external_source_block_count
        ):
            raise ValueError("planner and external blocks do not partition selected blocks")

    @property
    def constructed_physical_bytes(self) -> int:
        return self.constructed_active_bytes + self.physical_padding_bytes

    @property
    def logical_boundaries_aligned(self) -> bool:
        return (
            self.knapsack_bytes
            == self.unique_selected_source_bytes
            == self.constructed_active_bytes
        )


@dataclass(frozen=True, slots=True)
class _HierarchicalExactArtifact:
    construction: ThreeTierCacheConstruction
    selection: SelectionResult
    external_source_node_ids: tuple[int, ...]
    retained_source_node_ids: tuple[int, ...]
    packed_physical_bytes: int
    storage_accounting: ExactCacheStorageAccounting | None = None


def _dense_physical_cache_budget(
    full_state: FullKVState,
    config: RunConfig,
) -> DensePhysicalCacheBudget | None:
    """Quantize a byte budget to one shared dense prompt capacity.

    The eager HF cache has one physical sequence dimension shared by all
    layers and KV heads.  A logical byte sum is therefore not a physical
    budget.  A shared per-identity position capacity makes the materialized
    tensor size predictable and auditable before selection, including for
    fixed blocks larger than one token.
    """

    if config.cache.budget_unit is not BudgetUnit.BYTES:
        return None
    if full_state.block_size != config.cache.block_size:
        raise HuggingFaceRuntimeError("runtime and FullKV block sizes disagree")
    accounting_state = full_state
    accounting_block_size = config.cache.physical_budget_block_size
    accounting_policy = config.cache.physical_budget_mandatory_policy
    if accounting_block_size is not None or accounting_policy != "method":
        reference_block_size = (
            full_state.block_size if accounting_block_size is None else accounting_block_size
        )
        reference_mandatory = (
            tuple(full_state.mandatory_logical_positions)
            if accounting_policy in {"method", "question_and_boundaries"}
            else tuple(sorted({0, full_state.original_logical_sequence_length - 1}))
            if accounting_policy == "first_and_last"
            else (full_state.original_logical_sequence_length - 1,)
        )
        accounting_state = FullKVState.from_tensors(
            tuple((layer.key, layer.value) for layer in full_state.layers),
            modality_spans=full_state.modality_spans,
            token_ids=full_state.token_ids,
            block_size=reference_block_size,
            sequence_dimension=full_state.layers[0].key_sequence_dimension,
            head_dimension=full_state.layers[0].key_head_dimension,
            logical_positions=full_state.logical_positions.physical_to_logical,
            original_logical_sequence_length=full_state.original_logical_sequence_length,
            next_decode_position=full_state.next_decode_position,
            mandatory_logical_positions=reference_mandatory,
            source_class=full_state.source_class,
            source_kind=full_state.source_kind,
            cached_key_state=full_state.cached_key_state,
        )
    lengths = {layer.sequence_length for layer in full_state.layers}
    if len(lengths) != 1:
        raise HuggingFaceRuntimeError(
            "dense HF physical-byte budgeting requires a uniform source sequence length"
        )
    source_length = next(iter(lengths))
    bytes_per_identity: dict[tuple[int, int], int] = {}
    for block in accounting_state.blocks:
        if block.byte_size % block.position_count:
            raise HuggingFaceRuntimeError("KV block bytes are not divisible by its positions")
        identity = (block.layer, block.kv_head)
        bytes_per_token = block.byte_size // block.position_count
        previous = bytes_per_identity.setdefault(identity, bytes_per_token)
        if previous != bytes_per_token:
            raise HuggingFaceRuntimeError("KV bytes per token vary within one layer/head identity")
    bytes_per_position = sum(bytes_per_identity.values())
    source_bytes = bytes_per_position * source_length
    if source_bytes != full_state.active_bytes:
        raise HuggingFaceRuntimeError(
            "dense HF byte accounting does not reconstruct the FullKV tensor storage"
        )
    requested = min(
        math.ceil(source_bytes * config.cache.retention_ratio),
        config.cache.budget_value,
    )
    prompt_capacity = requested // bytes_per_position
    if config.cache.retention_ratio == 1.0:
        prompt_capacity = source_length
    else:
        # A dense cache is fastest when every layer/head has exactly the same
        # number of real positions. Quantize the target to a block-subset sum
        # so the selector can fill every physical slot without padding masks.
        grouped_signatures: dict[tuple[int, int], list[tuple[int, bool]]] = {
            identity: [] for identity in bytes_per_identity
        }
        for block in accounting_state.blocks:
            grouped_signatures[(block.layer, block.kv_head)].append(
                (block.position_count, block.mandatory)
            )
        signatures = {
            identity: tuple(sorted(values)) for identity, values in grouped_signatures.items()
        }
        unique_signatures = set(signatures.values())
        if len(unique_signatures) != 1:
            raise HuggingFaceRuntimeError(
                "dense mask-free packing requires identical block structures per layer/head"
            )
        signature = next(iter(unique_signatures))
        mandatory_positions = sum(size for size, mandatory in signature if mandatory)
        if mandatory_positions > prompt_capacity:
            raise HuggingFaceRuntimeError(
                "physical byte budget cannot represent the mandatory block subset"
            )
        # Selection operates on fixed-size blocks, but modality boundaries can
        # leave smaller fragments. Treating a fragment-dependent sum as a
        # generally fillable dense capacity is unsafe: greedy selection can
        # consume a different fragment combination and become stranded just
        # below that capacity. Quantize the shared physical extent itself to
        # the fixed block lattice. Mandatory-only capacity is the exception
        # when the requested budget is smaller than one aligned extent.
        optional_capacity = max(0, prompt_capacity - mandatory_positions)
        aligned_optional_capacity = (
            optional_capacity // accounting_state.block_size
        ) * accounting_state.block_size
        prompt_capacity = mandatory_positions + aligned_optional_capacity
    if prompt_capacity < 1:
        raise HuggingFaceRuntimeError(
            "physical byte budget cannot represent one KV position per layer/head"
        )
    prompt_capacity = min(prompt_capacity, source_length)
    target_bytes = prompt_capacity * bytes_per_position
    limits = {identity: prompt_capacity for identity in bytes_per_identity}
    mandatory_counts: dict[tuple[int, int], int] = {identity: 0 for identity in bytes_per_identity}
    for block in full_state.blocks:
        if block.mandatory:
            mandatory_counts[(block.layer, block.kv_head)] += block.position_count
    violations = {
        identity: count for identity, count in mandatory_counts.items() if count > prompt_capacity
    }
    if violations:
        identity, count = min(violations.items())
        raise HuggingFaceRuntimeError(
            "physical byte budget is below mandatory per-head capacity: "
            f"layer={identity[0]}, kv_head={identity[1]}, "
            f"required={count}, capacity={prompt_capacity}"
        )
    return DensePhysicalCacheBudget(source_bytes, target_bytes, prompt_capacity, limits)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional HF environment
        raise HuggingFaceRuntimeError("the unified HF runtime requires PyTorch") from error
    return torch


def _timed(exemplar: Any, action: Any) -> tuple[Any, float]:
    """Measure a phase with synchronized CUDA events or a host monotonic clock."""

    torch = _torch()
    device = getattr(exemplar, "device", None)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = action()
        end.record()
        torch.cuda.synchronize(device)
        return result, float(start.elapsed_time(end)) / 1000.0
    started = time.perf_counter()
    result = action()
    return result, time.perf_counter() - started


def _cache_payload_bytes(
    adapter: HuggingFaceMultimodalAdapter,
    cache: Any,
) -> int:
    """Count K/V payload bytes without treating views as new allocations."""

    if isinstance(cache, CacheSnapshot):
        layers = tuple((layer.key, layer.value) for layer in cache.layers)
    else:
        layers, _source_kind = adapter._legacy_layers(cache)
    return sum(tensor_storage_bytes(key) + tensor_storage_bytes(value) for key, value in layers)


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        raise HuggingFaceRuntimeError("runtime tensor contains NaN or infinity")
    return result


def _forecast_attention_by_node(
    full_state: FullKVState,
    prefill: PrefillOutput,
    rollout: IsolatedQueryRollout,
    config: RunConfig,
) -> dict[int, float]:
    """Pool actual eager prompt/draft attention into RoPE-aware block evidence."""

    by_head = _forecast_token_attention_by_head(full_state, prefill, rollout, config)
    return {
        node_id: float(
            by_head[(block.layer, block.kv_head)][list(block.physical_cache_indices)].sum()
        )
        for node_id, block in enumerate(full_state.blocks)
    }


def _route_head_attention(
    prompt_sample: np.ndarray[Any, Any] | None,
    draft_samples: Sequence[np.ndarray[Any, Any]],
    config: RunConfig,
) -> tuple[np.ndarray[Any, Any], float | None, float | None]:
    """Route one KV head between prompt safety and shifted draft evidence."""

    samples = ([] if prompt_sample is None else [prompt_sample]) + list(draft_samples)
    if not samples:
        raise HuggingFaceRuntimeError("forecast attention has no prompt or draft samples")
    if not (
        config.forecasting.adaptive_head_routing and prompt_sample is not None and draft_samples
    ):
        return np.mean(np.stack(samples, axis=0), axis=0), None, None
    draft_sample = np.mean(np.stack(draft_samples, axis=0), axis=0)
    epsilon = float(np.finfo(np.float64).tiny)
    prompt_distribution = prompt_sample / max(float(prompt_sample.sum()), epsilon)
    draft_distribution = draft_sample / max(float(draft_sample.sum()), epsilon)
    shift = 0.5 * float(np.abs(prompt_distribution - draft_distribution).sum())
    prompt_weight = (
        config.forecasting.stable_head_prompt_weight
        if shift <= config.forecasting.head_shift_threshold
        else config.forecasting.shifted_head_prompt_weight
    )
    values = prompt_weight * prompt_sample + (1.0 - prompt_weight) * draft_sample
    return values, shift, prompt_weight


def _prompt_token_attention_by_head(
    full_state: FullKVState,
    prefill: PrefillOutput,
    *,
    prompt_window: int,
) -> dict[tuple[int, int], np.ndarray[Any, Any]]:
    """Pool eager prompt attention on-device and synchronize it once.

    The former path copied every complete ``[H,Q,K]`` attention tensor to the
    host before reducing it.  MosaicKV consumes only one ``[KVH,K]`` prompt
    window summary per layer, so the large transfer and its per-layer
    synchronizations were pure orchestration overhead.  This reduction is the
    same grouped-query-head/window mean and retains all token-level evidence.
    """

    if prompt_window < 1:
        raise HuggingFaceRuntimeError("prompt attention pooling requires prompt_window >= 1")
    if not prefill.attention_weights:
        raise HuggingFaceRuntimeError("runtime prefill did not capture eager attention weights")
    attention_layers = tuple(prefill.attention_weights)
    if len(attention_layers) != len(full_state.layers):
        raise HuggingFaceRuntimeError("prefill attention layer count does not match the cache")
    if all(value.__class__.__module__.startswith("torch") for value in attention_layers):
        import torch

        pooled_layers: list[Any] = []
        minima: list[Any] = []
        shapes: list[tuple[int, int]] = []
        for attention, layer in zip(attention_layers, full_state.layers, strict=True):
            tensor = attention.detach().float()
            if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
                raise HuggingFaceRuntimeError("eager prompt attention must have shape [1,H,Q,K]")
            query_heads = int(tensor.shape[1])
            if query_heads % layer.kv_heads:
                raise HuggingFaceRuntimeError("query heads are not divisible by cache KV heads")
            sequence_length = layer.sequence_length
            if int(tensor.shape[-1]) < sequence_length:
                raise HuggingFaceRuntimeError(
                    "prompt attention key axis is shorter than the active cache"
                )
            window = min(prompt_window, int(tensor.shape[-2]))
            grouped = tensor[
                ...,
                -window:,
                :sequence_length,
            ].reshape(
                1,
                layer.kv_heads,
                query_heads // layer.kv_heads,
                window,
                sequence_length,
            )
            pooled = grouped.mean(dim=(0, 2, 3))
            pooled_layers.append(pooled)
            minima.append(tensor.amin())
            shapes.append((layer.kv_heads, sequence_length))
        flattened_counts = tuple(heads * length for heads, length in shapes)
        combined = torch.cat(
            (
                *(value.reshape(-1) for value in pooled_layers),
                torch.stack(minima),
            )
        )
        host = combined.detach().cpu().numpy().astype(np.float32, copy=False)
        pooled_count = sum(flattened_counts)
        if not bool(np.all(np.isfinite(host[:pooled_count]))):
            raise HuggingFaceRuntimeError("prompt token attention must be finite")
        if bool(np.any(host[pooled_count:] < 0)):
            raise HuggingFaceRuntimeError("prompt attention weights must be nonnegative")
        result: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
        offset = 0
        tiny = float(np.finfo(np.float64).tiny)
        for layer_index, ((heads, length), count) in enumerate(
            zip(shapes, flattened_counts, strict=True)
        ):
            matrix = host[offset : offset + count].reshape(heads, length)
            for kv_head in range(heads):
                result[(layer_index, kv_head)] = np.maximum(matrix[kv_head], tiny)
            offset += count
        return result
    prompt_layers = tuple(_to_numpy(value) for value in attention_layers)
    result = {}
    for layer_index, layer in enumerate(full_state.layers):
        prompt = prompt_layers[layer_index]
        if prompt.ndim != 4 or prompt.shape[0] != 1:
            raise HuggingFaceRuntimeError("eager prompt attention must have shape [1,H,Q,K]")
        query_heads = int(prompt.shape[1])
        if query_heads % layer.kv_heads:
            raise HuggingFaceRuntimeError("query heads are not divisible by cache KV heads")
        group = query_heads // layer.kv_heads
        window = min(prompt_window, int(prompt.shape[-2]))
        for kv_head in range(layer.kv_heads):
            start = kv_head * group
            values = prompt[
                0,
                start : start + group,
                -window:,
                : layer.sequence_length,
            ].mean(axis=(0, 1))
            result[(layer_index, kv_head)] = np.maximum(
                values,
                float(np.finfo(np.float64).tiny),
            )
    return result


def _prompt_query_token_attention_by_head(
    full_state: FullKVState,
    prefill: PrefillOutput,
    *,
    prompt_window: int,
) -> dict[tuple[int, int], np.ndarray[Any, Any]]:
    """Retain one attention distribution for each prompt query position.

    Grouped query heads are averaged into their KV head, but prompt query
    positions are deliberately not averaged.  This exposes a block that is
    critical to one OCR/question token even when its mean prompt-window mass
    is modest.  Torch inputs are reduced on-device and copied to the host in
    one combined transfer.
    """

    if prompt_window < 1:
        raise HuggingFaceRuntimeError("prompt-query attention pooling requires prompt_window >= 1")
    if not prefill.attention_weights:
        raise HuggingFaceRuntimeError("runtime prefill did not capture eager attention weights")
    attention_layers = tuple(prefill.attention_weights)
    if len(attention_layers) != len(full_state.layers):
        raise HuggingFaceRuntimeError("prefill attention layer count does not match the cache")
    if all(value.__class__.__module__.startswith("torch") for value in attention_layers):
        import torch

        pooled_layers: list[Any] = []
        minima: list[Any] = []
        shapes: list[tuple[int, int, int]] = []
        for attention, layer in zip(attention_layers, full_state.layers, strict=True):
            tensor = attention.detach().float()
            if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
                raise HuggingFaceRuntimeError("eager prompt attention must have shape [1,H,Q,K]")
            query_heads = int(tensor.shape[1])
            if query_heads % layer.kv_heads:
                raise HuggingFaceRuntimeError("query heads are not divisible by cache KV heads")
            sequence_length = layer.sequence_length
            if int(tensor.shape[-1]) < sequence_length:
                raise HuggingFaceRuntimeError(
                    "prompt attention key axis is shorter than the active cache"
                )
            window = min(prompt_window, int(tensor.shape[-2]))
            grouped = tensor[..., -window:, :sequence_length].reshape(
                1,
                layer.kv_heads,
                query_heads // layer.kv_heads,
                window,
                sequence_length,
            )
            pooled_layers.append(grouped.mean(dim=(0, 2)))
            minima.append(tensor.amin())
            shapes.append((layer.kv_heads, window, sequence_length))
        flattened_counts = tuple(heads * window * length for heads, window, length in shapes)
        combined = torch.cat(
            (
                *(value.reshape(-1) for value in pooled_layers),
                torch.stack(minima),
            )
        )
        host = combined.detach().cpu().numpy().astype(np.float32, copy=False)
        pooled_count = sum(flattened_counts)
        if not bool(np.all(np.isfinite(host[:pooled_count]))):
            raise HuggingFaceRuntimeError("prompt-query token attention must be finite")
        if bool(np.any(host[pooled_count:] < 0)):
            raise HuggingFaceRuntimeError("prompt attention weights must be nonnegative")
        result: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
        offset = 0
        tiny = float(np.finfo(np.float64).tiny)
        for layer_index, ((heads, window, length), count) in enumerate(
            zip(shapes, flattened_counts, strict=True)
        ):
            tensor = host[offset : offset + count].reshape(heads, window, length)
            for kv_head in range(heads):
                result[(layer_index, kv_head)] = np.maximum(tensor[kv_head], tiny)
            offset += count
        return result

    prompt_layers = tuple(_to_numpy(value) for value in attention_layers)
    result = {}
    for layer_index, layer in enumerate(full_state.layers):
        prompt = prompt_layers[layer_index]
        if prompt.ndim != 4 or prompt.shape[0] != 1:
            raise HuggingFaceRuntimeError("eager prompt attention must have shape [1,H,Q,K]")
        query_heads = int(prompt.shape[1])
        if query_heads % layer.kv_heads:
            raise HuggingFaceRuntimeError("query heads are not divisible by cache KV heads")
        group = query_heads // layer.kv_heads
        window = min(prompt_window, int(prompt.shape[-2]))
        for kv_head in range(layer.kv_heads):
            start = kv_head * group
            values = prompt[
                0,
                start : start + group,
                -window:,
                : layer.sequence_length,
            ].mean(axis=0)
            result[(layer_index, kv_head)] = np.maximum(
                values,
                float(np.finfo(np.float64).tiny),
            )
    return result


def _forecast_token_attention_and_shift_by_head(
    full_state: FullKVState,
    prefill: PrefillOutput,
    rollout: IsolatedQueryRollout,
    config: RunConfig,
) -> tuple[
    dict[tuple[int, int], np.ndarray[Any, Any]],
    dict[tuple[int, int], float],
]:
    """Return token evidence and label-blind prompt/draft shift per KV head."""

    if not prefill.attention_weights:
        raise HuggingFaceRuntimeError("runtime prefill did not capture eager attention weights")
    if not rollout.attention_steps:
        return (
            _prompt_token_attention_by_head(
                full_state,
                prefill,
                prompt_window=config.forecasting.prompt_window,
            ),
            {},
        )
    prompt_layers = tuple(_to_numpy(value) for value in prefill.attention_weights)
    if len(prompt_layers) != len(full_state.layers):
        raise HuggingFaceRuntimeError("prefill attention layer count does not match the cache")
    draft_steps = tuple(
        tuple(_to_numpy(value) for value in step) for step in rollout.attention_steps
    )
    result: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    shifts: dict[tuple[int, int], float] = {}
    for layer_index, layer in enumerate(full_state.layers):
        prompt = prompt_layers[layer_index]
        if prompt.ndim != 4 or prompt.shape[0] != 1:
            raise HuggingFaceRuntimeError("eager prompt attention must have shape [1,H,Q,K]")
        query_heads = int(prompt.shape[1])
        if query_heads % layer.kv_heads:
            raise HuggingFaceRuntimeError("query heads are not divisible by cache KV heads")
        group = query_heads // layer.kv_heads
        sequence_length = layer.sequence_length
        for kv_head in range(layer.kv_heads):
            query_start = kv_head * group
            query_end = query_start + group
            prompt_sample: np.ndarray[Any, Any] | None = None
            draft_samples: list[np.ndarray[Any, Any]] = []
            if config.forecasting.prompt_window:
                window = min(config.forecasting.prompt_window, int(prompt.shape[-2]))
                prompt_sample = prompt[
                    0,
                    query_start:query_end,
                    -window:,
                    :sequence_length,
                ].mean(axis=(0, 1))
            for step in draft_steps:
                attention = step[layer_index]
                if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[-2] != 1:
                    raise HuggingFaceRuntimeError("eager draft attention must have shape [1,H,1,K]")
                draft_samples.append(
                    attention[
                        0,
                        query_start:query_end,
                        0,
                        :sequence_length,
                    ].mean(axis=0)
                )
            values, shift, _prompt_weight = _route_head_attention(
                prompt_sample,
                draft_samples,
                config,
            )
            if values.shape != (sequence_length,):
                raise HuggingFaceRuntimeError("forecast token attention length is invalid")
            result[(layer_index, kv_head)] = np.maximum(
                values,
                float(np.finfo(np.float64).tiny),
            )
            if shift is not None:
                shifts[(layer_index, kv_head)] = shift
    return result, shifts


def _forecast_token_attention_by_head(
    full_state: FullKVState,
    prefill: PrefillOutput,
    rollout: IsolatedQueryRollout,
    config: RunConfig,
) -> dict[tuple[int, int], np.ndarray[Any, Any]]:
    """Return the exact token-level evidence pooled by the online selector."""

    return _forecast_token_attention_and_shift_by_head(
        full_state,
        prefill,
        rollout,
        config,
    )[0]


def _uncertainty_exact_head_routes(
    shifts: Mapping[tuple[int, int], float],
    fraction: float,
) -> tuple[str, ...]:
    """Select the highest prompt/draft-shift heads with deterministic ties."""

    if not 0 <= fraction <= 1:
        raise ValueError("uncertainty exact-head fraction must lie in [0, 1]")
    if not shifts or fraction == 0:
        return ()
    count = min(len(shifts), max(1, math.ceil(len(shifts) * fraction)))
    ranked = sorted(
        shifts,
        key=lambda identity: (-float(shifts[identity]), identity[0], identity[1]),
    )
    return tuple(f"{layer}:{head}" for layer, head in sorted(ranked[:count]))


def _oracle_token_attention_by_head(
    full_state: FullKVState,
    oracle: EvaluationOnlyOracleQueries,
) -> dict[tuple[int, int], np.ndarray[Any, Any]]:
    if not oracle.attention_steps:
        raise HuggingFaceRuntimeError("oracle diagnostics require captured FullKV attention")
    steps = tuple(tuple(_to_numpy(value) for value in step) for step in oracle.attention_steps)
    result: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    for layer_index, layer in enumerate(full_state.layers):
        exemplar = steps[0][layer_index]
        if exemplar.ndim != 4 or exemplar.shape[0] != 1 or exemplar.shape[-2] != 1:
            raise HuggingFaceRuntimeError("oracle decode attention must have shape [1,H,1,K]")
        query_heads = int(exemplar.shape[1])
        if query_heads % layer.kv_heads:
            raise HuggingFaceRuntimeError("oracle query heads are not divisible by KV heads")
        group = query_heads // layer.kv_heads
        for kv_head in range(layer.kv_heads):
            start = kv_head * group
            per_step = []
            for step in steps:
                attention = step[layer_index]
                per_step.append(
                    attention[
                        0,
                        start : start + group,
                        0,
                        : layer.sequence_length,
                    ].mean(axis=0)
                )
            result[(layer_index, kv_head)] = np.mean(np.stack(per_step, axis=0), axis=0)
    return result


def _forecast_quality_diagnostic(
    adapter: HuggingFaceMultimodalAdapter,
    prefill: PrefillOutput,
    full_state: FullKVState,
    forecast: QueryForecast,
    rollout: IsolatedQueryRollout,
    selection: SelectionResult,
    config: RunConfig,
) -> ForecastQualityDiagnostics:
    """Evaluation-only oracle audit; its outputs never enter online selection."""

    steps = config.diagnostics.forecast_oracle_steps
    if steps < 1:
        raise ValueError("forecast oracle diagnostics require a positive step count")
    oracle = collect_evaluation_only_true_future_queries(
        adapter,
        prefill,
        future_steps=steps,
        capture_attentions=True,
    )
    predicted = _forecast_token_attention_by_head(full_state, prefill, rollout, config)
    return _forecast_quality_from_oracle(
        full_state,
        forecast,
        oracle,
        predicted,
        selection,
    )


def _attention_diagnostic_inputs(
    full_state: FullKVState,
    predicted: Mapping[tuple[int, int], np.ndarray[Any, Any]],
    oracle: EvaluationOnlyOracleQueries,
    selection: SelectionResult | Iterable[int],
) -> dict[tuple[int, int], AttentionDiagnosticInput]:
    """Align actual selected blocks and oracle attention within each cache identity."""

    truth = _oracle_token_attention_by_head(full_state, oracle)
    selected = set(
        selection.selected_node_ids
        if isinstance(selection, SelectionResult)
        else (int(node_id) for node_id in selection)
    )
    inputs: dict[tuple[int, int], AttentionDiagnosticInput] = {}
    for identity in sorted(predicted):
        node_ids = tuple(
            node_id
            for node_id, block in enumerate(full_state.blocks)
            if (block.layer, block.kv_head) == identity
        )
        local_selected = tuple(
            local_index for local_index, node_id in enumerate(node_ids) if node_id in selected
        )
        inputs[identity] = AttentionDiagnosticInput(
            predicted_attention=predicted[identity],
            true_attention=truth[identity],
            block_positions=tuple(
                full_state.blocks[node_id].physical_cache_indices for node_id in node_ids
            ),
            selected_block_count=len(local_selected),
            selected_block_indices=local_selected,
        )
    return inputs


def _forecast_quality_from_oracle(
    full_state: FullKVState,
    forecast: QueryForecast,
    oracle: EvaluationOnlyOracleQueries,
    predicted: Mapping[tuple[int, int], np.ndarray[Any, Any]],
    selection: SelectionResult | Iterable[int],
) -> ForecastQualityDiagnostics:
    return evaluate_forecast_quality(
        forecast,
        oracle,
        _attention_diagnostic_inputs(full_state, predicted, oracle, selection),
    )


def _selection_quality_from_oracle(
    full_state: FullKVState,
    oracle: EvaluationOnlyOracleQueries,
    predicted: Mapping[tuple[int, int], np.ndarray[Any, Any]],
    selection: SelectionResult | Iterable[int],
) -> AttentionSelectionDiagnostics:
    return evaluate_attention_selection_quality(
        _attention_diagnostic_inputs(full_state, predicted, oracle, selection)
    )


def _uniform_token_attention_by_head(
    full_state: FullKVState,
) -> dict[tuple[int, int], np.ndarray[Any, Any]]:
    """No-forecast control: equal evidence for every source position."""

    return {
        (layer_index, kv_head): np.ones(layer.sequence_length, dtype=np.float64)
        for layer_index, layer in enumerate(full_state.layers)
        for kv_head in range(layer.kv_heads)
    }


def _block_attention_values_from_token_attention(
    full_state: FullKVState,
    by_head: Mapping[tuple[int, int], np.ndarray[Any, Any]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    # Cache blockization is normally contiguous. Pool every identity with one
    # prefix-sum pass instead of dispatching tens of thousands of tiny NumPy
    # reductions from Python. Retain an exact indexed fallback for deliberately
    # noncontiguous synthetic descriptors.
    grouped: dict[tuple[int, int], list[tuple[int, KVBlockDescriptor]]] = defaultdict(list)
    for node_id, block in enumerate(full_state.blocks):
        grouped[(block.layer, block.kv_head)].append((node_id, block))
    result = np.empty(len(full_state.blocks), dtype=np.float64)
    for identity, entries in grouped.items():
        source = np.asarray(by_head[identity], dtype=np.float64)
        prefix = np.empty(source.size + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(source, out=prefix[1:])
        contiguous: list[tuple[int, int, int]] = []
        irregular: list[tuple[int, KVBlockDescriptor]] = []
        for node_id, block in entries:
            positions = block.physical_cache_indices
            if positions[-1] - positions[0] + 1 == len(positions):
                contiguous.append((node_id, positions[0], positions[-1] + 1))
            else:
                irregular.append((node_id, block))
        if contiguous:
            starts = np.fromiter(
                (start for _node_id, start, _end in contiguous),
                dtype=np.intp,
                count=len(contiguous),
            )
            ends = np.fromiter(
                (end for _node_id, _start, end in contiguous),
                dtype=np.intp,
                count=len(contiguous),
            )
            masses = prefix[ends] - prefix[starts]
            node_ids = np.fromiter(
                (node_id for node_id, _start, _end in contiguous),
                dtype=np.intp,
                count=len(contiguous),
            )
            result[node_ids] = masses
        for node_id, block in irregular:
            result[node_id] = float(
                source[np.asarray(block.physical_cache_indices, dtype=np.intp)].sum()
            )
    return result


def _block_attention_from_token_attention(
    full_state: FullKVState,
    by_head: Mapping[tuple[int, int], np.ndarray[Any, Any]],
) -> dict[int, float]:
    """Compatibility wrapper for callers that require keyed block scores."""

    values = _block_attention_values_from_token_attention(full_state, by_head)
    return {node_id: float(value) for node_id, value in enumerate(values)}


def _block_query_attention_from_token_attention(
    full_state: FullKVState,
    by_head: Mapping[tuple[int, int], np.ndarray[Any, Any]],
) -> dict[int, np.ndarray[Any, Any]]:
    """Pool ``[prompt query, source token]`` attention into block vectors."""

    grouped: dict[tuple[int, int], list[tuple[int, KVBlockDescriptor]]] = defaultdict(list)
    for node_id, block in enumerate(full_state.blocks):
        grouped[(block.layer, block.kv_head)].append((node_id, block))
    result: dict[int, np.ndarray[Any, Any]] = {}
    for identity, entries in grouped.items():
        source = np.asarray(by_head[identity], dtype=np.float64)
        if source.ndim != 2 or not source.shape[0]:
            raise HuggingFaceRuntimeError(
                "prompt-query token attention must have shape [query, source]"
            )
        prefix = np.empty((source.shape[0], source.shape[1] + 1), dtype=np.float64)
        prefix[:, 0] = 0.0
        np.cumsum(source, axis=1, out=prefix[:, 1:])
        for node_id, block in entries:
            positions = block.physical_cache_indices
            if positions[-1] - positions[0] + 1 == len(positions):
                values = prefix[:, positions[-1] + 1] - prefix[:, positions[0]]
            else:
                values = source[:, np.asarray(positions, dtype=np.intp)].sum(axis=1)
            if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0)):
                raise HuggingFaceRuntimeError(
                    "prompt-query block attention must be finite and nonnegative"
                )
            result[node_id] = values
    return result


def _prompt_attention_by_node(
    full_state: FullKVState,
    prefill: PrefillOutput,
    *,
    prompt_window: int,
) -> dict[int, float]:
    """Pool prompt-window attention mass without using draft or forecast queries."""

    if prompt_window < 1:
        raise HuggingFaceRuntimeError("prompt_attention_topk requires prompt_window >= 1")
    result = _block_attention_from_token_attention(
        full_state,
        _prompt_token_attention_by_head(
            full_state,
            prefill,
            prompt_window=prompt_window,
        ),
    )
    if any(not math.isfinite(score) or score < 0 for score in result.values()):
        raise HuggingFaceRuntimeError("prompt attention block mass must be finite and nonnegative")
    return result


def _budget_source_cost(full_state: FullKVState, unit: BudgetUnit) -> int:
    if unit is BudgetUnit.BLOCKS:
        return len(full_state.blocks)
    if unit is BudgetUnit.RETAINED_SLOTS:
        return sum(block.position_count for block in full_state.blocks)
    if unit is BudgetUnit.BYTES:
        return full_state.active_bytes
    raise ValueError(f"unsupported runtime budget unit: {unit}")


def _mandatory_cost(full_state: FullKVState, unit: BudgetUnit) -> int:
    budget = SelectionBudget(1, unit)
    return sum(budget.cost(block) for block in full_state.blocks if block.mandatory)


def _mosaickv_effective_capacity(
    full_state: FullKVState,
    config: RunConfig,
    shared_target: int,
    position_limits: Mapping[tuple[int, int], int] | None,
) -> tuple[int, Mapping[tuple[int, int], int] | None]:
    """Resolve MosaicKV's auditable utilization of the shared hard cache cap."""

    fraction = config.selection.physical_capacity_fraction
    if config.cache.retention_ratio == 1.0 or fraction == 1.0:
        return shared_target, position_limits
    mandatory_cost = _mandatory_cost(full_state, config.cache.budget_unit)
    if position_limits is None:
        return max(mandatory_cost, math.floor(shared_target * fraction)), None

    mandatory_positions: Counter[tuple[int, int]] = Counter()
    for block in full_state.blocks:
        if block.mandatory:
            mandatory_positions[(block.layer, block.kv_head)] += block.position_count
    effective_limits = {
        identity: max(
            1,
            int(mandatory_positions[identity]),
            math.floor(int(limit) * fraction),
        )
        for identity, limit in position_limits.items()
    }
    if config.cache.budget_unit is BudgetUnit.BYTES:
        bytes_per_position: dict[tuple[int, int], int] = {}
        for block in full_state.blocks:
            if block.byte_size % block.position_count:
                raise HuggingFaceRuntimeError(
                    "MosaicKV physical-capacity scaling requires position-aligned blocks"
                )
            identity = (block.layer, block.kv_head)
            current = block.byte_size // block.position_count
            previous = bytes_per_position.setdefault(identity, current)
            if previous != current:
                raise HuggingFaceRuntimeError(
                    "MosaicKV KV bytes per position vary within one layer/head"
                )
        effective_target = sum(
            bytes_per_position[identity] * limit for identity, limit in effective_limits.items()
        )
    elif config.cache.budget_unit is BudgetUnit.RETAINED_SLOTS:
        effective_target = sum(effective_limits.values())
    else:
        effective_target = max(mandatory_cost, math.floor(shared_target * fraction))
    if effective_target > shared_target:
        raise HuggingFaceRuntimeError(
            "MosaicKV effective capacity exceeds the shared comparison budget"
        )
    return effective_target, effective_limits


def _graph_mandatory_cost(graph: SparseEvidenceGraph, unit: BudgetUnit) -> int:
    budget = SelectionBudget(1, unit)
    return sum(budget.cost(node.block) for node in graph.nodes if node.block.mandatory)


def _resolve_budget(full_state: FullKVState, config: RunConfig) -> tuple[int, int]:
    source = _budget_source_cost(full_state, config.cache.budget_unit)
    requested = math.ceil(source * config.cache.retention_ratio)
    target = min(requested, config.cache.budget_value)
    mandatory = _mandatory_cost(full_state, config.cache.budget_unit)
    if config.cache.retention_ratio == 1.0 and target != source:
        raise HuggingFaceRuntimeError(
            "retention 1.0 requires cache.budget_value to cover the complete FullKV cache"
        )
    if target < mandatory:
        raise HuggingFaceRuntimeError(
            f"retention budget {target} is below mandatory exact cost {mandatory}"
        )
    return source, target


def _similarity_shortlist(
    full_state: FullKVState,
    config: RunConfig,
    *,
    source_budget: int,
    target_budget: int,
    prompt_attention_by_node: (
        Mapping[int, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None
    ),
    head_uncertainty: Mapping[tuple[int, int], float] | None = None,
    eligible_source_node_ids: Sequence[int] | None = None,
) -> tuple[int, ...] | None:
    """Return prompt-ranked nodes for expensive graph similarity sources."""

    multiplier = config.graph.similarity_shortlist_multiplier
    if multiplier <= 0:
        return None
    if prompt_attention_by_node is None:
        raise HuggingFaceRuntimeError(
            "graph similarity shortlisting requires prompt-only block attention"
        )
    node_count = len(full_state.blocks)
    if len(prompt_attention_by_node) != node_count or (
        isinstance(prompt_attention_by_node, Mapping)
        and any(node_id not in prompt_attention_by_node for node_id in range(node_count))
    ):
        raise HuggingFaceRuntimeError(
            "graph similarity shortlist scores must cover every source block"
        )
    eligible_source_ids = (
        tuple(range(node_count))
        if eligible_source_node_ids is None
        else tuple(int(node_id) for node_id in eligible_source_node_ids)
    )
    if len(eligible_source_ids) != len(set(eligible_source_ids)) or any(
        node_id < 0 or node_id >= node_count for node_id in eligible_source_ids
    ):
        raise HuggingFaceRuntimeError("graph similarity shortlist eligible source IDs are invalid")
    if any(
        not math.isfinite(float(prompt_attention_by_node[node_id]))
        or float(prompt_attention_by_node[node_id]) < 0
        for node_id in eligible_source_ids
    ):
        raise HuggingFaceRuntimeError(
            "graph similarity shortlist scores must be finite and nonnegative"
        )
    estimated_selected_blocks = max(
        1,
        math.ceil(len(full_state.blocks) * target_budget / source_budget),
    )
    requested = max(
        config.graph.similarity_shortlist_min_nodes,
        math.ceil(estimated_selected_blocks * multiplier),
    )
    mandatory = {node_id for node_id in eligible_source_ids if full_state.blocks[node_id].mandatory}
    eligible_heads = {(block.layer, block.kv_head) for block in full_state.blocks}
    if config.graph.uncertain_head_only:
        eligible_heads = set(
            _uncertain_head_routes(
                eligible_heads,
                head_uncertainty,
                fraction=config.graph.uncertain_head_fraction,
                threshold=config.graph.uncertain_head_threshold,
                label="graph",
            )
        )

    # Screen coarse contiguous groups, then admit their original fixed-size
    # blocks to graph reranking. Cache granularity and byte accounting remain
    # identical to every baseline; only the inexpensive screening hierarchy is
    # coarse.
    groups: list[tuple[float, int, tuple[int, ...]]] = []
    nodes_by_stratum: defaultdict[tuple[int, int, object], list[int]] = defaultdict(list)
    for node_id in eligible_source_ids:
        block = full_state.blocks[node_id]
        if (block.layer, block.kv_head) in eligible_heads and node_id not in mandatory:
            nodes_by_stratum[(block.layer, block.kv_head, block.modality)].append(node_id)
    width = config.graph.screening_group_blocks
    for node_ids in nodes_by_stratum.values():
        ordered = sorted(
            node_ids,
            key=lambda node_id: (
                full_state.blocks[node_id].original_logical_positions[0],
                node_id,
            ),
        )
        for start in range(0, len(ordered), width):
            members = tuple(ordered[start : start + width])
            groups.append(
                (
                    sum(float(prompt_attention_by_node[node_id]) for node_id in members),
                    members[0],
                    members,
                )
            )
    groups.sort(key=lambda item: (-item[0], item[1]))
    selected = set(mandatory)
    nonmandatory_count = len(full_state.blocks) - sum(
        block.mandatory for block in full_state.blocks
    )
    fraction_limit = math.ceil(nonmandatory_count * config.graph.candidate_nonmandatory_fraction)
    candidate_limit = min(max(0, requested - len(mandatory)), fraction_limit)
    admitted = 0
    for _score, _tie, members in groups:
        if admitted >= candidate_limit:
            break
        remaining = candidate_limit - admitted
        accepted = members[:remaining]
        selected.update(accepted)
        admitted += len(accepted)
    return tuple(sorted(selected))


def _uncertain_head_routes(
    eligible_heads: Iterable[tuple[int, int]],
    head_uncertainty: Mapping[tuple[int, int], float] | None,
    *,
    fraction: float,
    threshold: float,
    label: str,
) -> tuple[tuple[int, int], ...]:
    """Return the deterministic highest-uncertainty layer/head routes."""

    expected = frozenset(eligible_heads)
    if not expected:
        return ()
    if head_uncertainty is None or set(head_uncertainty) != expected:
        raise HuggingFaceRuntimeError(
            f"uncertain-head {label} routing requires one uncertainty value per layer/head"
        )
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in head_uncertainty.values()):
        raise HuggingFaceRuntimeError(f"{label} head uncertainty must lie in [0, 1]")
    ordered = tuple(sorted(expected, key=lambda identity: (-head_uncertainty[identity], identity)))
    count = max(1, math.ceil(len(ordered) * fraction))
    routed = tuple(
        identity for identity in ordered[:count] if head_uncertainty[identity] >= threshold
    )
    return routed or (ordered[0],)


def _attention_uncertainty_by_head(
    token_attention: Mapping[tuple[int, int], np.ndarray[Any, Any]],
) -> dict[tuple[int, int], float]:
    """Normalized entropy router computed from already-pooled prompt evidence."""

    result: dict[tuple[int, int], float] = {}
    tiny = float(np.finfo(np.float64).tiny)
    for identity, raw in token_attention.items():
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0)):
            raise HuggingFaceRuntimeError("head-routing attention must be finite and nonnegative")
        total = float(values.sum())
        if values.size <= 1 or total <= 0:
            result[identity] = 0.0
            continue
        probability = np.maximum(values / total, tiny)
        entropy = -float(np.sum(probability * np.log(probability)))
        result[identity] = min(1.0, max(0.0, entropy / math.log(values.size)))
    return result


def _selection(
    objective: BudgetedObjective,
    value: int,
    unit: BudgetUnit,
    *,
    retention_one: bool,
    position_limits: Mapping[tuple[int, int], int] | None = None,
    fill_position_limits: bool = False,
    position_fill_quantum: int | None = None,
    required_node_ids: Iterable[int] = (),
    prompt_anchor_node_ids: Iterable[int] = (),
    evidence_node_ids: Iterable[int] = (),
    optimization_node_ids: Iterable[int] | None = None,
) -> SelectionResult:
    required_node_ids = tuple(required_node_ids)
    prompt_anchor_node_ids = tuple(prompt_anchor_node_ids)
    evidence_node_ids = tuple(evidence_node_ids)
    if retention_one:
        return select_all_exact(objective, SelectionBudget(value, unit))
    # In the byte-matched exact-only routed path, every planner block is one
    # token and each routed layer/head has an independently fixed dense
    # capacity. A generic global byte/cardinality pass cannot exchange
    # capacity across these routes, so it only adds heap/coverage-refresh
    # overhead. Select the deterministic local top-k set directly, then use
    # the common exact-subset recorder for auditable objective telemetry.
    if (
        objective.config.routed_head_only_planning
        and objective.config.algorithm == "vectorized_coverage"
        and position_limits is not None
        and fill_position_limits
        and position_fill_quantum == 1
        and not required_node_ids
        and not prompt_anchor_node_ids
        and not evidence_node_ids
        and all(node.block.position_count == 1 for node in objective.graph.nodes)
    ):
        return vectorized_unit_route_topk_select(
            objective,
            SelectionBudget(value, unit),
            position_limits=position_limits,
            optimization_node_ids=optimization_node_ids,
        )
    return lazy_greedy_select(
        objective,
        SelectionBudget(value, unit),
        position_limits=position_limits,
        fill_position_limits=fill_position_limits,
        position_fill_quantum=position_fill_quantum,
        required_node_ids=required_node_ids,
        prompt_anchor_node_ids=prompt_anchor_node_ids,
        evidence_node_ids=evidence_node_ids,
        optimization_node_ids=optimization_node_ids,
    )


def _bounded_reservation(
    objective: BudgetedObjective,
    budget_value: int,
    budget_unit: BudgetUnit,
    fraction: float,
    scores: Mapping[int, float],
    *,
    position_limits: Mapping[tuple[int, int], int] | None = None,
    position_fill_quantum: int | None = None,
    reserved_node_ids: Iterable[int] = (),
    per_head: bool = False,
) -> tuple[int, ...]:
    """Reserve ranked optional blocks after all earlier reservations.

    ``reserved_node_ids`` are occupied capacity, not merely exclusions.  Their
    byte/cardinality cost and per-head positions are therefore removed before
    the new fraction is allocated.  This makes mandatory, prototype-anchor,
    prompt-anchor, and evidence reservations one ordered budget transaction.
    """

    if fraction <= 0:
        return ()
    if position_fill_quantum is not None and position_fill_quantum < 1:
        raise ValueError("reservation position_fill_quantum must be positive")
    if per_head and position_limits is None:
        raise ValueError("per-head reservation requires position_limits")
    budget = SelectionBudget(max(1, budget_value), budget_unit)
    reserved = set(int(node_id) for node_id in reserved_node_ids)
    occupied = {
        node.node_id
        for node in objective.graph.nodes
        if node.block.mandatory or node.node_id in reserved
    }
    occupied_cost = sum(budget.cost(objective.graph.nodes[node_id].block) for node_id in occupied)
    allowance = math.floor(max(0, budget_value - occupied_cost) * fraction)
    chosen: list[int] = []
    spent = 0
    selected_positions: defaultdict[tuple[int, int], int] = defaultdict(int)
    for node_id in occupied:
        block = objective.graph.nodes[node_id].block
        identity = (block.layer, block.kv_head)
        selected_positions[identity] += block.position_count
    per_head_allowance = (
        {}
        if not per_head or position_limits is None
        else {
            identity: math.floor(max(0, int(limit) - selected_positions[identity]) * fraction)
            for identity, limit in position_limits.items()
        }
    )
    per_head_spent: defaultdict[tuple[int, int], int] = defaultdict(int)
    for node_id in sorted(
        scores,
        key=lambda candidate: (
            -(float(scores[candidate]) / budget.cost(objective.graph.nodes[candidate].block)),
            -float(scores[candidate]),
            candidate,
        ),
    ):
        block = objective.graph.nodes[node_id].block
        if node_id in occupied:
            continue
        if (
            position_fill_quantum is not None
            and not block.mandatory
            and block.position_count % position_fill_quantum != 0
        ):
            continue
        cost = budget.cost(block)
        identity = (block.layer, block.kv_head)
        limit = None if position_limits is None else position_limits.get(identity)
        if limit is not None and selected_positions[identity] + block.position_count > limit:
            continue
        if per_head and (
            block.position_count
            > int(per_head_allowance.get(identity, 0)) - per_head_spent[identity]
        ):
            continue
        if cost <= allowance - spent:
            chosen.append(node_id)
            spent += cost
            selected_positions[identity] += block.position_count
            per_head_spent[identity] += block.position_count
    return tuple(sorted(chosen))


def _structural_round_robin_evidence_scores(
    objective: BudgetedObjective,
) -> dict[int, float]:
    """Rank fine structural evidence while preventing one region from monopolizing reserve."""

    groups: dict[
        tuple[
            int,
            int,
            str,
            int | None,
            int | None,
            int | None,
            str | None,
            str | None,
            int | None,
            int | None,
        ],
        list[int],
    ] = {}
    for node in objective.graph.nodes:
        evidence = node.evidence
        if not evidence.has_structural_metadata:
            continue
        region_identity = (
            node.block.layer,
            node.block.kv_head,
            node.block.modality.value,
            evidence.image_index,
            evidence.page_index,
            evidence.frame_index,
            evidence.evidence_region,
            evidence.page_region,
            evidence.row_index,
            evidence.column_index,
        )
        groups.setdefault(region_identity, []).append(node.node_id)
    scores: dict[int, float] = {}
    for region_identity in sorted(groups, key=repr):
        members = sorted(
            groups[region_identity],
            key=lambda node_id: (
                -objective.utilities.for_node(node_id).local_utility,
                node_id,
            ),
        )
        # The first block in every fine region receives the largest score;
        # later blocks remain eligible but cannot crowd out unseen regions.
        for local_rank, node_id in enumerate(members):
            utility = objective.utilities.for_node(node_id).local_utility
            scores[node_id] = (1.0 / (local_rank + 1)) + max(0.0, utility) * 1e-6
    return scores


def _spatial_stratified_prompt_evidence_scores(
    objective: BudgetedObjective,
    prompt_attention_by_node: Mapping[int, float],
    *,
    grid_rows: int,
    grid_columns: int,
) -> dict[int, float]:
    """Prioritize prompt-relevant visual blocks across the complete patch grid.

    LLaVA exposes an authoritative vision-patch grid, but it does not expose OCR
    boxes.  Treating the image as a configurable set of spatial strata preserves
    document/chart coverage without inventing OCR metadata.  The first
    prompt-relevant block in every stratum outranks a second block from an
    already represented stratum; prompt mass and local utility break ties
    deterministically inside each coverage tier.
    """

    if grid_rows < 1 or grid_columns < 1:
        raise ValueError("evidence grid dimensions must be positive")
    node_ids = {node.node_id for node in objective.graph.nodes}
    if set(prompt_attention_by_node) != node_ids:
        raise ValueError("spatial evidence prompt scores must cover every graph node")
    if any(
        not math.isfinite(float(score)) or float(score) < 0
        for score in prompt_attention_by_node.values()
    ):
        raise ValueError("spatial evidence prompt scores must be finite and nonnegative")

    groups: defaultdict[tuple[int, int, int, int, int], list[int]] = defaultdict(list)
    for node in objective.graph.nodes:
        box = node.evidence.normalized_box
        if node.block.modality is not Modality.IMAGE or box is None:
            continue
        left, top, right, bottom = box
        column = min(grid_columns - 1, int(((left + right) * 0.5) * grid_columns))
        row = min(grid_rows - 1, int(((top + bottom) * 0.5) * grid_rows))
        groups[
            (
                node.block.layer,
                node.block.kv_head,
                -1 if node.evidence.image_index is None else node.evidence.image_index,
                row,
                column,
            )
        ].append(node.node_id)

    maximum_prompt = max(
        (
            float(prompt_attention_by_node[node_id])
            for members in groups.values()
            for node_id in members
        ),
        default=0.0,
    )
    prompt_scale = maximum_prompt if maximum_prompt > 0 else 1.0
    scores: dict[int, float] = {}
    for identity in sorted(groups):
        ranked = sorted(
            groups[identity],
            key=lambda node_id: (
                -float(prompt_attention_by_node[node_id]),
                -objective.utilities.for_node(node_id).local_utility,
                node_id,
            ),
        )
        for local_rank, node_id in enumerate(ranked):
            prompt_tie_break = float(prompt_attention_by_node[node_id]) / prompt_scale
            utility = objective.utilities.for_node(node_id).local_utility
            bounded_utility = utility / (1.0 + abs(utility))
            scores[node_id] = (
                2.0 / (local_rank + 1) + prompt_tie_break * 1e-3 + bounded_utility * 1e-6
            )
    return scores


def _evidence_preservation_scores(
    objective: BudgetedObjective,
    *,
    mode: str = "structural_round_robin",
    prompt_attention_by_node: Mapping[int, float] | None = None,
    grid_rows: int = 4,
    grid_columns: int = 4,
) -> dict[int, float]:
    """Dispatch the configured label-independent evidence reservation policy."""

    if mode == "structural_round_robin":
        return _structural_round_robin_evidence_scores(objective)
    if mode == "spatial_stratified_prompt":
        if prompt_attention_by_node is None:
            raise ValueError("spatial_stratified_prompt requires prompt attention scores")
        return _spatial_stratified_prompt_evidence_scores(
            objective,
            prompt_attention_by_node,
            grid_rows=grid_rows,
            grid_columns=grid_columns,
        )
    raise ValueError(f"unsupported evidence reservation mode: {mode!r}")


def _required_prototype_anchors(
    objective: BudgetedObjective,
    config: RunConfig,
    *,
    existing_anchor_ids: Iterable[int] = (),
    position_fill_quantum: int | None = None,
    prompt_attention_by_node: Mapping[int, float] | None = None,
) -> tuple[int, ...]:
    """Choose a sparse deterministic anchor cover for prototype feasibility.

    Nodes are partitioned only within a layer, KV head, and modality. Each
    contiguous segment contains at most ``group_size + 1`` nodes and respects
    the configured prototype position span. A mandatory node already supplies
    that segment's anchor; otherwise the highest-local-utility node is reserved
    exact with node ID as the tie-break. Universal sparse positional edges make
    every member directly reachable from the reserved anchor.
    """

    graph = objective.graph
    if position_fill_quantum is not None and position_fill_quantum < 1:
        raise ValueError("prototype-anchor position_fill_quantum must be positive")
    groups: dict[tuple[int, int, object], list[int]] = {}
    for node in graph.nodes:
        block = node.block
        groups.setdefault((block.layer, block.kv_head, block.modality), []).append(node.node_id)

    bounded_cover = config.prototypes.budget_planning_mode == "one_pass_bounded_anchor_cover"
    if bounded_cover and config.prototypes.maximum_prototypes_per_head is None:
        raise ValueError("bounded prototype anchor cover requires maximum_prototypes_per_head")
    if prompt_attention_by_node is not None and set(prompt_attention_by_node) != set(
        range(len(graph.nodes))
    ):
        raise ValueError("prototype anchor prompt scores must cover every graph node")
    required: list[int] = []
    bounded_candidates: defaultdict[
        tuple[int, int],
        list[tuple[float, float, int, tuple[int, ...]]],
    ] = defaultdict(list)
    existing = frozenset(int(node_id) for node_id in existing_anchor_ids)
    for identity in sorted(groups, key=lambda item: (item[0], item[1], str(item[2]))):
        if not config.prototypes.uses_prototype_block(
            identity[0],
            identity[1],
            identity[2],
        ):
            if config.prototypes.requires_exact_head(identity[0], identity[1]):
                required.extend(groups[identity])
            continue
        maximum_members = config.prototypes.group_size_for(identity[0], identity[1])
        maximum_span = config.prototypes.position_span_for(identity[0], identity[1])
        ordered = sorted(
            groups[identity],
            key=lambda node_id: (
                graph.nodes[node_id].block.original_logical_positions[0],
                node_id,
            ),
        )
        segment: list[int] = []

        def reserve_anchor(members: Sequence[int]) -> None:
            if not members:
                return
            preselected = tuple(
                node_id
                for node_id in members
                if graph.nodes[node_id].block.mandatory or node_id in existing
            )
            if preselected and not bounded_cover:
                return
            aligned = tuple(
                node_id
                for node_id in members
                if position_fill_quantum is None
                or graph.nodes[node_id].block.position_count % position_fill_quantum == 0
            )
            # Prefer a full fixed-size block for mask-free dense packing. If a
            # segment contains only modality-boundary fragments, retain the
            # ordinary best anchor and let the selector's residue-completion
            # pass find complementary fragments explicitly.
            candidates = preselected or aligned or tuple(members)
            anchor_id = min(
                candidates,
                key=lambda node_id: (
                    -(
                        0.0
                        if prompt_attention_by_node is None
                        else float(prompt_attention_by_node[node_id])
                    ),
                    -objective.utilities.for_node(node_id).local_utility,
                    node_id,
                ),
            )
            if not bounded_cover:
                required.append(anchor_id)
                return
            prompt_mass = sum(
                0.0
                if prompt_attention_by_node is None
                else float(prompt_attention_by_node[node_id])
                for node_id in members
            )
            utility_mass = sum(
                max(0.0, objective.utilities.for_node(node_id).local_utility) for node_id in members
            )
            head = (
                graph.nodes[anchor_id].block.layer,
                graph.nodes[anchor_id].block.kv_head,
            )
            bounded_candidates[head].append((prompt_mass, utility_mass, anchor_id, tuple(members)))

        for node_id in ordered:
            trial = [*segment, node_id]
            first_position = graph.nodes[trial[0]].block.original_logical_positions[0]
            last_position = graph.nodes[node_id].block.original_logical_positions[-1]
            exceeds_capacity = len(trial) > maximum_members + 1
            exceeds_span = bool(
                maximum_span is not None and last_position - first_position > maximum_span
            )
            if segment and (exceeds_capacity or exceeds_span):
                reserve_anchor(segment)
                segment = [node_id]
            else:
                segment = trial
        reserve_anchor(segment)
    if bounded_cover:
        maximum = config.prototypes.maximum_prototypes_per_head
        if maximum is None:  # pragma: no cover - validated above
            raise RuntimeError("bounded prototype cover lost its configured limit")
        for head in sorted(bounded_candidates):
            ranked = sorted(
                bounded_candidates[head],
                key=lambda item: (-item[0], -item[1], item[2], item[3]),
            )
            required.extend(
                anchor_id for _prompt, _utility, anchor_id, _members in ranked[:maximum]
            )
    return tuple(sorted(required))


def _prototype_aware_reservation_limits(
    graph: SparseEvidenceGraph,
    config: RunConfig,
    position_limits: Mapping[tuple[int, int], int] | None,
    required_anchor_ids: Iterable[int],
) -> Mapping[tuple[int, int], int] | None:
    """Leave physical room for provisional prototype slots during reservations.

    Prompt and evidence reservations run before exact selection. They must not
    consume slots already promised to the prototype anchors, otherwise the
    later one-pass planner sees a fully occupied head and can only fall back to
    exact selection. The reservation is conservative: assignment planning may
    later remove an empty prototype, but it can never discover less capacity
    than was advertised here.
    """

    if position_limits is None:
        return None
    candidate_anchor_ids = {
        *(int(node_id) for node_id in required_anchor_ids),
        *(node.node_id for node in graph.nodes if node.block.mandatory),
    }
    reserved_slots: Counter[tuple[int, int]] = Counter()
    for node_id in candidate_anchor_ids:
        if node_id < 0 or node_id >= len(graph.nodes):
            raise ValueError("prototype reservation anchor references a node outside the graph")
        block = graph.nodes[node_id].block
        if config.prototypes.uses_prototype_block(
            block.layer,
            block.kv_head,
            block.modality,
        ):
            reserved_slots[(block.layer, block.kv_head)] += 1
    limits = {
        identity: int(limit) - int(reserved_slots.get(identity, 0))
        for identity, limit in position_limits.items()
    }
    if any(limit < 1 for limit in limits.values()):
        raise InfeasibleSelectionBudgetError(
            "prototype slots consume the complete dense-cache reservation capacity: "
            f"limits={dict(position_limits)}, prototype_slots={dict(reserved_slots)}"
        )
    return limits


def _refined_budget_candidates(
    upper_failure: int,
    lower_infeasible: int,
    costs: Sequence[int],
    *,
    maximum_points: int = 512,
) -> tuple[int, ...]:
    """Refine only the coarse interval where feasibility changes.

    Selection changes only when a whole block can be added. The GCD of source
    block costs is therefore the finest useful byte/cardinality step. A bounded
    interpolation remains available for unusually heterogeneous layouts.
    """

    if upper_failure <= lower_infeasible + 1 or maximum_points < 1:
        return ()
    quantum = 0
    for cost in costs:
        quantum = math.gcd(quantum, int(cost))
    quantum = max(1, quantum)
    lattice = tuple(range(upper_failure - quantum, lower_infeasible, -quantum))
    if len(lattice) <= maximum_points:
        return lattice
    span = upper_failure - lower_infeasible
    return tuple(
        sorted(
            {
                lower_infeasible + math.floor(span * index / (maximum_points + 1))
                for index in range(1, maximum_points + 1)
            },
            reverse=True,
        )
    )


def _exact_construction(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    capabilities: AdapterCapabilities,
    config: RunConfig,
    key_transform: RotaryPrototypeKeyTransform | None,
    *,
    additional_exact_source_node_ids: Sequence[int] = (),
    active_budget: SelectionBudget | None = None,
) -> ThreeTierCacheConstruction:
    return construct_three_tier_cache(
        full_state,
        graph,
        selection,
        capabilities,
        prototype_config=replace(config.prototypes, enabled=False),
        residual_config=replace(config.residual, enabled=False),
        retention_ratio=config.cache.retention_ratio,
        key_transform=key_transform,
        exact_gather_mode=config.cache.exact_gather_mode,
        total_storage_budget_bytes=config.cache.total_storage_budget_bytes,
        additional_exact_source_node_ids=additional_exact_source_node_ids,
        active_budget=active_budget,
    )


def _uniform_attainable_prototype_position_limits(
    graph: SparseEvidenceGraph,
    position_limits: Mapping[tuple[int, int], int],
    reserved_slots_by_identity: Mapping[tuple[int, int], int],
    *,
    block_size: int,
    required_node_ids: Iterable[int],
    prompt_anchor_node_ids: Iterable[int],
    evidence_node_ids: Iterable[int],
    optimization_node_ids: Iterable[int] | None = None,
    allow_head_padding: bool = False,
) -> tuple[dict[tuple[int, int], int], int]:
    """Choose representable exact limits for a dense prototype cache.

    The dense HF cache has one physical sequence dimension, while modality
    boundaries can produce short source blocks. Reserving a one-token
    prototype slot can therefore leave an exact-cache limit (for example 62)
    that cannot be composed from required fragments and 16-token blocks. This
    planner intersects the position sums attainable by every layer/head and
    safely underfills the physical cap instead of asking the selector to solve
    an impossible equality. When explicitly enabled, a hybrid routed layout
    that has no common attainable length falls back to the largest attainable
    length in each head. The runtime then pads those heads to the returned
    physical length and masks every padding slot during eager attention.

    Optional boundary fragments are considered only for residue completion,
    matching :func:`lazy_greedy_select`; ordinary optional blocks must be
    aligned to ``block_size``. The returned exact limits include neither the
    reserved prototype slots nor padding.
    """

    if block_size < 1:
        raise ValueError("prototype layout block_size must be positive")
    identities = tuple(sorted(position_limits))
    if not identities:
        raise InfeasibleSelectionBudgetError(
            "prototype layout planning requires at least one layer/head limit"
        )
    graph_identities = {(node.block.layer, node.block.kv_head) for node in graph.nodes}
    missing = set(identities) - graph_identities
    if missing:
        raise InfeasibleSelectionBudgetError(
            f"prototype layout limits reference missing layer/head identities: {sorted(missing)}"
        )

    mandatory = {node.node_id for node in graph.nodes if node.block.mandatory}
    required = {int(node_id) for node_id in required_node_ids}
    prompt = {int(node_id) for node_id in prompt_anchor_node_ids}
    evidence = {int(node_id) for node_id in evidence_node_ids}
    hard = mandatory | required | prompt | evidence
    if any(node_id < 0 or node_id >= len(graph.nodes) for node_id in hard):
        raise ValueError("prototype layout required IDs reference a node outside the graph")
    optimization = (
        set(range(len(graph.nodes)))
        if optimization_node_ids is None
        else {int(node_id) for node_id in optimization_node_ids}
    )
    if any(node_id < 0 or node_id >= len(graph.nodes) for node_id in optimization):
        raise ValueError("prototype layout optimization IDs reference a node outside the graph")
    available = hard | optimization

    # Prompt, evidence, and prototype anchors may all be short
    # modality-boundary fragments. They contribute to ``hard_positions``
    # below; the same residue dynamic program used by the selector determines
    # whether optional fragments can complete an attainable dense length.

    nodes_by_identity: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for node in graph.nodes:
        nodes_by_identity[(node.block.layer, node.block.kv_head)].append(node.node_id)

    feasibility: dict[tuple[int, int], tuple[int, int, dict[int, int]]] = {}
    for identity in identities:
        identity_nodes = nodes_by_identity[identity]
        hard_positions = sum(
            graph.nodes[node_id].block.position_count
            for node_id in identity_nodes
            if node_id in hard
        )
        aligned_capacity = sum(
            graph.nodes[node_id].block.position_count
            for node_id in identity_nodes
            if node_id in available
            and node_id not in hard
            and graph.nodes[node_id].block.position_count % block_size == 0
        )
        # Store the smallest optional boundary footprint for each residue. It
        # leaves the greatest possible capacity for utility-ranked full blocks
        # and exactly mirrors the selector's deterministic completion rule.
        minimum_partial_positions: dict[int, int] = {0: 0}
        for node_id in identity_nodes:
            if node_id in hard or node_id not in available:
                continue
            positions = graph.nodes[node_id].block.position_count
            if positions % block_size == 0:
                continue
            updates = dict(minimum_partial_positions)
            for residue, current_positions in minimum_partial_positions.items():
                candidate_positions = current_positions + positions
                candidate_residue = (residue + positions) % block_size
                previous = updates.get(candidate_residue)
                if previous is None or candidate_positions < previous:
                    updates[candidate_residue] = candidate_positions
            minimum_partial_positions = updates
        feasibility[identity] = (
            hard_positions,
            aligned_capacity,
            minimum_partial_positions,
        )

    maximum_final_length = min(int(position_limits[identity]) for identity in identities)
    for final_length in range(maximum_final_length, 0, -1):
        exact_limits: dict[tuple[int, int], int] = {}
        viable = True
        for identity in identities:
            reserve = int(reserved_slots_by_identity.get(identity, 0))
            exact_length = final_length - reserve
            hard_positions, aligned_capacity, partial_positions = feasibility[identity]
            remaining = exact_length - hard_positions
            if remaining < 0:
                viable = False
                break
            partial = partial_positions.get(remaining % block_size)
            if partial is None or partial > remaining or remaining - partial > aligned_capacity:
                viable = False
                break
            exact_limits[identity] = exact_length
        if viable:
            return exact_limits, final_length

    if allow_head_padding:
        exact_limits = {}
        final_lengths: dict[tuple[int, int], int] = {}
        for identity in identities:
            reserve = int(reserved_slots_by_identity.get(identity, 0))
            maximum_exact_length = int(position_limits[identity]) - reserve
            hard_positions, aligned_capacity, partial_positions = feasibility[identity]
            for exact_length in range(maximum_exact_length, hard_positions - 1, -1):
                remaining = exact_length - hard_positions
                partial = partial_positions.get(remaining % block_size)
                if (
                    partial is not None
                    and partial <= remaining
                    and remaining - partial <= aligned_capacity
                ):
                    exact_limits[identity] = exact_length
                    final_lengths[identity] = exact_length + reserve
                    break
            else:
                raise InfeasibleSelectionBudgetError(
                    "no head-padded prototype layout is attainable under the physical cap: "
                    f"identity={identity}, cap={position_limits[identity]}, "
                    f"prototype_slots={reserve}, required_positions={hard_positions}, "
                    f"block_size={block_size}"
                )
        return exact_limits, max(final_lengths.values())

    required_positions = {identity: feasibility[identity][0] for identity in identities}
    raise InfeasibleSelectionBudgetError(
        "no uniform padding-free prototype layout is attainable under the physical caps: "
        f"caps={dict(position_limits)}, prototype_slots={dict(reserved_slots_by_identity)}, "
        f"required_positions={required_positions}, block_size={block_size}"
    )


def _validate_combined_active_layouts(
    measured_active_lengths: Mapping[tuple[int, int], int],
    routed_position_limits: Mapping[tuple[int, int], int],
    planned_routed_active_lengths: Mapping[tuple[int, int], int],
    external_capacity_records: Sequence[HeadLocalCapacityRecord],
) -> tuple[int | None, set[int], bool]:
    """Validate one atomic routed-plus-external dense-cache layout plan."""

    routed_identities = set(routed_position_limits)
    if set(planned_routed_active_lengths) != routed_identities:
        raise HuggingFaceRuntimeError(
            "routed prototype plan does not cover every routed layer/KV head"
        )
    external_by_identity = {
        (record.layer, record.kv_head): record for record in external_capacity_records
    }
    if len(external_by_identity) != len(external_capacity_records):
        raise HuggingFaceRuntimeError("external stable capacity routes must be unique")
    overlap = routed_identities.intersection(external_by_identity)
    if overlap:
        raise HuggingFaceRuntimeError(
            f"routed and external stable cache plans overlap: {sorted(overlap)}"
        )

    combined_limits = dict(routed_position_limits)
    combined_limits.update(
        {identity: record.position_limit for identity, record in external_by_identity.items()}
    )
    planned_lengths = dict(planned_routed_active_lengths)
    planned_lengths.update(
        {identity: record.selected_positions for identity, record in external_by_identity.items()}
    )
    missing = set(combined_limits) - set(measured_active_lengths)
    unexpected = set(measured_active_lengths) - set(combined_limits)
    over_limit = {
        identity: (measured_active_lengths[identity], limit)
        for identity, limit in combined_limits.items()
        if identity in measured_active_lengths and measured_active_lengths[identity] > limit
    }
    mismatches = {
        identity: (measured_active_lengths.get(identity), planned)
        for identity, planned in planned_lengths.items()
        if measured_active_lengths.get(identity) != planned
    }
    if missing or unexpected or over_limit or mismatches:
        raise HuggingFaceRuntimeError(
            "combined routed/external prototype layout violates its plan: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"over_limit={over_limit}, length_mismatches={mismatches}"
        )
    distinct_lengths = set(measured_active_lengths.values())
    return (
        max(distinct_lengths) if distinct_lengths else None,
        distinct_lengths,
        len(set(planned_lengths.values())) > 1,
    )


def _one_pass_prototype_construction(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    objective: BudgetedObjective,
    capabilities: AdapterCapabilities,
    config: RunConfig,
    target: int,
    position_limits: Mapping[tuple[int, int], int] | None,
    key_transform: RotaryPrototypeKeyTransform | None,
    required_anchors: tuple[int, ...],
    prompt_anchor_node_ids: tuple[int, ...],
    evidence_node_ids: tuple[int, ...],
    prototype_weight_by_node: Mapping[int, float],
    prototype_token_weight_by_head: Mapping[tuple[int, int], Any] | None,
    optimization_node_ids: tuple[int, ...] | None,
    additional_exact_source_node_ids: tuple[int, ...],
    external_exact_capacity_records: tuple[HeadLocalCapacityRecord, ...],
    total_active_budget: SelectionBudget,
) -> tuple[SelectionResult, ThreeTierCacheConstruction, float, float]:
    """Plan exact anchors and prototype slots as one deterministic transaction."""

    unit = config.cache.budget_unit
    mandatory_ids = tuple(node.node_id for node in graph.nodes if node.block.mandatory)
    bounded_anchor_cover = config.prototypes.budget_planning_mode == "one_pass_bounded_anchor_cover"
    anchor_candidates = (
        set(required_anchors) if bounded_anchor_cover else {*required_anchors, *mandatory_ids}
    )
    anchor_allowlist = tuple(
        sorted(
            node_id
            for node_id in anchor_candidates
            if not config.prototypes.requires_exact_head(
                graph.nodes[node_id].block.layer,
                graph.nodes[node_id].block.kv_head,
            )
            and config.prototypes.uses_prototype_head(
                graph.nodes[node_id].block.layer,
                graph.nodes[node_id].block.kv_head,
            )
            and config.prototypes.uses_prototype_block(
                graph.nodes[node_id].block.layer,
                graph.nodes[node_id].block.kv_head,
                graph.nodes[node_id].block.modality,
            )
        )
    )
    selection_seconds = 0.0
    tier_seconds = 0.0
    failure: str | None = None
    candidate: SelectionResult | None = None
    assignment_plan: PrototypeAssignmentPlan | None = None
    planned_routed_active_lengths: dict[tuple[int, int], int] = {}
    # First plan conservatively for every allowed anchor. Once selection is
    # known, descriptor-only assignment reveals which anchors actually own an
    # unselected member. Replan until that materialized anchor set is stable;
    # this removes phantom slots without risking a budget underestimate.
    planned_prototype_anchor_ids = anchor_allowlist
    seen_anchor_plans: set[tuple[int, ...]] = set()
    maximum_planning_rounds = len(anchor_allowlist) + 2
    for _planning_round in range(maximum_planning_rounds):
        assignment_plan = None
        if planned_prototype_anchor_ids in seen_anchor_plans:
            failure = (
                "prototype active-anchor planning did not reach a stable assignment: "
                f"repeated={planned_prototype_anchor_ids}"
            )
            candidate = None
            break
        seen_anchor_plans.add(planned_prototype_anchor_ids)
        if unit is BudgetUnit.BLOCKS or unit is BudgetUnit.RETAINED_SLOTS:
            prototype_reserve = len(planned_prototype_anchor_ids)
        elif unit is BudgetUnit.BYTES:
            prototype_reserve = sum(
                graph.nodes[node_id].block.byte_size // graph.nodes[node_id].block.position_count
                for node_id in planned_prototype_anchor_ids
            )
        else:  # pragma: no cover - controlled vocabulary
            raise HuggingFaceRuntimeError(f"unsupported prototype budget unit: {unit}")
        exact_budget = target - prototype_reserve
        exact_position_limits = position_limits
        if position_limits is not None:
            reserved_slots_by_identity: Counter[tuple[int, int]] = Counter(
                (
                    graph.nodes[node_id].block.layer,
                    graph.nodes[node_id].block.kv_head,
                )
                for node_id in planned_prototype_anchor_ids
            )
            routed_slot_counts = {
                int(reserved_slots_by_identity.get(identity, 0)) for identity in position_limits
            }
            allow_head_padding = len(routed_slot_counts) > 1
            try:
                exact_position_limits, _planned_final_length = (
                    _uniform_attainable_prototype_position_limits(
                        graph,
                        position_limits,
                        reserved_slots_by_identity,
                        block_size=full_state.block_size,
                        required_node_ids=required_anchors,
                        prompt_anchor_node_ids=prompt_anchor_node_ids,
                        evidence_node_ids=evidence_node_ids,
                        optimization_node_ids=optimization_node_ids,
                        allow_head_padding=allow_head_padding,
                    )
                )
                planned_routed_active_lengths = {
                    identity: (
                        exact_position_limits[identity]
                        + int(reserved_slots_by_identity.get(identity, 0))
                    )
                    for identity in position_limits
                }
            except InfeasibleSelectionBudgetError as error:
                failure = f"uniform attainable-length planning failed: {error}"
                candidate = None
                break
        if exact_budget < _graph_mandatory_cost(graph, unit):
            failure = (
                f"prototype anchor reserve {prototype_reserve} leaves exact budget "
                f"{exact_budget} below mandatory cost"
            )
            candidate = None
            break
        started = time.perf_counter()
        try:
            candidate = _selection(
                objective,
                exact_budget,
                unit,
                retention_one=False,
                position_limits=exact_position_limits,
                fill_position_limits=exact_position_limits is not None,
                position_fill_quantum=(
                    full_state.block_size if exact_position_limits is not None else None
                ),
                required_node_ids=required_anchors,
                prompt_anchor_node_ids=prompt_anchor_node_ids,
                evidence_node_ids=evidence_node_ids,
                optimization_node_ids=optimization_node_ids,
            )
        except InfeasibleSelectionBudgetError as error:
            failure = f"one-pass exact-anchor selection is infeasible: {error}"
            candidate = None
        selection_seconds += time.perf_counter() - started
        if candidate is None:
            break
        candidate = replace(candidate, budget=SelectionBudget(target, unit))
        assignment_plan = plan_prototype_assignments(
            graph,
            candidate,
            config.prototypes,
            prototype_anchor_node_ids=anchor_allowlist,
        )
        materialized_anchor_ids = assignment_plan.materialized_anchor_ids
        if assignment_plan.incompatibility is not None:
            failure = assignment_plan.incompatibility
            candidate = None
            break
        if materialized_anchor_ids == planned_prototype_anchor_ids:
            break
        if position_limits is not None:
            materialized_slots_by_identity: Counter[tuple[int, int]] = Counter(
                (
                    graph.nodes[node_id].block.layer,
                    graph.nodes[node_id].block.kv_head,
                )
                for node_id in materialized_anchor_ids
            )
            materialized_slot_counts = {
                int(materialized_slots_by_identity.get(identity, 0)) for identity in position_limits
            }
            try:
                materialized_exact_limits, _materialized_final_length = (
                    _uniform_attainable_prototype_position_limits(
                        graph,
                        position_limits,
                        materialized_slots_by_identity,
                        block_size=full_state.block_size,
                        required_node_ids=required_anchors,
                        prompt_anchor_node_ids=prompt_anchor_node_ids,
                        evidence_node_ids=evidence_node_ids,
                        optimization_node_ids=optimization_node_ids,
                        allow_head_padding=len(materialized_slot_counts) > 1,
                    )
                )
            except InfeasibleSelectionBudgetError as error:
                failure = f"materialized-anchor layout planning failed: {error}"
                candidate = None
                break
            # A smaller anchor count often changes only the nominal final
            # length, not the attainable exact-block lattice. Preserve the
            # already computed selection in that case and avoid a redundant
            # full lazy-greedy pass.
            if materialized_exact_limits == exact_position_limits:
                planned_prototype_anchor_ids = materialized_anchor_ids
                planned_routed_active_lengths = {
                    identity: (
                        materialized_exact_limits[identity]
                        + int(materialized_slots_by_identity.get(identity, 0))
                    )
                    for identity in position_limits
                }
                break
        planned_prototype_anchor_ids = materialized_anchor_ids
        candidate = None
    else:  # pragma: no cover - bounded defensive guard
        failure = "prototype active-anchor planning exceeded its convergence bound"
        candidate = None

    if candidate is not None:
        if assignment_plan is None:  # pragma: no cover - candidate and plan are created together
            raise HuggingFaceRuntimeError("prototype candidate is missing its assignment plan")
        estimated_cost, incompatibility = estimate_prototype_active_cost(
            graph,
            candidate,
            config.prototypes,
            position_limits=position_limits,
            prototype_anchor_node_ids=anchor_allowlist,
            assignment_plan=assignment_plan,
        )
        if estimated_cost is None:
            failure = incompatibility or "prototype assignment is incompatible"
        elif estimated_cost > target:
            failure = f"prototype active cost {estimated_cost} exceeds target {target}"
        else:
            started = time.perf_counter()
            construction = construct_three_tier_cache(
                full_state,
                graph,
                candidate,
                capabilities,
                prototype_config=config.prototypes,
                residual_config=config.residual,
                retention_ratio=config.cache.retention_ratio,
                key_transform=key_transform,
                prototype_weight_by_node=prototype_weight_by_node,
                prototype_token_weight_by_head=prototype_token_weight_by_head,
                exact_gather_mode=config.cache.exact_gather_mode,
                total_storage_budget_bytes=config.cache.total_storage_budget_bytes,
                prototype_anchor_node_ids=anchor_allowlist,
                assignment_plan=assignment_plan,
                additional_exact_source_node_ids=additional_exact_source_node_ids,
                active_budget=total_active_budget,
            )
            tier_seconds += time.perf_counter() - started
            if construction.mode is TierConstructionMode.THREE_TIER:
                if position_limits is None:
                    return candidate, construction, selection_seconds, tier_seconds
                active_lengths = {
                    (layout.layer, layout.kv_head): layout.active_cache_length
                    for layout in construction.active_layouts
                }
                (
                    measured_final_length,
                    distinct_lengths,
                    planned_head_padding,
                ) = _validate_combined_active_layouts(
                    active_lengths,
                    position_limits,
                    planned_routed_active_lengths,
                    external_exact_capacity_records,
                )
                if measured_final_length is not None:
                    physical_cap = min(position_limits.values())
                    if len(distinct_lengths) > 1:
                        if not planned_head_padding:  # pragma: no cover - mapping invariant
                            raise HuggingFaceRuntimeError(
                                "combined cache requires padding absent from its layout plan"
                            )
                        construction = replace(
                            construction,
                            reason=(
                                f"{construction.reason}; routed heads use validity-masked "
                                f"padding at physical length {measured_final_length}; logical "
                                f"head lengths={sorted(distinct_lengths)}"
                            ),
                        )
                    elif measured_final_length < physical_cap:
                        construction = replace(
                            construction,
                            reason=(
                                f"{construction.reason}; uniform attainable-length planner "
                                f"underfilled the padding-free physical cap from "
                                f"{physical_cap} to {measured_final_length} positions"
                            ),
                        )
                    return candidate, construction, selection_seconds, tier_seconds
                failure = "one-pass prototype layout has no active KV heads"
            else:
                failure = f"construction: {construction.reason}"

    started = time.perf_counter()
    fallback = _selection(
        objective,
        target,
        unit,
        retention_one=False,
        position_limits=position_limits,
        fill_position_limits=position_limits is not None,
        position_fill_quantum=(full_state.block_size if position_limits is not None else None),
        prompt_anchor_node_ids=prompt_anchor_node_ids,
        evidence_node_ids=evidence_node_ids,
        optimization_node_ids=optimization_node_ids,
    )
    selection_seconds += time.perf_counter() - started
    started = time.perf_counter()
    construction = _exact_construction(
        full_state,
        graph,
        fallback,
        capabilities,
        config,
        key_transform,
        additional_exact_source_node_ids=additional_exact_source_node_ids,
        active_budget=total_active_budget,
    )
    construction = replace(
        construction,
        mode=TierConstructionMode.EXACT_ONLY_INCOMPATIBLE,
        reason=f"one-pass prototype plan rejected: {failure or 'unknown incompatibility'}",
    )
    tier_seconds += time.perf_counter() - started
    return fallback, construction, selection_seconds, tier_seconds


def _prototype_construction(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    objective: BudgetedObjective,
    capabilities: AdapterCapabilities,
    config: RunConfig,
    target: int,
    position_limits: Mapping[tuple[int, int], int] | None,
    key_transform: RotaryPrototypeKeyTransform | None,
    required_anchors: tuple[int, ...],
    prompt_anchor_node_ids: tuple[int, ...],
    evidence_node_ids: tuple[int, ...],
    prototype_weight_by_node: Mapping[int, float],
    prototype_token_weight_by_head: Mapping[tuple[int, int], Any] | None,
    optimization_node_ids: tuple[int, ...] | None,
    additional_exact_source_node_ids: tuple[int, ...],
    external_exact_capacity_records: tuple[HeadLocalCapacityRecord, ...],
    total_active_budget: SelectionBudget,
) -> tuple[SelectionResult, ThreeTierCacheConstruction, float, float]:
    """Reserve active capacity for prototypes with bounded deterministic search."""

    unit = config.cache.budget_unit
    mandatory = _graph_mandatory_cost(graph, unit)
    selection_seconds = 0.0
    tier_seconds = 0.0
    if not capabilities.supports_prototype_merge or (
        config.method is MosaicKVMethod.MOSAICKV_FULL and not capabilities.supports_residual_repair
    ):
        started = time.perf_counter()
        selection = _selection(
            objective,
            target,
            unit,
            retention_one=False,
            position_limits=position_limits,
            fill_position_limits=position_limits is not None,
            position_fill_quantum=(full_state.block_size if position_limits is not None else None),
            prompt_anchor_node_ids=prompt_anchor_node_ids,
            evidence_node_ids=evidence_node_ids,
            optimization_node_ids=optimization_node_ids,
        )
        selection_seconds += time.perf_counter() - started
        started = time.perf_counter()
        construction = construct_three_tier_cache(
            full_state,
            graph,
            selection,
            capabilities,
            prototype_config=config.prototypes,
            residual_config=config.residual,
            retention_ratio=config.cache.retention_ratio,
            key_transform=key_transform,
            prototype_weight_by_node=prototype_weight_by_node,
            prototype_token_weight_by_head=prototype_token_weight_by_head,
            exact_gather_mode=config.cache.exact_gather_mode,
            total_storage_budget_bytes=config.cache.total_storage_budget_bytes,
            additional_exact_source_node_ids=additional_exact_source_node_ids,
            active_budget=total_active_budget,
        )
        tier_seconds += time.perf_counter() - started
        return selection, construction, selection_seconds, tier_seconds

    if config.prototypes.budget_planning_mode in {
        "one_pass_anchor_cover",
        "one_pass_bounded_anchor_cover",
    }:
        return _one_pass_prototype_construction(
            full_state,
            graph,
            objective,
            capabilities,
            config,
            target,
            position_limits,
            key_transform,
            required_anchors,
            prompt_anchor_node_ids,
            evidence_node_ids,
            prototype_weight_by_node,
            prototype_token_weight_by_head,
            optimization_node_ids,
            additional_exact_source_node_ids,
            external_exact_capacity_records,
            total_active_budget,
        )

    failures: Counter[str] = Counter()
    # Reserve the deterministic lower bound for one prototype slot per anchor
    # cover before invoking the expensive lazy selector. Starting at
    # ``target-1`` guarantees that every compressed prototype candidate is
    # over budget and made LLaVA repeat a large 32-head selection dozens of
    # times. Mandatory blocks can themselves anchor a segment, so include them
    # in the conservative slot set; a small over-reservation is refined below.
    prototype_anchor_lower_bound = tuple(
        sorted(
            {
                *required_anchors,
                *(node.node_id for node in graph.nodes if node.block.mandatory),
            }
        )
    )
    if unit is BudgetUnit.BLOCKS or unit is BudgetUnit.RETAINED_SLOTS:
        reserved_prototype_cost = len(prototype_anchor_lower_bound)
    elif unit is BudgetUnit.BYTES:
        reserved_prototype_cost = sum(
            graph.nodes[node_id].block.byte_size // graph.nodes[node_id].block.position_count
            for node_id in prototype_anchor_lower_bound
        )
    else:  # pragma: no cover - controlled vocabulary
        raise HuggingFaceRuntimeError(f"unsupported prototype budget unit: {unit}")
    upper = max(mandatory, target - reserved_prototype_cost)
    span = upper - mandatory
    if span <= 128:
        candidates = tuple(range(upper, mandatory - 1, -1))
    else:
        # Byte budgets commonly span millions of integer values. Searching
        # every byte is neither meaningful (blocks are indivisible) nor
        # tractable, so evaluate a deterministic 128-point coarse-to-fine
        # lattice. The prior method-specific branch left ``candidates``
        # undefined for ordinary MosaicKV byte budgets.
        candidates = tuple(
            sorted(
                {mandatory + math.floor(span * index / 127) for index in range(128)},
                reverse=True,
            )
        )
    pending = deque(candidates)
    representation_failure_budget: int | None = None
    refinement_inserted = False
    costs = tuple(SelectionBudget(1, unit).cost(node.block) for node in graph.nodes)
    while pending:
        exact_budget = pending.popleft()
        started = time.perf_counter()
        try:
            candidate = _selection(
                objective,
                exact_budget,
                unit,
                retention_one=False,
                position_limits=position_limits,
                required_node_ids=required_anchors,
                prompt_anchor_node_ids=prompt_anchor_node_ids,
                evidence_node_ids=evidence_node_ids,
                optimization_node_ids=optimization_node_ids,
            )
        except InfeasibleSelectionBudgetError as error:
            selection_seconds += time.perf_counter() - started
            failures[f"selection: {error}"] += 1
            if representation_failure_budget is not None and not refinement_inserted:
                pending.clear()
                pending.extend(
                    _refined_budget_candidates(
                        representation_failure_budget,
                        exact_budget,
                        costs,
                    )
                )
                refinement_inserted = True
            continue
        candidate = replace(candidate, budget=SelectionBudget(target, unit))
        selection_seconds += time.perf_counter() - started
        estimated_cost, incompatibility = estimate_prototype_active_cost(
            graph,
            candidate,
            config.prototypes,
            position_limits=position_limits,
        )
        if estimated_cost is None:
            failures[f"assignment: {incompatibility or 'unknown incompatibility'}"] += 1
            representation_failure_budget = exact_budget
            continue
        if estimated_cost > target:
            failures[f"active cost {estimated_cost} exceeds target {target}"] += 1
            representation_failure_budget = exact_budget
            continue
        # The estimate above already validates the deterministic assignment and
        # its exact active cost.  Materialize the chosen representation once,
        # including residuals when the method requests them.  The old two-pass
        # path first built every prototype with residuals disabled and then
        # rebuilt the identical prototypes with residuals enabled.  For long
        # videos that duplicated thousands of gather/RoPE/FP32 merge operations
        # and dominated compression latency without adding a correctness check.
        started = time.perf_counter()
        construction = construct_three_tier_cache(
            full_state,
            graph,
            candidate,
            capabilities,
            prototype_config=config.prototypes,
            residual_config=config.residual,
            retention_ratio=config.cache.retention_ratio,
            key_transform=key_transform,
            prototype_weight_by_node=prototype_weight_by_node,
            prototype_token_weight_by_head=prototype_token_weight_by_head,
            exact_gather_mode=config.cache.exact_gather_mode,
            total_storage_budget_bytes=config.cache.total_storage_budget_bytes,
            additional_exact_source_node_ids=additional_exact_source_node_ids,
            active_budget=total_active_budget,
        )
        tier_seconds += time.perf_counter() - started
        if construction.mode is TierConstructionMode.THREE_TIER:
            return candidate, construction, selection_seconds, tier_seconds
        failures[f"construction: {construction.reason}"] += 1
        representation_failure_budget = exact_budget
    started = time.perf_counter()
    fallback = _selection(
        objective,
        target,
        unit,
        retention_one=False,
        position_limits=position_limits,
        prompt_anchor_node_ids=prompt_anchor_node_ids,
        evidence_node_ids=evidence_node_ids,
        optimization_node_ids=optimization_node_ids,
    )
    selection_seconds += time.perf_counter() - started
    started = time.perf_counter()
    construction = _exact_construction(
        full_state,
        graph,
        fallback,
        capabilities,
        config,
        key_transform,
        additional_exact_source_node_ids=additional_exact_source_node_ids,
        active_budget=total_active_budget,
    )
    summary = "; ".join(f"{count}x {reason}" for reason, count in failures.most_common(4))
    construction = replace(
        construction,
        mode=TierConstructionMode.EXACT_ONLY_INCOMPATIBLE,
        reason=(
            "prototype budget search found no feasible representation"
            + (f": {summary}" if summary else "")
        ),
    )
    tier_seconds += time.perf_counter() - started
    return fallback, construction, selection_seconds, tier_seconds


def build_compression_plan(
    full_state: FullKVState,
    forecast: QueryForecast | None,
    forecast_attention_by_node: Mapping[int, float],
    capabilities: AdapterCapabilities,
    config: RunConfig,
    *,
    position_limits: Mapping[tuple[int, int], int] | None = None,
    attention_provenance: str = "eager_prompt_and_isolated_draft_attention_v1",
    key_transform: RotaryPrototypeKeyTransform | None = None,
    prompt_attentions: Sequence[Any] | None = None,
    prompt_attention_by_node: (
        Mapping[int, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None
    ) = None,
    prompt_query_attention_by_node: Mapping[int, Any] | None = None,
    prototype_token_weight_by_head: Mapping[tuple[int, int], Any] | None = None,
    graph_head_uncertainty: Mapping[tuple[int, int], float] | None = None,
) -> MosaicKVCompressionPlan:
    """Run graph, utility, selection, and tier construction for one prefill."""

    if config.method.is_full_cache:
        raise ValueError("FullKV does not run the MosaicKV compression planner")
    source_budget, shared_target = _resolve_budget(full_state, config)
    target, position_limits = _mosaickv_effective_capacity(
        full_state,
        config,
        shared_target,
        position_limits,
    )
    total_active_budget = SelectionBudget(target, config.cache.budget_unit)
    planner_target = target
    all_head_routes = {(block.layer, block.kv_head) for block in full_state.blocks}
    routed_head_routes: set[tuple[int, int]] | None = None
    optimization_node_ids: tuple[int, ...] | None = None
    stable_prompt_reference_ids: tuple[int, ...] = ()
    planner_source_node_ids: tuple[int, ...] | None = None
    external_exact_node_ids: tuple[int, ...] = ()
    external_exact_capacity_records: tuple[HeadLocalCapacityRecord, ...] = ()
    external_exact_logit_shortlist: tuple[HeadLocalPromptSelectionAlternative, ...] = ()
    prompt_attention_values = (
        None
        if prompt_attention_by_node is None
        else np.asarray(prompt_attention_by_node, dtype=np.float64)
        if not isinstance(prompt_attention_by_node, Mapping)
        else np.fromiter(
            (float(prompt_attention_by_node[node_id]) for node_id in range(len(full_state.blocks))),
            dtype=np.float64,
            count=len(full_state.blocks),
        )
    )
    if prompt_attention_values is not None and prompt_attention_values.shape != (
        len(full_state.blocks),
    ):
        raise HuggingFaceRuntimeError(
            "prompt attention scores must cover every source block exactly once"
        )
    stable_routing_seconds = 0.0
    similarity_shortlist_seconds = 0.0
    planner_mapping_seconds = 0.0
    if config.selection.routed_head_only_planning:
        if prompt_attention_by_node is None or graph_head_uncertainty is None:
            raise HuggingFaceRuntimeError(
                "routed-head-only planning requires prompt attention and head scores"
            )
        stable_routing_started = time.perf_counter()
        routed_head_routes = set(
            _uncertain_head_routes(
                all_head_routes,
                graph_head_uncertainty,
                fraction=config.prototypes.uncertain_head_fraction,
                threshold=config.prototypes.uncertain_head_threshold,
                label="routed-head-only planner",
            )
        )
        optimization_node_ids = tuple(
            node_id
            for node_id, block in enumerate(full_state.blocks)
            if (block.layer, block.kv_head) in routed_head_routes
        )
        if config.cache.retention_ratio < 1.0:
            exact_head_routes = {
                (block.layer, block.kv_head)
                for block in full_state.blocks
                if config.prototypes.requires_exact_head(block.layer, block.kv_head)
            }
            can_externalize_stable = bool(
                config.selection.stable_head_prompt_routing and position_limits is not None
            )
            if can_externalize_stable:
                if position_limits is None:  # pragma: no cover - guarded above
                    raise HuggingFaceRuntimeError(
                        "external stable routing requires dense position limits"
                    )
                stable_head_routes = tuple(
                    sorted(all_head_routes - routed_head_routes - exact_head_routes)
                )
                external_selection = select_prompt_attention_topk_ids_by_head(
                    full_state,
                    total_active_budget,
                    prompt_attention_by_node=(
                        prompt_attention_by_node
                        if isinstance(prompt_attention_by_node, Mapping)
                        else {}
                    ),
                    prompt_attention_scores=prompt_attention_values,
                    forecast_attention_by_node=(
                        forecast_attention_by_node
                        if config.selection.stable_head_adaptive_visual_swaps
                        or config.selection.stable_head_future_query_reranking
                        else None
                    ),
                    head_routes=stable_head_routes,
                    position_limits=position_limits,
                    minimum_visual_fraction=(config.selection.stable_head_visual_fraction),
                    adaptive_visual_swaps=(config.selection.stable_head_adaptive_visual_swaps),
                    minimum_text_mass_fraction=(
                        config.selection.stable_head_minimum_text_mass_fraction
                    ),
                    visual_gain_ratio=(config.selection.stable_head_visual_gain_ratio),
                    maximum_visual_swaps=(config.selection.stable_head_maximum_visual_swaps),
                    swap_risk_gate=config.selection.stable_head_swap_risk_gate,
                    maximum_normalized_text_loss=(
                        config.selection.stable_head_maximum_normalized_text_loss
                    ),
                    forecast_gain_ratio=(config.selection.stable_head_forecast_gain_ratio),
                    future_query_reranking=(config.selection.stable_head_future_query_reranking),
                    future_minimum_prompt_mass_fraction=(
                        config.selection.stable_head_future_minimum_prompt_mass_fraction
                    ),
                    future_gain_ratio=(config.selection.stable_head_future_gain_ratio),
                    maximum_future_replacements=(
                        config.selection.stable_head_maximum_future_replacements
                    ),
                    future_divergence_fail_closed=(
                        config.selection.stable_head_future_divergence_fail_closed
                    ),
                    future_minimum_prompt_forecast_tv=(
                        config.selection.stable_head_future_minimum_prompt_forecast_tv
                    ),
                    future_risk_gate=config.selection.stable_head_future_risk_gate,
                    future_maximum_normalized_prompt_loss=(
                        config.selection.stable_head_future_maximum_normalized_prompt_loss
                    ),
                    future_minimum_normalized_gain=(
                        config.selection.stable_head_future_minimum_normalized_gain
                    ),
                    prompt_query_attention_by_node=prompt_query_attention_by_node,
                    query_coverage_reranking=(
                        config.selection.stable_head_query_coverage_reranking
                    ),
                    query_coverage_minimum_prompt_mass_fraction=(
                        config.selection.stable_head_query_coverage_minimum_prompt_mass_fraction
                    ),
                    query_coverage_maximum_replacements=(
                        config.selection.stable_head_query_coverage_maximum_replacements
                    ),
                    query_coverage_minimum_normalized_gain=(
                        config.selection.stable_head_query_coverage_minimum_normalized_gain
                    ),
                    query_coverage_initial_selection=(
                        config.selection.stable_head_query_coverage_initial_selection
                    ),
                    query_coverage_initial_fraction=(
                        config.selection.stable_head_query_coverage_initial_fraction
                    ),
                    query_coverage_initial_prompt_weight=(
                        config.selection.stable_head_query_coverage_initial_prompt_weight
                    ),
                    query_coverage_initial_minimum_prompt_mass_fraction=(
                        config.selection.stable_head_query_coverage_initial_minimum_prompt_mass_fraction
                    ),
                    query_coverage_initial_objective=(
                        config.selection.stable_head_query_coverage_initial_objective
                    ),
                    query_coverage_initial_cap_fraction=(
                        config.selection.stable_head_query_coverage_initial_cap_fraction
                    ),
                    query_coverage_cross_head_allocation=(
                        config.selection.stable_head_query_coverage_cross_head_allocation
                    ),
                    query_coverage_cross_head_stability_gate=(
                        config.selection.stable_head_query_coverage_cross_head_stability_gate
                    ),
                    query_coverage_cross_head_stability_risk_mode=(
                        config.selection.stable_head_query_coverage_cross_head_stability_risk_mode
                    ),
                    query_coverage_cross_head_minimum_normalized_gain=(
                        config.selection.stable_head_query_coverage_cross_head_minimum_normalized_gain
                    ),
                    query_coverage_cross_head_gain_to_prompt_loss_ratio=(
                        config.selection.stable_head_query_coverage_cross_head_gain_to_prompt_loss_ratio
                    ),
                    query_coverage_output_impact_reservation=(
                        config.selection.stable_head_query_coverage_output_impact_reservation
                    ),
                    query_coverage_output_impact_fraction=(
                        config.selection.stable_head_query_coverage_output_impact_fraction
                    ),
                    query_coverage_layout_group_reservation=(
                        config.selection.stable_head_query_coverage_layout_group_reservation
                    ),
                    query_coverage_layout_group_fraction=(
                        config.selection.stable_head_query_coverage_layout_group_fraction
                    ),
                    query_coverage_layout_group_maximum_blocks=(
                        config.selection.stable_head_query_coverage_layout_group_maximum_blocks
                    ),
                    query_coverage_layout_group_row_bins=(
                        config.selection.stable_head_query_coverage_layout_group_row_bins
                    ),
                    query_coverage_layout_group_atomic_selection=(
                        config.selection.stable_head_query_coverage_layout_group_atomic_selection
                    ),
                    query_coverage_layout_group_atomic_selection_mode=(
                        config.selection.stable_head_query_coverage_layout_group_atomic_selection_mode
                    ),
                    query_coverage_layout_group_minimum_utilization_fraction=(
                        config.selection.stable_head_query_coverage_layout_group_minimum_utilization_fraction
                    ),
                    query_coverage_layout_group_minimum_normalized_gain=(
                        config.selection.stable_head_query_coverage_layout_group_minimum_normalized_gain
                    ),
                    query_coverage_layout_group_gain_to_risk_ratio=(
                        config.selection.stable_head_query_coverage_layout_group_gain_to_risk_ratio
                    ),
                    short_horizon_logit_stability=(
                        config.selection.stable_head_short_horizon_logit_stability
                    ),
                    progressive_logit_safe_eviction=(
                        config.selection.stable_head_progressive_logit_safe_eviction
                    ),
                    hierarchical_logit_safe_eviction=(
                        config.selection.stable_head_hierarchical_logit_safe_eviction
                    ),
                )
                external_exact_node_ids = external_selection.selected_node_ids
                external_exact_capacity_records = external_selection.capacity_records
                external_exact_logit_shortlist = external_selection.logit_stability_shortlist
                external_cost = sum(
                    total_active_budget.cost(full_state.blocks[node_id])
                    for node_id in external_exact_node_ids
                )
                planner_target = target - external_cost
                if planner_target < 1:
                    raise HuggingFaceRuntimeError(
                        "stable-head exact routing consumed the complete active budget"
                    )
                planner_head_routes = routed_head_routes | exact_head_routes
                planner_source_node_ids = tuple(
                    node_id
                    for node_id, block in enumerate(full_state.blocks)
                    if (block.layer, block.kv_head) in planner_head_routes
                )
            else:
                if config.selection.stable_head_prompt_routing:
                    prompt_attention_mapping = (
                        prompt_attention_by_node
                        if isinstance(prompt_attention_by_node, Mapping)
                        else {
                            node_id: float(prompt_attention_by_node[node_id])
                            for node_id in range(len(full_state.blocks))
                        }
                    )
                    stable_prompt_reference_ids = select_prompt_attention_topk_ids(
                        full_state,
                        total_active_budget,
                        prompt_attention_by_node=prompt_attention_mapping,
                        position_limits=position_limits,
                    )
                    stable_source_node_ids = set(stable_prompt_reference_ids)
                else:
                    # Without stable Prompt routing, stable heads remain in the
                    # ordinary planner so no source block becomes unreachable.
                    stable_source_node_ids = {
                        node_id
                        for node_id, block in enumerate(full_state.blocks)
                        if (block.layer, block.kv_head) not in routed_head_routes
                    }
                exact_route_source_node_ids = {
                    node_id
                    for node_id, block in enumerate(full_state.blocks)
                    if (block.layer, block.kv_head) in exact_head_routes
                }
                planner_source_node_ids = tuple(
                    sorted(
                        set(optimization_node_ids)
                        | stable_source_node_ids
                        | exact_route_source_node_ids
                    )
                )
        stable_routing_seconds = time.perf_counter() - stable_routing_started
    shortlist_started = time.perf_counter()
    similarity_candidate_node_ids = _similarity_shortlist(
        full_state,
        config,
        source_budget=source_budget,
        target_budget=target,
        prompt_attention_by_node=prompt_attention_by_node,
        head_uncertainty=graph_head_uncertainty,
        eligible_source_node_ids=optimization_node_ids,
    )
    similarity_shortlist_seconds = time.perf_counter() - shortlist_started
    if optimization_node_ids is not None:
        optimization_nodes = frozenset(optimization_node_ids)
        similarity_candidate_node_ids = tuple(
            node_id
            for node_id in (
                optimization_node_ids
                if similarity_candidate_node_ids is None
                else similarity_candidate_node_ids
            )
            if node_id in optimization_nodes
        )
    if planner_source_node_ids is not None:
        planner_sources = frozenset(planner_source_node_ids)
        similarity_candidate_node_ids = tuple(
            node_id
            for node_id in (
                planner_source_node_ids
                if similarity_candidate_node_ids is None
                else similarity_candidate_node_ids
            )
            if node_id in planner_sources
        )
    graph_started = time.perf_counter()
    graph = build_evidence_graph(
        full_state,
        config.graph,
        prompt_attentions=prompt_attentions,
        prompt_attention_query_window=config.forecasting.prompt_window,
        similarity_candidate_node_ids=similarity_candidate_node_ids,
        source_node_ids=planner_source_node_ids,
    )
    graph_seconds = time.perf_counter() - graph_started
    mapping_started = time.perf_counter()
    source_to_planner = {node.source_id: node.node_id for node in graph.nodes}
    planner_optimization_node_ids = (
        None
        if optimization_node_ids is None
        else tuple(
            source_to_planner[node_id]
            for node_id in optimization_node_ids
            if node_id in source_to_planner
        )
    )
    planner_forecast_attention_by_node = {
        node.node_id: float(forecast_attention_by_node[node.source_id]) for node in graph.nodes
    }
    planner_prompt_attention_by_node = (
        None
        if prompt_attention_by_node is None
        else {node.node_id: float(prompt_attention_by_node[node.source_id]) for node in graph.nodes}
    )
    planner_mapping_seconds = time.perf_counter() - mapping_started
    utility_started = time.perf_counter()
    utilities = compute_block_utilities(
        graph,
        config.utility,
        forecast_attention_by_node=planner_forecast_attention_by_node,
        attention_provenance=attention_provenance,
        rope_aware=True,
        value_aware_node_ids=planner_optimization_node_ids,
    )
    objective = BudgetedObjective(graph, utilities, config.selection)
    utility_seconds = time.perf_counter() - utility_started
    reservation_started = time.perf_counter()
    if config.selection.prompt_anchor_fraction and prompt_attention_by_node is None:
        raise HuggingFaceRuntimeError(
            "prompt-anchored selection requires prompt-only block attention"
        )
    prototype_path_enabled = bool(
        config.cache.retention_ratio < 1.0
        and config.method in {MosaicKVMethod.MOSAICKV_PROTO, MosaicKVMethod.MOSAICKV_FULL}
        and capabilities.supports_prototype_merge
        and (
            config.method is not MosaicKVMethod.MOSAICKV_FULL
            or capabilities.supports_residual_repair
        )
    )
    required_prototype_anchor_ids = (
        _required_prototype_anchors(
            objective,
            config,
            position_fill_quantum=(full_state.block_size if position_limits is not None else None),
            prompt_attention_by_node=planner_prompt_attention_by_node,
        )
        if prototype_path_enabled
        else ()
    )
    planner_head_routes = {(node.block.layer, node.block.kv_head) for node in graph.nodes}
    routed_position_limits = (
        None
        if position_limits is None
        else {
            identity: int(limit)
            for identity, limit in position_limits.items()
            if identity in planner_head_routes
        }
    )
    reservation_position_limits = (
        _prototype_aware_reservation_limits(
            graph,
            config,
            routed_position_limits,
            required_prototype_anchor_ids,
        )
        if prototype_path_enabled
        else routed_position_limits
    )
    reservation_fill_quantum = full_state.block_size if routed_position_limits is not None else None
    planning_position_limits = routed_position_limits
    prompt_anchor_node_ids = _bounded_reservation(
        objective,
        planner_target,
        config.cache.budget_unit,
        config.selection.prompt_anchor_fraction,
        ({} if planner_prompt_attention_by_node is None else planner_prompt_attention_by_node),
        position_limits=reservation_position_limits,
        position_fill_quantum=reservation_fill_quantum,
        reserved_node_ids=required_prototype_anchor_ids,
        per_head=(
            config.prototypes.budget_planning_mode == "one_pass_bounded_anchor_cover"
            and reservation_position_limits is not None
        ),
    )
    if config.selection.stable_head_prompt_routing:
        if (
            prompt_attention_by_node is None
            or planner_prompt_attention_by_node is None
            or graph_head_uncertainty is None
        ):
            raise HuggingFaceRuntimeError(
                "stable-head prompt routing requires prompt attention and head scores"
            )
        if routed_head_routes is None:
            routed_head_routes = set(
                _uncertain_head_routes(
                    all_head_routes,
                    graph_head_uncertainty,
                    fraction=config.prototypes.uncertain_head_fraction,
                    threshold=config.prototypes.uncertain_head_threshold,
                    label="stable-head prompt reservation",
                )
            )
        if external_exact_node_ids:
            stable_prompt_ids: tuple[int, ...] = ()
        elif not stable_prompt_reference_ids:
            stable_prompt_reference_ids = select_prompt_attention_topk_ids(
                full_state,
                SelectionBudget(target, config.cache.budget_unit),
                prompt_attention_by_node=(
                    prompt_attention_by_node
                    if isinstance(prompt_attention_by_node, Mapping)
                    else {
                        node_id: float(prompt_attention_by_node[node_id])
                        for node_id in range(len(full_state.blocks))
                    }
                ),
                position_limits=reservation_position_limits,
            )
        else:
            stable_prompt_ids = tuple(
                source_to_planner[node_id]
                for node_id in stable_prompt_reference_ids
                if node_id in source_to_planner
                if (
                    full_state.blocks[node_id].layer,
                    full_state.blocks[node_id].kv_head,
                )
                not in routed_head_routes
                and (
                    not config.selection.stable_head_uniform_fill
                    or full_state.blocks[node_id].mandatory
                    or full_state.blocks[node_id].position_count % full_state.block_size == 0
                )
            )
        if (
            not external_exact_node_ids
            and position_limits is not None
            and not config.selection.stable_head_uniform_fill
        ):
            stable_positions: Counter[tuple[int, int]] = Counter()
            for node_id in stable_prompt_ids:
                block = graph.nodes[node_id].block
                stable_positions[(block.layer, block.kv_head)] += block.position_count
            planning_position_limits = dict(position_limits)
            for identity in all_head_routes - routed_head_routes:
                planning_position_limits[identity] = int(stable_positions[identity])
            reservation_position_limits = planning_position_limits
        additional_prompt_ids = _bounded_reservation(
            objective,
            planner_target,
            config.cache.budget_unit,
            config.selection.prompt_anchor_fraction,
            planner_prompt_attention_by_node,
            position_limits=reservation_position_limits,
            position_fill_quantum=reservation_fill_quantum,
            reserved_node_ids=(
                *required_prototype_anchor_ids,
                *stable_prompt_ids,
            ),
            per_head=(
                config.prototypes.budget_planning_mode == "one_pass_bounded_anchor_cover"
                and reservation_position_limits is not None
            ),
        )
        prompt_anchor_node_ids = tuple(sorted(set(stable_prompt_ids) | set(additional_prompt_ids)))
    evidence_node_ids = _bounded_reservation(
        objective,
        planner_target,
        config.cache.budget_unit,
        config.selection.evidence_preservation_fraction,
        _evidence_preservation_scores(
            objective,
            mode=config.selection.evidence_reservation_mode,
            prompt_attention_by_node=planner_prompt_attention_by_node,
            grid_rows=config.selection.evidence_grid_rows,
            grid_columns=config.selection.evidence_grid_columns,
        ),
        position_limits=reservation_position_limits,
        position_fill_quantum=reservation_fill_quantum,
        reserved_node_ids=(*required_prototype_anchor_ids, *prompt_anchor_node_ids),
    )
    reservation_seconds = time.perf_counter() - reservation_started
    if config.cache.retention_ratio == 1.0:
        selector_started = time.perf_counter()
        selection = _selection(
            objective,
            source_budget,
            config.cache.budget_unit,
            retention_one=True,
        )
        if selection.selected_node_ids != tuple(range(len(full_state.blocks))):
            raise HuggingFaceRuntimeError("retention 1.0 did not select every source block")
        selector_seconds = time.perf_counter() - selector_started
        selection_seconds = utility_seconds + reservation_seconds + selector_seconds
        tier_started = time.perf_counter()
        construction = construct_three_tier_cache(
            full_state,
            graph,
            selection,
            capabilities,
            prototype_config=config.prototypes,
            residual_config=config.residual,
            retention_ratio=1.0,
            key_transform=key_transform,
            exact_gather_mode=config.cache.exact_gather_mode,
            total_storage_budget_bytes=config.cache.total_storage_budget_bytes,
        )
    elif config.method is MosaicKVMethod.MOSAICKV_EXACT:
        selector_started = time.perf_counter()
        selection = _selection(
            objective,
            planner_target,
            config.cache.budget_unit,
            retention_one=False,
            position_limits=planning_position_limits,
            fill_position_limits=planning_position_limits is not None,
            position_fill_quantum=(
                full_state.block_size if planning_position_limits is not None else None
            ),
            prompt_anchor_node_ids=prompt_anchor_node_ids,
            evidence_node_ids=evidence_node_ids,
            optimization_node_ids=planner_optimization_node_ids,
        )
        selector_seconds = time.perf_counter() - selector_started
        selection_seconds = utility_seconds + reservation_seconds + selector_seconds
        tier_started = time.perf_counter()
        try:
            construction = _exact_construction(
                full_state,
                graph,
                selection,
                capabilities,
                config,
                key_transform,
                additional_exact_source_node_ids=external_exact_node_ids,
                active_budget=total_active_budget,
            )
        except Exception as error:
            error.__dict__["mosaickv_external_exact_capacity_records"] = (
                external_exact_capacity_records
            )
            raise
    else:
        try:
            selection, construction, selector_seconds, tier_seconds = _prototype_construction(
                full_state,
                graph,
                objective,
                capabilities,
                config,
                planner_target,
                planning_position_limits,
                key_transform,
                required_prototype_anchor_ids,
                prompt_anchor_node_ids,
                evidence_node_ids,
                forecast_attention_by_node,
                prototype_token_weight_by_head,
                planner_optimization_node_ids,
                external_exact_node_ids,
                external_exact_capacity_records,
                total_active_budget,
            )
        except Exception as error:
            error.__dict__["mosaickv_external_exact_capacity_records"] = (
                external_exact_capacity_records
            )
            raise
        selection_seconds = utility_seconds + reservation_seconds + selector_seconds
        tier_started = None
    if tier_started is not None:
        tier_seconds = time.perf_counter() - tier_started
    if construction.mode is TierConstructionMode.THREE_TIER:
        effective_method = config.method.value
    elif construction.mode is TierConstructionMode.RETENTION_ONE:
        effective_method = f"{config.method.value}__retention_one_exact"
    elif config.method is MosaicKVMethod.MOSAICKV_EXACT:
        effective_method = config.method.value
    elif construction.mode is TierConstructionMode.EXACT_ONLY_UNSAFE:
        effective_method = f"{config.method.value}__mosaickv_exact_safety_fallback"
    else:
        effective_method = f"{config.method.value}__{construction.mode.value}"
    if config.ablation.variant is not AblationVariant.NONE:
        effective_method = f"{effective_method}__ablation_{config.ablation.variant.value}"
    graph_head_routes = tuple(
        sorted(
            {
                (full_state.blocks[node_id].layer, full_state.blocks[node_id].kv_head)
                for node_id in (similarity_candidate_node_ids or ())
                if not full_state.blocks[node_id].mandatory
            }
        )
    )
    return MosaicKVCompressionPlan(
        method=config.method,
        effective_method=effective_method,
        full_state=full_state,
        forecast=forecast,
        graph=graph,
        utilities=utilities,
        selection=selection,
        construction=construction,
        source_budget_value=source_budget,
        shared_active_budget_value=shared_target,
        active_budget_value=target,
        planner_active_budget_value=planner_target,
        configured_physical_capacity_fraction=(config.selection.physical_capacity_fraction),
        graph_seconds=graph_seconds,
        selection_seconds=selection_seconds,
        utility_seconds=utility_seconds,
        reservation_seconds=reservation_seconds,
        selector_seconds=selector_seconds,
        tier_seconds=tier_seconds,
        similarity_candidate_node_ids=similarity_candidate_node_ids,
        optimization_node_ids=optimization_node_ids,
        planner_source_node_ids=planner_source_node_ids,
        external_exact_node_ids=external_exact_node_ids,
        external_exact_capacity_records=external_exact_capacity_records,
        external_exact_logit_shortlist=external_exact_logit_shortlist,
        prompt_attention_by_node=prompt_attention_values,
        stable_routing_seconds=stable_routing_seconds,
        similarity_shortlist_seconds=similarity_shortlist_seconds,
        planner_mapping_seconds=planner_mapping_seconds,
        graph_head_routes=graph_head_routes,
        graph_routing_provenance=(
            "routed_head_only_prompt_shortlist_plus_normalized_head_entropy"
            if config.selection.routed_head_only_planning
            else "prompt_hierarchical_shortlist_plus_normalized_head_entropy"
            if config.graph.uncertain_head_only
            else (
                "prompt_hierarchical_shortlist"
                if similarity_candidate_node_ids is not None
                else "full_graph"
            )
        ),
    )


def _plan_has_live_residual_repair(
    method: MosaicKVMethod,
    plan: MosaicKVCompressionPlan | None,
    *,
    repair_enabled: bool = True,
    repair_policy: RepairPolicy = RepairPolicy.ENTROPY_OR_PROTOTYPE_RISK,
) -> bool:
    """Whether online decoding must pay for repair instrumentation."""

    return bool(
        method is MosaicKVMethod.MOSAICKV_FULL
        and repair_enabled
        and repair_policy is not RepairPolicy.NONE
        and plan is not None
        and plan.construction.mode is TierConstructionMode.THREE_TIER
        and plan.construction.prototypes
        and plan.construction.residual_storage.payloads
    )


def compare_runtime_retention_one(
    adapter: HuggingFaceMultimodalAdapter,
    prepared: PreparedInputs,
    config: RunConfig,
) -> ParityReport:
    """Compare packed retention-1 decoding with untouched FullKV from one prefill."""

    if config.method.is_full_cache or config.cache.retention_ratio != 1.0:
        raise ValueError("runtime parity requires a compressed method at retention ratio 1.0")
    if not (
        config.method.is_mosaickv
        or config.method.is_compressed_baseline
        or config.method.is_published_reimplementation
    ):
        raise ValueError("runtime parity method is unsupported")
    if config.generation.max_new_tokens < 2:
        raise ValueError("runtime parity requires at least two generated tokens")
    torch = _torch()
    prefill = adapter.prefill(
        prepared,
        capture_queries=config.method.is_mosaickv,
        capture_attentions=(
            config.method.is_mosaickv
            or config.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
            or config.method.is_published_reimplementation
        ),
        attention_query_window=(
            None
            if config.method.is_published_reimplementation
            else (
                max(1, config.forecasting.prompt_window)
                if config.method.is_mosaickv
                or config.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
                else None
            )
        ),
    )
    snapshot = adapter.extract_past_key_values(prefill.state.past_key_values)
    full_state = FullKVState.from_cache_snapshot(
        snapshot,
        modality_spans=prepared.modality_map,
        token_ids=prepared.model_inputs["input_ids"],
        block_size=config.cache.block_size,
        original_logical_sequence_length=prefill.state.logical_sequence_length,
        next_decode_position=prefill.state.next_decode_position,
        # The terminal prompt token is the only universally non-compressible
        # decode boundary.  A BOS token is useful but is not inherently
        # mandatory; marking both endpoints can make a valid 50% block budget
        # impossible for short multimodal prompts.
        mandatory_logical_positions=(
            tuple(range(prefill.state.logical_sequence_length))
            if config.method.is_published_reimplementation
            else (prefill.state.logical_sequence_length - 1,)
        ),
    )
    if config.output_preserving is not None:
        packed = pack_runtime_cache(
            adapter,
            full_state,
            MosaicKVState.retention_one(full_state),
            (),
            (),
            packing_mode=config.cache.packing_mode,
        )
    elif config.method.is_mosaickv:
        forecast, rollout = forecast_with_rollout(
            adapter,
            prefill,
            config.forecasting,
            capture_attentions=True,
        )
        prototype_token_attention = _forecast_token_attention_by_head(
            full_state,
            prefill,
            rollout,
            config,
        )
        mosaic_plan = build_compression_plan(
            full_state,
            forecast,
            _block_attention_from_token_attention(full_state, prototype_token_attention),
            adapter.capabilities,
            config,
            prompt_attentions=prefill.attention_weights,
            prototype_token_weight_by_head=prototype_token_attention,
        )
        packed = pack_runtime_cache(
            adapter,
            full_state,
            mosaic_plan.construction.state,
            mosaic_plan.construction.prototypes,
            (),
            packing_mode=config.cache.packing_mode,
        )
    elif config.method.is_compressed_baseline:
        prompt_attention = (
            _prompt_attention_by_node(
                full_state,
                prefill,
                prompt_window=config.forecasting.prompt_window,
            )
            if config.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
            else None
        )
        baseline_plan = build_exact_baseline_plan(
            full_state,
            config.method,
            config.cache,
            seed=config.execution.seed,
            prompt_attention_by_node=prompt_attention,
            value_chunk_size=config.graph.similarity_chunk_size,
        )
        packed = pack_runtime_cache(adapter, full_state, baseline_plan.state, (), ())
    elif config.method.is_lookm_reimplementation:
        lookm_plan = build_lookm_reimpl_plan(
            full_state,
            prefill.attention_weights,
            config.lookm,
            config.cache,
        )
        payloads, slot_records = lookm_runtime_payloads(lookm_plan)
        packed = pack_runtime_payloads(
            adapter,
            full_state,
            payloads,
            slot_records,
            packing_mode=config.cache.packing_mode,
        )
    elif config.method.is_prefixkv_reimplementation:
        prefix_profile = (
            load_prefixkv_profile(
                config.prefixkv.profile_path,
                model_id=config.model.id,
                model_revision=config.model.revision,
                target_retention_ratio=(
                    config.prefixkv.profile_reference_retention_ratio
                    if config.prefixkv.rescale_offline_profile_to_byte_budget
                    else config.cache.retention_ratio
                ),
                start_size=config.prefixkv.start_size,
                protect_size=config.prefixkv.protect_size,
            )
            if config.prefixkv.profile_path is not None
            else None
        )
        prefix_plan = build_prefixkv_reimpl_plan(
            full_state,
            prefill.attention_weights,
            config.prefixkv,
            config.cache,
            model_id=config.model.id,
            model_revision=config.model.revision,
            profile=prefix_profile,
        )
        payloads, slot_records = prefixkv_runtime_payloads(prefix_plan)
        packed = pack_runtime_payloads(
            adapter,
            full_state,
            payloads,
            slot_records,
            packing_mode=config.cache.packing_mode,
        )
    elif config.method.is_published_eviction_reimplementation:
        published_plan = build_published_kv_reimpl_plan(
            full_state,
            prefill.attention_weights,
            config.method,
            config.published_kv,
            config.cache,
        )
        payloads, slot_records = published_kv_runtime_payloads(published_plan)
        packed = pack_runtime_payloads(
            adapter,
            full_state,
            payloads,
            slot_records,
            packing_mode=config.cache.packing_mode,
        )
    else:
        vl_cache_plan = build_vl_cache_reimpl_plan(
            full_state,
            prefill.attention_weights,
            config.vl_cache,
            config.cache,
        )
        payloads, slot_records = vl_cache_runtime_payloads(vl_cache_plan)
        packed = pack_runtime_payloads(
            adapter,
            full_state,
            payloads,
            slot_records,
            packing_mode=config.cache.packing_mode,
        )
    reference_state = _clone_decode_state(adapter, prefill.state)
    candidate_state = _state_from_packed(adapter, prefill.state, packed)
    reference_token = prefill.next_token_id
    candidate_token = prefill.next_token_id.detach().clone()
    reference_tokens = [reference_token]
    candidate_tokens = [candidate_token]
    reference_logits = [prefill.logits]
    candidate_logits = [prefill.logits.detach().clone()]
    for _step in range(1, config.generation.max_new_tokens):
        reference = adapter.decode_one_token(
            reference_token,
            reference_state,
            capture_queries=False,
        )
        candidate = adapter.decode_one_token(
            candidate_token,
            candidate_state,
            capture_queries=False,
        )
        reference_state = reference.state
        candidate_state = candidate.state
        reference_token = reference.next_token_id
        candidate_token = candidate.next_token_id
        reference_tokens.append(reference_token)
        candidate_tokens.append(candidate_token)
        reference_logits.append(reference.logits)
        candidate_logits.append(candidate.logits)
    reference_ids = torch.cat(reference_tokens, dim=-1).detach().cpu().reshape(-1)
    candidate_ids = torch.cat(candidate_tokens, dim=-1).detach().cpu().reshape(-1)
    agreement = float((reference_ids == candidate_ids).float().mean().item())
    maximum = max(
        float(torch.max(torch.abs(reference.detach().float() - candidate.detach().float())).item())
        for reference, candidate in zip(reference_logits, candidate_logits, strict=True)
    )
    return ParityReport(
        comparison="unified_hf_runtime_retention_1_vs_fullkv",
        generated_tokens=int(reference_ids.numel()),
        token_agreement=agreement,
        maximum_logit_difference=maximum,
        reference_token_ids=tuple(int(value) for value in reference_ids.tolist()),
        candidate_token_ids=tuple(int(value) for value in candidate_ids.tolist()),
    )


def _axis(axis: int, rank: int) -> int:
    return axis if axis >= 0 else rank + axis


def _one_position(tensor: Any, sequence_axis: int, offset: int) -> Any:
    # A length-one basic slice is a view.  The previous index_select launched
    # one CUDA gather for every selected token before the subsequent pack
    # concatenated those tensors again (hundreds of thousands of launches for
    # LLaVA's 32 KV heads).
    index: list[Any] = [slice(None)] * int(tensor.ndim)
    index[_axis(sequence_axis, int(tensor.ndim))] = slice(offset, offset + 1)
    return tensor[tuple(index)]


def _tier_payloads(
    state: MosaicKVState,
    records: Sequence[PrototypeRecord],
    active_prototype_ids: Sequence[int],
) -> tuple[dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]], tuple[JsonObject, ...]]:
    payloads: dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]] = {}
    trace: list[JsonObject] = []
    for block_id, (block, key, value) in enumerate(
        zip(state.exact.blocks, state.exact.key_blocks, state.exact.value_blocks, strict=True)
    ):
        sequence_axis = _axis(-2, int(key.ndim))
        for offset, logical_position in enumerate(block.original_logical_positions):
            payloads.setdefault((block.layer, block.kv_head), []).append(
                (
                    logical_position,
                    "exact",
                    block_id,
                    _one_position(key, sequence_axis, offset),
                    _one_position(value, sequence_axis, offset),
                )
            )
    record_by_id = {record.prototype_id: record for record in records}
    for payload_offset, prototype_id in enumerate(active_prototype_ids):
        record = record_by_id[prototype_id]
        key = state.prototypes.prototype_keys[payload_offset]
        value = state.prototypes.prototype_values[payload_offset]
        payloads.setdefault((record.layer, record.kv_head), []).append(
            (record.anchor_logical_position, "prototype", prototype_id, key, value)
        )
    for identity in payloads:
        payloads[identity].sort(key=lambda item: (item[0], item[1] != "exact", item[2]))
        for slot, entry in enumerate(payloads[identity]):
            item: JsonObject = {
                "layer": identity[0],
                "kv_head": identity[1],
                "slot": slot,
                "logical_position": entry[0],
                "tier": entry[1],
                "source_id": entry[2],
            }
            if entry[1] == "prototype":
                record = record_by_id[int(entry[2])]
                item["represented_source_positions"] = record.represented_source_positions
                item["attention_log_bias"] = record.attention_log_bias
                item["attention_log_bias_scale"] = record.attention_log_bias_scale
            trace.append(item)
    return payloads, tuple(trace)


def _pack_three_tier_from_source(
    adapter: HuggingFaceMultimodalAdapter,
    full_state: FullKVState,
    state: MosaicKVState,
    records: Sequence[PrototypeRecord],
    active_prototype_ids: Sequence[int],
    *,
    tail_snapshot: CacheSnapshot | None,
    old_prompt_capacity: int | None,
    physical_prompt_capacity: int | None,
) -> PackedRuntimeCache:
    """Fuse exact gathering and prototype insertion once per source layer.

    Exact-tier payloads are unchanged views/copies of ``full_state``.  The
    generic mixed-tier packer expands them into one singleton tensor per
    retained position and concatenates those rows again.  On many-head VLMs
    that creates tens of thousands of Python objects and small CUDA launches.
    This path builds the identical logical slot table on the host, gathers all
    exact positions with one layer operation, then inserts the already-built
    prototypes with one indexed write.  It is deliberately limited to the
    optimized preallocated mode; the scalar packer remains the parity oracle.
    """

    torch = _torch()
    if tail_snapshot is not None and old_prompt_capacity is None:
        raise ValueError("tail packing requires the previous prompt capacity")
    active_ids = tuple(int(prototype_id) for prototype_id in active_prototype_ids)
    if len(active_ids) != len(set(active_ids)):
        raise HuggingFaceRuntimeError("active prototype IDs must be unique")
    if len(active_ids) != len(state.prototypes.prototype_keys):
        raise HuggingFaceRuntimeError(
            "active prototype IDs must align with compact prototype payloads"
        )
    record_by_id = {record.prototype_id: record for record in records}
    unknown = set(active_ids) - set(record_by_id)
    if unknown:
        raise HuggingFaceRuntimeError(
            f"active prototype IDs have no construction record: {sorted(unknown)}"
        )
    prototype_payloads = {
        prototype_id: (
            state.prototypes.prototype_keys[offset],
            state.prototypes.prototype_values[offset],
        )
        for offset, prototype_id in enumerate(active_ids)
    }

    # Entries are (logical position, tier, source ID, physical position).
    # Prototype rows have no source position and use ``None``.
    entries: dict[tuple[int, int], list[tuple[int, str, int, int | None]]] = {}
    for block_id, block in enumerate(state.exact.blocks):
        destination = entries.setdefault((block.layer, block.kv_head), [])
        destination.extend(
            (logical, "exact", block_id, physical)
            for logical, physical in zip(
                block.original_logical_positions,
                block.physical_cache_indices,
                strict=True,
            )
        )
    for prototype_id in active_ids:
        record = record_by_id[prototype_id]
        entries.setdefault((record.layer, record.kv_head), []).append(
            (record.anchor_logical_position, "prototype", prototype_id, None)
        )
    for values in entries.values():
        values.sort(key=lambda item: (item[0], item[1] != "exact", item[2]))

    selected_capacity = max((len(values) for values in entries.values()), default=0)
    if selected_capacity < 1:
        raise HuggingFaceRuntimeError("packed runtime cache has no active prompt slots")
    prompt_capacity = (
        selected_capacity if physical_prompt_capacity is None else physical_prompt_capacity
    )
    if prompt_capacity < selected_capacity:
        raise HuggingFaceRuntimeError(
            "selected cache exceeds the dense physical prompt capacity: "
            f"selected={selected_capacity}, capacity={prompt_capacity}"
        )

    layers: list[CacheLayerSnapshot] = []
    validity_masks: list[Any] = []
    prototype_slots: list[tuple[int, int, int, int]] = []
    prototype_attention_biases: list[tuple[int, int, int, int, float]] = []
    slot_records: list[JsonObject] = []
    finite_flags: list[Any] = []
    all_slots_valid = True

    def pack_tensor(
        source: Any,
        *,
        layer_entries: Sequence[Sequence[tuple[int, str, int, int | None]]],
        payload_index: int,
        head_axis: int,
        sequence_axis: int,
        tail: Any | None,
    ) -> tuple[Any, Any]:
        rank = int(source.ndim)
        normalized_head = _axis(head_axis, rank)
        normalized_sequence = _axis(sequence_axis, rank)
        canonical = source.movedim((normalized_head, normalized_sequence), (0, 1))
        head_count = int(canonical.shape[0])
        if len(layer_entries) != head_count:
            raise HuggingFaceRuntimeError("fused packing entries do not match KV heads")
        counts = tuple(len(values) for values in layer_entries)
        if any(count < 1 or count > prompt_capacity for count in counts):
            raise HuggingFaceRuntimeError(
                "fused packed head is empty or exceeds the physical prompt capacity"
            )
        physical_rows = tuple(
            tuple(0 if item[3] is None else int(item[3]) for item in values)
            + (0,) * (prompt_capacity - len(values))
            for values in layer_entries
        )
        indices = torch.as_tensor(physical_rows, dtype=torch.long, device=source.device)
        expanded = indices.reshape(
            head_count,
            prompt_capacity,
            *([1] * (rank - 2)),
        ).expand(head_count, prompt_capacity, *canonical.shape[2:])
        packed = torch.gather(canonical, 1, expanded)
        count_tensor = torch.as_tensor(counts, dtype=torch.long, device=source.device)
        validity = torch.arange(prompt_capacity, device=source.device).unsqueeze(
            0
        ) < count_tensor.unsqueeze(1)
        expanded_validity = validity.reshape(
            head_count,
            prompt_capacity,
            *([1] * (rank - 2)),
        )
        packed = packed.masked_fill(~expanded_validity, 0)

        prototype_indices: list[tuple[int, int, int]] = []
        for head, values in enumerate(layer_entries):
            prototype_indices.extend(
                (head, slot, int(item[2]))
                for slot, item in enumerate(values)
                if item[1] == "prototype"
            )
        if prototype_indices:
            prototype_rows: list[Any] = []
            for _head, _slot, prototype_id in prototype_indices:
                payload = prototype_payloads[prototype_id][payload_index]
                payload_canonical = payload.movedim(
                    (normalized_head, normalized_sequence),
                    (0, 1),
                )
                if (
                    int(payload_canonical.shape[0]) != 1
                    or int(payload_canonical.shape[1]) != 1
                    or tuple(payload_canonical.shape[2:]) != tuple(canonical.shape[2:])
                ):
                    raise HuggingFaceRuntimeError(
                        "prototype payload shape differs from its source cache layer"
                    )
                prototype_rows.append(payload_canonical[0, 0])
            head_indices = torch.as_tensor(
                tuple(item[0] for item in prototype_indices),
                dtype=torch.long,
                device=source.device,
            )
            slot_indices = torch.as_tensor(
                tuple(item[1] for item in prototype_indices),
                dtype=torch.long,
                device=source.device,
            )
            packed[head_indices, slot_indices] = torch.stack(prototype_rows, dim=0)

        if tail is not None:
            if old_prompt_capacity is None:  # pragma: no cover - checked by caller
                raise ValueError("tail packing requires the previous prompt capacity")
            canonical_tail = tail.movedim(
                (normalized_head, normalized_sequence),
                (0, 1),
            )
            if int(canonical_tail.shape[0]) != head_count:
                raise HuggingFaceRuntimeError("tail cache KV-head count changed during packing")
            tail_length = int(canonical_tail.shape[1]) - old_prompt_capacity
            if tail_length < 0:
                raise HuggingFaceRuntimeError("old prompt capacity exceeds the tail cache")
            if tail_length:
                packed = torch.cat(
                    (packed, canonical_tail[:, old_prompt_capacity:]),
                    dim=1,
                )
                validity = torch.cat(
                    (
                        validity,
                        torch.ones(
                            (head_count, tail_length),
                            dtype=torch.bool,
                            device=source.device,
                        ),
                    ),
                    dim=1,
                )
        return packed.movedim(
            (0, 1),
            (normalized_head, normalized_sequence),
        ), validity

    for layer_index, source_layer in enumerate(full_state.layers):
        layer_entries: list[Sequence[tuple[int, str, int, int | None]]] = []
        for kv_head in range(source_layer.kv_heads):
            values = entries.get((layer_index, kv_head), [])
            if not values:
                raise HuggingFaceRuntimeError(
                    f"packed cache removed every slot for layer={layer_index}, head={kv_head}"
                )
            layer_entries.append(values)
            all_slots_valid = all_slots_valid and len(values) == prompt_capacity
            for slot, item in enumerate(values):
                slot_record: JsonObject = {
                    "layer": layer_index,
                    "kv_head": kv_head,
                    "slot": slot,
                    "logical_position": item[0],
                    "tier": item[1],
                    "source_id": item[2],
                }
                if item[1] == "prototype":
                    prototype_id = int(item[2])
                    prototype = record_by_id[prototype_id]
                    slot_record["represented_source_positions"] = (
                        prototype.represented_source_positions
                    )
                    slot_record["attention_log_bias"] = prototype.attention_log_bias
                    slot_record["attention_log_bias_scale"] = prototype.attention_log_bias_scale
                    prototype_slots.append((prototype_id, layer_index, kv_head, slot))
                    bias = float(prototype.attention_log_bias)
                    if not math.isfinite(bias) or bias < 0:
                        raise HuggingFaceRuntimeError(
                            "prototype attention log bias must be finite and nonnegative"
                        )
                    if bias > 0:
                        prototype_attention_biases.append(
                            (prototype_id, layer_index, kv_head, slot, bias)
                        )
                slot_records.append(slot_record)
        tail_layer = None if tail_snapshot is None else tail_snapshot.layers[layer_index]
        packed_key, validity = pack_tensor(
            source_layer.key,
            layer_entries=tuple(layer_entries),
            payload_index=0,
            head_axis=source_layer.key_head_dimension,
            sequence_axis=source_layer.key_sequence_dimension,
            tail=None if tail_layer is None else tail_layer.key,
        )
        packed_value, value_validity = pack_tensor(
            source_layer.value,
            layer_entries=tuple(layer_entries),
            payload_index=1,
            head_axis=source_layer.value_head_dimension,
            sequence_axis=source_layer.value_sequence_dimension,
            tail=None if tail_layer is None else tail_layer.value,
        )
        if not torch.equal(validity, value_validity):
            raise HuggingFaceRuntimeError("fused K/V packing produced different validity masks")
        finite_flags.extend((torch.isfinite(packed_key).all(), torch.isfinite(packed_value).all()))
        layers.append(
            CacheLayerSnapshot(
                packed_key,
                packed_value,
                source_layer.key_sequence_dimension,
            )
        )
        validity_masks.append(validity)

    if not bool(torch.stack(finite_flags).all().item()):
        raise HuggingFaceRuntimeError("packed cache contains NaN or infinity")
    active_length = int(layers[0].key.shape[full_state.layers[0].key_sequence_dimension])
    snapshot = CacheSnapshot(
        tuple(layers),
        full_state.source_class,
        full_state.source_kind,
        active_length,
        adapter.capabilities.cached_key_state,
    )
    return PackedRuntimeCache(
        snapshot,
        tuple(validity_masks),
        prompt_capacity,
        all_slots_valid,
        tuple(sorted(prototype_slots)),
        tuple(sorted(prototype_attention_biases)),
        tuple(slot_records),
    )


def pack_runtime_payloads(
    adapter: HuggingFaceMultimodalAdapter,
    full_state: FullKVState,
    payloads: Mapping[
        tuple[int, int],
        Sequence[tuple[int, str, int, Any, Any]],
    ],
    slot_records: tuple[JsonObject, ...],
    *,
    tail_snapshot: CacheSnapshot | None = None,
    old_prompt_capacity: int | None = None,
    physical_prompt_capacity: int | None = None,
    prototype_log_bias_by_id: Mapping[int, float] | None = None,
    packing_mode: str = "per_head_reference",
) -> PackedRuntimeCache:
    """Pack algorithm-owned head payloads through the common HF cache interface."""

    torch = _torch()
    if packing_mode not in {
        "per_head_reference",
        "batched_layer",
        "preallocated_layer",
    }:
        raise ValueError(
            "packing_mode must be 'per_head_reference', 'batched_layer', or 'preallocated_layer'"
        )
    selected_capacity = max((len(items) for items in payloads.values()), default=0)
    if selected_capacity < 1:
        raise HuggingFaceRuntimeError("packed runtime cache has no active prompt slots")
    prompt_capacity = (
        selected_capacity if physical_prompt_capacity is None else physical_prompt_capacity
    )
    if prompt_capacity < selected_capacity:
        raise HuggingFaceRuntimeError(
            "selected cache exceeds the dense physical prompt capacity: "
            f"selected={selected_capacity}, capacity={prompt_capacity}"
        )
    if tail_snapshot is not None and old_prompt_capacity is None:
        raise ValueError("tail packing requires the previous prompt capacity")
    layers: list[CacheLayerSnapshot] = []
    validity_masks: list[Any] = []
    all_slots_valid = True
    prototype_slots: list[tuple[int, int, int, int]] = []
    prototype_attention_biases: list[tuple[int, int, int, int, float]] = []
    finite_flags: list[Any] = []
    for layer_index, source_layer in enumerate(full_state.layers):
        if packing_mode in {"batched_layer", "preallocated_layer"}:
            entries_by_head: list[Sequence[tuple[int, str, int, Any, Any]]] = []
            for kv_head in range(source_layer.kv_heads):
                entries = payloads.get((layer_index, kv_head), [])
                if not entries:
                    raise HuggingFaceRuntimeError(
                        f"packed cache removed every slot for layer={layer_index}, head={kv_head}"
                    )
                entries_by_head.append(entries)
                all_slots_valid = all_slots_valid and len(entries) == prompt_capacity
                for slot, entry in enumerate(entries):
                    if entry[1] == "prototype":
                        prototype_id = int(entry[2])
                        prototype_slots.append((prototype_id, layer_index, kv_head, slot))
                        bias = float((prototype_log_bias_by_id or {}).get(prototype_id, 0.0))
                        if not math.isfinite(bias) or bias < 0:
                            raise HuggingFaceRuntimeError(
                                "prototype attention log bias must be finite and nonnegative"
                            )
                        if bias > 0:
                            prototype_attention_biases.append(
                                (prototype_id, layer_index, kv_head, slot, bias)
                            )
            tail_layer = None if tail_snapshot is None else tail_snapshot.layers[layer_index]
            pack_layer = (
                _pack_preallocated_payload_heads
                if packing_mode == "preallocated_layer"
                else _pack_mixed_payload_heads
            )
            packed_key, validity = pack_layer(
                tuple(entries_by_head),
                payload_index=3,
                head_axis=source_layer.key_head_dimension,
                sequence_axis=source_layer.key_sequence_dimension,
                prompt_capacity=prompt_capacity,
                tail_tensor=None if tail_layer is None else tail_layer.key,
                old_prompt_capacity=old_prompt_capacity,
            )
            packed_value, _value_validity = pack_layer(
                tuple(entries_by_head),
                payload_index=4,
                head_axis=source_layer.value_head_dimension,
                sequence_axis=source_layer.value_sequence_dimension,
                prompt_capacity=prompt_capacity,
                tail_tensor=None if tail_layer is None else tail_layer.value,
                old_prompt_capacity=old_prompt_capacity,
            )
            finite_flags.extend(
                (torch.isfinite(packed_key).all(), torch.isfinite(packed_value).all())
            )
            layers.append(
                CacheLayerSnapshot(
                    packed_key,
                    packed_value,
                    source_layer.key_sequence_dimension,
                )
            )
            validity_masks.append(validity)
            continue
        head_keys: list[Any] = []
        head_values: list[Any] = []
        head_validity: list[Any] = []
        key_sequence = source_layer.key_sequence_dimension
        value_sequence = source_layer.value_sequence_dimension
        for kv_head in range(source_layer.kv_heads):
            entries = payloads.get((layer_index, kv_head), [])
            if not entries:
                raise HuggingFaceRuntimeError(
                    f"packed cache removed every slot for layer={layer_index}, head={kv_head}"
                )
            key = torch.cat(tuple(entry[3] for entry in entries), dim=key_sequence)
            value = torch.cat(tuple(entry[4] for entry in entries), dim=value_sequence)
            valid_count = len(entries)
            if valid_count < prompt_capacity:
                all_slots_valid = False
                key_shape = list(key.shape)
                key_shape[key_sequence] = prompt_capacity - valid_count
                value_shape = list(value.shape)
                value_shape[value_sequence] = prompt_capacity - valid_count
                key = torch.cat(
                    (key, torch.zeros(key_shape, dtype=key.dtype, device=key.device)),
                    dim=key_sequence,
                )
                value = torch.cat(
                    (value, torch.zeros(value_shape, dtype=value.dtype, device=value.device)),
                    dim=value_sequence,
                )
            if tail_snapshot is not None:
                tail_layer = tail_snapshot.layers[layer_index]
                tail_key_index: list[Any] = [slice(None)] * int(tail_layer.key.ndim)
                tail_value_index: list[Any] = [slice(None)] * int(tail_layer.value.ndim)
                tail_key_index[source_layer.key_head_dimension] = slice(kv_head, kv_head + 1)
                tail_value_index[source_layer.value_head_dimension] = slice(kv_head, kv_head + 1)
                tail_key_index[key_sequence] = slice(old_prompt_capacity, None)
                tail_value_index[value_sequence] = slice(old_prompt_capacity, None)
                key = torch.cat((key, tail_layer.key[tuple(tail_key_index)]), dim=key_sequence)
                value = torch.cat(
                    (value, tail_layer.value[tuple(tail_value_index)]), dim=value_sequence
                )
            tail_length = int(key.shape[key_sequence]) - prompt_capacity
            head_keys.append(key)
            head_values.append(value)
            head_validity.append(
                torch.tensor(
                    [True] * valid_count
                    + [False] * (prompt_capacity - valid_count)
                    + [True] * tail_length,
                    dtype=torch.bool,
                    device=key.device,
                )
            )
            for slot, entry in enumerate(entries):
                if entry[1] == "prototype":
                    prototype_id = int(entry[2])
                    prototype_slots.append((prototype_id, layer_index, kv_head, slot))
                    bias = float((prototype_log_bias_by_id or {}).get(prototype_id, 0.0))
                    if not math.isfinite(bias) or bias < 0:
                        raise HuggingFaceRuntimeError(
                            "prototype attention log bias must be finite and nonnegative"
                        )
                    if bias > 0:
                        prototype_attention_biases.append(
                            (prototype_id, layer_index, kv_head, slot, bias)
                        )
        packed_key = torch.cat(tuple(head_keys), dim=source_layer.key_head_dimension)
        packed_value = torch.cat(tuple(head_values), dim=source_layer.value_head_dimension)
        finite_flags.extend((torch.isfinite(packed_key).all(), torch.isfinite(packed_value).all()))
        layers.append(CacheLayerSnapshot(packed_key, packed_value, key_sequence))
        validity_masks.append(torch.stack(head_validity, dim=0))
    # Reduce all device-side validation in one host synchronization.  A sync
    # for K and V in every layer serialized dozens of tiny packing kernels.
    if not bool(torch.stack(finite_flags).all().item()):
        raise HuggingFaceRuntimeError("packed cache contains NaN or infinity")
    active_length = int(layers[0].key.shape[full_state.layers[0].key_sequence_dimension])
    if any(
        int(layer.key.shape[full_state.layers[index].key_sequence_dimension]) != active_length
        for index, layer in enumerate(layers)
    ):
        raise HuggingFaceRuntimeError("packed cache layers have inconsistent physical lengths")
    snapshot = CacheSnapshot(
        tuple(layers),
        full_state.source_class,
        full_state.source_kind,
        active_length,
        adapter.capabilities.cached_key_state,
    )
    return PackedRuntimeCache(
        snapshot,
        tuple(validity_masks),
        prompt_capacity,
        all_slots_valid,
        tuple(sorted(prototype_slots)),
        tuple(sorted(prototype_attention_biases)),
        slot_records,
    )


def _pack_mixed_payload_heads(
    entries_by_head: Sequence[Sequence[tuple[int, str, int, Any, Any]]],
    *,
    payload_index: int,
    head_axis: int,
    sequence_axis: int,
    prompt_capacity: int,
    tail_tensor: Any | None,
    old_prompt_capacity: int | None,
) -> tuple[Any, Any]:
    """Pack exact/prototype slots with one scatter and optional tail append.

    Every runtime payload represents one KV head and one sequence position.
    Canonicalizing those singleton axes lets us concatenate the payload rows
    once per layer, then scatter them into the padded ``[head, slot, ...]``
    storage.  This replaces per-head concatenation, padding, and final head
    concatenation while preserving the reference slot order byte-for-byte.
    """

    torch = _torch()
    if not entries_by_head or any(not entries for entries in entries_by_head):
        raise HuggingFaceRuntimeError("mixed cache packing requires every KV head")
    first: Any = entries_by_head[0][0][payload_index]
    rank = int(first.ndim)
    normalized_head = _axis(head_axis, rank)
    normalized_sequence = _axis(sequence_axis, rank)
    if normalized_head == normalized_sequence:
        raise HuggingFaceRuntimeError("packed cache head and sequence axes must differ")
    canonical_rows: list[Any] = []
    raw_counts: list[int] = []
    expected_feature_shape = tuple(
        first.movedim((normalized_head, normalized_sequence), (0, 1)).shape[2:]
    )
    for entries in entries_by_head:
        raw_counts.append(len(entries))
        for entry in entries:
            payload: Any = entry[payload_index]
            canonical = payload.movedim(
                (normalized_head, normalized_sequence),
                (0, 1),
            )
            if int(canonical.shape[0]) != 1 or int(canonical.shape[1]) != 1:
                raise HuggingFaceRuntimeError(
                    "mixed runtime payloads must contain one KV head and one position"
                )
            if tuple(canonical.shape[2:]) != expected_feature_shape:
                raise HuggingFaceRuntimeError("mixed runtime payload feature shapes differ")
            canonical_rows.append(canonical)
    if any(count > prompt_capacity for count in raw_counts):
        raise HuggingFaceRuntimeError("selected positions exceed the physical prompt capacity")
    rows = torch.cat(tuple(canonical_rows), dim=1).squeeze(0)
    device = rows.device
    head_indices = torch.tensor(
        tuple(head for head, count in enumerate(raw_counts) for _slot in range(count)),
        dtype=torch.long,
        device=device,
    )
    slot_indices = torch.tensor(
        tuple(slot for count in raw_counts for slot in range(count)),
        dtype=torch.long,
        device=device,
    )
    packed = rows.new_zeros((len(raw_counts), prompt_capacity, *rows.shape[1:]))
    packed[head_indices, slot_indices] = rows
    counts = torch.tensor(raw_counts, dtype=torch.long, device=device)
    validity = torch.arange(prompt_capacity, device=device).unsqueeze(0) < counts.unsqueeze(1)
    if tail_tensor is not None:
        if old_prompt_capacity is None:
            raise HuggingFaceRuntimeError("tail packing requires the old prompt capacity")
        tail = tail_tensor.movedim(
            (normalized_head, normalized_sequence),
            (0, 1),
        )
        if int(tail.shape[0]) != len(raw_counts):
            raise HuggingFaceRuntimeError("tail cache KV-head count changed during repair")
        tail_length = int(tail.shape[1]) - old_prompt_capacity
        if tail_length < 0:
            raise HuggingFaceRuntimeError("old prompt capacity exceeds the tail cache")
        packed = torch.cat((packed, tail[:, old_prompt_capacity:, ...]), dim=1)
        if tail_length:
            validity = torch.cat(
                (
                    validity,
                    torch.ones(
                        (len(raw_counts), tail_length),
                        dtype=torch.bool,
                        device=device,
                    ),
                ),
                dim=1,
            )
    return packed.movedim(
        (0, 1),
        (normalized_head, normalized_sequence),
    ), validity


def _pack_preallocated_payload_heads(
    entries_by_head: Sequence[Sequence[tuple[int, str, int, Any, Any]]],
    *,
    payload_index: int,
    head_axis: int,
    sequence_axis: int,
    prompt_capacity: int,
    tail_tensor: Any | None,
    old_prompt_capacity: int | None,
) -> tuple[Any, Any]:
    """Pack a layer into one preallocated tensor without padded concatenations.

    The reference path constructs and pads one tensor per KV head, then
    concatenates every head. The older batched path performs a large indexed
    scatter. This path allocates the final layer tensor once and copies each
    head's already ordered rows into its contiguous slice. Equal head counts
    therefore produce an all-valid dense cache with no decode-time mask hook.
    """

    torch = _torch()
    if not entries_by_head or any(not entries for entries in entries_by_head):
        raise HuggingFaceRuntimeError("preallocated packing requires every KV head")
    first: Any = entries_by_head[0][0][payload_index]
    rank = int(first.ndim)
    normalized_head = _axis(head_axis, rank)
    normalized_sequence = _axis(sequence_axis, rank)
    if normalized_head == normalized_sequence:
        raise HuggingFaceRuntimeError("packed cache head and sequence axes must differ")
    canonical_first = first.movedim(
        (normalized_head, normalized_sequence),
        (0, 1),
    )
    if int(canonical_first.shape[0]) != 1 or int(canonical_first.shape[1]) != 1:
        raise HuggingFaceRuntimeError(
            "preallocated runtime payloads must contain one KV head and one position"
        )
    feature_shape = tuple(int(size) for size in canonical_first.shape[2:])
    counts = tuple(len(entries) for entries in entries_by_head)
    if any(count > prompt_capacity for count in counts):
        raise HuggingFaceRuntimeError("selected positions exceed the physical prompt capacity")
    tail_length = 0
    canonical_tail = None
    if tail_tensor is not None:
        if old_prompt_capacity is None:
            raise HuggingFaceRuntimeError("tail packing requires the old prompt capacity")
        canonical_tail = tail_tensor.movedim(
            (normalized_head, normalized_sequence),
            (0, 1),
        )
        tail_length = int(canonical_tail.shape[1]) - old_prompt_capacity
        if tail_length < 0:
            raise HuggingFaceRuntimeError("old prompt capacity exceeds the tail cache")
        if int(canonical_tail.shape[0]) != len(entries_by_head):
            raise HuggingFaceRuntimeError("tail cache KV-head count changed during packing")
        if tuple(int(size) for size in canonical_tail.shape[2:]) != feature_shape:
            raise HuggingFaceRuntimeError("tail cache feature shape changed during packing")
    packed = canonical_first.new_zeros(
        (len(entries_by_head), prompt_capacity + tail_length, *feature_shape)
    )
    for head, entries in enumerate(entries_by_head):
        canonical_entries: list[Any] = []
        for entry in entries:
            payload: Any = entry[payload_index]
            canonical_entries.append(
                payload.movedim(
                    (normalized_head, normalized_sequence),
                    (0, 1),
                )
            )
        rows = torch.cat(tuple(canonical_entries), dim=1).squeeze(0)
        if tuple(int(size) for size in rows.shape[1:]) != feature_shape:
            raise HuggingFaceRuntimeError("mixed runtime payload feature shapes differ")
        packed[head, : len(entries)].copy_(rows)
    if canonical_tail is not None and tail_length:
        packed[:, prompt_capacity:].copy_(canonical_tail[:, old_prompt_capacity:])
    count_tensor = torch.as_tensor(counts, dtype=torch.long, device=packed.device)
    validity = torch.arange(
        prompt_capacity,
        dtype=torch.long,
        device=packed.device,
    ).unsqueeze(0) < count_tensor.unsqueeze(1)
    if tail_length:
        validity = torch.cat(
            (
                validity,
                torch.ones(
                    (len(entries_by_head), tail_length),
                    dtype=torch.bool,
                    device=packed.device,
                ),
            ),
            dim=1,
        )
    return packed.movedim(
        (0, 1),
        (normalized_head, normalized_sequence),
    ), validity


def _gather_dense_heads(
    tensor: Any,
    *,
    head_axis: int,
    sequence_axis: int,
    positions: Sequence[Sequence[int]],
    prompt_capacity: int,
    tail_tensor: Any | None,
    old_prompt_capacity: int | None,
) -> tuple[Any, Any]:
    """Gather different position rows for every KV head in one CUDA operation."""

    torch = _torch()
    rank = int(tensor.ndim)
    normalized_head = _axis(head_axis, rank)
    normalized_sequence = _axis(sequence_axis, rank)
    moved = tensor.movedim((normalized_head, normalized_sequence), (0, 1))
    head_count = int(moved.shape[0])
    if len(positions) != head_count:
        raise HuggingFaceRuntimeError("dense gather position rows do not match KV heads")
    raw_counts = [len(row) for row in positions]
    if any(count < 1 for count in raw_counts):
        raise HuggingFaceRuntimeError("packed cache removed every slot from a KV head")
    if any(count > prompt_capacity for count in raw_counts):
        raise HuggingFaceRuntimeError("selected positions exceed the physical prompt capacity")
    counts = torch.tensor(raw_counts, dtype=torch.long, device=tensor.device)
    index_rows = [tuple(row) + (0,) * (prompt_capacity - len(row)) for row in positions]
    indices = torch.tensor(index_rows, dtype=torch.long, device=tensor.device)
    expanded_indices = indices.reshape(head_count, prompt_capacity, *([1] * (rank - 2))).expand(
        head_count, prompt_capacity, *moved.shape[2:]
    )
    packed = torch.gather(moved, 1, expanded_indices)
    validity = torch.arange(prompt_capacity, device=tensor.device).unsqueeze(0) < counts.unsqueeze(
        1
    )
    expanded_validity = validity.reshape(head_count, prompt_capacity, *([1] * (rank - 2)))
    packed = packed.masked_fill(~expanded_validity, 0)
    if tail_tensor is not None:
        if old_prompt_capacity is None:
            raise HuggingFaceRuntimeError("tail packing requires the old prompt capacity")
        tail = tail_tensor.movedim((normalized_head, normalized_sequence), (0, 1))
        packed = torch.cat((packed, tail[:, old_prompt_capacity:, ...]), dim=1)
        tail_length = int(tail.shape[1]) - old_prompt_capacity
        if tail_length < 0:
            raise HuggingFaceRuntimeError("old prompt capacity exceeds the tail cache")
        if tail_length:
            validity = torch.cat(
                (
                    validity,
                    torch.ones((head_count, tail_length), dtype=torch.bool, device=tensor.device),
                ),
                dim=1,
            )
    return packed.movedim((0, 1), (normalized_head, normalized_sequence)), validity


def _pack_exact_runtime_blocks(
    adapter: HuggingFaceMultimodalAdapter,
    full_state: FullKVState,
    blocks: Sequence[KVBlockDescriptor],
    *,
    tail_snapshot: CacheSnapshot | None,
    old_prompt_capacity: int | None,
    physical_prompt_capacity: int | None,
) -> PackedRuntimeCache:
    """Vectorized materialization for exact-only selectors and baselines."""

    torch = _torch()
    entries: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for block_id, block in enumerate(blocks):
        for logical, physical in zip(
            block.original_logical_positions,
            block.physical_cache_indices,
            strict=True,
        ):
            entries.setdefault((block.layer, block.kv_head), []).append(
                (logical, physical, block_id)
            )
    for values in entries.values():
        values.sort(key=lambda item: (item[0], item[2]))
    selected_capacity = max((len(values) for values in entries.values()), default=0)
    if selected_capacity < 1:
        raise HuggingFaceRuntimeError("packed runtime cache has no active prompt slots")
    prompt_capacity = (
        selected_capacity if physical_prompt_capacity is None else physical_prompt_capacity
    )
    if prompt_capacity < selected_capacity:
        raise HuggingFaceRuntimeError("selected cache exceeds its dense physical prompt capacity")

    layers: list[CacheLayerSnapshot] = []
    validity_masks: list[Any] = []
    finite_flags: list[Any] = []
    slot_records: list[JsonObject] = []
    all_slots_valid = True
    for layer_index, source_layer in enumerate(full_state.layers):
        position_rows: list[tuple[int, ...]] = []
        for kv_head in range(source_layer.kv_heads):
            row = entries.get((layer_index, kv_head), [])
            if not row:
                raise HuggingFaceRuntimeError(
                    f"packed cache removed every slot for layer={layer_index}, head={kv_head}"
                )
            position_rows.append(tuple(item[1] for item in row))
            all_slots_valid = all_slots_valid and len(row) == prompt_capacity
            slot_records.extend(
                {
                    "layer": layer_index,
                    "kv_head": kv_head,
                    "slot": slot,
                    "logical_position": item[0],
                    "tier": "exact",
                    "source_id": item[2],
                }
                for slot, item in enumerate(row)
            )
        tail_layer = None if tail_snapshot is None else tail_snapshot.layers[layer_index]
        packed_key, validity = _gather_dense_heads(
            source_layer.key,
            head_axis=source_layer.key_head_dimension,
            sequence_axis=source_layer.key_sequence_dimension,
            positions=position_rows,
            prompt_capacity=prompt_capacity,
            tail_tensor=None if tail_layer is None else tail_layer.key,
            old_prompt_capacity=old_prompt_capacity,
        )
        packed_value, _value_validity = _gather_dense_heads(
            source_layer.value,
            head_axis=source_layer.value_head_dimension,
            sequence_axis=source_layer.value_sequence_dimension,
            positions=position_rows,
            prompt_capacity=prompt_capacity,
            tail_tensor=None if tail_layer is None else tail_layer.value,
            old_prompt_capacity=old_prompt_capacity,
        )
        finite_flags.extend((torch.isfinite(packed_key).all(), torch.isfinite(packed_value).all()))
        layers.append(
            CacheLayerSnapshot(
                packed_key,
                packed_value,
                source_layer.key_sequence_dimension,
            )
        )
        validity_masks.append(validity)
    if not bool(torch.stack(finite_flags).all().item()):
        raise HuggingFaceRuntimeError("packed cache contains NaN or infinity")
    active_length = int(layers[0].key.shape[full_state.layers[0].key_sequence_dimension])
    if any(
        int(layer.key.shape[full_state.layers[index].key_sequence_dimension]) != active_length
        for index, layer in enumerate(layers)
    ):
        raise HuggingFaceRuntimeError("packed cache layers have inconsistent physical lengths")
    return PackedRuntimeCache(
        CacheSnapshot(
            tuple(layers),
            full_state.source_class,
            full_state.source_kind,
            active_length,
            adapter.capabilities.cached_key_state,
        ),
        tuple(validity_masks),
        prompt_capacity,
        all_slots_valid,
        (),
        (),
        tuple(slot_records),
    )


def pack_runtime_cache(
    adapter: HuggingFaceMultimodalAdapter,
    full_state: FullKVState,
    state: MosaicKVState,
    records: Sequence[PrototypeRecord],
    active_prototype_ids: Sequence[int],
    *,
    tail_snapshot: CacheSnapshot | None = None,
    old_prompt_capacity: int | None = None,
    physical_prompt_capacity: int | None = None,
    packing_mode: str = "per_head_reference",
) -> PackedRuntimeCache:
    """Pack MosaicKV/simple-baseline tiers through the common HF interface."""

    key_state = str(getattr(full_state.cached_key_state, "value", full_state.cached_key_state))
    if state.prototypes.prototype_keys and key_state == "pre_rope":
        raise HuggingFaceRuntimeError(
            "HF prototype materialization requires an adapter-specific safe RoPE transform"
        )
    if not state.prototypes.prototype_keys and not active_prototype_ids:
        return _pack_exact_runtime_blocks(
            adapter,
            full_state,
            state.exact.blocks,
            tail_snapshot=tail_snapshot,
            old_prompt_capacity=old_prompt_capacity,
            physical_prompt_capacity=physical_prompt_capacity,
        )
    if packing_mode == "preallocated_layer":
        return _pack_three_tier_from_source(
            adapter,
            full_state,
            state,
            records,
            active_prototype_ids,
            tail_snapshot=tail_snapshot,
            old_prompt_capacity=old_prompt_capacity,
            physical_prompt_capacity=physical_prompt_capacity,
        )
    payloads, slot_records = _tier_payloads(state, records, active_prototype_ids)
    record_by_id = {record.prototype_id: record for record in records}
    return pack_runtime_payloads(
        adapter,
        full_state,
        payloads,
        slot_records,
        tail_snapshot=tail_snapshot,
        old_prompt_capacity=old_prompt_capacity,
        physical_prompt_capacity=physical_prompt_capacity,
        prototype_log_bias_by_id={
            prototype_id: record_by_id[prototype_id].attention_log_bias
            for prototype_id in active_prototype_ids
        },
        packing_mode=packing_mode,
    )


def _pack_prefixkv_ragged(
    adapter: HuggingFaceMultimodalAdapter,
    plan: PrefixKVCompressionPlan,
    original_cache: Any,
) -> PackedRuntimeCache:
    from mosaickv.adapters.huggingface.prefixkv_cache import build_ragged_snapshot

    snapshot = build_ragged_snapshot(adapter, plan, original_cache)
    return PackedRuntimeCache(
        snapshot=snapshot,
        validity_masks=(),
        prompt_capacity=max(plan.layer_cache_sizes),
        all_slots_valid=True,
        prototype_slots=(),
        prototype_attention_biases=(),
        slot_records=tuple(
            {
                "layer": layer.layer,
                "selected_logical_positions": list(layer.selected_logical_positions),
                "retained_positions": layer.retained_positions,
                "retained_bytes": layer.retained_bytes,
                "shared_across_kv_heads": True,
                "padding_bytes": 0,
            }
            for layer in plan.layers
        ),
        layerwise_logical_positions=tuple(
            layer.selected_logical_positions for layer in plan.layers
        ),
    )


def _state_from_packed(
    adapter: HuggingFaceMultimodalAdapter,
    original: DecodeState,
    packed: PackedRuntimeCache,
) -> DecodeState:
    torch = _torch()
    device = packed.snapshot.layers[0].key.device
    model_state = _packed_model_state(original, packed)
    return DecodeState(
        past_key_values=adapter.inject_past_key_values(packed.snapshot),
        attention_mask=torch.ones(
            (1, packed.snapshot.active_sequence_length),
            dtype=original.attention_mask.dtype,
            device=device,
        ),
        active_cache_length=packed.snapshot.active_sequence_length,
        logical_sequence_length=original.logical_sequence_length,
        next_decode_position=original.next_decode_position,
        modality_map=original.modality_map,
        model_state=model_state,
    )


def _packed_model_state(original: DecodeState, packed: PackedRuntimeCache) -> dict[str, Any]:
    model_state = dict(original.model_state)
    for stale_key in (
        "mosaickv_validity_buffers",
        "mosaickv_additive_mask_buffers",
        "mosaickv_mask_capacity",
    ):
        model_state.pop(stale_key, None)
    model_state.update(
        {
            "mosaickv_prompt_capacity": packed.prompt_capacity,
            "mosaickv_prototype_slots": packed.prototype_slots,
            "mosaickv_packed_all_valid": packed.all_slots_valid,
            "mosaickv_cache_all_valid": packed.all_slots_valid,
            "mosaickv_prototype_attention_biases": packed.prototype_attention_biases,
        }
    )
    if not packed.all_slots_valid or packed.prototype_attention_biases:
        model_state["mosaickv_validity_masks"] = packed.validity_masks
    else:
        model_state.pop("mosaickv_validity_masks", None)
    if packed.layerwise_logical_positions:
        model_state["prefixkv_ragged"] = True
        model_state["prefixkv_logical_positions"] = packed.layerwise_logical_positions
    return model_state


def construct_isolated_packed_decode_state(
    adapter: HuggingFaceMultimodalAdapter,
    reference: DecodeState,
    packed: PackedRuntimeCache,
    *,
    candidate_id: str,
) -> IsolatedDecodeStateHandle:
    """Own one disposable packed HF candidate and guard its FullKV reference."""

    torch = _torch()
    device = packed.snapshot.layers[0].key.device
    return construct_isolated_decode_state(
        adapter,
        reference,
        candidate_id=candidate_id,
        cache_snapshot=packed.snapshot,
        attention_mask=torch.ones(
            (1, packed.snapshot.active_sequence_length),
            dtype=reference.attention_mask.dtype,
            device=device,
        ),
        active_cache_length=packed.snapshot.active_sequence_length,
        model_state=_packed_model_state(reference, packed),
    )


def maintain_prefixkv_decode_cache(
    adapter: HuggingFaceMultimodalAdapter,
    state: DecodeState,
    plan: PrefixKVCompressionPlan,
    *,
    step_index: int,
) -> tuple[DecodeState, tuple[JsonObject, ...]]:
    """Apply PrefixKV's at-most-one fixed-offset eviction per layer after a decode."""

    if state.model_state.get("prefixkv_ragged") is True:
        from mosaickv.adapters.huggingface.prefixkv_cache import evict_ragged_prefixkv

        return evict_ragged_prefixkv(adapter, state, plan, step_index=step_index)

    if plan.config.decode_policy != "realized_ratio":
        raise HuggingFaceRuntimeError("new PrefixKV decode policies require a ragged cache state")

    torch = _torch()
    snapshot = adapter.extract_past_key_values(state.past_key_values)
    raw_masks = state.model_state.get("mosaickv_validity_masks")
    if raw_masks is None and state.model_state.get("mosaickv_packed_all_valid") is True:
        raw_masks = tuple(
            torch.ones(
                (
                    plan.full_state.layers[layer_plan.layer].kv_heads,
                    int(
                        snapshot.layers[layer_plan.layer].key.shape[
                            plan.full_state.layers[layer_plan.layer].key_sequence_dimension
                        ]
                    ),
                ),
                dtype=torch.bool,
                device=snapshot.layers[layer_plan.layer].key.device,
            )
            for layer_plan in plan.layers
        )
    if not isinstance(raw_masks, tuple) or len(raw_masks) != len(plan.layers):
        raise HuggingFaceRuntimeError("PrefixKV decode validity metadata is incomplete")
    keep_by_layer: list[tuple[int, ...]] = []
    events: list[JsonObject] = []
    for layer_plan, validity in zip(plan.layers, raw_masks, strict=True):
        if getattr(validity, "ndim", 0) != 2 or getattr(validity, "dtype", None) != torch.bool:
            raise HuggingFaceRuntimeError("PrefixKV validity masks must be rank-two bool tensors")
        if int(validity.shape[0]) != plan.full_state.layers[layer_plan.layer].kv_heads:
            raise HuggingFaceRuntimeError("PrefixKV validity KV-head count changed during decode")
        if not bool(torch.all(validity == validity[0:1])):
            raise HuggingFaceRuntimeError(
                "PrefixKV requires the same selected positions in every KV head"
            )
        valid_physical = tuple(
            int(value)
            for value in torch.nonzero(validity[0], as_tuple=False)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )
        target = state.logical_sequence_length * layer_plan.retention_ratio
        # Official PrefixKV casts the difference to int32 and removes at most
        # one position per decode call.
        should_evict = math.trunc(len(valid_physical) - target) > 0
        removed: int | None = None
        if should_evict and len(valid_physical) > 1:
            offset = min(layer_plan.eviction_offset, len(valid_physical) - 1)
            removed = valid_physical[offset]
            valid_physical = tuple(
                position for index, position in enumerate(valid_physical) if index != offset
            )
        keep_by_layer.append(valid_physical)
        events.append(
            {
                "step": step_index,
                "layer": layer_plan.layer,
                "logical_sequence_length": state.logical_sequence_length,
                "target_active_positions": target,
                "eviction_offset": layer_plan.eviction_offset,
                "removed_physical_position": removed,
                "active_positions_after": len(valid_physical),
            }
        )
    if not any(event["removed_physical_position"] is not None for event in events):
        return state, tuple(events)

    capacity = max(len(positions) for positions in keep_by_layer)
    layers: list[CacheLayerSnapshot] = []
    masks: list[Any] = []
    for layer_index, (source, positions) in enumerate(
        zip(snapshot.layers, keep_by_layer, strict=True)
    ):
        storage = plan.full_state.layers[layer_index]
        key_index = torch.tensor(positions, dtype=torch.long, device=source.key.device)
        value_index = key_index.to(device=source.value.device)
        key = torch.index_select(source.key, storage.key_sequence_dimension, key_index)
        value = torch.index_select(source.value, storage.value_sequence_dimension, value_index)
        padding = capacity - len(positions)
        if padding:
            key_shape = list(key.shape)
            value_shape = list(value.shape)
            key_shape[storage.key_sequence_dimension] = padding
            value_shape[storage.value_sequence_dimension] = padding
            key = torch.cat(
                (key, torch.zeros(key_shape, dtype=key.dtype, device=key.device)),
                dim=storage.key_sequence_dimension,
            )
            value = torch.cat(
                (value, torch.zeros(value_shape, dtype=value.dtype, device=value.device)),
                dim=storage.value_sequence_dimension,
            )
        layers.append(CacheLayerSnapshot(key, value, storage.key_sequence_dimension))
        masks.append(
            torch.tensor(
                [True] * len(positions) + [False] * padding,
                dtype=torch.bool,
                device=key.device,
            )
            .unsqueeze(0)
            .expand(storage.kv_heads, -1)
            .clone()
        )
    compacted = CacheSnapshot(
        tuple(layers),
        snapshot.source_class,
        snapshot.source_kind,
        capacity,
        snapshot.cached_key_state,
    )
    model_state = dict(state.model_state)
    model_state["mosaickv_validity_masks"] = tuple(masks)
    model_state["mosaickv_prompt_capacity"] = capacity
    return (
        DecodeState(
            past_key_values=adapter.inject_past_key_values(compacted),
            attention_mask=torch.ones(
                (1, capacity),
                dtype=state.attention_mask.dtype,
                device=state.attention_mask.device,
            ),
            active_cache_length=capacity,
            logical_sequence_length=state.logical_sequence_length,
            next_decode_position=state.next_decode_position,
            modality_map=state.modality_map,
            model_state=model_state,
        ),
        tuple(events),
    )


def maintain_elastic_cache_decode_cache(
    adapter: HuggingFaceMultimodalAdapter,
    state: DecodeState,
    plan: PublishedKVCompressionPlan,
    *,
    step_index: int,
) -> tuple[DecodeState, tuple[JsonObject, ...]]:
    """Apply Elastic Cache's fixed-point generation-stage elimination policy."""

    if plan.method is not MosaicKVMethod.ELASTIC_CACHE_REIMPL:
        raise HuggingFaceRuntimeError("Elastic decode maintenance requires elastic_cache_reimpl")
    torch = _torch()
    snapshot = adapter.extract_past_key_values(state.past_key_values)
    layer_heads = tuple(
        tuple(head for head in plan.heads if head.layer == layer)
        for layer in range(len(plan.full_state.layers))
    )
    raw_masks = state.model_state.get("mosaickv_validity_masks")
    if raw_masks is None and state.model_state.get("mosaickv_packed_all_valid") is True:
        raw_masks = tuple(
            torch.ones(
                (
                    storage.kv_heads,
                    int(snapshot.layers[layer].key.shape[storage.key_sequence_dimension]),
                ),
                dtype=torch.bool,
                device=snapshot.layers[layer].key.device,
            )
            for layer, storage in enumerate(plan.full_state.layers)
        )
    if not isinstance(raw_masks, tuple) or len(raw_masks) != len(layer_heads):
        raise HuggingFaceRuntimeError("Elastic Cache decode validity metadata is incomplete")

    keep_by_layer: list[tuple[int, ...]] = []
    events: list[JsonObject] = []
    for layer, (heads, validity) in enumerate(zip(layer_heads, raw_masks, strict=True)):
        if not heads:
            raise HuggingFaceRuntimeError("Elastic Cache plan is missing a decoder layer")
        storage = plan.full_state.layers[layer]
        if getattr(validity, "ndim", 0) != 2 or getattr(validity, "dtype", None) != torch.bool:
            raise HuggingFaceRuntimeError(
                "Elastic Cache validity masks must be rank-two bool tensors"
            )
        if int(validity.shape[0]) != storage.kv_heads:
            raise HuggingFaceRuntimeError(
                "Elastic Cache validity KV-head count changed during decode"
            )
        if not bool(torch.all(validity == validity[0:1])):
            raise HuggingFaceRuntimeError(
                "Elastic Cache requires shared physical positions for all KV heads in a layer"
            )
        if len({head.capacity for head in heads}) != 1:
            raise HuggingFaceRuntimeError(
                "Elastic Cache plan lost its shared layer-wise anchor capacity"
            )
        valid_physical = tuple(
            int(value)
            for value in torch.nonzero(validity[0], as_tuple=False)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )
        source_positions = heads[0].source_positions
        prompt_capacity = heads[0].capacity
        effective_retention = prompt_capacity / source_positions
        target = max(
            prompt_capacity,
            math.floor((source_positions + step_index) * effective_retention + 1e-12),
        )
        removed: int | None = None
        fixed_rank: int | None = None
        if len(valid_physical) > target:
            minimum_rank = min(plan.config.elastic_start_tokens, len(valid_physical) - 1)
            fixed_rank = max(
                minimum_rank,
                min(prompt_capacity + plan.config.elastic_decode_offset, len(valid_physical) - 1),
            )
            removed = valid_physical[fixed_rank]
            valid_physical = tuple(
                position for index, position in enumerate(valid_physical) if index != fixed_rank
            )
        keep_by_layer.append(valid_physical)
        events.append(
            {
                "step": step_index,
                "layer": layer,
                "source_prompt_positions": source_positions,
                "prompt_anchor_capacity": prompt_capacity,
                "effective_retention_ratio": effective_retention,
                "target_active_positions": target,
                "configured_decode_offset": plan.config.elastic_decode_offset,
                "fixed_truncation_rank": fixed_rank,
                "removed_physical_position": removed,
                "active_positions_after": len(valid_physical),
            }
        )
    if not any(event["removed_physical_position"] is not None for event in events):
        return state, tuple(events)

    capacity = max(len(positions) for positions in keep_by_layer)
    layers: list[CacheLayerSnapshot] = []
    masks: list[Any] = []
    for layer_index, (source, positions) in enumerate(
        zip(snapshot.layers, keep_by_layer, strict=True)
    ):
        storage = plan.full_state.layers[layer_index]
        key_index = torch.tensor(positions, dtype=torch.long, device=source.key.device)
        value_index = key_index.to(device=source.value.device)
        key = torch.index_select(source.key, storage.key_sequence_dimension, key_index)
        value = torch.index_select(source.value, storage.value_sequence_dimension, value_index)
        padding = capacity - len(positions)
        if padding:
            key_shape = list(key.shape)
            value_shape = list(value.shape)
            key_shape[storage.key_sequence_dimension] = padding
            value_shape[storage.value_sequence_dimension] = padding
            key = torch.cat(
                (key, torch.zeros(key_shape, dtype=key.dtype, device=key.device)),
                dim=storage.key_sequence_dimension,
            )
            value = torch.cat(
                (value, torch.zeros(value_shape, dtype=value.dtype, device=value.device)),
                dim=storage.value_sequence_dimension,
            )
        layers.append(CacheLayerSnapshot(key, value, storage.key_sequence_dimension))
        masks.append(
            torch.tensor(
                [True] * len(positions) + [False] * padding,
                dtype=torch.bool,
                device=key.device,
            )
            .unsqueeze(0)
            .expand(storage.kv_heads, -1)
            .clone()
        )
    compacted = CacheSnapshot(
        tuple(layers),
        snapshot.source_class,
        snapshot.source_kind,
        capacity,
        snapshot.cached_key_state,
    )
    model_state = dict(state.model_state)
    model_state["mosaickv_validity_masks"] = tuple(masks)
    model_state["mosaickv_prompt_capacity"] = capacity
    return (
        DecodeState(
            past_key_values=adapter.inject_past_key_values(compacted),
            attention_mask=torch.ones(
                (1, capacity),
                dtype=state.attention_mask.dtype,
                device=state.attention_mask.device,
            ),
            active_cache_length=capacity,
            logical_sequence_length=state.logical_sequence_length,
            next_decode_position=state.next_decode_position,
            modality_map=state.modality_map,
            model_state=model_state,
        ),
        tuple(events),
    )


def _clone_decode_state(adapter: HuggingFaceMultimodalAdapter, state: DecodeState) -> DecodeState:
    snapshot = adapter.extract_past_key_values(state.past_key_values)
    cloned_model_state: dict[str, Any] = {}
    for key, value in state.model_state.items():
        if isinstance(value, tuple):
            cloned_model_state[key] = tuple(
                item.detach().clone() if hasattr(item, "detach") else item for item in value
            )
        elif hasattr(value, "detach"):
            cloned_model_state[key] = value.detach().clone()
        else:
            cloned_model_state[key] = value
    return DecodeState(
        adapter.inject_past_key_values(snapshot),
        state.attention_mask.detach().clone(),
        state.active_cache_length,
        state.logical_sequence_length,
        state.next_decode_position,
        state.modality_map,
        cloned_model_state,
    )


def _attention_outputs(
    adapter: HuggingFaceMultimodalAdapter,
    attention_weights: Sequence[Any],
    state: DecodeState,
) -> tuple[Any, ...]:
    """Reconstruct pre-output-projection attention vectors from measured weights and V."""

    torch = _torch()
    live_cache = adapter.extract_past_key_values(state.past_key_values, clone=False)
    if isinstance(live_cache, CacheSnapshot):
        layers = tuple((layer.value, layer.sequence_dimension) for layer in live_cache.layers)
    else:
        legacy, _source_kind = adapter._legacy_layers(live_cache)
        layers = tuple((value, adapter._cache_sequence_dimension(value)) for _key, value in legacy)
    if len(attention_weights) != len(layers):
        raise HuggingFaceRuntimeError("attention/cache layer counts differ in fidelity diagnostic")
    outputs: list[Any] = []
    for layer_index, (attention, (value, sequence_dimension)) in enumerate(
        zip(attention_weights, layers, strict=True)
    ):
        if attention is None:
            raise HuggingFaceRuntimeError(
                f"layer {layer_index} did not expose eager attention weights"
            )
        probabilities = attention.detach().float()
        values = value.detach().float().movedim(sequence_dimension, -2)
        if probabilities.ndim != 4 or values.ndim != 4:
            raise HuggingFaceRuntimeError("fidelity diagnostic requires rank-four attention and V")
        query_heads = int(probabilities.shape[1])
        kv_heads = int(values.shape[-3])
        if kv_heads < 1 or query_heads % kv_heads:
            raise HuggingFaceRuntimeError("query/KV heads are incompatible in fidelity diagnostic")
        sequence = min(int(probabilities.shape[-1]), int(values.shape[-2]))
        if sequence < 1:
            raise HuggingFaceRuntimeError("fidelity diagnostic received an empty cache")
        expanded_values = values[..., -sequence:, :].repeat_interleave(
            query_heads // kv_heads,
            dim=1,
        )
        outputs.append(
            torch.matmul(probabilities[..., -1:, -sequence:], expanded_values).squeeze(-2)
        )
    return tuple(outputs)


def _kv_head_counts(adapter: HuggingFaceMultimodalAdapter, state: DecodeState) -> tuple[int, ...]:
    live_cache = adapter.extract_past_key_values(state.past_key_values, clone=False)
    if isinstance(live_cache, CacheSnapshot):
        values = tuple(layer.value for layer in live_cache.layers)
        sequence_axes = tuple(layer.sequence_dimension for layer in live_cache.layers)
    else:
        legacy, _source_kind = adapter._legacy_layers(live_cache)
        values = tuple(value for _key, value in legacy)
        sequence_axes = tuple(adapter._cache_sequence_dimension(value) for value in values)
    result: list[int] = []
    for value, sequence_axis in zip(values, sequence_axes, strict=True):
        normalized_sequence = sequence_axis % value.ndim
        head_axis = normalized_sequence - 1
        if head_axis < 0:
            raise HuggingFaceRuntimeError("cache layout has no KV-head axis before sequence")
        result.append(int(value.shape[head_axis]))
    return tuple(result)


def _attention_output_error(reference: Sequence[Any], candidate: Sequence[Any]) -> float:
    torch = _torch()
    if len(reference) != len(candidate) or not reference:
        raise HuggingFaceRuntimeError("attention-output diagnostic layer counts differ")
    numerator = torch.zeros((), dtype=torch.float64, device=reference[0].device)
    denominator = torch.zeros((), dtype=torch.float64, device=reference[0].device)
    for left, right in zip(reference, candidate, strict=True):
        if left.shape != right.shape:
            raise HuggingFaceRuntimeError("attention-output diagnostic shapes differ")
        delta = left.double() - right.double()
        numerator = numerator + torch.sum(delta * delta)
        denominator = denominator + torch.sum(left.double() * left.double())
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-24)).item())


def _attention_output_error_by_kv_head(
    reference: Sequence[Any],
    candidate: Sequence[Any],
    kv_heads_by_layer: Sequence[int],
) -> tuple[tuple[int, int, float], ...]:
    """Return relative attention-output error for each layer/KV-head group."""

    torch = _torch()
    if len(reference) != len(candidate) or len(reference) != len(kv_heads_by_layer):
        raise HuggingFaceRuntimeError("per-head fidelity diagnostic layer counts differ")
    result: list[tuple[int, int, float]] = []
    for layer, (left, right, kv_heads) in enumerate(
        zip(reference, candidate, kv_heads_by_layer, strict=True)
    ):
        if left.shape != right.shape or left.ndim != 3:
            raise HuggingFaceRuntimeError("per-head attention-output shapes differ")
        query_heads = int(left.shape[1])
        if kv_heads < 1 or query_heads % kv_heads:
            raise HuggingFaceRuntimeError("per-head fidelity query/KV grouping is invalid")
        query_heads_per_kv = query_heads // kv_heads
        reference_groups = left.double().reshape(
            int(left.shape[0]), kv_heads, query_heads_per_kv, int(left.shape[-1])
        )
        candidate_groups = right.double().reshape_as(reference_groups)
        reduction_axes = (0, 2, 3)
        numerator = torch.sum((reference_groups - candidate_groups) ** 2, dim=reduction_axes)
        denominator = torch.sum(reference_groups * reference_groups, dim=reduction_axes).clamp_min(
            1e-24
        )
        errors = torch.sqrt(numerator / denominator).detach().cpu().tolist()
        result.extend((layer, kv_head, float(errors[kv_head])) for kv_head in range(kv_heads))
    return tuple(result)


def _logit_kl(reference_logits: Any, candidate_logits: Any) -> float:
    torch = _torch()
    reference = reference_logits.detach().float()
    candidate = candidate_logits.detach().float()
    if reference.shape != candidate.shape:
        raise HuggingFaceRuntimeError("logit shapes differ in fidelity diagnostic")
    reference_log_probability = torch.log_softmax(reference, dim=-1)
    candidate_log_probability = torch.log_softmax(candidate, dim=-1)
    divergence = torch.sum(
        reference_log_probability.exp() * (reference_log_probability - candidate_log_probability),
        dim=-1,
    ).mean()
    return max(0.0, float(divergence.item()))


def _short_horizon_candidate_evaluation(
    candidate_id: str,
    *,
    kl_values: Sequence[float],
    maximum_logit_differences: Sequence[float],
    greedy_token_matches: Sequence[bool],
    maximum_kl_threshold: float,
    maximum_logit_difference_threshold: float,
) -> ShortHorizonLogitCandidateEvaluation:
    """Apply the preregistered V98 gate without benchmark labels."""

    if not kl_values or len(kl_values) != len(maximum_logit_differences):
        raise ValueError("short-horizon logit metrics must be nonempty and aligned")
    if len(kl_values) != len(greedy_token_matches):
        raise ValueError("short-horizon greedy-token evidence must align with logits")
    values = (*kl_values, *maximum_logit_differences)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("short-horizon logit metrics must be finite and nonnegative")
    maximum_kl = max(float(value) for value in kl_values)
    maximum_difference = max(float(value) for value in maximum_logit_differences)
    token_match = all(bool(value) for value in greedy_token_matches)
    return ShortHorizonLogitCandidateEvaluation(
        candidate_id=candidate_id,
        passed=bool(
            token_match
            and maximum_kl <= maximum_kl_threshold + 1e-12
            and maximum_difference <= maximum_logit_difference_threshold + 1e-12
        ),
        greedy_token_match=token_match,
        maximum_kl=maximum_kl,
        maximum_logit_difference=maximum_difference,
        evaluated_steps=len(kl_values),
    )


def probe_isolated_packed_candidate(
    adapter: HuggingFaceMultimodalAdapter,
    reference_state: DecodeState,
    packed: PackedRuntimeCache,
    *,
    candidate_id: str,
    teacher_forced_inputs: Sequence[Any],
    reference_logits: Sequence[Any],
) -> LogitProbeResult:
    """Evaluate one packed candidate without retaining a live cache reference."""

    if not teacher_forced_inputs or len(teacher_forced_inputs) != len(reference_logits):
        raise HuggingFaceRuntimeError(
            "isolated candidate probe inputs and reference logits must be nonempty and aligned"
        )
    divergences: list[float] = []
    differences: list[float] = []
    matches: list[bool] = []
    with construct_isolated_packed_decode_state(
        adapter,
        reference_state,
        packed,
        candidate_id=candidate_id,
    ) as handle:
        for teacher_input, full_logits in zip(
            teacher_forced_inputs,
            reference_logits,
            strict=True,
        ):
            output = adapter.decode_one_token(
                teacher_input,
                handle.state,
                capture_queries=False,
                capture_attentions=False,
            )
            handle.adopt(output.state)
            divergences.append(_logit_kl(full_logits, output.logits))
            differences.append(
                float(
                    _torch()
                    .max(
                        _torch().abs(full_logits.detach().float() - output.logits.detach().float())
                    )
                    .item()
                )
            )
            matches.append(
                int(full_logits.detach().argmax(dim=-1).reshape(-1)[0].item())
                == int(output.logits.detach().argmax(dim=-1).reshape(-1)[0].item())
            )
    return LogitProbeResult(
        greedy_token_match=all(matches),
        maximum_forward_kl=max(divergences),
        maximum_absolute_logit_difference=max(differences),
        evaluated_steps=len(divergences),
    )


def _run_short_horizon_logit_stability(
    adapter: HuggingFaceMultimodalAdapter,
    prefill: PrefillOutput,
    plan: MosaicKVCompressionPlan,
    config: RunConfig,
    *,
    key_transform: RotaryPrototypeKeyTransform | None,
    physical_prompt_capacity: int,
) -> tuple[MosaicKVCompressionPlan, ShortHorizonLogitStabilityDecision]:
    """Choose one same-byte exact cache using isolated FullKV logit probes.

    The probe teacher-forces the deterministic FullKV greedy prefix into each
    candidate so a cache perturbation is measured before trajectory drift. All
    probe states are disposable clones; the measured decode starts from a fresh
    packing of the selected construction.
    """

    if plan.method is not MosaicKVMethod.MOSAICKV_EXACT:
        raise HuggingFaceRuntimeError("short-horizon logit stability supports mosaickv_exact")
    requested = config.selection.stable_head_logit_stability_shortlist_size
    shortlist = plan.external_exact_logit_shortlist[:requested]
    if len(shortlist) != requested:
        raise HuggingFaceRuntimeError(
            "short-horizon logit stability did not produce the preregistered shortlist"
        )
    if shortlist[0].selected_node_ids != plan.external_exact_node_ids:
        raise HuggingFaceRuntimeError("primary logit-stability candidate differs from the plan")
    shortlist_costs = tuple(
        sum(plan.full_state.blocks[node_id].byte_size for node_id in item.selected_node_ids)
        for item in shortlist
    )
    if len(set(shortlist_costs)) != 1:
        raise HuggingFaceRuntimeError(
            "short-horizon logit-stability candidates must use identical exact KV bytes"
        )
    if len({item.candidate_id for item in shortlist}) != len(shortlist):
        raise HuggingFaceRuntimeError("short-horizon logit-stability candidate IDs must be unique")
    scalar = next(
        (item for item in shortlist if item.candidate_id == "scalar_prompt_topk_anchor"),
        None,
    )
    if scalar is None:
        raise HuggingFaceRuntimeError("logit-stability shortlist lacks its scalar anchor")

    horizon = config.selection.stable_head_logit_stability_horizon
    reference_state = _clone_decode_state(adapter, prefill.state)
    reference_input = prefill.next_token_id.detach().clone()
    teacher_forced_inputs: list[Any] = []
    reference_logits: list[Any] = []
    for _step in range(horizon):
        teacher_forced_inputs.append(reference_input.detach().clone())
        output = adapter.decode_one_token(
            reference_input,
            reference_state,
            capture_queries=False,
            capture_attentions=False,
        )
        reference_state = output.state
        reference_logits.append(output.logits.detach().clone())
        reference_input = output.next_token_id.detach().clone()

    active_budget = SelectionBudget(plan.active_budget_value, config.cache.budget_unit)
    constructions: dict[str, ThreeTierCacheConstruction] = {}
    evaluations: list[ShortHorizonLogitCandidateEvaluation] = []
    for alternative in shortlist:
        construction = (
            plan.construction
            if alternative.selected_node_ids == plan.external_exact_node_ids
            else _exact_construction(
                plan.full_state,
                plan.graph,
                plan.selection,
                adapter.capabilities,
                config,
                key_transform,
                additional_exact_source_node_ids=alternative.selected_node_ids,
                active_budget=active_budget,
            )
        )
        constructions[alternative.candidate_id] = construction
        packed = pack_runtime_cache(
            adapter,
            plan.full_state,
            construction.state,
            construction.prototypes,
            (),
            physical_prompt_capacity=physical_prompt_capacity,
            packing_mode=config.cache.packing_mode,
        )
        candidate_state = _state_from_packed(adapter, prefill.state, packed)
        divergences: list[float] = []
        differences: list[float] = []
        matches: list[bool] = []
        for teacher_input, full_logits in zip(
            teacher_forced_inputs,
            reference_logits,
            strict=True,
        ):
            output = adapter.decode_one_token(
                teacher_input,
                candidate_state,
                capture_queries=False,
                capture_attentions=False,
            )
            candidate_state = output.state
            divergences.append(_logit_kl(full_logits, output.logits))
            difference = float(
                _torch()
                .max(_torch().abs(full_logits.detach().float() - output.logits.detach().float()))
                .item()
            )
            differences.append(difference)
            matches.append(
                int(full_logits.detach().argmax(dim=-1).reshape(-1)[0].item())
                == int(output.logits.detach().argmax(dim=-1).reshape(-1)[0].item())
            )
        evaluations.append(
            _short_horizon_candidate_evaluation(
                alternative.candidate_id,
                kl_values=divergences,
                maximum_logit_differences=differences,
                greedy_token_matches=matches,
                maximum_kl_threshold=(config.selection.stable_head_logit_stability_maximum_kl),
                maximum_logit_difference_threshold=(
                    config.selection.stable_head_logit_stability_maximum_logit_difference
                ),
            )
        )

    first_passing = next((item for item in evaluations if item.passed), None)
    selected_id = scalar.candidate_id if first_passing is None else first_passing.candidate_id
    selected_alternative = next(item for item in shortlist if item.candidate_id == selected_id)
    selected_evaluation = next(item for item in evaluations if item.candidate_id == selected_id)
    selected_plan = replace(
        plan,
        effective_method=f"{plan.effective_method}__short_logit_{selected_id}",
        construction=constructions[selected_id],
        external_exact_node_ids=selected_alternative.selected_node_ids,
        external_exact_capacity_records=selected_alternative.capacity_records,
    )
    return selected_plan, ShortHorizonLogitStabilityDecision(
        enabled=True,
        shortlist_size=len(shortlist),
        evaluated_candidates=tuple(evaluations),
        selected_candidate_id=selected_id,
        selected_candidate_passed=selected_evaluation.passed,
        used_fail_closed_scalar_anchor=(selected_id == scalar.candidate_id),
        horizon=horizon,
        maximum_kl_threshold=config.selection.stable_head_logit_stability_maximum_kl,
        maximum_logit_difference_threshold=(
            config.selection.stable_head_logit_stability_maximum_logit_difference
        ),
    )


@dataclass(frozen=True, slots=True)
class _ProgressiveGroupReservation:
    protected_node_ids: frozenset[int]
    groups_by_route: Mapping[tuple[int, int], tuple[tuple[int, ...], ...]]
    candidate_count_by_route: Mapping[tuple[int, int], int]
    target_bytes_by_route: Mapping[tuple[int, int], int]
    protected_bytes_by_route: Mapping[tuple[int, int], int]
    protected_score_by_route: Mapping[tuple[int, int], float]


def _progressive_fixed_source_ids(plan: MosaicKVCompressionPlan) -> frozenset[int]:
    """Translate the compact planner selection into immutable source-block IDs."""

    return frozenset(
        plan.source_node_id(planner_node_id) for planner_node_id in plan.selection.selected_node_ids
    )


def _progressive_exact_construction(
    plan: MosaicKVCompressionPlan,
    capabilities: AdapterCapabilities,
    config: RunConfig,
    retained_source_node_ids: Iterable[int],
    *,
    key_transform: RotaryPrototypeKeyTransform | None,
    active_budget: SelectionBudget,
) -> tuple[ThreeTierCacheConstruction, SelectionResult, tuple[int, ...]]:
    """Construct an exact cache after partitioning planner and external IDs.

    Progressive eviction starts from FullKV, so its retained set can include
    routed graph nodes that were absent from the original compressed plan.
    Those nodes must be represented by graph-local selection IDs; only blocks
    outside the compact planner graph may use the external-exact path.
    """

    retained = frozenset(int(node_id) for node_id in retained_source_node_ids)
    if any(node_id < 0 or node_id >= len(plan.full_state.blocks) for node_id in retained):
        raise HuggingFaceRuntimeError(
            "progressive retained set contains a source block outside FullKV"
        )
    source_to_planner = {node.source_id: node.node_id for node in plan.graph.nodes}
    selected_planner_ids = tuple(
        sorted(source_to_planner[node_id] for node_id in retained if node_id in source_to_planner)
    )
    external_ids = tuple(sorted(retained - set(source_to_planner)))
    objective = BudgetedObjective(plan.graph, plan.utilities, config.selection)
    selection = select_exact_subset(objective, active_budget, selected_planner_ids)
    construction = _exact_construction(
        plan.full_state,
        plan.graph,
        selection,
        capabilities,
        config,
        key_transform,
        additional_exact_source_node_ids=external_ids,
        active_budget=active_budget,
    )
    return construction, selection, external_ids


def _exact_cache_storage_accounting(
    plan: MosaicKVCompressionPlan,
    retained_source_node_ids: Iterable[int],
    construction: ThreeTierCacheConstruction,
    external_source_node_ids: Iterable[int],
    *,
    knapsack_bytes: int,
    packed_physical_bytes: int,
) -> ExactCacheStorageAccounting:
    """Reconcile controller costs with exact construction and dense packing.

    The exact tier's storage boundary is the unique set of FullKV source
    descriptors.  Planner-local IDs are intentionally not counted here: they
    are translated back to immutable FullKV source IDs before accounting.
    This also prevents compact-planner and external-exact blocks from being
    double counted.
    """

    raw_retained = tuple(int(node_id) for node_id in retained_source_node_ids)
    retained = tuple(sorted(set(raw_retained)))
    external = frozenset(int(node_id) for node_id in external_source_node_ids)
    if len(retained) != len(raw_retained):
        raise HuggingFaceRuntimeError(
            "hierarchical retained source IDs must be unique at the storage boundary"
        )
    if any(node_id < 0 or node_id >= len(plan.full_state.blocks) for node_id in retained):
        raise HuggingFaceRuntimeError(
            "hierarchical accounting references a source block outside FullKV"
        )
    retained_set = frozenset(retained)
    if not external <= retained_set:
        raise HuggingFaceRuntimeError(
            "external-exact accounting contains a block outside the retained source set"
        )
    planner = retained_set - external
    constructed_ids = tuple(sorted(construction.exact_node_ids))
    if constructed_ids != retained:
        raise HuggingFaceRuntimeError(
            "exact construction membership differs from the hierarchical retained source set"
        )
    unique_source_bytes = sum(plan.full_state.blocks[node_id].byte_size for node_id in retained)
    planner_bytes = sum(plan.full_state.blocks[node_id].byte_size for node_id in planner)
    external_bytes = sum(plan.full_state.blocks[node_id].byte_size for node_id in external)
    constructed_active_bytes = construction.state.statistics.active_kv_bytes
    if packed_physical_bytes < constructed_active_bytes:
        raise HuggingFaceRuntimeError(
            "packed physical storage is smaller than its exact logical payload"
        )
    return ExactCacheStorageAccounting(
        knapsack_bytes=knapsack_bytes,
        unique_selected_source_bytes=unique_source_bytes,
        planner_tier_bytes=planner_bytes,
        external_exact_bytes=external_bytes,
        constructed_active_bytes=constructed_active_bytes,
        physical_padding_bytes=packed_physical_bytes - constructed_active_bytes,
        selected_source_block_count=len(retained),
        planner_source_block_count=len(planner),
        external_source_block_count=len(external),
    )


def _progressive_eviction_risks(plan: MosaicKVCompressionPlan) -> tuple[float, ...]:
    """Map compact planner utilities back to immutable FullKV source IDs once."""

    utility_by_source = {
        node.source_id: plan.utilities.for_node(node.node_id) for node in plan.graph.nodes
    }
    risks: list[float] = []
    for source_id in range(len(plan.full_state.blocks)):
        prompt = (
            0.0
            if plan.prompt_attention_by_node is None
            else float(plan.prompt_attention_by_node[source_id])
        )
        utility = utility_by_source.get(source_id)
        risks.append(
            prompt
            if utility is None
            else float(
                prompt
                + utility.forecast_attention_probability
                + utility.expected_attention_output_contribution
                + utility.value_novelty
                + utility.singleton_coverage_gain
                + utility.modality_rarity
            )
        )
    return tuple(risks)


def _progressive_group_reservation(
    plan: MosaicKVCompressionPlan,
    config: RunConfig,
    position_limits: Mapping[tuple[int, int], int],
    eviction_risks: Sequence[float],
) -> _ProgressiveGroupReservation:
    """Reserve coherent evidence groups before any V99 eviction proposal."""

    full_state = plan.full_state
    stable_routes = {
        (record.layer, record.kv_head) for record in plan.external_exact_capacity_records
    }
    fixed_ids = set(_progressive_fixed_source_ids(plan))
    groups = coherent_layout_groups(
        full_state,
        tuple(
            node_id
            for node_id, block in enumerate(full_state.blocks)
            if (block.layer, block.kv_head) in stable_routes
        ),
        maximum_blocks=(config.selection.stable_head_query_coverage_layout_group_maximum_blocks),
        row_bins=config.selection.stable_head_query_coverage_layout_group_row_bins,
    )
    candidates_by_route: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    for group in groups:
        first = full_state.blocks[group[0]]
        candidates_by_route[(first.layer, first.kv_head)].append(group)

    selected_by_route: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    target_bytes_by_route: dict[tuple[int, int], int] = {}
    protected_bytes_by_route: dict[tuple[int, int], int] = defaultdict(int)
    protected_score_by_route: dict[tuple[int, int], float] = defaultdict(float)
    protected_ids: set[int] = set()
    for route in sorted(stable_routes):
        route_groups = candidates_by_route.get(route, [])
        if not route_groups:
            target_bytes_by_route[route] = 0
            continue
        route_nodes = [
            node_id
            for node_id, block in enumerate(full_state.blocks)
            if (block.layer, block.kv_head) == route
        ]
        bytes_per_position = min(
            full_state.blocks[node_id].byte_size / full_state.blocks[node_id].position_count
            for node_id in route_nodes
        )
        target_positions = math.floor(
            position_limits[route]
            * config.selection.stable_head_query_coverage_initial_fraction
            * config.selection.stable_head_query_coverage_layout_group_fraction
        )
        target_bytes = int(target_positions * bytes_per_position)
        target_bytes_by_route[route] = target_bytes
        fixed_positions = sum(
            full_state.blocks[node_id].position_count
            for node_id in route_nodes
            if node_id in fixed_ids
        )
        protected_positions = 0
        ordered = sorted(
            route_groups,
            key=lambda group: (
                -sum(eviction_risks[node_id] for node_id in group),
                group,
            ),
        )
        for group in ordered:
            if protected_bytes_by_route[route] >= target_bytes:
                break
            new_nodes = tuple(
                node_id
                for node_id in group
                if node_id not in protected_ids and node_id not in fixed_ids
            )
            if not new_nodes:
                continue
            group_positions = sum(
                full_state.blocks[node_id].position_count for node_id in new_nodes
            )
            if fixed_positions + protected_positions + group_positions > position_limits[route]:
                continue
            selected_by_route[route].append(group)
            protected_ids.update(group)
            protected_positions += group_positions
            protected_bytes_by_route[route] += sum(
                full_state.blocks[node_id].byte_size for node_id in new_nodes
            )
            protected_score_by_route[route] += sum(eviction_risks[node_id] for node_id in new_nodes)
    return _ProgressiveGroupReservation(
        protected_node_ids=frozenset(protected_ids),
        groups_by_route={route: tuple(values) for route, values in selected_by_route.items()},
        candidate_count_by_route={
            route: len(candidates_by_route.get(route, ())) for route in stable_routes
        },
        target_bytes_by_route=target_bytes_by_route,
        protected_bytes_by_route=dict(protected_bytes_by_route),
        protected_score_by_route=dict(protected_score_by_route),
    )


def _progressive_capacity_records(
    plan: MosaicKVCompressionPlan,
    external_node_ids: frozenset[int],
    reservation: _ProgressiveGroupReservation,
    *,
    allow_fullkv_capacity_expansion: bool = False,
) -> tuple[HeadLocalCapacityRecord, ...]:
    full_state = plan.full_state
    by_route: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node_id in sorted(external_node_ids):
        block = full_state.blocks[node_id]
        by_route[(block.layer, block.kv_head)].append(node_id)
    result: list[HeadLocalCapacityRecord] = []
    for record in plan.external_exact_capacity_records:
        route = (record.layer, record.kv_head)
        node_ids = by_route.get(route, [])
        selected_positions = sum(full_state.blocks[node_id].position_count for node_id in node_ids)
        selected_bytes = sum(full_state.blocks[node_id].byte_size for node_id in node_ids)
        selected_visual = sum(
            full_state.blocks[node_id].position_count
            for node_id in node_ids
            if full_state.blocks[node_id].modality in {Modality.IMAGE, Modality.VIDEO}
        )
        groups = reservation.groups_by_route.get(route, ())
        protected_bytes = reservation.protected_bytes_by_route.get(route, 0)
        target_bytes = reservation.target_bytes_by_route.get(route, 0)
        position_limit = record.position_limit
        if selected_positions > position_limit:
            if not allow_fullkv_capacity_expansion:
                raise HuggingFaceRuntimeError("non-fallback selection exceeds its route capacity")
            # A fail-closed hierarchical decision may deliberately return
            # FullKV. Preserve that actual physical capacity in the audit
            # record; active-byte gates will reject it rather than disguising
            # it as a compressed candidate or crashing before measurement.
            position_limit = selected_positions
        result.append(
            replace(
                record,
                position_limit=position_limit,
                selected_positions=selected_positions,
                shortfall_positions=position_limit - selected_positions,
                selected_bytes=selected_bytes,
                selected_visual_positions=selected_visual,
                query_coverage_initial_layout_group_candidate_count=(
                    reservation.candidate_count_by_route.get(route, 0)
                ),
                query_coverage_initial_layout_group_protected_group_count=len(groups),
                query_coverage_initial_layout_group_protected_block_count=sum(
                    len(group) for group in groups
                ),
                query_coverage_initial_layout_group_protected_bytes=protected_bytes,
                query_coverage_initial_layout_group_protected_score=(
                    reservation.protected_score_by_route.get(route, 0.0)
                ),
                query_coverage_initial_layout_group_target_bytes=target_bytes,
                query_coverage_initial_layout_group_accepted_candidate_count=len(groups),
                query_coverage_initial_layout_group_knapsack_selected_count=len(groups),
                query_coverage_initial_layout_group_node_ids=groups,
            )
        )
    return tuple(result)


def _hierarchical_effective_position_limits(
    plan: MosaicKVCompressionPlan,
    dense_position_limits: Mapping[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """Intersect dense limits with already-attainable external-route limits."""

    routes = {(block.layer, block.kv_head) for block in plan.full_state.blocks}
    if set(dense_position_limits) != routes:
        raise HuggingFaceRuntimeError(
            "hierarchical dense position limits must cover every FullKV route"
        )
    effective = {route: int(limit) for route, limit in dense_position_limits.items()}
    if any(limit < 1 for limit in effective.values()):
        raise HuggingFaceRuntimeError("hierarchical position limits must be positive")
    seen: set[tuple[int, int]] = set()
    for record in plan.external_exact_capacity_records:
        route = (record.layer, record.kv_head)
        if route in seen:
            raise HuggingFaceRuntimeError("external capacity records contain duplicate routes")
        if route not in effective or record.position_limit < 1:
            raise HuggingFaceRuntimeError("external capacity record has an invalid route limit")
        seen.add(route)
        effective[route] = min(effective[route], int(record.position_limit))
    return effective


def _hierarchical_route_ladders(
    plan: MosaicKVCompressionPlan,
    config: RunConfig,
    position_limits: Mapping[tuple[int, int], int],
    *,
    feasible_baseline_node_ids_by_route: (Mapping[tuple[int, int], frozenset[int]] | None) = None,
    refinement_positions: int = 0,
) -> tuple[RouteCapacityLadder, ...]:
    """Build FullKV or position-feasible label-blind route ladders.

    A feasible refinement ladder is deliberately rebuilt from an already
    representable baseline.  Filtering a FullKV-to-cap ladder after the fact
    leaves the cap candidate as level zero and erases every positive-savings
    screening step.  Rebuilding from the baseline yields nested levels whose
    physical position counts all remain within the dense-cache limit.
    """

    if refinement_positions < 0:
        raise HuggingFaceRuntimeError("hierarchical refinement positions cannot be negative")
    if (feasible_baseline_node_ids_by_route is None) != (refinement_positions == 0):
        raise HuggingFaceRuntimeError(
            "feasible ladder baselines and positive refinement positions are required together"
        )

    full_state = plan.full_state
    # KVBlockDescriptor.byte_size is the exact-tier logical storage boundary:
    # FullKVState validates its sum against tensor numel*dtype bytes, and
    # ExactTier validates each gathered payload against the same descriptor.
    # Build every knapsack cost from this immutable source table so compact
    # planner IDs and external-exact IDs cannot introduce a second boundary.
    source_storage_costs = tuple(block.byte_size for block in full_state.blocks)
    if sum(source_storage_costs) != full_state.active_bytes:
        raise HuggingFaceRuntimeError("hierarchical source costs do not match FullKV exact storage")
    risks = _progressive_eviction_risks(plan)
    reservation = _progressive_group_reservation(plan, config, position_limits, risks)
    group_by_node: dict[int, str] = {}
    for route, groups in sorted(reservation.groups_by_route.items()):
        for group_index, group in enumerate(groups):
            group_id = f"{route[0]}:{route[1]}:{group_index}"
            for node_id in group:
                if node_id in group_by_node:
                    raise HuggingFaceRuntimeError("hierarchical protected layout groups overlap")
                group_by_node[node_id] = group_id
    fixed_ids = _progressive_fixed_source_ids(plan)
    route_blocks: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: list[RouteBlockCandidate] = []
    for node_id, block in enumerate(full_state.blocks):
        route = (block.layer, block.kv_head)
        route_blocks[route].append(node_id)
        risk = max(0.0, float(risks[node_id]))
        candidates.append(
            RouteBlockCandidate(
                node_id=node_id,
                layer=block.layer,
                kv_head=block.kv_head,
                byte_size=source_storage_costs[node_id],
                evidence_utility=risk,
                eviction_risk=risk,
                mandatory=block.mandatory or node_id in fixed_ids,
                atomic_group_id=group_by_node.get(node_id),
            )
        )
    if set(route_blocks) != set(position_limits):
        raise HuggingFaceRuntimeError(
            "hierarchical position limits must cover every FullKV layer/head route"
        )
    if feasible_baseline_node_ids_by_route is not None:
        if set(feasible_baseline_node_ids_by_route) != set(route_blocks):
            raise HuggingFaceRuntimeError(
                "feasible ladder baselines must cover every FullKV layer/head route"
            )
        candidate_by_id = {candidate.node_id: candidate for candidate in candidates}
        baseline_candidates: list[RouteBlockCandidate] = []
        for route, baseline_node_ids in sorted(feasible_baseline_node_ids_by_route.items()):
            expected = set(route_blocks[route])
            if not baseline_node_ids or not set(baseline_node_ids) <= expected:
                raise HuggingFaceRuntimeError(
                    "feasible ladder baseline contains an empty or cross-route selection"
                )
            baseline_candidates.extend(
                candidate_by_id[node_id] for node_id in sorted(baseline_node_ids)
            )
        candidates = baseline_candidates
        route_blocks = defaultdict(list)
        for candidate in candidates:
            route_blocks[candidate.route].append(candidate.node_id)
    target_bytes_by_route: dict[tuple[int, int], int] = {}
    for route, node_ids in sorted(route_blocks.items()):
        bytes_per_position = {
            source_storage_costs[node_id] // full_state.blocks[node_id].position_count
            for node_id in node_ids
        }
        if len(bytes_per_position) != 1 or any(
            source_storage_costs[node_id] % full_state.blocks[node_id].position_count
            for node_id in node_ids
        ):
            raise HuggingFaceRuntimeError(
                "hierarchical route blocks do not have a uniform integral byte cost"
            )
        unit = next(iter(bytes_per_position))
        full_bytes = sum(source_storage_costs[node_id] for node_id in node_ids)
        if feasible_baseline_node_ids_by_route is None:
            target_bytes_by_route[route] = min(
                full_bytes,
                max(1, int(position_limits[route]) * unit),
            )
        else:
            protected_bytes = sum(
                candidate.byte_size
                for candidate in candidates
                if candidate.route == route and candidate.protected
            )
            target_bytes_by_route[route] = max(
                protected_bytes,
                full_bytes - refinement_positions * unit,
            )
    built = build_route_capacity_ladders(candidates, target_bytes_by_route)
    if feasible_baseline_node_ids_by_route is None:
        return built
    return tuple(
        RouteCapacityLadder(
            route=ladder.route,
            levels=(
                replace(ladder.levels[0], kind=CapacityLevelKind.FEASIBLE_BASELINE),
                *ladder.levels[1:],
            ),
            protected_node_ids=ladder.protected_node_ids,
            target_active_bytes=ladder.target_active_bytes,
            target_reached=ladder.target_reached,
        )
        for ladder in built
    )


def _attach_feasible_baseline_reallocation_levels(
    ladders: Sequence[RouteCapacityLadder],
    position_limited_source_ladders: Sequence[RouteCapacityLadder],
) -> tuple[RouteCapacityLadder, ...]:
    """Retain same-capacity source swaps around each feasible baseline.

    Refinement ladders are built from baseline membership so every ordinary
    level is physically safe. That subset construction intentionally cannot
    propose restoring a block omitted at the baseline boundary. The original
    position-limited ladder already contains deterministic equal-byte swaps
    at that boundary; attach them as reallocation-only choices without
    changing any ordinary capacity level.
    """

    ordered = tuple(sorted(ladders, key=lambda item: item.route))
    source_by_route = {ladder.route: ladder for ladder in position_limited_source_ladders}
    if set(source_by_route) != {ladder.route for ladder in ordered}:
        raise HuggingFaceRuntimeError(
            "baseline reallocation sources must cover every feasible route ladder"
        )
    result: list[RouteCapacityLadder] = []
    for ladder in ordered:
        source = source_by_route[ladder.route]
        baseline_bytes = ladder.full_level.active_bytes
        ordinary = tuple(
            level
            for level in ladder.levels
            if level.kind is not CapacityLevelKind.REALLOCATION_ALTERNATIVE
        )
        alternatives = list(ladder.reallocation_levels)
        seen_memberships = {level.selected_node_ids for level in alternatives}
        for level in source.reallocation_levels:
            if (
                level.active_bytes == baseline_bytes
                and level.selected_node_ids not in seen_memberships
            ):
                alternatives.append(level)
                seen_memberships.add(level.selected_node_ids)
        rebuilt = tuple(
            replace(level, level_index=index)
            for index, level in enumerate((*ordinary, *alternatives))
        )
        result.append(
            RouteCapacityLadder(
                route=ladder.route,
                levels=rebuilt,
                protected_node_ids=ladder.protected_node_ids,
                target_active_bytes=ladder.target_active_bytes,
                target_reached=ladder.target_reached,
            )
        )
    return tuple(result)


def _hierarchical_reference_rollout(
    adapter: HuggingFaceMultimodalAdapter,
    prefill: PrefillOutput,
    horizon: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Capture one teacher-forced FullKV prefix from an isolated state."""

    teacher_inputs: list[Any] = []
    reference_logits: list[Any] = []
    reference_input = prefill.next_token_id.detach().clone()
    with construct_isolated_decode_state(
        adapter,
        prefill.state,
        candidate_id="v100-fullkv-teacher",
    ) as handle:
        for _step in range(horizon):
            teacher_inputs.append(reference_input.detach().clone())
            output = adapter.decode_one_token(
                reference_input,
                handle.state,
                capture_queries=False,
                capture_attentions=False,
            )
            handle.adopt(output.state)
            reference_logits.append(output.logits.detach().clone())
            reference_input = output.next_token_id.detach().clone()
    return tuple(teacher_inputs), tuple(reference_logits)


def _run_hierarchical_logit_safe_eviction(
    adapter: HuggingFaceMultimodalAdapter,
    prefill: PrefillOutput,
    plan: MosaicKVCompressionPlan,
    config: RunConfig,
    *,
    key_transform: RotaryPrototypeKeyTransform | None,
    position_limits: Mapping[tuple[int, int], int],
) -> tuple[MosaicKVCompressionPlan, BoundedHierarchicalProbeDecision]:
    """Run V100's bounded hierarchy using only disposable isolated states."""

    hierarchical_started = time.perf_counter()
    if plan.method is not MosaicKVMethod.MOSAICKV_EXACT:
        raise HuggingFaceRuntimeError("hierarchical logit-safe eviction supports mosaickv_exact")
    selection_config = config.selection
    effective_position_limits = _hierarchical_effective_position_limits(plan, position_limits)
    unrestricted_ladders = _hierarchical_route_ladders(plan, config, effective_position_limits)
    position_count_by_node = {
        node_id: block.position_count for node_id, block in enumerate(plan.full_state.blocks)
    }
    position_limited_ladders = restrict_route_capacity_ladders_to_position_limits(
        unrestricted_ladders,
        position_count_by_node,
        effective_position_limits,
    )
    feasible_baseline_node_ids_by_route = {
        ladder.route: frozenset(ladder.full_level.selected_node_ids)
        for ladder in position_limited_ladders
    }
    ladders = _hierarchical_route_ladders(
        plan,
        config,
        effective_position_limits,
        feasible_baseline_node_ids_by_route=feasible_baseline_node_ids_by_route,
        refinement_positions=2 * config.cache.block_size,
    )
    # The rebuilt ladder contains only subsets of the largest feasible
    # baseline. Keep this check explicit so a future candidate-builder change
    # cannot silently reintroduce a route-overflow probe.
    ladders = restrict_route_capacity_ladders_to_position_limits(
        ladders,
        position_count_by_node,
        effective_position_limits,
    )
    ladders = _attach_feasible_baseline_reallocation_levels(
        ladders,
        position_limited_ladders,
    )
    fullkv_fallback_selection = select_verified_capacity_levels(
        unrestricted_ladders,
        {ladder.route: (0,) for ladder in unrestricted_ladders},
        plan.active_budget_value,
    )
    teacher_inputs, reference_logits = _hierarchical_reference_rollout(
        adapter,
        prefill,
        selection_config.stable_head_hierarchical_horizon,
    )
    full_logical_bytes = sum(block.byte_size for block in plan.full_state.blocks)
    probe_budget = SelectionBudget(full_logical_bytes, BudgetUnit.BYTES)
    probe_config = replace(
        config,
        cache=replace(
            config.cache,
            budget_value=full_logical_bytes,
            total_storage_budget_bytes=full_logical_bytes,
        ),
    )
    # This cache contains only position-feasible ladder artifacts. FullKV
    # fallback uses the unrestricted ladder, whose level-zero signatures can
    # collide after feasible ladders are reindexed. It must therefore never
    # read this namespace.
    artifacts: dict[
        tuple[tuple[tuple[int, int], int], ...],
        _HierarchicalExactArtifact,
    ] = {}

    def probe(proposal: HierarchicalCapacityProposal) -> LogitProbeResult:
        level_indices = proposal.level_index_by_route()
        violations = proposal_route_capacity_violations(
            proposal,
            ladders,
            position_count_by_node,
            effective_position_limits,
            affected_routes_only=(proposal.kind is HierarchicalProposalKind.SCREENING),
        )
        if violations:
            attempted_active_bytes = sum(
                ladder.level(level_indices[ladder.route]).active_bytes for ladder in ladders
            )
            attempted_positions = max(item.selected_positions for item in violations)
            error = HuggingFaceRuntimeError(
                "hierarchical proposal exceeds a route position limit before logit probing"
            )
            error.mosaickv_failure_trace_payload = _route_capacity_failure_trace_payload(
                proposal,
                violations,
                attempted_active_kv_bytes=attempted_active_bytes,
                hard_active_kv_budget_bytes=plan.active_budget_value,
                attempted_prompt_capacity=attempted_positions,
                hard_prompt_capacity=max(effective_position_limits.values()),
            )
            raise error
        retained_ids = frozenset(
            node_id
            for ladder in ladders
            for node_id in ladder.level(level_indices[ladder.route]).selected_node_ids
        )
        construction, selection, external = _progressive_exact_construction(
            plan,
            adapter.capabilities,
            probe_config,
            retained_ids,
            key_transform=key_transform,
            active_budget=probe_budget,
        )
        maximum_positions = max(
            layout.active_cache_length for layout in construction.active_layouts
        )
        packed = pack_runtime_cache(
            adapter,
            plan.full_state,
            construction.state,
            (),
            (),
            physical_prompt_capacity=maximum_positions,
            packing_mode=config.cache.packing_mode,
        )
        knapsack_bytes = sum(
            ladder.level(level_indices[ladder.route]).active_bytes for ladder in ladders
        )
        storage_accounting = _exact_cache_storage_accounting(
            plan,
            tuple(sorted(retained_ids)),
            construction,
            external,
            knapsack_bytes=knapsack_bytes,
            packed_physical_bytes=_cache_payload_bytes(adapter, packed.snapshot),
        )
        artifacts[proposal.level_indices] = _HierarchicalExactArtifact(
            construction=construction,
            selection=selection,
            external_source_node_ids=external,
            retained_source_node_ids=tuple(sorted(retained_ids)),
            packed_physical_bytes=storage_accounting.constructed_physical_bytes,
            storage_accounting=storage_accounting,
        )
        return probe_isolated_packed_candidate(
            adapter,
            prefill.state,
            packed,
            candidate_id=proposal.proposal_id,
            teacher_forced_inputs=teacher_inputs,
            reference_logits=reference_logits,
        )

    tracker = ProbeBudgetTracker(
        selection_config.stable_head_hierarchical_maximum_candidate_evaluations,
        selection_config.stable_head_hierarchical_maximum_sequential_rounds,
    )

    def selection_is_route_feasible(selection: Any) -> bool:
        proposal = HierarchicalCapacityProposal(
            proposal_id="route-feasibility-only",
            kind=HierarchicalProposalKind.JOINT,
            level_indices=tuple(sorted(selection.level_index_by_route().items())),
            affected_routes=tuple(sorted(selection.level_index_by_route())),
        )
        return not proposal_route_capacity_violations(
            proposal,
            ladders,
            position_count_by_node,
            effective_position_limits,
        )

    decision = run_bounded_hierarchical_probe_loop(
        ladders,
        target_active_bytes=plan.active_budget_value,
        initial_partition_count=selection_config.stable_head_hierarchical_partition_count,
        maximum_refinement_depth=(
            selection_config.stable_head_hierarchical_maximum_refinement_depth
        ),
        maximum_rollback_rounds=(selection_config.stable_head_hierarchical_maximum_rollback_rounds),
        maximum_forward_kl=selection_config.stable_head_hierarchical_maximum_kl,
        maximum_absolute_logit_difference=(
            selection_config.stable_head_hierarchical_maximum_logit_difference
        ),
        required_steps=selection_config.stable_head_hierarchical_horizon,
        probe_budget=tracker,
        probe=probe,
        fixed_route_capacity_reallocation=True,
        reserved_reallocation_evaluations=(
            selection_config.stable_head_hierarchical_reserved_reallocation_evaluations
        ),
        reallocation_beam_width=(selection_config.stable_head_hierarchical_reallocation_beam_width),
        reallocation_maximum_combination_width=(
            selection_config.stable_head_hierarchical_reallocation_maximum_combination_width
        ),
        logit_feedback_restore_donor=(
            selection_config.stable_head_hierarchical_logit_feedback_restore_donor
        ),
        adaptive_no_pass_refinement=bool(
            selection_config.stable_head_hierarchical_logit_feedback_restore_donor
            and selection_config.stable_head_hierarchical_maximum_refinement_depth >= 3
        ),
        selection_feasible=selection_is_route_feasible,
        fullkv_fallback_selection=fullkv_fallback_selection,
    )
    selected_levels = tuple(
        (choice.route, choice.level_index) for choice in decision.selection.choices
    )
    selected_artifact = None if decision.used_fullkv_fallback else artifacts.get(selected_levels)
    if selected_artifact is None:
        artifact_ladders = unrestricted_ladders if decision.used_fullkv_fallback else ladders
        retained_ids = frozenset(
            node_id
            for ladder in artifact_ladders
            for node_id in ladder.level(
                decision.selection.level_index_by_route()[ladder.route]
            ).selected_node_ids
        )
        construction, selection, external = _progressive_exact_construction(
            plan,
            adapter.capabilities,
            probe_config,
            retained_ids,
            key_transform=key_transform,
            active_budget=probe_budget,
        )
        if decision.used_fullkv_fallback:
            # FullKV already has the common dense layout and no physical
            # padding. Do not materialize a second over-budget packed cache;
            # the caller rejects this fallback before final packing.
            packed_physical_bytes = construction.state.statistics.active_kv_bytes
        else:
            maximum_positions = max(
                layout.active_cache_length for layout in construction.active_layouts
            )
            packed = pack_runtime_cache(
                adapter,
                plan.full_state,
                construction.state,
                (),
                (),
                physical_prompt_capacity=maximum_positions,
                packing_mode=config.cache.packing_mode,
            )
            packed_physical_bytes = _cache_payload_bytes(adapter, packed.snapshot)
        selected_artifact = _HierarchicalExactArtifact(
            construction=construction,
            selection=selection,
            external_source_node_ids=external,
            retained_source_node_ids=tuple(sorted(retained_ids)),
            packed_physical_bytes=packed_physical_bytes,
        )
    construction = selected_artifact.construction
    selection = selected_artifact.selection
    external = selected_artifact.external_source_node_ids
    accounting = _exact_cache_storage_accounting(
        plan,
        selected_artifact.retained_source_node_ids,
        construction,
        external,
        knapsack_bytes=decision.selection.active_bytes,
        packed_physical_bytes=selected_artifact.packed_physical_bytes,
    )
    rejected_evaluation = next(
        (
            evaluation
            for evaluation in reversed(decision.evaluations)
            if not evaluation.passed
            and evaluation.proposal.kind
            in {
                HierarchicalProposalKind.JOINT,
                HierarchicalProposalKind.ROLLBACK,
                HierarchicalProposalKind.REALLOCATION,
            }
            and evaluation.proposal.level_indices in artifacts
        ),
        None,
    )
    rejected_artifact = (
        None
        if rejected_evaluation is None
        else artifacts[rejected_evaluation.proposal.level_indices]
    )
    rejected_accounting = (
        None if rejected_artifact is None else rejected_artifact.storage_accounting
    )
    if decision.used_fullkv_fallback and (
        rejected_evaluation is None or rejected_accounting is None
    ):
        raise HuggingFaceRuntimeError(
            "hierarchical FullKV fallback lost the rejected compressed proposal accounting"
        )
    if not accounting.logical_boundaries_aligned:
        decision_record = cast("JsonObject", asdict(decision))
        error = HuggingFaceRuntimeError(
            "hierarchical knapsack, source selection, and exact construction byte counts differ"
        )
        error.mosaickv_failure_trace_payload = _hierarchical_accounting_failure_trace_payload(
            decision_record,
            accounting,
            hard_active_kv_budget_bytes=plan.active_budget_value,
            elapsed_seconds=time.perf_counter() - hierarchical_started,
        )
        raise error
    reservation = _progressive_group_reservation(
        plan,
        config,
        effective_position_limits,
        _progressive_eviction_risks(plan),
    )
    selected_plan = replace(
        plan,
        effective_method=(
            f"{plan.effective_method}__hierarchical_fullkv_fallback"
            if decision.used_fullkv_fallback
            else f"{plan.effective_method}__hierarchical_logit_safe_eviction"
        ),
        selection=selection,
        construction=construction,
        external_exact_node_ids=external,
        external_exact_capacity_records=_progressive_capacity_records(
            plan,
            frozenset(external),
            reservation,
            allow_fullkv_capacity_expansion=decision.used_fullkv_fallback,
        ),
        hierarchical_storage_accounting=accounting,
        hierarchical_rejected_proposal_storage_accounting=rejected_accounting,
        hierarchical_rejected_proposal_id=(
            None if rejected_evaluation is None else rejected_evaluation.proposal.proposal_id
        ),
    )
    return selected_plan, decision


def _run_progressive_logit_safe_eviction(
    adapter: HuggingFaceMultimodalAdapter,
    prefill: PrefillOutput,
    plan: MosaicKVCompressionPlan,
    config: RunConfig,
    *,
    key_transform: RotaryPrototypeKeyTransform | None,
    position_limits: Mapping[tuple[int, int], int],
) -> tuple[MosaicKVCompressionPlan, ProgressiveLogitSafeEvictionDecision]:
    """Evict cumulatively from FullKV, accepting only FullKV-stable checkpoints."""

    if plan.method is not MosaicKVMethod.MOSAICKV_EXACT:
        raise HuggingFaceRuntimeError("progressive logit-safe eviction supports mosaickv_exact")
    full_state = plan.full_state
    fixed_ids = _progressive_fixed_source_ids(plan)
    eviction_risks = _progressive_eviction_risks(plan)
    reservation = _progressive_group_reservation(
        plan,
        config,
        position_limits,
        eviction_risks,
    )
    protected_ids = (
        {node_id for node_id, block in enumerate(full_state.blocks) if block.mandatory}
        | set(reservation.protected_node_ids)
        | set(fixed_ids)
    )
    current_external = set(range(len(full_state.blocks))) - set(fixed_ids)
    blocked_ids: set[int] = set()
    nodes_by_route: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node_id, block in enumerate(full_state.blocks):
        nodes_by_route[(block.layer, block.kv_head)].append(node_id)
    eviction_order = {
        route: tuple(
            sorted(
                (node_id for node_id in node_ids if node_id not in protected_ids),
                key=lambda node_id: (eviction_risks[node_id], node_id),
            )
        )
        for route, node_ids in nodes_by_route.items()
    }

    horizon = config.selection.stable_head_progressive_eviction_horizon
    reference_state = _clone_decode_state(adapter, prefill.state)
    reference_input = prefill.next_token_id.detach().clone()
    teacher_inputs: list[Any] = []
    reference_logits: list[Any] = []
    for _step in range(horizon):
        teacher_inputs.append(reference_input.detach().clone())
        output = adapter.decode_one_token(
            reference_input,
            reference_state,
            capture_queries=False,
            capture_attentions=False,
        )
        reference_state = output.state
        reference_logits.append(output.logits.detach().clone())
        reference_input = output.next_token_id.detach().clone()

    full_logical_bytes = sum(block.byte_size for block in full_state.blocks)
    probe_budget = SelectionBudget(full_logical_bytes, BudgetUnit.BYTES)
    probe_config = replace(
        config,
        cache=replace(
            config.cache,
            budget_value=full_logical_bytes,
            total_storage_budget_bytes=full_logical_bytes,
        ),
    )
    checkpoints: list[ProgressiveEvictionCheckpoint] = []
    pending: deque[tuple[int, ...]] = deque()
    batch_blocks = config.selection.stable_head_progressive_eviction_initial_batch_blocks
    maximum_probes = config.selection.stable_head_progressive_eviction_maximum_probes
    last_accepted_evaluation: ShortHorizonLogitCandidateEvaluation | None = None

    def route_positions(external_ids: set[int], route: tuple[int, int]) -> int:
        return sum(
            full_state.blocks[node_id].position_count
            for node_id in nodes_by_route[route]
            if node_id in external_ids or node_id in fixed_ids
        )

    def target_reached(external_ids: set[int]) -> bool:
        return all(
            route_positions(external_ids, route) <= limit
            for route, limit in position_limits.items()
        )

    def next_proposal() -> tuple[int, ...]:
        while pending:
            proposal = tuple(
                node_id
                for node_id in pending.popleft()
                if node_id in current_external
                and route_positions(
                    current_external,
                    (
                        full_state.blocks[node_id].layer,
                        full_state.blocks[node_id].kv_head,
                    ),
                )
                > position_limits[
                    (
                        full_state.blocks[node_id].layer,
                        full_state.blocks[node_id].kv_head,
                    )
                ]
            )
            if proposal:
                return proposal
        proposal_ids: list[int] = []
        for route, limit in sorted(position_limits.items()):
            excess = route_positions(current_external, route) - limit
            if excess <= 0:
                continue
            removed_positions = 0
            for node_id in eviction_order.get(route, ()):
                if node_id not in current_external or node_id in blocked_ids:
                    continue
                proposal_ids.append(node_id)
                removed_positions += full_state.blocks[node_id].position_count
                if removed_positions >= min(
                    excess,
                    batch_blocks * full_state.block_size,
                ):
                    break
        return tuple(sorted(set(proposal_ids)))

    while len(checkpoints) < maximum_probes and not target_reached(current_external):
        proposal = next_proposal()
        if not proposal:
            break
        candidate_external = current_external - set(proposal)
        construction, _probe_selection, _probe_external = _progressive_exact_construction(
            plan,
            adapter.capabilities,
            probe_config,
            fixed_ids | candidate_external,
            key_transform=key_transform,
            active_budget=probe_budget,
        )
        maximum_positions = max(
            layout.active_cache_length for layout in construction.active_layouts
        )
        packed = pack_runtime_cache(
            adapter,
            full_state,
            construction.state,
            (),
            (),
            physical_prompt_capacity=maximum_positions,
            packing_mode=config.cache.packing_mode,
        )
        candidate_state = _state_from_packed(adapter, prefill.state, packed)
        divergences: list[float] = []
        differences: list[float] = []
        matches: list[bool] = []
        for teacher_input, full_logits in zip(teacher_inputs, reference_logits, strict=True):
            output = adapter.decode_one_token(
                teacher_input,
                candidate_state,
                capture_queries=False,
                capture_attentions=False,
            )
            candidate_state = output.state
            divergences.append(_logit_kl(full_logits, output.logits))
            differences.append(
                float(
                    _torch()
                    .max(
                        _torch().abs(full_logits.detach().float() - output.logits.detach().float())
                    )
                    .item()
                )
            )
            matches.append(
                int(full_logits.detach().argmax(dim=-1).reshape(-1)[0].item())
                == int(output.logits.detach().argmax(dim=-1).reshape(-1)[0].item())
            )
        evaluation = _short_horizon_candidate_evaluation(
            f"progressive_probe_{len(checkpoints)}",
            kl_values=divergences,
            maximum_logit_differences=differences,
            greedy_token_matches=matches,
            maximum_kl_threshold=config.selection.stable_head_progressive_eviction_maximum_kl,
            maximum_logit_difference_threshold=(
                config.selection.stable_head_progressive_eviction_maximum_logit_difference
            ),
        )
        logical_bytes = sum(
            full_state.blocks[node_id].byte_size
            for node_id in fixed_ids | frozenset(candidate_external)
        )
        checkpoints.append(
            ProgressiveEvictionCheckpoint(
                probe_index=len(checkpoints),
                proposed_removed_blocks=len(proposal),
                accepted=evaluation.passed,
                greedy_token_match=evaluation.greedy_token_match,
                maximum_kl=evaluation.maximum_kl,
                maximum_logit_difference=evaluation.maximum_logit_difference,
                maximum_active_positions=maximum_positions,
                logical_active_bytes=logical_bytes,
            )
        )
        if evaluation.passed:
            current_external = candidate_external
            last_accepted_evaluation = evaluation
            pending.clear()
        elif len(proposal) > 1:
            midpoint = len(proposal) // 2
            pending.appendleft(proposal[midpoint:])
            pending.appendleft(proposal[:midpoint])
        else:
            blocked_ids.add(proposal[0])

    reached = target_reached(current_external)
    selected_plan = plan
    final_logical_bytes = plan.construction.state.statistics.active_kv_bytes
    if reached:
        final_construction, final_selection, final_external = _progressive_exact_construction(
            plan,
            adapter.capabilities,
            config,
            fixed_ids | current_external,
            key_transform=key_transform,
            active_budget=SelectionBudget(
                plan.active_budget_value,
                config.cache.budget_unit,
            ),
        )
        final_logical_bytes = final_construction.state.statistics.active_kv_bytes
        selected_plan = replace(
            plan,
            effective_method=f"{plan.effective_method}__progressive_logit_safe_eviction",
            selection=final_selection,
            construction=final_construction,
            external_exact_node_ids=final_external,
            external_exact_capacity_records=_progressive_capacity_records(
                plan,
                frozenset(current_external),
                reservation,
            ),
        )
    protected_groups = tuple(
        group for groups in reservation.groups_by_route.values() for group in groups
    )
    protected_bytes = sum(reservation.protected_bytes_by_route.values())
    protected_target = sum(reservation.target_bytes_by_route.values())
    selected_passed = bool(reached and last_accepted_evaluation is not None)
    return selected_plan, ProgressiveLogitSafeEvictionDecision(
        enabled=True,
        started_from_fullkv=True,
        target_reached=reached,
        selected_candidate_passed=selected_passed,
        used_target_fallback=not reached,
        horizon=horizon,
        maximum_kl_threshold=config.selection.stable_head_progressive_eviction_maximum_kl,
        maximum_logit_difference_threshold=(
            config.selection.stable_head_progressive_eviction_maximum_logit_difference
        ),
        maximum_probes=maximum_probes,
        probes_used=len(checkpoints),
        accepted_checkpoints=sum(item.accepted for item in checkpoints),
        rejected_checkpoints=sum(not item.accepted for item in checkpoints),
        initial_fullkv_logical_bytes=full_logical_bytes,
        target_active_bytes=plan.active_budget_value,
        final_logical_active_bytes=final_logical_bytes,
        protected_group_candidate_count=sum(reservation.candidate_count_by_route.values()),
        protected_group_count=len(protected_groups),
        protected_group_block_count=sum(len(group) for group in protected_groups),
        protected_group_bytes=protected_bytes,
        protected_group_target_bytes=protected_target,
        protected_group_score=sum(reservation.protected_score_by_route.values()),
        checkpoints=tuple(checkpoints),
    )


def _run_fidelity_diagnostic(
    adapter: HuggingFaceMultimodalAdapter,
    prefill: PrefillOutput,
    candidate_state: DecodeState,
    *,
    max_new_tokens: int,
    prefixkv_plan: PrefixKVCompressionPlan | None = None,
) -> FidelityDiagnostic:
    """Decode isolated greedy FullKV/candidate trajectories without a second prefill."""

    torch = _torch()
    reference_state = _clone_decode_state(adapter, prefill.state)
    compressed_state = _clone_decode_state(adapter, candidate_state)
    reference_token = prefill.next_token_id.detach().clone()
    candidate_token = prefill.next_token_id.detach().clone()
    reference_tokens = [reference_token]
    candidate_tokens = [candidate_token]
    divergences: list[float] = []
    attention_errors: list[float] = []
    attention_errors_by_head: dict[tuple[int, int], list[float]] = {}
    kv_heads_by_layer = _kv_head_counts(adapter, candidate_state)
    for _step in range(1, max_new_tokens):
        reference_output = adapter.decode_one_token(
            reference_token,
            reference_state,
            capture_queries=False,
            capture_attentions=True,
        )
        candidate_output = adapter.decode_one_token(
            candidate_token,
            compressed_state,
            capture_queries=False,
            capture_attentions=True,
        )
        divergences.append(_logit_kl(reference_output.logits, candidate_output.logits))
        reference_attention_outputs = _attention_outputs(
            adapter,
            reference_output.attention_weights,
            reference_output.state,
        )
        candidate_attention_outputs = _attention_outputs(
            adapter,
            candidate_output.attention_weights,
            candidate_output.state,
        )
        attention_errors.append(
            _attention_output_error(reference_attention_outputs, candidate_attention_outputs)
        )
        for layer, kv_head, error in _attention_output_error_by_kv_head(
            reference_attention_outputs,
            candidate_attention_outputs,
            kv_heads_by_layer,
        ):
            attention_errors_by_head.setdefault((layer, kv_head), []).append(error)
        reference_state = reference_output.state
        compressed_state = candidate_output.state
        if prefixkv_plan is not None:
            compressed_state, _ = maintain_prefixkv_decode_cache(
                adapter, compressed_state, prefixkv_plan, step_index=_step
            )
        reference_token = reference_output.next_token_id
        candidate_token = candidate_output.next_token_id
        reference_tokens.append(reference_token)
        candidate_tokens.append(candidate_token)
    reference_ids = torch.cat(reference_tokens, dim=-1).detach().reshape(-1)
    candidate_ids = torch.cat(candidate_tokens, dim=-1).detach().reshape(-1)
    agreement = float((reference_ids == candidate_ids).float().mean().item())
    return FidelityDiagnostic(
        token_agreement=agreement,
        sequence_agreement=bool(torch.equal(reference_ids, candidate_ids)),
        logit_kl=float(np.mean(divergences)) if divergences else 0.0,
        attention_output_error=float(np.mean(attention_errors)) if attention_errors else 0.0,
        attention_output_error_by_kv_head=tuple(
            (layer, head, float(np.mean(values)))
            for (layer, head), values in sorted(attention_errors_by_head.items())
        ),
    )


def _prototype_attention_mass(
    attention_weights: Sequence[Any],
    prototype_slots: Sequence[tuple[int, int, int, int]],
    validity_masks: Sequence[Any],
) -> dict[int, float]:
    if attention_weights and all(
        value.__class__.__module__.startswith("torch") for value in attention_weights
    ):
        import torch

        total_query_heads = sum(int(value.shape[1]) for value in attention_weights)
        if total_query_heads <= 0:
            raise HuggingFaceRuntimeError("decode attention exposes no query heads")
        grouped: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for prototype_id, layer, kv_head, slot in prototype_slots:
            grouped[(layer, kv_head)].append((prototype_id, slot))
        identifiers: list[int] = []
        device_masses: list[Any] = []
        for (layer, kv_head), entries in sorted(grouped.items()):
            attention = attention_weights[layer]
            query_heads = int(attention.shape[1])
            kv_heads = int(validity_masks[layer].shape[0])
            if query_heads % kv_heads:
                raise HuggingFaceRuntimeError("prototype attention has incompatible head groups")
            group = query_heads // kv_heads
            start = kv_head * group
            slots = torch.as_tensor(
                tuple(slot for _prototype_id, slot in entries),
                dtype=torch.long,
                device=attention.device,
            )
            # Reduce grouped query heads and query positions on device. Only
            # the compact prototype vector crosses the host boundary once.
            values = attention[0, start : start + group].index_select(-1, slots)
            normalized = values.sum(dim=(0, 1)) / (total_query_heads * int(attention.shape[-2]))
            identifiers.extend(prototype_id for prototype_id, _slot in entries)
            device_masses.append(normalized)
        if not device_masses:
            return {}
        host = torch.cat(device_masses).detach().float().cpu().tolist()
        device_result = dict(zip(identifiers, (float(value) for value in host), strict=True))
        total = sum(device_result.values())
        if total > 1 + 1e-7:
            raise HuggingFaceRuntimeError("normalized prototype attention mass exceeds one")
        return device_result

    arrays = tuple(_to_numpy(value) for value in attention_weights)
    total_query_heads = sum(int(array.shape[1]) for array in arrays)
    if total_query_heads <= 0:
        raise HuggingFaceRuntimeError("decode attention exposes no query heads")
    masses: dict[int, float] = {}
    for prototype_id, layer, kv_head, slot in prototype_slots:
        attention = arrays[layer]
        query_heads = int(attention.shape[1])
        kv_heads = int(validity_masks[layer].shape[0])
        if query_heads % kv_heads:
            raise HuggingFaceRuntimeError("prototype attention has incompatible head groups")
        group = query_heads // kv_heads
        start = kv_head * group
        value = float(attention[0, start : start + group, :, slot].sum())
        masses[prototype_id] = value / (total_query_heads * int(attention.shape[-2]))
    total = sum(masses.values())
    if total > 1 + 1e-7:
        raise HuggingFaceRuntimeError("normalized prototype attention mass exceeds one")
    return masses


def _decode_text(processor: Any, token_ids: Any) -> str:
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None:
        decoder = getattr(getattr(processor, "tokenizer", processor), "batch_decode", None)
    if decoder is None:
        raise HuggingFaceRuntimeError("processor/tokenizer does not provide batch_decode")
    answers = decoder(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answer = str(answers[0]) if answers else ""
    if not answer.strip():
        fallback = decoder(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        answer = str(fallback[0]) if fallback else ""
    if not answer.strip():
        raise HuggingFaceRuntimeError("decoded output is empty")
    return answer


def _trace_catalog(
    records: Iterable[Mapping[str, object]],
    *,
    compact: bool,
) -> JsonValue:
    """Serialize a catalog or compute its identical compact digest in one pass."""

    if not compact:
        return [cast("JsonObject", dict(record)) for record in records]
    digest = hashlib.sha256()
    count = 0
    for record in records:
        payload = cast("JsonObject", dict(record))
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
        count += 1
    if count == 0:
        return []
    return {
        "storage": "omitted_non_diagnostic",
        "record_count": count,
        "sha256": digest.hexdigest(),
    }


def _source_selection_records(
    plan: MosaicKVCompressionPlan,
    *,
    include_cost_rate: bool,
) -> Iterable[JsonObject]:
    """Expand compact planner decisions into the immutable source-node space.

    Expansion is intentionally invoked only by the deferred trace builder, so
    stable-head bypass bookkeeping cannot inflate online compression latency.
    Every omitted stable-head source block remains explicitly auditable.
    """

    decisions_by_source = {
        plan.source_node_id(decision.node_id): decision for decision in plan.selection.decisions
    }
    external_rank = {
        node_id: len(plan.selection.selection_order) + rank
        for rank, node_id in enumerate(plan.external_exact_node_ids)
    }
    for source_node_id, block in enumerate(plan.full_state.blocks):
        decision = decisions_by_source.get(source_node_id)
        if source_node_id in external_rank:
            external_record: JsonObject = {
                "node_id": source_node_id,
                "selected": True,
                "selection_rank": external_rank[source_node_id],
                "marginal_gain": 0.0,
                "cost": plan.selection.budget.cost(block),
                "reason": "stable_head_prompt_exact",
            }
            if include_cost_rate:
                external_record["marginal_gain_per_cost"] = 0.0
            yield external_record
            continue
        if decision is None:
            bypass_record: JsonObject = {
                "node_id": source_node_id,
                "selected": False,
                "selection_rank": None,
                "marginal_gain": 0.0,
                "cost": plan.selection.budget.cost(block),
                "reason": "routed_head_bypass_not_selected",
            }
            if include_cost_rate:
                bypass_record["marginal_gain_per_cost"] = 0.0
            yield bypass_record
            continue
        decision_record: JsonObject = {
            "node_id": source_node_id,
            "selected": decision.selected,
            "selection_rank": decision.selection_rank,
            "marginal_gain": decision.marginal_gain,
            "cost": decision.cost,
            "reason": decision.reason.value,
        }
        if include_cost_rate:
            decision_record["marginal_gain_per_cost"] = decision.marginal_gain_per_cost
        yield decision_record


def _selection_decision_evidence(
    plan: MosaicKVCompressionPlan | SelectionResult,
) -> JsonObject:
    """Retain lossless per-node selection evidence for explicit parity runs.

    Normal evaluation traces keep only compact catalog hashes.  A parity run
    can opt into this payload so exact semantic decisions can be compared
    separately from floating-point marginal gains without rerunning the
    compressor or weakening identity checks.
    """

    if isinstance(plan, MosaicKVCompressionPlan):
        records: list[JsonValue] = list(_source_selection_records(plan, include_cost_rate=True))
    else:
        # Preserve the focused helper contract used by unit tests and callers
        # that already own a source-space SelectionResult.
        records = [
            cast(
                "JsonObject",
                {
                    "node_id": decision.node_id,
                    "selected": decision.selected,
                    "selection_rank": decision.selection_rank,
                    "marginal_gain": decision.marginal_gain,
                    "marginal_gain_per_cost": decision.marginal_gain_per_cost,
                    "cost": decision.cost,
                    "reason": decision.reason.value,
                },
            )
            for decision in plan.decisions
        ]
    return {
        "schema_version": 1,
        "record_count": len(records),
        "records": records,
    }


def _forecast_trace(
    forecast: QueryForecast | None,
    *,
    compact_catalogs: bool = False,
) -> JsonObject:
    if forecast is None:
        return {
            "status": "disabled_by_declared_ablation",
            "attention_source": "uniform_per_source_position",
        }

    def head_records() -> Iterable[JsonObject]:
        for layer in forecast.layers:
            for head in layer:
                centroids = _to_numpy(head.normalized_centroids)
                weights = _to_numpy(head.forecast_weights)
                yield {
                    "layer": head.provenance.layer,
                    "kv_head": head.provenance.kv_head,
                    "prompt_sample_count": head.provenance.prompt_sample_count,
                    "draft_sample_count": head.provenance.draft_sample_count,
                    "centroid_shape": list(centroids.shape),
                    "centroid_norms": np.linalg.norm(centroids, axis=-1).tolist(),
                    "forecast_weights": weights.tolist(),
                }

    return cast(
        "JsonObject",
        {
            "provenance": asdict(forecast.provenance),
            "timing": asdict(forecast.timing),
            "heads": _trace_catalog(head_records(), compact=compact_catalogs),
        },
    )


def _utility_trace(plan: MosaicKVCompressionPlan) -> JsonObject:
    """Summarize commensurate utility signals without emitting another node catalog."""

    selected = set(plan.selection.selected_node_ids)

    def summary(values: Sequence[float]) -> JsonObject:
        array = np.asarray(values, dtype=np.float64)
        return {
            "minimum": float(array.min()),
            "mean": float(array.mean()),
            "maximum": float(array.max()),
        }

    blocks = plan.utilities.blocks
    local = tuple(block.local_utility for block in blocks)
    return {
        "normalization": plan.utilities.weights.normalization,
        "weights": cast("JsonObject", asdict(plan.utilities.weights)),
        "attention_probability": summary(
            tuple(block.forecast_attention_probability for block in blocks)
        ),
        "value_contribution": summary(tuple(block.value_contribution for block in blocks)),
        "uniqueness": summary(tuple(block.uniqueness for block in blocks)),
        "local_utility": summary(local),
        "negative_local_utility_fraction": sum(value < 0 for value in local) / len(local),
        "selected_mean_local_utility": float(
            np.mean([blocks[node_id].local_utility for node_id in sorted(selected)])
        ),
        "unselected_mean_local_utility": (
            float(
                np.mean([block.local_utility for block in blocks if block.node_id not in selected])
            )
            if len(selected) < len(blocks)
            else None
        ),
    }


def _mosaickv_request_wide_projection(plan: MosaicKVCompressionPlan) -> JsonObject:
    """Project exact source membership onto a serving-engine token-block table."""

    utility_by_source = {
        plan.source_node_id(graph_node_id): utility.local_utility
        for graph_node_id, utility in enumerate(plan.utilities.blocks)
    }
    exact_source_ids = tuple(sorted(set(plan.construction.exact_node_ids)))
    exact_source_set = frozenset(exact_source_ids)
    selected = tuple(
        (plan.full_state.blocks[source_id], utility_by_source.get(source_id, 0.0))
        for source_id in exact_source_ids
    )
    fallback = tuple(
        (plan.full_state.blocks[source_id], score)
        for source_id, score in sorted(utility_by_source.items())
        if source_id not in exact_source_set
    )
    ratio = plan.shared_active_budget_value / plan.source_budget_value
    return project_common_selector_to_request_blocks(
        plan.full_state,
        selected,
        method=plan.method.value,
        retention_ratio=min(1.0, ratio),
        fallback_candidates=fallback,
    ).to_json_object()


def _baseline_request_wide_projection(plan: BaselineCompressionPlan) -> JsonObject:
    """Project a simple exact baseline onto the same serving block boundary."""

    score_by_node = {decision.node_id: decision.score for decision in plan.selection.decisions}
    selected = tuple(
        (block, score_by_node[node_id])
        for node_id, block in zip(
            plan.selection.selected_node_ids,
            plan.selection.selected_blocks,
            strict=True,
        )
    )
    selected_node_ids = frozenset(plan.selection.selected_node_ids)
    fallback = tuple(
        (plan.full_state.blocks[decision.node_id], decision.score)
        for decision in plan.selection.decisions
        if decision.node_id not in selected_node_ids
    )
    ratio = plan.active_budget_value / plan.source_budget_value
    return project_common_selector_to_request_blocks(
        plan.full_state,
        selected,
        method=plan.method.value,
        retention_ratio=min(1.0, ratio),
        fallback_candidates=fallback,
    ).to_json_object()


def _token_block_lookup(full_state: FullKVState) -> dict[tuple[int, int, int], KVBlockDescriptor]:
    """Index token-sized source blocks for published-baseline projections."""

    lookup: dict[tuple[int, int, int], KVBlockDescriptor] = {}
    for block in full_state.blocks:
        if block.position_count != 1 or len(block.physical_cache_indices) != 1:
            raise HuggingFaceRuntimeError(
                "published-baseline vLLM projection requires token-sized HF source blocks"
            )
        identity = (block.layer, block.kv_head, block.physical_cache_indices[0])
        if identity in lookup:
            raise HuggingFaceRuntimeError(
                "published-baseline source cache contains a duplicate route/position"
            )
        lookup[identity] = block
    return lookup


def _lookm_request_wide_projection(plan: LookMCompressionPlan) -> JsonObject:
    """Project LOOK-M's paper-selected pivots without pretending to inject merged KV."""

    lookup = _token_block_lookup(plan.full_state)
    selected: list[tuple[KVBlockDescriptor, float]] = []
    fallback: list[tuple[KVBlockDescriptor, float]] = []
    for head in plan.heads:
        selected_positions = frozenset(head.selected_physical_positions)
        for position, score in enumerate(head.cumulative_attention_scores):
            item = (lookup[(head.layer, head.kv_head, position)], float(score))
            (selected if position in selected_positions else fallback).append(item)
    return project_common_selector_to_request_blocks(
        plan.full_state,
        tuple(selected),
        method=MosaicKVMethod.LOOKM_REIMPL.value,
        retention_ratio=plan.active_slots / plan.source_blocks,
        fallback_candidates=tuple(fallback),
    ).to_json_object()


def _prefixkv_request_wide_projection(plan: PrefixKVCompressionPlan) -> JsonObject:
    """Project PrefixKV layer-wise exact membership to shared vLLM blocks."""

    lookup = _token_block_lookup(plan.full_state)
    selected: list[tuple[KVBlockDescriptor, float]] = []
    fallback: list[tuple[KVBlockDescriptor, float]] = []
    for layer in plan.layers:
        selected_positions = frozenset(layer.selected_physical_positions)
        # PrefixKV stores a hash and aggregate rather than its full score vector.
        # Membership votes therefore carry the paper decision; this finite mean
        # is only a deterministic cross-layer tie breaker.
        mean_score = layer.attention_score_sum / layer.source_positions
        for head in range(plan.full_state.layers[layer.layer].kv_heads):
            for position in range(layer.source_positions):
                item = (lookup[(layer.layer, head, position)], mean_score)
                (selected if position in selected_positions else fallback).append(item)
    return project_common_selector_to_request_blocks(
        plan.full_state,
        tuple(selected),
        method=MosaicKVMethod.PREFIXKV_REIMPL.value,
        retention_ratio=plan.active_slots / plan.source_slots,
        fallback_candidates=tuple(fallback),
    ).to_json_object()


def _vl_cache_request_wide_projection(plan: VLCacheCompressionPlan) -> JsonObject:
    """Project VL-Cache per-head paper selections to shared vLLM blocks."""

    lookup = _token_block_lookup(plan.full_state)
    selected: list[tuple[KVBlockDescriptor, float]] = []
    fallback: list[tuple[KVBlockDescriptor, float]] = []
    for layer in plan.layers:
        for head in layer.heads:
            selected_positions = frozenset(head.selected_physical_positions)
            # As with PrefixKV, the retained membership is exact while the
            # persisted paper trace intentionally keeps only score provenance.
            mean_score = head.token_score_sum / head.source_positions
            for position in range(head.source_positions):
                item = (lookup[(head.layer, head.kv_head, position)], mean_score)
                (selected if position in selected_positions else fallback).append(item)
    return project_common_selector_to_request_blocks(
        plan.full_state,
        tuple(selected),
        method=MosaicKVMethod.VL_CACHE_REIMPL.value,
        retention_ratio=plan.active_slots / plan.source_slots,
        fallback_candidates=tuple(fallback),
    ).to_json_object()


def _prototype_diagnostics_by_head(plan: MosaicKVCompressionPlan) -> list[JsonObject]:
    grouped: dict[tuple[int, int], list[PrototypeRecord]] = {}
    for record in plan.construction.prototypes:
        grouped.setdefault((record.layer, record.kv_head), []).append(record)
    result: list[JsonObject] = []
    for (layer, head), records in sorted(grouped.items()):
        key_dispersion = tuple(record.diagnostics.key_dispersion for record in records)
        value_dispersion = tuple(record.diagnostics.value_dispersion for record in records)
        result.append(
            {
                "layer": layer,
                "kv_head": head,
                "prototype_count": len(records),
                "mean_key_dispersion": float(np.mean(key_dispersion)),
                "maximum_key_dispersion": max(key_dispersion),
                "mean_value_dispersion": float(np.mean(value_dispersion)),
                "maximum_value_dispersion": max(value_dispersion),
                "maximum_position_span": max(
                    record.diagnostics.position_span for record in records
                ),
                "maximum_cluster_size": max(record.diagnostics.cluster_size for record in records),
                "mean_represented_source_positions": float(
                    np.mean([record.represented_source_positions for record in records])
                ),
                "mean_attention_log_bias": float(
                    np.mean([record.attention_log_bias for record in records])
                ),
                "attention_log_bias_scale": records[0].attention_log_bias_scale,
                "attention_bias_basis": records[0].attention_bias_basis,
                "pooling_mode": records[0].pooling_mode,
                "anchor_position_strategy": records[0].anchor_position_strategy,
                "maximum_member_reconstruction_error": max(
                    member.reconstruction_error for record in records for member in record.members
                ),
            }
        )
    return result


def _external_stable_capacity_trace(
    records: Sequence[HeadLocalCapacityRecord],
) -> JsonObject:
    """Serialize stable-head physical-capacity use for success or failure traces."""

    return {
        "route_count": len(records),
        "requested_positions": sum(record.position_limit for record in records),
        "selected_positions": sum(record.selected_positions for record in records),
        "shortfall_positions": sum(record.shortfall_positions for record in records),
        "selected_bytes": sum(record.selected_bytes for record in records),
        "selected_visual_positions": sum(record.selected_visual_positions for record in records),
        "target_visual_positions": sum(record.target_visual_positions for record in records),
        "routes_missing_visual_target": sum(
            record.selected_visual_positions < record.target_visual_positions for record in records
        ),
        "adaptive_swap_count": sum(record.adaptive_swap_count for record in records),
        "baseline_text_prompt_mass": math.fsum(
            record.baseline_text_prompt_mass for record in records
        ),
        "selected_text_prompt_mass": math.fsum(
            record.selected_text_prompt_mass for record in records
        ),
        "normalized_visual_evidence_gain": math.fsum(
            record.normalized_visual_evidence_gain for record in records
        ),
        "normalized_text_prompt_mass_loss": math.fsum(
            record.normalized_text_prompt_mass_loss for record in records
        ),
        "normalized_forecast_visual_gain": math.fsum(
            record.normalized_forecast_visual_gain for record in records
        ),
        "normalized_forecast_text_loss": math.fsum(
            record.normalized_forecast_text_loss for record in records
        ),
        "adaptive_swap_candidate_count": sum(
            record.adaptive_swap_candidate_count for record in records
        ),
        "adaptive_swap_risk_rejected_count": sum(
            record.adaptive_swap_risk_rejected_count for record in records
        ),
        "future_rerank_count": sum(record.future_rerank_count for record in records),
        "baseline_selected_prompt_mass": math.fsum(
            record.baseline_selected_prompt_mass for record in records
        ),
        "selected_prompt_mass_after_future_rerank": math.fsum(
            record.selected_prompt_mass_after_future_rerank for record in records
        ),
        "normalized_future_query_gain": math.fsum(
            record.normalized_future_query_gain for record in records
        ),
        "normalized_future_prompt_mass_loss": math.fsum(
            record.normalized_future_prompt_mass_loss for record in records
        ),
        "future_rerank_candidate_count": sum(
            record.future_rerank_candidate_count for record in records
        ),
        "future_rerank_rejected_count": sum(
            record.future_rerank_rejected_count for record in records
        ),
        "future_rerank_risk_rejected_count": sum(
            record.future_rerank_risk_rejected_count for record in records
        ),
        "query_coverage_rerank_count": sum(
            record.query_coverage_rerank_count for record in records
        ),
        "baseline_query_coverage": math.fsum(record.baseline_query_coverage for record in records),
        "selected_query_coverage": math.fsum(record.selected_query_coverage for record in records),
        "normalized_query_coverage_gain": math.fsum(
            record.normalized_query_coverage_gain for record in records
        ),
        "query_coverage_baseline_prompt_mass": math.fsum(
            record.query_coverage_baseline_prompt_mass for record in records
        ),
        "query_coverage_selected_prompt_mass": math.fsum(
            record.query_coverage_selected_prompt_mass for record in records
        ),
        "query_coverage_normalized_prompt_loss": math.fsum(
            record.query_coverage_normalized_prompt_loss for record in records
        ),
        "query_coverage_candidate_count": sum(
            record.query_coverage_candidate_count for record in records
        ),
        "query_coverage_rejected_count": sum(
            record.query_coverage_rejected_count for record in records
        ),
        "query_coverage_initial_reserved_count": sum(
            record.query_coverage_initial_reserved_count for record in records
        ),
        "query_coverage_initial_changed_count": sum(
            record.query_coverage_initial_changed_count for record in records
        ),
        "query_coverage_initial_baseline": math.fsum(
            record.query_coverage_initial_baseline for record in records
        ),
        "query_coverage_initial_selected": math.fsum(
            record.query_coverage_initial_selected for record in records
        ),
        "query_coverage_initial_normalized_gain": math.fsum(
            record.query_coverage_initial_normalized_gain for record in records
        ),
        "query_coverage_initial_baseline_prompt_mass": math.fsum(
            record.query_coverage_initial_baseline_prompt_mass for record in records
        ),
        "query_coverage_initial_selected_prompt_mass": math.fsum(
            record.query_coverage_initial_selected_prompt_mass for record in records
        ),
        "query_coverage_initial_normalized_prompt_loss": math.fsum(
            record.query_coverage_initial_normalized_prompt_loss for record in records
        ),
        "query_coverage_initial_candidate_count": sum(
            record.query_coverage_initial_candidate_count for record in records
        ),
        "query_coverage_initial_reserved_bytes": sum(
            record.query_coverage_initial_reserved_bytes for record in records
        ),
        "query_coverage_initial_allocated_bytes": sum(
            record.query_coverage_initial_allocated_bytes for record in records
        ),
        "query_coverage_initial_cross_head_risk": math.fsum(
            record.query_coverage_initial_cross_head_risk for record in records
        ),
        "query_coverage_initial_proposed_normalized_gain": math.fsum(
            record.query_coverage_initial_proposed_normalized_gain for record in records
        ),
        "query_coverage_initial_proposed_normalized_prompt_loss": math.fsum(
            record.query_coverage_initial_proposed_normalized_prompt_loss for record in records
        ),
        "query_coverage_initial_proposed_output_impact_risk": math.fsum(
            record.query_coverage_initial_proposed_output_impact_risk for record in records
        ),
        "query_coverage_initial_output_impact_donor_count": sum(
            record.query_coverage_initial_output_impact_donor_count for record in records
        ),
        "query_coverage_initial_output_impact_protected_count": sum(
            record.query_coverage_initial_output_impact_protected_count for record in records
        ),
        "query_coverage_initial_output_impact_protected_bytes": sum(
            record.query_coverage_initial_output_impact_protected_bytes for record in records
        ),
        "query_coverage_initial_output_impact_protected_score": math.fsum(
            record.query_coverage_initial_output_impact_protected_score for record in records
        ),
        "query_coverage_initial_layout_group_candidate_count": sum(
            record.query_coverage_initial_layout_group_candidate_count for record in records
        ),
        "query_coverage_initial_layout_group_protected_group_count": sum(
            record.query_coverage_initial_layout_group_protected_group_count for record in records
        ),
        "query_coverage_initial_layout_group_protected_block_count": sum(
            record.query_coverage_initial_layout_group_protected_block_count for record in records
        ),
        "query_coverage_initial_layout_group_protected_bytes": sum(
            record.query_coverage_initial_layout_group_protected_bytes for record in records
        ),
        "query_coverage_initial_layout_group_protected_score": math.fsum(
            record.query_coverage_initial_layout_group_protected_score for record in records
        ),
        "query_coverage_initial_layout_group_target_bytes": max(
            (record.query_coverage_initial_layout_group_target_bytes for record in records),
            default=0,
        ),
        "query_coverage_initial_layout_group_accepted_candidate_count": sum(
            record.query_coverage_initial_layout_group_accepted_candidate_count
            for record in records
        ),
        "query_coverage_initial_layout_group_risk_rejected_count": sum(
            record.query_coverage_initial_layout_group_risk_rejected_count for record in records
        ),
        "query_coverage_initial_layout_group_pareto_candidate_count": sum(
            record.query_coverage_initial_layout_group_pareto_candidate_count for record in records
        ),
        "query_coverage_initial_layout_group_knapsack_selected_count": sum(
            record.query_coverage_initial_layout_group_knapsack_selected_count for record in records
        ),
        "query_coverage_initial_layout_group_joint_donor_candidate_count": sum(
            record.query_coverage_initial_layout_group_joint_donor_candidate_count
            for record in records
        ),
        "query_coverage_initial_layout_group_normalized_query_gain": math.fsum(
            record.query_coverage_initial_layout_group_normalized_query_gain for record in records
        ),
        "query_coverage_initial_layout_group_normalized_prompt_risk": math.fsum(
            record.query_coverage_initial_layout_group_normalized_prompt_risk for record in records
        ),
        "query_coverage_initial_layout_group_output_impact_risk": math.fsum(
            record.query_coverage_initial_layout_group_output_impact_risk for record in records
        ),
        "query_coverage_initial_stability_rejected_route_count": sum(
            record.query_coverage_initial_stability_rejected for record in records
        ),
        "query_coverage_initial_global_allocation_route_count": sum(
            record.query_coverage_initial_global_allocation for record in records
        ),
        "query_coverage_initial_allocated_route_count": sum(
            record.query_coverage_initial_allocated_bytes > 0 for record in records
        ),
        "forecast_divergence_eligible_route_count": sum(
            record.forecast_divergence_eligible for record in records
        ),
        "prompt_forecast_total_variation_min": (
            min(record.prompt_forecast_total_variation for record in records) if records else 0.0
        ),
        "prompt_forecast_total_variation_mean": (
            statistics.fmean(record.prompt_forecast_total_variation for record in records)
            if records
            else 0.0
        ),
        "prompt_forecast_total_variation_max": (
            max(record.prompt_forecast_total_variation for record in records) if records else 0.0
        ),
        "routes_with_adaptive_swaps": [
            cast("JsonObject", asdict(record)) for record in records if record.adaptive_swap_count
        ],
        "routes_with_risk_rejections": [
            cast("JsonObject", asdict(record))
            for record in records
            if record.adaptive_swap_risk_rejected_count
        ],
        "routes_with_future_reranking": [
            cast("JsonObject", asdict(record)) for record in records if record.future_rerank_count
        ],
        "routes_with_future_rerank_rejections": [
            cast("JsonObject", asdict(record))
            for record in records
            if record.future_rerank_rejected_count
        ],
        "routes_with_future_rerank_risk_rejections": [
            cast("JsonObject", asdict(record))
            for record in records
            if record.future_rerank_risk_rejected_count
        ],
        "routes_with_query_coverage_reranking": [
            cast("JsonObject", asdict(record))
            for record in records
            if record.query_coverage_rerank_count
        ],
        "routes_with_query_coverage_rejections": [
            cast("JsonObject", asdict(record))
            for record in records
            if record.query_coverage_rejected_count
        ],
        "routes_with_query_coverage_initial_selection": [
            [record.layer, record.kv_head]
            for record in records
            if record.query_coverage_initial_changed_count
        ],
        "query_coverage_initial_added_node_ids": {
            f"{record.layer}:{record.kv_head}": list(record.query_coverage_initial_added_node_ids)
            for record in records
            if record.query_coverage_initial_added_node_ids
        },
        "query_coverage_initial_layout_group_node_ids": {
            f"{record.layer}:{record.kv_head}": [
                list(group) for group in record.query_coverage_initial_layout_group_node_ids
            ]
            for record in records
            if record.query_coverage_initial_layout_group_node_ids
        },
        "routes_with_query_coverage_cross_head_allocation": [
            cast("JsonObject", asdict(record))
            for record in records
            if record.query_coverage_initial_global_allocation
            and (
                record.query_coverage_initial_allocated_bytes > 0
                or record.query_coverage_initial_cross_head_risk > 0
            )
        ],
        "routes_rejected_by_query_coverage_stability": [
            cast("JsonObject", asdict(record))
            for record in records
            if record.query_coverage_initial_stability_rejected
        ],
        "routes_with_shortfall": [
            cast("JsonObject", asdict(record)) for record in records if record.shortfall_positions
        ],
    }


def _prompt_mass_audit(
    full_state: FullKVState,
    prompt_attention_by_node: (
        Mapping[int, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None
    ),
    *,
    exact_source_node_ids: Iterable[int],
    prototype_source_node_ids: Iterable[int] = (),
) -> JsonObject:
    """Summarize retained prompt evidence without storing per-node scores."""

    if prompt_attention_by_node is None:
        return {"status": "unavailable"}
    node_ids = set(range(len(full_state.blocks)))
    if len(prompt_attention_by_node) != len(node_ids) or (
        isinstance(prompt_attention_by_node, Mapping) and set(prompt_attention_by_node) != node_ids
    ):
        raise HuggingFaceRuntimeError("prompt-mass audit scores do not cover the source cache")
    scores = {node_id: float(prompt_attention_by_node[node_id]) for node_id in sorted(node_ids)}
    if any(not math.isfinite(value) or value < 0 for value in scores.values()):
        raise HuggingFaceRuntimeError("prompt-mass audit scores must be finite and nonnegative")
    exact = frozenset(int(node_id) for node_id in exact_source_node_ids)
    prototype = frozenset(int(node_id) for node_id in prototype_source_node_ids)
    if not exact <= node_ids or not prototype <= node_ids:
        raise HuggingFaceRuntimeError("prompt-mass audit membership lies outside the source cache")
    if exact.intersection(prototype):
        raise HuggingFaceRuntimeError("exact and prototype prompt-mass memberships overlap")

    def mass(members: Iterable[int]) -> float:
        return math.fsum(scores[node_id] for node_id in members)

    total_mass = mass(node_ids)
    exact_mass = mass(exact)
    prototype_mass = mass(prototype)
    represented_mass = exact_mass + prototype_mass

    def fraction(value: float) -> float | None:
        return None if total_mass <= 0 else value / total_mass

    modalities: list[JsonValue] = []
    for modality in sorted(
        {block.modality for block in full_state.blocks},
        key=lambda item: item.value,
    ):
        members = {
            node_id for node_id, block in enumerate(full_state.blocks) if block.modality is modality
        }
        modality_total = mass(members)
        modality_exact = mass(members.intersection(exact))
        modality_prototype = mass(members.intersection(prototype))
        modalities.append(
            {
                "modality": modality.value,
                "source_block_count": len(members),
                "exact_block_count": len(members.intersection(exact)),
                "prototype_source_block_count": len(members.intersection(prototype)),
                "total_prompt_mass": modality_total,
                "exact_prompt_mass": modality_exact,
                "prototype_source_prompt_mass": modality_prototype,
                "represented_prompt_mass_fraction": (
                    None
                    if modality_total <= 0
                    else (modality_exact + modality_prototype) / modality_total
                ),
            }
        )
    return {
        "status": "measured_from_prompt_window_attention",
        "source_block_count": len(node_ids),
        "exact_block_count": len(exact),
        "prototype_source_block_count": len(prototype),
        "dropped_block_count": len(node_ids - exact - prototype),
        "total_prompt_mass": total_mass,
        "exact_prompt_mass": exact_mass,
        "prototype_source_prompt_mass": prototype_mass,
        "exact_prompt_mass_fraction": fraction(exact_mass),
        "represented_prompt_mass_fraction": fraction(represented_mass),
        "modalities": modalities,
    }


def _plan_trace(
    plan: MosaicKVCompressionPlan,
    *,
    compact_catalogs: bool = False,
    capture_selection_evidence: bool = False,
    prompt_attention_by_node: (
        Mapping[int, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None
    ) = None,
) -> JsonObject:
    selection = _source_selection_records(plan, include_cost_rate=False)
    prototypes = (
        {
            "prototype_id": record.prototype_id,
            "layer": record.layer,
            "kv_head": record.kv_head,
            "anchor_node_id": record.anchor_node_id,
            "assigned_node_ids": list(record.assigned_node_ids),
            "weights": [member.normalized_weight for member in record.members],
            "dispersion": record.diagnostics.dispersion,
            "position_span": record.diagnostics.position_span,
            "represented_source_positions": record.represented_source_positions,
            "attention_log_bias": record.attention_log_bias,
            "attention_log_bias_scale": record.attention_log_bias_scale,
            "attention_bias_basis": record.attention_bias_basis,
            "pooling_mode": record.pooling_mode,
            "anchor_position_strategy": record.anchor_position_strategy,
            "member_position_weight_entropy": [
                member.within_block_attention_entropy for member in record.members
            ],
        }
        for record in plan.construction.prototypes
    )
    edges = (
        {
            "source": plan.source_node_id(source),
            "target": plan.source_node_id(target),
            "weight": weight,
            "type": edge_type.value,
        }
        for source, target, weight, edge_type in zip(
            plan.graph.row_indices,
            plan.graph.column_indices,
            plan.graph.weights,
            plan.graph.edge_types,
            strict=True,
        )
    )
    active_bytes = plan.construction.state.statistics.active_kv_bytes
    residual_bytes = plan.construction.residual_storage.cpu_bytes
    budget_unit = plan.selection.budget.unit
    external_exact_bytes = sum(
        record.selected_bytes for record in plan.external_exact_capacity_records
    )
    eligible_residual_blocks = len(plan.construction.state.prototypes.source_blocks)
    stored_residual_blocks = len(plan.construction.residual_storage.payloads)
    prototype_source_node_ids = {
        node_id for record in plan.construction.prototypes for node_id in record.assigned_node_ids
    }
    return cast(
        "JsonObject",
        {
            "requested_method": plan.method.value,
            "effective_method": plan.effective_method,
            "request_wide_native_projection": _mosaickv_request_wide_projection(plan),
            "tier_mode": plan.construction.mode.value,
            "tier_reason": plan.construction.reason,
            "prototype_safety": cast("JsonObject", asdict(plan.construction.safety)),
            "prompt_mass_audit": _prompt_mass_audit(
                plan.full_state,
                prompt_attention_by_node,
                exact_source_node_ids=plan.construction.exact_node_ids,
                prototype_source_node_ids=prototype_source_node_ids,
            ),
            "source_budget_value": plan.source_budget_value,
            "shared_active_budget_value": plan.shared_active_budget_value,
            "active_budget_value": plan.active_budget_value,
            "planner_active_budget_value": plan.planner_active_budget_value,
            "configured_physical_capacity_fraction": (plan.configured_physical_capacity_fraction),
            "effective_budget_fraction": (
                plan.active_budget_value / plan.shared_active_budget_value
            ),
            "byte_budget_accounting": {
                "budget_unit": budget_unit.value,
                "source_bytes": (
                    plan.source_budget_value if budget_unit is BudgetUnit.BYTES else None
                ),
                "shared_hard_budget_bytes": (
                    plan.shared_active_budget_value if budget_unit is BudgetUnit.BYTES else None
                ),
                "effective_active_budget_bytes": (
                    plan.active_budget_value if budget_unit is BudgetUnit.BYTES else None
                ),
                "planner_budget_after_external_exact_bytes": (
                    plan.planner_active_budget_value if budget_unit is BudgetUnit.BYTES else None
                ),
                "external_exact_selected_bytes": (
                    external_exact_bytes if budget_unit is BudgetUnit.BYTES else None
                ),
                "planner_selected_bytes": (
                    plan.selection.active_bytes if budget_unit is BudgetUnit.BYTES else None
                ),
                "exact_active_bytes": (plan.construction.state.statistics.exact_kv_bytes),
                "prototype_active_bytes": (plan.construction.state.statistics.prototype_kv_bytes),
                "logical_active_bytes": active_bytes,
                "capacity_fraction_reduction_bytes": (
                    plan.shared_active_budget_value - plan.active_budget_value
                    if budget_unit is BudgetUnit.BYTES
                    else None
                ),
                "logical_underfill_vs_effective_budget_bytes": (
                    max(0, plan.active_budget_value - active_bytes)
                    if budget_unit is BudgetUnit.BYTES
                    else None
                ),
                "logical_underfill_vs_shared_hard_budget_bytes": (
                    max(0, plan.shared_active_budget_value - active_bytes)
                    if budget_unit is BudgetUnit.BYTES
                    else None
                ),
            },
            "selected_blocks": _trace_catalog(selection, compact=compact_catalogs),
            "prototypes": _trace_catalog(prototypes, compact=compact_catalogs),
            "graph_edges": _trace_catalog(edges, compact=compact_catalogs),
            "graph_diagnostics": asdict(plan.graph.diagnostics),
            "candidate_graph_routing": {
                "provenance": plan.graph_routing_provenance,
                "routed_head_only_planning": plan.optimization_node_ids is not None,
                "planner_node_count": len(plan.graph.nodes),
                "source_node_count": len(plan.full_state.blocks),
                "compact_node_id_space": plan.uses_compact_node_ids,
                "external_stable_exact_node_count": len(plan.external_exact_node_ids),
                "external_stable_capacity": _external_stable_capacity_trace(
                    plan.external_exact_capacity_records
                ),
                "optimization_node_count": (
                    None if plan.optimization_node_ids is None else len(plan.optimization_node_ids)
                ),
                "optimization_node_ids": (
                    None
                    if compact_catalogs or plan.optimization_node_ids is None
                    else list(plan.optimization_node_ids)
                ),
                "candidate_node_count": (
                    None
                    if plan.similarity_candidate_node_ids is None
                    else len(plan.similarity_candidate_node_ids)
                ),
                "candidate_node_ids": (
                    None
                    if compact_catalogs or plan.similarity_candidate_node_ids is None
                    else list(plan.similarity_candidate_node_ids)
                ),
                "graph_head_routes": [list(identity) for identity in plan.graph_head_routes],
            },
            "forecast_statistics": _forecast_trace(
                plan.forecast,
                compact_catalogs=compact_catalogs,
            ),
            "utility_diagnostics": _utility_trace(plan),
            "prototype_diagnostics_by_kv_head": _prototype_diagnostics_by_head(plan),
            "tier_storage_accounting": {
                "source_kv_bytes": plan.full_state.active_bytes,
                "exact_active_bytes": plan.construction.state.statistics.exact_kv_bytes,
                "prototype_active_bytes": (plan.construction.state.statistics.prototype_kv_bytes),
                "active_gpu_bytes": active_bytes,
                "residual_cpu_bytes": residual_bytes,
                "total_active_plus_residual_bytes": active_bytes + residual_bytes,
                "eligible_residual_source_blocks": eligible_residual_blocks,
                "stored_residual_source_blocks": stored_residual_blocks,
                "residual_source_block_fraction": (
                    0.0
                    if eligible_residual_blocks == 0
                    else stored_residual_blocks / eligible_residual_blocks
                ),
            },
            "selection_objective": {
                **cast("JsonObject", asdict(plan.selection.objective)),
                "selected_node_ids": list(plan.construction.exact_node_ids),
                "node_id_space": "full_kv_source",
            },
            "selection_decision_evidence": (
                _selection_decision_evidence(plan)
                if capture_selection_evidence
                else {"status": "not_captured"}
            ),
        },
    )


def _byte_budget_materialization_trace(
    dense_budget: DensePhysicalCacheBudget | None,
    config: RunConfig,
    *,
    source_prefill_kv_bytes: int,
    logical_prefill_active_kv_bytes: int,
    packed_prefill_active_kv_bytes: int,
    packed_prompt_capacity: int,
    plan: MosaicKVCompressionPlan | None,
) -> JsonObject | None:
    """Audit every stage between a requested byte cap and packed storage.

    The configured cap, dense-cache lattice, planner capacity fraction, tier
    construction, and physical padding are deliberately separate.  Collapsing
    them into one ``active_kv_bytes`` value previously hid double compression
    and made candidate caps appear binding when they were not.
    """

    if dense_budget is None:
        return None
    requested_by_retention = math.ceil(source_prefill_kv_bytes * config.cache.retention_ratio)
    requested_hard_budget = min(
        requested_by_retention,
        config.cache.budget_value,
    )
    if dense_budget.source_bytes != source_prefill_kv_bytes:
        raise HuggingFaceRuntimeError(
            "dense byte-budget source differs from materialized FullKV storage"
        )
    if dense_budget.target_bytes > requested_hard_budget:
        raise HuggingFaceRuntimeError(
            "dense byte-budget quantization exceeded the requested hard cap"
        )
    effective_budget = dense_budget.target_bytes if plan is None else plan.active_budget_value
    planner_budget = None if plan is None else plan.planner_active_budget_value
    external_exact_bytes = (
        None
        if plan is None
        else sum(record.selected_bytes for record in plan.external_exact_capacity_records)
    )
    exact_bytes = (
        logical_prefill_active_kv_bytes
        if plan is None
        else plan.construction.state.statistics.exact_kv_bytes
    )
    prototype_bytes = 0 if plan is None else plan.construction.state.statistics.prototype_kv_bytes
    selected_planner_bytes = None if plan is None else plan.selection.active_bytes
    hierarchical_accounting = None if plan is None else plan.hierarchical_storage_accounting
    return cast(
        "JsonObject",
        {
            "budget_unit": config.cache.budget_unit.value,
            "source_bytes": source_prefill_kv_bytes,
            "configured_active_byte_cap": config.cache.budget_value,
            "retention_requested_bytes": requested_by_retention,
            "requested_hard_budget_bytes": requested_hard_budget,
            "dense_attainable_budget_bytes": dense_budget.target_bytes,
            "dense_quantization_underfill_bytes": (
                requested_hard_budget - dense_budget.target_bytes
            ),
            "hard_prompt_capacity": dense_budget.prompt_capacity,
            "packed_prompt_capacity": packed_prompt_capacity,
            "physical_capacity_fraction": (
                1.0 if plan is None else plan.configured_physical_capacity_fraction
            ),
            "effective_active_budget_bytes": effective_budget,
            "capacity_fraction_reduction_bytes": (dense_budget.target_bytes - effective_budget),
            "planner_budget_after_external_exact_bytes": planner_budget,
            "external_exact_selected_bytes": external_exact_bytes,
            "planner_selected_bytes": selected_planner_bytes,
            "exact_logical_active_bytes": exact_bytes,
            "prototype_logical_active_bytes": prototype_bytes,
            "logical_active_bytes": logical_prefill_active_kv_bytes,
            "logical_underfill_vs_effective_budget_bytes": max(
                0,
                effective_budget - logical_prefill_active_kv_bytes,
            ),
            "logical_underfill_vs_dense_budget_bytes": max(
                0,
                dense_budget.target_bytes - logical_prefill_active_kv_bytes,
            ),
            "final_packed_active_bytes": packed_prefill_active_kv_bytes,
            "packed_padding_bytes": (
                packed_prefill_active_kv_bytes - logical_prefill_active_kv_bytes
            ),
            "knapsack_bytes": (
                None if hierarchical_accounting is None else hierarchical_accounting.knapsack_bytes
            ),
            "unique_selected_source_bytes": (
                None
                if hierarchical_accounting is None
                else hierarchical_accounting.unique_selected_source_bytes
            ),
            "planner_tier_bytes": (
                None
                if hierarchical_accounting is None
                else hierarchical_accounting.planner_tier_bytes
            ),
            "external_exact_bytes": (
                None
                if hierarchical_accounting is None
                else hierarchical_accounting.external_exact_bytes
            ),
            "constructed_active_bytes": (
                None
                if hierarchical_accounting is None
                else hierarchical_accounting.constructed_active_bytes
            ),
            "physical_padding_bytes": (
                None
                if hierarchical_accounting is None
                else hierarchical_accounting.physical_padding_bytes
            ),
            "hierarchical_logical_boundaries_aligned": (
                None
                if hierarchical_accounting is None
                else hierarchical_accounting.logical_boundaries_aligned
            ),
            "final_underfill_vs_dense_budget_bytes": max(
                0,
                dense_budget.target_bytes - packed_prefill_active_kv_bytes,
            ),
            "final_dense_budget_utilization": (
                packed_prefill_active_kv_bytes / dense_budget.target_bytes
            ),
        },
    )


def _repair_event_trace(
    event: RepairEvent,
    plan: MosaicKVCompressionPlan,
    *,
    compact: bool = False,
) -> JsonObject:
    """Enrich one repair event with source-block and cluster provenance."""

    record = event.to_record()
    prototypes = {prototype.prototype_id: prototype for prototype in plan.construction.prototypes}
    parent_by_node = {
        node_id: prototype.prototype_id
        for prototype in plan.construction.prototypes
        for node_id in prototype.assigned_node_ids
    }
    residual_by_node = {
        payload.source_node_id: payload for payload in plan.construction.residual_storage.payloads
    }
    restored_blocks: list[JsonObject] = []
    for node_id in event.restored_block_ids:
        block = plan.full_state.blocks[node_id]
        payload = residual_by_node[node_id]
        prototype = prototypes[parent_by_node[node_id]]
        member = next(member for member in prototype.members if member.node_id == node_id)
        restored_blocks.append(
            {
                "node_id": node_id,
                "prototype_id": parent_by_node[node_id],
                "layer": block.layer,
                "kv_head": block.kv_head,
                "modality": block.modality.value,
                "physical_cache_indices": list(block.physical_cache_indices),
                "original_logical_positions": list(block.original_logical_positions),
                "token_ids": None if block.token_ids is None else list(block.token_ids),
                "image_index": block.image_index,
                "frame_index": block.frame_index,
                "page_index": block.page_index,
                "region": None if block.region is None else list(block.region),
                "byte_size": block.byte_size,
                "prototype_weight": member.normalized_weight,
                "reconstruction_error": member.reconstruction_error,
                "risk_weighted_reconstruction_priority": (
                    event.signals.risk_for(prototype.prototype_id)
                    * member.normalized_weight
                    * member.reconstruction_error
                ),
                "residual_storage_dtype": payload.storage_dtype.value,
                "residual_placement": (
                    "cpu_pinned" if payload.key_pinned and payload.value_pinned else "cpu_pageable"
                ),
            }
        )
    clusters: list[JsonObject] = []
    triggered_ids = sorted({parent_by_node[node_id] for node_id in event.restored_block_ids})
    for prototype_id in triggered_ids:
        prototype = prototypes[prototype_id]
        clusters.append(
            {
                "prototype_id": prototype_id,
                "layer": prototype.layer,
                "kv_head": prototype.kv_head,
                "anchor_node_id": prototype.anchor_node_id,
                "anchor_logical_position": prototype.anchor_logical_position,
                "assigned_node_ids": list(prototype.assigned_node_ids),
                "attention_mass": event.signals.attention_mass_for(prototype_id),
                "dispersion": prototype.diagnostics.dispersion,
                "risk": event.signals.risk_for(prototype_id),
                "modality_composition": [
                    [modality.value, count]
                    for modality, count in prototype.diagnostics.modality_composition
                ],
                "minimum_logical_position": (prototype.diagnostics.minimum_logical_position),
                "maximum_logical_position": (prototype.diagnostics.maximum_logical_position),
                "position_span": prototype.diagnostics.position_span,
            }
        )
    record["triggered_prototype_clusters"] = cast("list[JsonValue]", clusters)
    record["restored_blocks"] = cast("list[JsonValue]", restored_blocks)
    if compact:
        for key in ("prototype_attention_masses", "prototype_risks"):
            values = cast("list[JsonValue]", record.pop(key, []))
            serialized_values = json.dumps(values, separators=(",", ":"), allow_nan=False).encode()
            record[f"{key}_summary"] = {
                "record_count": len(values),
                "sha256": hashlib.sha256(serialized_values).hexdigest(),
            }
    return record


def _baseline_plan_trace(
    plan: BaselineCompressionPlan,
    *,
    compact_catalogs: bool = False,
) -> JsonObject:
    selection = (
        {
            "node_id": decision.node_id,
            "selected": decision.selected,
            "selection_rank": decision.selection_rank,
            "score": decision.score,
            "cost": decision.cost,
            "reason": decision.reason.value,
            "stratum": [
                decision.stratum[0],
                decision.stratum[1],
                decision.stratum[2].value,
            ],
        }
        for decision in plan.selection.decisions
    )
    prompt_scores = (
        {decision.node_id: decision.score for decision in plan.selection.decisions}
        if plan.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
        else None
    )
    return cast(
        "JsonObject",
        {
            "requested_method": plan.method.value,
            "effective_method": (
                f"{plan.method.value}__retention_one_exact"
                if plan.state.is_retention_one
                else plan.method.value
            ),
            "tier_mode": "retention_one" if plan.state.is_retention_one else "exact_only_baseline",
            "tier_reason": "simple baselines retain exact source blocks only",
            "request_wide_native_projection": _baseline_request_wide_projection(plan),
            "source_budget_value": plan.source_budget_value,
            "active_budget_value": plan.active_budget_value,
            "selected_source_bytes": plan.selection.active_bytes,
            "budget_spent": plan.selection.budget_spent,
            "budget_unit": plan.selection.budget.unit.value,
            "baseline_seed": plan.selection.seed,
            "selection_provenance": plan.selection.score_provenance,
            "prompt_mass_audit": _prompt_mass_audit(
                plan.full_state,
                prompt_scores,
                exact_source_node_ids=plan.selection.selected_node_ids,
            ),
            "selected_blocks": _trace_catalog(selection, compact=compact_catalogs),
            "prototypes": [],
            "graph_edges": [],
            "graph_diagnostics": {"status": "not_used_by_simple_baseline"},
            "forecast_statistics": {
                "status": "not_used_by_simple_baseline",
                "score_provenance": plan.selection.score_provenance,
            },
        },
    )


def _lookm_plan_trace(plan: LookMCompressionPlan) -> JsonObject:
    """Label local paper equations distinctly from the pinned official code."""

    payload = plan.trace()
    payload.update(
        {
            "requested_method": MosaicKVMethod.LOOKM_REIMPL.value,
            "effective_method": MosaicKVMethod.LOOKM_REIMPL.value,
            "tier_mode": "lookm_paper_merge",
            "tier_reason": (
                "local paper-faithful LOOK-M equations; this is not official LOOK-M code"
            ),
            "request_wide_native_projection": _lookm_request_wide_projection(plan),
            "native_projection_semantics": (
                "paper-selected exact pivots projected to vLLM request blocks; LOOK-M KV "
                "merge is intentionally not injected"
            ),
            "source_budget_value": plan.source_blocks,
            "active_budget_value": plan.active_slots,
            "selected_blocks": [
                {
                    "layer": head.layer,
                    "kv_head": head.kv_head,
                    "selected_physical_positions": list(head.selected_physical_positions),
                    "important_physical_positions": list(head.important_physical_positions),
                    "recent_physical_positions": list(head.recent_physical_positions),
                }
                for head in plan.heads
            ],
            "prototypes": [],
            "graph_edges": [],
            "graph_diagnostics": {"status": "not_used_by_lookm_reimpl"},
            "forecast_statistics": {
                "status": "not_used_by_lookm_reimpl",
                "score_provenance": "paper_cumulative_prompt_attention_with_text_prior",
            },
            "selection_provenance": ("Wan_et_al_2024_equations_4_to_8_local_reimplementation"),
        }
    )
    return payload


def _prefixkv_plan_trace(plan: PrefixKVCompressionPlan) -> JsonObject:
    """Normalize PrefixKV paper provenance into the shared trace vocabulary."""

    evidence_variant = plan.profile is not None and plan.profile.evidence_variant
    return {
        "requested_method": MosaicKVMethod.PREFIXKV_REIMPL.value,
        "effective_method": plan.implementation_label,
        "tier_mode": "prefixkv_layerwise_exact_selection",
        "tier_reason": (
            "local evidence ablation; not paper-native PrefixKV"
            if evidence_variant
            else "paper layer profile and prompt-attention prefix selection"
        ),
        "request_wide_native_projection": _prefixkv_request_wide_projection(plan),
        "native_projection_semantics": (
            "HF-only evidence variant; native projection unsupported"
            if evidence_variant
            else "paper-selected layer positions projected to vLLM request-wide whole blocks"
        ),
        "selected_blocks": [
            {
                "layer": layer.layer,
                "retained_positions": layer.retained_positions,
                "positions": list(layer.selected_physical_positions),
            }
            for layer in plan.layers
        ],
        "prototypes": [],
        "graph_edges": [],
        "graph_diagnostics": {"status": "not_used_by_prefixkv_reimpl"},
        "forecast_statistics": {"status": "not_used_by_prefixkv_reimpl"},
        "selection_provenance": (
            "prefixkv_evidence_ablation_v1"
            if evidence_variant
            else "prefixkv_reimpl_paper_attention_and_layer_profile"
        ),
        "prefixkv": plan.trace(),
    }


def _vl_cache_plan_trace(plan: VLCacheCompressionPlan) -> JsonObject:
    """Normalize ICLR VL-Cache paper provenance into the shared trace vocabulary."""

    return {
        "requested_method": MosaicKVMethod.VL_CACHE_REIMPL.value,
        "effective_method": MosaicKVMethod.VL_CACHE_REIMPL.value,
        "tier_mode": "vl_cache_layer_adaptive_exact_selection",
        "tier_reason": "paper sparsity allocation and accumulated post-vision attention Top-K",
        "request_wide_native_projection": _vl_cache_request_wide_projection(plan),
        "native_projection_semantics": (
            "paper-selected layer/head positions projected to vLLM request-wide whole blocks"
        ),
        "selected_blocks": [
            {
                "layer": head.layer,
                "kv_head": head.kv_head,
                "positions": list(head.selected_physical_positions),
            }
            for layer in plan.layers
            for head in layer.heads
        ],
        "prototypes": [],
        "graph_edges": [],
        "graph_diagnostics": {"status": "not_used_by_vl_cache_reimpl"},
        "forecast_statistics": {"status": "not_used_by_vl_cache_reimpl"},
        "selection_provenance": "vl_cache_reimpl_Tu_et_al_ICLR_2025",
        "vl_cache": plan.trace(),
    }


def _published_kv_plan_trace(plan: PublishedKVCompressionPlan) -> JsonObject:
    """Normalize audited local paper policies into the shared trace vocabulary."""

    uses_merging = plan.method is MosaicKVMethod.ELASTIC_CACHE_REIMPL
    return {
        "requested_method": plan.method.value,
        "effective_method": plan.method.value,
        "tier_mode": (
            "published_layerwise_anchor_merge"
            if uses_merging
            else "published_exact_token_selection"
        ),
        "tier_reason": (
            "Elastic Cache layer-wise anchors and FP32 bucket means through the shared packer"
            if uses_merging
            else "audited paper policy through the shared exact-cache packer"
        ),
        "selected_blocks": [
            {
                "layer": head.layer,
                "kv_head": head.kv_head,
                "positions": list(head.selected_physical_positions),
                "protected_positions": list(head.protected_positions),
            }
            for head in plan.heads
        ],
        "prototypes": (
            [
                {
                    "layer": head.layer,
                    "kv_head": head.kv_head,
                    "anchor": anchor,
                    "source_membership": list(membership),
                }
                for head in plan.heads
                for anchor, membership in zip(
                    head.selected_physical_positions,
                    head.merge_memberships,
                    strict=True,
                )
            ]
            if uses_merging
            else []
        ),
        "graph_edges": [],
        "graph_diagnostics": {"status": "not_used_by_published_kv_reimplementation"},
        "forecast_statistics": {"status": "not_used_by_published_kv_reimplementation"},
        "selection_provenance": plan.score_provenance,
        "published_kv": plan.to_trace(),
    }


def _write_trace(path: Path, payload: JsonObject) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


_COMPACT_TRACE_LIST_KEYS = frozenset(
    {
        "graph_edges",
        "heads",
        "layers",
        "packed_slots",
        "prototypes",
        "selected_blocks",
    }
)


def _compact_trace_records(payload: JsonObject) -> None:
    """Replace large debug catalogs with deterministic audit summaries.

    Main-quality rows outside the fixed diagnostic subset retain hashes and
    counts rather than materializing petabytes of repeated graph/slot records.
    Diagnostic rows keep the complete debug payload.
    """

    for key, value in tuple(payload.items()):
        if key in _COMPACT_TRACE_LIST_KEYS and isinstance(value, list) and value:
            digest = hashlib.sha256()
            for record in value:
                digest.update(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                )
                digest.update(b"\n")
            payload[key] = {
                "storage": "omitted_non_diagnostic",
                "record_count": len(value),
                "sha256": digest.hexdigest(),
            }
        elif isinstance(value, dict) and key in {
            "forecast_statistics",
            "prefixkv",
            "vl_cache",
        }:
            _compact_trace_records(value)


class HuggingFaceMosaicKVModel:
    """Local-evaluation model integrating all MosaicKV stages around one HF adapter."""

    backend = "huggingface"

    def __init__(
        self,
        adapter: HuggingFaceMultimodalAdapter,
        config: RunConfig,
        *,
        trace_directory: str | Path,
    ) -> None:
        if config.execution.backend.value != "huggingface":
            raise ValueError("HuggingFaceMosaicKVModel requires execution.backend='huggingface'")
        self._hierarchical_gpu_debug_authorized = False
        if config.selection.stable_head_hierarchical_logit_safe_eviction:
            if os.environ.get("MOSAICKV_ENABLE_V100_GPU_DEBUG") != "1":
                raise HuggingFaceRuntimeError(
                    "hierarchical logit-safe eviction is GPU-unverified and requires the "
                    "explicit debug-only MOSAICKV_ENABLE_V100_GPU_DEBUG=1 authorization"
                )
            self._hierarchical_gpu_debug_authorized = True
        if config.execution.attention_implementation != "eager" and not config.method.is_full_cache:
            raise ValueError(
                "compressed HF methods currently require eager attention; "
                "SDPA and FlashAttention-2 are correctness-gated for FullKV"
            )
        if config.generation.do_sample or config.generation.temperature != 0:
            raise ValueError("the unified HF runtime requires deterministic greedy decoding")
        if config.generation.output_length_policy is not OutputLengthPolicy.FIXED_MAX_NEW_TOKENS:
            raise ValueError(
                "the unified HF runtime currently supports only fixed_max_new_tokens output"
            )
        self.adapter = adapter
        self.config = config
        self.trace_directory = Path(trace_directory)
        self.model_id = config.model.id
        self.method = config.method.value
        self.retention_ratio = config.cache.retention_ratio
        self._terminal_token_ids = terminal_token_ids(adapter.model, adapter.processor)
        self.supports_video = adapter.capabilities.video
        self._teacher_forced_outputs: dict[tuple[str, str], TeacherForcedNll] = {}
        self._prefixkv_profile: PrefixKVOfflineProfile | None = None
        if config.method.is_prefixkv_reimplementation and config.prefixkv.profile_path is not None:
            self._prefixkv_profile = load_prefixkv_profile(
                config.prefixkv.profile_path,
                model_id=config.model.id,
                model_revision=config.model.revision,
                target_retention_ratio=(
                    config.prefixkv.profile_reference_retention_ratio
                    if config.prefixkv.rescale_offline_profile_to_byte_budget
                    else config.cache.retention_ratio
                ),
                start_size=config.prefixkv.start_size,
                protect_size=config.prefixkv.protect_size,
            )
        if config.method.is_vl_cache_reimplementation:
            assert_vl_cache_calibration_disjoint(
                config.vl_cache.calibration_sample_ids,
                (),
            )

    def _trace_path(self, request: EvaluationRequest) -> Path:
        identity = hashlib.sha256(f"{request.run_id}\0{request.sample_id}".encode()).hexdigest()[
            :16
        ]
        base = self.trace_directory / request.run_id / f"{identity}.json"
        if not base.exists():
            return base
        # A process may be interrupted after its trace is committed but before
        # lmms-eval appends the scored JSONL row. Preserve that immutable trace
        # and allocate an auditable retry path instead of failing resume or
        # overwriting measured evidence.
        attempt = 1
        while True:
            candidate = base.with_name(f"{identity}-attempt-{attempt:03d}.json")
            if not candidate.exists():
                return candidate
            attempt += 1

    def _teacher_forced_target_ids(self, target: str) -> tuple[int, ...]:
        """Tokenize a PrefixKV PPL target without adding a second prompt."""

        tokenizer = getattr(self.adapter.processor, "tokenizer", self.adapter.processor)
        encoded = tokenizer(target, add_special_tokens=False, return_tensors="pt")
        input_ids = (
            encoded.get("input_ids")
            if isinstance(encoded, Mapping)
            else getattr(encoded, "input_ids", None)
        )
        if input_ids is None:
            raise HuggingFaceRuntimeError("processor tokenizer returned no target input_ids")
        target_ids = tuple(int(value) for value in input_ids.reshape(-1).tolist())
        if not target_ids:
            raise ValueError("teacher-forced target must contain at least one token")
        if len(target_ids) > self.config.generation.max_new_tokens:
            raise ValueError(
                "teacher-forced target exceeds generation.max_new_tokens; "
                "increase the configured limit rather than truncating the target"
            )
        return target_ids

    def teacher_forced_nll(
        self,
        request: EvaluationRequest,
        target: str,
    ) -> TeacherForcedNll:
        """Score human target tokens through the ordinary compressed cache path."""

        target_ids = self._teacher_forced_target_ids(target)
        key = (request.run_id, request.sample_id)
        if key in self._teacher_forced_outputs:
            raise HuggingFaceRuntimeError("duplicate pending teacher-forced request identity")
        kwargs = dict(request.generation_kwargs)
        kwargs["_mosaickv_teacher_forced_token_ids"] = target_ids
        self.generate(
            EvaluationRequest(
                run_id=request.run_id,
                sample_id=request.sample_id,
                task=request.task,
                messages=request.messages,
                generation_kwargs=kwargs,
            )
        )
        try:
            return self._teacher_forced_outputs.pop(key)
        except KeyError as error:
            raise HuggingFaceRuntimeError(
                "teacher-forced generation completed without NLL statistics"
            ) from error

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        """Run one explicit prefill/compress/decode request and always write a trace."""

        torch = _torch()
        trace_path = self._trace_path(request)
        decode_runtime_policy: JsonObject = {
            "cache_clone": "repair_provisional_steps_only",
            "attention_mask_growth": "amortized_device_buffer",
            "model_attention_mask": "omitted_when_all_physical_slots_are_valid",
            "cache_position": "amortized_logical_position_buffer",
            "finite_logit_validation": "single_deferred_sequence_reduction",
        }
        trace: JsonObject = {
            "schema_version": 1,
            "run_id": request.run_id,
            "sample_id": request.sample_id,
            "model": self.model_id,
            "backend": self.backend,
            "method": self.method,
            "retention_ratio": self.retention_ratio,
            "status": "started",
            "repair_events": [],
            "decode_runtime_policy": decode_runtime_policy,
            "ablation_variant": self.config.ablation.variant.value,
            "ablation_evaluation_only": self.config.ablation.evaluation_only,
        }
        if self.config.output_preserving is not None:
            trace["effective_method"] = self.config.output_preserving.effective_method
            trace["output_preserving_config"] = cast(
                "JsonObject",
                asdict(self.config.output_preserving),
            )
        started = time.perf_counter()
        raw_teacher_ids = request.generation_kwargs.get("_mosaickv_teacher_forced_token_ids")
        if raw_teacher_ids is None:
            teacher_forced_ids: tuple[int, ...] = ()
        elif (
            isinstance(raw_teacher_ids, tuple)
            and raw_teacher_ids
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in raw_teacher_ids
            )
        ):
            teacher_forced_ids = raw_teacher_ids
        else:
            raise ValueError(
                "_mosaickv_teacher_forced_token_ids must be a non-empty tuple "
                "of nonnegative integers"
            )
        teacher_forced_nll_sum = 0.0
        trace_instrumentation_seconds = 0.0
        deferred_trace_builder: Callable[[], JsonObject] | None = None
        cache_lifecycle: CacheLifecycle | None = None
        try:
            if self._prefixkv_profile is not None:
                self._prefixkv_profile.assert_evaluation_disjoint((request.sample_id,))
            if self.config.method.is_vl_cache_reimplementation:
                assert_vl_cache_calibration_disjoint(
                    self.config.vl_cache.calibration_sample_ids,
                    (request.sample_id,),
                )
            torch.manual_seed(self.config.execution.seed)
            if torch.cuda.is_available() and getattr(self.adapter.device, "type", "cpu") == "cuda":
                torch.cuda.manual_seed_all(self.config.execution.seed)
                torch.cuda.synchronize(self.adapter.device)
                torch.cuda.reset_peak_memory_stats(self.adapter.device)
            method = self.config.method
            self.adapter.set_attention_implementation(
                self.config.execution.attention_implementation
            )
            is_full_cache = method.is_full_cache
            retention_one_identity = not is_full_cache and self.config.cache.retention_ratio == 1.0
            fixed_diagnostic_sample = (
                self.config.diagnostics.enabled
                and request.sample_id in self.config.diagnostics.sample_ids
            )
            full_debug_trace = bool(
                fixed_diagnostic_sample and self.config.diagnostics.capture_full_trace
            )
            no_forecast_ablation = (
                self.config.ablation.variant is AblationVariant.NO_FUTURE_QUERY_FORECAST
            )
            needs_prompt_attention = not retention_one_identity and (
                (
                    method.is_mosaickv
                    and (
                        not no_forecast_ablation
                        or (self.config.graph.enabled and self.config.graph.attention_weight > 0)
                        or self.config.graph.similarity_shortlist_multiplier > 0
                    )
                )
                or method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
                or method.is_published_reimplementation
            )
            prepared = self.adapter.prepare_inputs(request.messages)
            prompt_evidence = None
            if (
                self.config.cache.physical_budget_mandatory_policy == "question_and_boundaries"
                or self.config.prefixkv.question_score_weight > 0
            ):
                from mosaickv.adapters.huggingface.prompt_evidence import map_prompt_evidence

                prompt_evidence = map_prompt_evidence(self.adapter, prepared, request.messages)

            def run_prefill() -> Any:
                return self.adapter.prefill(
                    prepared,
                    capture_queries=(
                        method.is_mosaickv
                        and not retention_one_identity
                        and not no_forecast_ablation
                        and self.config.output_preserving is None
                    ),
                    capture_attentions=needs_prompt_attention,
                    attention_query_window=(
                        None
                        if method.is_published_reimplementation
                        else (
                            max(1, self.config.forecasting.prompt_window)
                            if needs_prompt_attention
                            else None
                        )
                    ),
                    reduce_attention_to_kv_heads=(
                        (method.is_mosaickv or method is MosaicKVMethod.PROMPT_ATTENTION_TOPK)
                        and self.config.output_preserving is None
                    ),
                )

            image_video_encoder_seconds: float | None = None
            projector_seconds: float | None = None
            language_model_prefill_seconds: float | None = None
            is_cuda = (
                torch.cuda.is_available() and getattr(self.adapter.device, "type", "cpu") == "cuda"
            )
            if is_cuda:
                profiling = self.adapter.get_profiling_modules()
                audit = SynchronizationAudit()
                vision_timer = ModuleCudaTimer(
                    profiling.vision_encoder,
                    torch,
                    self.adapter.device,
                    audit,
                )
                projector_timer = ModuleCudaTimer(
                    profiling.projector,
                    torch,
                    self.adapter.device,
                    audit,
                )
                language_timer = ModuleCudaTimer(
                    profiling.language_model,
                    torch,
                    self.adapter.device,
                    audit,
                )

                def instrumented_prefill() -> Any:
                    with ExitStack() as stack:
                        stack.enter_context(vision_timer)
                        stack.enter_context(projector_timer)
                        stack.enter_context(language_timer)
                        return run_prefill()

                prefill, prefill_seconds = _timed(
                    prepared.model_inputs["input_ids"],
                    instrumented_prefill,
                )
                image_video_encoder_seconds = vision_timer.total_seconds
                projector_seconds = projector_timer.total_seconds
                language_model_prefill_seconds = language_timer.total_seconds
                if (
                    profiling.vision_includes_projector
                    and image_video_encoder_seconds is not None
                    and projector_seconds is not None
                ):
                    image_video_encoder_seconds = max(
                        0.0,
                        image_video_encoder_seconds - projector_seconds,
                    )
            else:
                prefill, prefill_seconds = _timed(
                    prepared.model_inputs["input_ids"],
                    run_prefill,
                )
            if not bool(torch.isfinite(prefill.logits).all()):
                raise HuggingFaceRuntimeError("prefill logits contain NaN or infinity")
            source_prefill_kv_bytes = _cache_payload_bytes(
                self.adapter,
                prefill.state.past_key_values,
            )
            # Original geometry, before selection: independently certifies the
            # largest feasible dense byte allocation in shared-cap analysis.
            trace["source_kv_layout"] = [
                {
                    "key_shape": list(key.shape),
                    "value_shape": list(value.shape),
                    "key_element_size": key.element_size(),
                    "value_element_size": value.element_size(),
                }
                for key, value in self.adapter._legacy_layers(prefill.state.past_key_values)[0]
            ]
            logical_prefill_active_kv_bytes = source_prefill_kv_bytes
            packed_prefill_active_kv_bytes = source_prefill_kv_bytes
            packed_prompt_capacity = prefill.state.active_cache_length
            cache_state_construction_seconds = 0.0
            forecast_seconds = 0.0
            planner_preparation_seconds = 0.0
            forecast_attention_projection_seconds = 0.0
            prompt_routing_evidence_seconds = 0.0
            head_route_configuration_seconds = 0.0
            graph_seconds = 0.0
            selection_seconds = 0.0
            tier_seconds = 0.0
            logit_stability_seconds = 0.0
            logit_stability_decision: ShortHorizonLogitStabilityDecision | None = None
            progressive_eviction_seconds = 0.0
            progressive_eviction_decision: ProgressiveLogitSafeEvictionDecision | None = None
            hierarchical_eviction_seconds = 0.0
            hierarchical_eviction_decision: BoundedHierarchicalProbeDecision | None = None
            packing_seconds = 0.0
            repair_state: RepairCacheState | None = None
            gpu_rescue_staging_seconds = 0.0
            gpu_rescue_transfer_ms = 0.0
            gpu_rescue_transfer_async = False
            gpu_rescue_initial_bytes = 0
            oracle_queries: EvaluationOnlyOracleQueries | None = None
            rollout: IsolatedQueryRollout | None = None
            predicted_token_attention: dict[tuple[int, int], np.ndarray[Any, Any]] | None = None
            prompt_token_attention: dict[tuple[int, int], np.ndarray[Any, Any]] | None = None
            prompt_block_attention: np.ndarray[Any, np.dtype[np.float64]] | None = None
            prompt_query_block_attention: dict[int, np.ndarray[Any, Any]] | None = None
            graph_head_uncertainty: dict[tuple[int, int], float] | None = None
            online_head_shifts: dict[tuple[int, int], float] = {}
            counterfactual_head_shifts: dict[tuple[int, int], float] = {}
            online_exact_head_routes: tuple[str, ...] = ()
            online_prototype_head_routes: tuple[str, ...] = ()
            draft_repair_distributions: tuple[Any, ...] = ()
            draft_repair_token_ids: tuple[int, ...] = ()
            dense_budget: DensePhysicalCacheBudget | None = None
            physical_prompt_capacity: int | None = None
            plan: MosaicKVCompressionPlan | None = None
            key_transform: RotaryPrototypeKeyTransform | None = None
            baseline_plan: BaselineCompressionPlan | None = None
            output_preserving_plan = None
            prototype_records: tuple[PrototypeRecord, ...] = ()
            lookm_plan: LookMCompressionPlan | None = None
            prefixkv_plan: PrefixKVCompressionPlan | None = None
            vl_cache_plan: VLCacheCompressionPlan | None = None
            published_kv_plan: PublishedKVCompressionPlan | None = None
            state = prefill.state
            if not is_full_cache and not retention_one_identity:
                cache_state_started = time.perf_counter()
                snapshot = self.adapter.extract_past_key_values(prefill.state.past_key_values)
                input_ids = prepared.model_inputs["input_ids"]
                recent_count = (
                    int(prefill.state.logical_sequence_length * self.config.lookm.recent_ratio)
                    if method.is_lookm_reimplementation
                    else 1
                )
                mandatory_positions: tuple[int, ...]
                if self.config.cache.physical_budget_mandatory_policy == "question_and_boundaries":
                    assert prompt_evidence is not None
                    mandatory_positions = tuple(
                        sorted(
                            {
                                0,
                                prefill.state.logical_sequence_length - 1,
                                *prompt_evidence.positions,
                            }
                        )
                    )
                elif self.config.cache.physical_budget_mandatory_policy == "first_and_last":
                    mandatory_positions = tuple(
                        sorted({0, int(prefill.state.logical_sequence_length) - 1})
                    )
                elif self.config.cache.physical_budget_mandatory_policy == "last_token":
                    mandatory_positions = (int(prefill.state.logical_sequence_length) - 1,)
                elif method.is_lookm_reimplementation:
                    mandatory_positions = tuple(
                        range(
                            prefill.state.logical_sequence_length - recent_count,
                            prefill.state.logical_sequence_length,
                        )
                    )
                elif method.is_prefixkv_reimplementation:
                    length = prefill.state.logical_sequence_length
                    mandatory_positions = tuple(
                        sorted(
                            set(range(min(self.config.prefixkv.start_size, length)))
                            | set(
                                range(
                                    max(0, length - self.config.prefixkv.protect_size),
                                    length,
                                )
                            )
                        )
                    )
                else:
                    mandatory_positions = (prefill.state.logical_sequence_length - 1,)
                full_state = FullKVState.from_cache_snapshot(
                    snapshot,
                    modality_spans=prepared.modality_map,
                    token_ids=input_ids,
                    block_size=self.config.cache.block_size,
                    original_logical_sequence_length=prefill.state.logical_sequence_length,
                    next_decode_position=prefill.state.next_decode_position,
                    mandatory_logical_positions=mandatory_positions,
                )
                cache_state_construction_seconds = time.perf_counter() - cache_state_started
                dense_budget = _dense_physical_cache_budget(full_state, self.config)
                physical_prompt_capacity = (
                    None if dense_budget is None else dense_budget.prompt_capacity
                )
                planning_config = (
                    self.config
                    if dense_budget is None
                    else replace(
                        self.config,
                        cache=replace(
                            self.config.cache,
                            budget_value=dense_budget.target_bytes,
                        ),
                    )
                )
                if self.config.output_preserving is not None:
                    from mosaickv.backends.output_preserving import (
                        select_output_preserving,
                        topk_anchor_capture,
                    )

                    if dense_budget is None:
                        raise HuggingFaceRuntimeError(
                            "output-preserving selection needs a dense cap"
                        )

                    if self.config.output_preserving.draft_validation:
                        rollout = run_isolated_greedy_query_rollout(
                            self.adapter,
                            prefill,
                            steps=self.config.forecasting.draft_steps,
                            capture_attentions=True,
                            capture_distributions=False,
                        )
                        forecast_seconds = (
                            rollout.cache_clone_seconds + rollout.draft_decode_seconds
                        )

                    def select_output_preserving_plan() -> Any:
                        prompt_scores = _block_attention_values_from_token_attention(
                            full_state,
                            _prompt_token_attention_by_head(
                                full_state,
                                replace(
                                    prefill,
                                    attention_weights=topk_anchor_capture(
                                        full_state,
                                        prefill.attention_weights,
                                    ),
                                ),
                                prompt_window=self.config.forecasting.prompt_window,
                            ),
                        )
                        return select_output_preserving(
                            full_state,
                            prefill.attention_weights,
                            self.config,
                            position_limits=dense_budget.position_limits,
                            target_bytes=dense_budget.target_bytes,
                            prompt_attention_scores=prompt_scores,
                            draft_attention_steps=(
                                () if rollout is None else rollout.attention_steps
                            ),
                            draft_cache_isolated=(
                                True if rollout is None else rollout.original_cache_unchanged
                            ),
                        )

                    output_preserving_plan, selection_seconds = _timed(
                        prefill.logits,
                        select_output_preserving_plan,
                    )
                    packed_state, tier_seconds = _timed(
                        prefill.logits,
                        lambda: output_preserving_plan.construct(full_state),
                    )
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = packed_state.statistics.active_kv_bytes
                elif method.is_mosaickv:
                    planner_wall_started = time.perf_counter()
                    key_transform_seconds = 0.0
                    if method in {
                        MosaicKVMethod.MOSAICKV_PROTO,
                        MosaicKVMethod.MOSAICKV_FULL,
                    } and (
                        self.config.prototypes.pooling_mode != "exact_medoid_key_forecast_value"
                    ):
                        key_transform, key_transform_seconds = _timed(
                            prefill.logits,
                            lambda: self.adapter.build_prototype_key_transform(prepared),
                        )
                    attention_provenance = "eager_prompt_and_isolated_draft_attention_v1"
                    if no_forecast_ablation:
                        projection_started = time.perf_counter()
                        forecast = None
                        predicted_token_attention = _uniform_token_attention_by_head(full_state)
                        attention = _block_attention_from_token_attention(
                            full_state,
                            predicted_token_attention,
                        )
                        attention_provenance = "declared_no_forecast_uniform_position_control_v1"
                        forecast_attention_projection_seconds += (
                            time.perf_counter() - projection_started
                        )
                    elif self.config.ablation.variant is AblationVariant.ORACLE_FUTURE_QUERY:
                        if not self.config.ablation.evaluation_only:
                            raise HuggingFaceRuntimeError(
                                "oracle forecasting is restricted to evaluation-only ablations"
                            )

                        def build_oracle_forecast() -> tuple[
                            QueryForecast,
                            EvaluationOnlyOracleQueries,
                        ]:
                            oracle = collect_evaluation_only_true_future_queries(
                                self.adapter,
                                prefill,
                                future_steps=self.config.diagnostics.forecast_oracle_steps,
                                capture_attentions=True,
                            )
                            forecast_value = build_query_forecast(
                                prefill.query_vectors.layers,
                                oracle.query_layers,
                                tuple(layer.kv_heads for layer in full_state.layers),
                                self.config.forecasting,
                                original_logical_sequence_length=(
                                    prefill.state.logical_sequence_length
                                ),
                                draft_cache_isolated=True,
                            )
                            return forecast_value, oracle

                        (forecast, oracle_queries), forecast_seconds = _timed(
                            prefill.logits,
                            build_oracle_forecast,
                        )
                        projection_started = time.perf_counter()
                        predicted_token_attention = _oracle_token_attention_by_head(
                            full_state,
                            oracle_queries,
                        )
                        attention = _block_attention_from_token_attention(
                            full_state,
                            predicted_token_attention,
                        )
                        attention_provenance = "evaluation_only_fullkv_true_future_attention_v1"
                        forecast_attention_projection_seconds += (
                            time.perf_counter() - projection_started
                        )
                    else:
                        forecast, rollout = forecast_with_rollout(
                            self.adapter,
                            prefill,
                            self.config.forecasting,
                            capture_attentions=True,
                            capture_distributions=(
                                self.config.repair.policy
                                in {RepairPolicy.DECISION_AWARE, RepairPolicy.ORACLE}
                            ),
                        )
                        forecast_seconds = forecast.timing.total
                        draft_repair_distributions = rollout.draft_distributions
                        draft_repair_token_ids = rollout.token_ids
                        projection_started = time.perf_counter()
                        (
                            predicted_token_attention,
                            online_head_shifts,
                        ) = _forecast_token_attention_and_shift_by_head(
                            full_state,
                            prefill,
                            rollout,
                            self.config,
                        )
                        attention = _block_attention_from_token_attention(
                            full_state,
                            predicted_token_attention,
                        )
                        forecast_attention_projection_seconds += (
                            time.perf_counter() - projection_started
                        )
                    # Repair counterfactuals are deliberately independent of
                    # the query distribution used for cache selection.  Run
                    # this after every forecast branch so a prompt-window
                    # selector can use an isolated FullKV draft solely as an
                    # online repair reference.
                    if self.config.repair.counterfactual_draft_steps:
                        repair_rollout = run_isolated_greedy_query_rollout(
                            self.adapter,
                            prefill,
                            steps=self.config.repair.counterfactual_draft_steps,
                            capture_attentions=(self.config.repair.counterfactual_route_heads),
                            capture_distributions=True,
                        )
                        draft_repair_distributions = repair_rollout.draft_distributions
                        draft_repair_token_ids = repair_rollout.token_ids
                        forecast_seconds += (
                            repair_rollout.cache_clone_seconds + repair_rollout.draft_decode_seconds
                        )
                        if self.config.repair.counterfactual_route_heads:
                            _, counterfactual_head_shifts = (
                                _forecast_token_attention_and_shift_by_head(
                                    full_state,
                                    prefill,
                                    repair_rollout,
                                    self.config,
                                )
                            )
                    route_config_started = time.perf_counter()
                    online_exact_head_routes = _uncertainty_exact_head_routes(
                        online_head_shifts,
                        self.config.forecasting.uncertainty_exact_head_fraction,
                    )
                    online_planning_config = planning_config
                    if online_exact_head_routes:
                        combined_routes = tuple(
                            sorted(
                                set(planning_config.prototypes.exact_head_routes)
                                | set(online_exact_head_routes),
                                key=lambda route: tuple(int(value) for value in route.split(":")),
                            )
                        )
                        online_planning_config = replace(
                            planning_config,
                            prototypes=replace(
                                planning_config.prototypes,
                                exact_head_routes=combined_routes,
                            ),
                        )
                    head_route_configuration_seconds += time.perf_counter() - route_config_started
                    needs_prompt_ranking = bool(
                        self.config.selection.prompt_anchor_fraction > 0
                        or self.config.graph.similarity_shortlist_multiplier > 0
                        or self.config.prototypes.uncertain_head_only
                        or self.config.selection.stable_head_query_coverage_reranking
                        or self.config.selection.stable_head_query_coverage_initial_selection
                    )
                    if needs_prompt_ranking:
                        prompt_evidence_started = time.perf_counter()
                        if (
                            self.config.selection.stable_head_query_coverage_reranking
                            or self.config.selection.stable_head_query_coverage_initial_selection
                        ):
                            prompt_query_token_attention = _prompt_query_token_attention_by_head(
                                full_state,
                                prefill,
                                prompt_window=self.config.forecasting.prompt_window,
                            )
                            # Keep the established scalar Prompt Top-K anchor
                            # on its existing reduction path. The per-query
                            # matrix is additional reranking evidence, not a
                            # substitute score computed in a new order.
                            prompt_token_attention = _prompt_token_attention_by_head(
                                full_state,
                                prefill,
                                prompt_window=self.config.forecasting.prompt_window,
                            )
                            prompt_query_block_attention = (
                                _block_query_attention_from_token_attention(
                                    full_state,
                                    prompt_query_token_attention,
                                )
                            )
                        else:
                            prompt_token_attention = (
                                predicted_token_attention
                                if (
                                    predicted_token_attention is not None
                                    and rollout is not None
                                    and not rollout.attention_steps
                                    and self.config.forecasting.prompt_window > 0
                                )
                                else _prompt_token_attention_by_head(
                                    full_state,
                                    prefill,
                                    prompt_window=self.config.forecasting.prompt_window,
                                )
                            )
                        if prompt_token_attention is None:  # pragma: no cover - assigned above
                            raise HuggingFaceRuntimeError(
                                "prompt token attention is unexpectedly absent"
                            )
                        prompt_block_attention = _block_attention_values_from_token_attention(
                            full_state,
                            prompt_token_attention,
                        )
                        graph_head_uncertainty = _attention_uncertainty_by_head(
                            prompt_token_attention
                        )
                        if counterfactual_head_shifts:
                            graph_head_uncertainty = counterfactual_head_shifts
                        prompt_routing_evidence_seconds += (
                            time.perf_counter() - prompt_evidence_started
                        )
                    if self.config.prototypes.uncertain_head_only and method in {
                        MosaicKVMethod.MOSAICKV_PROTO,
                        MosaicKVMethod.MOSAICKV_FULL,
                    }:
                        route_config_started = time.perf_counter()
                        all_head_routes = {
                            (block.layer, block.kv_head) for block in full_state.blocks
                        }
                        routed = _uncertain_head_routes(
                            all_head_routes,
                            graph_head_uncertainty,
                            fraction=self.config.prototypes.uncertain_head_fraction,
                            threshold=self.config.prototypes.uncertain_head_threshold,
                            label="prototype",
                        )
                        if (
                            self.config.selection.stable_head_prompt_routing
                            and not self.config.selection.routed_head_only_planning
                        ):
                            if prompt_block_attention is None:
                                raise HuggingFaceRuntimeError(
                                    "stable-head prompt routing requires prompt block attention"
                                )
                            routed_set = set(routed)
                            routed_attention = dict(attention)
                            for node_id, block in enumerate(full_state.blocks):
                                identity = (block.layer, block.kv_head)
                                if identity not in routed_set:
                                    routed_attention[node_id] = prompt_block_attention[node_id]
                            attention = routed_attention
                        exact_routes = {
                            tuple(int(value) for value in route.split(":"))
                            for route in online_planning_config.prototypes.exact_head_routes
                        }
                        online_prototype_head_routes = tuple(
                            f"{layer}:{head}"
                            for layer, head in routed
                            if (layer, head) not in exact_routes
                        )
                        selection_only = {
                            route
                            for route in (
                                online_planning_config.prototypes.selection_only_head_routes
                            )
                        }
                        selection_only.update(
                            f"{layer}:{head}"
                            for layer, head in all_head_routes
                            if (layer, head) not in routed and (layer, head) not in exact_routes
                        )
                        online_planning_config = replace(
                            online_planning_config,
                            prototypes=replace(
                                online_planning_config.prototypes,
                                selection_only_head_routes=tuple(
                                    sorted(
                                        selection_only,
                                        key=lambda route: tuple(
                                            int(value) for value in route.split(":")
                                        ),
                                    )
                                ),
                            ),
                        )
                        head_route_configuration_seconds += (
                            time.perf_counter() - route_config_started
                        )
                    plan = build_compression_plan(
                        full_state,
                        forecast,
                        attention,
                        self.adapter.capabilities,
                        online_planning_config,
                        position_limits=(
                            None if dense_budget is None else dense_budget.position_limits
                        ),
                        attention_provenance=attention_provenance,
                        key_transform=key_transform,
                        prompt_attentions=prefill.attention_weights,
                        prompt_attention_by_node=prompt_block_attention,
                        prompt_query_attention_by_node=prompt_query_block_attention,
                        prototype_token_weight_by_head=predicted_token_attention,
                        graph_head_uncertainty=graph_head_uncertainty,
                    )
                    if self.config.selection.stable_head_short_horizon_logit_stability:
                        if dense_budget is None:
                            raise HuggingFaceRuntimeError(
                                "short-horizon logit stability requires a dense byte budget"
                            )
                        stability_result, logit_stability_seconds = _timed(
                            prefill.logits,
                            lambda: _run_short_horizon_logit_stability(
                                self.adapter,
                                prefill,
                                plan,
                                self.config,
                                key_transform=key_transform,
                                physical_prompt_capacity=dense_budget.prompt_capacity,
                            ),
                        )
                        plan, logit_stability_decision = stability_result
                    if self.config.selection.stable_head_progressive_logit_safe_eviction:
                        if dense_budget is None:
                            raise HuggingFaceRuntimeError(
                                "progressive logit-safe eviction requires a dense byte budget"
                            )
                        progressive_result, progressive_eviction_seconds = _timed(
                            prefill.logits,
                            lambda: _run_progressive_logit_safe_eviction(
                                self.adapter,
                                prefill,
                                plan,
                                self.config,
                                key_transform=key_transform,
                                position_limits=dense_budget.position_limits,
                            ),
                        )
                        plan, progressive_eviction_decision = progressive_result
                    if self.config.selection.stable_head_hierarchical_logit_safe_eviction:
                        if dense_budget is None:
                            raise HuggingFaceRuntimeError(
                                "hierarchical logit-safe eviction requires a dense byte budget"
                            )
                        hierarchical_result, hierarchical_eviction_seconds = _timed(
                            prefill.logits,
                            lambda: _run_hierarchical_logit_safe_eviction(
                                self.adapter,
                                prefill,
                                plan,
                                self.config,
                                key_transform=key_transform,
                                position_limits=dense_budget.position_limits,
                            ),
                        )
                        plan, hierarchical_eviction_decision = hierarchical_result
                    planner_wall_seconds = time.perf_counter() - planner_wall_started
                    planner_preparation_seconds = max(
                        0.0,
                        planner_wall_seconds
                        - forecast_seconds
                        - key_transform_seconds
                        - plan.graph_seconds
                        - plan.selection_seconds
                        - plan.tier_seconds
                        - logit_stability_seconds
                        - progressive_eviction_seconds
                        - hierarchical_eviction_seconds,
                    )
                    if (
                        dense_budget is not None
                        and hierarchical_eviction_decision is not None
                        and hierarchical_eviction_decision.used_fullkv_fallback
                    ):
                        attempted_lengths = {
                            layout.active_cache_length
                            for layout in plan.construction.active_layouts
                        }
                        if not attempted_lengths:
                            raise HuggingFaceRuntimeError(
                                "hierarchical FullKV fallback has no active cache layouts"
                            )
                        attempted_capacity = max(attempted_lengths)
                        attempted_active_bytes = plan.construction.state.statistics.active_kv_bytes
                        failure_total_seconds = time.perf_counter() - started
                        accounted_failure_seconds = (
                            prefill_seconds
                            + cache_state_construction_seconds
                            + forecast_seconds
                            + planner_preparation_seconds
                            + plan.graph_seconds
                            + plan.selection_seconds
                            + plan.tier_seconds
                            + key_transform_seconds
                            + logit_stability_seconds
                            + progressive_eviction_seconds
                            + hierarchical_eviction_seconds
                        )
                        failure_timings = cast(
                            "JsonObject",
                            {
                                "prefill": prefill_seconds,
                                "image_video_encoder": image_video_encoder_seconds,
                                "projector": projector_seconds,
                                "language_model_prefill": language_model_prefill_seconds,
                                "cache_state_construction": (cache_state_construction_seconds),
                                "forecast": forecast_seconds,
                                "planner_preparation": planner_preparation_seconds,
                                "graph": plan.graph_seconds,
                                "utility_and_selection": plan.selection_seconds,
                                "tier_construction": (plan.tier_seconds + key_transform_seconds),
                                "logit_stability": logit_stability_seconds,
                                "progressive_eviction": progressive_eviction_seconds,
                                "hierarchical_eviction": hierarchical_eviction_seconds,
                                "cache_packing": 0.0,
                                "decode": 0.0,
                                "repair_transfer": 0.0,
                                "repair_redecode": 0.0,
                                "host_orchestration": max(
                                    0.0,
                                    failure_total_seconds - accounted_failure_seconds,
                                ),
                                "total_before_failure": failure_total_seconds,
                            },
                        )
                        decision_record = cast(
                            "JsonObject",
                            {
                                **asdict(hierarchical_eviction_decision),
                                "overhead_seconds": hierarchical_eviction_seconds,
                                "probe_state_isolation": (
                                    "clone_per_probe_discarded_after_evaluation"
                                ),
                            },
                        )
                        fallback_accounting = plan.hierarchical_storage_accounting
                        rejected_accounting = plan.hierarchical_rejected_proposal_storage_accounting
                        rejected_proposal_id = plan.hierarchical_rejected_proposal_id
                        if fallback_accounting is None:
                            raise HuggingFaceRuntimeError(
                                "hierarchical FullKV fallback is missing final accounting"
                            )
                        if rejected_accounting is None or rejected_proposal_id is None:
                            raise HuggingFaceRuntimeError(
                                "hierarchical FullKV fallback is missing rejected proposal "
                                "accounting"
                            )
                        failure_payload = _hard_budget_fallback_trace_payload(
                            decision_record,
                            attempted_active_kv_bytes=attempted_active_bytes,
                            hard_active_kv_budget_bytes=dense_budget.target_bytes,
                            attempted_prompt_capacity=attempted_capacity,
                            hard_prompt_capacity=dense_budget.prompt_capacity,
                            timing_breakdown=failure_timings,
                            fullkv_fallback_accounting=fallback_accounting,
                            rejected_compressed_proposal_accounting=rejected_accounting,
                            rejected_compressed_proposal_id=rejected_proposal_id,
                        )
                        error = HuggingFaceRuntimeError(
                            "hierarchical logit-safe eviction returned an over-budget "
                            "FullKV fallback; rejected before final cache packing"
                        )
                        error.mosaickv_failure_trace_payload = failure_payload
                        raise error
                    if dense_budget is not None:
                        hierarchical_fullkv_fallback = bool(
                            hierarchical_eviction_decision is not None
                            and hierarchical_eviction_decision.used_fullkv_fallback
                        )
                        measured_capacity = _mosaickv_physical_prompt_capacity(
                            plan,
                            dense_budget,
                            used_fullkv_fallback=hierarchical_fullkv_fallback,
                        )
                        # The common HF tensors use the largest logical head
                        # length as their physical sequence extent. Shorter
                        # routed heads are zero padded by the common packer and
                        # excluded with per-layer/head eager-attention masks.
                        physical_prompt_capacity = measured_capacity
                    graph_seconds = plan.graph_seconds
                    selection_seconds = plan.selection_seconds
                    tier_seconds = plan.tier_seconds + key_transform_seconds
                    packed_state = plan.construction.state
                    prototype_records = plan.construction.prototypes
                    logical_prefill_active_kv_bytes = (
                        plan.construction.state.statistics.active_kv_bytes
                    )
                    minimum_representable = (
                        None
                        if dense_budget is None
                        else dense_budget.minimum_block_representable_fill(
                            self.config.cache.block_size,
                            target_bytes=plan.active_budget_value,
                        )
                    )
                    if (
                        method is MosaicKVMethod.MOSAICKV_EXACT
                        and minimum_representable is not None
                        and not self.config.selection.stop_on_nonpositive_gain
                        and logical_prefill_active_kv_bytes < minimum_representable
                    ):
                        if dense_budget is None:  # pragma: no cover - narrowed above
                            raise AssertionError("dense underfill check requires a dense budget")
                        raise HuggingFaceRuntimeError(
                            "budget-filling MosaicKV selection underfilled the dense cache: "
                            f"selected={logical_prefill_active_kv_bytes}, "
                            f"minimum_representable={minimum_representable}, "
                            f"target={plan.active_budget_value}, "
                            f"shared_hard_cap={dense_budget.target_bytes}"
                        )
                elif method.is_compressed_baseline:
                    prompt_started = time.perf_counter()
                    prompt_attention = (
                        _prompt_attention_by_node(
                            full_state,
                            prefill,
                            prompt_window=self.config.forecasting.prompt_window,
                        )
                        if method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
                        else None
                    )
                    prompt_scoring_seconds = time.perf_counter() - prompt_started
                    baseline_plan = build_exact_baseline_plan(
                        full_state,
                        method,
                        planning_config.cache,
                        position_limits=(
                            None if dense_budget is None else dense_budget.position_limits
                        ),
                        seed=self.config.execution.seed,
                        prompt_attention_by_node=prompt_attention,
                        value_chunk_size=self.config.graph.similarity_chunk_size,
                    )
                    selection_seconds = prompt_scoring_seconds + baseline_plan.selection_seconds
                    tier_seconds = baseline_plan.tier_seconds
                    packed_state = baseline_plan.state
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = baseline_plan.selection.active_bytes
                elif method.is_lookm_reimplementation:
                    lookm_plan, selection_seconds = _timed(
                        prefill.logits,
                        lambda: build_lookm_reimpl_plan(
                            full_state,
                            prefill.attention_weights,
                            self.config.lookm,
                            planning_config.cache,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                    packed_state = None
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = lookm_plan.active_bytes
                elif method.is_vl_cache_reimplementation:
                    vl_cache_plan, selection_seconds = _timed(
                        prefill.logits,
                        lambda: build_vl_cache_reimpl_plan(
                            full_state,
                            prefill.attention_weights,
                            self.config.vl_cache,
                            planning_config.cache,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                    packed_state = None
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = vl_cache_plan.retained_bytes
                elif method.is_prefixkv_reimplementation:
                    prefixkv_plan, selection_seconds = _timed(
                        prefill.logits,
                        lambda: build_prefixkv_reimpl_plan(
                            full_state,
                            prefill.attention_weights,
                            self.config.prefixkv,
                            planning_config.cache,
                            model_id=self.config.model.id,
                            model_revision=self.config.model.revision,
                            profile=self._prefixkv_profile,
                            question_positions=prompt_evidence.positions if prompt_evidence else (),
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                    _assert_prefixkv_common_hf_representable(prefixkv_plan, self.adapter)
                    packed_state = None
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = prefixkv_plan.retained_bytes
                elif method.is_published_eviction_reimplementation:
                    published_kv_plan, selection_seconds = _timed(
                        prefill.logits,
                        lambda: build_published_kv_reimpl_plan(
                            full_state,
                            prefill.attention_weights,
                            method,
                            self.config.published_kv,
                            planning_config.cache,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                    packed_state = None
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = published_kv_plan.retained_bytes
                else:  # pragma: no cover - strict method vocabulary guards this
                    raise HuggingFaceRuntimeError(f"unsupported runtime method: {method.value}")
                if lookm_plan is not None:
                    lookm_payloads, lookm_slots = lookm_runtime_payloads(lookm_plan)
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: pack_runtime_payloads(
                            self.adapter,
                            full_state,
                            lookm_payloads,
                            lookm_slots,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                            packing_mode=self.config.cache.packing_mode,
                        ),
                    )
                elif prefixkv_plan is not None:
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: _pack_prefixkv_ragged(
                            self.adapter,
                            prefixkv_plan,
                            prefill.state.past_key_values,
                        ),
                    )
                elif vl_cache_plan is not None:
                    vl_cache_payloads, vl_cache_slots = vl_cache_runtime_payloads(vl_cache_plan)
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: pack_runtime_payloads(
                            self.adapter,
                            full_state,
                            vl_cache_payloads,
                            vl_cache_slots,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                            packing_mode=self.config.cache.packing_mode,
                        ),
                    )
                elif published_kv_plan is not None:
                    published_payloads, published_slots = published_kv_runtime_payloads(
                        published_kv_plan
                    )
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: pack_runtime_payloads(
                            self.adapter,
                            full_state,
                            published_payloads,
                            published_slots,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                            packing_mode=self.config.cache.packing_mode,
                        ),
                    )
                else:
                    if packed_state is None:
                        raise HuggingFaceRuntimeError("runtime cache plan has no active state")
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: pack_runtime_cache(
                            self.adapter,
                            full_state,
                            packed_state,
                            prototype_records,
                            tuple(record.prototype_id for record in prototype_records),
                            physical_prompt_capacity=physical_prompt_capacity,
                            packing_mode=self.config.cache.packing_mode,
                        ),
                    )
                state = _state_from_packed(self.adapter, prefill.state, packed)
                packed_prompt_capacity = packed.prompt_capacity
                packed_prefill_active_kv_bytes = _cache_payload_bytes(
                    self.adapter,
                    packed.snapshot,
                )
                if (
                    dense_budget is not None
                    and packed_prefill_active_kv_bytes > dense_budget.target_bytes
                ):
                    raise HuggingFaceRuntimeError(
                        "materialized cache exceeds its hard physical byte budget: "
                        f"active={packed_prefill_active_kv_bytes}, "
                        f"budget={dense_budget.target_bytes}"
                    )
                if _plan_has_live_residual_repair(
                    method,
                    plan,
                    repair_enabled=self.config.repair.enabled,
                    repair_policy=self.config.repair.policy,
                ):
                    if plan is None:  # pragma: no cover - helper guarantees this
                        raise HuggingFaceRuntimeError("repairable plan is unexpectedly absent")
                    repair_state = RepairCacheState.from_construction(
                        full_state,
                        plan.construction,
                    )
                    if self.config.residual.gpu_rescue_max_blocks > 0:
                        rescue_started = time.perf_counter()
                        (
                            repair_state,
                            gpu_rescue_transfer_ms,
                            gpu_rescue_transfer_async,
                        ) = stage_gpu_rescue_payloads(
                            repair_state,
                            max_blocks=self.config.residual.gpu_rescue_max_blocks,
                            max_bytes=self.config.residual.gpu_rescue_max_bytes,
                        )
                        gpu_rescue_staging_seconds = time.perf_counter() - rescue_started
                        gpu_rescue_initial_bytes = repair_state.gpu_rescue_bytes
                        tier_seconds += gpu_rescue_staging_seconds

                def build_deferred_cache_trace() -> JsonObject:
                    payload: JsonObject
                    if output_preserving_plan is not None:
                        payload = cast("JsonObject", output_preserving_plan.trace())
                        for key in ("selected_blocks", "anchor_selected_blocks"):
                            payload[key] = _trace_catalog(
                                cast("list[JsonObject]", payload[key]),
                                compact=not full_debug_trace,
                            )
                    elif plan is not None:
                        payload = _plan_trace(
                            plan,
                            compact_catalogs=not full_debug_trace,
                            capture_selection_evidence=(
                                self.config.diagnostics.capture_selection_evidence
                            ),
                            prompt_attention_by_node=prompt_block_attention,
                        )
                    elif baseline_plan is not None:
                        payload = _baseline_plan_trace(
                            baseline_plan,
                            compact_catalogs=not full_debug_trace,
                        )
                    elif lookm_plan is not None:
                        payload = _lookm_plan_trace(lookm_plan)
                    elif prefixkv_plan is not None:
                        payload = _prefixkv_plan_trace(prefixkv_plan)
                    elif vl_cache_plan is not None:
                        payload = _vl_cache_plan_trace(vl_cache_plan)
                    elif published_kv_plan is not None:
                        payload = _published_kv_plan_trace(published_kv_plan)
                    else:  # pragma: no cover - compressed branch always has a plan
                        raise HuggingFaceRuntimeError("compressed trace is missing its plan")
                    payload["packed_slots"] = _trace_catalog(
                        packed.slot_records,
                        compact=not full_debug_trace,
                    )
                    payload["packed_cache_all_slots_valid"] = packed.all_slots_valid
                    return payload

                deferred_trace_builder = build_deferred_cache_trace
            elif retention_one_identity:
                trace.update(
                    {
                        "requested_method": method.value,
                        "effective_method": (
                            f"{self.config.output_preserving.effective_method}__retention_one_exact"
                            if self.config.output_preserving is not None
                            else f"{method.value}__retention_one_exact"
                        ),
                        "tier_mode": "retention_one",
                        "tier_reason": (
                            "retention 1.0 reuses the untouched FullKV state; no "
                            "forecast, graph, selection, packing, prototype, residual, "
                            "quantization, offload, or repair operation was run"
                        ),
                        "selected_blocks": [],
                        "prototypes": [],
                        "graph_edges": [],
                        "graph_diagnostics": {"status": "not_used_by_retention_one_identity"},
                        "forecast_statistics": {"status": "not_used_by_retention_one_identity"},
                        "baseline_seed": self.config.execution.seed,
                        "selection_provenance": "retention_one_identity_no_selection",
                        "packed_slots": [],
                        "packed_cache_all_slots_valid": True,
                    }
                )
            else:
                trace.update(
                    {
                        "requested_method": method.value,
                        "effective_method": method.value,
                        "tier_mode": "full_cache",
                        "tier_reason": "no cache transformation was run",
                        "selected_blocks": [],
                        "prototypes": [],
                        "graph_edges": [],
                        "graph_diagnostics": {"status": "not_used_by_full_kv"},
                        "forecast_statistics": {"status": "not_used_by_full_kv"},
                        "baseline_seed": self.config.execution.seed,
                        "selection_provenance": "full_cache_no_selection",
                        "packed_slots": [],
                        "packed_cache_all_slots_valid": True,
                    }
                )

            if deferred_trace_builder is None:
                trace_started = time.perf_counter()
                if not full_debug_trace:
                    _compact_trace_records(trace)
                    trace["trace_detail"] = (
                        "compact_diagnostic"
                        if fixed_diagnostic_sample
                        else "compact_non_diagnostic"
                    )
                else:
                    trace["trace_detail"] = "full_debug"
                trace_instrumentation_seconds += time.perf_counter() - trace_started

            fidelity: FidelityDiagnostic | None = None
            fidelity_seconds = 0.0
            forecast_diagnostic_seconds = 0.0
            if (
                fixed_diagnostic_sample
                and self.config.diagnostics.forecast_oracle_steps > 0
                and plan is not None
            ):
                if predicted_token_attention is None:
                    raise HuggingFaceRuntimeError(
                        "MosaicKV oracle diagnostics are missing predicted attention"
                    )

                def run_forecast_diagnostic() -> (
                    ForecastQualityDiagnostics | AttentionSelectionDiagnostics
                ):
                    oracle = oracle_queries
                    if oracle is None:
                        oracle = collect_evaluation_only_true_future_queries(
                            self.adapter,
                            prefill,
                            future_steps=self.config.diagnostics.forecast_oracle_steps,
                            capture_attentions=True,
                        )
                    if plan.forecast is None:
                        return _selection_quality_from_oracle(
                            plan.full_state,
                            oracle,
                            predicted_token_attention,
                            plan.construction.exact_node_ids,
                        )
                    return _forecast_quality_from_oracle(
                        plan.full_state,
                        plan.forecast,
                        oracle,
                        predicted_token_attention,
                        plan.construction.exact_node_ids,
                    )

                forecast_diagnostic, forecast_diagnostic_seconds = _timed(
                    prefill.logits,
                    run_forecast_diagnostic,
                )
                forecast_payload = cast("JsonObject", asdict(forecast_diagnostic))
                forecast_payload["overhead_seconds"] = forecast_diagnostic_seconds
                forecast_payload["online_selection_isolation"] = (
                    "evaluation_only_oracle_is_the_declared_compressor_input"
                    if self.config.ablation.variant is AblationVariant.ORACLE_FUTURE_QUERY
                    else "evaluation_only_output_not_passed_to_compressor"
                )
                trace["forecast_quality_diagnostic"] = forecast_payload
            elif self.config.diagnostics.forecast_oracle_steps > 0:
                trace["forecast_quality_diagnostic"] = {
                    "status": (
                        "not_in_fixed_subset"
                        if not fixed_diagnostic_sample
                        else "not_applicable_to_non_mosaickv_method"
                    )
                }
            else:
                trace["forecast_quality_diagnostic"] = {"status": "disabled"}
            if (
                self.config.diagnostics.enabled
                and request.sample_id in self.config.diagnostics.sample_ids
            ):
                fidelity, fidelity_seconds = _timed(
                    prefill.logits,
                    lambda: _run_fidelity_diagnostic(
                        self.adapter,
                        prefill,
                        state,
                        max_new_tokens=self.config.generation.max_new_tokens,
                        prefixkv_plan=prefixkv_plan,
                    ),
                )
                if (
                    torch.cuda.is_available()
                    and getattr(self.adapter.device, "type", "cpu") == "cuda"
                ):
                    torch.cuda.synchronize(self.adapter.device)
                    torch.cuda.reset_peak_memory_stats(self.adapter.device)
                trace["fidelity_diagnostic"] = {
                    "status": "completed",
                    "sample_ledger_sha": self.config.diagnostics.sample_ledger_sha,
                    "token_agreement": fidelity.token_agreement,
                    "sequence_agreement": fidelity.sequence_agreement,
                    "logit_kl": fidelity.logit_kl,
                    "attention_output_error": fidelity.attention_output_error,
                    "attention_output_error_by_kv_head": [
                        {"layer": layer, "kv_head": head, "error": error}
                        for layer, head, error in fidelity.attention_output_error_by_kv_head
                    ],
                    "overhead_seconds": fidelity_seconds,
                    "trajectory": "independent_deterministic_greedy",
                    "attention_output_definition": "pre_output_projection_weighted_value",
                }
            elif self.config.diagnostics.enabled:
                trace["fidelity_diagnostic"] = {
                    "status": "not_in_fixed_subset",
                    "sample_ledger_sha": self.config.diagnostics.sample_ledger_sha,
                }
            else:
                trace["fidelity_diagnostic"] = {"status": "disabled"}

            # Observe before preallocation so unused backing capacity is not
            # confused with the post-compression byte target. All methods use
            # the same metadata-only instrumentation (no device copies/sync).
            cache_accounting_started = time.perf_counter()
            cache_lifecycle = CacheLifecycle.start(
                self.adapter._legacy_layers(state.past_key_values)[0]
            )
            trace_instrumentation_seconds += time.perf_counter() - cache_accounting_started
            preallocated_decode_cache = False
            if prefixkv_plan is None:
                preallocated_decode_cache = self.adapter.preallocate_decode_cache(
                    state,
                    self.config.generation.max_new_tokens - 1,
                )
            cache_accounting_started = time.perf_counter()
            cache_lifecycle.observe(self.adapter._legacy_layers(state.past_key_values)[0])
            trace_instrumentation_seconds += time.perf_counter() - cache_accounting_started
            decode_runtime_policy["kv_growth"] = (
                "preallocated_append_only_views"
                if preallocated_decode_cache
                else "backend_native_dynamic_growth"
            )
            decode_attention = (
                self.config.execution.decode_attention_implementation
                or self.config.execution.attention_implementation
            )
            if decode_attention != self.config.execution.attention_implementation:
                if repair_state is not None:
                    raise HuggingFaceRuntimeError(
                        "residual repair requires eager decode attention capture"
                    )
                if (
                    state.model_state.get("mosaickv_validity_masks") is not None
                    and decode_attention != "sdpa"
                ):
                    raise HuggingFaceRuntimeError(
                        "padded per-head cache masks are admitted only for eager or SDPA decode"
                    )
                self.adapter.set_attention_implementation(decode_attention)
            decode_runtime_policy["prefill_attention_implementation"] = (
                self.config.execution.attention_implementation
            )
            decode_runtime_policy["decode_attention_implementation"] = decode_attention

            if teacher_forced_ids:
                vocabulary = int(prefill.logits.shape[-1])
                first_target = teacher_forced_ids[0]
                if first_target >= vocabulary:
                    raise ValueError("teacher-forced target token exceeds model vocabulary")
                first_logits = prefill.logits.reshape(-1, vocabulary)[-1].float()
                teacher_forced_nll_sum += float(
                    torch.nn.functional.cross_entropy(
                        first_logits.reshape(1, -1),
                        torch.tensor(
                            [first_target],
                            dtype=torch.long,
                            device=first_logits.device,
                        ),
                        reduction="sum",
                    ).item()
                )
                token = torch.full_like(prefill.next_token_id, first_target)
                tokens = [token]
            else:
                tokens = [prefill.next_token_id]
                token = prefill.next_token_id
            decode_seconds = 0.0
            oracle_repair_diagnostic_seconds = 0.0
            per_token_decode_seconds: list[float] = []
            decode_finite_flags: list[Any] = []
            repair_events: list[RepairEvent] = []
            draft_repair_prefix_matches = True
            prefixkv_decode_events: list[JsonObject] = []
            elastic_cache_decode_events: list[JsonObject] = []
            oracle_reference_state = (
                _clone_decode_state(self.adapter, prefill.state)
                if repair_state is not None and self.config.repair.policy is RepairPolicy.ORACLE
                else None
            )
            decode_token_count = (
                len(teacher_forced_ids)
                if teacher_forced_ids
                else self.config.generation.max_new_tokens
            )
            for step_index in range(1, decode_token_count):
                # Hugging Face cache objects normally grow in place.  A
                # provisional repair step therefore needs an isolated copy so
                # it can be discarded and re-decoded, but every non-repair
                # path can advance the live state directly.  Cloning the full
                # cache unconditionally here made ordinary token decoding
                # O(sequence_length) in host-side copy work per step.
                transactional_repair = bool(repair_state is not None and preallocated_decode_cache)
                previous_active_cache_length = state.active_cache_length
                previous_logical_sequence_length = state.logical_sequence_length
                previous_next_decode_position = state.next_decode_position
                before = (
                    state
                    if repair_state is None or transactional_repair
                    else _clone_decode_state(self.adapter, state)
                )
                decode_token = token
                needs_attention = repair_state is not None
                oracle_reference = None
                if oracle_reference_state is not None:
                    oracle_reference, oracle_seconds = _timed(
                        decode_token,
                        lambda current=decode_token, current_state=oracle_reference_state: (
                            self.adapter.decode_one_token(
                                current,
                                current_state,
                                capture_queries=False,
                                capture_attentions=False,
                            )
                        ),
                    )
                    oracle_repair_diagnostic_seconds += oracle_seconds
                    oracle_reference_state = oracle_reference.state
                provisional, step_seconds = _timed(
                    decode_token,
                    lambda current=decode_token, current_state=before, capture=needs_attention: (
                        self.adapter.decode_one_token(
                            current,
                            current_state,
                            capture_queries=False,
                            capture_attentions=capture,
                        )
                    ),
                )
                decode_seconds += step_seconds
                per_token_decode_seconds.append(step_seconds)
                cache_accounting_started = time.perf_counter()
                cache_lifecycle.observe(
                    self.adapter._legacy_layers(provisional.state.past_key_values)[0]
                )
                trace_instrumentation_seconds += time.perf_counter() - cache_accounting_started
                # Queue device-side validation and reduce once after the
                # sequence.  Converting every scalar flag to bool here forced
                # a second CUDA synchronization on every token in addition to
                # the required timing boundary.
                decode_finite_flags.append(torch.isfinite(provisional.logits).all())
                if repair_state is None:
                    if prefixkv_plan is not None:

                        def maintain_prefix(
                            output: Any = provisional,
                            current_step: int = step_index,
                        ) -> tuple[DecodeState, tuple[JsonObject, ...]]:
                            return maintain_prefixkv_decode_cache(
                                self.adapter,
                                output.state,
                                prefixkv_plan,
                                step_index=current_step,
                            )

                        maintained, maintenance_seconds = _timed(
                            provisional.logits,
                            maintain_prefix,
                        )
                        state, maintenance_events = maintained
                        prefixkv_decode_events.extend(maintenance_events)
                        decode_seconds += maintenance_seconds
                        per_token_decode_seconds[-1] += maintenance_seconds
                    elif (
                        published_kv_plan is not None
                        and published_kv_plan.method is MosaicKVMethod.ELASTIC_CACHE_REIMPL
                    ):

                        def maintain_elastic(
                            output: Any = provisional,
                            current_step: int = step_index,
                        ) -> tuple[DecodeState, tuple[JsonObject, ...]]:
                            return maintain_elastic_cache_decode_cache(
                                self.adapter,
                                output.state,
                                published_kv_plan,
                                step_index=current_step,
                            )

                        maintained, maintenance_seconds = _timed(
                            provisional.logits,
                            maintain_elastic,
                        )
                        state, maintenance_events = maintained
                        elastic_cache_decode_events.extend(maintenance_events)
                        decode_seconds += maintenance_seconds
                        per_token_decode_seconds[-1] += maintenance_seconds
                    else:
                        state = provisional.state
                    cache_accounting_started = time.perf_counter()
                    cache_lifecycle.observe(self.adapter._legacy_layers(state.past_key_values)[0])
                    trace_instrumentation_seconds += time.perf_counter() - cache_accounting_started
                    if teacher_forced_ids:
                        target_id = teacher_forced_ids[step_index]
                        vocabulary = int(provisional.logits.shape[-1])
                        if target_id >= vocabulary:
                            raise ValueError("teacher-forced target token exceeds model vocabulary")
                        step_logits = provisional.logits.reshape(-1, vocabulary)[-1].float()
                        teacher_forced_nll_sum += float(
                            torch.nn.functional.cross_entropy(
                                step_logits.reshape(1, -1),
                                torch.tensor(
                                    [target_id],
                                    dtype=torch.long,
                                    device=step_logits.device,
                                ),
                                reduction="sum",
                            ).item()
                        )
                        token = torch.full_like(provisional.next_token_id, target_id)
                    else:
                        token = provisional.next_token_id
                    tokens.append(token)
                    continue

                if plan is None:
                    raise HuggingFaceRuntimeError(
                        "repair state exists without a MosaicKV compression plan"
                    )

                prototype_slots = cast(
                    "tuple[tuple[int, int, int, int], ...]",
                    before.model_state.get("mosaickv_prototype_slots", ()),
                )
                raw_validity_masks = before.model_state.get("mosaickv_validity_masks")
                if raw_validity_masks is None:
                    if before.model_state.get("mosaickv_packed_all_valid") is not True:
                        raise HuggingFaceRuntimeError(
                            "repair cache is missing packed validity metadata"
                        )
                    # The dense fast path intentionally omits per-layer masks.
                    # Repair only needs their KV-head dimension to map grouped
                    # query heads back to prototype slots, so synthesize the
                    # all-valid shape without reintroducing a decode-time mask.
                    raw_validity_masks = tuple(
                        torch.ones(
                            (layer.kv_heads, before.active_cache_length),
                            dtype=torch.bool,
                            device=provisional.logits.device,
                        )
                        for layer in plan.full_state.layers
                    )
                validity_masks = cast("tuple[Any, ...]", raw_validity_masks)
                masses = _prototype_attention_mass(
                    provisional.attention_weights,
                    prototype_slots,
                    validity_masks,
                )
                repaired_output: list[Any] = []
                repair_base_state = state
                repair_token = token
                repair_sink = repaired_output
                draft_distribution = None
                draft_offset = step_index - 1
                if draft_offset < len(draft_repair_distributions):
                    if draft_offset >= len(draft_repair_token_ids):
                        raise HuggingFaceRuntimeError(
                            "captured draft distributions are missing their prefix token IDs"
                        )
                    draft_input_id = int(decode_token.detach().reshape(-1)[0].item())
                    draft_repair_prefix_matches = bool(
                        draft_repair_prefix_matches
                        and draft_input_id == draft_repair_token_ids[draft_offset]
                    )
                    if draft_repair_prefix_matches:
                        draft_distribution = draft_repair_distributions[draft_offset]
                else:
                    draft_repair_prefix_matches = False

                def re_decode(
                    updated_repair: RepairCacheState,
                    base_state: DecodeState = repair_base_state,
                    decode_token: Any = repair_token,
                    sink: list[Any] = repair_sink,
                    use_transaction: bool = transactional_repair,
                    old_active_length: int = previous_active_cache_length,
                    old_logical_length: int = previous_logical_sequence_length,
                    old_next_position: int = previous_next_decode_position,
                    current_step_index: int = step_index,
                ) -> Any:
                    if use_transaction and not self.adapter.rollback_preallocated_decode_step(
                        base_state,
                        active_cache_length=old_active_length,
                        logical_sequence_length=old_logical_length,
                        next_decode_position=old_next_position,
                    ):
                        raise HuggingFaceRuntimeError(
                            "transactional repair could not rollback its provisional cache append"
                        )
                    old_snapshot = self.adapter.extract_past_key_values(base_state.past_key_values)
                    old_capacity = int(base_state.model_state["mosaickv_prompt_capacity"])
                    repacked = pack_runtime_cache(
                        self.adapter,
                        updated_repair.full_state,
                        updated_repair.mosaic_state,
                        updated_repair.active_prototype_records,
                        updated_repair.active_prototype_ids,
                        tail_snapshot=old_snapshot,
                        old_prompt_capacity=old_capacity,
                        packing_mode=self.config.cache.packing_mode,
                    )
                    repacked_state = _state_from_packed(self.adapter, base_state, repacked)
                    self.adapter.preallocate_decode_cache(
                        repacked_state,
                        max(1, self.config.generation.max_new_tokens - current_step_index),
                    )
                    output = self.adapter.decode_one_token(
                        decode_token,
                        repacked_state,
                        capture_queries=False,
                        capture_attentions=True,
                    )
                    sink.append(output)
                    return output.logits

                if self.config.repair.policy is RepairPolicy.ORACLE:
                    if oracle_reference is None:
                        raise HuggingFaceRuntimeError(
                            "oracle repair is missing its evaluation-only FullKV reference"
                        )
                    provisional_id = int(provisional.next_token_id.item())
                    reference_id = int(oracle_reference.next_token_id.item())
                    repair = evaluate_repair_decode_step(
                        repair_state,
                        self.config.repair,
                        step_index=step_index,
                        provisional_logits=provisional.logits,
                        prototype_attention_mass=masses,
                        re_decode=re_decode,
                        reference_token_id=reference_id,
                        oracle_should_repair=provisional_id != reference_id,
                        draft_distribution=draft_distribution,
                        key_transform=key_transform,
                    )
                else:
                    repair = repair_decode_step(
                        repair_state,
                        self.config.repair,
                        step_index=step_index,
                        provisional_logits=provisional.logits,
                        prototype_attention_mass=masses,
                        re_decode=re_decode,
                        draft_distribution=draft_distribution,
                        key_transform=key_transform,
                    )
                repair_state = repair.state
                repair_events.append(repair.event)
                if repair.event.triggered:
                    if len(repaired_output) != 1:
                        raise HuggingFaceRuntimeError("triggered repair did not re-decode once")
                    final = repaired_output[0]
                    state = final.state
                    token = final.next_token_id
                    scoring_logits = final.logits
                    decode_finite_flags.append(torch.isfinite(final.logits).all())
                else:
                    state = provisional.state
                    token = provisional.next_token_id
                    scoring_logits = provisional.logits
                cache_accounting_started = time.perf_counter()
                cache_lifecycle.observe(self.adapter._legacy_layers(state.past_key_values)[0])
                trace_instrumentation_seconds += time.perf_counter() - cache_accounting_started
                if teacher_forced_ids:
                    target_id = teacher_forced_ids[step_index]
                    vocabulary = int(scoring_logits.shape[-1])
                    if target_id >= vocabulary:
                        raise ValueError("teacher-forced target token exceeds model vocabulary")
                    step_logits = scoring_logits.reshape(-1, vocabulary)[-1].float()
                    teacher_forced_nll_sum += float(
                        torch.nn.functional.cross_entropy(
                            step_logits.reshape(1, -1),
                            torch.tensor(
                                [target_id],
                                dtype=torch.long,
                                device=step_logits.device,
                            ),
                            reduction="sum",
                        ).item()
                    )
                    token = torch.full_like(token, target_id)
                tokens.append(token)

            if decode_finite_flags:
                finite_logits, validation_seconds = _timed(
                    prefill.logits,
                    lambda: torch.stack(decode_finite_flags).all(),
                )
                trace_instrumentation_seconds += validation_seconds
                if not bool(finite_logits):
                    raise HuggingFaceRuntimeError("decode logits contain NaN or infinity")

            token_ids = torch.cat(tokens, dim=-1)
            timed_token_ids = tuple(
                int(value) for value in token_ids.detach().cpu().reshape(-1).tolist()
            )
            quality = decode_quality_output(
                self.adapter.processor,
                timed_token_ids,
                self._terminal_token_ids,
            )
            answer = quality.text
            live_cache = self.adapter.extract_past_key_values(state.past_key_values, clone=False)
            active_bytes = sum(
                tensor_storage_bytes(key) + tensor_storage_bytes(value)
                for key, value in self.adapter._legacy_layers(live_cache)[0]
            )
            residual_bytes = (
                repair_state.residual_storage.cpu_bytes
                if repair_state is not None
                else (plan.construction.residual_storage.cpu_bytes if plan is not None else 0)
            )
            total_seconds = max(
                0.0,
                time.perf_counter()
                - started
                - fidelity_seconds
                - forecast_diagnostic_seconds
                - oracle_repair_diagnostic_seconds
                - trace_instrumentation_seconds,
            )
            if deferred_trace_builder is not None:
                trace_started = time.perf_counter()
                trace.update(deferred_trace_builder())
                if not full_debug_trace:
                    _compact_trace_records(trace)
                    trace["trace_detail"] = (
                        "compact_diagnostic"
                        if fixed_diagnostic_sample
                        else "compact_non_diagnostic"
                    )
                else:
                    trace["trace_detail"] = "full_debug"
                trace_instrumentation_seconds += time.perf_counter() - trace_started
            repair_transfer = sum(event.transfer_time_ms for event in repair_events) / 1000.0
            repair_redecode = sum(event.re_decode_time_ms for event in repair_events) / 1000.0
            first_decode_step = per_token_decode_seconds[0] if per_token_decode_seconds else None
            steady_state_decode_mean = (
                float(np.mean(per_token_decode_seconds[1:]))
                if len(per_token_decode_seconds) > 1
                else None
            )
            # CUDA-event phase timings intentionally exclude host-only work in
            # mixed Python/CUDA actions.  Preserve that difference explicitly
            # instead of allowing the named phases to appear to sum to total
            # latency.  Repair transfer and re-decode are not included in
            # ``decode_seconds`` and therefore belong in the accounted sum.
            accounted_seconds = (
                prefill_seconds
                + cache_state_construction_seconds
                + forecast_seconds
                + planner_preparation_seconds
                + graph_seconds
                + selection_seconds
                + tier_seconds
                + logit_stability_seconds
                + progressive_eviction_seconds
                + hierarchical_eviction_seconds
                + packing_seconds
                + decode_seconds
                + repair_transfer
                + repair_redecode
            )
            host_orchestration_seconds = max(0.0, total_seconds - accounted_seconds)
            timings = RuntimePhaseTimings(
                prefill=prefill_seconds,
                image_video_encoder=image_video_encoder_seconds,
                projector=projector_seconds,
                language_model_prefill=language_model_prefill_seconds,
                cache_state_construction=cache_state_construction_seconds,
                forecast=forecast_seconds,
                planner_preparation=planner_preparation_seconds,
                graph=graph_seconds,
                utility_and_selection=selection_seconds,
                tier_construction=tier_seconds,
                logit_stability=logit_stability_seconds,
                progressive_eviction=progressive_eviction_seconds,
                hierarchical_eviction=hierarchical_eviction_seconds,
                cache_packing=packing_seconds,
                decode=decode_seconds,
                per_token_decode=tuple(per_token_decode_seconds),
                first_decode_step=first_decode_step,
                steady_state_decode_mean=steady_state_decode_mean,
                repair_transfer=repair_transfer,
                repair_redecode=repair_redecode,
                host_orchestration=host_orchestration_seconds,
                total=total_seconds,
            )
            timing_payload = cast("JsonObject", asdict(timings))
            timing_payload["trace_instrumentation"] = trace_instrumentation_seconds
            if repair_events and plan is None:
                raise HuggingFaceRuntimeError("repair events are missing their compression plan")
            repair_event_records = (
                []
                if plan is None
                else [
                    _repair_event_trace(event, plan, compact=not full_debug_trace)
                    for event in repair_events
                ]
            )
            residual_configuration: JsonObject | None = None
            if plan is not None:
                residual_payloads = plan.construction.residual_storage.payloads
                actual_placements = sorted(
                    {
                        (
                            "cpu_pinned"
                            if payload.key_pinned and payload.value_pinned
                            else "cpu_pageable"
                        )
                        for payload in residual_payloads
                    }
                )
                residual_configuration = {
                    "storage_dtype": self.config.residual.storage_dtype.value,
                    "requested_placement": (
                        "cpu_pinned"
                        if self.config.residual.require_pinned_memory
                        else "cpu_pageable"
                    ),
                    "actual_placements": cast("list[JsonValue]", actual_placements),
                    "all_payloads_pinned": (
                        plan.construction.residual_storage.all_payloads_pinned
                        if residual_payloads
                        else None
                    ),
                    "cpu_bytes_initial": plan.construction.residual_storage.cpu_bytes,
                    "eligible_source_blocks": len(plan.construction.state.prototypes.source_blocks),
                    "stored_source_blocks": len(residual_payloads),
                    "max_bytes": self.config.residual.max_bytes,
                    "max_source_blocks": self.config.residual.max_source_blocks,
                    "selection_policy": self.config.residual.selection_policy,
                    "total_storage_budget_bytes": (self.config.cache.total_storage_budget_bytes),
                    "active_plus_residual_bytes_initial": (
                        plan.construction.state.statistics.active_kv_bytes
                        + plan.construction.residual_storage.cpu_bytes
                    ),
                    "total_storage_budget_satisfied": (
                        self.config.cache.total_storage_budget_bytes is None
                        or (
                            plan.construction.state.statistics.active_kv_bytes
                            + plan.construction.residual_storage.cpu_bytes
                            <= self.config.cache.total_storage_budget_bytes
                        )
                    ),
                }
            byte_budget_materialization = _byte_budget_materialization_trace(
                dense_budget,
                self.config,
                source_prefill_kv_bytes=source_prefill_kv_bytes,
                logical_prefill_active_kv_bytes=logical_prefill_active_kv_bytes,
                packed_prefill_active_kv_bytes=packed_prefill_active_kv_bytes,
                packed_prompt_capacity=packed_prompt_capacity,
                plan=plan,
            )
            trace.update(
                {
                    "status": "completed",
                    "answer": answer,
                    "generated_token_ids": list(timed_token_ids),
                    "teacher_forced_evaluation": {
                        "enabled": bool(teacher_forced_ids),
                        "evaluated_tokens": len(teacher_forced_ids),
                        "negative_log_likelihood_sum": (
                            teacher_forced_nll_sum if teacher_forced_ids else None
                        ),
                        "target_was_not_truncated": True,
                    },
                    "quality_decoding": {
                        **quality.to_record(),
                        "configured_terminal_token_ids": list(self._terminal_token_ids),
                    },
                    "prompt_evidence": prompt_evidence.trace() if prompt_evidence else None,
                    "repair_events": cast("list[JsonValue]", repair_event_records),
                    "residual_configuration": residual_configuration,
                    "oracle_repair_diagnostic": {
                        "status": (
                            "completed_evaluation_only"
                            if oracle_reference_state is not None
                            else "not_used"
                        ),
                        "overhead_seconds_excluded_from_online_total": (
                            oracle_repair_diagnostic_seconds
                        ),
                        "trigger_definition": (
                            "provisional_token_differs_from_same_prefix_fullkv"
                            if oracle_reference_state is not None
                            else None
                        ),
                    },
                    "repair_draft_distribution": {
                        "available_steps": len(draft_repair_distributions),
                        "used_only_while_prefix_matches": True,
                        "final_prefix_match_state": draft_repair_prefix_matches,
                    },
                    "prefixkv_decode_events": cast("list[JsonValue]", prefixkv_decode_events),
                    "elastic_cache_decode_events": cast(
                        "list[JsonValue]", elastic_cache_decode_events
                    ),
                    "short_horizon_logit_stability": (
                        {"status": "disabled"}
                        if logit_stability_decision is None
                        else {
                            "status": "completed_label_blind",
                            **cast("JsonObject", asdict(logit_stability_decision)),
                            "overhead_seconds": logit_stability_seconds,
                            "trajectory": "fullkv_teacher_forced_short_horizon",
                            "probe_state_discarded_before_measured_decode": True,
                        }
                    ),
                    "progressive_logit_safe_eviction": (
                        {"status": "disabled"}
                        if progressive_eviction_decision is None
                        else {
                            "status": "completed_label_blind",
                            **cast(
                                "JsonObject",
                                asdict(progressive_eviction_decision),
                            ),
                            "overhead_seconds": progressive_eviction_seconds,
                            "trajectory": "progressive_fullkv_to_target_teacher_forced",
                            "probe_state_discarded_before_measured_decode": True,
                        }
                    ),
                    "hierarchical_logit_safe_eviction": (
                        {"status": "disabled"}
                        if hierarchical_eviction_decision is None
                        else {
                            "status": "completed_label_blind",
                            **cast(
                                "JsonObject",
                                asdict(hierarchical_eviction_decision),
                            ),
                            "overhead_seconds": hierarchical_eviction_seconds,
                            "trajectory": "bounded_hierarchical_fullkv_teacher_forced",
                            "probe_state_discarded_before_measured_decode": True,
                        }
                    ),
                    "timing_breakdown": timing_payload,
                    "planner_subphases": cast(
                        "JsonObject",
                        {
                            "utility": 0.0 if plan is None else plan.utility_seconds,
                            "hard_reservations": (
                                0.0 if plan is None else plan.reservation_seconds
                            ),
                            "budgeted_selector": (0.0 if plan is None else plan.selector_seconds),
                            "short_horizon_logit_stability": logit_stability_seconds,
                            "progressive_logit_safe_eviction": (progressive_eviction_seconds),
                            "hierarchical_logit_safe_eviction": (hierarchical_eviction_seconds),
                            "stable_head_routing": (
                                0.0 if plan is None else plan.stable_routing_seconds
                            ),
                            "similarity_shortlist": (
                                0.0 if plan is None else plan.similarity_shortlist_seconds
                            ),
                            "planner_source_mapping": (
                                0.0 if plan is None else plan.planner_mapping_seconds
                            ),
                            "forecast_attention_projection": (
                                forecast_attention_projection_seconds
                            ),
                            "prompt_routing_evidence": prompt_routing_evidence_seconds,
                            "head_route_configuration": (head_route_configuration_seconds),
                            "unattributed_preparation": max(
                                0.0,
                                planner_preparation_seconds
                                - forecast_attention_projection_seconds
                                - prompt_routing_evidence_seconds
                                - head_route_configuration_seconds
                                - (0.0 if plan is None else plan.stable_routing_seconds)
                                - (0.0 if plan is None else plan.similarity_shortlist_seconds)
                                - (0.0 if plan is None else plan.planner_mapping_seconds),
                            ),
                        },
                    ),
                    "source_prefill_kv_bytes": source_prefill_kv_bytes,
                    "logical_prefill_active_kv_bytes": logical_prefill_active_kv_bytes,
                    "packed_prefill_active_kv_bytes": packed_prefill_active_kv_bytes,
                    "packed_padding_kv_bytes": (
                        packed_prefill_active_kv_bytes - logical_prefill_active_kv_bytes
                    ),
                    "logical_cache_utilization": (
                        logical_prefill_active_kv_bytes / packed_prefill_active_kv_bytes
                    ),
                    "physical_budget_utilization": (
                        1.0
                        if dense_budget is None
                        else packed_prefill_active_kv_bytes / dense_budget.target_bytes
                    ),
                    "hard_active_kv_budget_bytes": (
                        None if dense_budget is None else dense_budget.target_bytes
                    ),
                    "byte_budget_materialization": byte_budget_materialization,
                    "packed_prompt_capacity": packed_prompt_capacity,
                    "hard_prompt_capacity": (
                        None if dense_budget is None else dense_budget.prompt_capacity
                    ),
                    "tier_active_cache_lengths": cast(
                        "list[JsonValue]",
                        []
                        if plan is None
                        else sorted(
                            {
                                layout.active_cache_length
                                for layout in plan.construction.active_layouts
                            }
                        ),
                    ),
                    "online_uncertainty_routing": cast(
                        "JsonObject",
                        {
                            "metric": "prompt_draft_attention_total_variation",
                            "configured_exact_head_fraction": (
                                self.config.forecasting.uncertainty_exact_head_fraction
                            ),
                            "selected_exact_head_routes": cast(
                                "list[JsonValue]",
                                [
                                    [int(value) for value in route.split(":")]
                                    for route in online_exact_head_routes
                                ],
                            ),
                            "head_shift_count": len(online_head_shifts),
                            "prototype_head_routes": cast(
                                "list[JsonValue]",
                                [
                                    [int(value) for value in route.split(":")]
                                    for route in online_prototype_head_routes
                                ],
                            ),
                            "stable_head_prompt_routing": (
                                self.config.selection.stable_head_prompt_routing
                            ),
                            "stable_head_uniform_fill": (
                                self.config.selection.stable_head_uniform_fill
                            ),
                            "stable_head_visual_fraction": (
                                self.config.selection.stable_head_visual_fraction
                            ),
                            "stable_head_adaptive_visual_swaps": (
                                self.config.selection.stable_head_adaptive_visual_swaps
                            ),
                            "stable_head_minimum_text_mass_fraction": (
                                self.config.selection.stable_head_minimum_text_mass_fraction
                            ),
                            "stable_head_visual_gain_ratio": (
                                self.config.selection.stable_head_visual_gain_ratio
                            ),
                            "stable_head_maximum_visual_swaps": (
                                self.config.selection.stable_head_maximum_visual_swaps
                            ),
                            "stable_head_swap_risk_gate": (
                                self.config.selection.stable_head_swap_risk_gate
                            ),
                            "stable_head_maximum_normalized_text_loss": (
                                self.config.selection.stable_head_maximum_normalized_text_loss
                            ),
                            "stable_head_forecast_gain_ratio": (
                                self.config.selection.stable_head_forecast_gain_ratio
                            ),
                            "stable_head_future_query_reranking": (
                                self.config.selection.stable_head_future_query_reranking
                            ),
                            "stable_head_future_minimum_prompt_mass_fraction": (
                                self.config.selection.stable_head_future_minimum_prompt_mass_fraction
                            ),
                            "stable_head_future_gain_ratio": (
                                self.config.selection.stable_head_future_gain_ratio
                            ),
                            "stable_head_maximum_future_replacements": (
                                self.config.selection.stable_head_maximum_future_replacements
                            ),
                            "stable_head_future_divergence_fail_closed": (
                                self.config.selection.stable_head_future_divergence_fail_closed
                            ),
                            "stable_head_future_minimum_prompt_forecast_tv": (
                                self.config.selection.stable_head_future_minimum_prompt_forecast_tv
                            ),
                            "stable_head_future_risk_gate": (
                                self.config.selection.stable_head_future_risk_gate
                            ),
                            "stable_head_future_maximum_normalized_prompt_loss": (
                                self.config.selection.stable_head_future_maximum_normalized_prompt_loss
                            ),
                            "stable_head_future_minimum_normalized_gain": (
                                self.config.selection.stable_head_future_minimum_normalized_gain
                            ),
                            "stable_head_query_coverage_reranking": (
                                self.config.selection.stable_head_query_coverage_reranking
                            ),
                            "stable_head_query_coverage_minimum_prompt_mass_fraction": (
                                self.config.selection.stable_head_query_coverage_minimum_prompt_mass_fraction
                            ),
                            "stable_head_query_coverage_maximum_replacements": (
                                self.config.selection.stable_head_query_coverage_maximum_replacements
                            ),
                            "stable_head_query_coverage_minimum_normalized_gain": (
                                self.config.selection.stable_head_query_coverage_minimum_normalized_gain
                            ),
                            "stable_head_query_coverage_initial_selection": (
                                self.config.selection.stable_head_query_coverage_initial_selection
                            ),
                            "stable_head_query_coverage_initial_fraction": (
                                self.config.selection.stable_head_query_coverage_initial_fraction
                            ),
                            "stable_head_query_coverage_initial_prompt_weight": (
                                self.config.selection.stable_head_query_coverage_initial_prompt_weight
                            ),
                            "stable_head_query_coverage_initial_minimum_prompt_mass_fraction": (
                                self.config.selection.stable_head_query_coverage_initial_minimum_prompt_mass_fraction
                            ),
                            "stable_head_query_coverage_initial_objective": (
                                self.config.selection.stable_head_query_coverage_initial_objective
                            ),
                            "stable_head_query_coverage_initial_cap_fraction": (
                                self.config.selection.stable_head_query_coverage_initial_cap_fraction
                            ),
                            "stable_head_query_coverage_cross_head_allocation": (
                                self.config.selection.stable_head_query_coverage_cross_head_allocation
                            ),
                            "stable_head_query_coverage_cross_head_stability_gate": (
                                self.config.selection.stable_head_query_coverage_cross_head_stability_gate
                            ),
                            "stable_head_query_coverage_cross_head_stability_risk_mode": (
                                self.config.selection.stable_head_query_coverage_cross_head_stability_risk_mode
                            ),
                            "stable_head_query_coverage_cross_head_minimum_normalized_gain": (
                                self.config.selection.stable_head_query_coverage_cross_head_minimum_normalized_gain
                            ),
                            "stable_head_query_coverage_cross_head_gain_to_prompt_loss_ratio": (
                                self.config.selection.stable_head_query_coverage_cross_head_gain_to_prompt_loss_ratio
                            ),
                            "stable_head_query_coverage_output_impact_reservation": (
                                self.config.selection.stable_head_query_coverage_output_impact_reservation
                            ),
                            "stable_head_query_coverage_output_impact_fraction": (
                                self.config.selection.stable_head_query_coverage_output_impact_fraction
                            ),
                            "stable_head_query_coverage_layout_group_reservation": (
                                self.config.selection.stable_head_query_coverage_layout_group_reservation
                            ),
                            "stable_head_query_coverage_layout_group_fraction": (
                                self.config.selection.stable_head_query_coverage_layout_group_fraction
                            ),
                            "stable_head_query_coverage_layout_group_maximum_blocks": (
                                self.config.selection.stable_head_query_coverage_layout_group_maximum_blocks
                            ),
                            "stable_head_query_coverage_layout_group_row_bins": (
                                self.config.selection.stable_head_query_coverage_layout_group_row_bins
                            ),
                            "stable_head_query_coverage_layout_group_atomic_selection": (
                                self.config.selection.stable_head_query_coverage_layout_group_atomic_selection
                            ),
                            "stable_head_query_coverage_layout_group_atomic_selection_mode": (
                                self.config.selection.stable_head_query_coverage_layout_group_atomic_selection_mode
                            ),
                            (
                                "stable_head_query_coverage_layout_group_"
                                "minimum_utilization_fraction"
                            ): (
                                self.config.selection.stable_head_query_coverage_layout_group_minimum_utilization_fraction
                            ),
                            "stable_head_query_coverage_layout_group_minimum_normalized_gain": (
                                self.config.selection.stable_head_query_coverage_layout_group_minimum_normalized_gain
                            ),
                            "stable_head_query_coverage_layout_group_gain_to_risk_ratio": (
                                self.config.selection.stable_head_query_coverage_layout_group_gain_to_risk_ratio
                            ),
                            "stable_head_short_horizon_logit_stability": (
                                self.config.selection.stable_head_short_horizon_logit_stability
                            ),
                            "stable_head_logit_stability_shortlist_size": (
                                self.config.selection.stable_head_logit_stability_shortlist_size
                            ),
                            "stable_head_logit_stability_horizon": (
                                self.config.selection.stable_head_logit_stability_horizon
                            ),
                            "stable_head_logit_stability_maximum_kl": (
                                self.config.selection.stable_head_logit_stability_maximum_kl
                            ),
                            "stable_head_logit_stability_maximum_logit_difference": (
                                self.config.selection.stable_head_logit_stability_maximum_logit_difference
                            ),
                            "stable_head_progressive_logit_safe_eviction": (
                                self.config.selection.stable_head_progressive_logit_safe_eviction
                            ),
                            "stable_head_progressive_eviction_initial_batch_blocks": (
                                self.config.selection.stable_head_progressive_eviction_initial_batch_blocks
                            ),
                            "stable_head_progressive_eviction_maximum_probes": (
                                self.config.selection.stable_head_progressive_eviction_maximum_probes
                            ),
                            "stable_head_progressive_eviction_horizon": (
                                self.config.selection.stable_head_progressive_eviction_horizon
                            ),
                            "stable_head_progressive_eviction_maximum_kl": (
                                self.config.selection.stable_head_progressive_eviction_maximum_kl
                            ),
                            "stable_head_progressive_eviction_maximum_logit_difference": (
                                self.config.selection.stable_head_progressive_eviction_maximum_logit_difference
                            ),
                            "stable_head_hierarchical_logit_safe_eviction": (
                                self.config.selection.stable_head_hierarchical_logit_safe_eviction
                            ),
                            "stable_head_hierarchical_partition_count": (
                                self.config.selection.stable_head_hierarchical_partition_count
                            ),
                            "stable_head_hierarchical_maximum_refinement_depth": (
                                self.config.selection.stable_head_hierarchical_maximum_refinement_depth
                            ),
                            "stable_head_hierarchical_maximum_candidate_evaluations": (
                                self.config.selection.stable_head_hierarchical_maximum_candidate_evaluations
                            ),
                            "stable_head_hierarchical_maximum_sequential_rounds": (
                                self.config.selection.stable_head_hierarchical_maximum_sequential_rounds
                            ),
                            "stable_head_hierarchical_maximum_rollback_rounds": (
                                self.config.selection.stable_head_hierarchical_maximum_rollback_rounds
                            ),
                            "stable_head_hierarchical_horizon": (
                                self.config.selection.stable_head_hierarchical_horizon
                            ),
                            "stable_head_hierarchical_maximum_kl": (
                                self.config.selection.stable_head_hierarchical_maximum_kl
                            ),
                            "stable_head_hierarchical_maximum_logit_difference": (
                                self.config.selection.stable_head_hierarchical_maximum_logit_difference
                            ),
                            "stable_head_hierarchical_reserved_reallocation_evaluations": (
                                self.config.selection.stable_head_hierarchical_reserved_reallocation_evaluations
                            ),
                            "stable_head_hierarchical_reallocation_beam_width": (
                                self.config.selection.stable_head_hierarchical_reallocation_beam_width
                            ),
                            "stable_head_hierarchical_reallocation_maximum_combination_width": (
                                self.config.selection.stable_head_hierarchical_reallocation_maximum_combination_width
                            ),
                            "stable_head_hierarchical_logit_feedback_restore_donor": (
                                self.config.selection.stable_head_hierarchical_logit_feedback_restore_donor
                            ),
                            "hierarchical_gpu_debug_authorized": (
                                self._hierarchical_gpu_debug_authorized
                            ),
                            "routed_head_only_planning": (
                                self.config.selection.routed_head_only_planning
                            ),
                            "counterfactual_draft_head_routing": bool(counterfactual_head_shifts),
                        },
                    ),
                    "active_kv_bytes": packed_prefill_active_kv_bytes,
                    "final_active_kv_bytes": active_bytes,
                    "cache_lifecycle": cache_lifecycle.record(),
                    "residual_kv_bytes": residual_bytes,
                    "gpu_rescue_kv_bytes": (
                        0 if repair_state is None else repair_state.gpu_rescue_bytes
                    ),
                    "gpu_rescue_initial_kv_bytes": gpu_rescue_initial_bytes,
                    "initial_total_gpu_kv_bytes": (
                        packed_prefill_active_kv_bytes + gpu_rescue_initial_bytes
                    ),
                    "gpu_rescue_staging": {
                        "configured_max_blocks": self.config.residual.gpu_rescue_max_blocks,
                        "configured_max_bytes": self.config.residual.gpu_rescue_max_bytes,
                        "staging_seconds": gpu_rescue_staging_seconds,
                        "transfer_time_ms": gpu_rescue_transfer_ms,
                        "transfer_was_asynchronous": gpu_rescue_transfer_async,
                    },
                    "peak_gpu_memory_allocated": (
                        int(torch.cuda.max_memory_allocated(self.adapter.device))
                        if is_cuda
                        else None
                    ),
                    "peak_gpu_memory_reserved": (
                        int(torch.cuda.max_memory_reserved(self.adapter.device))
                        if is_cuda
                        else None
                    ),
                    "trace_path": str(trace_path.resolve()),
                }
            )
            _write_trace(trace_path, trace)
            compression_seconds = (
                cache_state_construction_seconds
                + forecast_seconds
                + planner_preparation_seconds
                + graph_seconds
                + selection_seconds
                + tier_seconds
                + logit_stability_seconds
                + progressive_eviction_seconds
                + hierarchical_eviction_seconds
                + packing_seconds
            )
            peak_memory = (
                int(torch.cuda.max_memory_allocated(self.adapter.device))
                if torch.cuda.is_available()
                and getattr(self.adapter.device, "type", "cpu") == "cuda"
                else None
            )
            effective_method = str(trace.get("effective_method", self.method))
            if teacher_forced_ids:
                self._teacher_forced_outputs[(request.run_id, request.sample_id)] = (
                    TeacherForcedNll(
                        negative_log_likelihood_sum=teacher_forced_nll_sum,
                        evaluated_tokens=len(teacher_forced_ids),
                    )
                )
            return ModelGeneration(
                answer,
                GenerationMetrics(
                    ttft=prefill_seconds + compression_seconds,
                    prefill_time=prefill_seconds,
                    compression_time=compression_seconds,
                    decode_time=decode_seconds,
                    end_to_end_time=total_seconds,
                    generated_tokens=int(token_ids.numel()),
                    active_kv_bytes=packed_prefill_active_kv_bytes,
                    initial_active_kv_storage_bytes=cache_lifecycle.initial_storage_bytes,
                    final_active_kv_storage_bytes=cache_lifecycle.final_storage_bytes,
                    peak_active_kv_storage_bytes=cache_lifecycle.peak_storage_bytes,
                    residual_kv_bytes=residual_bytes,
                    peak_gpu_memory=peak_memory,
                    repair_count=sum(event.triggered for event in repair_events),
                    repaired_bytes=sum(event.restored_bytes for event in repair_events),
                    token_agreement=(None if fidelity is None else fidelity.token_agreement),
                    logit_kl=None if fidelity is None else fidelity.logit_kl,
                    attention_output_error=(
                        None if fidelity is None else fidelity.attention_output_error
                    ),
                ),
                effective_method=effective_method,
            )
        except Exception as error:
            if cache_lifecycle is not None:
                trace["partial_cache_lifecycle"] = {
                    **cache_lifecycle.record(),
                    "complete": False,
                    "final_scope": "last_observed_state_before_failure_not_completed_decode",
                }
            structured_failure = getattr(error, "mosaickv_failure_trace_payload", None)
            if isinstance(structured_failure, dict):
                trace.update(cast("JsonObject", structured_failure))
            failure_capacity_records = cast(
                "tuple[HeadLocalCapacityRecord, ...]",
                getattr(error, "mosaickv_external_exact_capacity_records", ()),
            )
            if failure_capacity_records:
                trace["candidate_graph_routing"] = {
                    "status": "partial_failure_before_plan_completion",
                    "external_stable_capacity": _external_stable_capacity_trace(
                        failure_capacity_records
                    ),
                }
            trace.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "end_to_end_time": time.perf_counter() - started,
                    "trace_path": str(trace_path.resolve()),
                }
            )
            if not trace_path.exists():
                _write_trace(trace_path, trace)
            raise


__all__ = [
    "HuggingFaceMosaicKVModel",
    "HuggingFaceRuntimeError",
    "MosaicKVCompressionPlan",
    "PackedRuntimeCache",
    "RuntimePhaseTimings",
    "build_compression_plan",
    "compare_runtime_retention_one",
    "construct_isolated_packed_decode_state",
    "maintain_prefixkv_decode_cache",
    "pack_runtime_cache",
    "pack_runtime_payloads",
    "probe_isolated_packed_candidate",
]
