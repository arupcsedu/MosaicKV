"""Audited bridge from common HF block selections to serving-engine blocks.

The common MosaicKV selector operates per layer and KV head.  vLLM 0.10.2
owns one request-wide physical token-block table.  This module performs the
only defensible projection between those boundaries: request blocks are ranked
by selected source-position votes, then by the source selector's score, with
deterministic block-index tie breaking and mandatory boundary preservation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from mosaickv.backends.native_blocks import NativeBlockSelection, require_scratch_path
from mosaickv.cache_state import FullKVState, KVBlockDescriptor, Modality
from mosaickv.types import JsonObject, JsonValue


class NativeSelectorPlanError(ValueError):
    """Raised when a selector projection cannot be consumed without guessing."""


@dataclass(frozen=True, slots=True)
class RequestWideSelectorProjection:
    """One common-selector decision projected onto request-wide token blocks."""

    method: str
    logical_positions: int
    block_size: int
    retention_ratio: float
    selected_blocks: tuple[int, ...]
    mandatory_blocks: tuple[int, ...]
    selected_position_votes: tuple[int, ...]
    selected_score_sums: tuple[float, ...]
    source_selected_positions: int
    source_total_positions: int
    selected_route_votes: tuple[int, ...] = ()
    normalized_utility_scores: tuple[float, ...] = ()
    evidence_route_votes: tuple[int, ...] = ()
    evidence_utility_scores: tuple[float, ...] = ()
    evidence_blocks: tuple[int, ...] = ()
    policy: str = "common_selector_request_wide_vote_projection_v2"

    def __post_init__(self) -> None:
        selection = self.selection
        if not self.method.strip():
            raise NativeSelectorPlanError("projection method must be non-empty")
        if len(self.selected_position_votes) != selection.source_blocks:
            raise NativeSelectorPlanError("projection vote vector has the wrong length")
        if len(self.selected_score_sums) != selection.source_blocks:
            raise NativeSelectorPlanError("projection score vector has the wrong length")
        if self.source_total_positions < 1 or not 0 <= self.source_selected_positions <= (
            self.source_total_positions
        ):
            raise NativeSelectorPlanError("projection source-position accounting is invalid")
        if any(value < 0 for value in self.selected_position_votes):
            raise NativeSelectorPlanError("projection votes must be nonnegative")
        if any(not math.isfinite(value) for value in self.selected_score_sums):
            raise NativeSelectorPlanError("projection scores must be finite")
        source_blocks = selection.source_blocks
        for name in (
            "selected_route_votes",
            "normalized_utility_scores",
            "evidence_route_votes",
            "evidence_utility_scores",
        ):
            values = getattr(self, name)
            if not values:
                values = tuple(0 for _ in range(source_blocks))
                object.__setattr__(self, name, values)
            if len(values) != source_blocks:
                raise NativeSelectorPlanError(f"projection {name} vector has the wrong length")
        if any(value < 0 for value in self.selected_route_votes):
            raise NativeSelectorPlanError("selected route votes must be nonnegative")
        if any(value < 0 for value in self.evidence_route_votes):
            raise NativeSelectorPlanError("evidence route votes must be nonnegative")
        if any(
            not math.isfinite(value) or value < 0
            for value in (*self.normalized_utility_scores, *self.evidence_utility_scores)
        ):
            raise NativeSelectorPlanError(
                "normalized projection utility scores must be finite and nonnegative"
            )
        if tuple(sorted(set(self.evidence_blocks))) != self.evidence_blocks:
            raise NativeSelectorPlanError(
                "projection evidence blocks must be strictly increasing and unique"
            )
        if self.evidence_blocks and (
            self.evidence_blocks[0] < 0 or self.evidence_blocks[-1] >= source_blocks
        ):
            raise NativeSelectorPlanError("projection evidence block is outside the prompt")
        if tuple(sorted(set(self.selected_blocks))) != self.selected_blocks:
            raise NativeSelectorPlanError(
                "projection selected blocks must be strictly increasing and unique"
            )
        if tuple(sorted(set(self.mandatory_blocks))) != self.mandatory_blocks:
            raise NativeSelectorPlanError(
                "projection mandatory blocks must be strictly increasing and unique"
            )
        if (
            not self.selected_blocks
            or self.selected_blocks[0] < 0
            or self.selected_blocks[-1] >= source_blocks
        ):
            raise NativeSelectorPlanError("projection selected block is outside the prompt")
        if (
            not self.mandatory_blocks
            or self.mandatory_blocks[0] < 0
            or self.mandatory_blocks[-1] >= source_blocks
        ):
            raise NativeSelectorPlanError("projection mandatory block is outside the prompt")
        if not set(self.mandatory_blocks).issubset(self.selected_blocks):
            raise NativeSelectorPlanError("projection omits a mandatory request block")

    @property
    def selection(self) -> NativeBlockSelection:
        return NativeBlockSelection(
            logical_positions=self.logical_positions,
            block_size=self.block_size,
            retention_ratio=self.retention_ratio,
            selected_blocks=self.selected_blocks,
            mandatory_blocks=self.mandatory_blocks,
            policy=self.policy,
        )

    def to_json_object(self) -> JsonObject:
        payload = cast("JsonObject", asdict(self))
        payload["schema_version"] = 2
        payload["source_blocks"] = self.selection.source_blocks
        payload["selected_positions"] = self.selection.selected_positions
        payload["selected_vote_fraction"] = (
            sum(self.selected_position_votes[index] for index in self.selected_blocks)
            / max(1, sum(self.selected_position_votes))
        )
        return payload


def _common_block_size(full_state: FullKVState) -> int:
    sizes = tuple(block.position_count for block in full_state.blocks)
    if not sizes:
        raise NativeSelectorPlanError("cannot project an empty FullKV state")
    return max(sizes)


def project_common_selector_to_request_blocks(
    full_state: FullKVState,
    selected: Sequence[tuple[KVBlockDescriptor, float]],
    *,
    method: str,
    retention_ratio: float,
    fallback_candidates: Sequence[tuple[KVBlockDescriptor, float]] = (),
) -> RequestWideSelectorProjection:
    """Project per-layer/head selections to vLLM's shared token-block table."""

    logical = full_state.original_logical_sequence_length
    block_size = _common_block_size(full_state)
    if not math.isfinite(retention_ratio) or not 0 < retention_ratio <= 1:
        raise NativeSelectorPlanError("retention_ratio must be finite and in (0, 1]")
    source_blocks = math.ceil(logical / block_size)
    votes = [0] * source_blocks
    scores = [0.0] * source_blocks
    selected_routes_by_block: dict[int, set[tuple[int, int]]] = defaultdict(set)
    evidence_routes_by_block: dict[int, set[tuple[int, int]]] = defaultdict(set)
    route_scores: dict[tuple[int, int], dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    evidence_blocks = {
        position // block_size
        for block in full_state.blocks
        if block.modality is not Modality.TEXT
        for position in block.original_logical_positions
    }
    source_selected_positions = 0
    for block, raw_score in selected:
        score = float(raw_score)
        if not math.isfinite(score):
            raise NativeSelectorPlanError("source selector score must be finite")
        source_selected_positions += block.position_count
        selected_positions_by_block: dict[int, int] = {}
        for position in block.original_logical_positions:
            if position < 0 or position >= logical:
                raise NativeSelectorPlanError("selected source position is outside the prompt")
            request_block = position // block_size
            selected_positions_by_block[request_block] = (
                selected_positions_by_block.get(request_block, 0) + 1
            )
        for request_block, position_count in selected_positions_by_block.items():
            votes[request_block] += position_count
            scores[request_block] += score * position_count
            route = (block.layer, block.kv_head)
            selected_routes_by_block[request_block].add(route)
            route_scores[route][request_block] += score * position_count
            if block.modality is not Modality.TEXT:
                evidence_routes_by_block[request_block].add(route)
    # Retain a complete, label-blind ranking below the original operating
    # point. This lets a serving-byte ceiling use otherwise idle capacity
    # without falling back to token index order.
    for block, raw_score in fallback_candidates:
        score = float(raw_score)
        if not math.isfinite(score):
            raise NativeSelectorPlanError("fallback selector score must be finite")
        fallback_positions_by_block: dict[int, int] = {}
        for position in block.original_logical_positions:
            if position < 0 or position >= logical:
                raise NativeSelectorPlanError("fallback source position is outside the prompt")
            request_block = position // block_size
            fallback_positions_by_block[request_block] = (
                fallback_positions_by_block.get(request_block, 0) + 1
            )
        for request_block, position_count in fallback_positions_by_block.items():
            scores[request_block] += score * position_count
            route_scores[(block.layer, block.kv_head)][request_block] += (
                score * position_count
            )

    routes = tuple(sorted({(block.layer, block.kv_head) for block in full_state.blocks}))
    normalized_utility = [0.0] * source_blocks
    for route in routes:
        values = [route_scores[route].get(index, 0.0) for index in range(source_blocks)]
        minimum = min(values)
        maximum = max(values)
        if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
            continue
        scale = maximum - minimum
        for index, value in enumerate(values):
            normalized_utility[index] += (value - minimum) / scale
    route_denominator = max(1, len(routes))
    normalized_utility = [value / route_denominator for value in normalized_utility]
    selected_route_votes = [
        len(selected_routes_by_block.get(index, set())) for index in range(source_blocks)
    ]
    evidence_route_votes = [
        len(evidence_routes_by_block.get(index, set())) for index in range(source_blocks)
    ]
    evidence_utility = [
        normalized_utility[index] if index in evidence_blocks else 0.0
        for index in range(source_blocks)
    ]

    mandatory = tuple(
        sorted(
            {0, source_blocks - 1}
            | {
                position // block_size
                for position in full_state.mandatory_logical_positions
            }
        )
    )
    target = source_blocks if retention_ratio == 1 else math.ceil(source_blocks * retention_ratio)
    target = min(source_blocks, max(target, len(mandatory)))
    chosen = set(mandatory)
    ranked = sorted(
        (index for index in range(source_blocks) if index not in chosen),
        key=lambda index: (-votes[index], -scores[index], index),
    )
    chosen.update(ranked[: target - len(chosen)])
    return RequestWideSelectorProjection(
        method=method,
        logical_positions=logical,
        block_size=block_size,
        retention_ratio=retention_ratio,
        selected_blocks=tuple(sorted(chosen)),
        mandatory_blocks=mandatory,
        selected_position_votes=tuple(votes),
        selected_score_sums=tuple(scores),
        source_selected_positions=source_selected_positions,
        source_total_positions=sum(block.position_count for block in full_state.blocks),
        selected_route_votes=tuple(selected_route_votes),
        normalized_utility_scores=tuple(normalized_utility),
        evidence_route_votes=tuple(evidence_route_votes),
        evidence_utility_scores=tuple(evidence_utility),
        evidence_blocks=tuple(sorted(evidence_blocks)),
    )


def retarget_request_wide_projection(
    projection: RequestWideSelectorProjection,
    *,
    target_block_size: int,
    target_retention_ratio: float,
) -> RequestWideSelectorProjection:
    """Coarsen a fine-grained selector projection for a serving cache.

    The HF selector may use smaller blocks than the serving engine.  Summing
    votes and scores over aligned source blocks preserves all selection
    evidence that can be represented by the coarser request-wide vLLM block
    table.  The serving budget is then applied *after* that lossless aligned
    aggregation, which avoids silently treating a 4-token HF block as a
    16-token vLLM block.
    """

    source_block_size = projection.block_size
    if target_block_size < source_block_size or target_block_size % source_block_size:
        raise NativeSelectorPlanError(
            "target block size must be an integer multiple of the source block size"
        )
    if not math.isfinite(target_retention_ratio) or not 0 < target_retention_ratio <= 1:
        raise NativeSelectorPlanError(
            "target retention ratio must be finite and in (0, 1]"
        )

    logical = projection.logical_positions
    target_source_blocks = math.ceil(logical / target_block_size)
    votes = [0] * target_source_blocks
    scores = [0.0] * target_source_blocks
    selected_route_votes = [0] * target_source_blocks
    normalized_utility = [0.0] * target_source_blocks
    evidence_route_votes = [0] * target_source_blocks
    evidence_utility = [0.0] * target_source_blocks
    evidence_blocks: set[int] = set()
    projected_rows = zip(
        projection.selected_position_votes,
        projection.selected_score_sums,
        projection.selected_route_votes,
        projection.normalized_utility_scores,
        projection.evidence_route_votes,
        projection.evidence_utility_scores,
        strict=True,
    )
    for source_index, (
        vote,
        score,
        route_vote,
        utility,
        evidence_vote,
        evidence_score,
    ) in enumerate(projected_rows):
        logical_start = source_index * source_block_size
        target_index = logical_start // target_block_size
        if target_index >= target_source_blocks:
            raise NativeSelectorPlanError("source projection exceeds its logical length")
        votes[target_index] += vote
        scores[target_index] += score
        # A route may contribute to several fine blocks inside one serving
        # block. Max aggregation caps repeated within-route support instead of
        # rewarding a serving block merely for spanning more fine blocks.
        selected_route_votes[target_index] = max(
            selected_route_votes[target_index], route_vote
        )
        normalized_utility[target_index] = max(
            normalized_utility[target_index], utility
        )
        evidence_route_votes[target_index] = max(
            evidence_route_votes[target_index], evidence_vote
        )
        evidence_utility[target_index] = max(
            evidence_utility[target_index], evidence_score
        )
        if source_index in projection.evidence_blocks:
            evidence_blocks.add(target_index)

    mandatory = tuple(
        sorted(
            {0, target_source_blocks - 1}
            | {
                (source_index * source_block_size) // target_block_size
                for source_index in projection.mandatory_blocks
            }
        )
    )
    target = (
        target_source_blocks
        if target_retention_ratio == 1
        else math.ceil(target_source_blocks * target_retention_ratio)
    )
    target = min(target_source_blocks, max(target, len(mandatory)))
    chosen = set(mandatory)
    ranked = sorted(
        (index for index in range(target_source_blocks) if index not in chosen),
        key=lambda index: (-votes[index], -scores[index], index),
    )
    chosen.update(ranked[: target - len(chosen)])
    return RequestWideSelectorProjection(
        method=projection.method,
        logical_positions=logical,
        block_size=target_block_size,
        retention_ratio=target_retention_ratio,
        selected_blocks=tuple(sorted(chosen)),
        mandatory_blocks=mandatory,
        selected_position_votes=tuple(votes),
        selected_score_sums=tuple(scores),
        source_selected_positions=projection.source_selected_positions,
        source_total_positions=projection.source_total_positions,
        selected_route_votes=tuple(selected_route_votes),
        normalized_utility_scores=tuple(normalized_utility),
        evidence_route_votes=tuple(evidence_route_votes),
        evidence_utility_scores=tuple(evidence_utility),
        evidence_blocks=tuple(sorted(evidence_blocks)),
        policy="common_selector_request_wide_vote_projection_v2",
    )


def fit_projection_to_competitor_budget(
    projection: RequestWideSelectorProjection,
    competitor: RequestWideSelectorProjection,
    *,
    minimum_reduction_fraction: float,
    minimum_evidence_blocks: int = 1,
    maximum_evidence_blocks: int = 2,
) -> RequestWideSelectorProjection:
    """Fill the largest whole-block budget that remains below a competitor.

    The rule is sample-label blind: the competitor defines only the physical
    byte ceiling. Selection within that ceiling continues to use MosaicKV's
    votes and utility scores. At this request-wide boundary every block has
    identical KV geometry, so block cardinality is exactly the active-byte
    budget.
    """

    if projection.logical_positions != competitor.logical_positions:
        raise NativeSelectorPlanError("candidate and competitor logical lengths differ")
    if projection.block_size != competitor.block_size:
        raise NativeSelectorPlanError("candidate and competitor block sizes differ")
    if (
        not math.isfinite(minimum_reduction_fraction)
        or not 0 < minimum_reduction_fraction < 1
    ):
        raise NativeSelectorPlanError("minimum reduction must be finite and in (0, 1)")
    if minimum_evidence_blocks < 0 or maximum_evidence_blocks < minimum_evidence_blocks:
        raise NativeSelectorPlanError("evidence reservation bounds are invalid")
    competitor_blocks = len(competitor.selected_blocks)
    maximum_blocks = math.floor(
        math.nextafter(
            (1.0 - minimum_reduction_fraction) * competitor_blocks,
            math.inf,
        )
    )
    if maximum_blocks < len(projection.mandatory_blocks):
        raise NativeSelectorPlanError(
            "competitor-relative byte ceiling cannot preserve mandatory request blocks"
        )
    source_blocks = projection.selection.source_blocks
    target = min(source_blocks, maximum_blocks)
    chosen = set(projection.mandatory_blocks)
    evidence_candidates = tuple(
        index for index in projection.evidence_blocks if index not in chosen
    )
    available_optional_slots = target - len(chosen)
    if evidence_candidates and available_optional_slots < minimum_evidence_blocks:
        raise NativeSelectorPlanError(
            "competitor-relative byte ceiling cannot satisfy the evidence reservation"
        )
    evidence_target = min(
        maximum_evidence_blocks,
        available_optional_slots,
        len(evidence_candidates),
    )
    if evidence_candidates:
        evidence_target = max(minimum_evidence_blocks, evidence_target)
    ranked_evidence = sorted(
        evidence_candidates,
        key=lambda index: (
            -projection.evidence_route_votes[index],
            -projection.evidence_utility_scores[index],
            -projection.selected_route_votes[index],
            -projection.normalized_utility_scores[index],
            index,
        ),
    )
    chosen.update(ranked_evidence[:evidence_target])
    ranked = sorted(
        (index for index in range(source_blocks) if index not in chosen),
        key=lambda index: (
            -projection.selected_route_votes[index],
            -projection.normalized_utility_scores[index],
            -projection.evidence_route_votes[index],
            -projection.evidence_utility_scores[index],
            -projection.selected_position_votes[index],
            index,
        ),
    )
    chosen.update(ranked[: target - len(chosen)])
    return RequestWideSelectorProjection(
        method=projection.method,
        logical_positions=projection.logical_positions,
        block_size=projection.block_size,
        retention_ratio=target / source_blocks,
        selected_blocks=tuple(sorted(chosen)),
        mandatory_blocks=projection.mandatory_blocks,
        selected_position_votes=projection.selected_position_votes,
        selected_score_sums=projection.selected_score_sums,
        source_selected_positions=projection.source_selected_positions,
        source_total_positions=projection.source_total_positions,
        selected_route_votes=projection.selected_route_votes,
        normalized_utility_scores=projection.normalized_utility_scores,
        evidence_route_votes=projection.evidence_route_votes,
        evidence_utility_scores=projection.evidence_utility_scores,
        evidence_blocks=projection.evidence_blocks,
        policy="common_selector_competitor_relative_evidence_budget_v2",
    )


@dataclass(frozen=True, slots=True)
class NativeSelectorPlanEntry:
    sample_id: str
    projection: RequestWideSelectorProjection
    selector_planning_seconds: float
    source_trace: str
    source_trace_sha256: str

    def __post_init__(self) -> None:
        if not self.sample_id or not math.isfinite(self.selector_planning_seconds):
            raise NativeSelectorPlanError("plan entry identity/timing is invalid")
        if self.selector_planning_seconds < 0:
            raise NativeSelectorPlanError("selector planning time must be nonnegative")
        if len(self.source_trace_sha256) != 64:
            raise NativeSelectorPlanError("source trace SHA-256 is invalid")

    def to_json_object(self) -> JsonObject:
        return {
            "sample_id": self.sample_id,
            "projection": self.projection.to_json_object(),
            "selector_planning_seconds": self.selector_planning_seconds,
            "source_trace": self.source_trace,
            "source_trace_sha256": self.source_trace_sha256,
        }


@dataclass(frozen=True, slots=True)
class NativeSelectorPlanCatalog:
    model_id: str
    model_revision: str
    task: str
    dataset_revision: str
    method: str
    entries: tuple[NativeSelectorPlanEntry, ...]
    catalog_sha256: str
    path: Path

    def entry(self, sample_id: str) -> NativeSelectorPlanEntry:
        matches = tuple(entry for entry in self.entries if entry.sample_id == sample_id)
        if len(matches) != 1:
            raise NativeSelectorPlanError(
                f"selector plan has {len(matches)} entries for sample {sample_id!r}"
            )
        return matches[0]


def _projection_from_json(value: object) -> RequestWideSelectorProjection:
    if not isinstance(value, Mapping):
        raise NativeSelectorPlanError("projection must be a JSON object")
    if value.get("schema_version") not in {1, 2}:
        raise NativeSelectorPlanError("unsupported projection schema")
    def integer_sequence(raw: object, label: str) -> tuple[int, ...]:
        if not isinstance(raw, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in raw
        ):
            raise NativeSelectorPlanError(f"{label} must be an integer list")
        return tuple(raw)

    def float_sequence(raw: object, label: str) -> tuple[float, ...]:
        if not isinstance(raw, list) or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in raw
        ):
            raise NativeSelectorPlanError(f"{label} must be a numeric list")
        return tuple(float(item) for item in raw)

    try:
        return RequestWideSelectorProjection(
            method=str(value["method"]),
            logical_positions=int(value["logical_positions"]),
            block_size=int(value["block_size"]),
            retention_ratio=float(value["retention_ratio"]),
            selected_blocks=integer_sequence(value["selected_blocks"], "selected_blocks"),
            mandatory_blocks=integer_sequence(value["mandatory_blocks"], "mandatory_blocks"),
            selected_position_votes=integer_sequence(
                value["selected_position_votes"], "selected_position_votes"
            ),
            selected_score_sums=float_sequence(
                value["selected_score_sums"], "selected_score_sums"
            ),
            source_selected_positions=int(value["source_selected_positions"]),
            source_total_positions=int(value["source_total_positions"]),
            selected_route_votes=integer_sequence(
                value.get("selected_route_votes", []), "selected_route_votes"
            ),
            normalized_utility_scores=float_sequence(
                value.get("normalized_utility_scores", []),
                "normalized_utility_scores",
            ),
            evidence_route_votes=integer_sequence(
                value.get("evidence_route_votes", []), "evidence_route_votes"
            ),
            evidence_utility_scores=float_sequence(
                value.get("evidence_utility_scores", []), "evidence_utility_scores"
            ),
            evidence_blocks=integer_sequence(
                value.get("evidence_blocks", []), "evidence_blocks"
            ),
            policy=str(value["policy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise NativeSelectorPlanError(f"invalid projection payload: {error}") from error


def load_native_selector_plan_catalog(
    path: str | Path,
    expected_sha256: str,
    *,
    model_id: str,
    model_revision: str,
    task: str,
    dataset_revision: str,
    method: str,
) -> NativeSelectorPlanCatalog:
    """Load a hash-bound selector catalog and reject every provenance mismatch."""

    source = require_scratch_path(path)
    payload_bytes = source.read_bytes()
    observed_sha = hashlib.sha256(payload_bytes).hexdigest()
    if observed_sha != expected_sha256:
        raise NativeSelectorPlanError(
            f"selector-plan SHA mismatch: observed={observed_sha}, expected={expected_sha256}"
        )
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise NativeSelectorPlanError(f"selector plan is not valid JSON: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise NativeSelectorPlanError("unsupported selector-plan catalog schema")
    expected = {
        "model_id": model_id,
        "model_revision": model_revision,
        "task": task,
        "dataset_revision": dataset_revision,
        "method": method,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise NativeSelectorPlanError(
                f"selector-plan {key}={payload.get(key)!r}, expected {expected_value!r}"
            )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise NativeSelectorPlanError("selector-plan entries must be a non-empty list")
    entries: list[NativeSelectorPlanEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise NativeSelectorPlanError("selector-plan entry must be an object")
        entries.append(
            NativeSelectorPlanEntry(
                sample_id=str(raw_entry.get("sample_id", "")),
                projection=_projection_from_json(raw_entry.get("projection")),
                selector_planning_seconds=float(raw_entry.get("selector_planning_seconds", -1)),
                source_trace=str(raw_entry.get("source_trace", "")),
                source_trace_sha256=str(raw_entry.get("source_trace_sha256", "")),
            )
        )
    identities = tuple(entry.sample_id for entry in entries)
    if len(set(identities)) != len(identities):
        raise NativeSelectorPlanError("selector-plan sample IDs must be unique")
    return NativeSelectorPlanCatalog(
        model_id=model_id,
        model_revision=model_revision,
        task=task,
        dataset_revision=dataset_revision,
        method=method,
        entries=tuple(entries),
        catalog_sha256=observed_sha,
        path=source,
    )


def catalog_payload(
    *,
    model_id: str,
    model_revision: str,
    task: str,
    dataset_revision: str,
    method: str,
    entries: Sequence[NativeSelectorPlanEntry],
) -> JsonObject:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "model_revision": model_revision,
        "task": task,
        "dataset_revision": dataset_revision,
        "method": method,
        "entries": cast(
            "list[JsonValue]", [entry.to_json_object() for entry in entries]
        ),
    }


def merge_native_selector_plan_catalogs(
    catalogs: Sequence[NativeSelectorPlanCatalog],
) -> JsonObject:
    """Merge disjoint one-sample catalogs for a persistent-engine isolation run."""

    if not catalogs:
        raise NativeSelectorPlanError("at least one selector catalog is required")
    first = catalogs[0]
    expected = (
        first.model_id,
        first.model_revision,
        first.task,
        first.dataset_revision,
        first.method,
    )
    entries: list[NativeSelectorPlanEntry] = []
    block_boundary: tuple[int, float] | None = None
    for catalog in catalogs:
        observed = (
            catalog.model_id,
            catalog.model_revision,
            catalog.task,
            catalog.dataset_revision,
            catalog.method,
        )
        if observed != expected:
            raise NativeSelectorPlanError("selector catalogs have incompatible provenance")
        for entry in catalog.entries:
            projection_boundary = (
                entry.projection.block_size,
                entry.projection.retention_ratio,
            )
            if block_boundary is None:
                block_boundary = projection_boundary
            elif (
                projection_boundary[0] != block_boundary[0]
                or not math.isclose(
                    projection_boundary[1],
                    block_boundary[1],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise NativeSelectorPlanError(
                    "persistent-engine selector entries require one block size and ratio"
                )
            entries.append(entry)
    identities = tuple(entry.sample_id for entry in entries)
    if len(set(identities)) != len(identities):
        raise NativeSelectorPlanError("merged selector catalogs contain duplicate sample IDs")
    entries.sort(key=lambda entry: entry.sample_id)
    return catalog_payload(
        model_id=first.model_id,
        model_revision=first.model_revision,
        task=first.task,
        dataset_revision=first.dataset_revision,
        method=first.method,
        entries=entries,
    )


__all__ = [
    "NativeSelectorPlanCatalog",
    "NativeSelectorPlanEntry",
    "NativeSelectorPlanError",
    "RequestWideSelectorProjection",
    "catalog_payload",
    "fit_projection_to_competitor_budget",
    "load_native_selector_plan_catalog",
    "merge_native_selector_plan_catalogs",
    "project_common_selector_to_request_blocks",
    "retarget_request_wide_projection",
]
