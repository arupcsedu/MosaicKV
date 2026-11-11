"""Independent cache-ownership intervals; never imports historical KV roles.

Assignments must be constructed by the source-proof validator. This pure CPU
reducer does not turn a classification assertion into a scientific certificate.
It ignores all inherited roles and computes two distinct interval bounds.
"""

from __future__ import annotations

POLICY = {
    "schema_version": 1,
    "persistent_classes": ["cache_payload", "cache_metadata"],
    "workspace_classes": [
        "cache_constructor_workspace",
        "cache_reader_workspace",
        "cache_history_workspace",
        "cache_auxiliary_workspace",
    ],
    "excluded_classes": [
        "normal_query",
        "normal_scores",
        "normal_context",
        "normal_model_activation",
        "model_weights",
    ],
    "auxiliary_validation_and_demand_capture_in_workspace_metric": True,
    "mixed_uses": "charge_whole_backing_for_whole_generation_if_any_included_use",
    "retirement": "actual_free_completed_or_physical_segment_retirement_only",
    "certified_owner_transitions": "not_implemented_never_inferred",
    "whole_device_HBM_or_allocator_reserved_not_certified": True,
}
CATEGORIES = frozenset(
    POLICY["persistent_classes"] + POLICY["workspace_classes"] + POLICY["excluded_classes"]
)
METRICS = {
    "persistent_KV_representation_plus_metadata": frozenset(POLICY["persistent_classes"]),
    "KV_representation_plus_cache_specific_workspace": frozenset(
        POLICY["persistent_classes"] + POLICY["workspace_classes"]
    ),
}


def interval_peak(records, accepted):
    """Aliases are allocator generations, never a sum over owner tensor views."""
    delta = {}
    for key, row in records.items():
        if key not in accepted:
            continue
        delta[row["start"]] = delta.get(row["start"], 0) + row["bytes"]
        if row["end"] is not None:
            delta[row["end"]] = delta.get(row["end"], 0) - row["bytes"]
    live = peak = 0
    for sequence in sorted(delta):
        live += delta[sequence]
        if live < 0:
            raise ValueError("negative allocator interval balance")
        peak = max(peak, live)
    return peak


def reduce_generations(
    records, assignments, *, complete_request, history_complete, structural_errors=()
):
    """Only interval arithmetic, not source review or a completed-GPU assertion."""
    if type(complete_request) is not bool or type(history_complete) is not bool:
        raise ValueError("strict boolean complete-window and allocator-history flags required")
    owners = {}
    for row in records:
        key = row["address"], row["generation"]
        if (
            key in owners
            or any(type(value) is not int or value < 0 for value in key)
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or type(row["start"]) is not int
            or row["start"] < -1
            or (
                row["end"] is not None
                and (type(row["end"]) is not int or row["end"] <= row["start"])
            )
        ):
            raise ValueError("unique valid raw allocator generations required")
        owners[key] = row
    if not set(assignments) <= set(owners):
        raise ValueError("classification references an absent allocation generation")
    uses = {key: set(assignments.get(key, ())) for key in owners}
    if any(not categories <= CATEGORIES for categories in uses.values()):
        raise ValueError("only the preregistered cache-specific categories are accepted")
    unknown = {key for key, categories in uses.items() if not categories}
    valid = complete_request and history_complete and not structural_errors and bool(owners)
    metrics = {}
    for name, included in METRICS.items():
        known = {key for key, categories in uses.items() if categories & included}
        lower = interval_peak(owners, known)
        upper = interval_peak(owners, known | unknown)
        metrics[name] = dict(
            lower_bound_bytes=lower if valid else None,
            upper_bound_bytes=upper if valid else None,
            observed_interval_lower_bytes=lower,
            observed_interval_upper_bytes=upper,
            complete_window_interval_bounds_valid=valid,
            scalar_interval_bounds_equal=valid and lower == upper,
            exact_peak_certified=False,
            source_review_validation_still_required=True,
        )
    return dict(
        policy=POLICY,
        metrics=metrics,
        allocation_generations=len(owners),
        unknown_generations=len(unknown),
        mixed_category_generations=sum(len(categories) > 1 for categories in uses.values()),
        categories=[
            dict(address=key[0], generation=key[1], categories=sorted(uses[key]))
            for key in sorted(owners)
        ],
        category_interval_peaks_not_additive={
            category: interval_peak(
                owners, {key for key, categories in uses.items() if category in categories}
            )
            for category in sorted(CATEGORIES)
        },
        inherited_allocator_roles_used=False,
        whole_device_memory_certified=False,
        historical_KV_role_definitions_modified=False,
        superiority_established=False,
    )
