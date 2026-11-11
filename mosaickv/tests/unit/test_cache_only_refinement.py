"""Synthetic interval arithmetic; none of these toy bytes are measured results."""

import copy

import pytest

from mosaickv.measurements.cache_only_refinement import POLICY, reduce_generations


def row(address, size, start, end, *, generation=0):
    return dict(
        address=address,
        generation=generation,
        bytes=size,
        start=start,
        end=end,
        role="model",
        evidence=["inherited role must not be trusted"],
    )


def reduce(records, uses, **kwargs):
    return reduce_generations(records, uses, complete_request=True, history_complete=True, **kwargs)


def test_two_metrics_keep_auxiliary_validation_and_demand_workspace_in_total():
    records = [row(10, 100, 0, 4), row(20, 40, 1, 2), row(30, 12, 1, 3)]
    result = reduce(
        records,
        {
            (10, 0): {"cache_payload"},
            (20, 0): {"cache_reader_workspace"},
            (30, 0): {"cache_auxiliary_workspace"},
        },
    )
    a = result["metrics"]["persistent_KV_representation_plus_metadata"]
    b = result["metrics"]["KV_representation_plus_cache_specific_workspace"]
    assert a["lower_bound_bytes"] == a["upper_bound_bytes"] == 100
    assert b["lower_bound_bytes"] == b["upper_bound_bytes"] == 152
    assert not a["exact_peak_certified"] and a["source_review_validation_still_required"]
    assert result["policy"]["auxiliary_validation_and_demand_capture_in_workspace_metric"]


def test_inherited_model_or_KV_role_never_classifies_unknown_generations():
    record = row(10, 100, 0, 4)
    for inherited in ("model", "non_kv", "kv_persistent"):
        record["role"] = inherited
        result = reduce([record], {})
        assert result["unknown_generations"] == 1
        for metric in result["metrics"].values():
            assert metric["lower_bound_bytes"] == 0 and metric["upper_bound_bytes"] == 100
            assert not metric["scalar_interval_bounds_equal"]


def test_alias_mixed_KV_and_nonKV_use_charges_whole_generation_exactly_once():
    records = [row(10, 100, 0, 5)]
    result = reduce(records, {(10, 0): ["cache_payload", "normal_query", "cache_payload"]})
    assert result["mixed_category_generations"] == 1
    for metric in result["metrics"].values():
        assert metric["lower_bound_bytes"] == 100
        assert metric["upper_bound_bytes"] == 100


def test_pointer_reuse_is_separate_generations_not_per_address_classification():
    records = [row(10, 100, 0, 2), row(10, 200, 3, 4, generation=1)]
    result = reduce(records, {(10, 0): {"cache_payload"}, (10, 1): {"normal_scores"}})
    assert result["allocation_generations"] == 2
    assert (
        result["metrics"]["persistent_KV_representation_plus_metadata"]["lower_bound_bytes"] == 100
    )


def test_free_requested_does_not_retire_until_actual_free_completed_endpoint():
    # Allocator reconstruction supplied end=4, even if a free was requested at1.
    record = row(10, 100, 0, 4)
    record["free_requested_at"] = 1
    result = reduce(
        [record, row(20, 40, 2, 3)], {(10, 0): {"cache_payload"}, (20, 0): {"cache_metadata"}}
    )
    assert (
        result["metrics"]["persistent_KV_representation_plus_metadata"]["lower_bound_bytes"] == 140
    )


def test_equal_scalar_bounds_may_hold_without_waiving_disjoint_unknown():
    result = reduce([row(10, 100, 0, 2), row(20, 20, 3, 4)], {(10, 0): {"cache_payload"}})
    assert result["unknown_generations"] == 1
    assert all(m["scalar_interval_bounds_equal"] for m in result["metrics"].values())
    assert not any(m["exact_peak_certified"] for m in result["metrics"].values())


def test_overlapping_unknown_keeps_bounds_open():
    result = reduce([row(10, 100, 0, 4), row(20, 20, 1, 3)], {(10, 0): {"cache_payload"}})
    assert (
        result["metrics"]["persistent_KV_representation_plus_metadata"]["upper_bound_bytes"] == 120
    )


@pytest.mark.parametrize("failure", ["incomplete", "history", "structural"])
def test_incomplete_or_structurally_invalid_window_never_certifies(failure):
    kwargs = dict(complete_request=True, history_complete=True)
    if failure == "incomplete":
        kwargs["complete_request"] = False
    elif failure == "history":
        kwargs["history_complete"] = False
    else:
        kwargs["structural_errors"] = ["missing_free_event"]
    result = reduce_generations([row(1, 100, 0, 2)], {(1, 0): {"cache_payload"}}, **kwargs)
    assert all(
        m["lower_bound_bytes"] is None and not m["scalar_interval_bounds_equal"]
        for m in result["metrics"].values()
    )


def test_invalid_generation_duplicates_or_unregistered_categories_are_rejected():
    record = row(1, 100, 0, 2)
    with pytest.raises(ValueError, match="unique"):
        reduce([record, copy.deepcopy(record)], {})
    with pytest.raises(ValueError, match="absent"):
        reduce([record], {(2, 0): {"cache_payload"}})
    with pytest.raises(ValueError, match="preregistered"):
        reduce([record], {(1, 0): {"all_attention_is_nonKV"}})
    with pytest.raises(ValueError, match="strict boolean"):
        reduce_generations([record], {}, complete_request="true", history_complete=True)


def test_historical_global_roles_are_not_changed():
    from mosaickv.measurements.kv_timeline import KV_ROLES

    before = KV_ROLES.copy()
    reduce([row(1, 100, 0, 2)], {(1, 0): {"normal_context"}})
    assert before == KV_ROLES and "cache_auxiliary_workspace" not in KV_ROLES
    assert POLICY["certified_owner_transitions"] == "not_implemented_never_inferred"
