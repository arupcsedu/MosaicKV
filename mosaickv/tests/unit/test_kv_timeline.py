from concurrent.futures import ThreadPoolExecutor

import pytest

from mosaickv.measurements.kv_timeline import AllocationTimeline


def event(action, addr, size=64):
    return dict(action=action, addr=addr, size=size, stream=0)


def report(ledger, **kwargs):
    return ledger.report(
        complete=True, history_complete=True, opaque_workspace_classified=True, **kwargs
    )


def test_peaks_use_joint_timeline_not_sum_of_phase_maxima():
    ledger = AllocationTimeline()
    ledger.ingest([event("alloc", 1, 100)])
    ledger.classify(1, "kv_persistent", "prefill source")
    ledger.ingest([event("alloc", 2, 70), event("alloc", 3, 20)])
    ledger.classify(2, "kv_workspace", "new source overlaps old source")
    ledger.classify(3, "non_kv", "logit output")
    ledger.ingest(
        [event("free_requested", 1, 100), event("free_completed", 1, 100), event("alloc", 4, 30)]
    )
    ledger.classify(4, "kv_workspace", "decode lease")
    result = report(ledger)
    assert result["exact_KV_peak_bytes"] == 170
    assert result["unknown_allocation_generations"] == 0


def test_address_reuse_is_new_generation_and_cannot_inherit_role():
    ledger = AllocationTimeline()
    ledger.ingest([event("alloc", 1)])
    ledger.classify(1, "model", "resident immutable model")
    ledger.ingest([event("free_requested", 1), event("free_completed", 1), event("alloc", 1)])
    result = report(ledger)
    assert result["unknown_allocation_generations"] == 1
    assert not result["exact_whole_request_KV_peak_certified"]
    assert [r["generation"] for r in result["records"]] == [0, 1]


def test_pending_free_remains_charged_and_aliases_count_once():
    ledger = AllocationTimeline()
    ledger.ingest([event("alloc", 1, 100)])
    ledger.classify(1, "kv_persistent", "source owner")
    ledger.classify(1, "kv_workspace", "aliased reconstruction view")
    ledger.ingest([event("free_requested", 1, 100), event("alloc", 2, 50)])
    ledger.classify(2, "kv_workspace", "other stream still overlaps source")
    assert report(ledger)["exact_KV_peak_bytes"] == 150


def test_cross_thread_registration_and_explicit_workspace_classification():
    ledger = AllocationTimeline()

    def allocate(index):
        with ledger.lock:
            ledger.ingest([event("alloc", index, 100)])
            ledger.classify(index, "kv_workspace", "async native construction workspace")

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(allocate, (1, 2)))
    assert report(ledger)["exact_KV_peak_bytes"] == 200
    incomplete = ledger.report(
        complete=True, history_complete=True, opaque_workspace_classified=False
    )
    assert not incomplete["exact_whole_request_KV_peak_certified"]


def test_incomplete_history_failed_request_and_unknown_fail_closed():
    ledger = AllocationTimeline()
    ledger.ingest([event("alloc", 1, 100)])
    result = ledger.report(
        complete=False, history_complete=False, opaque_workspace_classified=False
    )
    assert result["exact_KV_peak_bytes"] is None
    assert len(result["blocks_exact_certification"]) == 4
    assert result["KV_peak_upper_bound_including_unknown_bytes"] == 100


def test_conflicting_classification_and_lost_allocations_are_not_silenced():
    ledger = AllocationTimeline()
    ledger.ingest([event("alloc", 1)])
    ledger.classify(1, "non_kv", "vocabulary logits")
    with pytest.raises(ValueError, match="conflicting"):
        ledger.classify(1, "kv_persistent", "must not hide conflict")
    ledger.ingest([event("free_completed", 2), event("alloc", 1)])
    assert not report(ledger)["exact_whole_request_KV_peak_certified"]


def test_semantic_region_classifies_dead_workspace_without_reference_retention():
    ledger = AllocationTimeline()
    ledger.ingest([event("alloc", 1), event("free_completed", 1)])
    ledger.classify_range(0, 2, "kv_workspace", "bounded source constructor only")
    assert report(ledger)["exact_KV_peak_bytes"] == 64


def test_operator_scope_never_masquerades_as_whole_request_certificate():
    ledger = AllocationTimeline()
    ledger.ingest([event("alloc", 1)])
    ledger.classify(1, "kv_persistent", "source")
    result = report(ledger, scope="bounded_operator")
    assert result["exact_scoped_KV_peak_certified"]
    assert not result["exact_whole_request_KV_peak_certified"]


def test_stack_attribution_does_not_use_outer_model_frame_to_hide_KV_workspace():
    ledger = AllocationTimeline()
    ledger.ingest(
        [
            dict(
                event("alloc", 1),
                frames=[
                    dict(filename="/a/source.py", name="pack"),
                    dict(filename="/a/model.py", name="forward"),
                ],
            )
        ]
    )
    ledger.classify_stacks(
        [("source.py", "pack", "kv_workspace"), ("model.py", "forward", "non_kv")]
    )
    assert report(ledger)["exact_KV_peak_bytes"] == 64


def test_empty_capture_cannot_certify():
    assert not report(AllocationTimeline())["exact_whole_request_KV_peak_certified"]
