from copy import deepcopy

import pytest

from mosaickv.measurements.kv_timeline import AllocationTimeline, valid_whole_request_certificate
from mosaickv.measurements.ownership_review import review


def capture(*, filename="/project/mosaickv/evaluation/eager_core_operator.py", name="__init__"):
    ledger = AllocationTimeline()
    ledger.ingest(
        [
            dict(
                action="alloc",
                addr=128,
                size=64,
                stream=0,
                time_us=100,
                frames=[dict(filename=filename, name=name, line=113)],
            ),
            dict(action="free_requested", addr=128, size=64, stream=0, time_us=150),
            dict(action="free_completed", addr=128, size=64, stream=0, time_us=200),
        ]
    )
    return ledger


def report(ledger):
    return ledger.report(complete=True, history_complete=True, opaque_workspace_classified=False)


def test_cpu_review_classifies_without_changing_any_lifetime_or_raw_capture():
    original = report(capture())
    before = deepcopy(original)
    result = review(original)
    assert original == before
    assert valid_whole_request_certificate(result)
    assert result["exact_KV_peak_bytes"] == 64
    for key in ("address", "generation", "bytes", "start", "end", "time_us", "end_time_us"):
        assert result["records"][0].get(key) == original["records"][0].get(key)
    assert result["original_blocks_exact_certification"] == before["blocks_exact_certification"]


def test_unknown_source_and_unattributed_baseline_stay_blocked():
    result = review(report(capture(filename="/not_reviewed.py", name="forward")))
    assert result["unknown_allocation_generations"] == 1
    assert not valid_whole_request_certificate(result)
    ledger = capture()
    ledger.seed(256, 100)
    result = review(report(ledger))
    assert result["unknown_allocation_generations"] == 1
    assert not result["exact_whole_request_KV_peak_certified"]


@pytest.mark.parametrize(
    "blocker",
    ("allocator_history_incomplete_or_overwritten", "missing_alloc", "request_incomplete"),
)
def test_cpu_review_cannot_erase_lifecycle_or_execution_errors(blocker):
    ledger = capture()
    ledger.errors.append(blocker)
    result = review(report(ledger))
    assert blocker in result["blocks_exact_certification"]
    assert not valid_whole_request_certificate(result)


def test_ambiguous_rejected_marks_require_independent_complete_source_classification():
    ledger = capture()
    ledger.errors.append("ambiguous_deferred_owner_timestamp")
    result = review(report(ledger))
    assert valid_whole_request_certificate(result)
    assert result["rejected_timestamp_marks_not_used_for_final_attribution"]
    assert result["rejected_timestamp_mark_count"] is None
    assert "ambiguous_deferred_owner_timestamp" in result["original_blocks_exact_certification"]
    ledger = capture(filename="/unknown.py")
    ledger.errors.append("ambiguous_deferred_owner_timestamp")
    result = review(report(ledger))
    assert "ambiguous_deferred_owner_timestamp" in result["blocks_exact_certification"]
    assert not result["exact_whole_request_KV_peak_certified"]


def test_standard_language_attention_workspace_is_charged_conservatively():
    ledger = capture(
        filename="/env/transformers/models/llama/modeling_llama.py", name="eager_attention_forward"
    )
    ledger.classify_stacks(
        [("/transformers/models/llama/modeling_llama.py", "eager_attention_forward", "non_kv")]
    )
    result = review(report(ledger))
    assert result["conservative_attention_reclassifications"] == 1
    assert result["exact_KV_peak_bytes"] == 64


def test_conflicting_review_rules_fail_and_operator_scope_is_not_whole_request():
    raw = report(capture())
    with pytest.raises(ValueError, match="conflicting"):
        review(
            raw,
            rules=(
                ("eager_core_operator.py", "__init__", "non_kv"),
                ("eager_core_operator.py", "__init__", "kv_workspace"),
            ),
        )
    raw["scope"] = "bounded_operator"
    assert not review(raw)["exact_whole_request_KV_peak_certified"]


def test_independent_validator_rejects_tampered_peak_after_review():
    result = review(report(capture()))
    result["exact_KV_peak_bytes"] += 1
    assert not valid_whole_request_certificate(result)
