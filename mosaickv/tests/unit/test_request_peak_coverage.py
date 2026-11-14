import json
from copy import deepcopy
from types import MappingProxyType

import pytest

from mosaickv.evaluation.request_peak_coverage import coverage
from mosaickv.measurements.kv_timeline import AllocationTimeline, valid_whole_request_certificate


def synthetic_certificate(*, size=64, role="kv_workspace"):
    """CPU accounting fixture only; never a CUDA measurement or saved proof."""
    ledger = AllocationTimeline()
    ledger.ingest([dict(action="alloc", addr=128, size=size)])
    ledger.classify(128, role, "synthetic explicit ownership attribution")
    return ledger.report(complete=True, history_complete=True, opaque_workspace_classified=True)


def assert_blocked(report):
    assert valid_whole_request_certificate(report) is False
    result = coverage({}, {}, ownership_ledger=report)
    assert result["exact_whole_request_KV_peak_certified"] is False
    assert result["joint_whole_request_KV_peak_bytes"] is None
    assert "invalid_serialized_whole_request_certificate" in result["blocks_exact_certification"]


def test_even_complete_phase_counters_cannot_fabricate_time_aligned_request_peak():
    d = dict(
        prefill_storage=dict(tensor_results_observed=True, observed_prefill_cache_peak_bytes=100),
        tensor_result_auditing=True,
        decode_owned_peak_bytes=200,
        overlapping_native_construction=dict(report=dict(audited_operations=30)),
    )
    result = coverage(
        d,
        dict(extra_gpu_bytes=10, residual_cpu_bytes=12),
        source_encoding_audit=dict(peak_visible_bytes=250),
    )
    assert result["joint_whole_request_KV_peak_bytes"] is None
    assert result["exact_whole_request_KV_peak_certified"] is False
    assert len(result["blocks_exact_certification"]) == 2
    assert result["pinned_CPU_residual_bytes"] == 12


def test_missing_producer_audits_remain_separate_blockers():
    r = coverage({}, {})
    assert "prefill_encoder_tensor_results_not_observed" in r["blocks_exact_certification"]
    assert "source_codec_conversion_overlap_not_observed" in r["blocks_exact_certification"]


def test_valid_allocator_interval_proof_can_replace_legacy_phase_peaks():
    from mosaickv.measurements.kv_timeline import AllocationTimeline

    ledger = AllocationTimeline()
    ledger.ingest([dict(action="alloc", addr=123, size=64)])
    ledger.classify(123, "kv_workspace", "native construction workspace")
    report = ledger.report(complete=True, history_complete=True, opaque_workspace_classified=True)
    result = coverage({}, {}, ownership_ledger=report)
    assert result["exact_whole_request_KV_peak_certified"]
    assert result["joint_whole_request_KV_peak_bytes"] == 64
    assert result["legacy_phase_audit_diagnostics"]
    report["exact_KV_peak_bytes"] = 63
    assert not coverage({}, {}, ownership_ledger=report)["exact_whole_request_KV_peak_certified"]


@pytest.mark.parametrize(
    "field",
    [
        "exact_KV_peak_bytes",
        "known_KV_peak_lower_bound_bytes",
        "KV_peak_upper_bound_including_unknown_bytes",
        "allocation_generations",
        "unknown_allocation_generations",
        "events",
    ],
)
@pytest.mark.parametrize("value", [True, False, 1.0, 0.0, "1", None, [], {}, -1])
def test_serialized_peak_bytes_and_counts_require_plain_nonnegative_integers(field, value):
    report = synthetic_certificate(size=1)
    report[field] = value
    assert_blocked(report)


@pytest.mark.parametrize(
    "field",
    [
        "exact_KV_peak_bytes",
        "known_KV_peak_lower_bound_bytes",
        "KV_peak_upper_bound_including_unknown_bytes",
        "allocation_generations",
        "unknown_allocation_generations",
        "events",
        "records",
        "blocks_exact_certification",
    ],
)
def test_missing_serialized_proof_fields_fail_closed(field):
    report = synthetic_certificate()
    del report[field]
    assert_blocked(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("address", None),
        ("address", True),
        ("address", False),
        ("address", 0),
        ("address", -1),
        ("address", 128.0),
        ("address", "128"),
        ("address", []),
        ("address", {}),
        ("generation", None),
        ("generation", True),
        ("generation", False),
        ("generation", -1),
        ("generation", 0.0),
        ("generation", "0"),
        ("generation", []),
        ("generation", {}),
    ],
)
def test_allocation_identity_is_explicit_typed_and_not_boolean(field, value):
    report = synthetic_certificate()
    report["records"][0][field] = value
    assert_blocked(report)


@pytest.mark.parametrize("field", ["address", "generation", "evidence", "role", "bytes", "start"])
def test_missing_allocation_identity_attribution_or_interval_is_not_a_proof(field):
    report = synthetic_certificate()
    del report["records"][0][field]
    assert_blocked(report)


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        [],
        (),
        "synthetic attribution is not a list",
        {},
        True,
        False,
        [""],
        ["  \t\n"],
        [True],
        [0],
        [None],
        [[]],
        [{}],
        ["valid attribution", ""],
        ["valid attribution", None],
    ],
)
def test_every_classified_allocation_requires_nonempty_string_attribution(evidence):
    report = synthetic_certificate()
    report["records"][0]["evidence"] = evidence
    assert_blocked(report)


@pytest.mark.parametrize("row", [None, True, False, 1, "row", [], ()])
def test_nonmapping_allocation_rows_return_false_instead_of_becoming_certificates(row):
    report = synthetic_certificate()
    report["records"] = [row]
    assert_blocked(report)


@pytest.mark.parametrize("rows", [None, True, False, 1, "records", {}, []])
def test_missing_or_nonsequence_allocation_records_fail_closed(rows):
    report = synthetic_certificate()
    report["records"] = rows
    assert_blocked(report)


@pytest.mark.parametrize("report", [None, True, False, 0, 1, "report", [], ()])
def test_nonmapping_report_is_rejected_without_raising(report):
    assert valid_whole_request_certificate(report) is False


def test_boolean_byte_counter_and_missing_owner_reproductions_now_fail_closed():
    report = synthetic_certificate(size=1)
    report.update(
        exact_KV_peak_bytes=True,
        known_KV_peak_lower_bound_bytes=True,
        KV_peak_upper_bound_including_unknown_bytes=True,
        allocation_generations=True,
        unknown_allocation_generations=False,
    )
    assert_blocked(report)
    report = synthetic_certificate(size=1)
    report["records"][0].update(address=None, generation=None, evidence=[])
    assert_blocked(report)


def test_valid_zero_kv_peak_remains_integer_zero_and_certifies():
    report = synthetic_certificate(role="non_kv")
    assert report["exact_KV_peak_bytes"] == 0
    assert valid_whole_request_certificate(report) is True
    result = coverage({}, {}, ownership_ledger=report)
    assert result["exact_whole_request_KV_peak_certified"] is True
    assert type(result["joint_whole_request_KV_peak_bytes"]) is int
    assert result["joint_whole_request_KV_peak_bytes"] == 0
    assert result["blocks_exact_certification"] == []


@pytest.mark.parametrize("size", [1, 64, 2**53 + 1])
def test_valid_integer_peak_and_json_roundtrip_remain_exact_without_input_mutation(size):
    report = synthetic_certificate(size=size)
    serialized = json.loads(json.dumps(report))
    before = deepcopy(serialized)
    assert valid_whole_request_certificate(serialized) is True
    result = coverage({}, {}, ownership_ledger=serialized)
    assert result["joint_whole_request_KV_peak_bytes"] == size
    assert type(result["joint_whole_request_KV_peak_bytes"]) is int
    assert serialized == before
    serialized["exact_KV_peak_bytes"] += 1
    assert_blocked(serialized)


def test_valid_mapping_and_tuple_views_are_not_rejected_as_malformed_proofs():
    report = synthetic_certificate()
    report["records"] = tuple(MappingProxyType(row) for row in report["records"])
    report["records"][0]["evidence"].append("second explicit reviewed attribution")
    report["blocks_exact_certification"] = ()
    assert valid_whole_request_certificate(MappingProxyType(report)) is True


def test_integer_recomputation_keeps_joint_timeline_and_address_generations():
    ledger = AllocationTimeline()
    ledger.ingest([dict(action="alloc", addr=128, size=100)])
    ledger.classify(128, "kv_persistent", "synthetic first source")
    ledger.ingest([dict(action="alloc", addr=256, size=70)])
    ledger.classify(256, "kv_workspace", "synthetic overlapping workspace")
    ledger.ingest(
        [
            dict(action="free_completed", addr=128, size=100),
            dict(action="alloc", addr=128, size=30),
        ]
    )
    ledger.classify(128, "kv_persistent", "synthetic second generation")
    report = ledger.report(complete=True, history_complete=True, opaque_workspace_classified=True)
    assert valid_whole_request_certificate(report) is True
    assert report["exact_KV_peak_bytes"] == 170
    assert [row["generation"] for row in report["records"]] == [0, 0, 1]
    report["exact_KV_peak_bytes"] = 200
    assert_blocked(report)
