"""Accounting/control fixtures only; none is a measured experiment result."""

from types import SimpleNamespace

import pytest

from mosaickv.evaluation.paper_request_certification import (
    WholeRequestEnvelope,
    allocator_envelope,
    comparison_admission,
)


def envelope(**changes):
    return allocator_envelope(
        **dict(
            immutable_model_bytes=100,
            baseline_allocated_bytes=150,
            peak_allocated_bytes=250,
            peak_reserved_bytes=400,
            resident_model_unchanged=True,
            complete_request=True,
        )
        | changes
    )


def test_unknown_baseline_is_not_subtracted_or_misnamed_exact_peak():
    result = envelope()
    assert result["kv_peak_upper_bound_bytes"] == 150  # NOT 250 - 150
    assert result["kv_peak_upper_bound_certified"]
    assert result["exact_kv_peak_bytes"] is None
    assert not result["exact_kv_peak_certified"]
    assert envelope(complete_request=False)["kv_peak_upper_bound_bytes"] is None
    assert envelope(resident_model_unchanged=False)["kv_peak_upper_bound_bytes"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"immutable_model_bytes": -1},
        {"baseline_allocated_bytes": 99},
        {"peak_allocated_bytes": 149},
        {"peak_reserved_bytes": 249},
        {"peak_allocated_bytes": float("nan")},
        {"complete_request": 1},
    ],
)
def test_invalid_or_nan_measurements_fail(change):
    with pytest.raises(ValueError):
        envelope(**change)


def test_inner_reset_cannot_erase_construction_peak_and_exception_restores_api():
    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda: calls.append("sync"),
        memory_allocated=lambda: 150,
        reset_peak_memory_stats=lambda: calls.append("reset"),
        max_memory_allocated=lambda: 250,
        max_memory_reserved=lambda: 400,
    )
    audit = WholeRequestEnvelope(None, cuda=cuda)
    audit.model_backing = lambda: {("cuda:0", 7, 100): 100}
    with pytest.raises(RuntimeError, match="sample failure"), audit:
        cuda.reset_peak_memory_stats()
        raise RuntimeError("sample failure")
    assert calls.count("reset") == 1
    assert audit.certificate["interior_reset_attempts_suppressed"] == 1
    assert not audit.certificate["kv_peak_upper_bound_certified"]
    cuda.reset_peak_memory_stats()
    assert calls.count("reset") == 2
    with pytest.raises(RuntimeError, match="nonreusable"), audit:
        pass


def rows():
    return [
        dict(
            sample_id="a",
            method=m,
            status="completed",
            comparison_controls_sha256="a" * 64,
            initial_encoded_bytes=100,
            final_encoded_bytes=120,
            decode_owned_peak_bytes=130,
            used_fullkv_fallback=False,
            whole_request_envelope=envelope(),
        )
        for m in ("mosaickv_exact", "prompt_attention_topk")
    ]


def admit(data):
    return comparison_admission(
        data,
        sample_ids=["a"],
        methods=["mosaickv_exact", "prompt_attention_topk"],
        cap_bytes_by_sample={"a": 100},
    )


def test_allocator_certificate_alone_cannot_admit_paper_comparison():
    result = admit(rows())
    assert result["complete"] and not result["paper_ready"]
    assert not result["byte_matched_feasibility_pass"]
    assert {r["reason"] for r in result["blockers"]} == {"whole_request_KV_peak_not_exact"}
    assert not any(result["mechanism_activity"].values())


def test_unequal_caps_missing_peaks_failed_and_fallback_rows_stay_blocked():
    data = rows()
    data[1].update(initial_encoded_bytes=101, decode_owned_peak_bytes=float("nan"))
    result = admit(data)
    reasons = {r["reason"] for r in result["blockers"]}
    assert {
        "initial_cap_exceeded",
        "unmatched_initial_encoded_bytes",
        "missing_decode_owned_peak_bytes",
    } <= reasons
    data[1]["status"] = "failed"
    assert not admit(data)["complete"]
    assert "failed_prediction" in {r["reason"] for r in admit(data)["blockers"]}
    assert "missing_prediction" in {r["reason"] for r in admit(data[:1])["blockers"]}
    with pytest.raises(ValueError, match="duplicate"):
        admit([*data, data[0]])


def test_control_mismatch_or_fullkv_latency_comparison_rejected():
    data = rows()
    data[1]["comparison_controls_sha256"] = "b" * 64
    assert "controls_differ" in {r["reason"] for r in admit(data)["blockers"]}
    with pytest.raises(ValueError):
        comparison_admission(
            [],
            sample_ids=["a"],
            methods=["full_kv", "mosaickv_exact"],
            cap_bytes_by_sample={"a": 100},
        )


def test_missing_envelope_and_exact_flag_without_bytes_are_not_certificates():
    data = rows()
    data[0]["whole_request_envelope"] = None
    data[1]["whole_request_envelope"]["exact_kv_peak_certified"] = True
    reasons = {r["reason"] for r in admit(data)["blockers"]}
    assert "missing_request_envelope" in reasons
    assert "missing_exact_KV_peak_bytes" in reasons
