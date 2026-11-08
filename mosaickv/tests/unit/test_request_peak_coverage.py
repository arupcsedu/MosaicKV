from mosaickv.evaluation.request_peak_coverage import coverage


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
