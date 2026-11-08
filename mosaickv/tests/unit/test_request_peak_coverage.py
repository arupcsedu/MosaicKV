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
