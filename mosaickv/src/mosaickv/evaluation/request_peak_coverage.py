"""Do not silently substitute separate phase maxima for a request KV peak."""


def coverage(detail, mechanisms, *, source_encoding_audit=None, ownership_ledger=None):
    from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

    prefill = detail.get("prefill_storage", {})
    native = (detail.get("overlapping_native_construction") or {}).get("report") or {}
    issues = []
    if prefill.get("tensor_results_observed") is not True:
        issues.append("prefill_encoder_tensor_results_not_observed")
    if detail.get("tensor_result_auditing") is not True:
        issues.append("decode_tensor_results_not_observed")
    if not source_encoding_audit:
        issues.append("source_codec_conversion_overlap_not_observed")
    if native and not native.get("audited_operations", 0):
        issues.append("asynchronous_native_constructor_results_not_observed")
    # Those two gaps remain even if every local phase has its own observer.
    # A sum/max of unsynchronized partial counters is NOT an exact request peak.
    if not ownership_ledger or not ownership_ledger.get("includes_cross_thread_allocator_events"):
        issues.append("no_time_aligned_cross_thread_KV_ownership_ledger")
    if ownership_ledger:
        issues.extend(ownership_ledger.get("blocks_exact_certification", ()))
    else:
        issues.append("opaque_allocator_workspace_not_classified_as_KV_or_non_KV")
    legacy_issues = list(issues)
    certified = valid_whole_request_certificate(ownership_ledger)
    if certified:
        issues = []  # complete allocator proof supersedes separate phase counters
    elif ownership_ledger and ownership_ledger.get("exact_whole_request_KV_peak_certified"):
        issues.append("invalid_serialized_whole_request_certificate")
    return dict(
        observed_prefill_peak_bytes=prefill.get("observed_prefill_cache_peak_bytes"),
        observed_decode_peak_bytes=detail.get("decode_owned_peak_bytes"),
        source_encoding_overlap=source_encoding_audit,
        auxiliary_GPU_tier_bytes=mechanisms.get("extra_gpu_bytes"),
        pinned_CPU_residual_bytes=mechanisms.get("residual_cpu_bytes"),
        time_aligned_allocator_ownership=ownership_ledger,
        joint_whole_request_KV_peak_bytes=ownership_ledger["exact_KV_peak_bytes"]
        if certified
        else None,
        exact_whole_request_KV_peak_certified=certified,
        legacy_phase_audit_diagnostics=legacy_issues,
        blocks_exact_certification=issues,
        phase_peaks_not_added_or_maximized_as_exact_request_peak=True,
    )
