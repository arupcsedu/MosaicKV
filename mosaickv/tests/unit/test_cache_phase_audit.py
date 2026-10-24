import pytest

from mosaickv.evaluation.cache_phase_audit import audit_phase_trace


@pytest.mark.parametrize("teacher", [False, True])
def test_both_phases_require_initial_final_peak_physical_cap(teacher):
    trace = dict(
        status="completed",
        sample_id="synthetic",
        effective_method="fixture",
        teacher_forced_evaluation=dict(
            enabled=teacher,
            negative_log_likelihood_sum=2.0,
            evaluated_tokens=1,
            target_was_not_truncated=True,
        ),
        cache_lifecycle={f"{b}_active_kv_storage_bytes": 100 for b in ("initial", "final", "peak")},
    )
    assert audit_phase_trace(trace, 100)["phase"] == ("teacher_forced" if teacher else "generation")
    trace["cache_lifecycle"]["peak_active_kv_storage_bytes"] = 200
    with pytest.raises(ValueError, match="physical byte cap"):
        audit_phase_trace(trace, 100)
    assert audit_phase_trace(trace, None)  # FullKV reference is not byte capped.
