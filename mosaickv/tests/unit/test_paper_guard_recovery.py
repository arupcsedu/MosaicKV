import pytest
from evicastkv_paper_repair import EXPECTED_ERROR, failed_keys


def test_recovery_selects_only_guard_failures_not_quality_or_answers():
    rows = [
        dict(
            sample_index=3,
            method="mosaickv_exact",
            status="failed",
            error="AssertionError: " + EXPECTED_ERROR,
            answer="not inspected",
        ),
        dict(
            sample_index=7,
            method="full_kv",
            status="completed",
            generated_token_ids=[1] * 128,
            used_fullkv_fallback=False,
        ),
    ]
    assert failed_keys(rows) == [(3, "mosaickv_exact")]
    rows[0]["answer"] = "changed benchmark label must not affect execution selection"
    assert failed_keys(rows) == [(3, "mosaickv_exact")]
    rows[0]["error"] = "unrelated CUDA OOM"
    with pytest.raises(ValueError, match="not the registered"):
        failed_keys(rows)


def test_successful_but_incomplete_rows_are_not_reinterpreted_as_repairable():
    with pytest.raises(ValueError, match="incomplete"):
        failed_keys(
            [
                dict(
                    sample_index=0,
                    method="mosaickv_exact",
                    status="completed",
                    generated_token_ids=[1],
                    used_fullkv_fallback=False,
                )
            ]
        )
