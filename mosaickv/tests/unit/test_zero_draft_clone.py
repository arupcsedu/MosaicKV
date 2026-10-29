"""Synthetic numerical-equivalence gate tests, not experiment measurements."""

from copy import deepcopy

import evicastkv_zero_draft_clone as module
import pytest


def fixture():
    return dict(
        answer="fixture",
        nll_sum=2.0,
        evaluated_tokens=2,
        teacher_forced_ppl=2.0,
        rouge_l_f1=0.5,
        config_sha="fixture",
        audited_request_owned_peak_bytes=600,
        audited_decode_transaction_peak_bytes=200,
        metrics={
            **dict(zip(module.common.STORAGE, (100, 150, 200), strict=True)),
            "end_to_end_time": 1.0,
        },
    )


def test_equivalence_gate_separates_quality_storage_ownership_and_latency():
    old = fixture()
    new = deepcopy(old)
    new["audited_request_owned_peak_bytes"] = 450
    new["metrics"]["end_to_end_time"] = 2.0
    result = module.compare_rows(new, old)
    assert result["exact_quality_and_config_equivalence"]
    assert result["exact_live_storage_equivalence"]
    assert result["registered_owner_peak_reduced"]
    assert result["new_end_to_end_seconds"] > result["old_end_to_end_seconds"]


@pytest.mark.parametrize(
    "field", ["answer", "nll_sum", "evaluated_tokens", "rouge_l_f1", "config_sha"]
)
def test_quality_change_cannot_be_hidden_by_memory_reduction(field):
    old = fixture()
    new = deepcopy(old)
    new[field] = "different"
    new["audited_request_owned_peak_bytes"] = 450
    assert not module.compare_rows(new, old)["exact_quality_and_config_equivalence"]
