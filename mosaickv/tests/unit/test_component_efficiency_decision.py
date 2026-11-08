from copy import deepcopy

import pytest

from mosaickv.evaluation.component_efficiency_decision import build, markdown


def fixture():
    comparison = dict(
        paired_mean_ratio=0.97,
        adjusted_one_sided_upper_ratio=0.98,
        paired_latency_engineering_gate=True,
        original=dict(mean=1.0),
        inline=dict(mean=0.97),
    )
    storage = {
        k: [128]
        for k in ("initial_encoded_bytes", "final_encoded_bytes", "reported_counter_peak_bytes")
    }
    admission = dict(
        complete=True,
        errors=[],
        source_only_component_elimination_engineering_gate=True,
        results={
            str(i): dict(
                all_logits_bitwise=True,
                all_greedy_parent_agreement=True,
                storage={
                    arm: deepcopy(storage)
                    for arm in (
                        "common_storage_wrapped",
                        "common_storage_hook_free",
                        "retained_hybrid",
                    )
                },
                comparisons={
                    name: deepcopy(comparison)
                    for name in (
                        "common_storage_hook_free_versus_common_storage_wrapped",
                        "common_storage_hook_free_versus_retained_hybrid",
                        "retained_hybrid_versus_common_storage_wrapped",
                    )
                },
            )
            for i in (0, 1)
        },
    )
    codec = dict(
        complete=True,
        errors=[],
        CPU_storage_gate_passed=False,
        results={
            str(i): dict(original_code_bits_exact=True, savings_fraction=0.0, saved_bytes=0)
            for i in (0, 1)
        },
    )
    return admission, codec, deepcopy(codec), deepcopy(codec)


def test_all_arm_rows_are_contiguous_and_not_promoted_to_six_component_claim():
    result = build(*fixture())
    assert len(result["measured_requests"]) == 6
    assert not result["new_benchmark_quality_or_all_baseline_superiority_established"]
    text = markdown(result)
    assert text.index("| 1 | retained_hybrid") < text.index("Hook-free versus")
    assert "Do not round a 3.82% observation into 4%" in text
    assert "not active GPU KV" in text


def test_incomplete_or_different_storage_never_generates_a_measured_decision():
    data = fixture()
    data[0]["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        build(*data)
    data = fixture()
    data[0]["results"]["1"]["storage"]["common_storage_hook_free"]["final_encoded_bytes"] = [129]
    with pytest.raises(ValueError, match="boundaries"):
        build(*data)
