from copy import deepcopy

import pytest

from mosaickv.evaluation.six_stage_impact import STAGES, build, markdown


def fixtures():
    p = dict(
        all_128_logits_bitwise=True,
        labels_accessed=False,
        mechanism_evidence=dict(
            extra_gpu_bytes=17,
            residual_cpu_bytes=23,
            layers={
                "31": dict(
                    construction_phase_seconds=dict(
                        **{k: 0.001 for _, phases, _ in STAGES for k in phases},
                        route_and_descriptors=0.002,
                    )
                )
            },
        ),
    )
    panel = dict(
        execution_passed=True,
        natural_repair_events=0,
        effects={
            name: dict(
                full_minus_arm_cider=-1.0,
                retrospective_unadjusted_95_interval=[-2.0, 0.0],
                full_minus_arm_caption_changes=1,
            )
            for _, _, name in STAGES
        },
    )
    return [deepcopy(p), deepcopy(p)], panel, dict(complete=True, results={})


def test_missing_stage_memory_and_repair_latency_remain_unknown_not_zero():
    result = build(*fixtures())
    assert len(result["stages"]) == 6
    assert result["stages"][-1]["mean_profiled_milliseconds"] is None
    assert all(r["separately_audited_stage_KV_bytes"] is None for r in result["stages"])
    assert all(
        r["historical_16_image_unguarded_ablation"]["same_version_as_latency_run"] is False
        for r in result["stages"]
    )
    assert result["superiority_established"] is False
    text = markdown(result)
    assert "different, earlier" in text and "not isolated" in text
    assert "-1.00000" in text


def test_known_temporary_buffers_are_not_promoted_to_stage_peak_or_kv_storage():
    profiles, panel, latency = fixtures()
    for p in profiles:
        p["mechanism_evidence"]["inline_first_step_repair"] = dict(
            temporary_key_clone_backing_bytes=99, temporary_attention_residual_backing_bytes=8
        )
        p["mechanism_evidence"]["live_anchor_guard"] = dict(
            tail_graph=dict(record_buffer_activation_bytes=40)
        )
    result = build(profiles, panel, latency)
    repair = result["stages"][-1]
    assert repair["separately_audited_stage_KV_bytes"] is None
    assert all(
        v["stage_peak_certified"] is False for v in repair["known_first_step_repair_buffers"]
    )
    assert result["per_input_guard_activation_buffers"][0]["not_KV_bytes"] is True
    assert "not stage peak" in markdown(result)


@pytest.mark.parametrize("failure", ("missing_phase", "nan", "invalid_parity", "incomplete"))
def test_bad_profiles_cannot_generate_measured_report(failure):
    p, panel, latency = fixtures()
    if failure == "missing_phase":
        del p[0]["mechanism_evidence"]["layers"]["31"]["construction_phase_seconds"]["sparse_graph"]
    if failure == "nan":
        p[0]["mechanism_evidence"]["layers"]["31"]["construction_phase_seconds"]["sparse_graph"] = (
            float("nan")
        )
    if failure == "invalid_parity":
        p[0]["all_128_logits_bitwise"] = False
    if failure == "incomplete":
        latency["complete"] = False
    with pytest.raises((ValueError, KeyError)):
        build(p, panel, latency)
