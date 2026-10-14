"""Synthetic controls/statistics; these values are not experiment measurements."""

from dataclasses import replace

import pytest
from test_prefixkv_evidence import make_profile
from test_prefixkv_evidence_ablation import controls

from mosaickv.evaluation.hf_matched_validation import (
    CANDIDATE,
    CONTROL,
    PREFIX_METHODS,
    matched_configs,
    select_disjoint_ids,
    summarize,
)


def profiles(tmp_path):
    return {
        arm: make_profile(False, False, alpha).write(tmp_path / f"{arm}.json")
        for arm, alpha in [("control", 0.0), ("c_rank", 0.5)]
    }


def test_c_only_frozen_profiles_and_append_only_controls(tmp_path):
    cfgs = matched_configs(controls(), profiles(tmp_path))
    assert tuple(cfgs) == PREFIX_METHODS
    for label in (CONTROL, CANDIDATE):
        assert cfgs[label].prefixkv.decode_policy == "append_only"
        assert not cfgs[label].prefixkv.exclude_protected_from_calibration
        assert cfgs[label].cache == cfgs["prompt_attention_topk"].cache
    assert cfgs[CANDIDATE].prefixkv.question_score_weight == 0.5
    assert cfgs[CONTROL].prefixkv.question_score_weight == 0
    assert all(c.cache.physical_budget_mandatory_policy == "first_and_last" for c in cfgs.values())


def test_mandatory_policy_and_frozen_alpha_cannot_drift(tmp_path):
    c = controls()
    c["mosaickv_exact"] = replace(
        c["mosaickv_exact"], cache=replace(c["mosaickv_exact"].cache, block_size=2)
    )
    with pytest.raises(ValueError, match="cache policy"):
        matched_configs(c, profiles(tmp_path))
    paths = profiles(tmp_path / "new")
    paths["c_rank"] = paths["control"]
    with pytest.raises(ValueError, match="profile contract"):
        matched_configs(controls(), paths)


def test_lookm_is_separate_reimplementation_without_prefix_profile():
    cfgs = matched_configs(controls(), lane="lookm")
    assert cfgs["lookm_reimpl"].lookm.enabled
    assert not cfgs["lookm_reimpl"].prefixkv.enabled
    assert cfgs["lookm_reimpl"].cache == cfgs["mosaickv_exact"].cache


def test_disjoint_stages_are_id_only_and_fail_closed():
    v, big = select_disjoint_ids(list(range(400)), list(range(20)))
    assert len(v) == 64 and len(big) == 256
    assert not set(v) & set(big)
    assert not (set(v) | set(big)) & set(range(20))
    with pytest.raises(ValueError, match="insufficient"):
        select_disjoint_ids(list(range(50)), [])
    with pytest.raises(ValueError, match="duplicate"):
        select_disjoint_ids([1, 1], [])


def records():
    return [
        dict(
            variant=m,
            rows=[
                dict(
                    sample_id=s,
                    task_score=1.0,
                    answer="x",
                    status="completed",
                    active_kv_bytes=41943040,
                    final_active_kv_bytes=74973184,
                    residual_kv_bytes=0,
                    repair_count=0,
                    generated_tokens=64,
                    decode_evictions=0,
                    trace_validated=True,
                    input_controls_sha256="a" * 64,
                    cache_controls_sha256="b" * 64,
                    reference="x",
                    end_to_end_time=2.0,
                    compression_time=0.1,
                    decode_time=1.0,
                )
                for s in ["a", "b", "c", "d"]
            ],
        )
        for m in PREFIX_METHODS
    ]


def test_paired_summary_is_reproducible_and_excludes_fullkv_latency():
    a = summarize(records(), ["a", "b", "c", "d"])
    assert a == summarize(records(), ["a", "b", "c", "d"])
    assert a["gates_passed"] and a["memory_gate"]
    assert "vs_topk" not in a["methods"]["full_kv"]
    assert not a["paper_claim_authorized"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_active_kv_bytes", 75000000),
        ("active_kv_bytes", 41943039),
        ("decode_evictions", 1),
        ("trace_validated", False),
        ("end_to_end_time", float("nan")),
        ("generated_tokens", 63),
    ],
)
def test_matching_requires_real_bytes_traces_finite_timings_and_output(field, value):
    r = records()
    r[3]["rows"][0][field] = value
    assert not summarize(r, ["a", "b", "c", "d"])["gates_passed"]


def test_failures_and_duplicates_are_not_silently_dropped():
    r = records()
    r[3]["rows"].pop()
    assert not summarize(r, ["a", "b", "c", "d"])["gates_passed"]
    r = records()
    r[3]["rows"].append(r[3]["rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(r, ["a", "b", "c", "d"])


def test_zero_quality_or_quality_loss_cannot_promote():
    r = records()
    for rec in r:
        for row in rec["rows"]:
            row["task_score"] = 0.0
    assert not summarize(r, ["a", "b", "c", "d"])["gates_passed"]
    r = records()
    r[3]["rows"][0]["task_score"] = 0
    assert not summarize(r, ["a", "b", "c", "d"])["gates_passed"]
