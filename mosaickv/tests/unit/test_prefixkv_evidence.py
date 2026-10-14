"""Synthetic formula/byte/provenance tests, not benchmark quality measurements."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from mosaickv.baselines.prefixkv import (
    PrefixKVCalibrationObservation,
    PrefixKVOfflineProfile,
    PrefixKVReimplementationError,
    build_prefixkv_reimpl_plan,
    generate_prefixkv_byte_profile,
    mix_prefixkv_scores,
    prefixkv_calibration_observation,
    prefixkv_question_scores,
    search_prefixkv_layer_sizes,
)
from mosaickv.cache_state import FullKVState
from mosaickv.config import CacheConfig, ConfigurationError, PrefixKVConfig
from mosaickv.types import BudgetUnit


def captures():
    rng = np.random.default_rng(17)
    attentions = tuple(rng.random((1, 2, 40, 40), dtype=np.float32) for _ in range(4))
    observations = tuple(
        prefixkv_calibration_observation(
            name, attentions, question_positions=(30, 31, 32), question_mapping_sha256="a" * 64
        )
        for name in ("cal-1", "cal-2")
    )
    return observations, attentions


def make_profile(sink=False, protect=False, weight=0.0):
    obs, _ = captures()
    return generate_prefixkv_byte_profile(
        obs,
        byte_cap=768,
        bytes_per_layer_position=16,
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="b" * 40,
        dataset_id="synthetic",
        dataset_revision="c" * 40,
        calibration_split="train",
        evaluation_sample_ids=("eval-1",),
        seed=17,
        exclude_protected_from_calibration=sink,
        prompt_reservation_policy="question_and_boundaries" if protect else "boundaries",
        question_score_weight=weight,
    )


def make_plan(sink=False, protect=False, weight=0.0):
    p = make_profile(sink, protect, weight)
    _, attn = captures()
    full = FullKVState.from_tensors(
        tuple(
            (np.ones((1, 1, 40, 2), dtype=np.float32), np.ones((1, 1, 40, 2), dtype=np.float32))
            for _ in range(4)
        ),
        block_size=1,
        mandatory_logical_positions=(0, 30, 31, 32, 39) if protect else (0, 39),
    )
    cfg = PrefixKVConfig(
        enabled=True,
        profile_path="unused",
        rescale_offline_profile_to_byte_budget=True,
        profile_reference_retention_ratio=p.target_retention_ratio,
        hf_cache_transport="ragged_llama_eager_v1",
        decode_boundary_policy="protected",
        decode_policy="effective_profile_ratio",
        exclude_protected_from_calibration=sink,
        question_score_weight=weight,
    )
    cache = CacheConfig(
        768,
        BudgetUnit.BYTES,
        0.99,
        1,
        physical_budget_mandatory_policy="question_and_boundaries" if protect else "first_and_last",
    )
    plan = build_prefixkv_reimpl_plan(
        full,
        attn,
        cfg,
        cache,
        model_id=p.model_id,
        model_revision=p.model_revision,
        profile=p,
        question_positions=(30, 31, 32),
        physical_prompt_capacity=12,
    )
    return plan, p, cfg, cache


def test_optional_cdf_ignores_arbitrarily_large_protected_sink():
    scores = ((1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0), (1.0,) * 8)
    huge = ((1e12, *scores[0][1:-1], 1e12), scores[1])
    kwargs = dict(
        retention_ratio=0.5,
        start_size=1,
        protect_size=1,
        target_positions=8,
        search_policy="optional_cdf",
    )
    assert search_prefixkv_layer_sizes(scores, **kwargs) == search_prefixkv_layer_sizes(
        huge, **kwargs
    )
    assert sum(search_prefixkv_layer_sizes(huge, **kwargs)) == 8
    with pytest.raises(PrefixKVReimplementationError, match="mandatory"):
        search_prefixkv_layer_sizes(scores, **dict(kwargs, target_positions=3))


def test_question_mixture_normalizes_optional_mass_independently():
    original = ((10000.0, 2.0, 0.0, 0.0),)
    question = ((10000.0, 0.0, 6.0, 0.0),)
    assert mix_prefixkv_scores(original, question, ((0, 3),), 0.5) == ((0.0, 0.5, 0.5, 0.0),)
    assert mix_prefixkv_scores(original, question, ((0, 3),), 0) == ((0.0, 1.0, 0.0, 0.0),)
    assert mix_prefixkv_scores(original, question, ((0, 3),), 1) == ((0.0, 0.0, 1.0, 0.0),)
    assert mix_prefixkv_scores(((0.0,) * 4,), ((0.0,) * 4,), ((0, 3),), 0.5) == (
        (0.0, 0.5, 0.5, 0.0),
    )
    with pytest.raises(PrefixKVReimplementationError, match="square"):
        prefixkv_question_scores((np.ones((1, 1, 2, 4)),), (1,))


@pytest.mark.parametrize(
    "sink,protect,weight", list(product((False, True), (False, True), (0.0, 0.5)))
)
def test_all_factorial_arms_fill_same_bytes_and_bind_settings(sink, protect, weight):
    plan, p, cfg, cache = make_plan(sink, protect, weight)
    assert plan.retained_bytes == 768
    assert sum(plan.layer_cache_sizes) == 48
    assert plan.layers == make_plan(sink, protect, weight)[0].layers
    assert PrefixKVOfflineProfile.from_json_object(p.to_json_object()) == p
    assert plan.implementation_label.endswith("_reimpl")
    assert not plan.trace()["paper_native_algorithm"]
    for layer in plan.layers:
        assert set((0, 30, 31, 32, 39) if protect else (0, 39)) <= set(
            layer.selected_physical_positions
        )
    if sink or protect or weight:
        assert "evidence_policy" in p.to_json_object()
        with pytest.raises(PrefixKVReimplementationError, match="mismatch"):
            build_prefixkv_reimpl_plan(
                plan.full_state,
                captures()[1],
                replace(cfg, exclude_protected_from_calibration=not sink),
                cache,
                model_id=p.model_id,
                model_revision=p.model_revision,
                profile=p,
                question_positions=(30, 31, 32),
            )
    else:
        assert "evidence_policy" not in p.to_json_object()


def test_question_metadata_required_and_no_calibration_leakage():
    with pytest.raises(PrefixKVReimplementationError, match="mapping hash"):
        PrefixKVCalibrationObservation("x", ((1.0,) * 4,), (1,), ((1.0,) * 4,))
    p = make_profile(True, True, 0.5)
    with pytest.raises(PrefixKVReimplementationError, match="overlap"):
        p.assert_evaluation_disjoint(("cal-1",))
    payload = p.to_json_object()
    payload["evidence_policy"]["question_score_weight"] = 0.9
    with pytest.raises(PrefixKVReimplementationError, match="profile_sha256"):
        PrefixKVOfflineProfile.from_json_object(payload)


@pytest.mark.parametrize("bad", [True, -1, 1.1, float("nan"), float("inf"), "0.5"])
def test_strict_question_weight(bad):
    with pytest.raises(ConfigurationError, match="question_score_weight"):
        PrefixKVConfig(question_score_weight=bad)


def test_shared_policy_cannot_be_faked_by_question_indices_alone():
    plan, p, cfg, cache = make_plan(True, True, 0.5)
    with pytest.raises(PrefixKVReimplementationError, match="mandatory"):
        build_prefixkv_reimpl_plan(
            plan.full_state,
            captures()[1],
            cfg,
            cache,
            model_id=p.model_id,
            model_revision=p.model_revision,
            profile=p,
            question_positions=(30, 31),
        )
