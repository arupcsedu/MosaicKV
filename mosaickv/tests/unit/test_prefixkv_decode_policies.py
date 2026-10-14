from dataclasses import asdict, replace

import numpy as np
import pytest
from test_prefixkv_reimpl import _state

from mosaickv.baselines.prefixkv import (
    PrefixKVOfflineProfile,
    PrefixKVReimplementationError,
    build_prefixkv_reimpl_plan,
)
from mosaickv.config import CacheConfig, PrefixKVConfig
from mosaickv.types import BudgetUnit


def native_plan(*, length=101, cap=None, ratio=0.5, policy="profile_ratio"):
    state = _state(sequence=length)
    profile = PrefixKVOfflineProfile(
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
        target_retention_ratio=0.5,
        layer_forget_ratios=(0.411, 0.589),
        calibration_dataset_id="synthetic",
        calibration_dataset_revision="b" * 40,
        calibration_split="train",
        calibration_sample_ids=("cal",),
        calibration_seed=0,
        start_size=1,
        protect_size=1,
    )
    config = PrefixKVConfig(
        enabled=True,
        profile_path="unused.json",
        hf_cache_transport="ragged_llama_eager_v1",
        decode_boundary_policy="protected",
        allocation_policy="upstream_round",
        decode_policy="profile_ratio",
    )
    if policy != "profile_ratio":
        config = replace(config, allocation_policy="global_apportion", decode_policy=policy)
    return build_prefixkv_reimpl_plan(
        state,
        tuple(np.arange(length)[None, None, None, :] for _ in range(2)),
        config,
        CacheConfig(state.active_bytes if cap is None else cap, BudgetUnit.BYTES, ratio, 1),
        model_id=profile.model_id,
        model_revision=profile.model_revision,
        profile=profile,
    )


def test_native_uses_original_float_not_realized_ratio():
    plan = native_plan()
    assert plan.layer_cache_sizes == (59, 42)
    assert plan.layers[0].decode_retention_ratio == 1 - 0.411
    assert plan.layers[0].decode_retention_ratio != plan.layers[0].retention_ratio
    assert plan.trace()["allocation_policy"] == "upstream_round"


def test_native_rounding_cannot_silently_change_cap():
    plan = native_plan()
    with pytest.raises(PrefixKVReimplementationError, match="byte budget"):
        native_plan(cap=plan.retained_bytes - 1)


@pytest.mark.parametrize("policy", ["profile_ratio", "effective_profile_ratio", "append_only"])
def test_identity_remains_identity(policy):
    plan = native_plan(ratio=1.0, policy=policy)
    assert plan.layer_cache_sizes == (101, 101)
    assert all(layer.decode_retention_ratio == 1 for layer in plan.layers)


def test_append_only_label_and_same_prefill_bytes():
    plan = native_plan(policy="append_only")
    control = native_plan(policy="realized_ratio")
    assert plan.retained_bytes == control.retained_bytes
    assert plan.implementation_label == "prefixkv_selection_only_reimpl"
    assert not plan.trace()["paper_native_algorithm"]


def test_policy_configuration_fails_closed():
    config = native_plan().config
    assert PrefixKVConfig.from_mapping(asdict(config)) == config
    for changes in (
        {"decode_policy": "unknown"},
        {"allocation_policy": "unknown"},
        {"allocation_policy": "global_apportion"},
        {"rescale_offline_profile_to_byte_budget": True},
        {"hf_cache_transport": "common_guarded"},
    ):
        with pytest.raises(ValueError):
            replace(config, **changes)
