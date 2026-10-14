"""Synthetic CPU tests, not measured benchmark quality evidence."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from mosaickv.baselines.prefixkv import (
    PrefixKVCalibrationObservation,
    PrefixKVOfflineProfile,
    PrefixKVReimplementationError,
    build_prefixkv_reimpl_plan,
    generate_prefixkv_byte_profile,
    generate_prefixkv_profile,
    search_prefixkv_layer_sizes,
)
from mosaickv.cache_state import FullKVState
from mosaickv.config import CacheConfig, PrefixKVConfig
from mosaickv.types import BudgetUnit


def metadata():
    return dict(
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
        dataset_id="synthetic",
        dataset_revision="b" * 40,
        calibration_split="train",
        evaluation_sample_ids=("eval-only",),
        seed=7,
        start_size=1,
        protect_size=1,
    )


def observations():
    return tuple(
        PrefixKVCalibrationObservation(
            f"cal-{i}",
            ((100.0,) + (1.0,) * (n - 1), (1.0,) * n) * 2,
        )
        for i, n in enumerate((20, 40))
    )


def profile():
    return generate_prefixkv_byte_profile(
        observations(), byte_cap=24 * 16, bytes_per_layer_position=16, **metadata()
    )


def state(length=30):
    return FullKVState.from_tensors(
        tuple(
            (
                np.arange(length * 2, dtype=np.float32).reshape(1, 1, length, 2).copy(),
                np.ones((1, 1, length, 2), dtype=np.float32),
            )
            for _ in range(4)
        ),
        block_size=1,
        mandatory_logical_positions=(0, length - 1),
    )


def configuration(p, floor=0.0):
    return PrefixKVConfig(
        enabled=True,
        profile_path="unused",
        rescale_offline_profile_to_byte_budget=True,
        profile_reference_retention_ratio=p.target_retention_ratio,
        hf_cache_transport="ragged_llama_eager_v1",
        decode_boundary_policy="protected",
        decode_policy="effective_profile_ratio",
        layer_budget_floor_fraction=floor,
    )


def plan(p=None, floor=0.0, cap=24 * 16):
    p = p or profile()
    full = state()
    return build_prefixkv_reimpl_plan(
        full,
        (np.ones((1, 1, 1, 30), dtype=np.float32),) * 4,
        configuration(p, floor),
        CacheConfig(
            cap, BudgetUnit.BYTES, 0.99, 1, physical_budget_mandatory_policy="first_and_last"
        ),
        model_id=p.model_id,
        model_revision=p.model_revision,
        profile=p,
        physical_prompt_capacity=6,
    )


def test_upstream_protected_cdf_is_not_a_minimum_only():
    scores = ((100.0,) + (1.0,) * 19, (1.0,) * 20)
    legacy = search_prefixkv_layer_sizes(
        scores,
        retention_ratio=0.25,
        start_size=1,
        protect_size=1,
    )
    corrected = search_prefixkv_layer_sizes(
        scores,
        retention_ratio=0.25,
        start_size=1,
        protect_size=1,
        search_policy="upstream_protected_cdf",
    )
    assert legacy == (2, 8)
    assert corrected == (3, 7)
    assert sum(corrected) == sum(legacy)


def test_native_protected_count_convention_against_pinned_upstream():
    import importlib.util
    from pathlib import Path

    import torch

    path = Path(__file__).resolve().parents[3] / "third_party/PrefixKV/prefixkv.py"
    spec = importlib.util.spec_from_file_location("prefixkv_calibration_oracle", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    # Upstream permits <5 positions of global rounding error. Compare that
    # documented convention, not a falsely asserted bitwise profiling match.
    scores = torch.tensor([[100.0] + [1.0] * 99, [1.0] * 100])
    forget = upstream.obtain_cdf_num(scores, 150, 2)
    actual = search_prefixkv_layer_sizes(
        tuple(tuple(x) for x in scores.tolist()),
        retention_ratio=0.25,
        start_size=1,
        protect_size=1,
        search_policy="upstream_protected_cdf",
    )
    assert actual[0] == 3
    assert sum(actual) == 50
    assert sum(abs(a - (100 - b)) for a, b in zip(actual, forget, strict=True)) < 5


def test_byte_profile_uses_same_cap_for_different_prompt_lengths():
    p = profile()
    assert p.calibration_byte_cap == 384
    assert sum(p.calibration_mean_layer_positions) == 24
    for observation in observations():
        sizes = search_prefixkv_layer_sizes(
            observation.layer_scores,
            retention_ratio=p.target_retention_ratio,
            start_size=1,
            protect_size=1,
            target_positions=24,
            search_policy="upstream_protected_cdf",
        )
        assert sum(sizes) * 16 == 384
    assert p.profile_sha256 == profile().profile_sha256
    assert PrefixKVOfflineProfile.from_json_object(p.to_json_object()) == p
    assert plan().retained_bytes == 384
    assert plan().implementation_label == "prefixkv_byte_calibrated_reimpl"


def test_old_profile_schema_and_digest_are_unchanged():
    p = generate_prefixkv_profile(observations(), retention_ratio=0.5, **metadata())
    payload = p.to_json_object()
    assert "byte_calibration" not in payload
    assert "search_policy" not in payload
    assert PrefixKVOfflineProfile.from_json_object(payload) == p
    revised = replace(p, search_policy="upstream_protected_cdf")
    assert revised.profile_sha256 != p.profile_sha256


def test_byte_profile_rejects_leakage_and_different_runtime_cap():
    with pytest.raises(PrefixKVReimplementationError, match="overlap"):
        generate_prefixkv_byte_profile(
            observations(),
            byte_cap=384,
            bytes_per_layer_position=16,
            **{**metadata(), "evaluation_sample_ids": ("cal-0",)},
        )
    with pytest.raises(PrefixKVReimplementationError, match="recorded physical cap"):
        plan(cap=400)
    with pytest.raises(PrefixKVReimplementationError, match="recorded physical cap"):
        plan(
            p=replace(
                profile(),
                calibration_bytes_per_layer_position=32,
                calibration_mean_layer_positions=tuple(
                    x / 2 for x in profile().calibration_mean_layer_positions
                ),
            )
        )


@pytest.mark.parametrize("cap", [16, 16 * 200])
def test_infeasible_calibration_fails_instead_of_expanding_cap(cap):
    with pytest.raises(PrefixKVReimplementationError):
        generate_prefixkv_byte_profile(
            observations(), byte_cap=cap, bytes_per_layer_position=16, **metadata()
        )


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_floor_redistributes_same_budget_without_padding(fraction):
    p = plan(floor=fraction)
    assert p.retained_bytes == 384
    assert sum(p.layer_cache_sizes) == 24
    assert min(p.layer_cache_sizes) >= int(fraction * 6)
    assert p.layers == plan(floor=fraction).layers
    for layer in p.layers:
        assert layer.selected_physical_positions[0] == 0
        assert layer.selected_physical_positions[-1] == 29
    if fraction == 1:
        assert p.layer_cache_sizes == (6, 6, 6, 6)
    if fraction > 0:
        assert p.implementation_label == "prefixkv_capacity_floor_reimpl"


def test_starved_real_profile_shape_receives_a_floor_not_extra_bytes():
    p = generate_prefixkv_profile(observations(), retention_ratio=0.5, **metadata())
    p = replace(p, layer_forget_ratios=(0, 0.99, 0.99, 0.99))
    without = plan(p=p)
    guarded = plan(p=p, floor=1.0)
    assert without.layer_cache_sizes == (15, 3, 3, 3)
    assert guarded.layer_cache_sizes == (6, 6, 6, 6)
    assert without.retained_bytes == guarded.retained_bytes == 384


@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_strict_floor_configuration(invalid):
    with pytest.raises(ValueError, match="layer_budget_floor_fraction"):
        configuration(profile(), invalid)


def test_floor_rejects_native_label_and_roundtrips():
    cfg = configuration(profile(), 0.5)
    assert PrefixKVConfig.from_mapping(asdict(cfg)) == cfg
    for changes in [
        dict(decode_policy="append_only"),
        dict(allocation_policy="upstream_round"),
        dict(rescale_offline_profile_to_byte_budget=False),
    ]:
        with pytest.raises(ValueError, match="capacity-floor"):
            replace(cfg, **changes)


@pytest.mark.parametrize("scores", [((),), ((float("nan"),),), ((-1.0,),)])
def test_invalid_calibration_attention_is_rejected(scores):
    with pytest.raises(PrefixKVReimplementationError, match="finite"):
        search_prefixkv_layer_sizes(scores, retention_ratio=0.5, start_size=1, protect_size=1)


def test_profile_cli_byte_mode_and_no_overwrite(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/generate_prefixkv_profile.py"
    spec = importlib.util.spec_from_file_location("prefix_profile_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    captures = tmp_path / "captures.jsonl"
    captures.write_text(
        "\n".join(
            json.dumps({"sample_id": x.sample_id, "layer_scores": x.layer_scores})
            for x in observations()
        )
    )
    ids = tmp_path / "ids.json"
    ids.write_text('["eval-only"]')
    output = tmp_path / "profile.json"
    args = [
        str(script),
        "--attention-jsonl",
        str(captures),
        "--evaluation-sample-ids",
        str(ids),
        "--output",
        str(output),
        "--model",
        "test",
        "--model-revision",
        "a" * 40,
        "--dataset",
        "synthetic",
        "--dataset-revision",
        "b" * 40,
        "--calibration-split",
        "train",
        "--byte-cap",
        "384",
        "--bytes-per-layer-position",
        "16",
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 0
    saved = output.read_bytes()
    payload = PrefixKVOfflineProfile.from_json_object(json.loads(saved))
    assert payload.calibration_byte_cap == 384
    with pytest.raises(FileExistsError, match="overwrite"):
        cli.main()
    assert output.read_bytes() == saved
    monkeypatch.setattr(sys, "argv", [*args, "--search-policy", "legacy_cdf"])
    with pytest.raises(SystemExit):
        cli.main()


def test_in_memory_profile_identity_and_boundaries_fail_closed():
    with pytest.raises(PrefixKVReimplementationError, match="protected-boundary"):
        full = state()
        p = profile()
        build_prefixkv_reimpl_plan(
            full,
            (np.ones((1, 1, 1, 30), dtype=np.float32),) * 4,
            configuration(p),
            CacheConfig(384, BudgetUnit.BYTES, 0.99, 1),
            model_id=p.model_id,
            model_revision=p.model_revision,
            profile=replace(p, start_size=2),
        )
