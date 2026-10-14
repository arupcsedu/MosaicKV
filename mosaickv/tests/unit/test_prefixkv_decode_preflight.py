import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture
def worker(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "prefix_decode_worker", root / "scripts/run_prefixkv_decode_preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", ["docvqa", "textvqa"])
def test_corrected_variants_share_every_comparison_control(worker, task):
    import json

    from mosaickv.config import RunConfig, canonical_config_json

    root = Path(__file__).resolve().parents[3]
    configs = worker.configs(root, task)
    names = (
        "prompt_attention_topk",
        "byte_rescaled_prefixkv_reimpl",
        "prefixkv_selection_only_reimpl",
        "mosaickv_exact",
    )
    first = configs[names[0]]
    for name in names:
        cfg = configs[name]
        for field in ("cache", "execution", "generation", "model", "dataset"):
            assert getattr(cfg, field) == getattr(first, field)
        assert RunConfig.from_mapping(json.loads(canonical_config_json(cfg))) == cfg
    evict = configs["byte_rescaled_prefixkv_reimpl"]
    append = configs["prefixkv_selection_only_reimpl"]
    assert replace(evict.prefixkv, decode_policy="append_only") == append.prefixkv
    native = configs["native_r050_prefixkv_reimpl"]
    assert native.cache.physical_budget_block_size is None
    assert native.prefixkv.allocation_policy == "upstream_round"
    assert not native.prefixkv.rescale_offline_profile_to_byte_budget


def test_new_development_ids_do_not_overlap_prior_debug_or_calibration(worker):
    ids = [str(i) for i in range(32)]
    assert worker.derived_ids(ids) == tuple(ids[24:32])
    assert not set(worker.derived_ids(ids)) & set(ids[:24])
    with pytest.raises(ValueError):
        worker.derived_ids(ids[:31])
    with pytest.raises(ValueError):
        worker.derived_ids([*ids[:-1], ids[0]])


@pytest.mark.parametrize("task", ["textvqa", "docvqa"])
def test_allocation_ablations_preserve_shared_controls_and_frozen_profile(worker, tmp_path, task):
    import json

    from mosaickv.baselines.prefixkv import (
        PrefixKVCalibrationObservation,
        generate_prefixkv_byte_profile,
    )
    from mosaickv.config import RunConfig, canonical_config_json

    profile = generate_prefixkv_byte_profile(
        (PrefixKVCalibrationObservation("cal-only", ((1.0,) * 128,) * 32),),
        byte_cap=40 * 2**20,
        bytes_per_layer_position=16384,
        model_id=worker.MODEL,
        model_revision=worker.REVISION,
        dataset_id="synthetic",
        dataset_revision="b" * 40,
        calibration_split="train",
        evaluation_sample_ids=("eval-only",),
        seed=7,
    )
    path = profile.write(tmp_path / "frozen.json")
    arms = worker.allocation_configs(Path(__file__).resolve().parents[3], task, path, profile)
    assert len(arms) == 7
    for name, config in arms.items():
        if name != "full_kv":
            for field in ("cache", "execution", "generation", "model", "dataset"):
                assert getattr(config, field) == getattr(arms["prompt_attention_topk"], field)
        assert RunConfig.from_mapping(json.loads(canonical_config_json(config))) == config
    assert arms["prefixkv_capacity_floor_reimpl"].prefixkv.layer_budget_floor_fraction == 0.5
    assert arms["prefixkv_uniform_layer_reimpl"].prefixkv.profile_path is None
    bad = replace(profile, model_revision="c" * 40)
    with pytest.raises(ValueError, match="LLaVA profile"):
        worker.allocation_configs(Path(__file__).resolve().parents[3], task, path, bad)
