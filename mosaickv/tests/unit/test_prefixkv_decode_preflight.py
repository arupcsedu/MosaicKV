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
