from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mosaickv.evaluation.shared_byte_contract import file_digest, load_contract

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def preparer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "mosaickv/scripts"))
    path = ROOT / "mosaickv/scripts/prepare_prefixkv_native_validation.py"
    spec = importlib.util.spec_from_file_location("prefixkv_native_validation", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", ["llava_description", "mmvet_prefixkv"])
@pytest.mark.parametrize("cap", [40, 80, 120, 160, 240])
def test_native_configs_share_controls_and_preserve_frozen_policy(preparer, task, cap):
    contract = load_contract(ROOT / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml")
    cfgs = preparer.configurations(ROOT, task, contract, cap)
    assert set(cfgs) == {"full_kv", "prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact"}
    prefix = cfgs["prefixkv_reimpl"]
    assert prefix.prefixkv.decode_policy == "append_only"
    assert prefix.prefixkv.question_score_weight == 0
    assert prefix.prefixkv.layer_budget_floor_fraction == 0
    assert not prefix.prefixkv.exclude_protected_from_calibration
    assert prefix.prefixkv.profile_reference_retention_ratio == 0.5
    for cfg in cfgs.values():
        assert cfg.model == prefix.model
        assert cfg.dataset == prefix.dataset
        assert cfg.generation == prefix.generation
        assert cfg.execution == prefix.execution
        assert cfg.generation.max_new_tokens == 1024
        assert cfg.cache.physical_budget_mandatory_policy == "first_and_last"
        if cfg.method.value != "full_kv":
            assert cfg.cache == prefix.cache
            assert cfg.cache.budget_value == cap * 2**20


def test_cohort_is_seeded_id_only_and_order_independent(preparer):
    ids = [f"id-{i}" for i in range(100)]
    first = preparer.cohort_ids(ids, 3, "task")
    assert first == preparer.cohort_ids(ids[::-1], 3, "task")
    assert len(first) == len(set(first)) == 32
    assert first != preparer.cohort_ids(ids, 4, "task")
    with pytest.raises(ValueError, match="distinct identities"):
        preparer.cohort_ids(ids + ids, 3, "task")


def test_worker_import_and_disabled_scope_fail_before_gpu(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "mosaickv/scripts"))
    path = ROOT / "mosaickv/scripts/verify_prefixkv_native_fidelity.py"
    spec = importlib.util.spec_from_file_location("prefixkv_native_fidelity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = []
    monkeypatch.setattr(module, "_require_clean_source", lambda source, sha: checked.append(sha))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(dict(source_sha="a" * 40, fidelity_execution_authorized=False)))
    with pytest.raises(ValueError, match="authorization"):
        module.check_inputs(tmp_path, file_digest(plan_path), 0)
    assert checked == ["a" * 40]
    with pytest.raises(ValueError, match="plan hash"):
        module.check_inputs(tmp_path, "b" * 64, 0)


def test_native_fidelity_slurm_sets_deterministic_cublas_before_python():
    text = (ROOT / "mosaickv/slurm/prefixkv_native_fidelity.sbatch").read_text()
    export = text.index("CUBLAS_WORKSPACE_CONFIG=:4096:8")
    launch = text.index("verify_prefixkv_native_fidelity.py")
    assert export < launch
