"""CPU-only sweep contracts. Synthetic fixtures are never measured results."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "mosaickv/scripts"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import prepare_vllm_baseline_sweep as prepare
    import run_vllm_baseline_sweep as run

    return prepare, run


@pytest.mark.parametrize("task", ["mmstar", "textvqa", "docvqa"])
@pytest.mark.parametrize("cap", [40, 80, 120])
def test_all_methods_share_inputs_and_cap(modules, task, cap):
    from mosaickv.config import canonical_config_json, load_config

    _, run = modules
    arms = run.configurations(REPO, task, cap * 2**20)
    assert tuple(arms) == run.METHODS
    prompt = arms["prompt_attention_topk"]
    for name, config in arms.items():
        assert config.model == prompt.model
        assert config.dataset == prompt.dataset
        assert config.execution == prompt.execution
        assert config.generation == prompt.generation
        if name != "full_kv":
            assert config.cache == prompt.cache
            assert config.cache.budget_value == cap * 2**20
            assert config.cache.physical_budget_mandatory_policy == "first_and_last"
        assert not config.repair.enabled
        assert not config.prototypes.enabled
        assert not config.residual.enabled
    from prepare_vllm_byte_capped_mirror import project_backend_config

    prefix = json.loads(canonical_config_json(arms["prefixkv_reimpl"]))
    projected = project_backend_config(prefix)
    assert prefix["prefixkv"]["decode_policy"] == "effective_profile_ratio"
    assert projected["prefixkv"]["decode_policy"] == "realized_ratio"
    projected["execution"]["backend"] = "vllm"
    projected["cache"]["block_size"] = 16
    projected["cache"]["physical_budget_block_size"] = 16
    # Mapping parser tests that previously HF-only policy cannot poison vLLM loading.
    from mosaickv.config import RunConfig

    RunConfig.from_mapping(projected)
    assert callable(load_config)


def test_task_and_cap_fail_closed(modules):
    _, run = modules
    with pytest.raises(ValueError, match="unregistered"):
        run.configurations(REPO, "mathvista", 40 * 2**20)
    with pytest.raises(ValueError, match="unregistered"):
        run.configurations(REPO, "docvqa", 10 * 2**20)


def test_id_only_subset_preserves_revision_and_explicitly_not_heldout(modules):
    prepare, _ = modules
    parent = {
        "task": "docvqa",
        "dataset_id": "fixture",
        "dataset_revision": "a" * 40,
        "split": "validation",
        "sample_ids": [f"docvqa:{i}" for i in range(16)],
    }
    result = prepare.ledger_subset(parent)
    assert result["sample_ids"] == parent["sample_ids"][:8]
    assert result["dataset_revision"] == parent["dataset_revision"]
    assert result["heldout_claim"] is False
    parent["sample_ids"][3] = parent["sample_ids"][0]
    with pytest.raises(ValueError, match="unique"):
        prepare.ledger_subset(parent)


@pytest.mark.parametrize("attested,stage", [(False, "preflight"), (True, "campaign")])
def test_dirty_source_exception_cannot_authorize_campaign(modules, tmp_path, attested, stage):
    from prepare_vllm_byte_capped_mirror import development_source_binding

    with pytest.raises(ValueError, match="development preflight"):
        development_source_binding(tmp_path / "a.json" if attested else None, tmp_path, stage)


def test_development_inventory_rejects_tampering(modules, tmp_path, monkeypatch):
    from prepare_vllm_byte_capped_mirror import development_source_binding, sha

    import mosaickv.manifest

    monkeypatch.setattr(mosaickv.manifest, "_worktree_fingerprint", lambda _: "fingerprint")
    code = tmp_path / "worker.py"
    code.write_text("immutable fixture")
    record = tmp_path / "inventory.json"
    record.write_text(
        json.dumps(
            {
                "source_root": str(tmp_path),
                "patch_sha": "fingerprint",
                "files": {"worker.py": sha(code)},
            }
        )
    )
    assert development_source_binding(record, tmp_path, "preflight")["patch_sha"] == "fingerprint"
    code.write_text("changed")
    with pytest.raises(ValueError, match="inventory differs"):
        development_source_binding(record, tmp_path, "preflight")


def test_projection_does_not_modify_original_source_config(modules):
    from prepare_vllm_byte_capped_mirror import project_backend_config

    source = {
        "method": "prefixkv_reimpl",
        "prefixkv": {
            "decode_policy": "effective_profile_ratio",
            "hf_cache_transport": "ragged_llama_eager_v1",
        },
    }
    original = copy.deepcopy(source)
    projected = project_backend_config(source)
    assert source == original
    assert projected != source
    assert project_backend_config({"method": "lookm_reimpl"}) == {"method": "lookm_reimpl"}


def test_worker_can_be_loaded_from_sibling_import_boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "vllm_sweep_import_fixture", SCRIPTS / "run_vllm_baseline_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.METHODS[-1] == "mosaickv_exact"
