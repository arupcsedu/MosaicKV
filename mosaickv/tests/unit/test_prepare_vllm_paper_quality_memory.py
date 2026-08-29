from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml

from mosaickv.config import load_config
from mosaickv.types import MosaicKVMethod, RepairPolicy


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "prepare_vllm_paper_quality_memory.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_paper_plans", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_paper_protocol_freezes_scale_bootstrap_and_nonblocking_scope() -> None:
    package = Path(__file__).parents[2]
    path = package / "configs/experiments/paper_quality_memory_llava_vllm_v1.yaml"
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment = protocol["experiment"]
    assert protocol["execution_authorized"] is True
    assert experiment["tasks"] == ["mmstar", "textvqa", "docvqa"]
    assert experiment["methods"] == [
        "full_kv",
        "prompt_attention_topk",
        "mosaickv_exact",
    ]
    assert experiment["sample_count_per_task"] == 32
    assert experiment["bootstrap_resamples"] == 10_000
    assert protocol["scope"]["latency_is_not_a_paper_gate"] is True
    candidate_path = package / protocol["candidate"]["config"]
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == protocol["candidate"][
        "config_sha256"
    ]


def test_planning_configs_route_task_and_isolate_plan_only_behavior() -> None:
    package = Path(__file__).parents[2]
    template = load_config(package / "configs/paper_candidate_llava_v1.yaml")

    configs = SCRIPT.planning_configs(
        template,
        task="docvqa",
        seed=20270901,
        planner_output_length=1,
    )

    prompt = configs["prompt_attention_topk"]
    exact = configs["mosaickv_exact"]
    assert prompt.dataset.id == "lmms-lab/DocVQA"
    assert exact.dataset == prompt.dataset
    assert prompt.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
    assert not prompt.forecasting.enabled
    assert not prompt.graph.enabled
    assert not prompt.selection.enabled
    assert exact.method is MosaicKVMethod.MOSAICKV_EXACT
    assert exact.forecasting.enabled
    assert exact.graph.enabled
    assert exact.selection.enabled
    assert not exact.prototypes.enabled
    assert not exact.residual.enabled
    assert not exact.repair.enabled
    assert exact.repair.policy is RepairPolicy.NONE
    assert prompt.generation.max_new_tokens == exact.generation.max_new_tokens == 1
    assert prompt.execution.seed == exact.execution.seed == 20270901
