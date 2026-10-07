from __future__ import annotations

import importlib.util
from pathlib import Path

from mosaickv.config import load_config


def _module() -> object:
    path = Path(__file__).parents[2] / "scripts" / "run_baseline_fidelity_review_worker.py"
    spec = importlib.util.spec_from_file_location("byte_cap_review_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vl_cache_byte_capped_configs_share_the_physical_contract() -> None:
    root = Path(__file__).parents[2]
    module = _module()
    configs = module.byte_capped_configs(
        load_config(root / "configs/paper_candidate_llava_v1.yaml"),
        repository=root.parent,
        family="vl_cache",
        task="docvqa",
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        model_revision="2424fdd47412fccc66d91719126b420e9fbd7065",
        seed=7,
        output_length=64,
        byte_cap=40 * 2**20,
    )
    assert tuple(configs) == (
        "full_kv",
        "prompt_attention_topk",
        "vl_cache_reimpl",
        "mosaickv_exact",
    )
    compressed = [configs[name] for name in tuple(configs)[1:]]
    assert {config.cache for config in compressed} == {compressed[0].cache}
    assert compressed[0].cache.budget_value == 40 * 2**20
    assert compressed[0].cache.physical_budget_mandatory_policy == "last_token"

    mosaic = configs["mosaickv_exact"]
    # The latency-safe execution policy must not relax the physical-budget
    # contract.  It merely leaves deterministic Prompt Top-K routing outside
    # the graph for stable heads and bounds graph planning to uncertain heads.
    assert mosaic.selection.routed_head_only_planning is True
    assert mosaic.selection.stable_head_prompt_routing is True
    assert mosaic.selection.algorithm == "vectorized_coverage"
    assert mosaic.selection.physical_capacity_fraction == 1.0
    assert mosaic.graph.candidate_graph_mode == "shortlist_rerank"
    assert mosaic.graph.candidate_descriptor_pooling is True
    assert mosaic.graph.uncertain_head_only is True
    assert mosaic.graph.uncertain_head_fraction == 1.0 / 64.0
    assert mosaic.prototypes.enabled is False
    assert mosaic.prototypes.uncertain_head_only is True
    assert mosaic.prototypes.uncertain_head_fraction == 1.0 / 64.0
