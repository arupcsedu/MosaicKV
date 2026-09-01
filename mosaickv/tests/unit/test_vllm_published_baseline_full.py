from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _module(name: str, filename: str) -> Any:
    path = Path(__file__).parents[2] / "scripts" / filename
    sys.path.insert(0, str(Path(__file__).parents[2]))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_campaign_preparation_preserves_paired_backend_methods(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module(
        "mosaickv_test_prepare_vllm_published_baseline_full",
        "prepare_vllm_published_baseline_full.py",
    )
    tasks = {
        "mmstar": {**module.TASKS["mmstar"], "sample_count": 3},
        "textvqa": {**module.TASKS["textvqa"], "sample_count": 4},
        "docvqa": {**module.TASKS["docvqa"], "sample_count": 5},
    }
    monkeypatch.setattr(module, "TASKS", tasks)
    rows: list[dict[str, Any]] = []
    for task, route in tasks.items():
        for index in range(route["sample_count"]):
            sample_id = f"{task}:{index}"
            for backend, methods in module.SOURCE_METHODS_BY_BACKEND.items():
                rows.extend(
                    {
                        "task": task,
                        "sample_id": sample_id,
                        "backend_scope": backend,
                        "protocol_method": method,
                        "status": "completed",
                    }
                    for method in methods
                )
    source = tmp_path / "source.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    output = tmp_path / "campaign"
    source_sha = "a" * 40

    protocol = module.prepare(
        source,
        output,
        source_git_sha=source_sha,
        shard_size=2,
    )

    assert protocol["job_count"] == 7
    assert protocol["maximum_concurrent_a100s"] == 8
    assert protocol["cpus_per_job"] == 12
    assert protocol["time_limit"] == "02:00:00"
    index = json.loads((output / "inputs/job_index.json").read_text())
    assert len(index["jobs"]) == 7
    assert sum(job["sample_count"] for job in index["jobs"]) == 12
    assert {tuple(job["methods"]) for job in index["jobs"]} == {tuple(module.BASELINES)}


def test_full_hf_worker_uses_locked_baseline_configuration() -> None:
    module = _module(
        "mosaickv_test_run_vllm_published_baseline_full_hf_worker",
        "run_vllm_published_baseline_full_hf_worker.py",
    )
    package_root = Path(__file__).parents[2]
    template = module.load_config(package_root / "configs/paper_candidate_llava_v1.yaml")

    configs = module._configs(template, task="textvqa", seed=20271001, output_length=64)

    assert tuple(configs) == module.METHODS
    for method, config in configs.items():
        assert config.method.value == method
        assert config.cache.block_size == 1
        assert config.cache.physical_budget_block_size == 1
        assert config.cache.retention_ratio == 0.13
        assert config.execution.seed == 20271001
        assert config.generation.max_new_tokens == 64
    assert configs["prefixkv_reimpl"].prefixkv.profile_mode.value == "offline_profile"
    assert configs["lookm_reimpl"].lookm.merge_strategy.value == "pivotal"


def test_full_analyzer_pairs_mosaic_with_every_competitor(monkeypatch: Any) -> None:
    module = _module(
        "mosaickv_test_analyze_vllm_published_baseline_full",
        "analyze_vllm_published_baseline_full.py",
    )
    monkeypatch.setattr(module, "TASKS", {"mmstar": 2})
    rows: list[dict[str, Any]] = []
    for backend, mosaic_method in module.MOSAIC_METHOD.items():
        for sample in ("mmstar:0", "mmstar:1"):
            for method in (
                "prompt_attention_topk",
                *module.BASELINES,
                mosaic_method,
            ):
                rows.append(
                    {
                        "task": "mmstar",
                        "sample_id": sample,
                        "backend_scope": backend,
                        "protocol_method": method,
                        "task_score": 1.0,
                        "answer": "same",
                        "active_kv_bytes": 70 if method == mosaic_method else 100,
                    }
                )

    comparisons = module._comparisons(pd.DataFrame(rows))

    assert len(comparisons) == 8
    assert comparisons["memory_gate_30pct"].all()
    assert comparisons["quality_noninferiority_point_gate"].all()
