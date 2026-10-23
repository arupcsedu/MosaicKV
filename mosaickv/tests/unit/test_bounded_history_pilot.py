"""CPU-only plan/schema checks; no data access, jobs, or model downloads."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from mosaickv.config import BoundedDecodeConfig, RunConfig, canonical_config, config_sha256


def tooling():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "bounded_pilot", root / "mosaickv/scripts/bounded_history_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", ["llava_description", "mmvet_prefixkv"])
@pytest.mark.parametrize("cap", [40, 80])
def test_all_core_baselines_receive_same_lifetime_cap_and_generation(task, cap, tmp_path):
    module = tooling()
    route = {"dataset_id": "synthetic/native", "dataset_revision": "b" * 40, "split": "test"}
    configs = module.configurations(task, cap, route, tmp_path / "profile.json")
    reference = configs.pop("full_kv")
    assert reference.bounded_decode is None
    assert reference.cache.retention_ratio == 1.0
    for cfg in configs.values():
        assert cfg.cache.budget_value == cap * 2**20
        assert cfg.bounded_decode == BoundedDecodeConfig(32, 4)
        assert cfg.generation == reference.generation
        assert cfg.execution == reference.execution
        assert cfg.model == reference.model
        assert cfg.dataset == reference.dataset
        assert (
            cfg.cache.physical_budget_mandatory_policy
            == reference.cache.physical_budget_mandatory_policy
        )
        assert config_sha256(RunConfig.from_mapping(canonical_config(cfg))) == config_sha256(cfg)
    assert configs["prefixkv_reimpl"].prefixkv.decode_policy == "append_only"
    assert configs["prefixkv_reimpl"].prefixkv.hf_cache_transport == "ragged_llama_eager_v1"


def test_pilot_resources_and_scope_are_explicit():
    module = tooling()
    protocol = yaml.safe_load(module.PROTOCOL.read_text())
    assert protocol["max_global_a100s"] == 4
    assert protocol["cpus_per_worker"] == 12
    assert protocol["time_limit_seconds"] == 7200
    assert protocol["paper_campaign_authorized"] is False
    assert protocol["heldout_claim"] is False
    assert protocol["cross_budget_direct_comparisons"] == "forbidden"


@pytest.mark.parametrize("matched", [True, False])
def test_analysis_reduces_corpus_ppl_and_excludes_unequal_storage(tmp_path, monkeypatch, matched):
    """Synthetic CI records only: never experimental measurements."""
    import json
    import math

    import pandas as pd

    module = tooling()
    task = "llava_description"
    methods = ["full_kv", "prompt_attention_topk", "mosaickv_exact"]
    plan = dict(
        jobs=[
            dict(index=i, task=task, method=m, cap_mib=None if i == 0 else 40)
            for i, m in enumerate(methods)
        ],
        protocol=dict(
            samples_per_task=2,
            seed=7,
            tasks=[task],
            caps_mib=[40],
            methods=methods[1:],
            bootstrap_replicates=100,
        ),
    )
    monkeypatch.setattr(module, "inputs", lambda *_args: plan)
    for job in plan["jobs"]:
        observations, rows = [], []
        for sample_id, nll, tokens in [("ci-a", 2.0, 1), ("ci-b", 8.0, 7)]:
            for metric in ["teacher_forced_ppl", "rouge_l_f1"]:
                observations.append(
                    dict(
                        sample_id=sample_id,
                        metric_name=metric,
                        metric_scope="sample",
                        numerator=nll,
                        denominator=tokens,
                        metric_value=0.5,
                        method=job["method"],
                    )
                )
            rows.append(
                dict(
                    sample_id=sample_id,
                    end_to_end_time=1.0,
                    initial_active_kv_storage_bytes=40 * 2**20,
                    final_active_kv_storage_bytes=40 * 2**20,
                    peak_active_kv_storage_bytes=40 * 2**20
                    - (int(job["index"] == 2 and not matched)),
                )
            )
        path = tmp_path / f"observations-{job['index']}.parquet"
        pd.DataFrame(observations).to_parquet(path, index=False)
        module.write_once(
            tmp_path / "workers" / f"{job['index']:02d}.json",
            dict(
                plan_sha256="synthetic-ci-not-a-result",
                index=job["index"],
                artifacts={},
                record=dict(rows=rows, metric_observations_parquet_output=str(path)),
            ),
        )
    module.analyze(tmp_path, "synthetic-ci-not-a-result")
    report = json.loads((tmp_path / "analysis/report.json").read_text())
    summary = pd.read_parquet(tmp_path / "analysis/summary.parquet")
    assert summary.corpus_ppl.to_list() == pytest.approx([math.exp(10 / 8)] * 3)
    assert bool(report["comparisons"]) == matched
    assert bool(report["failures"]) != matched
