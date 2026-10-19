"""Synthetic preparation tests: no inference, downloads, or Slurm submission."""

import importlib.util
import sys
from pathlib import Path

import pytest

from mosaickv.config import DatasetConfig
from mosaickv.evaluation.worker_support import control_configs

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "native_shared_byte_prepare", SCRIPTS / "prepare_native_metric_shared_bytes.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
finally:
    sys.path.remove(str(SCRIPTS))


def test_label_blind_strata_are_balanced_disjoint_and_order_invariant():
    groups = {key: [f"{key}:{i}" for i in range(100)] for key in worker.MILEBENCH_SUBTASKS}
    excluded = [ids[0] for ids in groups.values()]
    a = worker.select_stages(groups, excluded=excluded)
    b = worker.select_stages(
        {key: list(reversed(ids)) for key, ids in groups.items()}, excluded=excluded
    )
    assert a == b
    assert len(a["validation"]) == 32 and len(a["larger"]) == 256
    assert not set(a["validation"]) & set(a["larger"])
    assert not set(excluded) & (set(a["validation"]) | set(a["larger"]))
    for stage in a.values():
        for offset in range(0, len(stage), 16):
            shard = stage[offset : offset + 16]
            assert {sid.split(":")[0] for sid in shard} == set(worker.MILEBENCH_SUBTASKS)


def test_selection_rejects_missing_coverage_and_duplicate_ids():
    with pytest.raises(ValueError, match="insufficient"):
        worker.select_stages({"a": ["1", "2"]})
    with pytest.raises(ValueError, match="duplicate"):
        worker.select_stages({"a": ["1"], "b": ["1"]})


def test_native_metric_worker_resources_and_deterministic_cublas_launch():
    text = (SCRIPTS.parent / "slurm/native_metric_shared_bytes.sbatch").read_text()
    for control in (
        "--cpus-per-task=12",
        "--gres=gpu:a100:1",
        "--time=02:00:00",
        "--reservation=bi_fox_dgx",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
    ):
        assert control in text


def test_frozen_contract_accepts_exact_coco_policy_and_rejects_retuning():
    from dataclasses import replace

    from mosaickv.evaluation.shared_byte_contract import (
        candidate_content_digest,
        load_contract,
        validate_configuration,
    )

    source = SCRIPTS.parents[1]
    contract = load_contract(source / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml")
    assert candidate_content_digest(source) == contract["candidate"]["content_sha256"]
    route = contract["lanes"]["vl_cache"]["tasks"]["coco_caption"]
    dataset = DatasetConfig(route["dataset_id"], route["dataset_revision"], route["split"])
    for cap in worker.CAPS:
        cfgs = worker.configurations(source, "coco_caption", dataset, cap)
        for cfg in cfgs.values():
            validate_configuration(contract, "vl_cache", "coco_caption", cfg)
    cfg = cfgs["mosaickv_exact"]
    retuned = replace(cfg, graph=replace(cfg.graph, uncertain_head_fraction=1 / 64))
    with pytest.raises(ValueError, match="policy changed"):
        validate_configuration(contract, "vl_cache", "coco_caption", retuned)
    changed_generation = replace(cfg, generation=replace(cfg.generation, max_new_tokens=32))
    with pytest.raises(ValueError, match="controls mismatch"):
        validate_configuration(contract, "vl_cache", "coco_caption", changed_generation)


def test_milebench_revision_is_pinned_and_rejects_another_revision():
    from dataclasses import replace

    from mosaickv.evaluation.shared_byte_contract import load_contract, validate_configuration

    source = SCRIPTS.parents[1]
    contract = load_contract(source / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml")
    cfg = worker.configurations(
        source,
        "milebench",
        DatasetConfig(
            "MileBench/MileBench",
            "53c7a58051ef88bacf76541d91f03f5ba2d71e7d",
            "official_evaluation",
        ),
        40,
    )["mosaickv_exact"]
    validate_configuration(contract, "lookm", "milebench", cfg)
    changed = replace(cfg, dataset=replace(cfg.dataset, revision="a" * 40))
    with pytest.raises(ValueError, match="dataset route mismatch"):
        validate_configuration(contract, "lookm", "milebench", changed)


@pytest.mark.parametrize("task", ["coco_caption", "milebench"])
@pytest.mark.parametrize("cap", [10, 20, 40, 80])
def test_shared_byte_configurations_hold_all_controls(task, cap):
    import math

    from mosaickv.evaluation.tasks import default_task_registry

    route = default_task_registry().resolve(task)
    dataset = DatasetConfig(route.dataset_id, "a" * 40, route.split)
    cfgs = worker.configurations(SCRIPTS.parents[1], task, dataset, cap)
    baseline = "vl_cache_reimpl" if task == "coco_caption" else "lookm_reimpl"
    assert set(cfgs) == {"full_kv", "prompt_attention_topk", baseline, "mosaickv_exact"}
    prompt = cfgs["prompt_attention_topk"]
    for method, cfg in cfgs.items():
        for field in ("model", "execution", "generation", "dataset"):
            assert getattr(cfg, field) == getattr(prompt, field)
        if method != "full_kv":
            assert cfg.cache == prompt.cache
            assert cfg.cache.budget_value == cap * 2**20
            if task == "coco_caption":
                for source_bytes in (8192, 70 * 2**20, 100 * 2**30):
                    assert math.ceil(source_bytes * cfg.cache.retention_ratio) == source_bytes
            assert cfg.cache.retention_ratio < 1.0  # Do not take the identity fast path.
        assert cfg.cache.block_size == 1
        assert not cfg.residual.enabled and not cfg.repair.enabled
        assert cfg.generation.max_new_tokens == 64 and not cfg.generation.do_sample
    assert cfgs["mosaickv_exact"].graph.uncertain_head_fraction == 1 / 256
    assert cfgs["full_kv"].cache.retention_ratio == 1
    assert prompt.dataset == dataset
    with pytest.raises(ValueError, match="dataset identity"):
        control_configs(
            prompt,
            task=task,
            seed=0,
            output_length=64,
            dataset=DatasetConfig("wrong/dataset", "a" * 40, "train"),
        )


@pytest.mark.parametrize("task", ["coco_caption", "milebench"])
def test_worker_exports_metrics_without_fabricating_cider_sample_score(tmp_path, monkeypatch, task):
    import pandas as pd

    from mosaickv.config import synthetic_smoke_config
    from mosaickv.evaluation import worker_support

    def fake_evaluate(args, config, adapter):
        assert args.metric_output and args.metric_observations_parquet_output
        assert args.metric_aggregates_parquet_output
        if task == "milebench":
            assert args.unified_milebench and args.sample_id_file is None
            assert args.paper_native_ledger_sha == "a" * 64
        else:
            assert args.sample_id_file_sha == "a" * 64
        path = tmp_path / "raw.parquet"
        pd.DataFrame(
            [
                dict(
                    sample_id="synthetic:1",
                    answer="caption",
                    reference="caption",
                    task_score=None,
                    active_kv_bytes=1024,
                    residual_kv_bytes=0,
                    compression_time=0.1,
                    decode_time=0.2,
                    end_to_end_time=0.3,
                    generated_tokens=64,
                    repair_count=0,
                    status="completed",
                    error=None,
                    manifest_path="synthetic-manifest.json",
                )
            ]
        ).to_parquet(path)
        return 0, dict(
            failed_samples=0,
            parquet_output=str(path),
            metric_observations_parquet_output=args.metric_observations_parquet_output,
            metric_aggregates_parquet_output=args.metric_aggregates_parquet_output,
        )

    monkeypatch.setattr(worker_support, "run_hf_evaluation", fake_evaluate)
    record = worker_support.record_hf_run(
        synthetic_smoke_config(),
        object(),
        task=task,
        run_id="fixture",
        output_root=tmp_path,
        ledger_path=tmp_path / "ledger.json",
        ledger_sha="a" * 64,
        sample_ids=("synthetic:1",),
        phase="synthetic_fixture",
        repetition=0,
        order=0,
        export_metrics=True,
        materialized_milebench=task == "milebench",
    )
    assert record["rows"][0]["task_score"] is None
    assert record["metric_aggregates_parquet_output"].endswith("metrics.parquet")


def test_worker_refuses_failed_metric_even_when_generation_succeeded(tmp_path, monkeypatch):
    from mosaickv.config import synthetic_smoke_config
    from mosaickv.evaluation import worker_support

    monkeypatch.setattr(
        worker_support,
        "run_hf_evaluation",
        lambda *a, **k: (0, {"failed_samples": 0, "metric_errors": ["CIDEr unavailable"]}),
    )
    with pytest.raises(RuntimeError, match="CIDEr unavailable"):
        worker_support.record_hf_run(
            synthetic_smoke_config(),
            object(),
            task="coco_caption",
            run_id="fixture",
            output_root=tmp_path,
            ledger_path=tmp_path / "ledger.json",
            ledger_sha="a" * 64,
            sample_ids=("synthetic:1",),
            phase="synthetic_fixture",
            repetition=0,
            order=0,
            export_metrics=True,
        )
