from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml


def _module(monkeypatch: pytest.MonkeyPatch) -> Any:
    scripts = Path(__file__).parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "prepare_paper_quality_hf_vllm_v3_continuation.py"
    spec = importlib.util.spec_from_file_location("paper_quality_hf_vllm_v3", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path], Path]:
    package = tmp_path / "mosaickv"
    protocol_path = package / "configs/experiments/protocol.yaml"
    candidate_path = package / "configs/candidate.yaml"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("schema_version: 1\n", encoding="utf-8")
    prior_root = tmp_path / "prior"
    (prior_root / "campaign_inputs").mkdir(parents=True)
    (prior_root / "analysis").mkdir(parents=True)
    (prior_root / "master_ledgers").mkdir(parents=True)
    prior_index = prior_root / "campaign_inputs/jobs.jsonl"
    prior_report = prior_root / "analysis/report.json"
    prior_rows = prior_root / "analysis/paired_rows.parquet"
    prior_index.write_text("{}\n", encoding="utf-8")
    prior_report.write_text("{}\n", encoding="utf-8")
    pd.DataFrame.from_records([{"status": "completed"}] * 4608).to_parquet(prior_rows)
    tasks = {
        "mmstar": {"dataset_id": "Lin-Chen/MMStar", "dataset_revision": "m", "split": "val"},
        "textvqa": {
            "dataset_id": "lmms-lab/textvqa",
            "dataset_revision": "t",
            "split": "validation",
        },
        "docvqa": {
            "dataset_id": "lmms-lab/DocVQA",
            "dataset_revision": "d",
            "split": "validation",
        },
    }
    prior_hashes: dict[str, str] = {}
    for task, contract in tasks.items():
        path = prior_root / "master_ledgers" / f"{task}.json"
        path.write_bytes(
            _canonical(
                {
                    "task": task,
                    **contract,
                    "sample_count": 2,
                    "sample_ids": [f"{task}-0", f"{task}-1"],
                }
            )
        )
        path.chmod(0o444)
        prior_hashes[task] = _sha(path)
    protocol = {
        "schema_version": 1,
        "experiment_id": "paper_quality_hf_vllm_llava15_v3_continuation",
        "classification": "fixed_preregistered_paper_quality_memory_continuation",
        "execution_authorized": True,
        "candidate": {
            "frozen_algorithm_commit": "a" * 40,
            "config": "configs/candidate.yaml",
            "config_sha256": _sha(candidate_path),
        },
        "reuse": {
            "prior_campaign_root": str(prior_root),
            "prior_index_sha256": _sha(prior_index),
            "prior_report_sha256": _sha(prior_report),
            "prior_paired_rows_sha256": _sha(prior_rows),
            "prior_master_ledger_sha256": prior_hashes,
            "reused_samples_per_task": 2,
        },
        "experiment": {
            "tasks": tasks,
            "target_sample_counts": {task: 4 for task in tasks},
            "continuation_sample_counts": {task: 2 for task in tasks},
            "samples_per_continuation_shard": 1,
            "selection_seed": 11,
            "output_length": 8,
            "precision": "fp16",
            "backends": {
                "hf_eager": {"methods": ["full_kv", "prompt_attention_topk", "mosaickv_full"]},
                "vllm": {"methods": ["full_kv", "prompt_attention_topk", "mosaickv_exact"]},
            },
        },
        "execution_plan": {"continuation_backend_jobs": 12},
    }
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_sha = _sha(protocol_path)
    full_ledgers: dict[str, Path] = {}
    for task, contract in tasks.items():
        path = tmp_path / "full" / f"{task}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            _canonical(
                {
                    "task": task,
                    **contract,
                    "classification": "fixed_preregistered_paper_quality_memory",
                    "canonical_eligible": True,
                    "source_git_sha": "a" * 40,
                    "source_git_dirty": False,
                    "selection_freeze_sha256": protocol_sha,
                    "sample_count": 4,
                    "sample_ids": [f"{task}-{index}" for index in range(4)],
                }
            )
        )
        path.chmod(0o444)
        full_ledgers[task] = path
    return protocol_path, full_ledgers, tmp_path / "repository"


def test_continuation_reuses_exact_prefix_and_writes_only_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module(monkeypatch)
    protocol, ledgers, repository = _fixture(tmp_path)
    monkeypatch.setattr(module, "_source_snapshot", lambda _repository: ("a" * 40, False))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    index = module.prepare_continuation(
        protocol_path=protocol,
        full_ledger_paths=ledgers,
        destination=tmp_path / "campaign",
        output_root=tmp_path / "runs",
        repository=repository,
    )
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    assert len(rows) == 12
    assert all(row["sample_count"] == 1 for row in rows)
    for row in rows:
        shard = json.loads(Path(row["sample_ledger_path"]).read_text())
        assert shard["sample_ids"][0] in {f"{row['task']}-2", f"{row['task']}-3"}
        assert shard["sample_ids"][0] not in {f"{row['task']}-0", f"{row['task']}-1"}


def test_continuation_rejects_a_changed_reused_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module(monkeypatch)
    protocol, ledgers, repository = _fixture(tmp_path)
    changed = ledgers["textvqa"]
    value = json.loads(changed.read_text())
    value["sample_ids"][0] = "different"
    changed.chmod(0o644)
    changed.write_bytes(_canonical(value))
    changed.chmod(0o444)
    monkeypatch.setattr(module, "_source_snapshot", lambda _repository: ("a" * 40, False))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    with pytest.raises(ValueError, match="does not exactly reuse"):
        module.prepare_continuation(
            protocol_path=protocol,
            full_ledger_paths=ledgers,
            destination=tmp_path / "campaign",
            output_root=tmp_path / "runs",
            repository=repository,
        )
