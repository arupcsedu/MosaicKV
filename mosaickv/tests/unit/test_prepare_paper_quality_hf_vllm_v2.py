from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml


def _module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "prepare_paper_quality_hf_vllm_v2.py"
    spec = importlib.util.spec_from_file_location("paper_quality_hf_vllm_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fixture(tmp_path: Path, *, authorized: bool = True) -> tuple[Path, dict[str, Path], Path]:
    package = tmp_path / "mosaickv"
    protocol_path = package / "configs" / "experiments" / "protocol.yaml"
    candidate_path = package / "configs" / "candidate.yaml"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("schema_version: 1\n", encoding="utf-8")
    source_sha = "a" * 40
    tasks = {
        "mmstar": {
            "dataset_id": "Lin-Chen/MMStar",
            "dataset_revision": "mm-rev",
            "split": "val",
        },
        "textvqa": {
            "dataset_id": "lmms-lab/textvqa",
            "dataset_revision": "text-rev",
            "split": "validation",
        },
        "docvqa": {
            "dataset_id": "lmms-lab/DocVQA",
            "dataset_revision": "doc-rev",
            "split": "validation",
        },
    }
    protocol = {
        "schema_version": 1,
        "experiment_id": "paper_quality_hf_vllm_llava15_v2",
        "classification": "fixed_preregistered_paper_quality_memory",
        "execution_authorized": authorized,
        "candidate": {
            "frozen_algorithm_commit": source_sha,
            "config": "configs/candidate.yaml",
            "config_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        },
        "experiment": {
            "tasks": tasks,
            "sample_count_per_task": 4,
            "shards_per_task": 2,
            "samples_per_shard": 2,
            "selection_seed": 11,
            "output_length": 8,
            "precision": "fp16",
            "backends": {
                "hf_eager": {"methods": ["full_kv", "prompt_attention_topk"]},
                "vllm": {"methods": ["full_kv", "mosaickv_exact"]},
            },
        },
    }
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    ledgers: dict[str, Path] = {}
    for task, contract in tasks.items():
        path = tmp_path / "ledgers" / f"{task}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical(
            {
                "schema_version": 1,
                "task": task,
                **contract,
                "classification": "fixed_preregistered_paper_quality_memory",
                "canonical_eligible": True,
                "source_git_sha": source_sha,
                "source_git_dirty": False,
                "selection_seed": 11,
                "selection_freeze_sha256": protocol_sha,
                "sample_count": 4,
                "sample_ids": [f"{task}-{index}" for index in range(4)],
            }
        )
        path.write_bytes(payload)
        path.chmod(0o444)
        ledgers[task] = path
    return protocol_path, ledgers, tmp_path / "repository"


def test_preparation_produces_twelve_shards_in_real_protocol_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    protocol_path, ledgers, repository = _fixture(tmp_path)
    monkeypatch.setattr(module, "_source_snapshot", lambda _repository: ("a" * 40, False))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    destination = tmp_path / "campaign"
    index = module.prepare_campaign(
        protocol_path=protocol_path,
        ledger_paths=ledgers,
        destination=destination,
        repository=repository,
        output_root=tmp_path / "runs",
    )
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    assert len(rows) == 12  # 2 backends * 3 tasks * 2 shards in this fixture.
    assert {(row["backend"], row["task"]) for row in rows} == {
        (backend, task)
        for backend in ("hf_eager", "vllm")
        for task in ("mmstar", "textvqa", "docvqa")
    }
    assert all(row["sample_count"] == 2 for row in rows)
    assert len(list((destination / "sample_shards").glob("*/*.json"))) == 6
    assert all(not (path.stat().st_mode & 0o222) for path in destination.rglob("*.json"))


def test_validation_rejects_a_mutated_sample_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    protocol_path, ledgers, repository = _fixture(tmp_path)
    monkeypatch.setattr(module, "_source_snapshot", lambda _repository: ("a" * 40, False))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    index = module.prepare_campaign(
        protocol_path=protocol_path,
        ledger_paths=ledgers,
        destination=tmp_path / "campaign",
        repository=repository,
        output_root=tmp_path / "runs",
    )
    row = json.loads(index.read_text().splitlines()[0])
    shard = Path(row["sample_ledger_path"])
    shard.chmod(0o644)
    shard.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample-shard ledger hash changed"):
        module.validate_job(index, 0, repository)


def test_disabled_protocol_cannot_prepare_or_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    protocol_path, ledgers, repository = _fixture(tmp_path, authorized=False)
    monkeypatch.setattr(module, "_source_snapshot", lambda _repository: ("a" * 40, False))
    with pytest.raises(PermissionError, match="has not been authorized"):
        module.prepare_campaign(
            protocol_path=protocol_path,
            ledger_paths=ledgers,
            destination=tmp_path / "campaign",
            repository=repository,
            output_root=tmp_path / "runs",
        )


def test_checked_in_protocol_has_paper_scale_and_explicit_authorization() -> None:
    module = _module()
    protocol_path = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "paper_quality_hf_vllm_llava15_v2.yaml"
    )
    protocol = module._load_protocol(
        protocol_path,
        require_execution_authorized=False,
    )
    experiment = protocol["experiment"]
    assert protocol["execution_authorized"] is True
    assert experiment["sample_count_per_task"] == 256
    assert experiment["shards_per_task"] == 4
    assert experiment["samples_per_shard"] == 64
    assert protocol["execution_plan"]["maximum_concurrent_a100_jobs"] == 4


def test_checked_in_continuation_reuses_prefix_and_caps_concurrency() -> None:
    module = _module()
    protocol_path = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "paper_quality_hf_vllm_llava15_v3_continuation.yaml"
    )
    protocol = module._load_protocol(
        protocol_path,
        require_execution_authorized=True,
    )
    experiment = protocol["experiment"]
    reused = protocol["reuse"]["reused_samples_per_task"]
    for task in ("mmstar", "textvqa", "docvqa"):
        assert experiment["target_sample_counts"][task] == (
            reused + experiment["continuation_sample_counts"][task]
        )
    assert sum(experiment["continuation_sample_counts"].values()) == 8232
    assert protocol["execution_plan"]["continuation_backend_jobs"] == 260
    assert protocol["execution_plan"]["maximum_concurrent_a100_jobs"] == 6
