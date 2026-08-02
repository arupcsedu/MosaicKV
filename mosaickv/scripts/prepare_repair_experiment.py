#!/usr/bin/env python3
"""Materialize the bounded Experiment-4 repair study as immutable task jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from mosaickv.config import (
    FidelityDiagnosticsConfig,
    RepairConfig,
    ResidualConfig,
    canonical_config_json,
    config_sha256,
)
from mosaickv.experiment_matrix import TASK_CATALOG, _base_config
from mosaickv.manifest import sha256_bytes
from mosaickv.types import (
    MosaicKVMethod,
    RepairCandidateRanking,
    RepairPolicy,
    ResidualStorageDType,
)


def _write_once(path: Path, payload: bytes, mode: int = 0o444) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable repair artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load_spec(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("repair study must be a schema-version-1 YAML object")
    tasks = value.get("tasks")
    sizes = value.get("subset_size_by_task")
    variants = value.get("variants")
    if not isinstance(tasks, list) or not isinstance(sizes, dict) or set(tasks) != set(sizes):
        raise ValueError("repair study tasks and subset_size_by_task must match")
    if not isinstance(variants, list) or not variants:
        raise ValueError("repair study variants must be a non-empty list")
    identifiers = [str(item.get("id")) for item in variants if isinstance(item, dict)]
    if len(identifiers) != len(variants) or len(set(identifiers)) != len(identifiers):
        raise ValueError("repair variant IDs must be present and unique")
    required_policies = {
        "none",
        "entropy",
        "prototype_risk",
        "entropy_or_prototype_risk",
        "oracle",
    }
    observed = {str(item.get("policy")) for item in variants if isinstance(item, dict)}
    if not required_policies <= observed:
        raise ValueError("repair variants do not cover every required policy")
    if value.get("primary_budget") != "packed_prefill_active_kv_bytes":
        raise ValueError("Experiment 4 requires the physical active-KV-byte budget")
    return cast("dict[str, Any]", value)


def _ledger(project_root: Path, task_key: str) -> tuple[Path, tuple[str, ...], str]:
    path = project_root / "configs" / "ledgers" / f"ablation_{task_key}.json"
    payload = path.read_bytes()
    value = json.loads(payload)
    task = TASK_CATALOG[task_key]
    expected = {
        "dataset_id": task.dataset_id,
        "dataset_revision": task.revision,
        "split": task.split,
        "task": task.task_name,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"repair ledger identity mismatch: {path}")
    sample_ids = tuple(str(item) for item in value.get("sample_ids", ()))
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"repair ledger must contain unique sample IDs: {path}")
    return path, sample_ids, sha256_bytes(payload)


def _cell_config(spec: dict[str, Any], variant: dict[str, Any], task_key: str) -> Any:
    is_reference = variant.get("kind") == "full_kv"
    method = MosaicKVMethod.FULL_KV if is_reference else MosaicKVMethod.MOSAICKV_FULL
    config = _base_config(
        model_key="qwen2_5_vl_3b",
        backend_key="hf_eager",
        method_name=method.value,
        task_key=task_key,
        retention_ratio=1.0 if is_reference else float(spec["retention_ratio"]),
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
    )
    if is_reference:
        return replace(config, diagnostics=FidelityDiagnosticsConfig(enabled=False))
    base = cast("dict[str, Any]", spec["base_repair"])
    policy = RepairPolicy(str(variant.get("policy", "none")))
    storage_dtype = ResidualStorageDType(str(variant.get("storage_dtype", base["storage_dtype"])))
    placement = str(variant.get("placement", base["placement"]))
    if placement not in {"cpu_pinned", "cpu_pageable"}:
        raise ValueError(f"unsupported residual placement: {placement}")
    preserve_partial = variant.get(
        "preserve_partial_prototype",
        base.get("preserve_partial_prototype", True),
    )
    if not isinstance(preserve_partial, bool):
        raise ValueError("preserve_partial_prototype must be boolean")
    return replace(
        config,
        cache=replace(config.cache, block_size=int(spec["block_size"])),
        execution=replace(config.execution, decode_attention_implementation="eager"),
        diagnostics=FidelityDiagnosticsConfig(enabled=False),
        selection=replace(
            config.selection,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=float(spec["minimum_stratum_fraction"]),
        ),
        prototypes=replace(
            config.prototypes,
            group_size=int(spec["prototype_group_size"]),
            max_position_span=int(spec["prototype_max_position_span"]),
        ),
        graph=replace(
            config.graph,
            max_neighbors=int(spec["graph_max_neighbors"]),
            fallback_max_position_span=int(spec["prototype_max_position_span"]),
        ),
        residual=ResidualConfig(
            enabled=True,
            storage_dtype=storage_dtype,
            require_pinned_memory=placement == "cpu_pinned",
        ),
        repair=RepairConfig(
            enabled=policy is not RepairPolicy.NONE,
            policy=policy,
            entropy_threshold=float(variant.get("entropy_threshold", base["entropy_threshold"])),
            prototype_risk_threshold=float(
                variant.get("prototype_risk_threshold", base["prototype_risk_threshold"])
            ),
            prototype_risk_min_entropy=float(
                variant.get(
                    "prototype_risk_min_entropy",
                    base.get("prototype_risk_min_entropy", 0.0),
                )
            ),
            max_blocks_per_step=(
                0
                if policy is RepairPolicy.NONE
                else int(variant.get("max_blocks_per_step", base["max_blocks_per_step"]))
            ),
            candidate_ranking=RepairCandidateRanking(
                str(
                    variant.get(
                        "candidate_ranking",
                        base.get(
                            "candidate_ranking",
                            RepairCandidateRanking.RISK_WEIGHTED_RECONSTRUCTION_ERROR.value,
                        ),
                    )
                )
            ),
            preserve_partial_prototype=preserve_partial,
            evaluation_only=bool(variant.get("evaluation_only", False)),
        ),
    )


def materialize(spec_path: Path, destination: Path, output_root: Path) -> Path:
    spec = _load_spec(spec_path)
    spec_sha = sha256_bytes(spec_path.read_bytes())
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("spec_sha") != spec_sha:
            raise RuntimeError("existing repair expansion has a different specification")
        return destination / "jobs.jsonl"
    project_root = spec_path.resolve().parents[2]
    destination.mkdir(parents=True, exist_ok=False)
    _write_once(destination / "study.yaml", spec_path.read_bytes())
    jobs: list[dict[str, Any]] = []
    config_count = 0
    for array_index, task_key in enumerate(cast("list[str]", spec["tasks"])):
        ledger_path, sample_ids, ledger_sha = _ledger(project_root, task_key)
        expected_size = int(cast("dict[str, int]", spec["subset_size_by_task"])[task_key])
        if len(sample_ids) != expected_size:
            raise ValueError(
                f"repair ledger for {task_key} has {len(sample_ids)} samples, expected "
                f"{expected_size}"
            )
        cells: list[dict[str, Any]] = []
        for raw_variant in cast("list[dict[str, Any]]", spec["variants"]):
            variant = cast("dict[str, Any]", raw_variant)
            config = _cell_config(spec, variant, task_key)
            digest = config_sha256(config)
            path = destination / "configs" / f"{config_count:04d}-{digest}.yaml"
            payload = cast("dict[str, object]", json.loads(canonical_config_json(config)))
            _write_once(path, yaml.safe_dump(payload, sort_keys=True).encode())
            identifier = str(variant["id"])
            name = str(variant["name"])
            cells.append(
                {
                    "variant_id": identifier,
                    "variant": name,
                    "config_path": str(path.resolve()),
                    "config_sha": digest,
                    "run_id": f"exp4-{spec_sha[:10]}-{task_key}-{identifier.lower()}-{name}",
                    "policy": (
                        "full_kv" if config.method.is_full_cache else config.repair.policy.value
                    ),
                    "entropy_threshold": config.repair.entropy_threshold,
                    "prototype_risk_threshold": config.repair.prototype_risk_threshold,
                    "prototype_risk_min_entropy": (config.repair.prototype_risk_min_entropy),
                    "max_blocks_per_step": config.repair.max_blocks_per_step,
                    "candidate_ranking": config.repair.candidate_ranking.value,
                    "preserve_partial_prototype": (config.repair.preserve_partial_prototype),
                    "storage_dtype": config.residual.storage_dtype.value,
                    "placement": (
                        "not_applicable"
                        if config.method.is_full_cache
                        else (
                            "cpu_pinned"
                            if config.residual.require_pinned_memory
                            else "cpu_pageable"
                        )
                    ),
                }
            )
            config_count += 1
        task = TASK_CATALOG[task_key]
        jobs.append(
            {
                "array_index": array_index,
                "task_key": task_key,
                "task": task.task_name,
                "dataset_id": task.dataset_id,
                "dataset_revision": task.revision,
                "sample_ids": list(sample_ids),
                "sample_id_file": str(ledger_path.resolve()),
                "sample_id_file_sha": ledger_sha,
                "output_root": str(output_root.resolve()),
                "trace_directory": str((output_root / "traces").resolve()),
                "cells": cells,
            }
        )
    index_payload = b"".join(_json_bytes(row) for row in jobs)
    _write_once(destination / "jobs.jsonl", index_payload)
    _write_once(
        destination / "manifest.json",
        _json_bytes(
            {
                "classification": spec["classification"],
                "config_count": config_count,
                "index_sha": hashlib.sha256(index_payload).hexdigest(),
                "job_count": len(jobs),
                "output_root": str(output_root.resolve()),
                "spec_sha": spec_sha,
            }
        ),
    )
    return destination / "jobs.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--spec",
        type=Path,
        default=project_root / "configs" / "experiments" / "repair_detailed_small.yaml",
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--run-output-root", required=True, type=Path)
    args = parser.parse_args()
    print(
        materialize(
            args.spec.resolve(),
            args.output_directory.resolve(),
            args.run_output_root.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
