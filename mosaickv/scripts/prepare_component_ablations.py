#!/usr/bin/env python3
"""Materialize the bounded Experiment-3 ablation study as immutable task jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.config import (
    AblationConfig,
    FidelityDiagnosticsConfig,
    ForecastingConfig,
    canonical_config_json,
    config_sha256,
)
from mosaickv.experiment_matrix import TASK_CATALOG, _base_config
from mosaickv.manifest import sha256_bytes
from mosaickv.types import AblationVariant, ForecastMode, MosaicKVMethod, RepairPolicy

EXPECTED_VARIANTS = tuple("ABCDEFGHIJKL")
METHOD_BY_VARIANT = {
    **{variant: MosaicKVMethod.MOSAICKV_FULL for variant in "ABCDEFGHKL"},
    "I": MosaicKVMethod.MOSAICKV_EXACT,
    "J": MosaicKVMethod.MOSAICKV_PROTO,
}


def _write_once(path: Path, payload: bytes, mode: int = 0o444) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable ablation artifact: {path}")
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
        raise ValueError("ablation study must be a schema-version-1 YAML object")
    variants = value.get("variants")
    if not isinstance(variants, list):
        raise ValueError("ablation study variants must be a list")
    identifiers = tuple(str(item.get("id")) for item in variants if isinstance(item, dict))
    if identifiers != EXPECTED_VARIANTS:
        raise ValueError(f"ablation variants must be ordered {EXPECTED_VARIANTS}")
    if value.get("primary_budget") != "packed_prefill_active_kv_bytes":
        raise ValueError("Experiment 3 requires a physical active-KV-byte budget")
    subset_sizes = value.get("subset_size_by_task")
    task_keys = value.get("tasks")
    if not isinstance(subset_sizes, dict) or not isinstance(task_keys, list):
        raise ValueError("ablation study must declare subset_size_by_task for every task")
    if set(subset_sizes) != set(task_keys):
        raise ValueError("subset_size_by_task keys must match the configured task list")
    if any(not isinstance(size, int) or size < 1 for size in subset_sizes.values()):
        raise ValueError("every component-ablation subset size must be a positive integer")
    return cast("dict[str, Any]", value)


def _ledger(project_root: Path, task_key: str) -> tuple[tuple[str, ...], str]:
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
        raise ValueError(f"ablation ledger identity mismatch: {path}")
    sample_ids = tuple(str(item) for item in value.get("sample_ids", ()))
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"ablation ledger must contain unique sample IDs: {path}")
    return sample_ids, sha256_bytes(payload)


def _variant_config(
    variant_id: str,
    variant_name: str,
    *,
    task_key: str,
    ratio: float,
    block_size: int,
    output_length: int,
    seed: int,
    oracle_steps: int,
    prototype_group_size: int,
    prototype_max_position_span: int,
    minimum_stratum_fraction: float,
    graph_max_neighbors: int,
    sample_ids: tuple[str, ...],
    ledger_sha: str,
) -> Any:
    method = METHOD_BY_VARIANT[variant_id]
    config = _base_config(
        model_key="qwen2_5_vl_3b",
        backend_key="hf_eager",
        method_name=method.value,
        task_key=task_key,
        retention_ratio=ratio,
        output_length=output_length,
        seed=seed,
    )
    ablation = AblationConfig(
        variant=AblationVariant(variant_name),
        evaluation_only=variant_id in {"K", "L"},
    )
    config = replace(
        config,
        cache=replace(config.cache, block_size=block_size),
        # Prototype-risk and repair require measured eager attention. Keep the
        # backend identical for every component-ablation row.
        execution=replace(config.execution, decode_attention_implementation="eager"),
        diagnostics=FidelityDiagnosticsConfig(
            enabled=True,
            sample_ids=sample_ids,
            sample_ledger_sha=ledger_sha,
            forecast_oracle_steps=oracle_steps,
        ),
        selection=replace(
            config.selection,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=minimum_stratum_fraction,
        ),
        prototypes=replace(
            config.prototypes,
            group_size=prototype_group_size,
            max_position_span=prototype_max_position_span,
        ),
        graph=replace(
            config.graph,
            max_neighbors=graph_max_neighbors,
            fallback_max_position_span=prototype_max_position_span,
        ),
    )
    if variant_id == "B":
        config = replace(
            config,
            forecasting=ForecastingConfig(
                enabled=False,
                mode=ForecastMode.PROMPT_WINDOW,
                prompt_window=16,
                draft_steps=0,
                centroid_count=4,
            ),
            ablation=ablation,
        )
    elif variant_id == "C":
        config = replace(
            config,
            forecasting=replace(
                config.forecasting,
                mode=ForecastMode.PROMPT_WINDOW,
                prompt_window=16,
                draft_steps=0,
            ),
        )
    elif variant_id == "D":
        config = replace(
            config,
            forecasting=replace(
                config.forecasting,
                mode=ForecastMode.DRAFT_ROLLOUT,
                prompt_window=0,
                draft_steps=oracle_steps,
            ),
        )
    elif variant_id == "E":
        config = replace(config, graph=replace(config.graph, enabled=False))
    elif variant_id == "F":
        config = replace(
            config,
            graph=replace(
                config.graph,
                attention_weight=0.0,
                spatial_weight=0.0,
                layout_weight=0.0,
                temporal_weight=0.0,
                same_region_weight=0.0,
                cross_modal_weight=0.0,
                fallback_weight=0.0,
            ),
        )
    elif variant_id == "G":
        config = replace(config, utility=replace(config.utility, lambda_v=0.0))
    elif variant_id == "H":
        config = replace(
            config,
            utility=replace(config.utility, lambda_v=0.0, lambda_o=0.0),
            selection=replace(config.selection, lambda_g=0.0, lambda_m=0.0),
        )
    elif variant_id == "K":
        config = replace(
            config,
            forecasting=replace(
                config.forecasting,
                mode=ForecastMode.DRAFT_ROLLOUT,
                prompt_window=0,
                draft_steps=oracle_steps,
            ),
        )
    elif variant_id == "L":
        config = replace(
            config,
            repair=replace(
                config.repair,
                policy=RepairPolicy.ORACLE,
                evaluation_only=True,
            ),
        )
    if variant_id != "B":
        config = replace(config, ablation=ablation)
    return config


def materialize(spec_path: Path, destination: Path, output_root: Path) -> Path:
    spec = _load_spec(spec_path)
    spec_sha = sha256_bytes(spec_path.read_bytes())
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("spec_sha") != spec_sha:
            raise RuntimeError("existing ablation expansion has a different specification")
        return destination / "jobs.jsonl"

    project_root = spec_path.resolve().parents[2]
    destination.mkdir(parents=True, exist_ok=False)
    _write_once(destination / "study.yaml", spec_path.read_bytes())
    variants = cast("list[dict[str, Any]]", spec["variants"])
    jobs: list[dict[str, Any]] = []
    config_count = 0
    for array_index, task_key in enumerate(cast("list[str]", spec["tasks"])):
        sample_ids, ledger_sha = _ledger(project_root, task_key)
        expected_size = int(cast("dict[str, int]", spec["subset_size_by_task"])[task_key])
        if len(sample_ids) != expected_size:
            raise ValueError(
                f"ablation ledger for {task_key} contains {len(sample_ids)} samples; "
                f"the study requires {expected_size}"
            )
        cells: list[dict[str, Any]] = []
        unsupported: list[dict[str, str]] = []
        for variant in variants:
            identifier = str(variant["id"])
            name = str(variant["name"])
            reason = variant.get("unsupported_reason")
            if reason is not None:
                unsupported.append(
                    {"variant_id": identifier, "variant": name, "reason": str(reason)}
                )
                continue
            config = _variant_config(
                identifier,
                name,
                task_key=task_key,
                ratio=float(spec["retention_ratio"]),
                block_size=int(spec["block_size"]),
                output_length=int(spec["output_length"]),
                seed=int(spec["seed"]),
                oracle_steps=int(spec["forecast_oracle_steps"]),
                prototype_group_size=int(spec["prototype_group_size"]),
                prototype_max_position_span=int(spec["prototype_max_position_span"]),
                minimum_stratum_fraction=float(spec["minimum_stratum_fraction"]),
                graph_max_neighbors=int(spec["graph_max_neighbors"]),
                sample_ids=sample_ids,
                ledger_sha=ledger_sha,
            )
            digest = config_sha256(config)
            path = destination / "configs" / f"{config_count:04d}-{digest}.yaml"
            payload = cast("dict[str, object]", json.loads(canonical_config_json(config)))
            _write_once(path, yaml.safe_dump(payload, sort_keys=True).encode())
            cells.append(
                {
                    "variant_id": identifier,
                    "variant": name,
                    "config_path": str(path.resolve()),
                    "config_sha": digest,
                    "run_id": (f"exp3-{spec_sha[:10]}-{task_key}-{identifier.lower()}-{name}"),
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
                "subset_size": len(sample_ids),
                "output_root": str(output_root.resolve()),
                "trace_directory": str((output_root / "traces").resolve()),
                "cells": cells,
                "unsupported": unsupported,
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
        default=project_root / "configs" / "experiments" / "component_ablations_small.yaml",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/component-ablations-matrix-v1"),
    )
    parser.add_argument(
        "--run-output-root",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/component-ablations-v1"),
    )
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
