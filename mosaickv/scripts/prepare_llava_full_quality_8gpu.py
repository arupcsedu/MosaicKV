#!/usr/bin/env python3
"""Materialize the immutable eight-GPU LLaVA full-quality campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.config import (
    FidelityDiagnosticsConfig,
    canonical_config_json,
    config_sha256,
)
from mosaickv.experiment_matrix import TASK_CATALOG, _base_config
from mosaickv.types import BudgetUnit, RepairPolicy


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable campaign artifact: {path}")
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


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_provenance() -> tuple[str, bool, bytes]:
    repository = _repository()
    git_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    tracked = subprocess.run(
        ["git", "-C", str(repository), "diff", "--binary", "HEAD", "--", "."],
        check=True,
        capture_output=True,
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    patches = [tracked]
    for raw_path in untracked_raw.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode()
        result = subprocess.run(
            ["git", "-C", str(repository), "diff", "--binary", "--no-index", "/dev/null", path],
            check=False,
            capture_output=True,
        )
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"could not preserve untracked source in patch: {path}")
        patches.append(result.stdout)
    return git_sha, dirty, b"".join(patches)


def _load_spec(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("campaign spec must be a schema-version-1 mapping")
    if int(value.get("worker_count", 0)) != 8 or int(value.get("workers_per_task", 0)) != 2:
        raise ValueError("this launcher requires eight workers and two workers per task")
    tasks = value.get("task_keys")
    methods = value.get("methods")
    if not isinstance(tasks, list) or len(tasks) != 4 or len(set(tasks)) != 4:
        raise ValueError("campaign requires four distinct image tasks")
    if not isinstance(methods, list) or len(methods) != 9 or len(set(methods)) != 9:
        raise ValueError("campaign requires all nine declared methods")
    return cast("dict[str, Any]", value)


def _load_frozen_profile(spec: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    profile_spec = cast("dict[str, Any]", spec["frozen_mosaickv_profile"])
    path = Path(str(profile_spec["path"])).resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    expected = str(profile_spec["sha256"])
    if profile.get("profile_sha256") != expected:
        raise ValueError("frozen MosaicKV profile identity does not match campaign spec")
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("frozen MosaicKV profile has no parameter mapping")
    return cast("dict[str, Any]", profile), path, _sha256(path.read_bytes())


def _source_ledger(path: Path, task_key: str) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    task = TASK_CATALOG[task_key]
    expected = {
        "schema_version": 1,
        "task": task.task_name,
        "dataset_id": task.dataset_id,
        "dataset_revision": task.revision,
        "split": task.split,
    }
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"source sample ledger identity mismatch: {path}")
    ids = value.get("sample_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError(f"source sample ledger IDs are invalid: {path}")
    return cast("dict[str, Any]", value), _sha256(payload)


def _diagnostics(sample_ids: tuple[str, ...], ledger_sha: str, *, mosaic: bool) -> Any:
    return FidelityDiagnosticsConfig(
        enabled=True,
        sample_ids=sample_ids,
        sample_ledger_sha=ledger_sha,
        forecast_oracle_steps=4 if mosaic else 0,
    )


def _common_budget(config: Any, method: str) -> Any:
    block_size = 1 if method in {"lookm_reimpl", "prefixkv_reimpl", "vl_cache_reimpl"} else 16
    return replace(
        config,
        cache=replace(
            config.cache,
            budget_unit=BudgetUnit.BYTES,
            block_size=block_size,
            physical_budget_block_size=16,
            physical_budget_mandatory_policy="last_token",
            exact_gather_mode="batched_layer",
            packing_mode="per_head_reference",
        ),
    )


def _resolved_config(
    *,
    spec: dict[str, Any],
    profile: dict[str, Any],
    method: str,
    task_key: str,
    ratio: float,
    seed: int,
    diagnostic_ids: tuple[str, ...],
    ledger_sha: str,
) -> Any:
    config = _base_config(
        model_key=str(spec["model_key"]),
        backend_key=str(spec["backend_key"]),
        method_name=method,
        task_key=task_key,
        retention_ratio=ratio,
        output_length=int(spec["output_length"]),
        seed=seed,
    )
    config = _common_budget(config, method)
    is_mosaic = method.startswith("mosaickv_")
    config = replace(config, diagnostics=_diagnostics(diagnostic_ids, ledger_sha, mosaic=is_mosaic))
    if not is_mosaic:
        return config

    parameters = cast("dict[str, Any]", profile["parameters"])
    routing = dict(cast("dict[str, Any]", parameters["routing"]))
    draft_steps = int(routing.pop("draft_steps"))
    prompt_window = int(routing.pop("prompt_window"))
    repair_values = dict(cast("dict[str, Any]", parameters["repair"]))
    repair_policy = RepairPolicy(str(repair_values.pop("policy")))
    if repair_policy is not RepairPolicy.NONE:
        raise ValueError("campaign spec expects the frozen repair-rejection policy")
    return replace(
        config,
        forecasting=replace(
            config.forecasting,
            draft_steps=draft_steps,
            prompt_window=prompt_window,
            adaptive_head_routing=True,
            centroid_execution_mode="batched_layer",
            **routing,
        ),
        graph=replace(
            config.graph,
            attention_host_transfer_mode="deferred_layer_vectorized",
            edge_accumulation_mode="preselected",
            **cast("dict[str, Any]", parameters["graph"]),
        ),
        utility=replace(
            config.utility,
            execution_mode="vectorized_edges",
            **cast("dict[str, Any]", parameters["utility"]),
        ),
        selection=replace(
            config.selection,
            stop_on_nonpositive_gain=False,
            **cast("dict[str, Any]", parameters["selection"]),
        ),
        prototypes=replace(
            config.prototypes,
            source_gather_mode="per_head_reference",
            **cast("dict[str, Any]", parameters["prototypes"]),
        ),
        residual=replace(config.residual, index_mode="compact"),
        repair=replace(config.repair, policy=repair_policy, **repair_values),
    )


def _cell_weight(method: str, ratio: float) -> float:
    if ratio == 1.0:
        return 1.0
    return {
        "random_kv": 3.0,
        "prompt_attention_topk": 3.2,
        "lookm_reimpl": 9.5,
        "prefixkv_reimpl": 5.5,
        "vl_cache_reimpl": 5.5,
        "mosaickv_exact": 5.0,
        "mosaickv_proto": 6.0,
        "mosaickv_full": 6.5,
    }[method]


def _balance(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bins: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
    totals = [0.0, 0.0]
    for cell in sorted(cells, key=lambda item: (-float(item["estimated_weight"]), item["run_id"])):
        target = 0 if totals[0] <= totals[1] else 1
        bins[target].append(cell)
        totals[target] += float(cell["estimated_weight"])
    for bucket in bins:
        bucket.sort(
            key=lambda item: (
                float(item["retention_ratio"]) != 1.0,
                str(item["requested_method"]),
                -float(item["retention_ratio"]),
                int(item["seed"]),
            )
        )
    if abs(totals[0] - totals[1]) > max(_cell_weight("lookm_reimpl", 0.5), 10.0):
        raise RuntimeError(f"worker estimates are unexpectedly imbalanced: {totals}")
    return bins


def materialize(spec_path: Path, destination: Path) -> Path:
    spec = _load_spec(spec_path)
    spec_payload = spec_path.read_bytes()
    spec_sha = _sha256(spec_payload)
    manifest_path = destination / "pilot_manifest.json"
    index_path = destination / "jobs.jsonl"
    if destination.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("spec_sha256") != spec_sha:
            raise RuntimeError("existing campaign was generated from a different spec")
        if _sha256(index_path.read_bytes()) != manifest.get("index_sha"):
            raise RuntimeError("existing campaign index SHA mismatch")
        return index_path

    profile, profile_path, profile_file_sha = _load_frozen_profile(spec)
    git_sha, git_dirty, source_patch = _source_provenance()
    if not git_dirty:
        raise RuntimeError(
            "this versioned campaign is explicitly classified as dirty-tree exploratory"
        )
    output_root = Path(str(spec["output_root"])).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse campaign output root: {output_root}")

    destination.mkdir(parents=True, exist_ok=False)
    _write_once(destination / "campaign_spec.yaml", spec_payload)
    _write_once(destination / "source.patch", source_patch)
    source_patch_sha = _sha256(source_patch)
    source_ledgers_root = Path(str(spec["source_ledgers_root"])).resolve()
    sample_count = int(spec["sample_count_per_task"])
    diagnostic_count = int(spec["diagnostic_sample_count_per_task"])
    calibration_ids = set(cast("list[str]", profile["calibration_sample_ids"]))
    ledgers: dict[str, tuple[Path, str, tuple[str, ...]]] = {}
    for task_key in cast("list[str]", spec["task_keys"]):
        source_path = source_ledgers_root / f"{task_key}.json"
        source, source_sha = _source_ledger(source_path, task_key)
        selected = tuple(
            sample_id
            for sample_id in cast("list[str]", source["sample_ids"])
            if sample_id not in calibration_ids
        )[:sample_count]
        if len(selected) != sample_count:
            raise RuntimeError(f"not enough leakage-free samples for {task_key}")
        task = TASK_CATALOG[task_key]
        ledger_payload = _canonical_bytes(
            {
                "schema_version": 1,
                "task": task.task_name,
                "dataset_id": task.dataset_id,
                "dataset_revision": task.revision,
                "split": task.split,
                "selection_seed": int(spec["deterministic_method_seed"]),
                "selection_policy": "ordered_parent_ledger_excluding_frozen_calibration_ids",
                "parent_ledger_path": str(source_path),
                "parent_ledger_sha256": source_sha,
                "frozen_profile_sha256": profile["profile_sha256"],
                "sample_count": len(selected),
                "sample_ids": list(selected),
            }
        )
        ledger_path = destination / "sample_ledgers" / f"{task_key}.json"
        _write_once(ledger_path, ledger_payload)
        ledgers[task_key] = (ledger_path.resolve(), _sha256(ledger_payload), selected)

    configs_root = destination / "configs"
    all_cells: dict[str, list[dict[str, Any]]] = {}
    config_count = 0
    unique_configs: set[str] = set()
    methods = cast("list[str]", spec["methods"])
    ratios = tuple(float(value) for value in cast("list[float]", spec["retention_ratios"]))
    random_multiseed = {
        float(value) for value in cast("list[float]", spec["random_multiseed_ratios"])
    }
    all_seeds = tuple(int(value) for value in cast("list[int]", spec["selection_seeds"]))
    deterministic_seed = int(spec["deterministic_method_seed"])
    for task_key in cast("list[str]", spec["task_keys"]):
        ledger_path, ledger_sha, selected = ledgers[task_key]
        diagnostic_ids = selected[:diagnostic_count]
        task_cells: list[dict[str, Any]] = []
        for method in methods:
            method_ratios = (1.0,) if method == "full_kv" else ratios
            for ratio in method_ratios:
                seeds = (
                    all_seeds
                    if method == "random_kv" and ratio in random_multiseed
                    else (deterministic_seed,)
                )
                for seed in seeds:
                    config = _resolved_config(
                        spec=spec,
                        profile=profile,
                        method=method,
                        task_key=task_key,
                        ratio=ratio,
                        seed=seed,
                        diagnostic_ids=diagnostic_ids,
                        ledger_sha=ledger_sha,
                    )
                    digest = config_sha256(config)
                    if digest in unique_configs:
                        raise RuntimeError(f"duplicate resolved configuration: {digest}")
                    unique_configs.add(digest)
                    config_path = configs_root / f"{config_count:04d}-{digest}.yaml"
                    config_payload = cast(
                        "dict[str, object]", json.loads(canonical_config_json(config))
                    )
                    _write_once(
                        config_path,
                        cast("str", yaml.safe_dump(config_payload, sort_keys=True)).encode(),
                    )
                    ratio_label = str(ratio).replace(".", "p")
                    run_id = (
                        f"llava-full-q-v1-{spec_sha[:10]}-{task_key}-{method}-"
                        f"r{ratio_label}-s{seed}"
                    )
                    result_label = (
                        str(cast("dict[str, Any]", spec["mosaickv_full_status"])["result_label"])
                        if method == "mosaickv_full"
                        else (
                            "generalized_prefixkv_reimpl" if method == "prefixkv_reimpl" else method
                        )
                    )
                    task_cells.append(
                        {
                            "config_path": str(config_path.resolve()),
                            "config_sha": digest,
                            "requested_method": method,
                            "result_label": result_label,
                            "retention_ratio": ratio,
                            "seed": seed,
                            "role": (
                                "random_independent_seed"
                                if method == "random_kv" and len(seeds) > 1
                                else "primary"
                            ),
                            "run_id": run_id,
                            "estimated_weight": _cell_weight(method, ratio),
                        }
                    )
                    config_count += 1
        all_cells[task_key] = task_cells

    jobs: list[dict[str, Any]] = []
    for task_key in cast("list[str]", spec["task_keys"]):
        task = TASK_CATALOG[task_key]
        ledger_path, ledger_sha, _selected = ledgers[task_key]
        for local_worker, cells in enumerate(_balance(all_cells[task_key])):
            array_index = len(jobs)
            jobs.append(
                {
                    "array_index": array_index,
                    "backend": spec["backend_key"],
                    "pilot_revision": spec["campaign_revision"],
                    "classification": spec["classification"],
                    "cells": cells,
                    "dataset_id": task.dataset_id,
                    "dataset_revision": task.revision,
                    "ordinary_reference": local_worker == 0,
                    "output_root": str(output_root),
                    "sample_id_file": str(ledger_path),
                    "sample_id_file_sha": ledger_sha,
                    "seed": deterministic_seed,
                    "source_git_sha": git_sha,
                    "source_git_dirty": git_dirty,
                    "source_patch_sha256": source_patch_sha,
                    "subset_size": None,
                    "task": task.task_name,
                    "task_key": task_key,
                    "task_worker_index": local_worker,
                    "trace_directory": str(output_root / "traces"),
                }
            )
    if len(jobs) != int(spec["worker_count"]):
        raise RuntimeError("campaign did not materialize exactly eight workers")
    if config_count != 156:
        raise RuntimeError(f"campaign expected 156 cells, observed {config_count}")

    index_payload = b"".join(_canonical_bytes(job) for job in jobs)
    _write_once(index_path, index_payload)
    manifest = {
        "schema_version": 1,
        "classification": spec["classification"],
        "spec_sha256": spec_sha,
        "index_sha": _sha256(index_payload),
        "source_git_sha": git_sha,
        "source_git_dirty": git_dirty,
        "source_patch_path": str((destination / "source.patch").resolve()),
        "source_patch_sha256": source_patch_sha,
        "frozen_profile_path": str(profile_path),
        "frozen_profile_file_sha256": profile_file_sha,
        "frozen_profile_sha256": profile["profile_sha256"],
        "job_count": len(jobs),
        "config_count": config_count,
        "expected_evaluation_rows": config_count * sample_count,
        "ordinary_reference_rows": len(cast("list[str]", spec["task_keys"])) * sample_count,
        "output_root": str(output_root),
        "sample_ledgers": {
            task_key: {"path": str(path), "sha256": sha, "sample_count": len(ids)}
            for task_key, (path, sha, ids) in ledgers.items()
        },
        "worker_cell_counts": [len(cast("list[dict[str, Any]]", job["cells"])) for job in jobs],
        "worker_estimated_weights": [
            sum(
                float(cell["estimated_weight"])
                for cell in cast("list[dict[str, Any]]", job["cells"])
            )
            for job in jobs
        ],
    }
    _write_once(manifest_path, _canonical_bytes(manifest))
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/experiments/llava_full_quality_8gpu_v1.yaml"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/campaigns/llava-full-quality-8gpu-v1"),
    )
    args = parser.parse_args()
    index = materialize(args.spec.resolve(), args.output_directory.resolve())
    print(str(index.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
