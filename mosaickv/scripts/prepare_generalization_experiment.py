#!/usr/bin/env python3
"""Materialize immutable calibration or frozen Experiment-5 evaluation jobs."""

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
from mosaickv.experiment_matrix import MODEL_CATALOG, TASK_CATALOG, _base_config
from mosaickv.generalization import assert_disjoint_sample_sets, canonical_hyperparameters
from mosaickv.manifest import sha256_bytes
from mosaickv.types import (
    ForecastMode,
    MosaicKVMethod,
    RepairCandidateRanking,
    RepairPolicy,
)


def _write_once(path: Path, payload: bytes, mode: int = 0o444) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable Experiment-5 artifact: {path}")
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


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _spec(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Experiment-5 specification must be a schema-version-1 object")
    if value.get("primary_budget") != "packed_prefill_active_kv_bytes":
        raise ValueError("Experiment 5 requires the physical packed-prefill byte budget")
    if float(value.get("time_budget_hours", 0.0)) > 3.5:
        raise ValueError("bounded Experiment 5 may not exceed 3.5 hours")
    calibration = value.get("calibration")
    evaluation = value.get("evaluation")
    if not isinstance(calibration, dict) or not isinstance(evaluation, dict):
        raise ValueError("calibration and evaluation sections are required")
    candidates = calibration.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("calibration requires at least two predeclared candidates")
    identifiers = [item.get("id") for item in candidates if isinstance(item, dict)]
    if len(identifiers) != len(candidates) or len(set(identifiers)) != len(identifiers):
        raise ValueError("calibration candidate IDs must be present and unique")
    return cast("dict[str, Any]", value)


def _ledger(
    project_root: Path, relative_path: str, task_key: str
) -> tuple[Path, tuple[str, ...], str]:
    path = project_root / relative_path
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
        raise ValueError(f"Experiment-5 ledger identity mismatch: {path}")
    sample_ids = tuple(str(item) for item in value.get("sample_ids", ()))
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Experiment-5 ledger must contain unique sample IDs: {path}")
    return path.resolve(), sample_ids, sha256_bytes(payload)


def _resolved_hyperparameters(
    spec: dict[str, Any], candidate: dict[str, Any] | None = None
) -> dict[str, Any]:
    fixed = dict(cast("dict[str, Any]", spec["fixed_hyperparameters"]))
    if candidate is not None:
        fixed["forecast_mode"] = candidate["forecast_mode"]
        fixed["draft_steps"] = candidate["draft_steps"]
    return canonical_hyperparameters(fixed)


def _run_config(
    spec: dict[str, Any],
    hyperparameters: dict[str, Any],
    *,
    model_key: str,
    task_key: str,
    method: MosaicKVMethod,
    sample_ids: tuple[str, ...],
    ledger_sha: str,
    prefill_attention: str = "eager",
    decode_attention: str | None = None,
) -> Any:
    is_full = method is MosaicKVMethod.FULL_KV
    config = _base_config(
        model_key=model_key,
        backend_key=(
            "hf_sdpa"
            if prefill_attention == "sdpa"
            else (
                "hf_flash_attention_2" if prefill_attention == "flash_attention_2" else "hf_eager"
            )
        ),
        method_name=method.value,
        task_key=task_key,
        retention_ratio=1.0 if is_full else float(hyperparameters["retention_ratio"]),
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
    )
    if is_full:
        return replace(
            config,
            execution=replace(
                config.execution,
                attention_implementation=prefill_attention,
                decode_attention_implementation=decode_attention,
            ),
            diagnostics=FidelityDiagnosticsConfig(enabled=False),
        )
    mode = ForecastMode(str(hyperparameters["forecast_mode"]))
    config = replace(
        config,
        execution=replace(
            config.execution,
            attention_implementation=prefill_attention,
            decode_attention_implementation=decode_attention,
        ),
        cache=replace(config.cache, block_size=int(hyperparameters["block_size"])),
        forecasting=replace(
            config.forecasting,
            mode=mode,
            prompt_window=int(hyperparameters["prompt_window"]),
            draft_steps=int(hyperparameters["draft_steps"]),
            centroid_count=int(hyperparameters["centroid_count"]),
        ),
        graph=replace(
            config.graph,
            max_neighbors=int(hyperparameters["graph_max_neighbors"]),
            fallback_max_position_span=int(hyperparameters["prototype_max_position_span"]),
        ),
        selection=replace(
            config.selection,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=float(hyperparameters["minimum_stratum_fraction"]),
        ),
        prototypes=replace(
            config.prototypes,
            group_size=int(hyperparameters["prototype_group_size"]),
            max_position_span=int(hyperparameters["prototype_max_position_span"]),
        ),
        diagnostics=FidelityDiagnosticsConfig(
            enabled=True,
            sample_ids=sample_ids,
            sample_ledger_sha=ledger_sha,
            forecast_oracle_steps=4,
        ),
    )
    if method is MosaicKVMethod.MOSAICKV_FULL:
        config = replace(
            config,
            residual=ResidualConfig(enabled=True, require_pinned_memory=True),
            repair=RepairConfig(
                enabled=True,
                policy=RepairPolicy(str(hyperparameters["repair_policy"])),
                entropy_threshold=float(hyperparameters["repair_entropy_threshold"]),
                prototype_risk_threshold=float(hyperparameters["repair_prototype_risk_threshold"]),
                prototype_risk_min_entropy=float(
                    hyperparameters["repair_prototype_risk_min_entropy"]
                ),
                max_blocks_per_step=int(hyperparameters["repair_max_blocks_per_step"]),
                candidate_ranking=RepairCandidateRanking(
                    str(hyperparameters["repair_candidate_ranking"])
                ),
                preserve_partial_prototype=bool(hyperparameters["preserve_partial_prototype"]),
            ),
        )
    return config


def _cell(
    destination: Path,
    ordinal: int,
    config: Any,
    *,
    cell_id: str,
    role: str,
    run_id: str,
    task_key: str,
) -> dict[str, Any]:
    digest = config_sha256(config)
    path = destination / "configs" / f"{ordinal:04d}-{digest}.yaml"
    payload = cast("dict[str, object]", json.loads(canonical_config_json(config)))
    _write_once(path, yaml.safe_dump(payload, sort_keys=True).encode())
    return {
        "cell_id": cell_id,
        "role": role,
        "task_key": task_key,
        "method": config.method.value,
        "prefill_attention": config.execution.attention_implementation,
        "decode_attention": (
            config.execution.decode_attention_implementation
            or config.execution.attention_implementation
        ),
        "config_path": str(path.resolve()),
        "config_sha": digest,
        "run_id": run_id,
    }


def _write_matrix(
    destination: Path,
    *,
    spec: dict[str, Any],
    spec_sha: str,
    output_root: Path,
    phase: str,
    jobs: list[dict[str, Any]],
    config_count: int,
    frozen_sha: str | None,
) -> Path:
    index_payload = b"".join(_json_bytes(job) for job in jobs)
    _write_once(destination / "jobs.jsonl", index_payload)
    _write_once(
        destination / "manifest.json",
        _json_bytes(
            {
                "classification": spec["classification"],
                "config_count": config_count,
                "experiment_id": spec["experiment_id"],
                "frozen_sha": frozen_sha,
                "index_sha": hashlib.sha256(index_payload).hexdigest(),
                "job_count": len(jobs),
                "output_root": str(output_root.resolve()),
                "phase": phase,
                "spec_sha": spec_sha,
            }
        ),
    )
    return destination / "jobs.jsonl"


def prepare_calibration(spec_path: Path, destination: Path, output_root: Path) -> Path:
    spec = _spec(spec_path)
    if destination.exists():
        return destination / "jobs.jsonl"
    destination.mkdir(parents=True, exist_ok=False)
    _write_once(destination / "study.yaml", spec_path.read_bytes())
    project_root = spec_path.resolve().parents[2]
    calibration = cast("dict[str, Any]", spec["calibration"])
    task_key = str(calibration["task"])
    model_key = str(calibration["model"])
    ledger_path, sample_ids, ledger_sha = _ledger(
        project_root, str(calibration["ledger"]), task_key
    )
    spec_sha = sha256_bytes(spec_path.read_bytes())
    cells: list[dict[str, Any]] = []
    reference_hyperparameters = _resolved_hyperparameters(
        spec, cast("dict[str, Any]", calibration["candidates"][0])
    )
    cells.append(
        _cell(
            destination,
            0,
            _run_config(
                spec,
                reference_hyperparameters,
                model_key=model_key,
                task_key=task_key,
                method=MosaicKVMethod.FULL_KV,
                sample_ids=sample_ids,
                ledger_sha=ledger_sha,
            ),
            cell_id="full_reference",
            role="calibration_reference",
            run_id=f"exp5-cal-{spec_sha[:10]}-full",
            task_key=task_key,
        )
    )
    for ordinal, candidate in enumerate(
        cast("list[dict[str, Any]]", calibration["candidates"]), start=1
    ):
        candidate_id = str(candidate["id"])
        cells.append(
            _cell(
                destination,
                ordinal,
                _run_config(
                    spec,
                    _resolved_hyperparameters(spec, candidate),
                    model_key=model_key,
                    task_key=task_key,
                    method=MosaicKVMethod.MOSAICKV_EXACT,
                    sample_ids=sample_ids,
                    ledger_sha=ledger_sha,
                ),
                cell_id=candidate_id,
                role="calibration_candidate",
                run_id=f"exp5-cal-{spec_sha[:10]}-{candidate_id}",
                task_key=task_key,
            )
        )
    jobs = [
        {
            "array_index": 0,
            "job_kind": "calibration",
            "model_key": model_key,
            "model_id": MODEL_CATALOG[model_key].model_id,
            "task_key": task_key,
            "task": TASK_CATALOG[task_key].task_name,
            "dataset_revision": TASK_CATALOG[task_key].revision,
            "sample_ids": list(sample_ids),
            "sample_ledger_path": str(ledger_path),
            "sample_ledger_sha": ledger_sha,
            "output_root": str(output_root.resolve()),
            "trace_directory": str((output_root / "traces").resolve()),
            "cells": cells,
        }
    ]
    return _write_matrix(
        destination,
        spec=spec,
        spec_sha=spec_sha,
        output_root=output_root,
        phase="calibration",
        jobs=jobs,
        config_count=len(cells),
        frozen_sha=None,
    )


def prepare_evaluation(
    spec_path: Path,
    frozen_path: Path,
    destination: Path,
    output_root: Path,
) -> Path:
    spec = _spec(spec_path)
    if destination.exists():
        return destination / "jobs.jsonl"
    frozen_payload = frozen_path.read_bytes()
    frozen = yaml.safe_load(frozen_payload)
    if not isinstance(frozen, dict) or frozen.get("schema_version") != 1:
        raise ValueError("frozen hyperparameter artifact is invalid")
    if frozen.get("experiment_id") != spec["experiment_id"]:
        raise ValueError("frozen hyperparameters belong to another experiment")
    hyperparameters = canonical_hyperparameters(cast("dict[str, Any]", frozen["hyperparameters"]))
    frozen_sha = sha256_bytes(frozen_payload)
    project_root = spec_path.resolve().parents[2]
    calibration = cast("dict[str, Any]", spec["calibration"])
    _calibration_path, calibration_ids, calibration_ledger_sha = _ledger(
        project_root, str(calibration["ledger"]), str(calibration["task"])
    )
    if frozen.get("calibration_ledger_sha") != calibration_ledger_sha:
        raise ValueError("frozen artifact calibration-ledger SHA mismatch")
    evaluation = cast("dict[str, Any]", spec["evaluation"])
    task_ledgers: dict[str, tuple[Path, tuple[str, ...], str]] = {}
    for task_key, relative in cast("dict[str, str]", evaluation["tasks"]).items():
        task_ledgers[task_key] = _ledger(project_root, relative, task_key)
    assert_disjoint_sample_sets(
        calibration_ids,
        {task: value[1] for task, value in task_ledgers.items()},
    )
    destination.mkdir(parents=True, exist_ok=False)
    _write_once(destination / "study.yaml", spec_path.read_bytes())
    _write_once(destination / "frozen_hyperparameters.yaml", frozen_payload)
    spec_sha = sha256_bytes(spec_path.read_bytes())
    jobs: list[dict[str, Any]] = []
    ordinal = 0
    for model_key in cast("list[str]", evaluation["models"]):
        cells: list[dict[str, Any]] = []
        task_records: list[dict[str, Any]] = []
        for task_key, (ledger_path, sample_ids, ledger_sha) in task_ledgers.items():
            task_records.append(
                {
                    "task_key": task_key,
                    "task": TASK_CATALOG[task_key].task_name,
                    "dataset_revision": TASK_CATALOG[task_key].revision,
                    "sample_ids": list(sample_ids),
                    "sample_ledger_path": str(ledger_path),
                    "sample_ledger_sha": ledger_sha,
                }
            )
            for role, method in (
                ("full_reference", MosaicKVMethod.FULL_KV),
                ("frozen_algorithmic", MosaicKVMethod.MOSAICKV_EXACT),
            ):
                cells.append(
                    _cell(
                        destination,
                        ordinal,
                        _run_config(
                            spec,
                            hyperparameters,
                            model_key=model_key,
                            task_key=task_key,
                            method=method,
                            sample_ids=sample_ids,
                            ledger_sha=ledger_sha,
                        ),
                        cell_id=f"{task_key}_{role}",
                        role=role,
                        run_id=(
                            f"exp5-eval-{spec_sha[:10]}-{frozen_sha[:10]}-"
                            f"{model_key}-{task_key}-{role}"
                        ),
                        task_key=task_key,
                    )
                )
                ordinal += 1
            adaptation = cast("dict[str, str]", evaluation["architecture_adaptations"]).get(
                model_key
            )
            if adaptation == MosaicKVMethod.MOSAICKV_FULL.value:
                cells.append(
                    _cell(
                        destination,
                        ordinal,
                        _run_config(
                            spec,
                            hyperparameters,
                            model_key=model_key,
                            task_key=task_key,
                            method=MosaicKVMethod.MOSAICKV_FULL,
                            sample_ids=sample_ids,
                            ledger_sha=ledger_sha,
                        ),
                        cell_id=f"{task_key}_architecture_adapted_full",
                        role="architecture_adapted_full",
                        run_id=(
                            f"exp5-eval-{spec_sha[:10]}-{frozen_sha[:10]}-"
                            f"{model_key}-{task_key}-adapted-full"
                        ),
                        task_key=task_key,
                    )
                )
                ordinal += 1
        jobs.append(
            {
                "array_index": len(jobs),
                "job_kind": "model_generalization",
                "model_key": model_key,
                "model_id": MODEL_CATALOG[model_key].model_id,
                "output_root": str(output_root.resolve()),
                "trace_directory": str((output_root / "traces").resolve()),
                "tasks": task_records,
                "cells": cells,
            }
        )
    backend = cast("dict[str, Any]", spec["backend_parity"])
    task_key = str(backend["task"])
    _ledger_path, all_ids, _ledger_sha = task_ledgers[task_key]
    requested_ids = tuple(str(item) for item in backend["sample_ids"])
    if not set(requested_ids) <= set(all_ids):
        raise ValueError("backend-parity sample IDs are not in the held-out ledger")
    task = TASK_CATALOG[task_key]
    backend_ledger_payload = (
        json.dumps(
            {
                "dataset_id": task.dataset_id,
                "dataset_revision": task.revision,
                "sample_ids": list(requested_ids),
                "schema_version": 1,
                "split": task.split,
                "task": task.task_name,
            },
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    backend_ledger_path = destination / "ledgers" / "backend_parity.json"
    _write_once(backend_ledger_path, backend_ledger_payload)
    backend_ledger_sha = sha256_bytes(backend_ledger_payload)
    for raw_cell in cast("list[dict[str, Any]]", backend["cells"]):
        method = MosaicKVMethod(str(raw_cell["method"]))
        cell = _cell(
            destination,
            ordinal,
            _run_config(
                spec,
                hyperparameters,
                model_key=str(backend["model"]),
                task_key=task_key,
                method=method,
                sample_ids=requested_ids,
                ledger_sha=backend_ledger_sha,
                prefill_attention=str(raw_cell["prefill"]),
                decode_attention=str(raw_cell["decode"]),
            ),
            cell_id=str(raw_cell["id"]),
            role="backend_parity",
            run_id=(f"exp5-backend-{spec_sha[:10]}-{frozen_sha[:10]}-{raw_cell['id']}"),
            task_key=task_key,
        )
        ordinal += 1
        jobs.append(
            {
                "array_index": len(jobs),
                "job_kind": "backend_parity",
                "model_key": str(backend["model"]),
                "model_id": MODEL_CATALOG[str(backend["model"])].model_id,
                "task_key": task_key,
                "task": TASK_CATALOG[task_key].task_name,
                "dataset_revision": TASK_CATALOG[task_key].revision,
                "sample_ids": list(requested_ids),
                "sample_ledger_path": str(backend_ledger_path.resolve()),
                "sample_ledger_sha": backend_ledger_sha,
                "output_root": str(output_root.resolve()),
                "trace_directory": str((output_root / "traces").resolve()),
                "cells": [cell],
            }
        )
    return _write_matrix(
        destination,
        spec=spec,
        spec_sha=spec_sha,
        output_root=output_root,
        phase="frozen_evaluation",
        jobs=jobs,
        config_count=ordinal,
        frozen_sha=frozen_sha,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibration", "evaluation"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    if args.phase == "calibration":
        index = prepare_calibration(
            args.spec.resolve(), args.destination.resolve(), args.output_root.resolve()
        )
    else:
        if args.frozen is None:
            parser.error("evaluation requires --frozen")
        index = prepare_evaluation(
            args.spec.resolve(),
            args.frozen.resolve(),
            args.destination.resolve(),
            args.output_root.resolve(),
        )
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
