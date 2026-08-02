#!/usr/bin/env python3
"""Select and immutably freeze Experiment-5 hyperparameters from calibration only."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from mosaickv.generalization import (
    CalibrationCandidate,
    canonical_hyperparameters,
    is_completed_forecast_diagnostic,
    select_calibration_winner,
)
from mosaickv.manifest import sha256_bytes


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen hyperparameters: {path}")
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
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _trace(trace_root: Path, run_id: str, sample_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((trace_root / run_id).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("sample_id") == sample_id and value.get("status") == "completed":
            matches.append((path, cast("dict[str, Any]", value)))
    if len(matches) != 1:
        raise RuntimeError(f"expected one trace for {run_id}/{sample_id}, found {len(matches)}")
    return matches[0]


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("calibration metric is missing or non-finite")
    return sum(values) / len(values)


def freeze(index: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    jobs = _rows(index)
    if len(jobs) != 1 or jobs[0].get("job_kind") != "calibration":
        raise RuntimeError("freezing requires the one-job Experiment-5 calibration index")
    matrix_manifest_path = index.parent / "manifest.json"
    matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    if matrix_manifest.get("phase") != "calibration":
        raise RuntimeError("cannot freeze from a non-calibration matrix")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("experiment_id") != matrix_manifest["experiment_id"]:
        raise RuntimeError("calibration matrix and Experiment-5 specification differ")
    if sha256_bytes(spec_path.read_bytes()) != matrix_manifest["spec_sha"]:
        raise RuntimeError("calibration matrix specification SHA mismatch")
    job = jobs[0]
    output_root = Path(job["output_root"])
    trace_root = output_root / "traces"
    cells = cast("list[dict[str, Any]]", job["cells"])
    source_provenance: set[tuple[str, bool, str]] = set()
    artifact_records: list[dict[str, str]] = []
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    traces_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        run_root = output_root / str(cell["run_id"])
        result_path = run_root / "results.jsonl"
        manifest_path = run_root / "manifest.json"
        if not result_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"calibration cell is incomplete: {cell['cell_id']}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        if source["config_sha"] != cell["config_sha"]:
            raise RuntimeError(f"calibration config SHA mismatch: {cell['cell_id']}")
        source_provenance.add(
            (str(source["git_sha"]), bool(source["git_dirty"]), str(source["patch_sha"]))
        )
        rows = _rows(result_path)
        if len(rows) != len(job["sample_ids"]):
            raise RuntimeError(f"calibration row count mismatch: {cell['cell_id']}")
        if any(row.get("status") != "completed" or row.get("error") is not None for row in rows):
            raise RuntimeError(f"calibration has a failed sample: {cell['cell_id']}")
        cell_traces: list[dict[str, Any]] = []
        trace_paths: list[Path] = []
        for row in rows:
            trace_path, trace = _trace(trace_root, str(cell["run_id"]), str(row["sample_id"]))
            trace_paths.append(trace_path)
            cell_traces.append(trace)
        rows_by_cell[str(cell["cell_id"])] = rows
        traces_by_cell[str(cell["cell_id"])] = cell_traces
        artifact_records.append(
            {
                "cell_id": str(cell["cell_id"]),
                "config_sha": str(cell["config_sha"]),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha": sha256_bytes(manifest_path.read_bytes()),
                "result_path": str(result_path.resolve()),
                "result_sha": sha256_bytes(result_path.read_bytes()),
                "trace_set_sha": sha256_bytes(
                    b"".join(path.read_bytes() for path in sorted(trace_paths))
                ),
            }
        )
    if len(source_provenance) != 1:
        raise RuntimeError(f"calibration source provenance differs: {sorted(source_provenance)}")
    calibration = cast("dict[str, Any]", spec["calibration"])
    candidate_specs = {
        str(candidate["id"]): cast("dict[str, Any]", candidate)
        for candidate in cast("list[dict[str, Any]]", calibration["candidates"])
    }
    measured: list[CalibrationCandidate] = []
    metric_records: list[dict[str, Any]] = []
    for candidate_id in candidate_specs:
        rows = rows_by_cell[candidate_id]
        traces = traces_by_cell[candidate_id]
        token_agreement: list[float] = []
        selected_recall: list[float] = []
        regret: list[float] = []
        for trace in traces:
            fidelity = cast("dict[str, Any]", trace["fidelity_diagnostic"])
            forecast = cast("dict[str, Any]", trace["forecast_quality_diagnostic"])
            if fidelity.get("status") != "completed" or not is_completed_forecast_diagnostic(
                forecast
            ):
                raise RuntimeError(f"candidate {candidate_id} lacks completed diagnostics")
            token_agreement.append(float(fidelity["token_agreement"]))
            selected_recall.append(float(forecast["mean_selected_block_recall"]))
            regret.append(float(forecast["mean_forecast_regret"]))
        candidate = CalibrationCandidate(
            candidate_id=candidate_id,
            mean_task_score=_mean([float(row["task_score"]) for row in rows]),
            mean_token_agreement=_mean(token_agreement),
            mean_selected_block_recall=_mean(selected_recall),
            mean_forecast_regret=_mean(regret),
            mean_end_to_end_seconds=_mean([float(row["end_to_end_time"]) for row in rows]),
        )
        measured.append(candidate)
        metric_records.append(
            {
                "candidate_id": candidate.candidate_id,
                "mean_task_score": candidate.mean_task_score,
                "mean_token_agreement": candidate.mean_token_agreement,
                "mean_selected_block_recall": candidate.mean_selected_block_recall,
                "mean_forecast_regret": candidate.mean_forecast_regret,
                "mean_end_to_end_seconds": candidate.mean_end_to_end_seconds,
            }
        )
    winner = select_calibration_winner(tuple(measured))
    fixed = dict(cast("dict[str, Any]", spec["fixed_hyperparameters"]))
    winner_spec = candidate_specs[winner.candidate_id]
    fixed["forecast_mode"] = winner_spec["forecast_mode"]
    fixed["draft_steps"] = winner_spec["draft_steps"]
    hyperparameters = canonical_hyperparameters(fixed)
    provenance = next(iter(source_provenance))
    result = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "classification": spec["classification"],
        "calibration_index_sha": matrix_manifest["index_sha"],
        "calibration_ledger_sha": job["sample_ledger_sha"],
        "calibration_sample_ids": job["sample_ids"],
        "calibration_model": job["model_id"],
        "calibration_task": job["task_key"],
        "winner_rule": calibration["winner_rule"],
        "winner": winner.candidate_id,
        "candidate_metrics": metric_records,
        "hyperparameters": hyperparameters,
        "source": {
            "git_sha": provenance[0],
            "git_dirty": provenance[1],
            "patch_sha": provenance[2],
        },
        "artifacts": artifact_records,
        "matrix_manifest_path": str(matrix_manifest_path.resolve()),
        "matrix_manifest_sha": sha256_bytes(matrix_manifest_path.read_bytes()),
    }
    payload = yaml.safe_dump(result, sort_keys=True).encode()
    _write_once(output, payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = freeze(args.index.resolve(), args.spec.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "winner": result["winner"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
