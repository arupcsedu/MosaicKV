#!/usr/bin/env python3
"""Validate two fail-fast canaries and submit the remaining sweep arms."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.evaluation.coco_shared_bytes import STORAGE_FIELDS, validate_caption_rows


def _passed_worker(path: Path) -> dict[str, Any]:
    attempts = sorted(path.glob("attempt-*/worker.json"))
    records = [json.loads(item.read_text()) for item in attempts]
    passed = [record for record in records if record.get("status") == "passed"]
    if len(records) != 1 or len(passed) != 1:
        raise ValueError(f"canary requires exactly one passed attempt: {path}")
    return passed[0]


def _nonempty(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty {label}: {path}")


def validate_canaries(root: Path, plan_sha256: str) -> dict[str, Any]:
    """Require complete manifests, rows, traces, and physical storage accounting."""

    plan_path = root / "plan.json"
    if digest(plan_path) != plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    if plan.get("canary_indices") != [0, 12]:
        raise ValueError("unexpected canary registration")
    samples = json.loads(Path(plan["ledger"]).read_text())["samples"]
    expected_ids = set(plan["sample_ids"])
    reports = []
    for index in plan["canary_indices"]:
        job = plan["jobs"][index]
        worker = _passed_worker(Path(job["output"]))
        if (
            worker.get("plan_sha256") != plan_sha256
            or worker.get("source_base_git_sha") != plan["source_base_git_sha"]
            or worker.get("source_content_sha256") != plan["source_content_sha256"]
        ):
            raise ValueError("canary provenance mismatch")
        rows = worker["record"]["rows"]
        validate_caption_rows(rows, samples, plan["output_tokens"])
        if {row["sample_id"] for row in rows} != expected_ids:
            raise ValueError("canary sample population mismatch")
        manifests = {Path(row["manifest_path"]) for row in rows}
        if len(manifests) != 1:
            raise ValueError("canary rows do not share one immutable manifest")
        manifest_path = manifests.pop()
        _nonempty(manifest_path, "manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("evaluation", {}).get("evaluator_commit") != plan["source_base_git_sha"]:
            raise ValueError("manifest evaluator commit does not match the detached worktree")
        raw_path = manifest_path.parent / "results.jsonl"
        aggregate_path = manifest_path.parent / "aggregate.parquet"
        _nonempty(raw_path, "raw prediction JSONL")
        _nonempty(aggregate_path, "aggregate Parquet")
        if len([line for line in raw_path.read_text().splitlines() if line.strip()]) != len(rows):
            raise ValueError("raw prediction denominator mismatch")
        if manifest.get("artifacts", {}).get("raw_output_sha") != digest(raw_path):
            raise ValueError("raw prediction hash mismatch")
        for key in ("metric_observations_parquet_output", "metric_aggregates_parquet_output"):
            _nonempty(Path(worker["record"][key]), key)
        traces = worker["traces"]
        if set(traces) != expected_ids:
            raise ValueError("canary trace population mismatch")
        for sample_id, trace_name in traces.items():
            trace_path = Path(trace_name)
            _nonempty(trace_path, f"trace for {sample_id}")
            trace = json.loads(trace_path.read_text())
            if trace.get("status") != "completed" or trace.get("sample_id") != sample_id:
                raise ValueError("incomplete or misbound trace")
            lifecycle = trace.get("cache_lifecycle")
            if not isinstance(lifecycle, dict):
                raise ValueError("trace lacks cache lifecycle accounting")
            for field in STORAGE_FIELDS:
                if type(lifecycle.get(field)) is not int or lifecycle[field] <= 0:
                    raise ValueError(f"trace lacks {field}")
            if trace.get("effective_method") != job["effective_method"]:
                raise ValueError("canary effective method mismatch")
            if job["label"] == "output_preserving_individual":
                reconstruction = trace.get("output_reconstruction")
                if (
                    not reconstruction
                    or reconstruction["final_error"] > reconstruction["initial_error"] + 1e-6
                ):
                    raise ValueError("output-preserving objective is missing or increasing")
        storage = {
            field: sorted({row[field] for row in rows})
            for field in STORAGE_FIELDS
        }
        if job["cap_bytes"] is not None and max(storage[STORAGE_FIELDS[0]]) > job["cap_bytes"]:
            raise ValueError("canary exceeds its registered initial physical-byte cap")
        reports.append(
            {
                "index": index,
                "label": job["label"],
                "rows": len(rows),
                "manifest": str(manifest_path),
                "manifest_sha256": digest(manifest_path),
                "raw_predictions": str(raw_path),
                "raw_predictions_sha256": digest(raw_path),
                "storage_bytes": storage,
            }
        )
    return {
        "status": "passed",
        "classification": plan["classification"],
        "plan_sha256": plan_sha256,
        "source_base_git_sha": plan["source_base_git_sha"],
        "source_content_sha256": plan["source_content_sha256"],
        "canaries": reports,
    }


def _submit(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip().split(";", maxsplit=1)[0]


def promote(args: argparse.Namespace) -> None:
    root, source = args.root.resolve(), args.source.resolve()
    report = validate_canaries(root, args.plan_sha256)
    write_once(root / "canary" / "report.json", report)
    plan = json.loads((root / "plan.json").read_text())
    indices = ",".join(str(index) for index in plan["remaining_indices"])
    slurm = root / "slurm"
    remaining_id = _submit(
        [
            "sbatch",
            "--parsable",
            "--hold",
            f"--array={indices}%4",
            f"--output={slurm}/%A_%a.out",
            f"--error={slurm}/%A_%a.err",
            str(source / "mosaickv/slurm/output_preserving_sweep.sbatch"),
            str(source),
            str(root),
            args.plan_sha256,
        ]
    )
    policy_path = root / "remaining_a100_policy.json"
    state_path = root / "remaining_a100_state.json"
    write_once(
        policy_path,
        {
            "timezone": "America/New_York",
            "maximum_worker_minutes": 120,
            "maximum_a100s": 4,
            "window": "all_times",
            "arrays": [int(remaining_id)],
            "adopt_existing_holds": [int(remaining_id)],
            "blocked_arrays": {},
        },
    )
    controller_id = _submit(
        [
            "sbatch",
            "--parsable",
            "--job-name=mk-op-controller",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=256M",
            "--time=02:00:00",
            "--partition=bii",
            f"--output={slurm}/%j-remaining-controller.out",
            f"--error={slurm}/%j-remaining-controller.err",
            "--wrap",
            (
                f"/scratch/djy8hg/env/mosaickv/bin/python "
                f"{source}/mosaickv/scripts/control_a100_schedule.py "
                f"--policy {policy_path} --state {state_path} --apply --watch"
            ),
        ]
    )
    analysis_id = _submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{remaining_id}",
            f"--output={slurm}/%j-analysis.out",
            f"--error={slurm}/%j-analysis.err",
            str(source / "mosaickv/slurm/output_preserving_sweep_analysis.sbatch"),
            str(source),
            str(root),
            args.plan_sha256,
        ]
    )
    write_once(
        root / "remaining-submission.json",
        {
            **report,
            "remaining_array_job_id": remaining_id,
            "controller_job_id": controller_id,
            "analysis_job_id": analysis_id,
            "remaining_indices": plan["remaining_indices"],
        },
    )
    print((root / "remaining-submission.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    promote(parser.parse_args())
