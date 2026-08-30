#!/usr/bin/env python3
"""Validate V3 continuation rows and merge them with the immutable V2 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from analyze_paper_quality_hf_vllm_v2 import (
    BACKENDS,
    METHODS,
    TASKS,
    _finite,
    _object,
    _paths,
    paired_task_summary,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_continuation(
    index_path: Path,
    *,
    protocol: dict[str, Any],
    protocol_sha: str,
) -> tuple[pd.DataFrame, list[str]]:
    rows = [
        _object(json.loads(line), f"{index_path}:{number}")
        for number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1)
        if line
    ]
    if len(rows) != 260:
        raise ValueError(f"expected 260 continuation jobs, found {len(rows)}")
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for job in rows:
        if job.get("protocol_sha256") != protocol_sha:
            raise ValueError("continuation job protocol hash changed")
        ledger_path = Path(str(job["sample_ledger_path"]))
        spec_path = Path(str(job["job_spec_path"]))
        if _sha(ledger_path) != job["sample_ledger_sha256"]:
            raise ValueError(f"continuation shard hash changed: {ledger_path}")
        if _sha(spec_path) != job["job_spec_sha256"]:
            raise ValueError(f"continuation job spec hash changed: {spec_path}")
        ledger = _object(json.loads(ledger_path.read_text(encoding="utf-8")), str(ledger_path))
        expected_ids = tuple(str(value) for value in cast("list[str]", ledger["sample_ids"]))
        job_root = (
            Path(str(job["output_root"]))
            / str(job["backend"])
            / str(job["task"])
            / str(job["run_namespace"])
        )
        completion = _object(
            json.loads((job_root / "job_complete.json").read_text(encoding="utf-8")),
            "job completion",
        )
        if (
            completion.get("status") != "complete"
            or completion.get("source_git_sha") != job["source_git_sha"]
            or completion.get("sample_ledger_sha256") != job["sample_ledger_sha256"]
        ):
            raise ValueError(f"continuation completion seal changed: {job_root}")
        for method in METHODS[str(job["backend"])]:
            result_path, manifest_path = _paths(job, method)
            manifest = _object(
                json.loads(manifest_path.read_text(encoding="utf-8")), str(manifest_path)
            )
            source = _object(manifest.get("source"), "manifest.source")
            model = _object(manifest.get("model"), "manifest.model")
            dataset = _object(manifest.get("dataset"), "manifest.dataset")
            execution = _object(manifest.get("execution"), "manifest.execution")
            if (
                source.get("git_dirty") is not False
                or source.get("canonical_eligible") is not True
                or source.get("git_sha") != job["source_git_sha"]
            ):
                raise ValueError(f"noncanonical continuation manifest: {manifest_path}")
            candidate = _object(protocol["candidate"], "candidate")
            if (
                model.get("id") != candidate["model_id"]
                or model.get("revision") != candidate["model_revision"]
                or dataset.get("revision") != job["dataset_revision"]
            ):
                raise ValueError(f"model/dataset identity changed: {manifest_path}")
            if job["backend"] == "vllm" and execution.get("request_isolation_verified") is not True:
                failures.append(
                    f"vllm isolation missing: {job['task']}/{job['shard_index']}/{method}"
                )
            observed = [
                _object(json.loads(line), f"{result_path}:{number}")
                for number, line in enumerate(
                    result_path.read_text(encoding="utf-8").splitlines(), 1
                )
                if line
            ]
            if len(observed) != len(expected_ids) or tuple(
                sorted(str(value["sample_id"]) for value in observed)
            ) != tuple(sorted(expected_ids)):
                raise ValueError(f"continuation sample coverage changed: {result_path}")
            for value in observed:
                if value.get("status") != "completed" or value.get("error") is not None:
                    raise ValueError(f"failed continuation sample: {result_path}")
                for key in (
                    "task_score",
                    "active_kv_bytes",
                    "residual_kv_bytes",
                    "compression_time",
                    "ttft",
                    "prefill_time",
                    "decode_time",
                    "end_to_end_time",
                    "generated_tokens",
                    "peak_gpu_memory",
                ):
                    _finite(value, key)
                effective_method = str(value.get("method", ""))
                if not effective_method.startswith(method):
                    raise ValueError(f"effective method label changed: {effective_method}")
                records.append(
                    {
                        **value,
                        "backend_scope": job["backend"],
                        "protocol_method": method,
                        "effective_method": effective_method,
                        "shard_index": job["shard_index"],
                        "source_git_sha": job["source_git_sha"],
                        "protocol_sha256": job["protocol_sha256"],
                        "sample_ledger_sha256": job["sample_ledger_sha256"],
                        "result_path": str(result_path),
                        "cohort": "v3_continuation",
                    }
                )
    return pd.DataFrame.from_records(records), failures


def analyze(
    *, index_path: Path, protocol_path: Path, repository: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    protocol = _object(yaml.safe_load(protocol_path.read_text(encoding="utf-8")), "protocol")
    if (
        protocol.get("experiment_id") != "paper_quality_hf_vllm_llava15_v3_continuation"
        or protocol.get("execution_authorized") is not True
    ):
        raise ValueError("V3 continuation protocol is not authorized")
    protocol_sha = _sha(protocol_path)
    reuse = _object(protocol["reuse"], "reuse")
    experiment = _object(protocol["experiment"], "experiment")
    gates = _object(protocol["primary_gates"], "gates")
    prior_root = Path(str(reuse["prior_campaign_root"])).resolve()
    prior_path = prior_root / "analysis/paired_rows.parquet"
    if _sha(prior_path) != reuse["prior_paired_rows_sha256"]:
        raise ValueError("reused V2 paired rows changed")
    prior = pd.read_parquet(prior_path)
    if len(prior) != 4608 or set(prior["status"]) != {"completed"}:
        raise ValueError("reused V2 rows are incomplete")
    prior = prior.copy()
    prior["cohort"] = "v2_reused"

    source_sha = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("composite analysis requires a clean worktree")
    runtime_paths = (
        "mosaickv/src/mosaickv",
        "mosaickv/configs/paper_candidate_llava_v1.yaml",
        "mosaickv/scripts/build_vllm_selector_plan.py",
        "mosaickv/slurm/paper_quality_hf_vllm_llava15_v2.sbatch",
    )
    runtime_diff = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            str(reuse["prior_source_git_sha"]),
            source_sha,
            "--",
            *runtime_paths,
        ),
        cwd=repository,
        check=False,
    )
    if runtime_diff.returncode != 0:
        raise ValueError("runtime or frozen candidate differs from the reused V2 campaign")

    continuation, failures = _collect_continuation(
        index_path, protocol=protocol, protocol_sha=protocol_sha
    )
    expected_new_rows = (
        sum(
            int(value)
            for value in _object(experiment["continuation_sample_counts"], "counts").values()
        )
        * len(BACKENDS)
        * 3
    )
    if len(continuation) != expected_new_rows:
        raise ValueError(
            f"expected {expected_new_rows} continuation rows, found {len(continuation)}"
        )
    frame = pd.concat((prior, continuation), ignore_index=True, sort=False)
    keys = ("backend_scope", "task", "protocol_method", "sample_id")
    if frame.duplicated(list(keys)).any():
        raise ValueError("composite report contains duplicate backend/task/method/sample rows")
    target_counts = _object(experiment["target_sample_counts"], "target counts")
    for backend in BACKENDS:
        for task in TASKS:
            expected = int(target_counts[task])
            for method in METHODS[backend]:
                count = len(
                    frame[
                        (frame.backend_scope == backend)
                        & (frame.task == task)
                        & (frame.protocol_method == method)
                    ]
                )
                if count != expected:
                    raise ValueError(
                        f"composite count {count} != {expected} for {backend}/{task}/{method}"
                    )
    for task in TASKS:
        identities = {
            backend: set(
                str(value)
                for value in frame[
                    (frame.backend_scope == backend)
                    & (frame.task == task)
                    & (frame.protocol_method == "full_kv")
                ].sample_id
            )
            for backend in BACKENDS
        }
        if identities["hf_eager"] != identities["vllm"]:
            raise ValueError(f"composite HF/vLLM sample identity mismatch for {task}")

    summaries: list[dict[str, Any]] = []
    minimum_reduction = float(gates["minimum_per_sample_active_memory_reduction_fraction"])
    for backend_index, backend in enumerate(BACKENDS):
        for task_index, task in enumerate(TASKS):
            summary, task_failures = paired_task_summary(
                frame,
                backend=backend,
                task=task,
                minimum_reduction=minimum_reduction,
                bootstrap_resamples=int(experiment["bootstrap_resamples"]),
                bootstrap_seed=(
                    int(experiment["bootstrap_seed"]) + 10 * backend_index + task_index
                ),
            )
            summaries.append(summary)
            failures.extend(task_failures)
    summary_frame = pd.DataFrame.from_records(summaries)
    report = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "classification": protocol["classification"],
        "source_git_shas": sorted(set(str(value) for value in frame.source_git_sha)),
        "runtime_equivalence_verified": True,
        "candidate_config_sha256": protocol["candidate"]["config_sha256"],
        "protocol_sha256": protocol_sha,
        "reused_samples_per_task": reuse["reused_samples_per_task"],
        "target_sample_counts": target_counts,
        "total_unique_samples": int(sum(int(value) for value in target_counts.values())),
        "primary_row_count": len(frame),
        "latency_is_secondary_nonblocking": True,
        "hf_candidate_scope": sorted(
            set(
                frame[
                    (frame.backend_scope == "hf_eager") & (frame.protocol_method == "mosaickv_full")
                ].effective_method
            )
        ),
        "vllm_candidate_scope": "mosaickv_exact__vllm_requestwide_projection",
        "passed": not failures,
        "failures": failures,
        "task_results": summaries,
    }
    return frame, summary_frame, report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Composite 9,000-sample HF/vLLM quality-memory report",
        "",
        f"Overall gate: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        f"Reused samples per task: {report['reused_samples_per_task']}",
        f"Total unique samples: {report['total_unique_samples']}",
        f"Primary result rows: {report['primary_row_count']}",
        "",
        "| Backend | Task | N | FullKV | Prompt Top-K | MosaicKV | Delta | 95% CI | "
        "Mean smaller | Minimum smaller | Agreement |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in cast("list[dict[str, Any]]", report["task_results"]):
        lines.append(
            f"| {item['backend']} | {item['task']} | {item['sample_count']} | "
            f"{item['fullkv_mean_official_score']:.4f} | "
            f"{item['prompt_topk_mean_official_score']:.4f} | "
            f"{item['mosaickv_mean_official_score']:.4f} | "
            f"{item['paired_mean_score_delta']:+.4f} | "
            f"[{item['paired_bootstrap_score_delta_95_ci_low']:+.4f}, "
            f"{item['paired_bootstrap_score_delta_95_ci_high']:+.4f}] | "
            f"{item['mean_active_memory_reduction_fraction']:.2%} | "
            f"{item['minimum_active_memory_reduction_fraction']:.2%} | "
            f"{item['answer_agreement_with_prompt_topk']:.2%} |"
        )
    if report["failures"]:
        lines.extend(
            ["", "## Gate failures", ""]
            + [f"- {value}" for value in cast("list[str]", report["failures"])]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    frame, summary, report = analyze(
        index_path=args.index.resolve(),
        protocol_path=args.protocol.resolve(),
        repository=repository,
    )
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "paired_rows_9000.parquet", index=False)
    summary.to_parquet(output / "backend_task_summary_9000.parquet", index=False)
    (output / "report_9000.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report_9000.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output)}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
