#!/usr/bin/env python3
"""Locate native-arithmetic drift on four already consumed numerical inputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

from evicastkv_campaign import SOURCE, environment, resolved_config

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once


def worker(root):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.evaluation.fox_operator_audit import boundary_localization, padding_localization
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    protocol = json.loads((root / "protocol.json").read_text())
    clean_source(SOURCE, protocol["source_sha"])
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("one allocated GPU required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic cuBLAS setting must precede Python")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20271014)
    result = dict(
        cases=[],
        errors=[],
        environment=environment(),
        gpu=asdict(capture_gpu_environment(torch)),
        source_sha=protocol["source_sha"],
    )
    try:
        parent = Path(protocol["numerical_parent"])
        if sha(parent / "results.json") != protocol["numerical_results_sha256"]:
            raise ValueError("consumed numerical evidence changed")
        parent_protocol = json.loads((parent / "protocol.json").read_text())
        for workload, spec in parent_protocol["workloads"].items():
            path = Path(spec["parent"]) / "protocol_lock.json"
            if sha(path) != spec["protocol_sha256"]:
                raise ValueError("original input protocol changed")
            lock = json.loads(path.read_text())
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = lock["arithmetic"][
                "allow_fp16_reduced_precision_reduction"
            ]
            cfg = resolved_config(lock, workload, "full_kv", None)
            cohort = lock["cohorts"][workload]["validation"]
            ledger = load_paper_native_ledger(
                cohort["path"],
                expected_sha256=cohort["sha256"],
                task=default_task_registry().resolve(
                    "milebench" if workload == "milebench" else "coco_caption"
                ),
                dataset_revision=cfg.dataset.revision,
            )
            samples = {s.sample_id: s for s in ledger.samples}
            adapter = load_hf_adapter(
                cfg.model.id,
                revision=cfg.model.revision,
                model_kwargs=_model_kwargs(cfg, local_files_only=True),
                processor_kwargs={"local_files_only": True},
            )
            versions = [(p, p._version, p.dtype) for p in adapter.model.parameters()]
            for sid in spec["sample_ids"]:
                sample = samples[sid]
                prepared = adapter.prepare_inputs(
                    build_multimodal_messages(
                        sample.prompt,
                        tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media),
                    )
                )
                if workload == "milebench":
                    full = adapter.prefill(
                        prepared, capture_queries=False, capture_attentions=False
                    )
                    records = padding_localization(
                        adapter, full.state, int(full.next_token_id.item())
                    )
                    del full
                else:
                    records = [boundary_localization(adapter, prepared)]
                case = dict(
                    workload=workload,
                    sample_id=sid,
                    operators=records,
                    model=asdict(cfg.model),
                    dataset=asdict(cfg.dataset),
                    policy="native_fp16",
                    measurement_type="instrumented_operator_diagnostic_not_latency_or_quality",
                )
                file = root / "cases" / f"{workload}-{sid}.json"
                write_once(file, case)
                result["cases"].append(dict(path=str(file), sha256=sha(file)))
                del prepared, records, case
            if any(p._version != v or p.dtype != dtype for p, v, dtype in versions):
                raise AssertionError("weights changed")
            del versions, adapter
            import gc

            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        result["errors"].append(
            dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        )
    write_once(root / "results.json", result)
    if result["errors"]:
        raise SystemExit(1)


def analyze(root):
    raw = root / "results.json"
    results = (
        json.loads(raw.read_text()) if raw.exists() else dict(cases=[], errors=["worker missing"])
    )
    summaries = []
    for binding in results["cases"]:
        if sha(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("operator captures changed")
        case = json.loads(Path(binding["path"]).read_text())
        rows = case["operators"]
        failures = (
            [r for r in rows if r["logits"]["normalized_linf"] > 8 * 2**-10]
            if case["workload"] == "milebench"
            else [r for r in rows if r["logits"]["absolute_linf"] > 1e-4]
        )
        summaries.append(
            dict(
                workload=case["workload"],
                sample_id=case["sample_id"],
                first_nonidentical_operator_by_step=[
                    r["first_nonidentical_operator"] for r in rows
                ],
                first_registered_logit_failure_step=failures[0].get("step", 0)
                if failures
                else None,
                maximum_normalized_logit_error=max(r["logits"]["normalized_linf"] for r in rows),
                maximum_absolute_logit_error=max(r["logits"]["absolute_linf"] for r in rows),
            )
        )
    write_once(
        root / "analysis.json",
        dict(
            status="diagnosis_complete_not_promotion"
            if len(summaries) == 4 and not results["errors"]
            else "blocked_incomplete",
            cases=summaries,
            errors=results["errors"],
            thresholds_changed=False,
            no_arithmetic_or_weight_change=True,
            fresh_confirmation_authorized=False,
            notes=(
                "First nonidentical operator is localization, "
                "not a causal proof or a new tolerance."
            ),
        ),
    )


def submit(root, numerical_parent):
    revision = clean_source(SOURCE)
    write_once(
        root / "protocol.json",
        dict(
            source_sha=revision,
            numerical_parent=str(numerical_parent),
            numerical_results_sha256=sha(numerical_parent / "results.json"),
            native_policy_only=True,
            fresh_inputs=False,
            no_answers_used=True,
            padding_steps=16,
            A_B_original_absolute_tolerance=1e-4,
            padding_normwise_tolerance=8 * 2**-10,
            confirmation_authorized=False,
        ),
    )
    (root / "slurm").mkdir()
    base = ["sbatch", "--parsable", "--account=bii_dsc_community"]
    gpu = (
        subprocess.check_output(
            [
                *base,
                f"--output={root}/slurm/worker-%j.out",
                f"--error={root}/slurm/worker-%j.err",
                str(SOURCE / "mosaickv/slurm/evicastkv_operator_audit.sbatch"),
                str(SOURCE),
                str(root),
                revision,
                "worker",
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    cpu = (
        subprocess.check_output(
            [
                *base,
                "--partition=bii",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:05:00",
                f"--dependency=afterany:{gpu}",
                f"--output={root}/slurm/analysis-%j.out",
                f"--error={root}/slurm/analysis-%j.err",
                str(SOURCE / "mosaickv/slurm/evicastkv_operator_audit_analysis.sbatch"),
                str(SOURCE),
                str(root),
                revision,
                "analyze",
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    write_once(root / "submission.json", dict(worker=gpu, analysis=cpu, source_sha=revision))
    print(json.dumps(dict(worker=gpu, analysis=cpu)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("submit", "worker", "analyze"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--numerical-parent", type=Path)
    a = p.parse_args()
    if a.mode == "submit":
        submit(a.root, a.numerical_parent)
    else:
        (worker if a.mode == "worker" else analyze)(a.root)
