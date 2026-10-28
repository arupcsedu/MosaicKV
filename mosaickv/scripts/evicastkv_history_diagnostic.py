#!/usr/bin/env python3
"""Same consumed inputs: local roundoff versus accumulated cache-history drift."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

from evicastkv_campaign import SOURCE, environment, resolved_config
from evicastkv_reduction_diagnostic import CASES, reduction_policy

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once


def worker(root):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.evaluation.fox_history_control import diagnose_history
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    protocol = json.loads((root / "protocol.json").read_text())
    clean_source(SOURCE, protocol["source_sha"])
    result = dict(
        status="failed",
        cases=[],
        errors=[],
        source_sha=protocol["source_sha"],
        protocol_sha256=sha(root / "protocol.json"),
    )
    try:
        if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
            raise RuntimeError("one allocated GPU required")
        gres = subprocess.check_output(
            ["squeue", "-h", "-u", os.environ["USER"], "-t", "RUNNING,COMPLETING", "-o", "%b"],
            text=True,
        )
        allocated = sum(
            int(m.group(1))
            for line in gres.splitlines()
            if (m := re.search(r"gpu(?::[a-zA-Z0-9_-]+)?:(\d+)", line))
        )
        if allocated > 6:
            raise RuntimeError("aggregate user GPU ceiling six exceeded; diagnostic refuses work")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("cuBLAS setting must precede Python")
        parent = Path(protocol["parent"])
        lock = json.loads((parent / "protocol_lock.json").read_text())
        if sha(parent / "protocol_lock.json") != protocol["parent_protocol_sha256"]:
            raise ValueError("parent protocol changed")
        cfg = resolved_config(lock, "milebench", "full_kv", None)
        cohort = lock["cohorts"]["milebench"]["validation"]
        ledger = load_paper_native_ledger(
            cohort["path"],
            expected_sha256=cohort["sha256"],
            task=default_task_registry().resolve("milebench"),
            dataset_revision=cfg.dataset.revision,
        )
        samples = {s.sample_id: s for s in ledger.samples if s.sample_id in CASES}
        torch.set_num_threads(12)
        torch.manual_seed(lock["seed"])
        torch.use_deterministic_algorithms(True)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        result.update(
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            aggregate_user_gpus_at_entry=allocated,
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            model=asdict(cfg.model),
            dataset=asdict(cfg.dataset),
        )
        with reduction_policy(True):
            for sid in CASES:
                sample = samples[sid]
                prepared = adapter.prepare_inputs(
                    build_multimodal_messages(
                        sample.prompt,
                        tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media),
                    )
                )
                prefill = adapter.prefill(
                    prepared,
                    capture_queries=True,
                    capture_attentions=True,
                    attention_query_window=16,
                    reduce_attention_to_kv_heads=True,
                )
                record = diagnose_history(
                    adapter, prefill.state, int(prefill.next_token_id.item()), steps=16
                )
                result["cases"].append(
                    dict(
                        sample_id=sid,
                        input_sha256=digest(
                            dict(prompt=sample.prompt, media=[m.sha256 for m in sample.media])
                        ),
                        **record,
                    )
                )
                del prefill, prepared
                settle(adapter)
        result["status"] = "completed_numerical_diagnostic_not_benchmark"
    except Exception as error:
        result["errors"].append(
            dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        )
    write_once(root / "results.json", result)
    if result["errors"]:
        raise SystemExit(1)


def analyze(root):
    clean_source(SOURCE)
    path = root / "results.json"
    result = (
        json.loads(path.read_text()) if path.exists() else dict(cases=[], errors=["worker missing"])
    )
    complete = not result["errors"] and {c["sample_id"] for c in result["cases"]} == set(CASES)
    summaries = []
    for case in result["cases"]:
        if len(case["records"]) != 16:
            complete = False
        maxima = {
            name: max(r[name + "_normalized_error"] for r in case["records"])
            for name in ("persistent", "rebased", "synchronized")
        }
        bc = max(r["rebased_synchronized_max_absolute"] for r in case["records"])
        diagnosis = (
            "original_persistent_failure_not_reproduced_in_this_diagnostic"
            if maxima["persistent"] <= 8 * 2**-10
            else "generated_history_roundoff_contributes"
            if maxima["persistent"] > 8 * 2**-10
            and maxima["rebased"] <= 8 * 2**-10
            and maxima["synchronized"] <= 8 * 2**-10
            and bc <= 1e-4
            else "within_step_geometry_or_other_roundoff_also_exceeds_contract"
        )
        summaries.append(
            dict(
                sample_id=case["sample_id"],
                maxima=maxima,
                rebased_synchronized_max_absolute=bc,
                diagnosis=diagnosis,
            )
        )
    write_once(
        root / "analysis.json",
        dict(
            status="diagnosed" if complete else "incomplete",
            cases=summaries,
            errors=result["errors"],
            results_sha256=sha(path) if path.exists() else None,
            numerical_tolerances_unchanged=True,
            model_cache_dtype_changed=False,
            original_failed_gates_reclassified=False,
            milebench_quality_authorized=False,
            evaluation_only_history_replacement_not_a_runtime_method=True,
        ),
    )
    with (root / "report.md").open("x") as stream:
        stream.write("# Consumed-input cache-history diagnostic\n\n")
        stream.write(
            "This is not quality evidence, a deployable repair, or a passed original gate.\n\n"
        )
        for item in summaries:
            stream.write(
                f"- {item['sample_id']}: {item['diagnosis']}; normalized errors {item['maxima']}; "
                "rebased/synchronized absolute difference "
                f"{item['rebased_synchronized_max_absolute']}.\n"
            )
        stream.write(
            "\nOriginal padding failures remain negative. "
            "No MileBench quality or fresh-cohort job is submitted.\n"
        )


def submit(parent, root):
    source_sha = clean_source(SOURCE)
    write_once(
        root / "protocol.json",
        dict(
            source_sha=source_sha,
            parent=str(parent),
            parent_protocol_sha256=sha(parent / "protocol_lock.json"),
            sample_ids=CASES,
            conditions=[
                "persistent_padding",
                "rebased_identical_history",
                "synchronized_identical_history",
            ],
            steps=16,
            normwise_tolerance=8 * 2**-10,
            between_same_history_absolute_tolerance=1e-4,
            model_and_cache_precision="fp16",
            reduced_precision_reductions=True,
            evaluator_history_replacement_only=True,
            online_method_modified=False,
            reference_annotations_used=False,
            quality_successor_authorized=False,
            resources=dict(
                gpus=1, cpus=12, wall_minutes=20, main_campaign_ceiling=4, aggregate_user_ceiling=6
            ),
        ),
    )
    gpu = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                f"--output={root}/worker-%j.out",
                f"--error={root}/worker-%j.err",
                str(SOURCE / "mosaickv/slurm/evicastkv_history_diagnostic.sbatch"),
                str(SOURCE),
                str(root),
                source_sha,
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    cpu = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                "--partition=bii",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:05:00",
                f"--dependency=afterany:{gpu}",
                f"--output={root}/analysis-%j.out",
                f"--error={root}/analysis-%j.err",
                str(SOURCE / "mosaickv/slurm/evicastkv_history_analysis.sbatch"),
                str(SOURCE),
                str(root),
                source_sha,
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    write_once(root / "submission.json", dict(worker=gpu, analysis=cpu, source_sha=source_sha))
    print(json.dumps(dict(worker=gpu, analysis=cpu)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.mode == "submit":
        submit(args.parent, args.root)
    elif args.mode == "worker":
        worker(args.root)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
