#!/usr/bin/env python3
"""Consumed-input numerical diagnostic; no task scoring, no threshold tuning."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from evicastkv_campaign import SOURCE, environment, resolved_config

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

CASES = ("ActionPrediction-0035", "ActionPrediction-0014")


@contextmanager
def reduction_policy(allow_reduced):
    import torch

    if type(allow_reduced) is not bool:
        raise ValueError("reduction policy must be an explicit boolean")
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_reduced
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous


def prepare(parent, root):
    source_sha = clean_source(SOURCE)
    lock = json.loads((parent / "protocol_lock.json").read_text())
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    gate = parent / "analysis/correctness/GATE_correctness.json"
    if json.loads(gate.read_text())["status"] != "blocked":
        raise ValueError("this diagnostic is restricted to the preserved failed numerical gate")
    cohort = lock["cohorts"]["milebench"]["validation"]
    if sha(Path(cohort["path"])) != cohort["sha256"]:
        raise ValueError("consumed-input cohort changed")
    if not set(CASES).issubset(cohort["sample_ids"]):
        raise ValueError("diagnostic identities are not in the already consumed cohort")
    write_once(
        root / "diagnostic_protocol.json",
        dict(
            source_sha=source_sha,
            source_root=str(SOURCE),
            parent=str(parent),
            parent_protocol_sha256=sha(parent / "protocol_lock.json"),
            failed_gate_sha256=sha(gate),
            sample_ids=CASES,
            conditions=[True, False],
            steps=16,
            retention_one_tolerance=1e-4,
            unpruned_normwise_tolerance=8 * 2**-10,
            unchanged=[
                "model",
                "processor",
                "inputs",
                "cache_dtype",
                "attention_backend",
                "thresholds",
            ],
            changed_only="torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
            benchmark_labels_used=False,
            reference_annotations_not_passed_to_model=True,
            measurement_type="consumed_input_numerical_diagnostic",
            promotion_requires=("both strict controls pass and one default failure is reproduced"),
            successor_root=str(root.parent / "fox-review-20261028-v5-strict-reduction"),
            documentation="https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms",
        ),
    )


def worker(root):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.evaluation.fox_numerics import unpruned_padding_control
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    protocol = json.loads((root / "diagnostic_protocol.json").read_text())
    clean_source(SOURCE, protocol["source_sha"])
    record = dict(
        source_sha=protocol["source_sha"],
        status="failed",
        conditions=[],
        errors=[],
        protocol_sha256=sha(root / "diagnostic_protocol.json"),
    )
    try:
        if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
            raise RuntimeError("one allocated GPU is required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("cuBLAS deterministic setting must precede Python")
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
        record.update(
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            model=asdict(cfg.model),
            dataset=asdict(cfg.dataset),
            slurm_job_id=os.environ["SLURM_JOB_ID"],
        )
        for allow in protocol["conditions"]:
            with reduction_policy(allow):
                for sid in protocol["sample_ids"]:
                    sample = samples[sid]
                    prepared = adapter.prepare_inputs(
                        build_multimodal_messages(
                            sample.prompt,
                            tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media),
                        )
                    )
                    comparisons = {
                        name: asdict(function(adapter, prepared, max_new_tokens=16))
                        for name, function in (
                            ("ordinary_generate", compare_with_generate),
                            ("reinjection", compare_cache_reinjection),
                        )
                    }
                    prefill = adapter.prefill(
                        prepared,
                        capture_queries=True,
                        capture_attentions=True,
                        attention_query_window=16,
                        reduce_attention_to_kv_heads=True,
                    )
                    controls = unpruned_padding_control(
                        adapter, prefill.state, int(prefill.next_token_id.item()), steps=16
                    )
                    dtype = adapter._legacy_layers(prefill.state.past_key_values)[0][0][0].dtype
                    tolerance = 8 * torch.finfo(dtype).eps
                    passed = all(
                        v["token_agreement"] == 1 and v["maximum_logit_difference"] <= 1e-4
                        for v in comparisons.values()
                    ) and all(item["normalized_linf"] <= tolerance for item in controls)
                    record["conditions"].append(
                        dict(
                            sample_id=sid,
                            allow_fp16_reduced_precision_reduction=allow,
                            cache_dtype=str(dtype),
                            passed=passed,
                            fullkv_parity=comparisons,
                            input_sha256=digest(
                                dict(prompt=sample.prompt, media=[m.sha256 for m in sample.media])
                            ),
                            control=controls,
                            maximum_normalized_error=max(i["normalized_linf"] for i in controls),
                            unchanged_normwise_tolerance=tolerance,
                        )
                    )
                    del prefill, prepared
                    settle(adapter)
        record["status"] = "completed_diagnostic_not_quality_evidence"
    except Exception as error:
        record["errors"].append(
            dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        )
    write_once(root / "diagnostic_results.json", record)
    if record["errors"]:
        raise SystemExit(1)


def analyze(root):
    clean_source(SOURCE)
    protocol = json.loads((root / "diagnostic_protocol.json").read_text())
    path = root / "diagnostic_results.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else dict(conditions=[], errors=["worker did not publish"])
    )
    if path.exists() and (
        result.get("source_sha") != protocol["source_sha"]
        or result.get("protocol_sha256") != sha(root / "diagnostic_protocol.json")
    ):
        result["errors"].append("diagnostic source/protocol drift")
    conditions = result["conditions"]
    strict = [r for r in conditions if r["allow_fp16_reduced_precision_reduction"] is False]
    default = [r for r in conditions if r["allow_fp16_reduced_precision_reduction"] is True]
    promote = (
        not result.get("errors")
        and len(strict) == len(CASES)
        and {r["sample_id"] for r in strict} == set(CASES)
        and all(r["passed"] for r in strict)
        and len(default) == len(CASES)
        and {r["sample_id"] for r in default} == set(CASES)
        and any(not r["passed"] for r in default)
    )
    write_once(
        root / "GATE_reduction_diagnostic.json",
        dict(
            status="passed" if promote else "blocked",
            strict_policy_supported=promote,
            old_results_reclassified=False,
            thresholds_unchanged=True,
            model_or_cache_dtype_changed=False,
            diagnostic_sha256=sha(path) if path.exists() else None,
            source_sha=protocol["source_sha"],
            conditions=[
                {k: v for k, v in r.items() if k not in ("control", "fullkv_parity")}
                for r in conditions
            ],
            errors=result.get("errors", []),
        ),
    )
    if promote:
        from evicastkv_campaign import prepare as prepare_campaign
        from evicastkv_campaign import submit
        from evicastkv_finalize import schedule as schedule_final
        from evicastkv_packet import schedule_watch

        newroot = Path(protocol["successor_root"])
        prepare_campaign(newroot, strict_fp16_reductions=True)
        inventory = Path(protocol["parent"]) / "history-inventory.json"
        if inventory.exists():
            write_once(newroot / "history-inventory.json", json.loads(inventory.read_text()))
        write_once(
            newroot / "arithmetic_revision.json",
            dict(
                diagnostic=str(root),
                gate_sha256=sha(root / "GATE_reduction_diagnostic.json"),
                old_negative_protocol_sha256=protocol["parent_protocol_sha256"],
                old_quality_or_latency_reused=False,
                all_methods_use_strict_reductions=True,
            ),
        )
        submit(newroot, "smoke")
        submission = json.loads((newroot / "jobs/smoke-submission.json").read_text())
        schedule_watch(newroot, "smoke", submission["analysis"])
        packet_submission = json.loads((newroot / "jobs/smoke-packet-submission.json").read_text())
        schedule_final(newroot, newroot, "smoke", packet_submission["job"])


def submit(parent, root):
    prepare(parent, root)
    worker_id = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                f"--output={root}/worker-%j.out",
                f"--error={root}/worker-%j.err",
                str(SOURCE / "mosaickv/slurm/evicastkv_reduction_diagnostic.sbatch"),
                str(SOURCE),
                str(root),
                clean_source(SOURCE),
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
                "--cpus-per-task=4",
                "--mem=16G",
                "--time=00:20:00",
                f"--dependency=afterany:{worker_id}",
                f"--output={root}/analysis-%j.out",
                f"--error={root}/analysis-%j.err",
                str(SOURCE / "mosaickv/slurm/evicastkv_reduction_analysis.sbatch"),
                str(SOURCE),
                str(root),
                clean_source(SOURCE),
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    write_once(
        root / "submission.json",
        dict(worker=worker_id, analysis=cpu, source_sha=clean_source(SOURCE)),
    )
    print(json.dumps(dict(worker=worker_id, analysis=cpu)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.mode == "submit":
        if args.parent is None:
            parser.error("submit requires the preserved --parent run")
        submit(args.parent, args.root)
    elif args.mode == "worker":
        worker(args.root)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
