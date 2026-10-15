#!/usr/bin/env python3
"""Run one retrospective output-preserving sweep arm on one isolated A100."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from prepare_output_preserving_sweep import source_digest
from run_coco_shared_bytes import gpu_snapshot
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import config_sha256, load_config
from mosaickv.evaluation.coco_shared_bytes import processor_signature, validate_caption_rows
from mosaickv.evaluation.paper_native_lanes import load_paper_native_ledger
from mosaickv.evaluation.tasks import default_task_registry
from mosaickv.evaluation.worker_support import record_hf_run, settle


def run(args: argparse.Namespace) -> None:
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.backends.hf_runtime import (
        HuggingFaceMosaicKVModel,
        compare_runtime_retention_one,
    )
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload
    from mosaickv.hf_cli import _model_kwargs

    root = args.root.resolve()
    if digest(root / "plan.json") != args.plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads((root / "plan.json").read_text())
    if plan["classification"] not in {
        "retrospective_development_only_not_paper_evidence",
        "preregistered_fresh_development_validation_not_paper_evidence",
    }:
        raise ValueError("wrong experiment classification")
    source = Path(plan["source_root"])
    if source_digest(source) != plan["source_content_sha256"]:
        raise ValueError("development source snapshot changed")
    observed_sha = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_sha != plan["source_base_git_sha"]:
        raise ValueError("source Git SHA differs from the preregistered plan")
    observed_dirty = bool(
        subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if observed_dirty != plan["source_worktree_dirty"]:
        raise ValueError("source worktree cleanliness differs from the preregistered plan")
    parent_plan = Path(plan["parent_plan"])
    if digest(parent_plan) != plan["parent_plan_sha256"]:
        raise ValueError("reviewed parent plan changed")
    parent_signature = json.loads(parent_plan.read_text()).get("processor_signature")
    if parent_signature != plan["processor_signature"]:
        raise ValueError("processor signature differs from reviewed parent plan")
    if not 0 <= args.index < len(plan["jobs"]):
        raise ValueError("unregistered array index")
    job = plan["jobs"][args.index]
    config = load_config(job["config"])
    if config_sha256(config) != job["config_sha256"]:
        raise ValueError("config hash mismatch")
    ledger_path = Path(plan["ledger"])
    if digest(ledger_path) != plan["ledger_sha256"]:
        raise ValueError("ledger hash mismatch")
    ledger = load_paper_native_ledger(
        ledger_path,
        expected_sha256=plan["ledger_sha256"],
        task=default_task_registry().resolve("coco_caption"),
        dataset_revision=plan["dataset_revision"],
    )
    if tuple(s.sample_id for s in ledger.samples) != tuple(plan["sample_ids"]):
        raise ValueError("sample identities/order changed")
    output = Path(job["output"])
    attempt = output / f"attempt-{os.getenv('SLURM_JOB_ID', 'local')}-{time.time_ns()}"
    summary = {
        "status": "running",
        "classification": plan["classification"],
        "heldout_claim": False,
        "plan_sha256": args.plan_sha256,
        "source_base_git_sha": plan["source_base_git_sha"],
        "source_content_sha256": plan["source_content_sha256"],
        "source_worktree_dirty": observed_dirty,
        "index": args.index,
        "job": job,
    }
    started = time.monotonic()
    try:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one allocated CUDA device is required")
        torch.set_num_threads(plan["cpus_per_worker"])
        torch.manual_seed(config.execution.seed)
        torch.use_deterministic_algorithms(True)
        adapter = load_hf_adapter(
            config.model.id,
            revision=config.model.revision,
            model_kwargs=_model_kwargs(config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        observed_processor = processor_signature(adapter.processor)
        if observed_processor != plan["processor_signature"]:
            raise ValueError("processor/preprocessing/tokenization differs from reviewed ledger")
        summary["gpu_before"] = gpu_snapshot(torch)
        first = ledger.samples[0]
        messages = build_multimodal_messages(
            first.prompt,
            tuple(MediaItem(media.kind, _media_payload(media)) for media in first.media),
        )
        if args.index == 0:
            identity = load_config(plan["retention_one_config"])
            if config_sha256(identity) != plan["retention_one_config_sha256"]:
                raise ValueError("retention-one configuration changed")
            prepared = adapter.prepare_inputs(messages)
            parity = compare_runtime_retention_one(adapter, prepared, identity)
            if parity.token_agreement != 1 or parity.maximum_logit_difference > 1e-5:
                raise RuntimeError("real-checkpoint retention-one parity failed")
            write_once(attempt / "retention-one-parity.json", asdict(parity))
            del prepared
        runtime = HuggingFaceMosaicKVModel(
            adapter, config, trace_directory=attempt / "warmup-traces"
        )
        for warmup in range(plan["warmups"]):
            settle(adapter)
            measured = runtime.generate(
                EvaluationRequest(
                    run_id=f"warmup-{warmup}",
                    sample_id=first.sample_id,
                    task="coco_caption",
                    messages=messages,
                    generation_kwargs={},
                )
            )
            write_once(
                attempt / f"warmup-{warmup}.json",
                {"answer": measured.answer, "metrics": asdict(measured.metrics)},
            )
        del runtime
        settle(adapter)
        record = record_hf_run(
            config,
            adapter,
            task="coco_caption",
            run_id=f"quality-{job['label']}",
            output_root=attempt / "measurement",
            ledger_path=ledger_path,
            ledger_sha=plan["ledger_sha256"],
            sample_ids=tuple(plan["sample_ids"]),
            phase="quality",
            repetition=0,
            order=args.index,
            export_metrics=True,
            materialized_coco=True,
        )
        validate_caption_rows(record["rows"], json.loads(ledger_path.read_text())["samples"], 64)
        traces = sorted(
            (attempt / "measurement/traces/quality-output_preserving_sweep").glob("*.json")
        )
        if not traces:
            traces = sorted((attempt / "measurement/traces").rglob("*.json"))
        trace_by_sample = {}
        for path in traces:
            payload = json.loads(path.read_text())
            if payload.get("status") == "completed":
                trace_by_sample[payload["sample_id"]] = str(path)
        if set(trace_by_sample) != set(plan["sample_ids"]):
            raise ValueError("missing or ambiguous completed traces")
        for path in map(Path, trace_by_sample.values()):
            trace = json.loads(path.read_text())
            effective = trace.get("effective_method", trace["method"])
            if effective != job["effective_method"]:
                raise ValueError(f"effective method mismatch: {effective}")
            if job["label"].startswith("output_preserving"):
                audit = trace.get("output_reconstruction")
                if not audit or audit["final_error"] > audit["initial_error"] + 1e-6:
                    raise ValueError("missing/increasing reconstruction objective")
        summary.update(
            status="passed",
            record=record,
            traces=trace_by_sample,
            gpu_after=gpu_snapshot(torch),
        )
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["worker_wall_seconds"] = time.monotonic() - started
        write_once(attempt / "worker.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--index", type=int, required=True)
    run(parser.parse_args())
