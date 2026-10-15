#!/usr/bin/env python3
"""One method per immutable shard; resume only verified successful samples.

All artifacts (including failures and warmups) remain append-only. Per-sample
CIDEr artifacts are diagnostics, NOT the campaign's corpus score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import config_sha256, load_config
from mosaickv.evaluation.coco_shared_bytes import (
    control_hashes,
    processor_signature,
    validate_caption_rows,
)
from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
from mosaickv.evaluation.shared_byte_contract import (
    candidate_content_digest,
    load_contract,
    validate_configuration,
)
from mosaickv.evaluation.tasks import default_task_registry


def verify_plan(root, plan_sha, index):
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("plan hash mismatch")
    plan = json.loads((root / "plan.json").read_text())
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, plan["source_sha"])
    if plan["source_root"] != str(source) or plan["task"] != "coco_caption":
        raise ValueError("wrong source or task route")
    if not 0 <= index < len(plan["jobs"]):
        raise ValueError("unregistered job index")
    job = plan["jobs"][index]
    if job["stage"] == "final":
        registration = Path(plan["final_registration"])
        if (
            not plan.get("final_execution_authorized")
            or digest(registration) != plan["final_registration_sha256"]
        ):
            raise ValueError("final stage lacks reviewed hash-bound sample-size registration")
        gate_path = Path(plan["parent_intermediate_report"])
        gate = json.loads(gate_path.read_text())
        if (
            digest(gate_path) != plan["parent_intermediate_report_sha256"]
            or gate.get("status") != "passed"
        ):
            raise ValueError("256-example curves have not passed")
    elif job["stage"] != "validation":
        gate_path = root / "analysis" / "validation" / "report.json"
        gate = json.loads(gate_path.read_text())
        if gate.get("status") != "passed" or gate.get("plan_sha256") != plan_sha:
            raise ValueError("32-example validation has not passed")
        promotion = json.loads((root / "larger-authorization.json").read_text())
        if promotion != {
            "stage": "larger",
            "plan_sha256": plan_sha,
            "validation_sha256": digest(gate_path),
        }:
            raise ValueError("larger stage lacks hash-bound promotion")
    elif not plan["validation_execution_authorized"]:
        raise ValueError("validation disabled")
    contract_path = Path(plan["comparison_contract"])
    if digest(contract_path) != plan["comparison_contract_sha256"]:
        raise ValueError("comparison contract changed")
    contract = load_contract(contract_path)
    if candidate_content_digest(source) != plan["candidate_content_sha256"]:
        raise ValueError("candidate source changed")
    if digest(Path(job["ledger"])) != job["ledger_sha"]:
        raise ValueError("sample ledger changed")
    cfg = load_config(job["config"])
    if config_sha256(cfg) != job["config_sha"] or cfg.method.value != job["method"]:
        raise ValueError("configuration changed")
    validate_configuration(contract, "vl_cache", "coco_caption", cfg)
    ledger = load_paper_native_ledger(
        job["ledger"],
        expected_sha256=job["ledger_sha"],
        task=default_task_registry().resolve("coco_caption"),
        dataset_revision=cfg.dataset.revision,
    )
    if tuple(s.sample_id for s in ledger.samples) != tuple(job["sample_ids"]):
        raise ValueError("shard sample mismatch")
    return source, plan, job, cfg, ledger


def verified_success(sample_root, plan_sha, config_sha):
    """Only completed hash-verified records can be reused, never BF16 history."""
    matches = []
    for path in sorted(sample_root.glob("attempt-*/success.json")):
        record = json.loads(path.read_text())
        if record["plan_sha256"] != plan_sha or record["config_sha"] != config_sha:
            raise ValueError("resume provenance mismatch")
        for name, sha in record["bound_files"].items():
            if digest(Path(name)) != sha:
                raise ValueError("resume artifact changed")
        matches.append(record)
    if len(matches) > 1:
        raise ValueError("duplicate completed sample; do not silently choose a result")
    return matches[0] if matches else None


def completed_sample(plan, job, sid):
    """A final curve may reuse only explicitly hash-bound compatible pilots."""
    plan_sha = plan["_plan_sha"]
    local = verified_success(
        Path(job["output"]) / "samples" / hashlib.sha256(sid.encode()).hexdigest(),
        plan_sha,
        job["config_sha"],
    )
    reuse = plan.get("reuse_records", {}).get(json.dumps([job["method"], job["cap_mib"], sid]))
    if reuse is None:
        return local
    if local is not None:
        raise ValueError("sample executed again despite registered reuse")
    path = Path(reuse["path"])
    if digest(path) != reuse["sha256"]:
        raise ValueError("reused completion record changed")
    record = verified_success(path.parent.parent, reuse["plan_sha256"], job["config_sha"])
    if record is None or record["source_sha"] != plan["source_sha"]:
        raise ValueError("reused source/config/sample incompatible")
    return record


def gpu_snapshot(torch):
    from mosaickv.measurements.telemetry import capture_gpu_environment

    snapshot = asdict(capture_gpu_environment(torch))
    visible = [d for d in snapshot["devices"] if d["visible_to_process"]]
    if (
        snapshot["query_error"]
        or len(visible) != 1
        or snapshot["background_processes"]
        or "A100" not in visible[0]["name"]
    ):
        raise RuntimeError("require one isolated, observable A100")
    return snapshot


def run(args):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.worker_support import record_hf_run, settle
    from mosaickv.hf_cli import _model_kwargs

    _source, plan, job, cfg, ledger = verify_plan(args.root, args.plan_sha, args.index)
    plan["_plan_sha"] = args.plan_sha
    output = Path(job["output"])
    attempt_name = f"attempt-{os.getenv('SLURM_JOB_ID', 'local')}-{time.time_ns()}"
    attempt = output / attempt_name
    started = time.monotonic()
    summary = dict(
        status="running",
        plan_sha256=args.plan_sha,
        index=args.index,
        source_sha=plan["source_sha"],
        failures=[],
        resumed=[],
        completed=[],
        samples=len(ledger.samples),
    )
    try:
        pending = []
        for sample in ledger.samples:
            sample_root = output / "samples" / hashlib.sha256(sample.sample_id.encode()).hexdigest()
            if completed_sample(plan, job, sample.sample_id):
                summary["resumed"].append(sample.sample_id)
            else:
                pending.append((sample, sample_root))
        if not pending:
            summary["status"] = "passed"
            return
        if torch.cuda.device_count() != 1:
            raise RuntimeError("one allocated CUDA device required")
        torch.set_num_threads(12)
        torch.manual_seed(cfg.execution.seed)
        torch.use_deterministic_algorithms(True)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        if processor_signature(adapter.processor) != plan["processor_signature"]:
            raise ValueError("loaded processor differs from frozen preprocessing/tokenization")
        summary["gpu_before"] = gpu_snapshot(torch)
        first = pending[0][0]
        messages = build_multimodal_messages(
            first.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in first.media)
        )
        prepared = adapter.prepare_inputs(messages)
        parity = {}
        for name, check in (
            ("ordinary_generate", compare_with_generate),
            ("retention_one", compare_cache_reinjection),
        ):
            report = check(adapter, prepared, max_new_tokens=plan["parity_tokens"])
            parity[name] = asdict(report)
            if report.token_agreement != 1 or report.maximum_logit_difference > plan["parity_atol"]:
                raise RuntimeError(f"FullKV parity failed: {name}")
        write_once(attempt / "parity.json", parity)
        del prepared
        runtime = HuggingFaceMosaicKVModel(adapter, cfg, trace_directory=attempt / "warmup-traces")
        for warmup in range(plan["warmups"]):
            settle(adapter)
            result = runtime.generate(
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
                dict(answer=result.answer, metrics=asdict(result.metrics)),
            )
        del runtime
        warmup_bindings = {str(p): digest(p) for p in attempt.glob("*.json")}
        for sample, sample_root in pending:
            sample_started = time.monotonic()
            sample_attempt = sample_root / attempt_name
            try:
                # Save completed sample boundaries rather than losing the
                # entire shard on timeout. Never initiate work past 115 min.
                if time.monotonic() - started > 6900:
                    raise RuntimeError("shard deadline: retry remaining samples only")
                selected = dict(
                    schema_version=1,
                    task=ledger.task,
                    dataset_id=ledger.dataset_id,
                    dataset_revision=ledger.dataset_revision,
                    split=ledger.split,
                    samples=[
                        dict(
                            sample_id=sample.sample_id,
                            prompt=sample.prompt,
                            references=list(sample.references),
                            media=[
                                dict(kind=m.kind.value, path=str(m.path), sha256=m.sha256)
                                for m in sample.media
                            ],
                        )
                    ],
                )
                sample_ledger = sample_attempt / "input.json"
                write_once(sample_ledger, selected)
                settle(adapter)
                before = gpu_snapshot(torch)
                device = next(d for d in before["devices"] if d["visible_to_process"])
                gpu_controls = dict(
                    name=device["name"],
                    driver=device["driver_version"],
                    power_limit_w=device["power_limit_w"],
                    torch_version=before["torch_version"],
                    cuda_runtime=before["cuda_runtime"],
                    clock_policy="unlocked_observed_not_modified",
                    visible_gpus=1,
                    background_processes=0,
                )
                record = record_hf_run(
                    cfg,
                    adapter,
                    task="coco_caption",
                    run_id="quality",
                    output_root=sample_attempt,
                    ledger_path=sample_ledger,
                    ledger_sha=digest(sample_ledger),
                    sample_ids=(sample.sample_id,),
                    phase="quality",
                    repetition=0,
                    order=args.index,
                    export_metrics=True,
                    materialized_coco=True,
                )
                after = gpu_snapshot(torch)
                validate_caption_rows(record["rows"], selected["samples"], plan["output_tokens"])
                trace_files = list((sample_attempt / "traces" / "quality").glob("*.json"))
                if len(trace_files) != 1:
                    raise ValueError("missing/ambiguous trace")
                files = [
                    *sample_attempt.rglob("*.json"),
                    *sample_attempt.rglob("*.jsonl"),
                    *sample_attempt.rglob("*.parquet"),
                ]
                record.update(
                    plan_sha256=args.plan_sha,
                    source_sha=plan["source_sha"],
                    config_sha=job["config_sha"],
                    trace=str(trace_files[0]),
                    gpu_before=before,
                    gpu_after=after,
                    bound_files={**warmup_bindings, **{str(p): digest(p) for p in files}},
                    warmups=plan["warmups"],
                    gpu_controls=gpu_controls,
                    control_hashes=control_hashes(
                        cfg, selected["samples"][0], plan["processor_signature"], gpu_controls
                    ),
                    sample_wall_seconds=time.monotonic() - sample_started,
                )
                write_once(sample_attempt / "success.json", record)
                summary["completed"].append(sample.sample_id)
            except Exception as error:
                failure = dict(
                    sample_id=sample.sample_id,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    plan_sha256=args.plan_sha,
                    config_sha=job["config_sha"],
                )
                write_once(sample_attempt / "failure.json", failure)
                summary["failures"].append(failure)
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    break
        summary["status"] = "passed" if not summary["failures"] else "failed"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["worker_wall_seconds"] = time.monotonic() - started
        write_once(attempt / "worker.json", summary)
    if summary["status"] != "passed":
        raise RuntimeError("shard contains failures; preserved for selective retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--index", type=int, required=True)
    run(parser.parse_args())
