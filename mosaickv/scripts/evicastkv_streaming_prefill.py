#!/usr/bin/env python3
"""Bounded same-consumed-input streaming prefill parity/transaction audit."""

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import evicastkv_shared_source_runtime as control

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
REFERENCE = Path("/scratch/djy8hg/runs/evicastkv/common-exact-history-consumed-20261101-v2")
previous = control.previous


def prepare(root, *, timing=False):
    if root.exists():
        raise FileExistsError("new immutable streaming-prefill root required")
    old = previous.read(REFERENCE / "plan.json")
    decision = previous.read(REFERENCE / "analysis/decision.json")
    if (
        not decision["complete"]
        or decision["failures"]
        or not decision["common_exact_history_transport_gate"]
        or not decision["audited_history_memory_gate"]
    ):
        raise ValueError("complete common exact-history evidence required")
    bound = dict(old["bound_files"])
    for path in (REFERENCE / "plan.json", REFERENCE / "analysis/decision.json"):
        bound[str(path)] = sha(path)
    jobs = []
    for job in old["jobs"]:
        if job["evaluation_variant"] != "source3_direct_fp32":
            continue
        jobs.append(dict(job, index=len(jobs), reference_worker=job["index"]))
        folder = REFERENCE / f"workers/{job['index']}"
        producer = previous.read(folder / "result.json")
        if producer["status"] != "completed" or producer["failures"]:
            raise ValueError("complete direct-source producer required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in producer["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("source producer changed")
            bound[str(folder / name)] = digest
    if len(jobs) != 2:
        raise ValueError("only the same two consumed inputs permitted")
    path = SOURCE / "mosaickv/docs/streaming_prefill_20261101.md"
    bound[str(path)] = sha(path)
    if timing:
        audit_root = Path("/scratch/djy8hg/runs/evicastkv/streaming-prefill-consumed-20261101-v1")
        check = previous.read(audit_root / "analysis/decision.json")
        if (
            not check["complete"]
            or check["failures"]
            or not check["streaming_payload_and_output_gate"]
            or not check["lower_observed_cache_transaction_peak_than_native_prefill_lower_bound"]
        ):
            raise ValueError("complete streaming payload/logit/teacher/transaction gate required")
        for path in (audit_root / "plan.json", audit_root / "analysis/decision.json"):
            bound[str(path)] = sha(path)
        for i in range(2):
            folder = audit_root / f"workers/{i}"
            record = previous.read(folder / "result.json")
            bound[str(folder / "result.json")] = sha(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("streaming audit evidence changed")
                bound[str(folder / name)] = digest
        path = SOURCE / "mosaickv/docs/streaming_prefill_timing_20261101.md"
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            jobs=jobs,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_streaming_source_paired_endtoend_timing"
            if timing
            else "consumed_streaming_prefill_transaction_diagnostic",
            reference_root=str(REFERENCE),
            maximum_simultaneous_workers=2,
            maximum_worker_minutes=45,
            variants=["lossless_full_prompt", "streaming_direct_source"],
            evaluation_variants=["lossless_full_prompt", "streaming_direct_source"],
            no_baseline_comparison_authorized=True,
            no_fresh_or_large_campaign_authorized=True,
            paper_campaign_authorized=False,
            independent_quality_N=0,
            original_source_encoder_and_memberships_must_match=True,
            no_model_weight_or_position_change=True,
            no_offload=True,
            instrumentation_not_production_latency=True,
            paired_streaming_timing=timing,
            timing_warmups=5 if timing else 0,
            timing_repetitions=20 if timing else 0,
        ),
    )
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.direct_source_constructor import DirectSourcePolicy, construct
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.sequential_source_history import paired_run
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.evaluation.source_teacher_protocol import load_targets
    from mosaickv.evaluation.storage_teacher_forcing import score_formats
    from mosaickv.evaluation.streaming_prefill import (
        RetiringPrefillCache,
        SealAfterAttention,
        lossless_kv_encoder,
    )
    from mosaickv.evaluation.streaming_source_constructor import StreamingDirectSource
    from mosaickv.evaluation.text_source_storage import TextSourceCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    payload = previous.read(job["inputs"][0])
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw.jsonl").touch(exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    cache, arm = None, "setup"
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one isolated A100 and deterministic CUDA workspace required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
        if (
            cfg.model.id,
            cfg.model.revision,
            cfg.model.precision.value,
            cfg.execution.attention_implementation,
        ) != (
            "llava-hf/llava-1.5-7b-hf",
            "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            "bf16",
            "eager",
        ):
            raise NotImplementedError("pinned LLaVA BF16/eager geometry only")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        ref_folder = REFERENCE / f"workers/{job['reference_worker']}"
        cap = previous.read(ref_folder / "construction.json")["cap_bytes"]
        profile = tuple(previous.read(Path(plan["profile_path"])))
        with (
            SourceCapture(adapter) as source,
            TextSourceCapture(adapter, prepared) as text,
            PhaseCapture(adapter) as phase,
        ):
            reference = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        native, _ = adapter._legacy_layers(reference.state.past_key_values)
        original_native_bytes = backing_bytes(tuple(t for pair in native for t in pair))
        with torch.inference_mode():
            packets, checks = construct(
                adapter,
                prepared,
                reference,
                source,
                text,
                phase,
                profile,
                cap,
                policy=DirectSourcePolicy(head_group=4, lossless_minimum=False),
            )
        packet_hash = control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
        packet_bytes = backing_bytes(tuple(t for p in packets for t in p.tensors()))
        memberships = checks["source_memberships"]
        first_logits = reference.logits.detach().cpu()
        del native, packets, checks, reference, source, text, phase
        gc.collect()
        torch.cuda.synchronize()
        write_once(
            out / "offline_constructor_control.json",
            dict(
                payload_sha256=packet_hash,
                physical_bytes=packet_bytes,
                original_native_prefill_KV_bytes=original_native_bytes,
                source_memberships=memberships,
                reference_build_is_not_the_streaming_memory_measurement=True,
            ),
        )
        full = torch.load(
            Path(plan["parent"]) / f"workers/{job['input_index']}/logits.pt",
            map_location="cpu",
            weights_only=True,
        )["0-full_kv-ordinary"]
        source_expected = torch.load(
            ref_folder / "source3_direct_fp32-logits.pt",
            map_location="cpu",
            weights_only=True,
        )["greedy"]
        manifest = previous.read(ref_folder / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            evaluator_commit=plan["source_sha"],
            baseline_commit="not_used",
            environment=control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            variant="streamed_execution_of_frozen_shared_source_precision_variant",
            no_baseline_comparison=True,
            independent_quality_N=0,
            instrumentation_not_production_latency=True,
            config_sha256=hashlib.sha256(
                json.dumps(
                    dict(parent_config_sha=manifest["config_sha256"], plan_sha=expected),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            actual_timing_protocol=dict(
                warmups=5 if plan.get("paired_streaming_timing", False) else 0,
                repetitions=20 if plan.get("paired_streaming_timing", False) else 1,
                order="randomized_pairs" if plan.get("paired_streaming_timing", False) else "fixed",
                tensor_result_audit=not plan.get("paired_streaming_timing", False),
                not_paper_latency=True,
            ),
        )
        write_once(out / "manifest.json", manifest)
        if plan.get("paired_streaming_timing", False):
            from mosaickv.evaluation.streaming_prefill_timing import order, request, summary

            messages = previous.load_messages(payload)
            trials, seen = [], set()
            for entry in order(plan["seed"] + index):
                arm = entry["variant"]
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                values, measured, formats = request(
                    adapter, messages, profile, cap, entry["variant"]
                )
                if not torch.equal(values, source_expected):
                    raise RuntimeError("timed source execution changed a 128-logit trajectory bit")
                if measured["initial_encoded_bytes"] != packet_bytes:
                    raise RuntimeError("timed source execution changed cache backing bytes")
                if entry["variant"] not in seen:
                    if (
                        control.digest_tensors(tuple(t for p in formats for t in p.tensors()))
                        != packet_hash
                    ):
                        raise RuntimeError("minimal-summary capture changed the frozen payload")
                    seen.add(entry["variant"])
                row = dict(
                    **entry,
                    **measured,
                    status="completed",
                    logits_bitwise=True,
                    sample_id=payload["sample_id"],
                    task=payload["task"],
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                    config_sha256=manifest["config_sha256"],
                    manifest_path=str(out / "manifest.json"),
                    measurement_type="ablation_measured",
                    torch_peak_allocated=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved=torch.cuda.max_memory_reserved(),
                )
                with (out / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                trials.append(row)
                del values, formats, measured, row
            write_once(out / "timing_summary.json", summary(trials, seed=plan["seed"] + index))
            result["status"] = "completed"
            return
        for arm in plan["evaluation_variants"]:
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            cache = RetiringPrefillCache(len(adapter._language_layers()), audit=True)
            encoder = (
                lossless_kv_encoder
                if arm == "lossless_full_prompt"
                else StreamingDirectSource(adapter, prepared, cache, profile)
            )
            fresh = replace(
                prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache)
            )
            started = time.monotonic()
            with SealAfterAttention(adapter, cache, encoder):
                pref = adapter.prefill(fresh, capture_queries=False, capture_attentions=False)
            if arm == "streaming_direct_source":
                encoder.finish()
            torch.cuda.synchronize()
            prefill_wall = time.monotonic() - started
            if not torch.equal(pref.logits.cpu(), first_logits):
                raise RuntimeError("streaming retirement changed original prefill logits")
            formats = cache.formats()
            if arm == "streaming_direct_source":
                if tuple(encoder.memberships) != memberships:
                    raise RuntimeError("streaming changed the global allocation or selections")
                if (
                    control.digest_tensors(tuple(t for p in formats for t in p.tensors()))
                    != packet_hash
                ):
                    raise RuntimeError("streaming changed an encoded payload bit")
                if backing_bytes(tuple(t for p in formats for t in p.tensors())) != packet_bytes:
                    raise RuntimeError("streaming changed initial backing/sharing")
            gc.collect()
            storage = cache.record()
            if storage["original_KV_tensor_objects_alive"]:
                raise RuntimeError("retired native prefill tensors remain alive")
            write_once(out / (arm + "-prefill.json"), storage)
            runtime = replace(pref, state=replace(pref.state, past_key_values=None))
            values, details = paired_run(
                adapter,
                runtime,
                formats,
                tokens=128,
                sealed=False,
                audit_history=True,
            )
            reference_logits = full if arm == "lossless_full_prompt" else source_expected
            if not torch.equal(values, reference_logits):
                raise RuntimeError("streamed formats changed the 128-token reference trajectory")
            torch.save(values, out / (arm + "-greedy.pt"))
            targets, provenance = load_targets(plan, payload, adapter.processor.tokenizer)
            teacher, logits = score_formats(
                adapter,
                runtime,
                formats,
                targets,
                maximum=1024,
                sequential_source_history=True,
                audit_history=True,
                return_logits=True,
            )
            prior_rows = [
                json.loads(x) for x in (ref_folder / "raw.jsonl").read_text().splitlines()
            ]
            prior = next(r for r in prior_rows if r["variant"] == "source3_direct_fp32")
            if arm == "lossless_full_prompt":
                prior_teacher = previous.read(ref_folder / "teacher_control.json")["ordinary"]
                old_logits = torch.load(
                    ref_folder / "teacher_fullkv_logits.pt", map_location="cpu", weights_only=True
                )
                teacher_bitwise = torch.equal(logits, old_logits)
                del old_logits
            else:
                prior_teacher = prior["teacher_forced"]
                teacher_bitwise = (
                    control.digest_tensors((logits,))
                    == prior_teacher["storage"]["raw_history_control"][
                        "source_history_logit_sha256"
                    ]
                )
            if not teacher_bitwise or any(
                teacher[k] != prior_teacher[k]
                for k in (
                    "nll_sum",
                    "per_token_nll",
                    "conditional_greedy_tokens",
                    "evaluated_token_count",
                )
            ):
                raise RuntimeError("streaming changed complete teacher-forcing logits or NLL")
            row = dict(
                status="completed",
                variant=arm,
                sample_id=payload["sample_id"],
                task=payload["task"],
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                manifest_path=str(out / "manifest.json"),
                config_sha256=manifest["config_sha256"],
                measurement_type="ablation_measured",
                prefill_storage=storage,
                generation_storage=details["storage"],
                teacher_forced=teacher,
                teacher_provenance=provenance,
                prefill_seconds=prefill_wall,
                prefill_logits_bitwise=True,
                generation_128_logits_bitwise=True,
                complete_teacher_logits_bitwise=True,
                original_native_prefill_KV_lower_bound=original_native_bytes,
                cache_transaction_observed_peak=max(
                    storage["observed_prefill_cache_peak_bytes"],
                    details["storage"]["peak_active_kv_storage_bytes"],
                ),
                whole_request_gpu_peak_certified=False,
                torch_peak_allocated=torch.cuda.max_memory_allocated(),
                torch_peak_reserved=torch.cuda.max_memory_reserved(),
                no_baseline_or_independent_quality_claim=True,
            )
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            del (
                teacher,
                logits,
                values,
                details,
                runtime,
                pref,
                fresh,
                formats,
                cache,
                encoder,
                targets,
            )
            del provenance, prior_rows, prior, prior_teacher, row
            cache = None
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        status="failed",
                        variant=arm,
                        sample_id=payload["sample_id"],
                        task=payload["task"],
                        error=result["failures"][-1],
                        source_sha=plan["source_sha"],
                        plan_sha256=expected,
                        partial_cache=None
                        if cache is None
                        else cache.record(require_complete=False),
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)
    if result["status"] != "completed":
        raise SystemExit(1)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    rows, failures = [], []
    for i in range(2):
        try:
            folder = root / f"workers/{i}"
            producer = previous.read(folder / "result.json")
            if producer["source_sha"] != plan["source_sha"] or producer["plan_sha256"] != expected:
                raise ValueError("streaming producer provenance changed")
            for name, digest in producer["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("streaming artifact changed")
            failures.extend(producer["failures"])
            rows.extend(json.loads(x) for x in (folder / "raw.jsonl").read_text().splitlines())
        except (KeyError, ValueError, OSError) as exc:
            failures.append(str(exc))
    timing = plan.get("paired_streaming_timing", False)
    complete = (
        len(rows) == (100 if timing else 4)
        and not failures
        and all(r["status"] == "completed" for r in rows)
    )
    if timing:
        summaries = []
        try:
            summaries = [previous.read(root / f"workers/{i}/timing_summary.json") for i in range(2)]
        except (KeyError, ValueError, OSError) as exc:
            complete = False
            failures.append(str(exc))
        write_once(
            root / "analysis/decision.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                complete=complete,
                failures=failures,
                rows=rows,
                summaries=summaries,
                own_parent_similar_time_gate=complete
                and all(s["similar_time_gate_upper_ratio_1_05"] for s in summaries),
                independent_quality_N=0,
                no_baseline_or_paper_superiority_claim=True,
                memory_evidence_must_use_separate_audited_canary=True,
            ),
        )
        return
    write_once(
        root / "analysis/decision.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            complete=complete,
            failures=failures,
            rows=rows,
            streaming_payload_and_output_gate=complete,
            lower_observed_cache_transaction_peak_than_native_prefill_lower_bound=complete
            and all(
                r["cache_transaction_observed_peak"] < r["original_native_prefill_KV_lower_bound"]
                for r in rows
                if r["variant"] == "streaming_direct_source"
            ),
            whole_request_gpu_peak_certified=False,
            independent_quality_N=0,
            baseline_superiority_established=False,
            fresh_or_large_campaign_authorized=False,
        ),
    )


def submit(root, expected):
    original = control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_streaming_prefill.sbatch")
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(control.parent.subprocess, "check_output", scoped):
        control.parent.submit(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["prepare", "prepare_timing", "cpu", "submit", "worker", "analyze"]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode in ("prepare", "prepare_timing"):
        prepare(args.root, timing=args.mode == "prepare_timing")
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
