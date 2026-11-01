#!/usr/bin/env python3
"""Same-consumed-input, all-method streaming construction identity gate."""

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


def method_of(job):
    variant = job["evaluation_variant"]
    if variant == "source3_direct_fp32":
        return "mosaickv_exact"
    if variant.startswith("baseline_") and variant.endswith("__fp32"):
        method = variant.removeprefix("baseline_").removesuffix("__fp32")
        if method in (
            "prompt_attention_topk",
            "prefixkv_reimpl",
            "vl_cache_reimpl",
            "lookm_reimpl",
        ):
            return method
    raise ValueError("only the five frozen FP32-header precision variants permitted")


def row_is_complete(row):
    if row.get("status") != "completed":
        return False
    flags = (
        "payload_and_memberships_identical",
        "generation_128_logits_bitwise",
        "complete_teacher_logits_bitwise",
        "prefill_logits_bitwise",
    )
    if not all(row.get(key) is True for key in flags):
        return False
    prefill = row.get("prefill_storage", {})
    generation = row.get("generation_storage", {})
    return (
        prefill.get("original_KV_tensor_objects_alive") == 0
        and prefill.get("CPU_offload_or_residual_bytes") == 0
        and prefill.get("final_prefill_cache_bytes", 0) > 0
        and prefill.get("final_prefill_cache_bytes")
        == generation.get("initial_active_kv_storage_bytes")
        and prefill.get("observed_prefill_cache_peak_bytes", 0)
        >= prefill["final_prefill_cache_bytes"]
        and row.get("teacher_forced", {}).get("evaluated_token_count", 0) > 0
    )


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable common-streaming diagnostic root required")
    old = previous.read(REFERENCE / "plan.json")
    check = previous.read(REFERENCE / "analysis/decision.json")
    if (
        not check["complete"]
        or check["failures"]
        or not check["common_exact_history_transport_gate"]
    ):
        raise ValueError("complete common lossless-history transport gate required")
    bound = dict(old["bound_files"])
    for path in (
        REFERENCE / "plan.json",
        REFERENCE / "analysis/decision.json",
        SOURCE / "mosaickv/docs/common_streaming_prefill_20261101.md",
    ):
        bound[str(path)] = sha(path)
    for job in old["jobs"]:
        method_of(job)
        folder = REFERENCE / f"workers/{job['index']}"
        record = previous.read(folder / "result.json")
        if record["status"] != "completed" or record["failures"]:
            raise ValueError("all ten original method/input producers must be complete")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in record["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("frozen original result changed")
            bound[str(folder / name)] = digest
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_all_method_streaming_construction_identity",
            reference_root=str(REFERENCE),
            maximum_simultaneous_workers=6,
            maximum_worker_minutes=45,
            baseline_comparison_admissible=False,
            common_storage_controls_not_byte_matched_algorithm_evidence=True,
            no_fresh_or_large_campaign_authorized=True,
            paper_campaign_authorized=False,
            independent_quality_N=0,
            same_final_payload_and_memberships_required=True,
            VL_all_layer_statistics_not_approximated=True,
            numerical_guard="bitwise_128_generation_and_complete_teacher_logits_to_own_parent",
            original_six_mechanisms_disabled_in_this_precision_variant=True,
        ),
    )
    print(sha(root / "plan.json"))


def offline(adapter, prepared, cfg, job, profile, profile_path, cap):
    """Independent frozen public all-layer implementation; validation only."""
    from contextlib import ExitStack

    from mosaickv.evaluation.direct_source_constructor import DirectSourcePolicy, construct
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.evaluation.text_source_storage import TextSourceCapture, optimize

    method = method_of(job)
    with ExitStack() as stack:
        text = stack.enter_context(TextSourceCapture(adapter, prepared))
        queries = stack.enter_context(PromptQueryCapture(adapter))
        if method == "mosaickv_exact":
            source = stack.enter_context(SourceCapture(adapter))
            phase = stack.enter_context(PhaseCapture(adapter))
        pref = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
    with __import__("torch").inference_mode():
        if method == "mosaickv_exact":
            packets, proof = construct(
                adapter,
                prepared,
                pref,
                source,
                text,
                phase,
                profile,
                cap,
                policy=DirectSourcePolicy(head_group=4, lossless_minimum=False),
            )
            memberships = proof["source_memberships"]
        else:
            packets, proof = bundle(
                adapter,
                prepared,
                pref,
                queries,
                cfg,
                method,
                profile_path,
                0.5,
                policies=("head3_boundary8",),
                allocation_mode="balanced" if method == "prompt_attention_topk" else "profile",
            )
            memberships = proof["source_memberships"]
            packets = tuple(
                optimize(packet, text.records[i], memberships[i], text.positions, block.self_attn)[
                    0
                ]
                for i, (packet, block) in enumerate(
                    zip(packets["head3_boundary8"], adapter._language_layers(), strict=True)
                )
            )
    return packets, memberships, pref.logits.detach().cpu()


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.source_teacher_protocol import load_targets
    from mosaickv.evaluation.storage_runs import run
    from mosaickv.evaluation.storage_teacher_forcing import score_formats
    from mosaickv.evaluation.streaming_baselines import SelectedRetiringCache, StreamingBaseline
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention
    from mosaickv.evaluation.streaming_source_constructor import StreamingDirectSource
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.common.inputs(root, expected)
    job, cache = plan["jobs"][index], None
    payload, method = previous.read(job["inputs"][0]), method_of(job)
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw.jsonl").touch(exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
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
            raise NotImplementedError("pinned LLaVA BF16 eager only")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        folder = REFERENCE / f"workers/{index}"
        cap = previous.read(folder / "construction.json")["cap_bytes"]
        profile = tuple(previous.read(Path(plan["profile_path"])))
        prior_rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
        prior = next(r for r in prior_rows if r["variant"] == job["evaluation_variant"])
        prior_logits = torch.load(
            folder / (job["evaluation_variant"] + "-logits.pt"),
            weights_only=True,
            map_location="cpu",
        )["greedy"]
        packets, memberships, first_logits = offline(
            adapter, prepared, cfg, job, profile, plan["profile_path"], cap
        )
        packet_hash = control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
        packet_bytes = backing_bytes(tuple(t for p in packets for t in p.tensors()))
        if packet_bytes != prior["storage"]["initial_active_kv_storage_bytes"]:
            raise RuntimeError("offline reference changed original encoded cache bytes")
        write_once(
            out / "offline_payload.json",
            dict(
                payload_sha256=packet_hash,
                physical_bytes=packet_bytes,
                memberships=memberships,
                original_prefill_KV_lower_bound=len(profile)
                * prepared.logical_sequence_length
                * 32
                * 128
                * 4,
            ),
        )
        del packets
        gc.collect()
        torch.cuda.synchronize()
        manifest = previous.read(folder / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            evaluator_commit=plan["source_sha"],
            config_sha256=hashlib.sha256(
                json.dumps(
                    dict(parent=manifest["config_sha256"], plan=expected), sort_keys=True
                ).encode()
            ).hexdigest(),
            environment=control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            method=method,
            evaluation_variant=job["evaluation_variant"],
            representation_adaptation_not_official_method=True,
            instrumentation_not_production_latency=True,
            independent_quality_N=0,
        )
        write_once(out / "manifest.json", manifest)
        torch.cuda.reset_peak_memory_stats()
        cache = (
            RetiringPrefillCache(len(profile), audit=True)
            if method == "mosaickv_exact"
            else SelectedRetiringCache(len(profile), audit=True)
        )
        encoder = (
            StreamingDirectSource(adapter, prepared, cache, profile)
            if method == "mosaickv_exact"
            else StreamingBaseline(prepared, cache, profile, method)
        )
        fresh = replace(prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache))
        started = time.monotonic()
        with SealAfterAttention(adapter, cache, encoder):
            pref = adapter.prefill(fresh, capture_queries=False, capture_attentions=False)
        with torch.inference_mode():
            encoder.finish() if method == "mosaickv_exact" else encoder.finish(adapter)
        torch.cuda.synchronize()
        prefill_seconds = time.monotonic() - started
        if tuple(tuple(map(tuple, layer)) for layer in encoder.memberships) != tuple(
            tuple(map(tuple, layer)) for layer in memberships
        ):
            raise RuntimeError("streaming changed selected source membership")
        packets = cache.formats()
        if (
            control.digest_tensors(tuple(t for p in packets for t in p.tensors())) != packet_hash
            or backing_bytes(tuple(t for p in packets for t in p.tensors())) != packet_bytes
        ):
            raise RuntimeError("streaming changed original payload bits or backing sharing")
        if not torch.equal(pref.logits.cpu(), first_logits):
            raise RuntimeError("streaming changed original prefill logits")
        gc.collect()
        prefill_storage = cache.record()
        if prefill_storage["original_KV_tensor_objects_alive"]:
            raise RuntimeError("retired native cache tensors remain alive")
        write_once(out / "prefill_storage.json", prefill_storage)
        runtime = replace(
            pref, state=replace(pref.state, past_key_values=None), attention_weights=()
        )
        del pref, fresh, cache, encoder
        cache = None
        gc.collect()
        values, details = run(
            adapter,
            runtime,
            packets,
            tokens=128,
            audit=True,
            sealed=True,
            sequential_lease=True,
            source_history=True,
            sequential_source_history=True,
            source_history_constructor="direct_batched_checks",
            consolidated_history=True,
        )
        if not torch.equal(values, prior_logits):
            raise RuntimeError("streaming changed original 128-vocabulary-logit trajectory")
        torch.save(values, out / "greedy_logits.pt")
        targets, provenance = load_targets(plan, payload, adapter.processor.tokenizer)
        teacher, teacher_logits = score_formats(
            adapter,
            runtime,
            packets,
            targets,
            maximum=1024,
            sequential_source_history=True,
            audit_history=True,
            return_logits=True,
        )
        old_teacher = prior["teacher_forced"]
        if control.digest_tensors((teacher_logits,)) != old_teacher["storage"][
            "raw_history_control"
        ]["source_history_logit_sha256"] or any(
            teacher[k] != old_teacher[k]
            for k in (
                "nll_sum",
                "per_token_nll",
                "conditional_greedy_tokens",
                "evaluated_token_count",
            )
        ):
            raise RuntimeError("streaming changed complete teacher vocabulary logits or NLL")
        row = dict(
            status="completed",
            variant=job["evaluation_variant"],
            method=method,
            sample_id=payload["sample_id"],
            task=payload["task"],
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            config_sha256=manifest["config_sha256"],
            manifest_path=str(out / "manifest.json"),
            measurement_type="ablation_measured",
            payload_and_memberships_identical=True,
            generation_128_logits_bitwise=True,
            complete_teacher_logits_bitwise=True,
            prefill_logits_bitwise=True,
            teacher_forced=teacher,
            teacher_provenance=provenance,
            prefill_storage=prefill_storage,
            generation_storage=details["storage"],
            prefill_seconds=prefill_seconds,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            independent_quality_N=0,
            same_source_capacity_not_encoded_byte_matched=True,
            whole_request_gpu_peak_certified=False,
        )
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        status="failed",
                        method=method,
                        variant=job["evaluation_variant"],
                        sample_id=payload["sample_id"],
                        task=payload["task"],
                        error=result["failures"][-1],
                        partial_cache=None
                        if cache is None
                        else cache.record(require_complete=False),
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
        raise
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    rows, failures = [], []
    for job in plan["jobs"]:
        try:
            folder = root / f"workers/{job['index']}"
            record = previous.read(folder / "result.json")
            if record["source_sha"] != plan["source_sha"] or record["plan_sha256"] != expected:
                raise ValueError("producer provenance mismatch")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("immutable worker artifact changed")
            failures.extend(record["failures"])
            rows.extend(
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            )
        except (OSError, KeyError, ValueError) as exc:
            failures.append(f"worker {job['index']}: {exc}")
    identities = [(r.get("method"), r.get("task"), r.get("sample_id")) for r in rows]
    complete = (
        len(rows) == 10
        and len(set(identities)) == 10
        and not failures
        and all(row_is_complete(r) for r in rows)
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            complete=complete,
            failures=failures,
            rows=rows,
            all_five_methods_streaming_identity_gate=complete,
            independent_quality_N=0,
            byte_matched_algorithm_superiority_established=False,
            whole_request_gpu_peak_certified=False,
            fresh_or_large_campaign_authorized=False,
            baseline_time_comparison_requires_separate_repeated_measurement=True,
        ),
    )


def submit(root, expected):
    original = control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_common_streaming.sbatch")
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(control.parent.subprocess, "check_output", scoped):
        control.parent.submit(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
