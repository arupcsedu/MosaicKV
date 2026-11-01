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
AUDIT = Path("/scratch/djy8hg/runs/evicastkv/exact-minimum-streaming-consumed-20261101-v1")
RANGE_AUDIT = Path(
    "/scratch/djy8hg/runs/evicastkv/compact-quantizer-workspace-consumed-20261101-v1"
)
CODE_AUDIT = Path(
    "/scratch/djy8hg/runs/evicastkv/lossless-layer-code-delta-gpu-consumed-20261101-retry3"
)
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


def prepare(root, *, timing=False, exact_minimum=False, optimized=False):
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
    jobs = old["jobs"]
    optimized_bytes = {}
    if optimized:
        if not timing:
            raise ValueError("optimized execution is restricted to the registered timing follow-up")
        proof = previous.read(CODE_AUDIT / "analysis/decision.json")
        if (
            not proof["complete"]
            or proof["failures"]
            or not proof["all_five_methods_bitwise"]
            or not proof["source_net_storage_gate"]
        ):
            raise ValueError(
                "complete code replay, quality identity and net-storage proof required"
            )
        if any(
            r["constructed_code_payload"]["enabled"]
            for r in proof["rows"]
            if r["method"] != "mosaickv_exact"
        ):
            raise ValueError("do not disable a baseline codec that showed net savings")
        for path in (
            CODE_AUDIT / "plan.json",
            CODE_AUDIT / "analysis/decision.json",
            RANGE_AUDIT / "plan.json",
            SOURCE / "mosaickv/docs/optimized_source_request_timing_20261101.md",
        ):
            bound[str(path)] = sha(path)
        for j in jobs:
            if method_of(j) == "mosaickv_exact":
                folder = CODE_AUDIT / f"workers/{j['index']}"
                row = json.loads((folder / "raw.jsonl").read_text().splitlines()[0])
                optimized_bytes[str(j["input_index"])] = row["initial_active_bytes"]
                for name in ("result.json", "raw.jsonl"):
                    bound[str(folder / name)] = sha(folder / name)
    if exact_minimum:
        diagnostic = Path(
            "/scratch/djy8hg/runs/evicastkv/streaming-packet-diagnostic-consumed-20261101-v1/analysis/decision.json"
        )
        status = previous.read(diagnostic)
        if not status["complete"] or status["failures"]:
            raise ValueError("complete consumed boundary-refit localization required")
        for path in (diagnostic, SOURCE / "mosaickv/docs/exact_minimum_streaming_20261101.md"):
            bound[str(path)] = sha(path)
    if timing:
        audit = previous.read(AUDIT / "analysis/decision.json")
        if (
            not audit["complete"]
            or audit["failures"]
            or not audit["all_five_methods_streaming_identity_gate"]
        ):
            raise ValueError(
                "all-method payload/generation/teacher streaming identity must pass first"
            )
        for path in (
            AUDIT / "plan.json",
            AUDIT / "analysis/decision.json",
            SOURCE / "mosaickv/docs/common_streaming_timing_20261101.md",
        ):
            bound[str(path)] = sha(path)
        for job in jobs:
            folder = AUDIT / f"workers/{job['index']}"
            record = previous.read(folder / "result.json")
            bound[str(folder / "result.json")] = sha(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("streaming correctness evidence changed")
                bound[str(folder / name)] = digest
        jobs = [
            dict(job, reference_worker=job["index"], index=i)
            for i, job in enumerate(j for j in jobs if method_of(j) == "mosaickv_exact")
        ]
        exact_minimum = previous.read(AUDIT / "plan.json").get("exact_minimum_only", False)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_all_method_streaming_randomized_timing"
            if timing
            else "consumed_all_method_streaming_construction_identity",
            jobs=jobs,
            reference_root=str(REFERENCE),
            maximum_simultaneous_workers=2 if timing else 6,
            maximum_worker_minutes=75 if timing else 45,
            paired_common_streaming_timing=timing,
            optimized_source_runtime=optimized,
            optimized_source_initial_bytes=optimized_bytes,
            common_componentwise_fitting=optimized,
            code_delta_methods=["mosaickv_exact"] if optimized else [],
            baseline_codec_disabled_only_after_registered_no_savings_evidence=optimized,
            require_lower_observed_constructor_peak_than_all_baselines=optimized,
            exact_minimum_only=exact_minimum,
            timing_warmups=5 if timing else 0,
            timing_repetitions=20 if timing else 0,
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
    job, cache, timing_context = plan["jobs"][index], None, {}
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
        if plan.get("paired_common_streaming_timing", False):
            timing_worker(root, expected, index, adapter, plan, job, payload, timing_context)
            result["status"] = "completed"
            return
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
        original_packet_bytes = packet_bytes
        minimum_proof = None
        if plan.get("exact_minimum_only", False):
            from mosaickv.evaluation.exact_minimum_metadata import prove_and_compact

            packets, minimum_proof = prove_and_compact(adapter, packets)
            packet_hash = control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
            packet_bytes = backing_bytes(tuple(t for p in packets for t in p.tensors()))
            if packet_bytes >= original_packet_bytes:
                raise RuntimeError("lossless-minimum variant did not reduce persistent bytes")
        write_once(
            out / "offline_payload.json",
            dict(
                payload_sha256=packet_hash,
                physical_bytes=packet_bytes,
                original_FP32_minimum_packet_bytes=original_packet_bytes,
                exact_minimum_proof=minimum_proof,
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
            exact_minimum_only=plan.get("exact_minimum_only", False),
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
            StreamingDirectSource(
                adapter,
                prepared,
                cache,
                profile,
                minimum_only=plan.get("exact_minimum_only", False),
            )
            if method == "mosaickv_exact"
            else StreamingBaseline(
                prepared, cache, profile, method, minimum_only=plan.get("exact_minimum_only", False)
            )
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
            payload_identity_reference="losslessly_compacted_frozen_parent"
            if minimum_proof
            else "frozen_parent",
            exact_minimum_only=plan.get("exact_minimum_only", False),
            original_FP32_minimum_packet_bytes=original_packet_bytes,
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
                    )
                    | timing_context,
                    allow_nan=False,
                )
                + "\n"
            )
        raise
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def timing_worker(root, expected, index, adapter, plan, job, payload, timing_context):
    import torch

    from mosaickv.evaluation.common_streaming_timing import METHODS, order, request, summarize
    from mosaickv.measurements.telemetry import capture_gpu_environment

    out = root / f"workers/{index}"
    originals = previous.read(AUDIT / "plan.json")["jobs"]
    references = {}
    for original in originals:
        if original["input_index"] == job["input_index"]:
            method = method_of(original)
            folder = AUDIT / f"workers/{original['index']}"
            references[method] = dict(
                payload=previous.read(folder / "offline_payload.json"),
                logits=torch.load(
                    folder / "greedy_logits.pt", weights_only=True, map_location="cpu"
                ),
                memory=previous.read(folder / "prefill_storage.json"),
            )
    if set(references) != set(METHODS):
        raise ValueError("all five frozen method references required")
    manifest = previous.read(REFERENCE / f"workers/{job['reference_worker']}/manifest.json")
    manifest.update(
        source_sha=plan["source_sha"],
        source_dirty=False,
        plan_sha256=expected,
        config_sha256=hashlib.sha256(
            json.dumps(
                dict(parent=manifest["config_sha256"], plan=expected), sort_keys=True
            ).encode()
        ).hexdigest(),
        evaluator_commit=plan["source_sha"],
        environment=control.environment(),
        gpu=asdict(capture_gpu_environment(torch)),
        methods=list(METHODS),
        representation_adaptations_not_official_methods=True,
        independent_quality_N=0,
        randomized_same_engine_methods=True,
        run_id=str(out),
        parent_run_id=manifest.get("run_id"),
        warmups_per_method=5,
        repetitions_per_method=20,
        optimized_source_runtime=plan.get("optimized_source_runtime", False),
        code_delta_methods=plan.get("code_delta_methods", []),
        initial_bytes_include_request_replay_workspace=True,
        output_predictions=128,
        tensor_result_auditing=False,
        warm_weights_and_media=True,
        disk_media_and_model_loading_excluded=True,
        shared_encoded_byte_budget_claim=False,
    )
    write_once(out / "manifest.json", manifest)
    profile = tuple(previous.read(Path(plan["profile_path"])))
    messages = previous.load_messages(payload)
    rows, seen = [], set()
    for entry in order(plan["seed"] + index):
        timing_context.update(entry)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        method = entry["method"]
        optimized = plan.get("optimized_source_runtime", False)
        reference = references[method]
        validator = None
        if optimized and method == "mosaickv_exact" and method not in seen:
            if not entry["warmup"]:
                raise RuntimeError("initial packet audit may only occur on excluded warmup")

            def validator(packets, *, reference=reference):
                if (
                    control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
                    != reference["payload"]["payload_sha256"]
                ):
                    raise RuntimeError("optimized constructor changed frozen input packet bits")

        values, packets, measured = request(
            adapter,
            messages,
            profile,
            method,
            minimum_only=plan.get("exact_minimum_only", False),
            componentwise_fitting=optimized,
            code_delta=method in plan.get("code_delta_methods", []),
            input_packet_validator=validator,
        )
        if not torch.equal(values, reference["logits"]):
            raise RuntimeError("timed request changed frozen 128-token vocabulary logits")
        expected_bytes = (
            plan["optimized_source_initial_bytes"][str(job["input_index"])]
            if optimized and method == "mosaickv_exact"
            else reference["payload"]["physical_bytes"]
        )
        if measured["initial_encoded_bytes"] != expected_bytes:
            raise RuntimeError("timed request changed frozen initial physical bytes")
        if (
            measured["original_precompression_packet_bytes"]
            != reference["payload"]["physical_bytes"]
        ):
            raise RuntimeError("timed constructor changed original backing storage")
        if method not in seen:
            if not (optimized and method == "mosaickv_exact") and (
                control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
                != reference["payload"]["payload_sha256"]
            ):
                raise RuntimeError("timed request changed encoded payload bits")
            seen.add(method)
        row = dict(
            **entry,
            **measured,
            run_id=str(out),
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
        rows.append(row)
        del values, packets, measured, row
    timing_context.clear()
    write_once(out / "timing_summary.json", summarize(rows, plan["seed"] + index))


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
    if plan.get("paired_common_streaming_timing", False):
        summaries = []
        try:
            summaries = [previous.read(root / f"workers/{i}/timing_summary.json") for i in range(2)]
        except (OSError, KeyError, ValueError) as exc:
            failures.append(str(exc))
        complete = (
            len(rows) == 250 and not failures and all(r["status"] == "completed" for r in rows)
        )
        write_once(
            root / "analysis/decision.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                complete=complete,
                failures=failures,
                rows=rows,
                summaries=summaries,
                all_method_similar_time_gate=complete
                and all(
                    p["similar_time_gate_upper_ratio_1_05"]
                    for s in summaries
                    for p in s["pairs"].values()
                ),
                independent_quality_N=0,
                no_byte_matched_algorithm_or_paper_superiority_claim=True,
                memory_evidence_must_use_separate_audit=True,
            ),
        )
        return
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
    plan = previous.common.inputs(root, expected)
    if plan.get("optimized_source_runtime", False):
        proof = previous.read(RANGE_AUDIT / "analysis/decision.json")
        range_jobs = previous.read(RANGE_AUDIT / "plan.json")["jobs"]
        if not optimized_controls_pass(proof, range_jobs):
            write_once(
                root / "blocked.json",
                dict(
                    GPU_submitted=False,
                    reason="all-method common-range packet/storage/prefill identity did not pass",
                    parent_decision_sha256=sha(RANGE_AUDIT / "analysis/decision.json"),
                ),
            )
            return
        write_once(
            root / "optimized_parent_proof.json",
            dict(
                path=str(RANGE_AUDIT / "analysis/decision.json"),
                sha256=sha(RANGE_AUDIT / "analysis/decision.json"),
            ),
        )
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
    plan = previous.common.inputs(root, expected)
    if plan.get("exact_minimum_only") and not plan.get("paired_common_streaming_timing"):
        launched = previous.read(root / "submission.json")
        followup = original(
            [
                "sbatch",
                "--parsable",
                "--partition=bii",
                "--account=bii_dsc_community",
                "--cpus-per-task=2",
                "--mem=8G",
                "--time=00:10:00",
                f"--dependency=afterany:{launched['cpu_analysis']}",
                "--job-name=mk-minimum-advance",
                f"--output={root}/slurm/advance-%j.out",
                f"--error={root}/slurm/advance-%j.err",
                str(SOURCE / "mosaickv/slurm/evicastkv_common_streaming.sbatch"),
                str(SOURCE),
                str(root),
                expected,
                "advance",
            ],
            text=True,
        ).strip()
        write_once(
            root / "registered_followup.json",
            dict(
                job_id=followup,
                scope="conditional_same_consumed_inputs_timing_only",
                paper_campaign_authorized=False,
            ),
        )


def optimized_controls_pass(proof, jobs):
    if (
        not proof.get("complete")
        or proof.get("failures")
        or len(proof.get("cases", [])) != 10
        or len(jobs) != 10
    ):
        return False
    if {c["case"] for c in proof["cases"]} != set(range(10)):
        return False
    peaks = {}
    for case in proof["cases"]:
        row = case["observations"]
        if not all(
            row.get(k) is True
            for k in (
                "payload_bitwise",
                "prefill_logits_bitwise",
                "physical_packet_bytes_identical",
            )
        ):
            return False
        memory = row["memory"]
        if (
            memory.get("tensor_results_observed") is not True
            or memory.get("original_KV_tensor_objects_alive") != 0
            or memory.get("observed_prefill_cuda_peak_bytes", 0) <= 0
        ):
            return False
        job = jobs[case["case"]]
        key = (job["input_index"], method_of(job))
        if key in peaks:
            return False
        peaks[key] = memory["observed_prefill_cuda_peak_bytes"]
    return all(
        peaks[i, "mosaickv_exact"]
        < min(
            peaks[i, m]
            for m in ("prompt_attention_topk", "prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl")
        )
        for i in (0, 1)
    )


def register_optimized_timing(root, expected):
    """Finite continuation: reuse same-SHA CPU proof, wait for ALL GPU controls."""
    plan = previous.common.inputs(root, expected)
    if not plan.get("optimized_source_runtime") or not plan.get("paired_common_streaming_timing"):
        raise ValueError("only the registered consumed optimized timing may continue")
    producer = previous.read(RANGE_AUDIT / "plan.json")
    proof = previous.read(RANGE_AUDIT / "cpu.json")
    if (
        producer["source_sha"] != plan["source_sha"]
        or proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["xml_sha256"] != sha(RANGE_AUDIT / "cpu.xml")
    ):
        raise ValueError("complete same-immutable-source CPU suite required")
    for name in ("cpu.json", "cpu.xml"):
        os.link(RANGE_AUDIT / name, root / name)
    write_once(
        root / "cpu_reuse.json",
        dict(
            source_sha=plan["source_sha"],
            producer_root=str(RANGE_AUDIT),
            cpu_json_sha256=sha(RANGE_AUDIT / "cpu.json"),
            cpu_xml_sha256=sha(RANGE_AUDIT / "cpu.xml"),
            same_complete_suite_same_source=True,
        ),
    )
    submission = previous.read(RANGE_AUDIT / "submission.json")
    job = control.parent.subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=8G",
            "--time=00:10:00",
            f"--dependency=afterany:{submission['cpu_analysis']}",
            "--job-name=mk-optimized-time-admit",
            f"--output={root}/slurm/admit-%j.out",
            f"--error={root}/slurm/admit-%j.err",
            str(SOURCE / "mosaickv/slurm/evicastkv_common_streaming.sbatch"),
            str(SOURCE),
            str(root),
            expected,
            "submit",
        ],
        text=True,
    ).strip()
    write_once(
        root / "registered_followup.json",
        dict(
            gpu_admission_job=job,
            parent_analysis=submission["cpu_analysis"],
            finite_scope="same_consumed_inputs_complete_requests_only",
            fresh_or_large_campaign_authorized=False,
        ),
    )


def advance(root, expected):
    plan = previous.common.inputs(root, expected)
    decision = previous.read(root / "analysis/decision.json")
    if (
        not plan.get("exact_minimum_only")
        or not decision["complete"]
        or decision["failures"]
        or not decision["all_five_methods_streaming_identity_gate"]
    ):
        write_once(
            root / "timing_promotion.json",
            dict(
                authorized=False,
                reason="common exact-minimum identity gate did not pass",
                paper_campaign_authorized=False,
            ),
        )
        return
    target = root.parent / "exact-minimum-streaming-timing-consumed-20261101-v1"
    prepare(target, timing=True)
    for name in ("cpu.json", "cpu.xml"):
        os.link(root / name, target / name)
    write_once(
        target / "cpu_reuse.json",
        dict(
            source_sha=plan["source_sha"],
            producer_root=str(root),
            cpu_json_sha256=sha(root / "cpu.json"),
            cpu_xml_sha256=sha(root / "cpu.xml"),
            same_complete_suite_same_immutable_source=True,
        ),
    )
    write_once(
        root / "timing_promotion.json",
        dict(authorized=True, target=str(target), paper_campaign_authorized=False),
    )
    submit(target, sha(target / "plan.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "prepare_timing",
            "prepare_exact_minimum",
            "prepare_optimized_timing",
            "register_optimized_timing",
            "cpu",
            "submit",
            "worker",
            "analyze",
            "advance",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode in (
        "prepare",
        "prepare_timing",
        "prepare_exact_minimum",
        "prepare_optimized_timing",
    ):
        prepare(
            args.root,
            timing=args.mode in ("prepare_timing", "prepare_optimized_timing"),
            exact_minimum=args.mode == "prepare_exact_minimum",
            optimized=args.mode == "prepare_optimized_timing",
        )
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
