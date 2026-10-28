"""Allocated-GPU execution only. Diagnostic clones never enter timing workers."""

from __future__ import annotations

import contextlib
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from evicastkv_campaign import environment, load_lock, resolved_config

from mosaickv.evaluation.fox_campaign import (
    BUDGETS,
    METHODS,
    CampaignRow,
    completed_trace,
    digest,
    sha,
    unique_storage_bytes,
    write_once,
)


def generation_observer(adapter):
    """A timing-only wrapper: never captures queries/attentions beyond policy."""

    class Observer:
        def __init__(self):
            self.phases = {}
            self.steps = []
            self.prefill_kwargs = None
            self.prefill_finished = None
            self.emissions = []
            self.capture_first_logits = False
            self.first_logits = None
            self.output_completed = None
            self.completion_memory = None

        def complete(self, answer):
            import torch

            torch.cuda.synchronize(adapter.device)
            self.output_completed = time.perf_counter()
            self.completion_memory = dict(
                allocated=torch.cuda.max_memory_allocated(adapter.device),
                reserved=torch.cuda.max_memory_reserved(adapter.device),
            )

        def emit(self, index, token):
            # A real consumer handoff, not a retrospective prefill-time proxy.
            # CPU transfer completes token availability before timestamping.
            ids = tuple(token.detach().cpu().reshape(-1).tolist())
            self.emissions.append((index, time.perf_counter(), ids))

        @contextlib.contextmanager
        def install(self):
            original_prepare, original_prefill, original_decode = (
                adapter.prepare_inputs,
                adapter.prefill,
                adapter.decode_one_token,
            )

            def prepare(*a, **kw):
                start = time.perf_counter()
                with __import__("torch").profiler.record_function("evicastkv/input_preparation"):
                    result = original_prepare(*a, **kw)
                self.phases["input_preparation_wall"] = time.perf_counter() - start
                return result

            def prefill(*a, **kw):
                self.prefill_kwargs = kw
                start = time.perf_counter()
                with __import__("torch").profiler.record_function(
                    "evicastkv/prefill_with_policy_observation"
                ):
                    result = original_prefill(*a, **kw)
                __import__("torch").cuda.synchronize(adapter.device)
                if self.capture_first_logits:
                    self.first_logits = result.logits.detach().float().cpu()
                self.prefill_finished = time.perf_counter()
                self.phases["prefill_observed_wall"] = self.prefill_finished - start
                return result

            def decode(*a, **kw):
                start = time.perf_counter()
                with __import__("torch").profiler.record_function("evicastkv/decode_one_token"):
                    result = original_decode(*a, **kw)
                self.steps.append(time.perf_counter() - start)
                return result

            with (
                patch.object(adapter, "prepare_inputs", prepare),
                patch.object(adapter, "prefill", prefill),
                patch.object(adapter, "decode_one_token", decode),
            ):
                yield self

    return Observer()


@contextlib.contextmanager
def layout_capture(adapter):
    """Diagnostic-only original-layout reference, independent of packer masks."""
    import torch

    import mosaickv.backends.hf_runtime as runtime

    original = runtime._state_from_packed
    capture = {}

    def record(local_adapter, source, packed):
        result = original(local_adapter, source, packed)
        reference = runtime._clone_decode_state(adapter, source)
        compact = runtime._clone_decode_state(adapter, result)
        source_layers = adapter._legacy_layers(reference.past_key_values)[0]
        compact_layers = adapter._legacy_layers(compact.past_key_values)[0]
        masks = []
        positions = {}
        for item in packed.slot_records:
            if item["tier"] != "exact":
                raise AssertionError("campaign must retain original exact KV")
            positions.setdefault((int(item["layer"]), int(item["kv_head"])), []).append(
                (int(item["slot"]), int(item["logical_position"]))
            )
        for layer, (key, value) in enumerate(source_layers):
            if key.ndim != 4 or adapter._cache_sequence_dimension(key) != 2:
                raise NotImplementedError("oracle currently validates B,H,N,D adapters only")
            mask = torch.zeros(key.shape[1:3], dtype=torch.bool, device=key.device)
            for head in range(key.shape[1]):
                entries = sorted(positions[(layer, head)])
                ids = [p for _, p in entries]
                if ids != sorted(set(ids)):
                    raise AssertionError("nonmonotonic/duplicate logical positions")
                slots = [s for s, _ in entries]
                mask[head, ids] = True
                if not torch.equal(
                    key[:, head, ids], compact_layers[layer][0][:, head, slots]
                ) or not torch.equal(value[:, head, ids], compact_layers[layer][1][:, head, slots]):
                    raise AssertionError("retained KV changed")
            masks.append(mask)
        reference.model_state["mosaickv_validity_masks"] = tuple(masks)
        reference.model_state["mosaickv_packed_all_valid"] = False
        reference.model_state["mosaickv_cache_all_valid"] = False
        capture.update(
            reference=reference,
            compact=compact,
            independent_positions={
                identity: tuple(p for _, p in sorted(entries))
                for identity, entries in positions.items()
            },
            positions_sha256=digest(packed.slot_records),
            actual_unique_backing_bytes=unique_storage_bytes(
                [t for pair in compact_layers for t in pair]
            ),
            original_logical_length=source.logical_sequence_length,
            next_decode_position=source.next_decode_position,
            retained_entries=sum(len(v) for v in positions.values()),
            padded_entries=sum(k.shape[1] * k.shape[2] for k, _ in compact_layers),
        )
        return result

    with patch.object(runtime, "_state_from_packed", record):
        yield capture


def check_layout(adapter, capture, token_ids):
    if not capture or len(token_ids) < 17:
        raise AssertionError("missing compressed representation or forced continuation")
    reference, compact = capture.pop("reference"), capture.pop("compact")
    from mosaickv.evaluation.fox_numerics import validate_independent_layout

    positions = capture.pop("independent_positions")
    return dict(
        capture, **validate_independent_layout(adapter, reference, compact, positions, token_ids)
    )


def run_worker(root: Path, stage: str, index: int, spec_sha: str):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.config import canonical_config, config_sha256
    from mosaickv.evaluation.coco_shared_bytes import processor_signature
    from mosaickv.evaluation.fox_measurement import ProcessSampler, phase_ranges
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    spec = root / "jobs" / f"{stage}.json"
    if sha(spec) != spec_sha:
        raise ValueError("job specification changed")
    if json.loads(spec.read_text())["protocol_sha256"] != sha(root / "protocol_lock.json"):
        raise ValueError("job specification belongs to a different protocol")
    lock = load_lock(root)
    job = json.loads(spec.read_text())["jobs"][index]
    workload = job["workload"]
    attempt = f"{os.environ.get('SLURM_JOB_ID', 'not_slurm')}-{index}-{time.time_ns()}"
    output = root / "outputs" / stage / f"{index:03d}" / attempt
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        stage=stage,
        index=index,
        status="failed",
        source_sha=lock["source_sha"],
        records=[],
        errors=[],
        attempt=attempt,
    )
    start = time.perf_counter()
    attention_transport = contextlib.ExitStack()
    try:
        if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one allocated GPU required; no login-node GPU execution")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("deterministic cuBLAS must be set before Python starts")
        torch.set_num_threads(12)
        arithmetic = lock.get("arithmetic", dict(allow_fp16_reduced_precision_reduction=True))
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = arithmetic[
            "allow_fp16_reduced_precision_reduction"
        ]
        torch.manual_seed(lock["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = resolved_config(lock, workload, "full_kv", None)
        cohort = lock["cohorts"][workload][job["subset"]]
        task = "coco_caption" if workload == "coco" else "milebench"
        ledger = load_paper_native_ledger(
            cohort["path"],
            expected_sha256=cohort["sha256"],
            task=default_task_registry().resolve(task),
            dataset_revision=cfg.dataset.revision,
        )
        samples = [ledger.samples[i] for i in job["indices"]]
        load_start = time.perf_counter()
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        policy = arithmetic.get("attention_layout_policy", "native")
        if policy == "valid_slot_decode":
            if stage not in {"correctness", "representation_canary"}:
                raise RuntimeError("valid-slot transport is registered for correctness only")
            from mosaickv.evaluation.fox_valid_slots import valid_slot_diagnostic

            attention_transport.enter_context(valid_slot_diagnostic(adapter))
        elif policy != "native":
            raise RuntimeError("unsupported attention-layout policy")
        summary["model_load_seconds"] = time.perf_counter() - load_start
        parameter_devices = sorted({str(p.device) for p in adapter.model.parameters()})
        if any(
            p.device.type != "cuda" or p.dtype != torch.float16 for p in adapter.model.parameters()
        ):
            raise RuntimeError(
                "registered single-GPU FP16 lane forbids automatic CPU offload "
                "or mixed parameter precision"
            )
        summary["model_parameter_devices"] = parameter_devices
        processor_sha = digest(processor_signature(adapter.processor))
        gpu = asdict(capture_gpu_environment(torch))
        visible = [x for x in gpu["devices"] if x["visible_to_process"]]
        if len(visible) != 1 or "A100" not in visible[0]["name"]:
            raise RuntimeError("registered campaign requires one A100")
        if gpu["query_error"] or gpu["background_processes"]:
            raise RuntimeError("GPU isolation cannot be established from telemetry")
        # Shared-node processes on OTHER allocated devices do not invalidate isolation.
        gpu_uuid = visible[0]["uuid"] if "uuid" in visible[0] else visible[0].get("gpu_uuid")
        write_once(
            output / "environment.json",
            dict(
                environment=environment(),
                gpu=gpu,
                source_sha=lock["source_sha"],
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                processor_sha256=processor_sha,
                backend="huggingface",
                attention="eager",
                arithmetic_policy=arithmetic,
                actual_arithmetic_flags=dict(
                    allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    allow_fp16_accumulation=torch.backends.cuda.matmul.allow_fp16_accumulation,
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                ),
                seed=lock["seed"],
                measurement_type="measured_development",
                slurm_job_id=os.environ["SLURM_JOB_ID"],
                cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"),
                vllm="not_used",
                sglang="not_used",
                CUDA=str(torch.version.cuda),
            ),
        )
        scorer_path = (
            Path(__import__("mosaickv.evaluation.milebench", fromlist=["__file__"]).__file__)
            if workload == "milebench"
            else Path(__import__("lmms_eval.tasks.coco_cap.utils", fromlist=["__file__"]).__file__)
        )
        scorer_sha = sha(scorer_path)
        # Every model-loaded job carries parity on its first registered input.
        first = samples[0]
        messages = build_multimodal_messages(
            first.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in first.media)
        )
        prepared = adapter.prepare_inputs(messages)
        parity = {}
        for name, check in (
            ("ordinary_generate", compare_with_generate),
            ("retention_one", compare_cache_reinjection),
        ):
            report = asdict(check(adapter, prepared, max_new_tokens=16))
            parity[name] = report
            if (
                report["token_agreement"] != 1
                or report["maximum_logit_difference"]
                > lock["tolerances"]["retention_one_max_logit"]
            ):
                raise AssertionError(f"FullKV parity failed: {name}")
        from dataclasses import replace

        from mosaickv.backends.hf_runtime import compare_runtime_retention_one

        identity = resolved_config(lock, workload, "mosaickv_exact", BUDGETS[workload][1])
        identity = replace(
            identity,
            cache=replace(identity.cache, retention_ratio=1.0, budget_value=2**63 - 1),
            generation=replace(identity.generation, max_new_tokens=16),
        )
        packed_identity = asdict(compare_runtime_retention_one(adapter, prepared, identity))
        parity["mosaickv_runtime_retention_one"] = packed_identity
        if (
            packed_identity["token_agreement"] != 1
            or packed_identity["maximum_logit_difference"] > 1e-4
        ):
            raise AssertionError("MosaicKV packed retention-one parity failed")
        if stage == "evidence_canary":
            for variant in ("evidence-routing-v1", "evidence-coverage-v1", "evidence-both-v1"):
                candidate = resolved_config(lock, workload, "mosaickv_exact", 40, variant=variant)
                candidate = replace(
                    candidate,
                    cache=replace(candidate.cache, retention_ratio=1.0, budget_value=2**63 - 1),
                    generation=replace(candidate.generation, max_new_tokens=16),
                )
                report = asdict(compare_runtime_retention_one(adapter, prepared, candidate))
                parity[variant] = report
                if report["token_agreement"] != 1 or report["maximum_logit_difference"] > 1e-4:
                    raise AssertionError(f"experimental retention-one parity failed: {variant}")
        write_once(output / "retention_one.json", parity)
        del prepared
        if stage == "smoke":
            arms = [
                ("full_kv", None, "complete"),
                ("mosaickv_exact", BUDGETS[workload][0], "complete"),
            ]
        elif stage in {"correctness", "representation_canary"}:
            arms = [("mosaickv_exact", cap, "complete") for cap in BUDGETS[workload]]
        elif stage == "mechanisms":
            arms = [
                ("mosaickv_exact", cap, f"route-{fraction}")
                for fraction in (0, 1 / 256, 1 / 16, 1 / 4)
                for cap in BUDGETS[workload]
            ]
            arms += [
                ("mosaickv_exact", cap, variant)
                for variant in (
                    "mechanism-complete",
                    "routed-prompt",
                    "coverage-off",
                    "value-off",
                    "cross-modal-off",
                )
                for cap in BUDGETS[workload]
            ]
            arms += [
                ("prompt_attention_topk", cap, "route-zero-control") for cap in BUDGETS[workload]
            ]
            arms += [
                ("mosaickv_exact", cap, f"random-{seed}")
                for seed in (13, 29, 47)
                for cap in BUDGETS[workload]
            ]
        else:
            arms = [("full_kv", None, "complete")] + [
                (m, b, "complete")
                for b in BUDGETS[workload]
                for m in METHODS[workload]
                if m != "full_kv"
            ]
        if "arms" in job:
            arms = [tuple(a) for a in job["arms"]]
        # Five full-request warmups per arm on its actual shape regime; no prefills shared.
        if stage in {"measurement", "optimization"}:
            for method, cap, variant in arms:
                current = resolved_config(lock, workload, method, cap, variant=variant)
                warmup_arm = (
                    f"{method}-{cap}-{variant}" if stage == "optimization" else f"{method}-{cap}"
                )
                runtime = HuggingFaceMosaicKVModel(
                    adapter, current, trace_directory=output / "warmups" / warmup_arm
                )
                for repeat in range(lock["warmups"]):
                    settle(adapter)
                    warm_sample = samples[repeat % len(samples)]
                    warm_messages = build_multimodal_messages(
                        warm_sample.prompt,
                        tuple(MediaItem(m.kind, _media_payload(m)) for m in warm_sample.media),
                    )
                    from mosaickv.evaluation.fox_optimization import budget_implementation

                    with budget_implementation(lock, variant, required=stage == "optimization"):
                        result = runtime.generate(
                            EvaluationRequest(
                                f"warmup-{repeat}", warm_sample.sample_id, task, warm_messages, {}
                            )
                        )
                    write_once(
                        output / "warmups" / warmup_arm / f"{repeat}.json",
                        dict(metrics=asdict(result.metrics)),
                    )
                del runtime
        repeats = lock["repeats"] if stage in {"measurement", "optimization"} else 1
        for position, sample in enumerate(samples):
            for repeat in range(repeats):
                offset = (job["indices"][position] + repeat) % len(arms)
                ordered = arms[offset:] + arms[:offset]
                if stage == "optimization":
                    from mosaickv.evaluation.fox_optimization import factorial_order

                    ordered = factorial_order(arms, job["indices"][position], repeat)
                for order_position, (method, cap, variant) in enumerate(ordered):
                    key = f"{job['indices'][position]:03d}-{method}-{cap}-{variant}-r{repeat}"
                    record_path = output / "records" / f"{key}.json"
                    current = resolved_config(lock, workload, method, cap, variant=variant)
                    runtime_config_sha = config_sha256(current)
                    config_sha = digest(
                        dict(runtime_config=canonical_config(current), arithmetic=arithmetic)
                    )
                    trace_dir = output / "traces" / key
                    observer = generation_observer(adapter)
                    observer.capture_first_logits = stage == "scoring"
                    row = dict(
                        run_id=key,
                        sample_id=sample.sample_id,
                        media_group_ids=tuple(m.sha256 for m in sample.media),
                        workload=workload,
                        subtask=getattr(sample, "milebench_dataset", None),
                        method=method,
                        effective_method=None,
                        budget_bytes=int(cap * 2**20) if cap else None,
                        source_sha=lock["source_sha"],
                        config_sha=config_sha,
                        manifest_path=str(output / "environment.json"),
                        manifest_sha256=sha(output / "environment.json"),
                        input_sha=digest(
                            dict(prompt=sample.prompt, media=[m.sha256 for m in sample.media])
                        ),
                        processor_sha=processor_sha,
                        scorer_sha=scorer_sha,
                        attempt=attempt,
                        repeat=repeat,
                        status="failed",
                        gpu_uuid=gpu_uuid,
                        slurm_job_id=os.environ["SLURM_JOB_ID"],
                    )
                    try:
                        if time.perf_counter() - start > 6800:
                            raise TimeoutError("two-hour shard safety boundary reached")
                        write_once(
                            output / "resolved-configs" / f"{key}.json", canonical_config(current)
                        )
                        settle(adapter)
                        torch.cuda.reset_peak_memory_stats()
                        baseline_allocated = torch.cuda.memory_allocated()
                        baseline_reserved = torch.cuda.memory_reserved()
                        runtime = HuggingFaceMosaicKVModel(
                            adapter,
                            current,
                            trace_directory=trace_dir,
                            token_observer=observer.emit,
                            completion_observer=observer.complete,
                        )
                        diagnostic = stage in {"correctness", "representation_canary"} or (
                            stage == "evidence_canary" and method != "full_kv"
                        )
                        from mosaickv.evaluation.fox_mechanisms import mechanism_audit

                        mechanism = stage in {"mechanisms", "full_graph_pilot"}
                        profiling = stage == "profiling"
                        from mosaickv.evaluation.fox_optimization import budget_implementation

                        with (
                            budget_implementation(lock, variant, required=stage == "optimization"),
                            layout_capture(adapter)
                            if diagnostic
                            else contextlib.nullcontext({}) as captured,
                            observer.install(),
                            phase_ranges(),
                            mechanism_audit(
                                variant,
                                sample.sample_id,
                                stable_policy=resolved_config(
                                    lock, workload, "mosaickv_exact", cap
                                ),
                                raw_terms_directory=output / "mechanism-terms" / key,
                            )
                            if mechanism
                            else contextlib.nullcontext({}) as activity,
                            ProcessSampler(gpu_uuid) as sampler,
                            torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ],
                                record_shapes=False,
                                profile_memory=True,
                                with_stack=False,
                            )
                            if profiling
                            else contextlib.nullcontext(None) as profiler,
                        ):
                            torch.cuda.synchronize()
                            wall = time.perf_counter()
                            messages = build_multimodal_messages(
                                sample.prompt,
                                tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media),
                            )
                            input_load_seconds = time.perf_counter() - wall
                            result = runtime.generate(
                                EvaluationRequest(key, sample.sample_id, task, messages, {})
                            )
                            torch.cuda.synchronize()
                            returned_wall_seconds = time.perf_counter() - wall
                            if observer.output_completed is None:
                                raise RuntimeError("missing output-completion handoff")
                            wall_seconds = observer.output_completed - wall
                        if profiling:
                            profile_dir = output / "profiler" / key
                            profile_dir.mkdir(parents=True, exist_ok=True)
                            profiler.export_chrome_trace(str(profile_dir / "trace.json"))
                            averages = [
                                dict(
                                    name=item.key,
                                    count=item.count,
                                    self_cpu_us=item.self_cpu_time_total,
                                    inclusive_cpu_us=item.cpu_time_total,
                                    self_device_us=getattr(item, "self_device_time_total", None),
                                    inclusive_device_us=getattr(item, "device_time_total", None),
                                )
                                for item in profiler.key_averages()
                            ]
                            write_once(profile_dir / "operators.json", averages)
                        write_once(output / "telemetry" / f"{key}.json", sampler.record())
                        if mechanism:
                            write_once(output / "mechanisms" / f"{key}.json", activity)
                        traces = [completed_trace(trace_dir, key)]
                        trace = json.loads(traces[0].read_text())
                        tokens = trace.get("generated_token_ids", trace.get("token_ids"))
                        if tokens is None:
                            raise AssertionError("trace has no generated token IDs")
                        if method == "full_kv" and (
                            observer.prefill_kwargs.get("capture_attentions")
                            or observer.prefill_kwargs.get("capture_queries")
                        ):
                            raise AssertionError("FullKV executed unused policy observations")
                        if diagnostic:
                            representation = check_layout(adapter, captured, tokens)
                            write_once(
                                output / "correctness" / f"{key}.json",
                                representation,
                            )
                            if representation["status"] != "passed":
                                raise AssertionError(
                                    "independent representation/control gate failed; "
                                    "complete report preserved"
                                )
                        metrics = asdict(result.metrics)
                        if len(tokens) != 64 or len(observer.steps) != 63:
                            raise AssertionError("locked fixed output/forward schedule changed")
                        if cap and metrics["initial_active_kv_storage_bytes"] > cap * 2**20:
                            raise AssertionError("physical initial budget exceeded")
                        row.update(
                            status="completed",
                            answer=result.answer,
                            effective_method=result.effective_method or method,
                            token_ids=tuple(tokens),
                            emitted_tokens=len(tokens),
                            forward_steps=len(observer.steps),
                            cache_append_steps=len(observer.steps),
                            initial_kv_bytes=metrics["initial_active_kv_storage_bytes"],
                            final_kv_bytes=metrics["final_active_kv_storage_bytes"],
                            peak_kv_bytes=metrics["peak_active_kv_storage_bytes"],
                            peak_allocated_bytes=observer.completion_memory["allocated"],
                            peak_reserved_bytes=observer.completion_memory["reserved"],
                            e2e_wall_seconds=wall_seconds,
                            metrics=metrics,
                            timings=dict(
                                observer.phases,
                                input_load_seconds=input_load_seconds,
                                full_measurement_return_wall_seconds=returned_wall_seconds,
                                post_output_instrumentation_seconds=returned_wall_seconds
                                - wall_seconds,
                                per_token_forward_wall=observer.steps,
                                first_logit_ready_seconds=observer.prefill_finished - wall,
                                first_token_emitted_seconds=observer.emissions[0][1] - wall,
                                second_token_emitted_seconds=observer.emissions[1][1] - wall,
                                last_token_emitted_seconds=observer.emissions[-1][1] - wall,
                                token_emission_seconds=[t - wall for _, t, _ in observer.emissions],
                                emission_status="local_cpu_token_consumer_handoff_not_network_stream",
                                baseline_allocated_bytes=baseline_allocated,
                                baseline_reserved_bytes=baseline_reserved,
                                sampled_telemetry_path=str(output / "telemetry" / f"{key}.json"),
                                profiler_contaminated=profiling,
                            ),
                            trace_path=str(traces[0]),
                        )
                        if workload == "milebench" and stage not in {
                            "smoke",
                            "correctness",
                            "representation_canary",
                        }:
                            from mosaickv.evaluation.milebench import (
                                route_milebench_dataset,
                                score_milebench_sample,
                            )

                            route = route_milebench_dataset(
                                sample.milebench_dataset, sample.milebench_question_type
                            )
                            row["score"] = score_milebench_sample(
                                route, result.answer, sample.references[0], choices=sample.choices
                            )
                        row = asdict(CampaignRow(**row))
                        row.update(
                            variant=variant,
                            references=list(sample.references),
                            choices=list(getattr(sample, "choices", ())),
                            question_type=str(
                                getattr(sample, "milebench_question_type", "not_applicable")
                            ),
                            trace_sha256=sha(traces[0]),
                            boundary="A",
                            actual_method=current.method.value,
                            runtime_config_sha256=runtime_config_sha,
                            arithmetic_policy=arithmetic,
                            config_hash_contract="canonical runtime config plus arithmetic policy",
                            resolved_config_path=str(output / "resolved-configs" / f"{key}.json"),
                            resolved_config_file_sha256=sha(
                                output / "resolved-configs" / f"{key}.json"
                            ),
                            quality_decoding=trace.get("quality_decoding"),
                            experiment_arm_sha256=digest(
                                dict(
                                    runtime_config=canonical_config(current),
                                    variant=variant,
                                    requested_method=method,
                                    arithmetic=arithmetic,
                                )
                            ),
                            mechanism_audit_path=str(output / "mechanisms" / f"{key}.json")
                            if mechanism
                            else None,
                            mechanism_audit_sha256=sha(output / "mechanisms" / f"{key}.json")
                            if mechanism
                            else None,
                        )
                        if stage == "scoring":
                            from mosaickv.evaluation.fox_boundary import run_boundary_b

                            try:
                                boundary = run_boundary_b(
                                    adapter,
                                    current,
                                    messages,
                                    EvaluationRequest(
                                        key + "-B", sample.sample_id, task, messages, {}
                                    ),
                                    output / "boundary-traces" / key,
                                )
                                first_logits = boundary.pop("first_logits")
                                first_A = observer.first_logits
                                boundary.update(
                                    first_token_A=int(first_A.argmax(-1).item()),
                                    first_token_A_probability=float(
                                        first_A.softmax(-1).max().item()
                                    ),
                                    first_token_B_probability=float(
                                        first_logits.softmax(-1).max().item()
                                    ),
                                    A_B_first_logit_max_difference=float(
                                        (first_A - first_logits).abs().max().item()
                                    ),
                                    first_token_changed=int(first_A.argmax(-1).item())
                                    != boundary["first_token_id"],
                                    A_answer=row["answer"],
                                    A_B_answer_changed=row["answer"] != boundary["answer"],
                                )
                                if method == "full_kv":
                                    boundary["fullkv_chunked_logit_control_passed"] = bool(
                                        (first_A - first_logits).abs().max().item() <= 1e-4
                                    )
                                if workload == "milebench":
                                    boundary["task_score"] = score_milebench_sample(
                                        route,
                                        boundary["answer"],
                                        sample.references[0],
                                        choices=sample.choices,
                                    )
                            except Exception as error:
                                boundary = dict(
                                    boundary="B",
                                    status="blocked",
                                    error=f"{type(error).__name__}: {error}",
                                    primary_A_row_preserved=True,
                                )
                            write_once(output / "boundary" / f"{key}.json", boundary)
                    except Exception as error:
                        row.update(
                            status="failed",
                            error=f"{type(error).__name__}: {error}",
                            variant=variant,
                        )
                        summary["errors"].append(dict(key=key, error=row["error"]))
                        write_once(
                            output / "errors" / f"{key}.json",
                            dict(error=row["error"], traceback=traceback.format_exc()),
                        )
                    row["arm_execution_position"] = order_position
                    write_once(record_path, row)
                    summary["records"].append(
                        dict(path=str(record_path), sha256=sha(record_path), status=row["status"])
                    )
                    del runtime
                    settle(adapter)
        summary["status"] = "passed" if not summary["errors"] else "failed"
    except Exception as error:
        summary["errors"].append(
            dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        )
    finally:
        attention_transport.close()
        summary["wall_seconds"] = time.perf_counter() - start
        write_once(output / "worker.json", summary)
    if summary["status"] != "passed":
        raise SystemExit(1)
