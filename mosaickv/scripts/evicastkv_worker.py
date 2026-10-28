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
    import torch

    if not capture or len(token_ids) < 17:
        raise AssertionError("missing compressed representation or forced continuation")
    reference, compact = capture.pop("reference"), capture.pop("compact")
    differences = []
    agreements = []
    # A separate FP16 representation contract, never a relaxation of retention-1.
    atol = rtol = 8 * torch.finfo(torch.float16).eps
    for index in range(16):
        token = torch.tensor([[token_ids[index]]], device=adapter.device, dtype=torch.long)
        a = adapter.decode_one_token(
            token, reference, capture_queries=False, capture_attentions=False
        )
        b = adapter.decode_one_token(
            token, compact, capture_queries=False, capture_attentions=False
        )
        differences.append(float((a.logits.float() - b.logits.float()).abs().max().item()))
        agreements.append(bool(torch.equal(a.next_token_id, b.next_token_id)))
        if not torch.allclose(a.logits, b.logits, atol=atol, rtol=rtol):
            raise AssertionError(
                f"masked-source/packed representation mismatch at step {index}: {differences[-1]}"
            )
        if a.state.next_decode_position != b.state.next_decode_position:
            raise AssertionError("logical decode positions changed")
    return dict(
        capture,
        forced_steps=16,
        maximum_logit_difference=max(differences),
        per_step_max_logit_difference=differences,
        greedy_agreement=sum(agreements) / 16,
        tolerance=dict(
            atol=atol, rtol=rtol, contract="separate_fp16_8epsilon_representation_control"
        ),
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
    try:
        if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one allocated GPU required; no login-node GPU execution")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("deterministic cuBLAS must be set before Python starts")
        torch.set_num_threads(12)
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
        summary["model_load_seconds"] = time.perf_counter() - load_start
        processor_sha = digest(processor_signature(adapter.processor))
        gpu = asdict(capture_gpu_environment(torch))
        visible = [x for x in gpu["devices"] if x["visible_to_process"]]
        if len(visible) != 1 or "A100" not in visible[0]["name"]:
            raise RuntimeError("registered campaign requires one A100")
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
        write_once(output / "retention_one.json", parity)
        del prepared
        if stage == "smoke":
            arms = [
                ("full_kv", None, "complete"),
                ("mosaickv_exact", BUDGETS[workload][0], "complete"),
            ]
        elif stage == "correctness":
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
        else:
            arms = [("full_kv", None, "complete")] + [
                (m, b, "complete")
                for b in BUDGETS[workload]
                for m in METHODS[workload]
                if m != "full_kv"
            ]
        # Five full-request warmups per arm on its actual shape regime; no prefills shared.
        if stage == "measurement":
            for method, cap, variant in arms:
                current = resolved_config(lock, workload, method, cap, variant=variant)
                runtime = HuggingFaceMosaicKVModel(
                    adapter, current, trace_directory=output / "warmups" / f"{method}-{cap}"
                )
                for repeat in range(lock["warmups"]):
                    settle(adapter)
                    result = runtime.generate(
                        EvaluationRequest(f"warmup-{repeat}", first.sample_id, task, messages, {})
                    )
                    write_once(
                        output / "warmups" / f"{method}-{cap}" / f"{repeat}.json",
                        dict(metrics=asdict(result.metrics)),
                    )
                del runtime
        repeats = lock["repeats"] if stage == "measurement" else 1
        for position, sample in enumerate(samples):
            for repeat in range(repeats):
                offset = (position + repeat) % len(arms)
                ordered = arms[offset:] + arms[:offset]
                for method, cap, variant in ordered:
                    key = f"{job['indices'][position]:03d}-{method}-{cap}-{variant}-r{repeat}"
                    record_path = output / "records" / f"{key}.json"
                    current = resolved_config(lock, workload, method, cap, variant=variant)
                    config_sha = config_sha256(current)
                    trace_dir = output / "traces" / key
                    observer = generation_observer(adapter)
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
                        messages = build_multimodal_messages(
                            sample.prompt,
                            tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media),
                        )
                        write_once(
                            output / "resolved-configs" / f"{key}.json", canonical_config(current)
                        )
                        settle(adapter)
                        torch.cuda.reset_peak_memory_stats()
                        runtime = HuggingFaceMosaicKVModel(
                            adapter,
                            current,
                            trace_directory=trace_dir,
                            token_observer=observer.emit,
                        )
                        diagnostic = stage == "correctness"
                        with (
                            layout_capture(adapter)
                            if diagnostic
                            else contextlib.nullcontext({}) as captured,
                            observer.install(),
                        ):
                            torch.cuda.synchronize()
                            wall = time.perf_counter()
                            result = runtime.generate(
                                EvaluationRequest(key, sample.sample_id, task, messages, {})
                            )
                            torch.cuda.synchronize()
                            wall_seconds = time.perf_counter() - wall
                        traces = list(trace_dir.glob("*.json"))
                        if len(traces) != 1:
                            raise AssertionError("one complete trace required")
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
                            write_once(
                                output / "correctness" / f"{key}.json",
                                check_layout(adapter, captured, tokens),
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
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            e2e_wall_seconds=wall_seconds,
                            metrics=metrics,
                            timings=dict(
                                observer.phases,
                                per_token_forward_wall=observer.steps,
                                first_logit_ready_seconds=observer.prefill_finished - wall,
                                first_token_emitted_seconds=observer.emissions[0][1] - wall,
                                second_token_emitted_seconds=observer.emissions[1][1] - wall,
                                last_token_emitted_seconds=observer.emissions[-1][1] - wall,
                                token_emission_seconds=[t - wall for _, t, _ in observer.emissions],
                                emission_status="local_cpu_token_consumer_handoff_not_network_stream",
                            ),
                            trace_path=str(traces[0]),
                        )
                        if workload == "milebench":
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
                        )
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
        summary["wall_seconds"] = time.perf_counter() - start
        write_once(output / "worker.json", summary)
    if summary["status"] != "passed":
        raise SystemExit(1)
