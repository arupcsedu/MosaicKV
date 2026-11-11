#!/usr/bin/env python3
"""Paired lossless construction engineering on two consumed inputs only.

Both schedules retain every original BF16 K/V bit, identical prompt packets and
identical source-history decoding. Full requests include processor/media,
prefill, exact construction, retirement and 127 fixed-prefix decode updates.
Independent native controls and post-request tensor copies/hashes are outside
timers. Common output-logit collection inside storage_runs.run remains included.
No comparison to a nearest baseline, fresh quality claim or KV-peak certificate.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_lossless_source_control as exact
import evicastkv_streaming_lossless_source as streamed

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v13/run-20261111-1/streaming")
PARENT_SHA = "cc2768b2ba93e13682bc0d58e0e49a7856efd4c1"
ARMS = ("nonstreaming_lossless", "streaming_lossless")
GATE = "GATE_lossless_streaming_timing.json"
POLICY = dict(
    label="consumed_paired_streaming_vs_nonstreaming_lossless_full_request",
    parent_source_sha=PARENT_SHA,
    source_precision="original_BF16_all_positions_exact_sparse_bitpatches",
    constructor="unchanged_lossless_source_control.construct_all_word_validation",
    streaming_encoder="unchanged_StreamingLosslessSource",
    warmups=5,
    timed_repetitions=20,
    predictions=128,
    continuation_updates=127,
    seed=20261111,
    guard=exact.POLICY["guard"],
    native_repeat_atol=exact.POLICY["native_repeat_atol"],
    preflight="both_arms_full_ATen_audit_and_independent_native_all_KV_words",
    timing="no_per_ATen_audit_or_profiler_common_constructor_observer_disabled",
    reader="same_generic_two_buffer_source_history_direct_no_graphs",
    constructor_all_word_validation_retained=True,
    output_logit_collection_and_host_transfer_inside_reader_included_identically=True,
    independent_reference_and_post_request_checks_excluded=True,
    timing_peak_field="incomplete_decode_owned_snapshot_not_audited_peak",
    allocator_envelope_is_not_KV_peak=True,
    paired_bootstrap_replicates=10000,
    paired_latency_ratio_upper_at_most=1.0,
    workers=2,
    worker_A100s=1,
    worker_CPUs=12,
    worker_minutes=60,
    soft_worker_seconds=3500,
    no_new_ids=True,
    independent_quality_N=0,
    no_nearest_baseline_or_paper_quality_claim=True,
)


def orders(seed):
    generator = random.Random(seed)
    result = []
    for block in range(POLICY["warmups"] + POLICY["timed_repetitions"]):
        pair = list(ARMS)
        generator.shuffle(pair)
        result.append(
            dict(
                phase="warmup" if block < POLICY["warmups"] else "timed",
                trial=block if block < POLICY["warmups"] else block - POLICY["warmups"],
                method_order=pair,
            )
        )
    return result


def require_parent(root):
    root = Path(root)
    gate, protocol, result = (
        exact.read(root / name) for name in (streamed.GATE, "protocol.json", "worker/result.json")
    )
    if (
        gate.get("passed") is not True
        or gate.get("source_sha") != PARENT_SHA
        or protocol.get("source_sha") != PARENT_SHA
        or result.get("source_sha") != PARENT_SHA
        or gate.get("failures") != []
        or result.get("status") != "completed"
        or result.get("failures") != []
        or result.get("completed_samples") != 2
        or any(
            gate.get(field) is not True
            for field in (
                "independent_native_prefill_KV_and_logits_bitwise",
                "all_originals_retired_before_decode",
                "all_32_and_128_guards_pass",
            )
        )
    ):
        raise ValueError("Actual passed immutable V13 streaming prerequisite required")
    bindings = {
        str(root / name): sha(root / name)
        for name in (streamed.GATE, "protocol.json", "worker/result.json")
    }
    required = {"raw.jsonl", "logits.pt", "prefill_0.pt", "prefill_1.pt", "GATE_cuda_operator.json"}
    if not required <= set(result.get("outputs", {})):
        raise ValueError("Complete V13 streaming output manifest required")
    for name, expected in result["outputs"].items():
        if Path(name).name != name or sha(root / "worker" / name) != expected:
            raise ValueError("V13 streaming output hash changed")
        bindings[str(root / "worker" / name)] = expected
    return protocol, bindings


def prepare(root, *, cpu_gate_root=None, parent=PARENT):
    from evicastkv_review_followups import cpu_gate

    root, parent = Path(root), Path(parent)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("New immutable scratch timing root required")
    revision = clean_source(SOURCE)
    previous, bindings = require_parent(parent)
    if cpu_gate_root is None:
        raise ValueError("Complete current-source CPU gate required")
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    jobs = previous["jobs"]
    if len(jobs) != 2 or {job["task"] for job in jobs} != {"llava_description", "mmvet_prefixkv"}:
        raise ValueError("Only the same two consumed source inputs are permitted")
    bindings.update(previous["bindings"])
    for path in (
        Path(__file__),
        SOURCE / "mosaickv/src/mosaickv/evaluation/lossless_source_control.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/streaming_lossless_source.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/streaming_prefill.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/storage_runs.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/source_history.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/graphed_storage.py",
        SOURCE / "mosaickv/env/common/requirements.lock",
        Path(cpu_gate_root) / "full-cpu/cpu-tests.json",
    ):
        bindings[str(path)] = sha(path)
    write_once(
        root / "protocol.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            parent=str(parent.resolve()),
            historical_native_parent=previous["parent"],
            jobs=jobs,
            policy=POLICY,
            bindings=bindings,
            CPU_gate=cpu,
            orders={str(job["index"]): orders(POLICY["seed"] + job["index"]) for job in jobs},
            no_confirmation_or_paper_authorization=True,
        ),
    )
    return dict(workers=2, GPUs_per_worker=1, CPUs_per_worker=12, worker_minutes=60)


def validate(root):
    plan = exact.read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["policy"]) != digest(POLICY) or any(
        sha(Path(path)) != expected for path, expected in plan["bindings"].items()
    ):
        raise ValueError("Frozen timing source/configuration/evidence changed")
    require_parent(plan["parent"])
    return plan


def tensor_hash(tensors):
    import torch

    state = hashlib.sha256()
    for tensor in tensors:
        state.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        state.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return state.hexdigest()


def history_hash(cache):
    """Hash valid payload only, never uninitialized spare tail capacity."""
    tensors = []
    used = {layer.tail_used for layer in cache.layers}
    if len(used) != 1:
        raise ValueError("Synchronized generated-history tails required")
    for layer in cache.layers:
        for packet in layer.source_sealed:
            tensors.extend(packet.tensors())
        tensors.extend(t[..., : layer.tail_used, :] for t in layer.history)
        tensors.append(layer.normalized_tail[:, : layer.tail_used])
    if cache.phase_tail is not None:
        length = next(iter(used))
        tensors.extend(t[:, :length] for t in cache.phase_tail[:2])
        tensors.append(cache.phase_tail[2][:length])
    tensors.extend(cache.history_position_chunks)
    tensors.extend(t for pair in cache.history_rotary_chunks for t in pair)
    return tensor_hash(tensors)


def _refs(tensors):
    result = []
    for tensor in tensors:
        seen = set()
        while tensor is not None and id(tensor) not in seen:
            seen.add(id(tensor))
            result.append(weakref.ref(tensor))
            tensor = tensor._base
    return result


def nonstreaming_construct(adapter, prepared, prefill, normalized, rotary, *, audit):
    import torch

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.lossless_source_control import construct

    if audit:
        return exact.lossless_formats(
            adapter,
            prepared,
            prefill,
            SimpleNamespace(records={}, windows={}, priors={}),
            normalized,
            rotary,
        )
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    if set(normalized.records) != set(range(len(pairs))):
        raise ValueError("One full normalized source per original layer required")
    formats = []
    for index in range(len(pairs)):
        pair = pairs[index]
        positions = torch.arange(pair[0].shape[-2], dtype=torch.int64, device=pair[0].device)
        formats.append(
            construct(
                normalized.records[index],
                pair,
                adapter._language_layers()[index].self_attn,
                rotary.phases,
                positions,
                maximum_persistent_bytes=backing_bytes(pair),
                workspace_observer=None,
            )
        )
    return tuple(formats), [], {"constructor_all_word_validation": True, "audit": False}


def request(adapter, payload, arm, *, prefix, audit, independent_control=None):
    import evicastkv_request_validation as runtime
    import torch

    from mosaickv.evaluation import lossless_source_control, storage_runs, streaming_lossless_source
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention

    if arm not in ARMS or type(audit) is not bool or len(prefix) != 128:
        raise ValueError("Registered schedule, complete prefix and explicit audit required")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event, stop_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    started, cpu_started = time.perf_counter(), time.thread_time()
    process_started, phases = time.process_time(), {}
    start_event.record()

    def timed(name, function):
        begin = time.perf_counter()
        value = function()
        phases[name] = time.perf_counter() - begin
        return value

    prepared = timed(
        "media_and_processor", lambda: adapter.prepare_inputs(runtime.load_messages(payload))
    )
    if arm == ARMS[0]:
        with SourceCapture(adapter) as normalized, PhaseCapture(adapter) as rotary:
            prefill = timed(
                "native_prefill_with_required_source_capture",
                lambda prepared=prepared: adapter.prefill(
                    prepared, capture_queries=False, capture_attentions=False
                ),
            )
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        original_refs = _refs(tuple(t for pair in pairs for t in pair))
        raw = backing_bytes(tuple(t for pair in pairs for t in pair))
        packets, _, construction = timed(
            "lossless_construction",
            lambda prefill=prefill, prepared=prepared, normalized=normalized, rotary=rotary: (
                nonstreaming_construct(adapter, prepared, prefill, normalized, rotary, audit=audit)
            ),
        )
        prefill.state.past_key_values = None
        normalized.records.clear()
        rotary.phases = None
        del pairs, normalized, rotary
    else:
        cache = RetiringPrefillCache(len(adapter._language_layers()), audit=audit)
        encoder = streaming_lossless_source.StreamingLosslessSource(adapter, prepared, cache)
        fresh = dataclasses.replace(
            prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache)
        )
        original_constructor = lossless_source_control.construct

        def common_no_observer(*args, **kwargs):
            kwargs["workspace_observer"] = None
            return original_constructor(*args, **kwargs)

        # Only instrumentation is disabled, identically to the direct timed
        # nonstreaming constructor. No arithmetic/check/cap is bypassed.
        constructor = original_constructor if audit else common_no_observer
        with (
            patch.object(streaming_lossless_source, "construct", constructor),
            SealAfterAttention(adapter, cache, encoder),
        ):
            prefill = timed(
                "prefill_including_lossless_construction",
                lambda fresh=fresh: adapter.prefill(
                    fresh, capture_queries=False, capture_attentions=False
                ),
            )
        construction, packets = encoder.finish(), cache.formats()
        raw, original_refs = construction["original_raw_KV_bytes"], tuple(cache.original_refs)
        prefill.state.past_key_values = None
        fresh.model_inputs.clear()
        del fresh, cache, encoder
    begin = time.perf_counter()
    prepared.model_inputs.clear()
    del prepared
    gc.collect()
    if any(ref() is not None for ref in original_refs):
        raise ValueError("Original native K/V survives before common reader decoding")
    phases["retire_original_sources"] = time.perf_counter() - begin
    # A separate independent control is used only during untimed preflight.
    if independent_control is not None:
        if not audit:
            raise ValueError("Independent CPU reference copies forbidden in timed requests")
        streamed.verify_prefill(adapter, packets, prefill, independent_control)
    initial_packet_hash = (
        tensor_hash(tuple(t for p in packets for t in p.tensors())) if audit else None
    )
    holder, native_cache_builder = [], storage_runs.cache_from_formats

    def capture_cache(*args, holder=holder, **kwargs):
        value = native_cache_builder(*args, **kwargs)
        holder.append(value)
        return value

    with patch.object(storage_runs, "cache_from_formats", capture_cache):
        logits, record = timed(
            "reader_construction_and_127_decode_updates",
            lambda prefill=prefill, packets=packets: storage_runs.run(
                adapter,
                prefill,
                packets,
                tokens=128,
                forced=prefix,
                audit=audit,
                source_history=True,
                source_history_constructor="direct",
                shared_workspace=True,
                workspace_graph=False,
                initial_physical_byte_cap=raw,
            ),
        )
    stop_event.record()
    torch.cuda.synchronize()
    timing = dict(
        full_request_wall_seconds=time.perf_counter() - started,
        calling_thread_CPU_seconds=time.thread_time() - cpu_started,
        process_CPU_seconds=time.process_time() - process_started,
        CUDA_stream_interval_ms=start_event.elapsed_time(stop_event),
        CUDA_interval_contains_host_launch_gaps_not_pure_kernel_time=True,
    )
    allocator_peak = int(torch.cuda.max_memory_allocated())
    allocator_reserved_peak = int(torch.cuda.max_memory_reserved())
    if len(holder) != 1 or holder[0].workspace:
        raise ValueError("Exactly one released request-private decode workspace required")
    packet_hash = tensor_hash(tuple(t for p in packets for t in p.tensors()))
    if audit and packet_hash != initial_packet_hash:
        raise ValueError("Independent preflight reader mutated prompt source packet")
    history = history_hash(holder[0])
    refs = _refs(tuple(t for p in packets for t in p.tensors()))
    storage = record["storage"]
    result = dict(
        arm=arm,
        timing=timing,
        phase_wall_seconds=phases,
        audited_precheck=audit,
        source_packet_sha256=packet_hash,
        generated_history_valid_payload_sha256=history,
        original_native_KV_retired=True,
        persistent_format_bytes=backing_bytes(tuple(t for p in packets for t in p.tensors())),
        storage=storage,
        whole_request_allocator_peak_bytes=allocator_peak,
        whole_request_allocator_reserved_peak_bytes=allocator_reserved_peak,
        allocator_snapshot_precedes_post_response_hash_copies=True,
        phase_intervals_are_host_wall_without_added_per_phase_CUDA_barriers=True,
        response_boundary_includes_media_prefill_constructor_retirement_reader_and_logit_collection=True,
        excludes="independent_native_reference_and_CPU_KV_controls,post_response_hashes_and_cleanup",
        whole_request_KV_peak_certified=False,
        timing_peak_is_incomplete_snapshot=not audit,
        constructor_all_word_validation=True,
        no_per_ATen_audit_during_timing=not audit,
        storage_audit_scope="common_decode_owned_tensor_backing_not_whole_request"
        if audit
        else "incomplete_decode_owned_snapshot",
        no_shared_byte_baseline_or_quality_claim=True,
    )
    holder.clear()
    del packets, prefill
    gc.collect()
    if any(ref() is not None for ref in refs):
        raise ValueError("Request source owners leak across paired repetitions")
    result["private_source_objects_retired_after_response"] = True
    return logits, result


def match_pair(left, right, *, audited):
    if left["arm"] != ARMS[0] or right["arm"] != ARMS[1]:
        raise ValueError("Two distinct registered schedules required")
    fields = ["initial_active_kv_storage_bytes", "final_active_kv_storage_bytes"]
    if audited:
        fields.append("peak_active_kv_storage_bytes")
    if any(left["storage"][key] != right["storage"][key] for key in fields) or any(
        left[key] != right[key]
        for key in (
            "source_packet_sha256",
            "generated_history_valid_payload_sha256",
            "persistent_format_bytes",
        )
    ):
        raise ValueError(
            "Lossless packet/history bits or declared physical storage boundaries differ"
        )
    for row in (left, right):
        if (
            row["audited_precheck"] is not audited
            or row["original_native_KV_retired"] is not True
            or row["constructor_all_word_validation"] is not True
            or row["private_source_objects_retired_after_response"] is not True
            or row["storage"]["history_updates_per_layer"] != [127] * 32
            or row["storage"]["outstanding_layer_leases"] != 0
            or any(
                type(row["storage"][key]) is not int or row["storage"][key] < 1 for key in fields
            )
        ):
            raise ValueError(
                "Complete unchanged history/retirement/physical-byte controls required"
            )
    return True


def worker(root, index=0):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_review_source_prefix import (
        MODEL,
        comparisons,
        forced_prefix,
        label_blind_payload,
        native_reference,
    )
    from evicastkv_source_protection_diagnostic import require_isolated_slurm_gpu

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.hf_cli import _model_kwargs

    root = Path(root)
    plan = validate(root)
    if type(index) is not int or not 0 <= index < 2:
        raise ValueError("Exactly one of two registered input workers required")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    start, rows, failures, snapshots = time.monotonic(), [], [], {}
    result = dict(status="failed", source_sha=plan["source_sha"], index=index)
    expected = [
        (block, arm) for block in plan["orders"][str(index)] for arm in block["method_order"]
    ]
    try:
        gpu = require_isolated_slurm_gpu(torch)
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(POLICY["seed"])
        job = plan["jobs"][index]
        config = RunConfig.from_mapping(job["config"])
        if (
            (config.model.id, config.model.revision) != MODEL
            or config.model.precision.value != "bf16"
            or config.execution.attention_implementation != "eager"
        ):
            raise ValueError("Frozen pinned BF16 eager full-MHA model required")
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                protocol_sha256=sha(root / "protocol.json"),
                model=dataclasses.asdict(config.model),
                config_sha256=config_sha256(config),
                environment=environment(),
                GPU=gpu,
                job_id=os.environ.get("SLURM_JOB_ID"),
                consumed_only=True,
                independent_quality_N=0,
                no_nearest_baseline_claim=True,
                CUDA=torch.version.cuda,
            ),
        )
        _, adapter = exact.operator_before_model(
            lambda: _operator(torch, out),
            lambda: load_hf_adapter(
                MODEL[0],
                revision=MODEL[1],
                model_kwargs=_model_kwargs(config, local_files_only=True),
                processor_kwargs={"local_files_only": True},
            ),
        )
        with (
            torch.inference_mode(),
            exact.bounded_adapter(adapter, start + POLICY["soft_worker_seconds"]),
        ):
            payload = label_blind_payload(job["input"])
            native, native_record = native_reference(adapter, payload)
            historical = torch.load(
                Path(plan["historical_native_parent"]) / f"workers/{index}/logits.pt",
                map_location="cpu",
                weights_only=True,
            )["fullkv_native_reference"]
            repeat = comparisons(historical, native)
            if (
                repeat["prefix_128"]["token_agreement"] != 1.0
                or repeat["prefix_128"]["maximum_logit_difference"] > POLICY["native_repeat_atol"]
            ):
                raise ValueError("Native historical 128-token repeat control failed")
            snapshots["native"] = native
            control = streamed.native_prefill_control(adapter, payload)
            prefix = forced_prefix(native)
            audited, preflight_logits = {}, {}
            for arm in ARMS:
                logits, row = request(
                    adapter, payload, arm, prefix=prefix, audit=True, independent_control=control
                )
                diagnostics = comparisons(native, logits)
                if any(
                    diagnostics[name]["registered_guard_pass"] is not True
                    for name in (
                        "registered_guard_32",
                        "cache_dependent_guard_32",
                        "prefix_64",
                        "prefix_128",
                    )
                ):
                    raise ValueError("Audited numerical preflight failed; timing blocked")
                audited[arm], preflight_logits[arm] = dict(row, diagnostics=diagnostics), logits
                snapshots[f"preflight_{arm}"] = logits
            if tensor_hash((preflight_logits[ARMS[0]],)) != tensor_hash(
                (preflight_logits[ARMS[1]],)
            ):
                raise ValueError("Audited schedules differ in 128 actual logit bits")
            match_pair(audited[ARMS[0]], audited[ARMS[1]], audited=True)
            write_once(
                out / "GATE_preflight.json",
                dict(
                    passed=True,
                    source_sha=plan["source_sha"],
                    rows=audited,
                    native_repeat=repeat,
                    native_record=native_record,
                    packet_constructor_sha256=sha(
                        SOURCE / "mosaickv/src/mosaickv/evaluation/lossless_source_control.py"
                    ),
                ),
            )
            del control
            gc.collect()
            for block, arm in expected:
                row = dict(block, arm=arm, status="failed", sample_id=job["sample_id"])
                try:
                    if time.monotonic() - start >= POLICY["soft_worker_seconds"]:
                        raise TimeoutError("Frozen paired timing worker deadline reached")
                    logits, record = request(adapter, payload, arm, prefix=prefix, audit=False)
                    observed_logit_hash = tensor_hash((logits,))
                    if observed_logit_hash != tensor_hash((preflight_logits[arm],)):
                        raise ValueError("Timed request changed audited 128-logit bits")
                    if any(
                        record[key] != audited[arm][key]
                        for key in (
                            "source_packet_sha256",
                            "generated_history_valid_payload_sha256",
                            "persistent_format_bytes",
                        )
                    ):
                        raise ValueError(
                            "Timed lossless packet/history differs from audited precheck"
                        )
                    for key in ("initial_active_kv_storage_bytes", "final_active_kv_storage_bytes"):
                        if record["storage"][key] != audited[arm]["storage"][key]:
                            raise ValueError(
                                "Timed physical endpoint differs from audited precheck"
                            )
                    row.update(
                        record,
                        status="completed",
                        all_128_logits_bitwise=True,
                        all_128_logits_sha256=observed_logit_hash,
                    )
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    failures.append(
                        dict(phase=row["phase"], trial=row["trial"], arm=arm, error=row["error"])
                    )
                rows.append(row)
                with (out / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                if failures:
                    break
    except Exception as error:
        failures.append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        for block, arm in expected[len(rows) :]:
            row = dict(
                block,
                arm=arm,
                status="blocked_not_executed",
                reason="earlier correctness/setup/deadline failure",
            )
            rows.append(row)
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        torch.save(snapshots, out / "preflight_logits.pt")
        result.update(
            status="completed" if not failures and len(rows) == 50 else "failed",
            observations=len(rows),
            completed_observations=sum(row["status"] == "completed" for row in rows),
            failures=failures,
            wall_seconds=time.monotonic() - start,
            outputs={path.name: sha(path) for path in out.iterdir() if path.is_file()},
        )
        write_once(out / "result.json", result)
    return result


def _operator(torch, out):
    try:
        with torch.inference_mode():
            result = streamed.cuda_operator_control()
    except Exception as error:
        write_once(
            out / "GATE_cuda_operator.json",
            dict(
                passed=False,
                error=f"{type(error).__name__}: {error}",
                pretrained_checkpoint_loaded=False,
            ),
        )
        raise
    write_once(out / "GATE_cuda_operator.json", result)
    return result


def validate_trials(rows, expected_order):
    expected = [
        (block["phase"], block["trial"], arm)
        for block in expected_order
        for arm in block["method_order"]
    ]
    keys = [(row["phase"], row["trial"], row["arm"]) for row in rows]
    if (
        len(rows) != 50
        or keys != expected
        or any(
            row["status"] != "completed" or row["all_128_logits_bitwise"] is not True
            for row in rows
        )
    ):
        raise ValueError("All fifty unique registered warmup/timed observations required")
    for index in range(0, len(rows), 2):
        pair = {row["arm"]: row for row in rows[index : index + 2]}
        match_pair(pair[ARMS[0]], pair[ARMS[1]], audited=False)
        for row in pair.values():
            if row["method_order"] != expected_order[index // 2]["method_order"] or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for key, value in row["timing"].items()
                if key.endswith("seconds") or key.endswith("_ms")
            ):
                raise ValueError("Complete finite frozen-order wall/CPU/CUDA observations required")
    return {
        arm: [row for row in rows if row["phase"] == "timed" and row["arm"] == arm] for arm in ARMS
    }


def analyze(root):
    import torch
    from evicastkv_review_source_prefix import (
        analysis_cpu_controls,
        comparisons,
        verify_recomputed_diagnostics,
    )

    from mosaickv.evaluation.paired_inline_latency import summarize

    root, plan = Path(root), validate(root)
    analysis_cpu_controls()
    failures, summaries, bindings = [], [], {}
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        try:
            result = exact.read(folder / "result.json")
            if (
                result["source_sha"] != plan["source_sha"]
                or result["status"] != "completed"
                or result["failures"]
            ):
                raise ValueError("Complete immutable paired producer required")
            for name, expected in result["outputs"].items():
                if Path(name).name != name or sha(folder / name) != expected:
                    raise ValueError("Paired timing producer artifact changed")
                bindings[str(folder / name)] = expected
            preflight = exact.read(folder / "GATE_preflight.json")
            if preflight["passed"] is not True or preflight["source_sha"] != plan["source_sha"]:
                raise ValueError("Audited precheck failed or source changed")
            match_pair(preflight["rows"][ARMS[0]], preflight["rows"][ARMS[1]], audited=True)
            saved = torch.load(
                folder / "preflight_logits.pt", map_location="cpu", weights_only=True
            )
            for arm in ARMS:
                recomputed = comparisons(saved["native"], saved[f"preflight_{arm}"])
                verify_recomputed_diagnostics(recomputed, preflight["rows"][arm]["diagnostics"])
                if any(
                    recomputed[name]["registered_guard_pass"] is not True
                    for name in (
                        "registered_guard_32",
                        "cache_dependent_guard_32",
                        "prefix_64",
                        "prefix_128",
                    )
                ):
                    raise ValueError("Recomputed audited numerical guard failed")
            if tensor_hash((saved[f"preflight_{ARMS[0]}"],)) != tensor_hash(
                (saved[f"preflight_{ARMS[1]}"],)
            ):
                raise ValueError("Saved audited 128-logit bits differ")
            rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            groups = validate_trials(rows, plan["orders"][str(job["index"])])
            if any(
                row.get("all_128_logits_sha256") != tensor_hash((saved[f"preflight_{arm}"],))
                for arm in ARMS
                for row in groups[arm]
            ):
                raise ValueError("Timed logit hashes do not bind audited saved predictions")
            stats = summarize(
                [row["timing"]["full_request_wall_seconds"] for row in groups[ARMS[0]]],
                [row["timing"]["full_request_wall_seconds"] for row in groups[ARMS[1]]],
                seed=POLICY["seed"] + job["index"],
            )
            summaries.append(
                dict(
                    index=job["index"],
                    task=job["task"],
                    full_request_latency=stats,
                    audited_storage=preflight["rows"][ARMS[0]]["storage"],
                    no_whole_request_KV_peak_claim=True,
                )
            )
        except Exception as error:
            failures.append(dict(index=job["index"], error=f"{type(error).__name__}: {error}"))
    complete = not failures and len(summaries) == 2
    gate = dict(
        status="completed_paired_lossless_construction_diagnostic"
        if complete
        else "blocked_paired_lossless_timing",
        source_sha=plan["source_sha"],
        complete=complete,
        failures=failures,
        summaries=summaries,
        all_strict_paired_latency_gates_pass=complete
        and all(
            row["full_request_latency"]["paired_latency_engineering_gate"] for row in summaries
        ),
        measured_inputs=2,
        independent_quality_N=0,
        superiority_over_nearest_baseline_established=False,
        whole_request_KV_peak_certified=False,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
        bindings=bindings,
    )
    write_once(root / GATE, gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    value = (
        prepare(args.root, cpu_gate_root=args.cpu_gate_root, parent=args.parent)
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
