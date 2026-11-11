#!/usr/bin/env python3
"""Fresh allocator trace for the streaming exact-BF16 source variant only.

This is not the old three-bit ownership request, a retry, or a peak certificate.
The window starts before media preparation and includes layer-local sealing,
fixed-prefix decoding and private-source cleanup. Named model owners are the
only baseline non-KV exemption; unexplained allocations remain unknown.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

import evicastkv_lossless_streaming_timing as timing
import evicastkv_source_ownership_closure as declared
import evicastkv_streaming_lossless_source as streamed
from evicastkv_source_ownership_v12 import _gzip_once
from evicastkv_source_trace_adjudication import load as load_metadata
from evicastkv_source_trace_codec import document_digests, encode

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.measurements.kv_timeline import CudaOwnershipSession

SOURCE = Path(__file__).resolve().parents[2]
KIND = "streaming_lossless_BF16_whole_request_ownership_v15"
GATE = "GATE_streaming_lossless_ownership.json"
POLICY = dict(
    method=KIND,
    predictions=128,
    continuation_updates=127,
    guard=timing.POLICY["guard"],
    first_prefill_logit_tolerance=0.001,
    max_allocator_entries=1_000_000,
    max_owner_marks=100_000,
    max_observations=512,
    worker_A100s=1,
    worker_CPUs=12,
    worker_minutes=20,
    soft_worker_seconds=1100,
    decode_checkpoints=[1, 16, 32, 64, 127],
    reference="saved_V13_FullKV_prefix_CPU_evaluator_only_no_reference_prefill_in_window",
    payload="all_original_positions_lossless_BF16_source_exact_bitpatches",
    observer="plain_CudaOwnershipSession_and_typed_DeclaredOwners_no_projection_hooks",
    source_rules="explicit_live_owners_only_no_legacy_stack_rules",
    unknown_baseline_extras_remain_unknown=True,
    final_snapshot_required=True,
    exact_KV_peak_certification_authorized=False,
    no_fresh_samples=True,
    no_latency_or_quality_claim=True,
)


def model_catalog(model):
    """Metadata only: exact backing identities/names, never parameter copies."""
    owners = {}
    for family, entries in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, value in entries:
            if not value.numel():
                continue
            storage = value.untyped_storage()
            key = str(value.device), storage.data_ptr(), storage._cdata
            try:
                version = value._version
            except RuntimeError:
                version = "untracked_inference_tensor"
            owner = owners.setdefault(
                key,
                dict(
                    device=str(value.device),
                    address=storage.data_ptr(),
                    storage_cdata=storage._cdata,
                    bytes=storage.nbytes(),
                    names=[],
                ),
            )
            if owner["bytes"] != storage.nbytes():
                raise ValueError("Model owner has inconsistent backing identity")
            owner["names"].append(
                dict(
                    name=name,
                    family=family,
                    shape=list(value.shape),
                    stride=list(value.stride()),
                    dtype=str(value.dtype),
                    storage_offset=value.storage_offset(),
                    version=version,
                )
            )
    rows = sorted(
        owners.values(), key=lambda row: (row["device"], row["address"], row["storage_cdata"])
    )
    if not rows:
        raise ValueError("Explicit actual named model parameter/buffer owners required")
    return dict(
        owners=rows,
        owner_catalog_sha256=digest(rows),
        total_backing_bytes=sum(row["bytes"] for row in rows),
        baseline_other_allocations_classified=False,
        tensor_values_or_GPU_owners_retained=False,
    )


def mark_baseline_model(session, catalog):
    """Named owners must match actual seeded backing bytes, not a subtraction."""
    matched = []
    for owner in catalog["owners"]:
        if not owner["device"].startswith("cuda"):
            continue
        key = session.timeline.active.get(owner["address"])
        row = session.timeline.allocations.get(key, {})
        if not row.get("baseline") or row.get("bytes") != owner["bytes"]:
            raise ValueError("Named model owner does not match an actual prewindow allocation")
        session.timeline.classify(
            owner["address"],
            "model",
            "pinned immutable resident model parameters/buffers actual named catalog "
            + catalog["owner_catalog_sha256"],
        )
        matched.append(
            dict(
                address=owner["address"],
                generation=key[1],
                bytes=owner["bytes"],
                names=owner["names"],
            )
        )
    if not matched:
        raise ValueError("At least one actual CUDA named model owner required")
    return matched


class TypedOwners:
    """Reuse weak owner tracking while assigning explicit KV role categories."""

    def __init__(self, session=None):
        self.ledger = declared.DeclaredOwners()
        self.session = session
        self.current_kind = ContextVar(f"lossless_owner_role_{id(self)}", default=None)
        self.ledger.session = self if session is not None else None

    def mark(self, values, ignored_old_role, evidence):
        kind = self.current_kind.get()
        if kind not in {"kv_persistent", "kv_workspace", "kv_shared_workspace"}:
            raise ValueError("An explicit source/lease/validation owner role is required")
        self.session.mark(values, kind, evidence)

    def track(self, name, values, *, kind="kv_persistent"):
        token = self.current_kind.set(kind)
        try:
            self.ledger.track(name, values)
        finally:
            self.current_kind.reset(token)

    def observe(self, phase, **extra):
        return self.ledger.observe(phase, **extra)


@contextmanager
def instrument(adapter, owners, *, deadline=None):
    """Observe actual cache APIs without adding forbidden K/V projection hooks."""
    from mosaickv.evaluation import (
        source_history,
        storage_runs,
        streaming_lossless_source,
        streaming_prefill,
    )
    from mosaickv.evaluation.graphed_storage import SharedSourceCache
    from mosaickv.evaluation.request_ownership import cache_tensors
    from mosaickv.evaluation.sealed_history import SealedHistoryLayer

    native_update = streaming_prefill.RetiringLayer.update
    native_encode = streaming_lossless_source.StreamingLosslessSource.__call__
    native_observe = streaming_prefill.RetiringPrefillCache.observe
    native_cache = storage_runs.cache_from_formats
    native_decode_observe = SharedSourceCache.observe
    native_history = SealedHistoryLayer.update
    native_source_input = source_history.SourceHistoryCache.source_input
    native_seal = source_history.seal_source_full_tails
    native_decode = adapter.decode_one_token
    counts = dict(
        prefill_layers=0,
        sealed_layers=0,
        decode_updates=0,
        history_seal_calls=0,
        constructor_owner_notifications=0,
        reader_owner_notifications=0,
    )
    in_encoder = ContextVar("lossless_encoder_scope", default=False)

    def check():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Bounded streaming ownership deadline reached")

    def update(layer, key, value, *args, **kwargs):
        check()
        owners.track(f"original_prefill_KV_layer_{layer.index}", (key, value))
        result = native_update(layer, key, value, *args, **kwargs)
        counts["prefill_layers"] += 1
        owners.observe(f"original_KV_layer_{layer.index}_published")
        return result

    def encode_layer(encoder, index, attention, original, args, kwargs, output):
        check()
        owners.track(
            f"normalized_input_shared_layer_{index}",
            kwargs.get("hidden_states", args[0] if args else None),
            kind="kv_shared_workspace",
        )
        owners.track(
            f"original_RoPE_shared_layer_{index}",
            kwargs.get("position_embeddings"),
            kind="kv_shared_workspace",
        )
        token = in_encoder.set(True)
        try:
            packet = native_encode(encoder, index, attention, original, args, kwargs, output)
        finally:
            in_encoder.reset(token)
        from mosaickv.evaluation.lossless_source_control import KIND as SOURCE_KIND

        if packet.kind != SOURCE_KIND:
            raise ValueError("Only the new exact lossless source packet is admissible")
        owners.track(f"lossless_source_packet_layer_{index}", packet.tensors())
        counts["sealed_layers"] += 1
        owners.observe(f"lossless_source_layer_{index}_constructed")
        return packet

    def observe_prefill(cache, extra=()):
        if in_encoder.get() and extra:
            owners.track("explicit_lossless_constructor_temporary", extra, kind="kv_workspace")
            counts["constructor_owner_notifications"] += 1
        return native_observe(cache, extra)

    def cache(*args, **kwargs):
        check()
        owners.observe("originals_retired_before_reader")
        value = native_cache(*args, **kwargs)
        owners.track("reader_initial_payload_and_metadata", cache_tensors(value))
        owners.observe("reader_constructed")
        return value

    def observe_decode(cache, extra=()):
        if extra:
            owners.track("explicit_reader_or_history_temporary", extra, kind="kv_workspace")
            counts["reader_owner_notifications"] += 1
        return native_decode_observe(cache, extra)

    def history(layer, key, value, *args, **kwargs):
        owners.track("incoming_generated_KV", (key, value), kind="kv_workspace")
        result = native_history(layer, key, value, *args, **kwargs)
        owners.track(
            f"generated_raw_tail_layer_{getattr(layer, 'prompt_length', 'unknown')}_{id(layer)}",
            layer.history,
        )
        return result

    def source_input(cache, index, hidden, rotary, positions):
        result = native_source_input(cache, index, hidden, rotary, positions)
        owners.track(
            f"generated_normalized_tail_layer_{index}", cache.layers[index].normalized_tail
        )
        owners.track("generated_original_phase_and_position_tail", cache.phase_tail)
        return result

    def seal(cache):
        result = native_seal(cache)
        if result:
            counts["history_seal_calls"] += 1
            for index, layer in enumerate(cache.layers):
                owners.track(
                    f"sealed_history_layer_{index}",
                    tuple(t for packet in layer.source_sealed for t in packet.tensors()),
                )
            owners.track(
                "sealed_history_original_positions_and_phases",
                (cache.history_position_chunks, cache.history_rotary_chunks),
            )
        return result

    def decode(*args, **kwargs):
        check()
        value = native_decode(*args, **kwargs)
        counts["decode_updates"] += 1
        if counts["decode_updates"] in POLICY["decode_checkpoints"]:
            owners.observe(f"decode_update_{counts['decode_updates']}_returned")
        return value

    with ExitStack() as stack:
        for owner, name, replacement in (
            (streaming_prefill.RetiringLayer, "update", update),
            (streaming_lossless_source.StreamingLosslessSource, "__call__", encode_layer),
            (streaming_prefill.RetiringPrefillCache, "observe", observe_prefill),
            (storage_runs, "cache_from_formats", cache),
            (SharedSourceCache, "observe", observe_decode),
            (SealedHistoryLayer, "update", history),
            (source_history.SourceHistoryCache, "source_input", source_input),
            (source_history, "seal_source_full_tails", seal),
            (adapter, "decode_one_token", decode),
        ):
            stack.enter_context(patch.object(owner, name, replacement))
        yield counts


def archive(path, session, owners, catalog, *, source_sha):
    """Typed complete DAG, including the actual final allocator snapshot."""
    if session.snapshot is None or not session.finished:
        raise ValueError("Finished actual allocator session/final snapshot required")
    capture = session.report(opaque_workspace_classified=False)
    capture.update(
        exact_whole_request_KV_peak_certified=False,
        exact_scoped_KV_peak_certified=False,
        exact_KV_peak_bytes=None,
        unknowns_not_assumed_non_KV=True,
        legacy_stack_rules_used=False,
    )
    payload = dict(
        schema=KIND,
        source_sha=source_sha,
        device=session.device,
        baseline_trace_count=session.baseline_trace_count,
        actual_events=session.timeline.events,
        allocator_records=capture["records"],
        stack_table=capture["stack_table"],
        capture_metadata=capture,
        declared_owner_observations=owners.ledger.observations,
        actual_owner_marks=session.marks,
        baseline_model_catalog=catalog,
        final_allocator_snapshot=session.snapshot,
        installed_and_local_source_hashes=declared.stack_source_hashes(capture["stack_table"]),
    )
    validate_bundle(payload)
    document = encode(payload)
    size = _gzip_once(path, document)
    semantic = document_digests(
        document,
        tuple(
            (key,)
            for key in (
                "actual_events",
                "allocator_records",
                "final_allocator_snapshot",
                "capture_metadata",
                "baseline_model_catalog",
            )
        ),
    )
    return dict(
        schema=KIND,
        path=str(path),
        sha256=sha(path),
        encoded_JSON_bytes=size,
        source_sha=source_sha,
        semantic_sha256={key[0]: value for key, value in semantic.items()},
        codec_statistics=document["statistics"],
        complete_final_snapshot_preserved=True,
        exact_whole_request_KV_peak_certified=False,
    )


def validate_bundle(value):
    if value.get("schema") != KIND or not isinstance(value.get("final_allocator_snapshot"), dict):
        raise ValueError("Distinct complete streaming-lossless ownership schema required")
    snapshot = value["final_allocator_snapshot"]
    if not {"segments", "device_traces"} <= set(snapshot):
        raise ValueError("Actual complete final allocator snapshot required")
    events = value["actual_events"]
    original = snapshot["device_traces"][value["device"]][value["baseline_trace_count"] :]
    if (
        len(events) != value["capture_metadata"]["events"]
        or len(events) != len(original)
        or any(
            dict((key, item) for key, item in event.items() if key != "sequence") != actual
            or event["sequence"] != index
            for index, (event, actual) in enumerate(zip(events, original, strict=True))
        )
    ):
        raise ValueError("Every final allocator event must match captured sequence exactly")
    if value["baseline_model_catalog"]["owner_catalog_sha256"] != digest(
        value["baseline_model_catalog"]["owners"]
    ):
        raise ValueError("Exact baseline model-owner catalog hash changed")
    if value["capture_metadata"]["exact_whole_request_KV_peak_certified"] is not False:
        raise ValueError("Source-refinement diagnostic cannot claim exact KV peak certification")
    return True


def load_bundle(path, receipt):
    if sha(path) != receipt["sha256"]:
        raise ValueError("Streaming lossless trace hash changed")
    value, document = load_metadata(
        path, expected_json_bytes=receipt["encoded_JSON_bytes"], with_codec=True
    )
    validate_bundle(value)
    actual = document_digests(document, tuple((key,) for key in receipt["semantic_sha256"]))
    if (
        any(actual[(key,)] != expected for key, expected in receipt["semantic_sha256"].items())
        or value["source_sha"] != receipt["source_sha"]
    ):
        raise ValueError("Complete trace semantic/source binding changed")
    return value


def verify_request(record, counts):
    if (
        record.get("arm") != timing.ARMS[1]
        or record.get("audited_precheck") is not True
        or record.get("original_native_KV_retired") is not True
        or record.get("private_source_objects_retired_after_response") is not True
        or record.get("constructor_all_word_validation") is not True
        or record.get("whole_request_KV_peak_certified") is not False
        or record["storage"]["history_updates_per_layer"] != [127] * 32
        or record["storage"]["outstanding_layer_leases"] != 0
        or counts["prefill_layers"] != 32
        or counts["sealed_layers"] != 32
        or counts["decode_updates"] != 127
        or counts["history_seal_calls"] != 7
    ):
        raise ValueError("Complete exact source/private-cache/history controls required")
    return True


def prepare(root, *, cpu_gate_root=None, parent=timing.PARENT):
    from evicastkv_review_followups import cpu_gate

    root = Path(root)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("New immutable scratch lossless ownership root required")
    revision = clean_source(SOURCE)
    prior, bindings = timing.require_parent(parent)
    if cpu_gate_root is None:
        raise ValueError("Complete current immutable CPU suite required")
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    bindings.update(prior["bindings"])
    for path in (
        Path(__file__),
        Path(timing.__file__),
        SOURCE / "mosaickv/src/mosaickv/evaluation/lossless_source_control.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/streaming_lossless_source.py",
        SOURCE / "mosaickv/src/mosaickv/measurements/kv_timeline.py",
        Path(declared.__file__),
        SOURCE / "mosaickv/scripts/evicastkv_source_trace_codec.py",
        Path(cpu_gate_root) / "full-cpu/cpu-tests.json",
    ):
        bindings[str(path)] = sha(path)
    write_once(
        root / "protocol.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            parent=str(Path(parent).resolve()),
            jobs=prior["jobs"],
            bindings=bindings,
            policy=POLICY,
            cpu_gate=cpu,
            expected_workers=2,
            no_new_confirmation=True,
        ),
    )
    return dict(workers=2, GPUs_per_worker=1, CPUs_per_worker=12, worker_minutes=20)


def validate(root):
    plan = declared.read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy"] != POLICY or any(
        sha(Path(path)) != expected for path, expected in plan["bindings"].items()
    ):
        raise ValueError("Lossless ownership source/configuration/evidence changed")
    timing.require_parent(plan["parent"])
    return plan


def worker(root, index=0):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_review_source_prefix import (
        MODEL,
        comparisons,
        forced_prefix,
        label_blind_payload,
    )
    from evicastkv_source_protection_diagnostic import require_isolated_slurm_gpu

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.hf_cli import _model_kwargs

    root = Path(root)
    plan = validate(root)
    if type(index) is not int or not 0 <= index < 2:
        raise ValueError("Exactly one of two consumed workers required")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    failures = []
    session = None
    owners = None
    catalog = None
    result = dict(source_sha=plan["source_sha"], index=index, status="failed", method=KIND)
    try:
        gpu = require_isolated_slurm_gpu(torch)
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        job = plan["jobs"][index]
        config = RunConfig.from_mapping(job["config"])
        if (
            (config.model.id, config.model.revision) != MODEL
            or config.model.precision.value != "bf16"
            or config.execution.attention_implementation != "eager"
        ):
            raise ValueError("Pinned BF16 full-MHA eager checkpoint required")
        with torch.inference_mode():
            op = streamed.cuda_operator_control()
        write_once(out / "GATE_cuda_operator.json", op)
        if op.get("passed") is not True:
            raise ValueError("Actual CUDA source/streaming control failed before weights")
        adapter = load_hf_adapter(
            MODEL[0],
            revision=MODEL[1],
            model_kwargs=_model_kwargs(config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = label_blind_payload(job["input"])
        saved = torch.load(
            Path(plan["parent"]) / "worker/logits.pt", map_location="cpu", weights_only=True
        )
        reference = saved[f"native_{index}"]
        prefix = forced_prefix(reference)
        catalog = model_catalog(adapter.model)
        write_once(out / "baseline_model_catalog.json", catalog)
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                protocol_sha256=sha(root / "protocol.json"),
                model=dataclasses.asdict(config.model),
                environment=environment(),
                GPU=gpu,
                method=KIND,
                baseline_catalog_sha256=sha(out / "baseline_model_catalog.json"),
                baseline_model_metadata_digest=catalog["owner_catalog_sha256"],
                job_id=os.environ.get("SLURM_JOB_ID"),
                no_labels=True,
                no_native_reference_forward_inside_window=True,
            ),
        )
        session = CudaOwnershipSession(
            max_entries=POLICY["max_allocator_entries"],
            deferred_marks=True,
            clear_library_workspace=False,
        )
        owners = TypedOwners(session)
        with torch.inference_mode(), session:
            baseline_matches = mark_baseline_model(session, catalog)
            write_once(
                out / "baseline_model_allocator_matches.json",
                dict(
                    catalog_sha256=catalog["owner_catalog_sha256"],
                    matches=baseline_matches,
                    other_baseline_owners_exempted=False,
                ),
            )
            owners.observe("before_media")
            with instrument(
                adapter, owners, deadline=start + POLICY["soft_worker_seconds"]
            ) as counts:
                logits, record = timing.request(
                    adapter,
                    payload,
                    timing.ARMS[1],
                    prefix=prefix,
                    audit=True,
                    independent_control=None,
                )
            owners.observe("after_request_private_cleanup")
        if catalog != model_catalog(adapter.model):
            raise ValueError("Actual named model owners changed during window")
        verify_request(record, counts)
        diagnostics = comparisons(reference, logits)
        prefill_difference = float((reference[0] - logits[0]).abs().max())
        torch.save(dict(reference=reference, candidate=logits), out / "logits.pt")
        write_once(
            out / "raw.json",
            dict(
                method=KIND,
                index=index,
                record=record,
                counters=counts,
                diagnostics=diagnostics,
                first_prefill_logit_difference=prefill_difference,
                source_sha=plan["source_sha"],
            ),
        )
        if (
            any(
                diagnostics[name]["registered_guard_pass"] is not True
                for name in (
                    "registered_guard_32",
                    "cache_dependent_guard_32",
                    "prefix_64",
                    "prefix_128",
                )
            )
            or prefill_difference > POLICY["first_prefill_logit_tolerance"]
        ):
            raise ValueError("Unchanged native prefill/32/128 numerical guard failed")
        if (
            counts["prefill_layers"] != 32
            or counts["sealed_layers"] != 32
            or counts["decode_updates"] != 127
        ):
            raise ValueError("Incomplete source retirement/history instrumentation")
        result.update(status="completed_numerical_diagnostic", all_32_128_guards_pass=True)
    except Exception as error:
        failures.append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        if session is not None and session.finished and owners is not None and catalog is not None:
            try:
                receipt = archive(
                    out / "allocator_trace.json.gz",
                    session,
                    owners,
                    catalog,
                    source_sha=plan["source_sha"],
                )
                write_once(out / "trace_receipt.json", receipt)
                result["complete_trace_with_final_snapshot_archived"] = True
            except Exception as error:
                failures.append(dict(archive_error=f"{type(error).__name__}: {error}"))
        result.update(
            status=result["status"] if not failures else "failed",
            failures=failures,
            wall_seconds=time.monotonic() - start,
            outputs={path.name: sha(path) for path in out.iterdir() if path.is_file()},
        )
        write_once(out / "result.json", result)
    return result


def analyze(root):
    import torch
    from evicastkv_review_source_prefix import (
        analysis_cpu_controls,
        comparisons,
        verify_recomputed_diagnostics,
    )

    root = Path(root)
    plan = validate(root)
    analysis_cpu_controls()
    failures = []
    summaries = []
    for job in plan["jobs"]:
        out = root / f"workers/{job['index']}"
        try:
            result = declared.read(out / "result.json")
            if (
                result["source_sha"] != plan["source_sha"]
                or result["status"] != "completed_numerical_diagnostic"
                or result["failures"]
                or result.get("complete_trace_with_final_snapshot_archived") is not True
            ):
                raise ValueError(
                    "Complete numerical request and fresh final-snapshot trace required"
                )
            for name, expected in result["outputs"].items():
                if Path(name).name != name or sha(out / name) != expected:
                    raise ValueError("Ownership producer output changed")
            value = load_bundle(
                out / "allocator_trace.json.gz", declared.read(out / "trace_receipt.json")
            )
            raw = declared.read(out / "raw.json")
            verify_request(raw["record"], raw["counters"])
            saved = torch.load(out / "logits.pt", map_location="cpu", weights_only=True)
            diagnostics = comparisons(saved["reference"], saved["candidate"])
            verify_recomputed_diagnostics(diagnostics, raw["diagnostics"])
            prefill_difference = float((saved["reference"][0] - saved["candidate"][0]).abs().max())
            if (
                prefill_difference != raw["first_prefill_logit_difference"]
                or prefill_difference > POLICY["first_prefill_logit_tolerance"]
            ):
                raise ValueError("Saved native prefill tolerance control failed")
            if any(
                diagnostics[name]["registered_guard_pass"] is not True
                for name in (
                    "registered_guard_32",
                    "cache_dependent_guard_32",
                    "prefix_64",
                    "prefix_128",
                )
            ):
                raise ValueError("Recomputed unchanged numerical guard failed")
            capture = value["capture_metadata"]
            structural = {
                "overlapping_address_without_free_completed",
                "free_without_observed_allocation",
                "partial_physical_segment_retirement",
                "final_live_backing_snapshot_mismatch",
                "CUDA_OOM_in_request",
                "empty_allocator_capture",
            }
            if (
                capture["complete_request"] is not True
                or capture["allocator_history_complete"] is not True
                or structural.intersection(capture["blocks_exact_certification"])
            ):
                raise ValueError("Whole allocator event structure/final snapshot is incomplete")
            summaries.append(
                dict(
                    index=job["index"],
                    task=job["task"],
                    capture_complete=capture["complete_request"],
                    allocator_history_complete=capture["allocator_history_complete"],
                    unknown_allocation_generations=capture["unknown_allocation_generations"],
                    known_KV_peak_lower_bound_bytes=capture["known_KV_peak_lower_bound_bytes"],
                    upper_including_unknown_bytes=capture[
                        "KV_peak_upper_bound_including_unknown_bytes"
                    ],
                    capture_blockers=capture["blocks_exact_certification"],
                    final_snapshot_preserved=True,
                    exact_whole_request_KV_peak_certified=False,
                )
            )
        except Exception as error:
            failures.append(dict(index=job["index"], error=f"{type(error).__name__}: {error}"))
    gate = dict(
        status="completed_lossless_ownership_diagnostic_not_certified"
        if not failures and len(summaries) == 2
        else "blocked_lossless_ownership",
        source_sha=plan["source_sha"],
        summaries=summaries,
        failures=failures,
        complete_numerical_and_trace_controls=not failures and len(summaries) == 2,
        exact_whole_request_KV_peak_certified=False,
        source_proof_refinement_required=True,
        independent_quality_N=0,
        superiority_established=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / GATE, gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--parent", type=Path, default=timing.PARENT)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    value = (
        prepare(args.root, cpu_gate_root=args.cpu_gate_root, parent=args.parent)
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
