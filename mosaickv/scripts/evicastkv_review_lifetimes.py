#!/usr/bin/env python3
"""Consumed-input whole-request allocator capture, never endpoint certification.

Use existing HF transaction instrumentation plus time-aligned allocator events.
Unknown generations remain unknown and block exact KV-peak certification.
Every request is instrumented identically; its timing is NOT latency evidence.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evicastkv_campaign import SOURCE, load_lock
from evicastkv_graph_profile import registered_plan

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

STAGE = "request_lifetimes"
ARMS = [
    ["full_kv", None, "complete"],
    ["prompt_attention_topk", 40, "complete"],
    ["mosaickv_exact", 40, "complete"],
]


def prepare(root, parent):
    if root.exists():
        raise FileExistsError("ownership diagnostic uses a new immutable root")
    revision = clean_source(SOURCE)
    lock, bindings, identities = registered_plan(parent)
    lock.update(name="fox_request_lifetimes_v1", source_root=str(SOURCE), source_sha=revision)
    lock["resources"].update(
        maximum_a100s=6,
        array_concurrency=2,
        six_a100s_explicitly_authorized=True,
        cpus_per_worker=12,
        wall_minutes=120,
    )
    lock["authorization"].update(automatic_stage_progression=False, confirmation_authorized=False)
    lock["historical_bindings"].update(bindings)
    lock[STAGE] = dict(
        sample_ids=identities,
        reference_root=str(parent),
        instrumented_only=True,
        max_allocator_entries=500000,
        unresolved_generations_never_assumed_non_KV=True,
        arithmetic_unchanged=True,
        fresh_inputs=False,
    )
    write_once(root / "protocol_lock.json", lock)
    write_once(
        root / "jobs" / f"{STAGE}.json",
        dict(
            protocol_sha256=sha(root / "protocol_lock.json"),
            jobs=[
                dict(workload="coco", subset="profiler", indices=[i], stage=STAGE, arms=ARMS)
                for i in range(2)
            ],
        ),
    )
    return dict(workers=2, spec_sha256=sha(root / "jobs" / f"{STAGE}.json"))


def flatten_pairs(pairs):
    return tuple(tensor for pair in pairs for tensor in pair)


@contextlib.contextmanager
def capture_generate(directory):
    import torch

    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.measurements.kv_ownership import KVOwnershipLedger
    from mosaickv.measurements.kv_timeline import CudaOwnershipSession

    original_generate = HuggingFaceMosaicKVModel.generate
    original_track = KVOwnershipLedger.track

    def generate(runtime, request):
        if Path(request.run_id).name != request.run_id:
            raise ValueError("unsafe request identity")
        session = CudaOwnershipSession(
            max_entries=500000, deferred_marks=True, clear_library_workspace=True
        )
        status, error = "failed", None
        result = None

        def track(ledger, role, pairs):
            materialized = tuple(pairs)
            original_track(ledger, role, materialized)
            session.mark(
                flatten_pairs(materialized),
                "kv_persistent",
                "existing HF explicit KV owner: " + role,
            )

        try:
            with session, contextlib.ExitStack() as stack:
                session.mark(
                    (*runtime.adapter.model.parameters(), *runtime.adapter.model.buffers()),
                    "model",
                    "pinned resident model parameters and buffers",
                )
                for layer in runtime.adapter._language_layers():
                    for name in ("k_proj", "v_proj"):
                        module = getattr(layer.self_attn, name)

                        def projected(_module, _args, output):
                            if isinstance(output, torch.Tensor):
                                session.mark(
                                    (output,),
                                    "kv_workspace",
                                    "actual language K/V projection output",
                                )

                        handle = module.register_forward_hook(projected)
                        stack.callback(handle.remove)
                stack.enter_context(patch.object(KVOwnershipLedger, "track", track))
                flags = dict(
                    request.generation_kwargs,
                    _mosaickv_capture_kv_ownership=True,
                    _mosaickv_capture_kv_transactions=True,
                )
                result = original_generate(runtime, replace(request, generation_kwargs=flags))
            status = "completed"
            return result
        except BaseException as exception:
            error = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            # These functions allocate only cache storage or explicit masks.
            # No broad model-forward/non-KV exemption is permitted.
            rules = (
                (
                    "/transformers/models/mistral/modeling_mistral.py",
                    "eager_attention_forward",
                    "kv_shared_workspace",
                ),
                (
                    "/transformers/models/mistral/modeling_mistral.py",
                    "apply_rotary_pos_emb",
                    "kv_shared_workspace",
                ),
                (
                    "/transformers/models/mistral/modeling_mistral.py",
                    "rotate_half",
                    "kv_shared_workspace",
                ),
                ("/transformers/cache_utils.py", "update", "kv_workspace"),
                ("/mosaickv/backends/hf_runtime.py", "_clone_decode_state", "kv_persistent"),
                ("/mosaickv/backends/hf_runtime.py", "_state_from_packed", "kv_workspace"),
            )
            session.timeline.classify_stacks(rules)
            classified = bool(session.timeline.allocations) and all(
                r["role"] != "unknown" for r in session.timeline.allocations.values()
            )
            report = session.report(opaque_workspace_classified=classified)
            report.update(
                status=status,
                error=error,
                run_id=request.run_id,
                sample_id=request.sample_id,
                instrumented_timing_not_headline=True,
                rules=[list(r) for r in rules],
                original_source_retirement_not_assumed=True,
            )
            write_once(directory / f"{request.run_id}.json", report)

    with patch.object(HuggingFaceMosaicKVModel, "generate", generate):
        yield


def worker(root, index, spec_sha):
    from evicastkv_worker import run_worker

    lock = load_lock(root)
    if index not in range(2) or not lock.get(STAGE):
        raise ValueError("only two consumed-input ownership workers registered")
    with capture_generate(root / "allocator" / str(index)):
        run_worker(root, STAGE, index, spec_sha)


def analyze(root):
    from evicastkv_analyze import collect
    from evicastkv_optimization import semantic_check

    lock = load_lock(root)
    _, _, rows, errors = collect(root, STAGE)
    _, _, reference, parent_errors = collect(Path(lock[STAGE]["reference_root"]), "profiling")
    errors.extend(parent_errors)
    reference = {(r["sample_id"], r["method"], r["budget_bytes"]): r for r in reference}
    reports = []
    for row in rows:
        try:
            if row["status"] != "completed":
                raise ValueError("failed ownership request retained")
            # Generation audit flags are not behavior-affecting config fields.
            semantic_check(reference[(row["sample_id"], row["method"], row["budget_bytes"])], row)
            paths = list((root / "allocator").glob(f"*/{row['run_id']}.json"))
            if len(paths) != 1:
                raise ValueError("exactly one time-aligned capture required")
            report = json.loads(paths[0].read_text())
            if report["status"] != "completed" or not report["complete_request"]:
                raise ValueError("incomplete allocator request")
            unknown = [r for r in report["records"] if r["role"] == "unknown"]
            stacks = report["stack_table"]
            summary = dict(
                sample_id=row["sample_id"],
                method=row["method"],
                allocator_path=str(paths[0]),
                allocator_sha256=sha(paths[0]),
                initial_active_bytes=row["initial_kv_bytes"],
                final_active_bytes=row["final_kv_bytes"],
                peak_active_bytes=row["peak_kv_bytes"],
                **{
                    k: report[k]
                    for k in (
                        "exact_whole_request_KV_peak_certified",
                        "exact_KV_peak_bytes",
                        "known_KV_peak_lower_bound_bytes",
                        "KV_peak_upper_bound_including_unknown_bytes",
                        "blocks_exact_certification",
                        "unknown_allocation_generations",
                    )
                },
            )
            summary["largest_unknown_generations"] = [
                dict(
                    bytes=r["bytes"],
                    frames=stacks[r["stack_id"]] if "stack_id" in r else [],
                    baseline=r["baseline"],
                )
                for r in sorted(unknown, key=lambda r: r["bytes"], reverse=True)[:20]
            ]
            reports.append(summary)
        except Exception as exception:
            errors.append(dict(run_id=row.get("run_id"), error=str(exception)))
    if len(rows) != 6 or len(reports) != 6:
        errors.append(dict(error="all two-input by three-method captures required"))
    result = dict(
        status="passed_capture" if not errors else "failed_capture",
        source_sha=lock["source_sha"],
        rows=len(rows),
        errors=errors,
        whole_request_KV_peak_certified=not errors
        and all(r["exact_whole_request_KV_peak_certified"] for r in reports),
        consumed_inputs_only=True,
        fresh_confirmation_authorized=False,
        headline_latency_eligible=False,
    )
    write_once(root / "analysis/ownership.json", reports)
    write_once(root / "analysis/GATE.json", result)
    return result
