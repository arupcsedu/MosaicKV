#!/usr/bin/env python3
"""Bounded nonstreaming BF16 source ownership study, not an exact peak license.

Only the consumed MM-Vet request and its registered source policy are used.
Declared live-owner observations augment the existing cross-thread allocator
capture. Unknown generations remain unknown: checkpoint maxima do not certify
transients, whole-device memory, or a superiority claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import time
import weakref
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.measurements.kv_timeline import AllocationTimeline

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/audits/fox-review-v7-closure-20261110/source-prefix")
ARM = "source3_attention_optional"
POLICY = dict(
    arm=ARM,
    parent_index=1,
    predictions=128,
    continuation_updates=127,
    max_allocator_entries=1000000,
    max_owner_marks=100000,
    max_observations=512,
    soft_worker_seconds=1100,
    decode_checkpoints=[1, 16, 32, 64, 127],
    source_bits=3,
    protection_fractions=[0.0, 0.03, 0.05],
    no_policy_change=True,
    no_benchmark_labels=True,
    no_latency_claim=True,
    no_fresh_cohort=True,
    exact_peak_certification_authorized=False,
)
BLIND_SPOTS = [
    "Declared checkpoints do not observe every constructor or decoder transient.",
    "Unreviewed allocator generations remain unknown, never assumed non-KV.",
    "The allocator upper envelope excludes vendor allocations outside PyTorch.",
    "Requested tensor backing bytes exclude allocator rounding and reservation.",
    "CPU residuals and observer CPU metadata are separate from CUDA KV owners.",
    "Earlier line-specific stack attributions are not accepted as source certification.",
]


def read(path):
    return json.loads(Path(path).read_text())


def tensors(value):
    """Traverse explicit tensor containers, never arbitrary model attributes."""
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


class DeclaredOwners:
    """Weak references only; observations must not extend GPU-owner lifetimes."""

    def __init__(self):
        self.roles, self.marked, self.observations = {}, {}, []
        self.session = None
        self.marks = 0

    def track(self, role, values):
        values = tuple(tensors(values))
        self.roles[role] = tuple(weakref.ref(value) for value in values)
        fresh = []
        for value in values:
            if not value.numel():
                continue
            storage = value.untyped_storage()
            key = str(value.device), storage.data_ptr(), storage._cdata
            previous = self.marked.get(key)
            if previous is None or previous() is None:
                fresh.append(value)
                self.marked[key] = weakref.ref(value)
        self.marks += len(fresh)
        if self.marks > POLICY["max_owner_marks"]:
            raise ValueError("bounded declared-owner mark budget exceeded")
        if fresh and self.session is not None:
            self.session.mark(fresh, "kv_persistent", f"source_ownership_declared:{role}")

    def observe(self, phase, **extra):
        if len(self.observations) >= POLICY["max_observations"]:
            raise ValueError("bounded owner-observation count exceeded")
        before = time.time_ns() // 1000
        owners = {}
        for role, refs in self.roles.items():
            for reference in refs:
                value = reference()
                if value is None or not value.numel():
                    continue
                storage = value.untyped_storage()
                key = str(value.device), storage.data_ptr(), storage._cdata
                row = owners.setdefault(
                    key,
                    dict(
                        device=str(value.device),
                        address=storage.data_ptr(),
                        storage_cdata=storage._cdata,
                        bytes=storage.nbytes(),
                        roles=[],
                    ),
                )
                if role not in row["roles"]:
                    row["roles"].append(role)
        after = time.time_ns() // 1000
        result = dict(
            phase=phase,
            begin_us=before,
            end_us=after,
            owners=list(owners.values()),
            declared_cuda_bytes=sum(
                row["bytes"] for row in owners.values() if row["device"].startswith("cuda")
            ),
            declared_CPU_bytes=sum(
                row["bytes"] for row in owners.values() if row["device"].startswith("cpu")
            ),
            checkpoint_not_exact_transient_peak=True,
            **extra,
        )
        self.observations.append(result)
        return result


def generation_for_observation(records, owner, observation, clock):
    """Require the complete observation interval inside exactly one generation."""
    if not clock or not owner["device"].startswith("cuda"):
        return None
    begin = observation["begin_us"] - clock["maximum_offset_us"]
    end = observation["end_us"] - clock["minimum_offset_us"]
    matches = []
    for row in records:
        birth = float("-inf") if row["baseline"] else row.get("time_us")
        death = float("inf") if row["end"] is None else row.get("end_time_us")
        if (
            row["address"] == owner["address"]
            and row["bytes"] == owner["bytes"]
            and birth is not None
            and death is not None
            and birth <= begin
            and death >= end
        ):
            matches.append(row)
    return matches[0] if len(matches) == 1 else None


def conservative_report(capture, observations, *, events):
    """Rebuild bounds, ignoring inherited unreviewed stack classifications."""
    rows = [dict(row, evidence=list(row.get("evidence", ()))) for row in capture["records"]]
    for row in rows:
        # Actual resident parameter/buffer identity is the sole non-KV exemption.
        model = any(
            "pinned immutable resident model parameters/buffers" in text for text in row["evidence"]
        )
        row["role"] = "model" if model else "unknown"
    by_identity = {(row["address"], row["generation"]): row for row in rows}
    matched, unresolved = 0, []
    for observation in observations:
        for owner in observation["owners"]:
            if not owner["device"].startswith("cuda"):
                continue
            match = generation_for_observation(
                rows, owner, observation, capture.get("clock_calibration")
            )
            if match is None:
                unresolved.append(
                    dict(
                        phase=observation["phase"],
                        address=owner["address"],
                        bytes=owner["bytes"],
                        reason="generation_not_unique",
                    )
                )
                continue
            key = match["address"], match["generation"]
            row = by_identity[key]
            if row["role"] == "model":
                raise ValueError("declared KV aliases an immutable model owner")
            row["role"] = "kv_persistent"
            row["evidence"].append(f"declared_live_owner:{observation['phase']}")
            owner["allocation_generation"] = row["generation"]
            matched += 1
    # The independent event reducer counts aliases once and preserves reuse.
    timeline = AllocationTimeline()
    timeline.allocations = {(row["address"], row["generation"]): row for row in rows}
    if len(events) != capture["events"]:
        raise ValueError("actual allocator event count differs from captured record")
    timeline.events = events
    timeline.stacks = capture["stack_table"]
    timeline.errors = list(capture.get("blocks_exact_certification", ()))
    derived = timeline.report(
        complete=capture["complete_request"],
        history_complete=capture["allocator_history_complete"],
        opaque_workspace_classified=False,
    )
    structural_errors = {
        "overlapping_address_without_free_completed",
        "free_without_observed_allocation",
        "partial_physical_segment_retirement",
        "final_live_backing_snapshot_mismatch",
        "CUDA_OOM_in_request",
        "empty_allocator_capture",
    }
    bounds_valid = bool(
        capture["allocator_history_complete"]
        and capture["complete_request"]
        and not structural_errors.intersection(capture.get("blocks_exact_certification", ()))
    )
    if not bounds_valid:
        derived["KV_peak_upper_bound_including_unknown_bytes"] = None
        derived["known_KV_peak_lower_bound_bytes"] = None
    derived.update(
        exact_whole_request_KV_peak_certified=False,
        exact_scoped_KV_peak_certified=False,
        exact_KV_peak_bytes=None,
        whole_device_memory_certified=False,
        declared_owner_marks_linked=matched,
        unresolved_owner_observations=unresolved,
        observed_declared_cuda_peak_lower_bound_bytes=max(
            (item["declared_cuda_bytes"] for item in observations), default=0
        ),
        unknown_not_assumed_non_KV=True,
        allocator_interval_bounds_valid=bounds_valid,
        inherited_stack_classifications_discarded=True,
        blind_spots=BLIND_SPOTS,
        checkpoint_observations_not_exact_transient_certificate=True,
        superiority_established=False,
    )
    return derived


def stack_source_hashes(stack_table):
    """Pin real operator/source files; unavailable JIT frames stay explicit."""
    result = {}
    for filename in sorted({frame[0] for frames in stack_table for frame in frames}):
        path = Path(filename)
        if (
            path.is_absolute()
            and path.suffix in {".py", ".pyi", ".cpp", ".cc", ".h", ".cu"}
            and path.is_file()
        ):
            result[filename] = dict(sha256=sha(path), canonical_path=str(path.resolve()))
        else:
            result[filename] = dict(status="source_not_available_not_waived")
    return result


def unknown_clusters(report):
    """Allocation totals and simultaneous interval peaks are different columns."""
    groups = {}
    for row in report["records"]:
        if row["role"] != "unknown":
            continue
        key = row.get("stack_id")
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in groups.items():
        delta = {}
        for row in rows:
            delta[row["start"]] = delta.get(row["start"], 0) + row["bytes"]
            if row["end"] is not None:
                delta[row["end"]] = delta.get(row["end"], 0) - row["bytes"]
        live = peak = 0
        for sequence in sorted(delta):
            live += delta[sequence]
            peak = max(peak, live)
        births = [row["time_us"] for row in rows if row.get("time_us") is not None]
        deaths = [row["end_time_us"] for row in rows if row.get("end_time_us") is not None]
        result.append(
            dict(
                stack_id=key,
                frames=report["stack_table"][key] if key is not None else [],
                allocation_generations=len(rows),
                total_allocated_bytes=sum(row["bytes"] for row in rows),
                simultaneous_peak_bytes=peak,
                maximum_generation_bytes=max(row["bytes"] for row in rows),
                first_allocator_time_us=min(births) if births else None,
                last_completed_free_time_us=max(deaths) if deaths else None,
                source_baseline_without_stack=key is None,
                classified_non_KV=False,
            )
        )
    return sorted(result, key=lambda row: (-row["simultaneous_peak_bytes"], str(row["stack_id"])))


@contextmanager
def instrument(adapter, ledger, *, deadline=None):
    """Audit wrappers only; do not construct replacement K/V or change selection."""
    import evicastkv_review_source_controls as controls
    import evicastkv_review_source_prefix as source

    from mosaickv.evaluation import request_ownership, shared_source_format, storage_runs
    from mosaickv.evaluation.graphed_storage import SharedSourceCache

    original_prefill = adapter.prefill
    original_decode = adapter.decode_one_token
    original_formats = controls._formats
    original_chosen = source._formats_with_ceiling
    original_choose = source.choose_attainable
    original_cache = storage_runs.cache_from_formats
    original_observe = SharedSourceCache.observe
    original_construct = shared_source_format.construct
    original_fit = shared_source_format.fit_transport
    counts = dict(builders=0, constructs=0, fits=0, updates=0)

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("bounded source ownership diagnostic soft deadline reached")

    def prefill(*args, **kwargs):
        check_deadline()
        output = original_prefill(*args, **kwargs)
        ledger.track("original_prefill_KV", adapter._legacy_layers(output.state.past_key_values)[0])
        ledger.observe("prefill_returned")
        return output

    def build(*args, **kwargs):
        check_deadline()
        ledger.track("normalized_source_capture", args[4].records)
        ledger.track("prompt_attention_capture", (args[3].records, args[3].windows, args[3].priors))
        ledger.track("RoPE_phase_capture", args[5].phases)
        counts["builders"] += 1
        ledger.observe(f"candidate_{counts['builders']}_before")
        formats, proof = original_formats(*args, **kwargs)
        ledger.track(
            f"candidate_{counts['builders']}_formats",
            tuple(t for packet in formats for t in packet.tensors()),
        )
        ledger.observe(f"candidate_{counts['builders']}_returned")
        return formats, proof

    def construct(*args, **kwargs):
        check_deadline()
        output = original_construct(*args, **kwargs)
        counts["constructs"] += 1
        ledger.track(f"construct_{counts['constructs']}", output.tensors())
        return output

    def fit(*args, **kwargs):
        check_deadline()
        output = original_fit(*args, **kwargs)
        counts["fits"] += 1
        ledger.track(f"fit_{counts['fits']}", output[0].tensors())
        return output

    def chosen(*args, **kwargs):
        output = original_chosen(*args, **kwargs)
        ledger.track("selected_formats", tuple(t for p in output[0] for t in p.tensors()))
        ledger.observe("finite_ladder_selected_returned")
        return output

    def choose(*args, **kwargs):
        ledger.observe("finite_ladder_all_candidates_live_before_choice")
        return original_choose(*args, **kwargs)

    def cache(*args, **kwargs):
        check_deadline()
        ledger.observe("sources_retired_before_cache_construction")
        output = original_cache(*args, **kwargs)
        ledger.track("decode_cache_history_and_workspace", request_ownership.cache_tensors(output))
        ledger.observe("decode_cache_constructed")
        return output

    def observe(cache, extras=()):
        result = original_observe(cache, extras)
        ledger.track("decode_cache_history_and_workspace", request_ownership.cache_tensors(cache))
        ledger.track("current_decode_explicit_temporary", extras)
        return result

    def decode(*args, **kwargs):
        check_deadline()
        output = original_decode(*args, **kwargs)
        counts["updates"] += 1
        if counts["updates"] in POLICY["decode_checkpoints"]:
            ledger.observe(f"decode_update_{counts['updates']}_returned")
        return output

    with ExitStack() as stack:
        for obj, name, replacement in (
            (adapter, "prefill", prefill),
            (adapter, "decode_one_token", decode),
            (controls, "_formats", build),
            (source, "_formats_with_ceiling", chosen),
            (source, "choose_attainable", choose),
            (shared_source_format, "construct", construct),
            (shared_source_format, "fit_transport", fit),
            (storage_runs, "cache_from_formats", cache),
            (SharedSourceCache, "observe", observe),
        ):
            stack.enter_context(patch.object(obj, name, replacement))
        yield counts
        if counts["updates"] != POLICY["continuation_updates"]:
            raise ValueError("ownership capture did not cover the complete registered decode")


def prepare(root, parent=PARENT):
    from evicastkv_review_source_prefix import MODEL, label_blind_payload

    root, parent = Path(root), Path(parent)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch ownership-study root required")
    plan = read(parent / "plan.json")
    old = read(parent / "workers/1/result.json")
    if old["status"] != "completed" or old["failures"]:
        raise ValueError("completed consumed parent worker required")
    job = plan["jobs"][1]
    payload = label_blind_payload(job["inputs"][0])
    if payload["task"] != "mmvet_prefixkv":
        raise ValueError("one previously consumed MM-Vet input only")
    paths = [
        parent / "plan.json",
        parent / "workers/1/result.json",
        Path(job["inputs"][0]),
        *(Path(item["path"]) for item in payload["media"]),
    ]
    bindings = {str(path): sha(path) for path in paths}
    for name, expected in old["outputs"].items():
        path = parent / "workers/1" / name
        if sha(path) != expected:
            raise ValueError("parent worker artifact changed")
        bindings[str(path)] = expected
    for path in (
        Path(__file__),
        SOURCE / "mosaickv/scripts/evicastkv_review_source_prefix.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_source_controls.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/request_ownership.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/storage_runs.py",
        SOURCE / "mosaickv/src/mosaickv/measurements/kv_timeline.py",
        SOURCE / "mosaickv/src/mosaickv/measurements/ownership_review.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/shared_source_format.py",
        SOURCE / "mosaickv/env/common/requirements.lock",
    ):
        bindings[str(path)] = sha(path)
    config = json.loads(json.dumps(job.get("config", plan["config"])))
    config["execution"]["seed"] = 13
    write_once(root / "resolved_config.json", config)
    bindings[str(root / "resolved_config.json")] = sha(root / "resolved_config.json")
    protocol = dict(
        source_sha=clean_source(SOURCE),
        source_root=str(SOURCE),
        parent=str(parent),
        config=config,
        job=job,
        model=MODEL,
        policy=POLICY,
        bindings=bindings,
        expected_workers=1,
        worker_GPUs=1,
        worker_CPUs=12,
        wall_minutes=20,
        new_cohort=False,
        paper_claim=False,
    )
    write_once(root / "protocol.json", protocol)
    return dict(workers=1, protocol_sha256=sha(root / "protocol.json"))


def validate(root):
    plan = read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy"] != json.loads(json.dumps(POLICY)):
        raise ValueError("bounded ownership-study policy changed")
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable source ownership binding changed: {path}")
    return plan


def worker(root, index=0):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_review_source_prefix import (
        MODEL,
        comparisons,
        forced_prefix,
        label_blind_payload,
        request,
    )
    from evicastkv_source_protection_diagnostic import require_isolated_slurm_gpu

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import request_ownership
    from mosaickv.hf_cli import _model_kwargs

    root = Path(root)
    plan = validate(root)
    if index != 0:
        raise ValueError("one consumed ownership worker only")
    out = root / "worker"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"])
    ledger, capture, holders = DeclaredOwners(), {}, []
    started = time.monotonic()
    try:
        gpu = require_isolated_slurm_gpu(torch)
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        cfg = RunConfig.from_mapping(plan["config"])
        if (
            (cfg.model.id, cfg.model.revision) != MODEL
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
        ):
            raise ValueError("registered BF16 eager full-MHA LLaVA checkpoint required")
        payload = label_blind_payload(plan["job"]["inputs"][0])
        old_logits = torch.load(
            Path(plan["parent"]) / "workers/1/logits.pt", map_location="cpu", weights_only=True
        )
        reference = old_logits["fullkv_native_reference"]
        prefix = forced_prefix(reference)
        adapter = load_hf_adapter(
            MODEL[0],
            revision=MODEL[1],
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                config_sha256=config_sha256(cfg),
                policy_sha256=digest(POLICY),
                model=dataclasses.asdict(cfg.model),
                dataset=dataclasses.asdict(cfg.dataset),
                environment=environment(),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                processor_revision=MODEL[1],
                GPU=gpu,
                cuda_runtime=torch.version.cuda,
                backend="huggingface",
                attention="eager",
                seed=13,
                vllm="not_used",
                sglang="not_used",
                measurement_type="source_owner_construction_diagnostic",
                result_category="unified_comparison",
                result_origin="locally_measured",
                slurm_job_id=__import__("os").environ["SLURM_JOB_ID"],
                conditioning_prefix_sha256=digest(prefix),
                reference_prefix_used_only_for_decode=True,
                sample_id=payload["sample_id"],
                benchmark_labels_read=False,
                timings_instrumented_not_comparative=True,
            ),
        )
        original = request_ownership._RequestSession

        def session(*args, **kwargs):
            value = original(*args, **kwargs)
            ledger.session = value
            holders.append(value)
            return value

        with (
            torch.inference_mode(),
            patch.object(request_ownership, "_RequestSession", session),
            request_ownership.audit_request(adapter, max_entries=POLICY["max_allocator_entries"]),
            instrument(
                adapter, ledger, deadline=started + POLICY["soft_worker_seconds"]
            ) as counters,
        ):
            logits, record = request(
                adapter,
                payload,
                cfg,
                ARM,
                prefix=prefix,
                cap=plan["job"]["initial_ceiling_bytes"],
            )
            ledger.observe("request_returned")
        torch.save(logits, out / "logits.pt")
        numerical = comparisons(reference, logits)
        repeat = comparisons(old_logits[ARM], logits)
        write_once(
            out / "raw.json",
            dict(
                arm=ARM,
                record=record,
                numerical=numerical,
                parent_source_repeat=repeat,
                counters=counters,
            ),
        )
        result.update(
            status="completed",
            continuation_updates=counters["updates"],
            repeat_all_tokens_agree=repeat["prefix_128"]["token_agreement"] == 1,
            registered_guard_pass=numerical["registered_guard_32"]["registered_guard_pass"],
        )
    except Exception as error:
        result["failures"].append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        if holders:
            try:
                capture = holders[0].report(opaque_workspace_classified=False)
                report = conservative_report(
                    capture, ledger.observations, events=holders[0].timeline.events
                )
                with gzip.open(out / "allocator_trace.json.gz", "xt") as stream:
                    json.dump(
                        dict(
                            actual_events=holders[0].timeline.events,
                            allocator_records=capture["records"],
                            stack_table=capture["stack_table"],
                            clock_calibration=capture.get("clock_calibration"),
                            inherited_stack_labels_not_admitted=True,
                            installed_and_local_source_hashes=stack_source_hashes(
                                capture["stack_table"]
                            ),
                        ),
                        stream,
                        sort_keys=True,
                        allow_nan=False,
                    )
                with gzip.open(out / "allocator_ledger.json.gz", "xt") as stream:
                    json.dump(report, stream, sort_keys=True, allow_nan=False)
            except Exception as error:
                result["failures"].append(dict(error=f"ownership reduction: {error}"))
        write_once(
            out / "declared_owners.json",
            dict(observations=ledger.observations, marks=ledger.marks, blind_spots=BLIND_SPOTS),
        )
        result["outputs"] = {path.name: sha(path) for path in out.iterdir() if path.is_file()}
        result["instrumented_worker_seconds_before_result"] = time.monotonic() - started
        if result["failures"]:
            result["status"] = "failed"
        write_once(out / "result.json", result)
    return result


def analyze(root):
    root = Path(root)
    plan = validate(root)
    result = read(root / "worker/result.json")
    for name, expected in result["outputs"].items():
        if sha(root / "worker" / name) != expected:
            raise ValueError("source ownership raw artifact changed")
    report_path = root / "worker/allocator_ledger.json.gz"
    if not report_path.exists():
        gate = dict(
            status="blocked_incomplete_ownership_study",
            source_sha=plan["source_sha"],
            errors=[*result["failures"], dict(error="allocator capture unavailable")],
            source_whole_request_KV_peak_certified=False,
            whole_device_memory_certified=False,
            superiority_established=False,
            fresh_confirmation_authorized=False,
        )
        write_once(root / "GATE_source_ownership_closure.json", gate)
        return gate
    with gzip.open(report_path, "rt") as stream:
        report = json.load(stream)
    if report["exact_whole_request_KV_peak_certified"] or report["exact_KV_peak_bytes"] is not None:
        raise ValueError("bounded declared-owner study cannot mint an exact certificate")
    write_once(
        root / "analysis/unknown_allocation_clusters.json",
        dict(
            clusters=unknown_clusters(report),
            unrecognized_stacks_never_assumed_non_KV=True,
            raw_trace_sha256=sha(root / "worker/allocator_trace.json.gz"),
            scientific_scope="CPU source-specific review priorities, not a classification waiver",
        ),
    )
    declared = read(root / "worker/declared_owners.json")
    phases = {item["phase"] for item in declared["observations"]}
    required_phases = {
        "prefill_returned",
        "candidate_1_before",
        "finite_ladder_selected_returned",
        "finite_ladder_all_candidates_live_before_choice",
        "sources_retired_before_cache_construction",
        "decode_cache_constructed",
        "decode_update_127_returned",
        "request_returned",
    }
    errors = list(result["failures"])
    if not required_phases <= phases:
        errors.append(
            dict(
                error="required source-owner boundaries not all observed",
                missing_phases=sorted(required_phases - phases),
            )
        )
    retirement = next(
        (
            item
            for item in declared["observations"]
            if item["phase"] == "sources_retired_before_cache_construction"
        ),
        None,
    )
    original_remaining = (
        None
        if retirement is None
        else sum(
            item["bytes"] for item in retirement["owners"] if "original_prefill_KV" in item["roles"]
        )
    )
    if original_remaining:
        errors.append(
            dict(
                error="original prefill K/V owner remains alive before decode",
                bytes=original_remaining,
            )
        )
    gate = dict(
        status="completed_bounded_ownership_study"
        if result["status"] == "completed" and not errors
        else "blocked_incomplete_ownership_study",
        source_sha=plan["source_sha"],
        errors=errors,
        known_KV_peak_lower_bound_bytes=report["known_KV_peak_lower_bound_bytes"],
        observed_declared_cuda_peak_lower_bound_bytes=report[
            "observed_declared_cuda_peak_lower_bound_bytes"
        ],
        PyTorch_KV_upper_envelope_including_unknown_bytes=report[
            "KV_peak_upper_bound_including_unknown_bytes"
        ],
        unknown_allocation_generations=report["unknown_allocation_generations"],
        allocator_history_complete=report["allocator_history_complete"],
        source_whole_request_KV_peak_certified=False,
        whole_device_memory_certified=False,
        report_sha256=sha(root / "worker/allocator_ledger.json.gz"),
        original_prefill_and_captures_included=True,
        required_phase_boundaries_observed=required_phases <= phases,
        observed_original_prefill_bytes_before_decode=original_remaining,
        generation_reuse_not_collapsed=True,
        observed_checkpoints_not_exact_transients=True,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        blind_spots=BLIND_SPOTS,
    )
    write_once(root / "GATE_source_ownership_closure.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=PARENT)
    args = parser.parse_args()
    result = (
        prepare(args.root, args.parent)
        if args.mode == "prepare"
        else globals()[args.mode](args.root)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
