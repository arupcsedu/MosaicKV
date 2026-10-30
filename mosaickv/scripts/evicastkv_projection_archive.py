#!/usr/bin/env python3
"""Bounded shared-input archive feasibility; same consumed inputs, no labels."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import evicastkv_early_prefix as messages_api
import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
from evicastkv_campaign import environment

from mosaickv.config import RunConfig
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/lossless-fused-consumed-20261029-v3")
PARENT_SHA = "e538caf12011239a3d218a2620106ec20e9c11b36dedd13341b85c00edabd42d"


def prepare(root, *, runtime=False, union_restore=False, source_timing=False):
    if union_restore and not runtime:
        raise ValueError("union restoration requires the isolated runtime controls")
    if source_timing and (not runtime or union_restore):
        raise ValueError("timing requires original-selection runtime controls, not rejected union")
    source = clean_source(SOURCE)
    if root.exists() or sha(PARENT / "plan.json") != PARENT_SHA:
        raise ValueError("new immutable root and pinned consumed inputs required")
    plan = common.read(PARENT / "plan.json")
    for field in ("paired_transport_benchmark", "fused_unpack", "gpu_constructor", "trajectory"):
        plan.pop(field, None)
    bound = dict(plan["bound_files"])
    for name in (
        "plan.json",
        "analysis/GATE.json",
        "transport_timing_analysis/GATE.json",
        "transport_timing_analysis/manifest.json",
    ):
        bound[str(PARENT / name)] = sha(PARENT / name)
    plan.update(
        source_sha=source,
        bound_files=bound,
        protocol="shared_normalized_projection_input_archive_feasibility_v1",
        intervention="store_union_normalized_inputs_recompute_original_shape_KV_then_original_RoPE",
        same_consumed_inputs_only=True,
        selector_changes_authorized=False,
        numerical_control="bitwise_retained_KV_comparison_to_original_source_at_every_layer",
        not_installed_cache_backend=True,
        benchmark_quality_measured=False,
        whole_request_peak_certified=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        shared_format_applied_to_both_selectors=True,
    )
    if runtime:
        previous = Path("/scratch/djy8hg/runs/evicastkv/projection-source-feasibility-20261030-v1")
        gate = common.read(previous / "analysis/GATE.json")
        if gate["status"] != "complete_consumed_component_diagnostic" or gate["failures"]:
            raise ValueError("complete retained-bit source reconstruction required")
        for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
            bound[str(previous / name)] = sha(previous / name)
        plan.update(
            protocol="isolated_projection_archive_greedy_and_workspace_control_v2",
            runtime_controls=True,
            tokens=32,
            source_layer_bound=8,
            not_installed_cache_backend=False,
            experimental_evaluation_transport_only=True,
            policies=["kv_only", "hybrid", "bounded_sources"],
            whole_request_peak_certified=False,
            latency_comparison_authorized=False,
        )
    if union_restore:
        previous = Path("/scratch/djy8hg/runs/evicastkv/projection-source-runtime-20261030-v2")
        gate = common.read(previous / "analysis/GATE.json")
        if gate["runtime_controls_passed"] is not True or gate["failures"]:
            raise ValueError("complete source-cache runtime identity required")
        for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
            bound[str(previous / name)] = sha(previous / name)
        plan.update(
            protocol="source_union_restoration_consumed_causal_control_v3",
            union_restore_controls=True,
            selector_changes_authorized=True,
            bounded_selector_change="all_heads_receive_union_of_already_archived_rows",
            no_donor_eviction=True,
            no_benchmark_label_access=True,
            same_realized_byte_budget_claim=False,
            representation_correctness_reference="independent_original_KV_gather_at_expanded_positions",
            original_guard_unchanged=True,
        )
    if source_timing:
        previous = Path("/scratch/djy8hg/runs/evicastkv/source-union-restoration-20261030-v3")
        gate = common.read(previous / "analysis/GATE.json")
        if gate["status"] != "complete_consumed_component_diagnostic" or gate["failures"]:
            raise ValueError("completed consumed causal diagnostic required")
        for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
            bound[str(previous / name)] = sha(previous / name)
        plan.update(
            protocol="original_selection_projection_source_paired_timing_v4",
            source_timing_controls=True,
            selector_changes_authorized=False,
            source_timing=dict(
                warmups=5,
                timed_repetitions=20,
                tokens=32,
                variants=["raw_separate", "encoded_KV", "hybrid_source"],
                seed=20261030,
                randomized_within_paired_block=True,
                workspace_instrumentation_disabled_inside_timing=True,
            ),
            timing_boundary="same_selection_shared_prefill_constructor_and_decode_not_request",
            transport_timing_comparison_authorized=True,
            memory_evidence="separate_instrumented_runtime_controls_not_timed_peak",
            causal_union_remains_negative=True,
            no_fresh_or_quality_campaign_promotion=True,
        )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def worker(root, expected, index):
    import inspect

    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import compare_with_generate
    from mosaickv.evaluation.delimiter_control import source_sha
    from mosaickv.evaluation.lossless_bf16_kernel import encode
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.projection_archive import (
        ProjectionCapture,
        build_archive,
        reconstruct_archive,
    )
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        task=job["task"],
        sample_id=job["sample_id"],
    )
    started = time.monotonic()
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one allocated A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic workspace missing")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(cfg.execution.seed)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        messages = messages_api.redacted_messages(common.read(job["input"]))
        parity = compare_with_generate(adapter, adapter.prepare_inputs(messages), max_new_tokens=16)
        if parity.token_agreement != 1 or parity.maximum_logit_difference > 0.001:
            raise ValueError("ordinary model identity failed")
        write_once(out / "parity.json", asdict(parity))
        with ProjectionCapture(adapter) as capture:
            prefill = adapter.prefill(adapter.prepare_inputs(messages), capture_queries=False)
        capture.validate()
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        checksum = source_sha(pairs)
        if plan.get("runtime_controls"):
            from mosaickv.evaluation.projection_cache import runtime_controls

            all_positions = [
                [[p for p in range(k.shape[-2])] for _ in range(k.shape[1])] for k, _ in pairs
            ]
            identity = runtime_controls(
                adapter, prefill, capture, all_positions, policies=("bounded_sources",)
            )
            write_once(out / "runtime_retention_one.json", identity)
        folder = Path(job["delimiter_worker"])
        frozen = torch.load(folder / "captures.pt", weights_only=True, map_location="cpu")
        transactions = common.read(folder / "transactions.json")
        source_file = Path(inspect.getsourcefile(type(adapter._language_layers()[0].self_attn)))
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                source_dirty=False,
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                environment=environment(),
                gpu=asdict(capture_gpu_environment(torch)),
                cuda_runtime=torch.version.cuda,
                installed_transformers_source=str(source_file),
                installed_transformers_source_sha256=sha(source_file),
                measurement_type="evaluation_only_projection_archive_runtime_correctness"
                if plan.get("runtime_controls")
                else "consumed_projection_archive_component_feasibility",
                result_category="unified_comparison",
                result_origin="locally_measured",
                evaluator_commit=plan["source_sha"],
                baseline_commit=plan["source_sha"],
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                reference_output_sha256=sha(folder / "captures.pt"),
                annotation_targets_used=False,
                experimental_live_decode_storage=bool(plan.get("runtime_controls")),
                not_production_cache_backend=True,
                not_request_latency_measurement=True,
                paired_source_transport_timing=bool(plan.get("source_timing_controls")),
            ),
        )
        rows = []
        for arm in job["arms"]:
            role = arm["role"]
            if checksum != transactions[role]["null_repack"]["transaction"]["source_KV_sha256"]:
                raise ValueError("original source KV changed")
            selections = frozen[role]["anchor"]["positions"]
            for encoded in (False, True):
                owners, hybrid_owners, records, expected_payload = [], [], [], 0
                for layer_index, ((key, value), positions) in enumerate(
                    zip(pairs, selections, strict=True)
                ):
                    archive = build_archive(
                        capture.records[layer_index], positions, encoded=encoded
                    )
                    owners.extend(archive.storage_tensors())
                    indices = archive.selected[None, :, :, None].expand(
                        1, key.shape[1], len(positions[0]), key.shape[-1]
                    )
                    reference = (key.gather(2, indices), value.gather(2, indices))
                    expected_payload += backing_bytes(reference)
                    baseline_bodies = tuple(encode(t) for t in reference)
                    baseline_owners = tuple(t for b in baseline_bodies for t in b.buffers)
                    source_cost, baseline_cost = (
                        backing_bytes(archive.storage_tensors()),
                        backing_bytes(baseline_owners),
                    )
                    choose_source = source_cost < baseline_cost
                    hybrid_owners.extend(
                        archive.storage_tensors() if choose_source else baseline_owners
                    )
                    torch.cuda.synchronize()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    reconstructed = reconstruct_archive(
                        archive, adapter._language_layers()[layer_index].self_attn
                    )
                    end.record()
                    torch.cuda.synchronize()
                    equality = [
                        torch.equal(a, b) for a, b in zip(reference, reconstructed, strict=True)
                    ]
                    record = dict(
                        layer=layer_index,
                        role=role,
                        encoded_source=encoded,
                        union_tokens=archive.union.numel(),
                        per_head_tokens=len(positions[0]),
                        source_component_bytes_with_rotary=source_cost,
                        encoded_KV_component_bytes=baseline_cost,
                        conservative_hybrid_chooses_source=choose_source,
                        bitwise_K=equality[0],
                        bitwise_V=equality[1],
                        one_reconstruction_cuda_ms=start.elapsed_time(end),
                    )
                    records.append(record)
                    if not all(equality):
                        write_once(out / f"{role}-{encoded}-partial.json", records)
                        raise RuntimeError(
                            f"retained BF16 reconstruction differs at layer {layer_index}"
                        )
                rows.append(
                    dict(
                        role=role,
                        encoded_source=encoded,
                        layers=records,
                        selected_KV_payload_bytes=expected_payload,
                        archive_component_unique_backing_bytes=backing_bytes(tuple(owners)),
                        conservative_hybrid_component_unique_backing_bytes=backing_bytes(
                            tuple(hybrid_owners)
                        ),
                        not_installed_in_decode=True,
                        whole_request_peak_certified=False,
                    )
                )
                del (
                    owners,
                    hybrid_owners,
                    baseline_bodies,
                    baseline_owners,
                    reconstructed,
                    reference,
                    archive,
                )
                settle(adapter)
            if plan.get("runtime_controls"):
                from mosaickv.evaluation.projection_cache import runtime_controls

                controls = runtime_controls(adapter, prefill, capture, selections)
                write_once(out / f"{role}-runtime_controls.json", controls)
            if plan.get("source_timing_controls"):
                from mosaickv.evaluation.projection_timing import benchmark

                trials = benchmark(
                    adapter, prefill, capture, selections, seed=plan["source_timing"]["seed"]
                )
                write_once(out / f"{role}-source_timing.json", trials)
            if plan.get("union_restore_controls"):
                from mosaickv.config import DraftAllocationGuardConfig
                from mosaickv.evaluation.union_restoration import evaluate_union

                full = torch.load(
                    Path(job["parent_worker"]) / "captures.pt",
                    map_location="cpu",
                    weights_only=True,
                )["logits"]["full_kv_reference"]
                control = evaluate_union(
                    adapter,
                    prefill,
                    capture,
                    selections,
                    common.read(job["input"])["fullkv_prefix_tokens"],
                    full_logits=full,
                    anchor_logits=frozen[role]["anchor"]["logits"],
                    guard_config=DraftAllocationGuardConfig.from_mapping(plan["guard"]),
                    maximum_reference_kl=plan["maximum_reference_kl"],
                    maximum_logit_difference=plan["maximum_logit_difference"],
                )
                write_once(out / f"{role}-union_controls.json", control)
        if source_sha(pairs) != checksum:
            raise RuntimeError("archive feasibility mutated original source")
        write_once(out / "measurements.json", rows)
        result.update(
            status="completed", outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}
        )
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = common.inputs(root, expected)
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    rows, failures, inputs, controls, unions = [], [], {}, [], []
    timing_trials, timing_summary = [], []
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        path = folder / "result.json"
        if not path.exists():
            failures.append(dict(worker=job["index"], error="missing result"))
            continue
        result = common.read(path)
        inputs[str(path)] = sha(path)
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("producer source/plan changed")
        if result["status"] != "completed" or result["failures"]:
            failures.append(dict(worker=job["index"], errors=result["failures"]))
            continue
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("worker output changed")
        for record in common.read(folder / "measurements.json"):
            if len(record["layers"]) != 32 or not all(
                r["bitwise_K"] and r["bitwise_V"] for r in record["layers"]
            ):
                raise ValueError("incomplete bitwise source reconstruction")
            rows.append(dict(task=job["task"], sample_id=job["sample_id"], **record))
        if plan.get("runtime_controls"):
            identity = common.read(folder / "runtime_retention_one.json")["bounded_sources"]
            if not identity["bitwise_logits_equal_reference"] or identity["tokens"] != 32:
                raise ValueError("source-archive retention-one identity failed")
            for arm in job["arms"]:
                for policy, control in common.read(
                    folder / f"{arm['role']}-runtime_controls.json"
                ).items():
                    if (
                        not control["bitwise_logits_equal_reference"]
                        or control["tokens"] != 32
                        or control["maximum_logit_difference"] != 0
                    ):
                        raise ValueError("source-archive greedy identity failed")
                    controls.append(
                        dict(task=job["task"], role=arm["role"], policy=policy, **control)
                    )
        if plan.get("union_restore_controls"):
            for arm in job["arms"]:
                control = common.read(folder / f"{arm['role']}-union_controls.json")
                if not control["representation_bitwise_equal_original_expanded_KV"]:
                    raise ValueError("union representation reference failed")
                unions.append(dict(task=job["task"], role=arm["role"], **control))
        if plan.get("source_timing_controls"):
            from mosaickv.evaluation.projection_timing import summarize

            for arm in job["arms"]:
                data = common.read(folder / f"{arm['role']}-source_timing.json")
                timing_trials.extend(dict(t, task=job["task"], role=arm["role"]) for t in data)
                timing_summary.extend(
                    dict(t, task=job["task"], role=arm["role"]) for t in summarize(data)
                )
    gate = dict(
        status="complete_consumed_component_diagnostic"
        if len(rows) == 8
        and not failures
        and (not plan.get("runtime_controls") or len(controls) == 12)
        and (not plan.get("source_timing_controls") or len(timing_summary) == 12)
        else "failed",
        failures=failures,
        comparisons=rows,
        not_installed_in_decode=not plan.get("runtime_controls", False),
        production_backend=False,
        union_controls=unions,
        union_relative_guards_passed=(
            len(unions) == 4
            and not failures
            and all(r["unchanged_relative_guard"]["accepted"] for r in unions)
        )
        if plan.get("union_restore_controls")
        else None,
        union_absolute_limits_passed=(
            len(unions) == 4
            and not failures
            and all(r["absolute_fullkv_limits_passed"] for r in unions)
        )
        if plan.get("union_restore_controls")
        else None,
        runtime_controls=controls,
        source_timing_summary=timing_summary,
        source_timing_complete=(len(timing_summary) == 12 and not failures)
        if plan.get("source_timing_controls")
        else None,
        runtime_controls_passed=(len(controls) == 12 and not failures)
        if plan.get("runtime_controls")
        else None,
        benchmark_quality_measured=False,
        whole_request_peak_certified=False,
        selector_superiority_established=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(out / "GATE.json", gate)
    if plan.get("source_timing_controls"):
        import pandas as pd

        pd.DataFrame(timing_trials).to_parquet(out / "timing_trials.parquet", index=False)
        pd.DataFrame(timing_summary).to_parquet(out / "timing_summary.parquet", index=False)
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs=inputs,
            outputs={p.name: sha(p) for p in out.iterdir()},
        ),
    )
    print(
        json.dumps(dict(status=gate["status"], failures=failures, completed_components=len(rows)))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    parser.add_argument("--runtime-controls", action="store_true")
    parser.add_argument("--union-restore-controls", action="store_true")
    parser.add_argument("--source-timing-controls", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(
            prepare(
                args.root,
                runtime=args.runtime_controls,
                union_restore=args.union_restore_controls,
                source_timing=args.source_timing_controls,
            )
        )
    elif args.mode == "submit":
        launch.submit(
            args.root,
            args.plan_sha,
            script_name="evicastkv_projection_archive.sbatch",
            worker_minutes=20
            if common.read(args.root / "plan.json").get("source_timing_controls")
            else 10,
            memory_gb=16,
        )
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, args.index)
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
