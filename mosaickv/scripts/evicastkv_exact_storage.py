#!/usr/bin/env python3
"""Finite four-stage consumed-input storage study; never paper promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
BASE = Path("/scratch/djy8hg/runs/evicastkv/exact-storage-stages-20261030-v1")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/streamed-source-paired-timing-20261030-v9")
PARENT_SHA = "5b0da3d148de4e0ca1793440591ff47b1a1028b982e89400bc2ca3479f87c008"
STAGES = {
    1: "per_token_storage",
    2: "numerical_patch_density",
    3: "direct_buffer_construction",
    4: "sealed_history",
}


def evidence_path(stem, suffix):
    # Selector IDs contain decimal points; with_suffix would truncate the ID.
    return stem.parent / (stem.name + suffix)


def variants_for(stage):
    if stage == 1:
        return [("kv", False, False), ("token", False, False)]
    if stage == 2:
        return [("kv", False, False), ("patched", False, False)]
    if stage == 3:
        return [("kv", True, False), ("token", False, False), ("token", True, False)]
    if stage == 4:
        return [(mode, True, sealed) for mode in ("kv", "token") for sealed in (False, True)]
    raise ValueError("only the four registered stages are permitted")


def prepare(root, stage):
    if type(stage) is not int or stage not in STAGES or root.exists():
        raise ValueError("new immutable root and stage in [1, 4] required")
    source = clean_source(SOURCE)
    if sha(PARENT / "plan.json") != PARENT_SHA:
        raise ValueError("consumed parent changed")
    parent = common.read(PARENT / "plan.json")
    parent_gate = common.read(PARENT / "analysis/GATE.json")
    if parent_gate["status"] != "complete_consumed_component_diagnostic" or parent_gate["failures"]:
        raise ValueError("parent numerical/completeness diagnostic did not pass")
    # Copy immutable inputs, not obsolete experimental switches from the parent.
    plan = dict(
        jobs=parent["jobs"],
        maximum_a100s=6,
        requested_workers=2,
        cpus_per_worker=12,
        wall_minutes=20,
        parent_plan_sha256=PARENT_SHA,
        selector_changes_authorized=False,
        original_positions_and_RoPE_preserved=True,
        no_tolerance_relaxation=True,
        automatic_promotion=False,
        construction_peak_certified=False,
    )
    bound = dict(parent["bound_files"])
    for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
        bound[str(PARENT / name)] = sha(PARENT / name)
    if stage > 1:
        previous = root.parent / f"stage-{stage - 1}"
        gate = common.read(previous / "analysis/GATE.json")
        if gate["status"] != "complete_exact_storage_diagnostic" or gate["failures"]:
            raise ValueError("previous stage completeness/numerical gate did not pass")
        for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
            bound[str(previous / name)] = sha(previous / name)
    plan.update(
        source_sha=source,
        bound_files=bound,
        stage=stage,
        stage_name=STAGES[stage],
        protocol="consumed_exact_storage_four_stage_v1",
        maximum_stages=4,
        same_consumed_inputs_only=True,
        same_memberships=True,
        common_formats_for_both_selectors=True,
        numerical_control="original_KV_bits_and_ordinary_cache_32_token_logits_bitwise",
        patch_repeats=3,
        source_degree_rule="2*retaining_KV_heads>total_KV_heads",
        choice_rule="minimum_actual_serialized_component_bytes_including_indices_patches_and_RoPE",
        history_chunk_tokens=16,
        output_lengths=[32, 128] if stage == 4 else [32],
        finite_next_stage_authorized=stage < 4,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        benchmark_quality_measured=False,
        whole_request_peak_certified=False,
        timing_kind="instrumented_single_control_not_comparable_latency",
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def control(
    adapter, prefill, capture, positions, reference, *, mode, direct, sealed, tokens, evidence
):
    import gc

    import torch

    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.exact_storage import construct
    from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport

    torch.cuda.synchronize()
    started = time.monotonic()
    cache = construct(
        adapter,
        prefill,
        capture,
        positions,
        mode=mode,
        direct=direct,
        sealed_history=sealed,
        updates=tokens,
        audit=True,
    )
    torch.cuda.synchronize()
    construction = time.monotonic() - started
    write_once(
        evidence_path(evidence, ".construction.json"),
        dict(
            storage=cache.storage_record(),
            construction_seconds=construction,
            mode=mode,
            direct_buffer=direct,
            sealed_history=sealed,
            tokens=tokens,
        ),
    )
    length = max(len(h[0]) for h in positions)
    state = DecodeState(
        cache,
        torch.ones((1, length), device=adapter.device, dtype=torch.long),
        length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
        prefill.state.modality_map,
        dict(
            prefill.state.model_state, mosaickv_packed_all_valid=True, mosaickv_cache_all_valid=True
        ),
    )
    values, token = [prefill.logits.detach().reshape(-1).float()], prefill.next_token_id
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    with IsolatedStorageTransport(adapter, cache):
        start.record()
        for _ in range(tokens - 1):
            output = adapter.decode_one_token(token, state, capture_queries=False)
            values.append(output.logits.detach().reshape(-1).float())
            token = output.next_token_id
        end.record()
    torch.cuda.synchronize()
    wall, cuda_ms = time.monotonic() - started, start.elapsed_time(end)
    logits = torch.stack(values).cpu()
    if not torch.isfinite(logits).all() or not torch.equal(logits, reference):
        torch.save(
            dict(actual_logits=logits, ordinary_reference_logits=reference),
            evidence_path(evidence, ".failure.pt"),
        )
        write_once(
            evidence_path(evidence, ".failure.json"),
            dict(
                storage=cache.storage_record(),
                maximum_logit_difference=float((logits - reference).abs().max()),
                finite=bool(torch.isfinite(logits).all()),
                construction_seconds=construction,
                instrumented_decode_wall_seconds=wall,
                instrumented_decode_cuda_ms=cuda_ms,
            ),
        )
        raise RuntimeError(f"{mode}/direct={direct}/sealed={sealed} changed greedy logits")
    if state.next_decode_position != prefill.state.next_decode_position + tokens - 1:
        raise RuntimeError("logical/RoPE positions changed")
    record = cache.storage_record()
    if record["outstanding_layer_leases"] or record["history_updates_per_layer"] != [
        tokens - 1
    ] * len(positions):
        raise RuntimeError("incomplete history or leaked state")
    row = dict(
        mode=mode,
        direct_buffer=direct,
        sealed_history=sealed,
        tokens=tokens,
        token_ids=logits.argmax(-1).tolist(),
        logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        reference_logits_sha256=hashlib.sha256(reference.numpy().tobytes()).hexdigest(),
        bitwise_logits=True,
        maximum_logit_difference=0.0,
        storage=record,
        construction_seconds=construction,
        instrumented_decode_wall_seconds=wall,
        instrumented_decode_cuda_ms=cuda_ms,
        not_comparable_latency=True,
        selector_changed=False,
        used_fullkv_fallback=False,
    )
    del cache, state, values, logits, output
    gc.collect()
    return row


def worker(root, expected, index):
    import inspect

    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import compare_with_generate
    from mosaickv.evaluation.delimiter_control import source_sha
    from mosaickv.evaluation.exact_storage import gpu_controls
    from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference
    from mosaickv.evaluation.projection_archive import ProjectionCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.inputs(root, expected)
    job, stage = plan["jobs"][index], plan["stage"]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        stage=stage,
        task=job["task"],
        sample_id=job["sample_id"],
    )
    started = time.monotonic()
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one allocated A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUDA workspace missing")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        write_once(out / "synthetic_cuda_controls.json", gpu_controls())
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
            raise RuntimeError("ordinary model parity failed")
        write_once(out / "ordinary_parity.json", asdict(parity))
        with ProjectionCapture(adapter) as capture:
            prefill = adapter.prefill(adapter.prepare_inputs(messages), capture_queries=False)
        capture.validate()
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        before = source_sha(pairs)
        module = Path(inspect.getsourcefile(type(adapter._language_layers()[0].self_attn)))
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                config_sha256=expected,
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                environment=environment(),
                gpu=asdict(capture_gpu_environment(torch)),
                cuda_runtime=torch.version.cuda,
                installed_attention_source_sha256=sha(module),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                result_category="unified_comparison",
                result_origin="locally_measured",
                measurement_type="consumed_exact_storage_component_and_runtime_diagnostic",
                evaluator_commit=plan["source_sha"],
                baseline_commit=plan["source_sha"],
                reference_artifact_sha256=sha(Path(job["delimiter_worker"]) / "captures.pt"),
                annotation_targets_used=False,
                no_task_quality_or_end_to_end_latency_claim=True,
            ),
        )
        full = tuple(tuple(tuple(range(k.shape[-2])) for _ in range(k.shape[1])) for k, _ in pairs)
        n = 128 if stage == 4 else 32
        reference = standard_greedy_reference(adapter, prefill, full, tokens=n)
        identity = control(
            adapter,
            prefill,
            capture,
            full,
            reference,
            mode="patched" if stage == 2 else "token",
            direct=stage >= 3,
            sealed=stage == 4,
            tokens=n,
            evidence=out / "retention-one",
        )
        write_once(out / "retention_one.json", identity)
        frozen = torch.load(
            Path(job["delimiter_worker"]) / "captures.pt", map_location="cpu", weights_only=True
        )
        rows = []
        for arm in job["arms"]:
            role = arm["role"]
            positions = frozen[role]["anchor"]["positions"]
            for tokens in plan["output_lengths"]:
                reference = standard_greedy_reference(adapter, prefill, positions, tokens=tokens)
                torch.save(reference, out / f"{role}-reference-{tokens}.pt")
                for mode, direct, sealed in variants_for(stage):
                    evidence = out / f"{role}-{tokens}-{mode}-{int(direct)}-{int(sealed)}"
                    record = control(
                        adapter,
                        prefill,
                        capture,
                        positions,
                        reference,
                        mode=mode,
                        direct=direct,
                        sealed=sealed,
                        tokens=tokens,
                        evidence=evidence,
                    )
                    row = dict(role=role, task=job["task"], sample_id=job["sample_id"], **record)
                    rows.append(row)
                    write_once(
                        out / f"{role}-{tokens}-{mode}-{int(direct)}-{int(sealed)}.json", row
                    )
        if source_sha(pairs) != before:
            raise RuntimeError("original prefill source mutated")
        write_once(out / "controls.json", rows)
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = common.inputs(root, expected)
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    rows, failures, inputs = [], [], {}
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        path = folder / "result.json"
        if not path.exists():
            failures.append(dict(worker=job["index"], error="missing result"))
            continue
        result = common.read(path)
        inputs[str(path)] = sha(path)
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("producer identity changed")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("raw output changed")
        if result["status"] != "completed":
            failures.append(
                dict(
                    worker=job["index"],
                    errors=result["failures"],
                    partial_outputs=result["outputs"],
                )
            )
            continue
        if not common.read(folder / "retention_one.json")["bitwise_logits"]:
            raise ValueError("retention-one parity failed")
        data = common.read(folder / "controls.json")
        keys = [
            (r["role"], r["tokens"], r["mode"], r["direct_buffer"], r["sealed_history"])
            for r in data
        ]
        expected_keys = {
            (arm["role"], tokens, mode, direct, sealed)
            for arm in job["arms"]
            for tokens in plan["output_lengths"]
            for mode, direct, sealed in variants_for(plan["stage"])
        }
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError("missing, duplicate or unexpected control")
        if any(not r["bitwise_logits"] or r["used_fullkv_fallback"] for r in data):
            raise ValueError("inexact storage or fallback")
        for row in data:
            if (row["task"], row["sample_id"]) != (job["task"], job["sample_id"]):
                raise ValueError("consumed input identity changed")
            if row["logits_sha256"] != row["reference_logits_sha256"]:
                raise ValueError("reference logits changed")
            if any(
                not math.isfinite(row[k]) or row[k] < 0
                for k in (
                    "construction_seconds",
                    "instrumented_decode_wall_seconds",
                    "instrumented_decode_cuda_ms",
                )
            ):
                raise ValueError("missing or invalid synchronized timing")
            storage = row["storage"]
            if storage["peak_active_kv_storage_bytes"] < max(
                storage["initial_active_kv_storage_bytes"], storage["final_active_kv_storage_bytes"]
            ):
                raise ValueError("invalid peak accounting")
        rows.extend(data)
    count = {1: 8, 2: 8, 3: 12, 4: 32}[plan["stage"]]
    complete = len(rows) == count and not failures
    gate = dict(
        status="complete_exact_storage_diagnostic" if complete else "failed",
        stage=plan["stage"],
        failures=failures,
        completed_controls=len(rows),
        expected_controls=count,
        controls=rows,
        independent_quality_or_superiority_established=False,
        fresh_or_paper_campaign_authorized=False,
        whole_request_peak_certified=False,
    )
    write_once(out / "GATE.json", gate)
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs=inputs,
            outputs={p.name: sha(p) for p in out.iterdir()},
        ),
    )
    print(json.dumps(dict(status=gate["status"], failures=failures, controls=len(rows))))
    if complete and plan["finite_next_stage_authorized"]:
        next_root = root.parent / f"stage-{plan['stage'] + 1}"
        digest = prepare(next_root, plan["stage"] + 1)
        launch.submit(
            next_root,
            digest,
            script_name="evicastkv_exact_storage.sbatch",
            worker_minutes=20,
            memory_gb=16,
        )
        write_once(
            out / "advancement.json",
            dict(
                next_root=str(next_root),
                next_plan_sha256=digest,
                source_sha=plan["source_sha"],
                scope="next_fixed_consumed_storage_stage_only",
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 5))
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.root, args.stage))
    elif args.mode == "submit":
        launch.submit(
            args.root,
            args.plan_sha,
            script_name="evicastkv_exact_storage.sbatch",
            worker_minutes=20,
            memory_gb=16,
        )
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, args.index)
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
