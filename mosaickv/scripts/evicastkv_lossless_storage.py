#!/usr/bin/env python3
"""Same consumed exact selections, shared lossless-storage feasibility diagnostic."""

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

from mosaickv.config import DraftAllocationGuardConfig, RunConfig, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/donor-isolation-consumed-20261029-v1")
PARENT_SHA = "da8cd788ff54844cdcf9c26b41f05c1eabd504ce79e6721a28dcccc6541d4b42"
MODES = ("raw_control", "lossless_storage")


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists() or sha(PARENT / "plan.json") != PARENT_SHA:
        raise ValueError("new immutable root and pinned consumed parent required")
    plan = common.read(PARENT / "plan.json")
    gate = common.read(PARENT / "analysis/GATE.json")
    if gate["status"] != "complete_consumed_diagnostic" or gate["failures"]:
        raise ValueError("prior diagnostic incomplete")
    bound = dict(plan["bound_files"])
    for name in (
        "plan.json",
        "analysis/GATE.json",
        "development_decision.json",
        "derived_interpretation_v1/measurements.json",
    ):
        bound[str(PARENT / name)] = sha(PARENT / name)
    plan.update(
        source_sha=source,
        bound_files=bound,
        modes=MODES,
        protocol="shared_lossless_BF16_exact_prompt_storage_feasibility_v1",
        parent_plan_sha256=PARENT_SHA,
        intervention="lossless_integer_exponent_encoding_of_unchanged_saved_exact_prompt",
        codec=dict(
            block_values=1024,
            exponent_code_bits=3,
            ordinary_exponents=7,
            literal_escape=True,
            raw_fallback_on_expansion=True,
        ),
        interpretation="common_storage_optimization_for_BOTH_methods_not_selector_superiority",
        donor_selection="none",
        evaluation_only_frozen_donor_mask_authorized=False,
        evaluation_only_lossless_storage_authorized=True,
        guard_unchanged=True,
        selector_changes_authorized=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        shared_prefill_diagnostic_not_end_to_end_timing=True,
        construction_peak_certified=False,
        whole_request_peak_certified=False,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def kernel_control():
    """GPU bit-pattern and nonaligned-shape checks before any model experiment."""
    import torch

    from mosaickv.evaluation.lossless_bf16 import pack_cpu, unpack_into

    torch.manual_seed(0)
    fixtures = (
        torch.arange(65536).to(torch.int16).view(torch.bfloat16).reshape(1, 1, 512, 128),
        torch.randn(1, 3, 257, 17).bfloat16(),
    )
    rows = []
    for tensor in fixtures:
        encoded = pack_cpu(tensor, raw_fallback=False).to("cuda")
        target = torch.full(
            (*tensor.shape[:2], tensor.shape[-2] + 3, tensor.shape[-1]),
            -99,
            device="cuda",
            dtype=torch.bfloat16,
        )
        unpack_into(encoded, target)
        torch.cuda.synchronize()
        assert torch.equal(
            target[..., : tensor.shape[-2], :].cpu().view(torch.int16), tensor.view(torch.int16)
        ), "GPU codec changed source bits"
        assert bool((target[..., tensor.shape[-2] :, :] == -99).all()), "history overwritten"
        rows.append(dict(shape=list(tensor.shape), bitwise=True, history_unchanged=True))
    return rows


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.delimiter_control import source_sha
    from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport, construct_cache
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
    started, captured, rows = time.monotonic(), {}, {}
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("unchanged deterministic workspace required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        write_once(out / "kernel_controls.json", dict(checks=kernel_control()))
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(cfg.execution.seed)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = common.read(job["input"])
        messages = messages_api.redacted_messages(payload)
        targets = payload["fullkv_prefix_tokens"]
        assert len(targets) == 32
        parent_folder = Path(job["delimiter_worker"])
        original = torch.load(parent_folder / "captures.pt", weights_only=True, map_location="cpu")
        transactions = common.read(parent_folder / "transactions.json")
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=expected,
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                arms=job["arms"],
                codec=plan["codec"],
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                annotation_targets_used=False,
                measurement_type="evaluation_only_shared_lossless_storage_feasibility",
                result_category="unified_comparison",
                result_origin="locally_measured",
                study_track="shared_storage_diagnostic_NOT_Fox_confirmation_or_native_PrefixKV",
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                evaluator_commit=plan["source_sha"],
                baseline_commit=plan["source_sha"],
                interpretation=plan["interpretation"],
                reference_output_sha256=sha(parent_folder / "captures.pt"),
                timing_is_shared_prefill_diagnostic_not_request_latency=True,
            ),
        )
        parity = {}
        for name, fn in (
            ("generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ):
            report = fn(adapter, adapter.prepare_inputs(messages), max_new_tokens=16)
            parity[name] = asdict(report)
            assert report.token_agreement == 1 and report.maximum_logit_difference <= 0.001
        write_once(out / "parity.json", parity)
        prefill = adapter.prefill(adapter.prepare_inputs(messages))
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        source_checksum = source_sha(pairs)
        for arm in job["arms"]:
            role = arm["role"]
            positions = original[role]["anchor"]["positions"]
            assert (
                source_checksum
                == transactions[role]["null_repack"]["transaction"]["source_KV_sha256"]
            )
            captured[role], rows[role] = {}, {}
            for mode in MODES:
                before = time.monotonic()
                cache = construct_cache(
                    adapter,
                    prefill.state.past_key_values,
                    positions,
                    encode=mode == "lossless_storage",
                )
                construction_seconds = time.monotonic() - before
                state = DecodeState(
                    cache,
                    torch.ones(
                        (1, max(len(h[0]) for h in positions)),
                        device=adapter.device,
                        dtype=torch.long,
                    ),
                    max(len(h[0]) for h in positions),
                    prefill.state.logical_sequence_length,
                    prefill.state.next_decode_position,
                    prefill.state.modality_map,
                    dict(
                        prefill.state.model_state,
                        mosaickv_packed_all_valid=True,
                        mosaickv_cache_all_valid=True,
                    ),
                )
                values = [prefill.logits.detach().reshape(-1).float().cpu()]
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                with IsolatedStorageTransport(adapter, cache):
                    for token in targets[:-1]:
                        output = adapter.decode_one_token(
                            torch.tensor([[token]], device=adapter.device),
                            state,
                            capture_queries=False,
                        )
                        values.append(output.logits.detach().reshape(-1).float().cpu())
                end.record()
                torch.cuda.synchronize()
                logits = torch.stack(values)
                storage = cache.storage_record()
                if not torch.equal(logits, original[role]["anchor"]["logits"]):
                    difference = (logits - original[role]["anchor"]["logits"]).abs()
                    raise RuntimeError(
                        f"{role}/{mode} changed saved anchor logits: max={float(difference.max())}"
                    )
                if state.logical_sequence_length != prefill.state.logical_sequence_length + 31:
                    raise RuntimeError("logical positions did not advance exactly once per token")
                assert storage["outstanding_layer_leases"] == 0
                assert storage["history_updates_per_layer"] == [31] * len(positions)
                captured[role][mode] = logits
                rows[role][mode] = dict(
                    config_sha=config_sha256(RunConfig.from_mapping(arm["config"])),
                    storage=storage,
                    construction_seconds=construction_seconds,
                    diagnostic_decode_wall_seconds=time.monotonic() - before,
                    diagnostic_decode_cuda_ms=begin.elapsed_time(end),
                    diagnostic_cuda_max_memory_allocated=torch.cuda.max_memory_allocated(),
                    diagnostic_cuda_max_memory_reserved=torch.cuda.max_memory_reserved(),
                    bitwise_logits_equal_saved_anchor=True,
                    source_prompt_checksum=source_checksum,
                    no_selection_change=True,
                    no_mandatory_eviction=True,
                    no_quantization=True,
                    no_FullKV_fallback=True,
                    original_KV_values_and_logical_positions_preserved=True,
                )
                write_once(out / f"{role}-{mode}.json", rows[role][mode])
                del output, state, cache
                settle(adapter)
        if source_sha(pairs) != source_checksum:
            raise RuntimeError("source prefill cache was modified")
        torch.save(captured, out / "logits.pt")
        write_once(out / "measurements.json", rows)
        result.update(
            status="completed",
            completed_replays=4,
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        )
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}"))
        torch.save(captured, out / "partial_logits.pt")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_once(out / "result.json", result)


def analyze(root, expected):
    import pandas as pd
    import torch

    from mosaickv.backends.hf_allocation_guard import decide_draft_stability
    from mosaickv.evaluation.delimiter_control import compare_prefix_logits

    plan = common.inputs(root, expected)
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    rows, failures = [], []
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        if not (folder / "result.json").exists():
            failures.append(dict(worker=job["index"], error="missing worker result"))
            continue
        result = common.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            failures.append(dict(worker=job["index"], errors=result["failures"]))
            continue
        assert result["source_sha"] == plan["source_sha"] and result["plan_sha256"] == expected
        for name, digest in result["outputs"].items():
            assert sha(folder / name) == digest
        measurements = common.read(folder / "measurements.json")
        old = common.read(Path(job["delimiter_worker"]) / "transactions.json")
        captured = torch.load(folder / "logits.pt", map_location="cpu", weights_only=True)
        full = torch.load(
            Path(job["parent_worker"]) / "captures.pt", map_location="cpu", weights_only=True
        )["logits"]["full_kv_reference"]
        for role, modes in measurements.items():
            raw, encoded = (modes[m] for m in MODES)
            assert torch.equal(captured[role]["raw_control"], captured[role]["lossless_storage"])
            guard_rows = compare_prefix_logits(
                full, captured[role]["raw_control"], captured[role]["lossless_storage"]
            )
            guard = decide_draft_stability(
                guard_rows, DraftAllocationGuardConfig.from_mapping(plan["guard"])
            )
            row = dict(
                task=job["task"],
                role=role,
                bitwise_logits_equal_anchor=True,
                unchanged_relative_guard_passed=guard["accepted"],
                base_selection_absolute_limits_passed=all(
                    r["candidate"]["logit_kl"] <= plan["maximum_reference_kl"]
                    and r["candidate"]["maximum_logit_difference"]
                    <= plan["maximum_logit_difference"]
                    for r in guard_rows
                ),
                construction_seconds=encoded["construction_seconds"],
                diagnostic_decode_wall_seconds=encoded["diagnostic_decode_wall_seconds"],
                raw_control_diagnostic_decode_wall_seconds=raw["diagnostic_decode_wall_seconds"],
                construction_and_whole_request_peak_certified=False,
            )
            for key in common.STORAGE:
                row["encoded_" + key] = encoded["storage"][key]
                row["raw_control_" + key] = raw["storage"][key]
                row["saved_anchor_" + key] = old[role]["anchor"]["storage"][key]
                row[key + "_reduction_vs_saved_anchor"] = (
                    1 - encoded["storage"][key] / old[role]["anchor"]["storage"][key]
                )
            row["all_decode_boundaries_lower_than_saved_anchor"] = all(
                encoded["storage"][k] < old[role]["anchor"]["storage"][k] for k in common.STORAGE
            )
            rows.append(row)
    pd.DataFrame(rows).to_parquet(out / "summary.parquet", index=False)
    gate = dict(
        status="complete_consumed_diagnostic" if len(rows) == 4 and not failures else "failed",
        failures=failures,
        comparisons=rows,
        storage_feasibility_passed=bool(
            len(rows) == 4
            and not failures
            and all(
                r["unchanged_relative_guard_passed"]
                and r["all_decode_boundaries_lower_than_saved_anchor"]
                for r in rows
            )
        ),
        selector_superiority_established=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        common_optimization_applied_to_both_methods=True,
    )
    write_once(out / "GATE.json", gate)
    text = (
        "# Shared lossless prompt-storage feasibility\n\n"
        "Same two consumed inputs and unchanged selected K/V for both methods. FullKV parity, "
        "GPU exhaustive bit-pattern checks and bitwise 32-prefix equality are required. "
        "No new selection, donor, reservation, precision, attention backend or guard is used. "
        "The BF16 storage representation changes, not the BF16 attention inputs. Generated "
        "history remains raw and append-only.\n\n"
    )
    if rows:
        text += pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
    text += (
        "Storage counts all encoded component tensors (including bases and literal offsets), "
        "raw fallbacks, generated-history allocations, growth overlap and temporary current-layer "
        "full K/V. It excludes the inactive original prefill source and construction workspaces; "
        "**construction/whole-request KV and HBM peaks are not certified**. Shared-prefill decode "
        "and CPU-packing times are diagnostic, not warmed repeated end-to-end performance. "
        "Bitwise preservation retains the anchor's existing errors: absolute FullKV fidelity "
        "can still fail. No benchmark scores or independent quality claim are produced. "
        "Both Top-K and EviCastKV get this common optimization; its savings cannot establish "
        "EviCastKV selection-policy superiority over an equally optimized competitor.\n"
    )
    (out / "report.md").write_text(text)
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs={str(p): sha(p) for p in (root / "workers").rglob("result.json")},
            outputs={p.name: sha(p) for p in out.iterdir()},
        ),
    )
    write_once(
        root / "recursive_decision.json",
        dict(
            decision="review_common_storage_feasibility_not_selector_superiority",
            storage_feasibility_passed=gate["storage_feasibility_passed"],
            no_fresh_or_large_campaign=True,
            construction_peak_audit_required=True,
        ),
    )
    print(json.dumps(gate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    a = parser.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root))
    elif a.mode == "submit":
        launch.submit(
            a.root,
            a.plan_sha,
            script_name="evicastkv_lossless_storage.sbatch",
            worker_minutes=20,
            memory_gb=16,
        )
    elif a.mode == "worker":
        worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
