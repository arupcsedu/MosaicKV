#!/usr/bin/env python3
"""Finite native-operator and consumed-input original-source reader diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT_SHA = "29805aa4e4f610368d6e1f4165a45e6920a3dc85"
POLICY = dict(
    id="V24_native_broadcast_bounded_original_source_diagnostic_v1",
    consumed_inputs_only=True,
    fresh_cohort_authorized=False,
    whole_request_KV_peak_certified=False,
    paper_campaign_authorized=False,
    source="original_BF16_normalized_input_with_original_KV_bit_patches",
    native_full_token_axis=True,
    native_dispatch="zero_stride_batch32_broadcast",
    dispatch_operator_evidence_SHA="29805aa4e4f610368d6e1f4165a45e6920a3dc85",
    token_reduction_tiling=False,
    workspace_head_token_slots=1024,
    dtype="bf16",
    attention="eager",
    original_49_MiB_all_source_lossless_claim=False,
    exact_byte_matching="actual_source_packet_cost_intersected_with_raw_dense_lattice",
    dummy_padding_or_unused_reservations=False,
    output_predictions=128,
    greedy_token_agreement_required=1.0,
    logit_absolute_tolerance=0.001,
    registered_KL_limit=0.02,
    registered_maximum_logit_difference=1.0,
    disabled_mechanisms=[
        "forecast",
        "evidence_graph",
        "utility",
        "selector",
        "prototype",
        "residual_repair",
    ],
)
ARMS = ("original_source_bounded", "raw_topk_bounded", "raw_retention_1_bounded")


def parity(expected, actual):
    import torch

    if (
        expected.shape != actual.shape
        or expected.ndim != 2
        or expected.shape[0] != 128
        or expected.dtype != torch.float32
        or actual.dtype != torch.float32
        or not torch.isfinite(expected).all()
        or not torch.isfinite(actual).all()
    ):
        raise ValueError("complete finite 128-row FP32 logits required")
    difference = float((expected - actual).abs().max())
    agreement = int((expected.argmax(-1) == actual.argmax(-1)).sum())
    return dict(
        prediction_agreement=agreement,
        predictions=128,
        maximum_logit_difference=difference,
        bitwise_logit_parity=bool(torch.equal(expected, actual)),
        pass_128=bool(agreement == 128 and difference <= POLICY["logit_absolute_tolerance"]),
    )


def read(path):
    return json.loads(Path(path).read_text())


def validate(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy_sha256"] != digest(POLICY) or digest(plan["policy"]) != digest(POLICY):
        raise ValueError("bounded reader immutable policy changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"bounded reader immutable binding changed: {name}")
    return plan


def prepare(root):
    import evicastkv_review_source_prefix as fixed

    revision = clean_source(SOURCE)
    _, jobs, bindings = fixed._bind_parent(fixed.DEFAULT_PARENT)
    original = Path("/scratch/djy8hg/workdir/fox-review-v20/run-20261111-1")
    dispatch_evidence = Path(
        "/scratch/djy8hg/workdir/fox-review-v23-dispatch/run-20261111-1/diagnostic"
    )
    evidence = read(dispatch_evidence / "result.json")
    if evidence["source_sha"] != PARENT_SHA or not evidence["summary"]["broadcast32"]["all_equal"]:
        raise ValueError("hash-bound native broadcast CUDA dispatch gate required")
    for name in ("manifest.json", "result.json"):
        path = dispatch_evidence / name
        bindings[str(path)] = sha(path)
    for index in range(2):
        for name in ("result.json", "logits.pt"):
            path = original / f"workers/{index}/{name}"
            bindings[str(path)] = sha(path)
    write_once(
        Path(root) / "plan.json",
        dict(
            source_sha=revision,
            policy=POLICY,
            policy_sha256=digest(POLICY),
            jobs=jobs,
            bindings=bindings,
            original_reference_root=str(original),
            expected_workers=2,
        ),
    )


def operator(root):
    import torch

    from mosaickv.evaluation.bounded_native_attention_broadcast import selftest

    plan = validate(root)
    out = Path(root) / "operator"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    errors, report = [], {}
    try:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUBLAS workspace config must precede Python")
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one A100 operator diagnostic required")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        report = selftest(device="cuda")
    except BaseException as exc:
        errors.append(
            dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        )
    write_once(
        out / "result.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=sha(Path(root) / "plan.json"),
            operator=report,
            failures=errors,
            wall_seconds=time.monotonic() - start,
            SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    return not errors


def worker(root, index):
    import gc
    from dataclasses import asdict

    import evicastkv_review_source_prefix as fixed
    import torch
    from evicastkv_campaign import environment
    from evicastkv_request_validation import load_messages

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.bounded_native_cache import RawHeadPacket, run
    from mosaickv.evaluation.bounded_reader_budget import FunctionalRawIncrement, solve_exact_budget
    from mosaickv.evaluation.bounded_source_format import construct, reconfigure_cached_heads
    from mosaickv.evaluation.exact_selected_control import (
        build_raw_formats,
        mandatory_positions,
        run_native_same_selected,
        select_uniform_positions,
    )
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = validate(root)
    operator_result = read(root / "operator/result.json")
    if (
        operator_result["failures"]
        or not operator_result["operator"]["all_bitwise_passed"]
        or operator_result["source_sha"] != plan["source_sha"]
    ):
        raise ValueError("native CUDA operator gate failed; pretrained workers blocked")
    if index not in (0, 1):
        raise ValueError("only the two consumed identities allowed")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    rows, errors, tensors = [], [], {}
    start = time.monotonic()
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one deterministic A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUBLAS config must precede Python")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(12)
        torch.manual_seed(13)
        job = plan["jobs"][index]
        config = json.loads(json.dumps(job["config"]))
        config["execution"]["seed"] = 13
        cfg = RunConfig.from_mapping(config)
        if (cfg.model.id, cfg.model.revision) != fixed.MODEL or cfg.model.precision.value != "bf16":
            raise ValueError("same pinned BF16 LLaVA required")
        payload = fixed.label_blind_payload(job["inputs"][0])
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=sha(root / "plan.json"),
                config_sha256=digest(dict(config=config, policy=POLICY)),
                dataset=config["dataset"],
                model=config["model"],
                environment=environment(),
                GPU=asdict(capture_gpu_environment(torch)),
                seed=13,
                backend="hf",
                attention="eager",
                measurement_type="two_consumed_input_bounded_reader_engineering_diagnostic",
                vllm="not_used",
                sglang="not_used",
                benchmark_answers_read=False,
                old_49_MiB_lossless_all_source_unsupported=True,
            ),
        )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        prepared = adapter.prepare_inputs(load_messages(payload))
        with (
            PromptQueryCapture(adapter, summaries=True, normalized=False, prior=False) as captured,
            SourceCapture(adapter) as normalized,
            PhaseCapture(adapter) as rotary,
        ):
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        references = torch.load(
            Path(plan["original_reference_root"]) / f"workers/{index}/logits.pt",
            weights_only=True,
            map_location="cpu",
        )
        reference = references["FullKV_native_greedy"]
        if not torch.equal(prefill.logits.reshape(-1).float().cpu(), reference[0]):
            raise ValueError("fresh prefill differs from immutable original FullKV")
        tensors["FullKV_reference"] = reference
        prefix = fixed.forced_prefix(reference)
        attention_modules = [layer.self_attn for layer in adapter._language_layers()]
        torch.cuda.synchronize()
        constructor_start = time.monotonic()
        packets = tuple(
            construct(
                normalized.records[i],
                tuple(pair),
                attention_modules[i],
                rotary.phases,
                take_source_ownership=True,
            )
            for i, pair in enumerate(pairs)
        )
        source_base = backing_bytes(tuple(t for p in packets for t in p.tensors()))
        n, heads = packets[0].shape[-2], packets[0].shape[1]
        mandatory = mandatory_positions(prepared, n)
        increments = tuple(
            FunctionalRawIncrement(i, h, p.raw_head_bytes, p.patch_bytes_by_head[h])
            for i, p in enumerate(packets)
            for h in range(heads - 1)
        )
        witness = solve_exact_budget(
            source_base,
            increments,
            source_positions=n,
            reserved_source_heads=tuple((i, heads - 1) for i in range(len(packets))),
            baseline_min_positions=len(mandatory),
        )
        write_once(out / "budget-search.json", witness.as_dict())
        if not witness.supported:
            raise ValueError("no actually attainable exact source/raw byte intersection found")
        selected_heads = set(witness.selected_raw_heads)
        packets = tuple(
            reconfigure_cached_heads(
                p, tuple(pairs[i]), tuple(h for h in range(heads) if (i, h) in selected_heads)
            )
            for i, p in enumerate(packets)
        )
        actual_source = backing_bytes(tuple(t for p in packets for t in p.tensors()))
        if actual_source != witness.matched_bytes:
            raise RuntimeError("constructed source owners disagree with actual-cost witness")
        positions = tuple(
            select_uniform_positions(
                captured.windows[i][0].float().mean(-2), witness.baseline_positions, mandatory
            )
            for i in range(len(packets))
        )
        formats = build_raw_formats(adapter, pairs, positions, mandatory=mandatory)
        raw_packets = tuple(RawHeadPacket(tuple(b.buffers[0] for b in f.pair)) for f in formats)
        actual_raw = backing_bytes(tuple(t for p in raw_packets for t in p.tensors()))
        if actual_source != actual_raw:
            raise RuntimeError("actually constructed raw/source initial payloads do not match")
        torch.cuda.synchronize()
        construction = dict(
            source_base_bytes=source_base,
            matched_packet_bytes=actual_source,
            raw_baseline_positions=witness.baseline_positions,
            functional_raw_heads=len(selected_heads),
            source_packets=[p.diagnostics for p in packets],
            CPU_position_metadata_bytes=backing_bytes(tuple(f.original_positions for f in formats)),
            constructor_seconds=time.monotonic() - constructor_start,
            old_49_MiB_lossless_all_source_unsupported=source_base > 49 * 2**20,
            budget_changed_explicitly=True,
            no_padding_or_unused_fill=True,
            verifier_original_prefill_and_source_captures_retained=True,
            verifier_owners_excluded_from_constructed_decode_cache_boundary=True,
            whole_request_KV_peak_certified=False,
        )
        write_once(out / "construction.json", construction)
        same_selected, _ = run_native_same_selected(
            adapter, prefill, formats, tokens=128, forced=prefix
        )
        tensors["raw_native_same_selected"] = same_selected
        for arm in ARMS:
            if arm == ARMS[0]:
                arm_packets, expected = packets, reference
            elif arm == ARMS[1]:
                arm_packets, expected = raw_packets, same_selected
            else:
                arm_packets = tuple(
                    RawHeadPacket(tuple(t.contiguous().clone() for t in pair)) for pair in pairs
                )
                expected = reference
            logits, measured = run(
                adapter, prefill, arm_packets, forced=prefix, slots=1024, dispatch="broadcast32"
            )
            tensors[arm] = logits
            parity_record = parity(expected, logits)
            row = dict(
                arm=arm,
                status="completed",
                sample_id=payload["sample_id"],
                task=payload["task"],
                source_sha=plan["source_sha"],
                plan_sha256=sha(root / "plan.json"),
                parity=parity_record,
                measured=measured,
                FullKV_diagnostic=fixed.comparisons(reference, logits),
                fullkv_fallback=False,
                construction=construction,
            )
            rows.append(row)
            write_once(out / "traces" / f"{arm}.json", row)
            gc.collect()
    except BaseException as exc:
        errors.append(
            dict(
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                worker_index=index,
                stage_or_arm=locals().get("arm", "construction_or_setup"),
            )
        )
    finally:
        if tensors:
            torch.save(tensors, out / "logits.pt")
        with (out / "raw.jsonl").open("x") as stream:
            for item in rows + [dict(status="failed", **error) for error in errors]:
                stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        write_once(
            out / "result.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=sha(root / "plan.json"),
                actual_rows=len(rows),
                failures=errors,
                outputs=outputs,
                wall_seconds=time.monotonic() - start,
                SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
                SLURM_ARRAY_JOB_ID=os.environ.get("SLURM_ARRAY_JOB_ID"),
                SLURM_ARRAY_TASK_ID=os.environ.get("SLURM_ARRAY_TASK_ID"),
            ),
        )


def analyze(root):
    import evicastkv_review_source_prefix as fixed
    import torch

    root = Path(root)
    plan = validate(root)
    fixed.analysis_cpu_controls()
    failures, observations = [], []
    for index in range(2):
        try:
            out = root / f"workers/{index}"
            result = read(out / "result.json")
            if result["failures"] or result["actual_rows"] != 3:
                raise ValueError(f"worker {index} incomplete: {result['failures']}")
            if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
                root / "plan.json"
            ):
                raise ValueError("worker producer provenance mismatch")
            for name, expected in result["outputs"].items():
                if sha(out / name) != expected:
                    raise ValueError(f"worker immutable output changed: {name}")
            rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
            if len(rows) != 3 or tuple(row["arm"] for row in rows) != ARMS:
                raise ValueError("exactly three registered ordered arms required")
            saved = torch.load(out / "logits.pt", weights_only=True, map_location="cpu")
            for row in rows:
                comparator = (
                    saved["raw_native_same_selected"]
                    if row["arm"] == "raw_topk_bounded"
                    else saved["FullKV_reference"]
                )
                recomputed = parity(comparator, saved[row["arm"]])
                if digest(recomputed) != digest(row["parity"]) or any(
                    type(row["parity"][key]) is not bool
                    for key in ("pass_128", "bitwise_logit_parity")
                ):
                    raise ValueError("saved parity diagnostics mismatch")
                fixed.verify_recomputed_diagnostics(
                    fixed.comparisons(saved["FullKV_reference"], saved[row["arm"]]),
                    row["FullKV_diagnostic"],
                )
            boundaries = (
                "initial_active_kv_storage_bytes",
                "final_active_kv_storage_bytes",
                "peak_active_kv_storage_bytes",
            )
            matched = all(
                rows[0]["measured"]["storage"][key] == rows[1]["measured"]["storage"][key]
                for key in boundaries
            )
            observations.append(
                dict(
                    worker=index,
                    sample_id=rows[0]["sample_id"],
                    parity_all_arms_128=all(row["parity"]["pass_128"] for row in rows),
                    constructed_initial_final_decode_peak_exactly_matched=matched,
                    source_storage=rows[0]["measured"]["storage"],
                    raw_storage=rows[1]["measured"]["storage"],
                    source_parity=rows[0]["parity"],
                    raw_parity=rows[1]["parity"],
                    retention_1_parity=rows[2]["parity"],
                )
            )
        except BaseException as exc:
            failures.append(dict(worker=index, error=f"{type(exc).__name__}: {exc}"))
    complete = not failures and len(observations) == 2
    gate = dict(
        source_sha=plan["source_sha"],
        plan_sha256=sha(root / "plan.json"),
        complete=complete,
        failures=failures,
        observations=observations,
        source_fidelity_128_pass=complete and all(o["parity_all_arms_128"] for o in observations),
        constructed_decode_boundary_byte_match_pass=complete
        and all(o["constructed_initial_final_decode_peak_exactly_matched"] for o in observations),
        old_49_MiB_lossless_source_unsupported=True,
        whole_request_KV_peak_certified=False,
        independent_quality_confirmation=False,
        paper_campaign_authorized=False,
        numerical_tolerances_unchanged=True,
    )
    write_once(root / "analysis/GATE_bounded_reader_v24.json", gate)
    return gate


def submit(root, revision, label, mode, *, dependencies=(), gpu=False, array=False):
    root = Path(root)
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox24-{label}",
        f"--partition={'bii-gpu' if gpu else 'bii,standard'}",
        f"--cpus-per-task={12 if gpu or mode == 'analyze' else 4}",
        f"--mem={'64G' if gpu else '24G'}",
        f"--time=00:{30 if mode == 'tests' else 15 if array else 5}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if gpu:
        command.extend(["--gres=gpu:a100:1", "--reservation=bi_fox_dgx"])
    if array:
        command.append("--array=0-1%2")
    if dependencies:
        if any(not str(job).isdigit() for job in dependencies):
            raise ValueError("numeric acknowledged dependencies required")
        command.append("--dependency=afterany:" + ":".join(map(str, dependencies)))
    command.extend(
        [
            str(SOURCE / "mosaickv/slurm/fox_bounded_reader_v24.sbatch"),
            str(SOURCE),
            str(root),
            mode,
            revision,
        ]
    )
    write_once(root / f"jobs/{label}-request.json", dict(command=command, source_sha=revision))
    output = subprocess.check_output(command, text=True, timeout=30).strip()
    job = output.split(";")[0]
    if not job.isdigit():
        raise ValueError("uncertain Slurm acknowledgement; do not duplicate submission")
    write_once(
        root / f"jobs/{label}-submission.json",
        dict(job=job, command=command, source_sha=revision, automatic_retry=False),
    )
    return job


def launch(root):
    revision = clean_source(SOURCE)
    modules = subprocess.check_output(
        ["git", "submodule", "status", "--recursive"], cwd=SOURCE, text=True
    )
    if any(line[:1] in ("-", "+", "U") for line in modules.splitlines()):
        raise ValueError("all six pinned submodules must be initialized and unchanged")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT_SHA, revision], cwd=SOURCE, check=True
    )
    root = Path(root).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if not root.is_relative_to(allowed) or root == allowed:
        raise ValueError("new immutable scratch output directory required")
    root.mkdir(parents=True, exist_ok=False)
    (root / "jobs").mkdir()
    (root / "slurm").mkdir()
    prepare(root)
    write_once(
        root / "launch.json",
        dict(
            source_sha=revision,
            policy_sha256=digest(POLICY),
            maximum_a100s=6,
            registered_operator_gpus=1,
            registered_model_workers=2,
            worker_cpus=12,
            consumed_inputs_only=True,
            whole_request_KV_peak_certified=False,
            automatic_finite_afterany_chain=True,
            continuous_polling=False,
        ),
    )
    tests = submit(root, revision, "cpu", "tests")
    op = submit(root, revision, "operator", "operator", gpu=True)
    dispatch = submit(root, revision, "dispatch", "dispatch", dependencies=(tests, op))
    print(json.dumps(dict(cpu=tests, operator=op, dispatch=dispatch, root=str(root))))


def dispatch(root):
    root = Path(root)
    plan = validate(root)
    errors = []
    try:
        cpu = read(root / "full-cpu/cpu-tests.json")
        if cpu["status"] != "passed" or cpu["source_sha"] != plan["source_sha"]:
            raise ValueError("new complete CPU suite did not pass")
        operator_result = read(root / "operator/result.json")
        if operator_result["failures"] or not operator_result["operator"]["all_bitwise_passed"]:
            raise ValueError("native CUDA operator parity did not pass")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    workers = None
    if not errors:
        workers = submit(root, plan["source_sha"], "gpu", "worker", gpu=True, array=True)
    analysis = submit(
        root,
        plan["source_sha"],
        "analysis",
        "analyze",
        dependencies=() if workers is None else (workers,),
    )
    write_once(
        root / "jobs/dispatch-result.json",
        dict(
            blockers=errors, workers=workers, analysis=analysis, model_workers_blocked=bool(errors)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "operator", "worker", "analyze", "launch", "dispatch")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker" and args.index not in (0, 1):
        parser.error("worker requires --index 0 or 1")
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "operator":
        operator(args.root)
    elif args.mode == "worker":
        worker(args.root, args.index)
    elif args.mode == "analyze":
        analyze(args.root)
    elif args.mode == "launch":
        launch(args.root)
    else:
        dispatch(args.root)


if __name__ == "__main__":
    main()
