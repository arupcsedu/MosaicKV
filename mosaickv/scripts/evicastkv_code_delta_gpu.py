#!/usr/bin/env python3
"""Bounded lossless-code replay/model diagnostic on two CONSUMED inputs only."""

import argparse
import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import evicastkv_common_streaming as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, previous = common.SOURCE, common.previous
REFERENCE = common.AUDIT
CPU_STUDY = Path("/scratch/djy8hg/runs/evicastkv/source-code-delta-cpu-consumed-20261101-v1")


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable code-delta GPU diagnostic root required")
    old = previous.read(REFERENCE / "plan.json")
    check = previous.read(REFERENCE / "analysis/decision.json")
    feasible = previous.read(CPU_STUDY / "decision.json")
    if (
        not check["all_five_methods_streaming_identity_gate"]
        or check["failures"]
        or not feasible["GPU_implementation_feasibility_gate"]
        or feasible["failures"]
    ):
        raise ValueError("complete frozen all-method identity and CPU feasibility required")
    bound = dict(old["bound_files"])
    for path in (
        REFERENCE / "plan.json",
        REFERENCE / "analysis/decision.json",
        CPU_STUDY / "plan.json",
        CPU_STUDY / "decision.json",
        SOURCE / "mosaickv/docs/lossless_code_delta_gpu_20261101.md",
    ):
        bound[str(path)] = sha(path)
    for job in old["jobs"]:
        folder = REFERENCE / f"workers/{job['index']}"
        record = previous.read(folder / "result.json")
        if record["status"] != "completed" or record["failures"]:
            raise ValueError("all frozen producers must be complete")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, expected in record["outputs"].items():
            if sha(folder / name) != expected:
                raise ValueError("frozen producer changed")
            bound[str(folder / name)] = expected
    jobs = [
        dict(j, reference_worker=j["index"], index=i)
        for i, j in enumerate(j for j in old["jobs"] if common.method_of(j) == "mosaickv_exact")
    ]
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            jobs=jobs,
            bound_files=bound,
            stage="consumed_lossless_GPU_code_chain_model_diagnostic",
            maximum_worker_minutes=90,
            maximum_simultaneous_workers=2,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
            GPU_consumed_diagnostic_authorized=True,
            exact_minimum_only=True,
            identical_128_logits_and_complete_teacher_logits_required=True,
            code_replay_microbenchmark_warmups=5,
            code_replay_microbenchmark_repetitions=20,
            source_decode_pair_warmups=5,
            source_decode_pair_repetitions=20,
            prefill_peak_improvement_claim_authorized=False,
            common_codec_offered_to_all_five_methods=True,
            fallback_is_original_same_method_not_FullKV=True,
            comparison_is_common_source_capacity_not_equal_encoded_bytes=True,
        ),
    )
    print(sha(root / "plan.json"))


def controls():
    import numpy as np
    import torch

    from mosaickv.evaluation.lossless_code_delta import encode as oracle
    from mosaickv.evaluation.lossless_code_delta_gpu import decode, encode

    generator = torch.Generator(device="cuda").manual_seed(20261101)
    cases = []
    for mode in ("identical", "sparse", "random", "mixed"):
        previous_bytes = torch.randint(
            0, 256, (4096,), device="cuda", dtype=torch.uint8, generator=generator
        )
        current = previous_bytes.clone()
        if mode == "sparse":
            current[::129] ^= 1
        elif mode == "random":
            current.random_(0, 256, generator=generator)
        elif mode == "mixed":
            current[128:256].random_(0, 256, generator=generator)
            current[::257] ^= 1
        packet = encode(previous_bytes, current)
        cpu = oracle(previous_bytes.cpu().numpy(), current.cpu().numpy())
        if packet.raw != cpu.raw or not np.array_equal(packet.payload.cpu().numpy(), cpu.payload):
            raise RuntimeError("GPU serialization differs from CPU reference")
        if not packet.raw and not np.array_equal(
            packet.offsets.cpu().numpy().view(np.uint32), cpu.offsets
        ):
            raise RuntimeError("GPU raw flags/offsets differ from CPU reference")
        for inplace in (False, True):
            old = previous_bytes.clone()
            value = decode(old, packet, old if inplace else None)
            if not torch.equal(value, current):
                raise RuntimeError("GPU replay or in-place predecessor changed bytes")
        cases.append(
            dict(
                mode=mode,
                raw_fallback=packet.raw,
                exact=True,
                serialized_bytes=packet.serialized_bytes,
            )
        )
    current = torch.zeros(128, dtype=torch.uint8, device="cuda")
    if (
        not encode(None, current).raw
        or not encode(torch.zeros(256, dtype=torch.uint8, device="cuda"), current).raw
    ):
        raise RuntimeError("missing/changed predecessor must use raw fallback")
    return cases


def construct(adapter, messages, profile, method):
    import torch

    from mosaickv.evaluation.streaming_baselines import SelectedRetiringCache, StreamingBaseline
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention
    from mosaickv.evaluation.streaming_source_constructor import StreamingDirectSource

    prepared = adapter.prepare_inputs(messages)
    cache = (
        RetiringPrefillCache(len(profile), audit=False)
        if method == "mosaickv_exact"
        else SelectedRetiringCache(len(profile), audit=False)
    )
    encoder = (
        StreamingDirectSource(adapter, prepared, cache, profile, minimum_only=True)
        if method == "mosaickv_exact"
        else StreamingBaseline(prepared, cache, profile, method, minimum_only=True)
    )
    fresh = replace(prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache))
    with SealAfterAttention(adapter, cache, encoder):
        pref = adapter.prefill(fresh, capture_queries=False, capture_attentions=False)
    with torch.inference_mode():
        encoder.finish() if method == "mosaickv_exact" else encoder.finish(adapter)
    packets = cache.formats()
    runtime = replace(pref, state=replace(pref.state, past_key_values=None), attention_weights=())
    prepared.model_inputs.clear()
    del prepared, pref, fresh, cache, encoder
    gc.collect()
    return runtime, packets


def decode_run(adapter, runtime, packets):
    from mosaickv.evaluation.storage_runs import run

    return run(
        adapter,
        runtime,
        packets,
        tokens=128,
        audit=False,
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        source_history_constructor="direct_batched_checks",
        consolidated_history=True,
    )


def source_timing(adapter, runtime, original, compressed, expected_logits, output):
    import numpy as np
    import torch

    rng, rows = random.Random(20261101), []
    for trial in range(-5, 20):
        order = [False, True]
        rng.shuffle(order)
        for position, enabled in enumerate(order):
            gc.collect()
            values, measured = decode_run(adapter, runtime, compressed if enabled else original)
            if not torch.equal(values, expected_logits):
                raise RuntimeError("timed lossless replay changed a vocabulary logit")
            row = dict(
                trial=trial,
                warmup=trial < 0,
                order_index=position,
                code_delta=enabled,
                wall_seconds=measured["decode_wall_seconds"],
                cuda_ms=measured["decode_cuda_ms"],
                logits_bitwise=True,
                includes_construction=False,
                includes_cache_tensor_audit=False,
                independent_quality_N=0,
            )
            with (output / "decode_trials.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
    old = np.asarray(
        [
            next(r["wall_seconds"] for r in rows if r["trial"] == i and not r["code_delta"])
            for i in range(20)
        ]
    )
    new = np.asarray(
        [
            next(r["wall_seconds"] for r in rows if r["trial"] == i and r["code_delta"])
            for i in range(20)
        ]
    )
    ids = np.random.default_rng(20261101).integers(0, 20, (10000, 20))
    ratios = new[ids].mean(1) / old[ids].mean(1)
    write_once(
        output / "source_decode_timing.json",
        dict(
            old_mean=float(old.mean()),
            new_mean=float(new.mean()),
            ratio=float(new.mean() / old.mean()),
            paired_CI95=np.quantile(ratios, [0.025, 0.975]).tolist(),
            old_p50=float(np.median(old)),
            old_p95=float(np.quantile(old, 0.95)),
            new_p50=float(np.median(new)),
            new_p95=float(np.quantile(new, 0.95)),
            engineering_repetitions_not_independent_examples=True,
            not_end_to_end_or_baseline_latency_evidence=True,
        ),
    )


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.common_streaming_timing import METHODS
    from mosaickv.evaluation.layer_code_delta import bind, convert
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.source_teacher_protocol import load_targets
    from mosaickv.evaluation.storage_teacher_forcing import score_formats
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    output = root / f"workers/{index}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw.jsonl").touch(exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    method = "primitive_controls"
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one A100 and deterministic launch environment required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        manifest = previous.read(REFERENCE / f"workers/{job['reference_worker']}/manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            config_sha256=hashlib.sha256((expected + str(index)).encode()).hexdigest(),
            evaluator_commit=plan["source_sha"],
            environment=common.control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            measurement_type="diagnostic_measured",
            independent_quality_N=0,
            variant="common_lossless_layer_code_delta",
            whole_request_peak_claim=False,
            representation_not_official_baseline=True,
        )
        write_once(output / "manifest.json", manifest)
        write_once(output / "primitive_controls.json", controls())
        cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = previous.read(job["inputs"][0])
        messages = previous.load_messages(payload)
        profile = tuple(previous.read(Path(plan["profile_path"])))
        originals = previous.read(REFERENCE / "plan.json")["jobs"]
        source = None
        for method in METHODS:
            reference_job = next(
                j
                for j in originals
                if j["input_index"] == job["input_index"] and common.method_of(j) == method
            )
            folder = REFERENCE / f"workers/{reference_job['index']}"
            reference = previous.read(folder / "offline_payload.json")
            prior = json.loads((folder / "raw.jsonl").read_text().splitlines()[0])
            expected_logits = torch.load(
                folder / "greedy_logits.pt", weights_only=True, map_location="cpu"
            )
            runtime, packets = construct(adapter, messages, profile, method)
            if (
                common.control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
                != reference["payload_sha256"]
            ):
                raise RuntimeError("original streamed packet differs from frozen all-method parent")
            if not torch.equal(runtime.logits.cpu().reshape(-1), expected_logits[0].reshape(-1)):
                raise RuntimeError("original prefill logits changed")
            torch.cuda.synchronize()
            start = time.monotonic()
            encoded, construction = convert(packets)
            torch.cuda.synchronize()
            construction["GPU_encode_and_proof_wall_seconds"] = time.monotonic() - start
            bound = bind(encoded)
            actual = backing_bytes(tuple(t for p in bound for t in p.tensors()))
            if actual != construction["net_active_bytes"]:
                raise RuntimeError(
                    "persistent bytes plus charged replay scratch differ from backing storage"
                )
            for i, (original, updated, block) in enumerate(
                zip(packets, bound, adapter._language_layers(), strict=True)
            ):
                view = updated.replay.view(i) if hasattr(updated, "replay") else updated
                a, b = original.materialize(block.self_attn), view.materialize(block.self_attn)
                if not all(
                    torch.equal(x.view(torch.int16), y.view(torch.int16))
                    for x, y in zip(a, b, strict=True)
                ):
                    raise RuntimeError("lossless code replay changed actual reconstructed K/V")
                del a, b, view
            del bound
            values, details = decode_run(adapter, runtime, encoded)
            if not torch.equal(values, expected_logits):
                raise RuntimeError("lossless replay changed frozen 128-token vocabulary logits")
            targets, provenance = load_targets(plan, payload, adapter.processor.tokenizer)
            teacher, teacher_logits = score_formats(
                adapter,
                runtime,
                encoded,
                targets,
                maximum=1024,
                sequential_source_history=True,
                audit_history=False,
                return_logits=True,
            )
            original_teacher = prior["teacher_forced"]
            if common.control.digest_tensors((teacher_logits,)) != original_teacher["storage"][
                "raw_history_control"
            ]["source_history_logit_sha256"] or any(
                teacher[k] != original_teacher[k]
                for k in (
                    "nll_sum",
                    "per_token_nll",
                    "conditional_greedy_tokens",
                    "evaluated_token_count",
                )
            ):
                raise RuntimeError("lossless replay changed complete teacher logits or NLL")
            if details["storage"]["initial_active_kv_storage_bytes"] != actual:
                raise RuntimeError("runtime active-byte accounting omitted replay workspace")
            row = dict(
                status="completed",
                method=method,
                sample_id=payload["sample_id"],
                task=payload["task"],
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                manifest_path=str(output / "manifest.json"),
                config_sha256=manifest["config_sha256"],
                measurement_type="diagnostic_measured",
                constructed_code_payload=construction,
                decoded_KV_bitwise=True,
                generation_128_logits_bitwise=True,
                complete_teacher_logits_bitwise=True,
                initial_active_bytes=actual,
                generation_storage=details["storage"],
                parent_generation_storage=prior["generation_storage"],
                teacher_forced=teacher,
                teacher_provenance=provenance,
                prefill_peak_not_improved=True,
                original_control_packets_simultaneously_resident=True,
                whole_request_GPU_peak_certified=False,
                independent_quality_N=0,
            )
            with (output / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if method == "mosaickv_exact" and construction["enabled"]:
                source = (runtime, packets, encoded, expected_logits)
            del packets, encoded, values, teacher_logits, details, row
            gc.collect()
        if source is not None:
            source_timing(adapter, *source, output)
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        with (output / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(dict(status="failed", method=method, error=result["failures"][-1]))
                + "\n"
            )
        raise
    finally:
        result["outputs"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        write_once(output / "result.json", result)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    rows, failures, timing = [], [], []
    for i in range(2):
        folder = root / f"workers/{i}"
        try:
            record = previous.read(folder / "result.json")
            if record["source_sha"] != plan["source_sha"] or record["plan_sha256"] != expected:
                raise ValueError("producer source or plan mismatch")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer artifact changed")
            failures.extend(record["failures"])
            rows.extend(
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            )
            if (folder / "source_decode_timing.json").exists():
                timing.append(dict(case=i, **previous.read(folder / "source_decode_timing.json")))
        except (OSError, ValueError) as exc:
            failures.append(dict(case=i, error=str(exc)))
    complete = len(rows) == 10 and not failures and all(r["status"] == "completed" for r in rows)
    source = [r for r in rows if r["method"] == "mosaickv_exact" and r["status"] == "completed"]
    write_once(
        root / "analysis/decision.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            complete=complete,
            failures=failures,
            rows=rows,
            source_decode_timing=timing,
            all_five_methods_bitwise=complete,
            source_net_storage_gate=complete
            and len(source) == 2
            and all(r["constructed_code_payload"]["enabled"] for r in source),
            actual_integrated_storage_not_estimate=True,
            prefill_peak_improvement_established=False,
            benchmark_superiority_established=False,
            independent_quality_N=0,
            paper_or_fresh_campaign_authorized=False,
        ),
    )


def submit(root, expected):
    original = common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_code_delta_gpu.sbatch")
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(common.control.parent.subprocess, "check_output", scoped):
        common.control.parent.submit(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
