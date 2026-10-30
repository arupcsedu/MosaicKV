#!/usr/bin/env python3
"""Consumed tensor capture -> two bounded CPU codec studies -> no GPU promotion."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import evicastkv_paired_storage_baselines as paired

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base = paired.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/predictive-storage-consumed-20261030-v1")
original_writer = paired.write_once


def prepare(root, phase):
    passed = Path("/scratch/djy8hg/runs/evicastkv/direct-constructor-consumed-20261030-v1")
    gate = base.common.read(passed / "analysis/GATE.json")
    if gate["failures"] or gate["status"] != "complete_consumed_engineering_diagnostic":
        raise ValueError("direct constructor must pass before tensor-only feasibility")

    def register(path, value):
        if path.name == "plan.json":
            for name in (
                "plan.json",
                "analysis/GATE.json",
                "analysis/manifest.json",
                "constructor_analysis/summary.json",
            ):
                value["bound_files"][str(passed / name)] = sha(passed / name)
            value.update(
                protocol="lossless_row_predictor_CPU_feasibility_v1",
                predictor_kinds=["none", "xor16", "ordered_delta16"],
                codec_kinds=["bf16_exponents", "sparse_high_byte"],
                predictor_rows=16,
                predictor_header_bytes=4,
                minimum_evicast_prospective_reduction=0.03,
                cpu_codec_workers=2,
                cpu_codec_cpus=4,
                cpu_codec_minutes=30,
                finite_cpu_followup_authorized=True,
                automatic_gpu_decoder_authorized=False,
                no_attention_or_selection_change=True,
                no_quality_claim=True,
                maximum_capture_bytes_per_worker=2 * 2**30,
            )
        original_writer(path, value)

    paired.write_once = register
    try:
        return paired.prepare(root, phase)
    finally:
        paired.write_once = original_writer


def capture(adapter, prefill, captured, frozen, job, cfg, full_reference, prepared, out, plan):
    import torch

    from mosaickv.evaluation.direct_constructor import construct
    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.storage_baselines import baseline_payload

    counts = []
    for method in paired.METHODS:
        positions, pairs, selection = baseline_payload(
            adapter,
            prepared,
            prefill,
            cfg,
            method,
            frozen,
            cfg.bounded_decode.incremental_history.layer_profile_path,
        )
        if pairs is None:
            formats, _ = construct(adapter, prefill, captured, positions)
        else:
            formats = tuple(
                ExactFormat("kv", tuple(k.shape), pair=(encode(k), encode(v))) for k, v in pairs
            )
        tensors = []
        for layer, format in enumerate(formats):
            bodies = [("KV_or_direct_KV", p) for p in format.pair]
            if format.archive is not None:
                bodies.append(("normalized_source", format.archive.source))
            for body_index, (role, body) in enumerate(bodies):
                if body.shared_literal_window:
                    raise ValueError("prompt payloads must not borrow history literal windows")
                tensors.append(
                    dict(
                        layer=layer,
                        body_index=body_index,
                        role=role,
                        shape=body.shape,
                        raw=body.raw,
                        buffers=tuple(t.cpu() for t in body.buffers),
                        original_GPU_payload_bytes=backing_bytes(body.buffers),
                    )
                )
        payload = sum(t["original_GPU_payload_bytes"] for t in tensors)
        actual = backing_bytes(tuple(t for f in formats for t in f.tensors()))
        if payload > actual:
            raise RuntimeError("prompt payload ownership is not independently additive")
        artifact = dict(
            method=method,
            selection=json.loads(json.dumps(selection)),
            bodies=tensors,
            original_GPU_initial_component_bytes=actual,
            original_GPU_payload_bytes=payload,
            non_payload_bytes=actual - payload,
            no_model_weights_or_labels=True,
        )
        torch.save(artifact, out / f"{method}-codec-input.pt")
        if (
            sum(p.stat().st_size for p in out.glob("*-codec-input.pt"))
            > plan["maximum_capture_bytes_per_worker"]
        ):
            raise RuntimeError("bounded prompt capture exceeded two GiB; stop without more exports")
        counts.append(
            dict(
                method=method,
                payload_bytes=payload,
                actual_initial_bytes=actual,
                tensor_count=len(tensors),
                captured_prompt_only=True,
            )
        )
        del formats, tensors, artifact
    return [
        dict(
            method="all_primary_codec_capture",
            methods=counts,
            benchmark_labels_used=False,
            model_outputs_not_used_for_codec_selection=True,
        )
    ]


def advance(root, expected):
    paired.original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    if gate["failures"]:
        return
    launch_cpu(root, expected)


def launch_cpu(root, expected):
    base.common.inputs(root, expected)
    (root / "slurm").mkdir(exist_ok=True)
    (root / "analysis").mkdir(exist_ok=True)
    script = str(base.SOURCE / "mosaickv/slurm/evicastkv_predictive_feasibility.sbatch")
    args = [str(base.SOURCE), str(root), expected]
    cpu = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                "--partition=bii",
                "--job-name=mk-codec-feasibility",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=4",
                "--mem=12G",
                "--time=00:30:00",
                "--array=0-1%2",
                f"--output={root}/slurm/codec-%A_%a.out",
                f"--error={root}/slurm/codec-%A_%a.err",
                script,
                *args,
                "codec_worker",
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    final = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                "--partition=bii",
                "--job-name=mk-codec-analysis",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=2",
                "--mem=4G",
                "--time=00:10:00",
                f"--dependency=afterany:{cpu}",
                f"--output={root}/slurm/codec-analysis-%j.out",
                f"--error={root}/slurm/codec-analysis-%j.err",
                script,
                *args,
                "codec_aggregate",
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    write_once(
        root / "analysis/cpu_advancement.json",
        dict(
            cpu_array=cpu,
            cpu_aggregate=final,
            no_further_GPU_authorization=True,
            finite_cpu_only=True,
        ),
    )


def prepare_cpu_retry(root):
    parent = ROOT
    if root.exists():
        raise FileExistsError("new immutable CPU-only retry root required")
    gate = base.common.read(parent / "codec_analysis/GATE.json")
    if not gate["failures"] or any(
        "mosaickv.types.BudgetUnit" not in " ".join(f.get("errors", [])) for f in gate["failures"]
    ):
        raise ValueError("retry only the observed metadata-enum load boundary")
    original = base.common.read(parent / "plan.json")
    bound = dict(original["bound_files"])
    for name in (
        "plan.json",
        "analysis/GATE.json",
        "codec_analysis/GATE.json",
        "codec_analysis/manifest.json",
    ):
        bound[str(parent / name)] = sha(parent / name)
    for job in original["jobs"]:
        folder = parent / f"workers/{job['index']}"
        path = folder / "result.json"
        result = base.common.read(path)
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("retry requires completed unchanged GPU captures")
        bound[str(path)] = sha(path)
        for name, digest in result["outputs"].items():
            bound[str(folder / name)] = digest
    original.update(
        source_sha=clean_source(base.SOURCE),
        bound_files=bound,
        capture_root=str(parent),
        capture_source_sha=original["source_sha"],
        capture_plan_sha256=sha(parent / "plan.json"),
        CPU_only_retry=True,
        automatic_gpu_decoder_authorized=False,
        model_inference_authorized=False,
        protocol="CPU_only_enum_metadata_repair_no_new_capture_v1",
    )
    write_once(root / "plan.json", original)
    return sha(root / "plan.json")


def codec_worker(root, expected, index):
    import torch

    from mosaickv.evaluation.lossless_bf16 import PackedBF16, pack_cpu, unpack_cpu
    from mosaickv.evaluation.predictive_storage import (
        decode_prediction,
        encode_prediction,
        pack_sparse_high,
        unpack_sparse_high,
    )

    plan = base.common.inputs(root, expected)
    job = plan["jobs"][index]
    parent = Path(plan.get("capture_root", root)) / f"workers/{index}"
    manifest = base.common.read(parent / "result.json")
    if (
        manifest["status"] != "completed"
        or manifest["failures"]
        or manifest["source_sha"] != plan.get("capture_source_sha", plan["source_sha"])
        or manifest["plan_sha256"] != plan.get("capture_plan_sha256", expected)
    ):
        raise ValueError("complete source-matched GPU capture required")
    out = root / f"codec_workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        task=job["task"],
        sample_id=job["sample_id"],
        methods=[],
    )
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        for method in paired.METHODS:
            path = parent / f"{method}-codec-input.pt"
            if sha(path) != manifest["outputs"][path.name]:
                raise ValueError("captured prompt tensor changed")
            from mosaickv.types import BudgetUnit

            # Only this repository enum is allowlisted, only inside this
            # hash-verified load. Never switch to unrestricted pickle loading.
            with torch.serialization.safe_globals([BudgetUnit]):
                artifact = torch.load(path, map_location="cpu", weights_only=True)
            rows = []
            for j, body in enumerate(artifact["bodies"]):
                packed = PackedBF16(tuple(body["shape"]), body["raw"], tuple(body["buffers"]))
                original = unpack_cpu(packed)
                proposals = []
                for kind in plan["predictor_kinds"]:
                    prediction = encode_prediction(original, kind)
                    for codec in plan["codec_kinds"]:
                        began = time.monotonic()
                        if codec == "bf16_exponents":
                            packet = pack_cpu(prediction)
                            payload = sum(t.untyped_storage().nbytes() for t in packet.buffers)
                            decoded = unpack_cpu(packet)
                        elif codec == "sparse_high_byte":
                            packet = pack_sparse_high(prediction)
                            payload = packet.storage_bytes
                            decoded = unpack_sparse_high(packet)
                        else:
                            raise ValueError("unregistered codec")
                        reconstructed = decode_prediction(decoded, kind)
                        if not torch.equal(
                            reconstructed.view(torch.int16), original.view(torch.int16)
                        ):
                            raise RuntimeError("lossless predictor changed an original BF16 bit")
                        legacy = kind == "none" and codec == "bf16_exponents"
                        if legacy and payload != body["original_GPU_payload_bytes"]:
                            raise RuntimeError("CPU/GPU baseline codec sizes disagree")
                        proposals.append(
                            dict(
                                kind=kind,
                                codec=codec,
                                serialized_packet_bytes=payload
                                + (0 if legacy else plan["predictor_header_bytes"]),
                                tensor_bitwise_equal=True,
                                cpu_roundtrip_seconds=time.monotonic() - began,
                            )
                        )
                        del packet, decoded, reconstructed
                    del prediction
                chosen = min(
                    proposals,
                    key=lambda p: (
                        p["serialized_packet_bytes"],
                        p["kind"] != "none" or p["codec"] != "bf16_exponents",
                        p["kind"],
                        p["codec"],
                    ),
                )
                row = dict(
                    method=method,
                    layer=body["layer"],
                    body_index=body["body_index"],
                    role=body["role"],
                    original_payload_bytes=body["original_GPU_payload_bytes"],
                    chosen=chosen,
                    candidates=proposals,
                )
                write_once(out / f"{method}-tensor-{j}.json", row)
                rows.append(row)
                del original
            new_payload = sum(r["chosen"]["serialized_packet_bytes"] for r in rows)
            prospective = artifact["non_payload_bytes"] + new_payload
            summary = dict(
                method=method,
                original_measured_GPU_initial_bytes=artifact[
                    "original_GPU_initial_component_bytes"
                ],
                measured_CPU_serialized_packet_bytes=new_payload,
                prospective_installed_bytes_NOT_GPU_verified=prospective,
                prospective_reduction=1
                - prospective / artifact["original_GPU_initial_component_bytes"],
                tensor_count=len(rows),
                predicted_tensor_count=sum(r["chosen"]["kind"] != "none" for r in rows),
                all_tensor_bits_preserved=True,
                not_an_inference_or_quality_result=True,
            )
            write_once(out / f"{method}-summary.json", summary)
            result["methods"].append(summary)
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def aggregate(root, expected):
    plan = base.common.inputs(root, expected)
    failures, rows, evidence = [], [], {}
    for job in plan["jobs"]:
        folder = root / f"codec_workers/{job['index']}"
        path = folder / "result.json"
        if not path.exists():
            failures.append(dict(index=job["index"], error="missing CPU worker"))
            continue
        result = base.common.read(path)
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("codec worker provenance changed")
        evidence[str(path)] = sha(path)
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("CPU codec evidence changed")
        if result["status"] != "completed" or result["failures"]:
            failures.append(dict(index=job["index"], errors=result["failures"]))
        elif {r["method"] for r in result["methods"]} != set(paired.METHODS) or len(
            result["methods"]
        ) != 6:
            failures.append(dict(index=job["index"], error="incomplete or duplicate method rows"))
        rows.extend(
            dict(task=job["task"], sample_id=job["sample_id"], **r) for r in result["methods"]
        )
    eligible = (
        not failures
        and len(rows) == 12
        and all(
            r["prospective_reduction"] >= plan["minimum_evicast_prospective_reduction"]
            for r in rows
            if r["method"] == "mosaickv_exact"
        )
    )
    out = root / "codec_analysis"
    out.mkdir()
    write_once(
        out / "GATE.json",
        dict(
            status="complete_CPU_feasibility" if not failures else "failed",
            failures=failures,
            rows=rows,
            eligible_for_separately_versioned_GPU_decoder=eligible,
            automatic_GPU_submission=False,
            quality_improvement_established=False,
            no_installed_GPU_storage_or_latency_claim=True,
            fresh_campaign_authorized=False,
        ),
    )
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            evidence=evidence,
            outputs={"GATE.json": sha(out / "GATE.json")},
        ),
    )


def main():
    paired.install()
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_predictive_feasibility.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, capture, advance
    if len(sys.argv) > 1 and sys.argv[1] in (
        "codec_worker",
        "codec_aggregate",
        "prepare_cpu_retry",
        "submit_cpu_retry",
    ):
        parser = argparse.ArgumentParser()
        parser.add_argument("mode")
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--plan-sha")
        parser.add_argument("--index", type=int)
        args = parser.parse_args()
        if args.mode == "prepare_cpu_retry":
            print(prepare_cpu_retry(args.root))
        elif args.mode == "submit_cpu_retry":
            plan = base.common.inputs(args.root, args.plan_sha)
            if not plan.get("CPU_only_retry"):
                raise ValueError("CPU-only retry scope required")
            launch_cpu(args.root, args.plan_sha)
        elif args.mode == "codec_worker":
            codec_worker(args.root, args.plan_sha, args.index)
        else:
            aggregate(args.root, args.plan_sha)
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "submit" and "--root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--root") + 1])
            if base.common.read(root / "plan.json").get("CPU_only_retry"):
                raise ValueError("CPU-only retry forbids GPU submission")
        base.main()


if __name__ == "__main__":
    main()
