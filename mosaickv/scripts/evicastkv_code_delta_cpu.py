#!/usr/bin/env python3
"""CPU-only serialized-byte feasibility on existing consumed source captures."""

import argparse
import hashlib
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/protected-source-cpu-20261101-v2")


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable CPU-only code-delta study root required")
    original = previous.read(PARENT / "plan.json")
    bound = dict(original["bound_files"])
    for p in (PARENT / "plan.json", SOURCE / "mosaickv/docs/lossless_code_delta_cpu_20261101.md"):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            original,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="CPU_only_consumed_unprotected_source_code_delta_feasibility",
            code_block_bytes=128,
            minimum_net_code_fraction_saved=0.10,
            unchanged_quantization_bits=3,
            include_largest_decoder_code_scratch=True,
            actual_frozen_GPU_packets_not_reconstructed=True,
            no_model_or_GPU_execution_authorized=True,
            independent_quality_N=0,
            no_fresh_or_large_campaign_authorized=True,
            benchmark_answers_or_references_accessed=False,
        ),
    )
    print(sha(root / "plan.json"))


def study(root, expected):
    import numpy as np
    import torch

    from mosaickv.evaluation.lossless_code_delta import decode, encode
    from mosaickv.evaluation.precision_tier import pack

    plan = previous.common.inputs(root, expected)
    proof = previous.read(root / "cpu.json")
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("complete clean same-source CPU suite required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    records, cases, failures = [], [], []
    for job in plan["jobs"]:
        raw_input = torch.load(job["capture"], map_location="cpu", weights_only=True, mmap=True)
        original, encoded, previous_codes = [], [], None
        output = root / f"case-{job['index']}"
        output.mkdir(exist_ok=False)
        manifest = dict(
            run_id=str(output),
            started_at_utc=datetime.now(UTC).isoformat(),
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            config_sha256=hashlib.sha256((expected + str(job["index"])).encode()).hexdigest(),
            environment=environment(),
            gpu=dict(type="not_used", count=0, driver="not_used"),
            model=dict(id=plan["model_id"], revision=plan["model_revision"], precision="bf16"),
            dataset=dict(
                id="consumed_normalized_source_capture",
                revision=sha(Path(job["capture"])),
                split="two_consumed_inputs",
            ),
            capture_producer_manifest=dict(
                path=job["producer_manifest"], sha256=sha(Path(job["producer_manifest"]))
            ),
            backend="CPU_serialization_only",
            attention_implementation="not_used",
            measurement_type="diagnostic_measured",
            result_category="unified_comparison",
            result_origin="locally_measured",
            seed=20261101,
            baseline_commit="not_used",
            evaluator_commit=plan["source_sha"],
            actual_GPU_cache_or_model_quality_claim=False,
        )
        write_once(output / "manifest.json", manifest)
        try:
            for layer, x in sorted(raw_input["normalized"].items()):
                if (
                    x.dtype != torch.bfloat16
                    or x.ndim != 3
                    or x.shape[0] != 1
                    or x.shape[-1] != 4096
                ):
                    raise ValueError("original captured full-MHA BF16 source required")
                body = pack(x[:, None], bits=3, axis=2)
                current = body.codes.numpy()
                started = time.monotonic()
                packet = encode(previous_codes, current)
                recovered = decode(previous_codes, packet)
                if not np.array_equal(recovered, current):
                    raise RuntimeError("lossless code delta changed an original byte")
                digest = hashlib.sha256(
                    packet.offsets.tobytes() + packet.payload.tobytes()
                ).hexdigest()
                row = dict(
                    case=job["index"],
                    layer=layer,
                    status="completed",
                    original_code_bytes=current.nbytes,
                    serialized_bytes=packet.serialized_bytes,
                    offsets_bytes=packet.offsets.nbytes,
                    raw_fallback=packet.raw,
                    payload_sha256=digest,
                    original_codes_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
                    roundtrip_exact=True,
                    CPU_encode_decode_seconds=time.monotonic() - started,
                    manifest_path=str(output / "manifest.json"),
                    measurement_type="diagnostic_measured",
                )
                records.append(row)
                original.append(current.copy())
                encoded.append(packet)
                previous_codes = current
            previous_codes = None
            for packet, expected_codes in zip(encoded, original, strict=True):
                previous_codes = decode(previous_codes, packet)
                if not np.array_equal(previous_codes, expected_codes):
                    raise RuntimeError("complete source-code chain replay differs")
            old_bytes = sum(a.nbytes for a in original)
            new_bytes = sum(p.serialized_bytes for p in encoded)
            scratch = max(a.nbytes for a in original) if any(not p.raw for p in encoded) else 0
            net = (old_bytes - new_bytes - scratch) / old_bytes
            cases.append(
                dict(
                    case=job["index"],
                    layers=len(encoded),
                    original_code_bytes=old_bytes,
                    serialized_code_bytes=new_bytes,
                    planned_decoder_scratch_bytes=scratch,
                    net_code_fraction_saved_including_planned_scratch=net,
                    minimum_feasibility_gate=bool(net >= plan["minimum_net_code_fraction_saved"]),
                    actual_GPU_HBM_measurement=False,
                    independent_quality_N=0,
                )
            )
        except Exception as exc:
            failures.append(dict(case=job["index"], error=f"{type(exc).__name__}: {exc}"))
        write_once(output / "records.json", [r for r in records if r["case"] == job["index"]])
        del original, encoded, raw_input, previous_codes
    complete = not failures and len(cases) == 2 and all(c["layers"] == 32 for c in cases)
    if not all(
        math.isfinite(c["net_code_fraction_saved_including_planned_scratch"]) for c in cases
    ):
        raise ValueError("nonfinite serialized-byte measurement")
    write_once(
        root / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            complete=complete,
            failures=failures,
            cases=cases,
            GPU_implementation_feasibility_gate=complete
            and all(c["minimum_feasibility_gate"] for c in cases),
            GPU_model_validation_authorized=False,
            actual_GPU_storage_reduction_established=False,
            independent_quality_N=0,
            no_new_inference_executed=True,
            baseline_equivalent_codec_study_still_required=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "study"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    else:
        study(args.root, args.plan_sha)
