#!/usr/bin/env python3
"""Bounded CUDA feasibility diagnostic; synthetic results, not benchmark quality."""

import argparse
import os
import platform
import random
import subprocess
from functools import partial
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]


def worker(root: Path) -> None:
    import numpy as np
    import torch

    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_bf16_kernel import encode
    from mosaickv.evaluation.packed_attention import attend, packet

    source_sha = clean_source(SOURCE)
    root.mkdir(parents=True, exist_ok=False)
    seed = 20261030
    config = dict(
        seed=seed,
        lengths=[1, 7, 8, 9, 31, 128, 300, 604, 2048],
        heads=32,
        dimension=128,
        warmups=5,
        repetitions=20,
        calls_per_trial=10,
        numerical_max_attention_output_abs=0.015625,
        relative_latency_gate=1.05,
        larger_cohort_authorized=False,
    )
    write_once(root / "config.json", config)
    import importlib.metadata

    environment = {
        name: importlib.metadata.version(name)
        if name in {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
        else "not_installed"
        for name in ["torch", "transformers", "triton", "vllm", "sglang"]
    }
    write_once(
        root / "manifest.json",
        dict(
            source_git_sha=source_sha,
            source_dirty=False,
            config_sha256=sha(root / "config.json"),
            python=platform.python_version(),
            packages=environment,
            seed=seed,
            model_id="not_used",
            model_revision="not_used",
            dataset="synthetic_tensor_fixture",
            dataset_revision="generator_seed_20261030_v1",
            measurement_type="synthetic_cuda_diagnostic",
            backend="experimental_hf_eager_compatible_arithmetic",
            attention="packed_integer_decode",
            cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            gpu_count=1,
            gpu_and_driver=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            no_large_model_loaded=True,
        ),
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    order = random.Random(seed)
    rows, trials = [], []
    try:
        for length in config["lengths"]:
            shape = (1, 32, length, 128)
            raw = [
                torch.randn(shape, generator=generator, device="cuda").bfloat16() for _ in range(2)
            ]
            query = torch.randn((1, 32, 1, 128), generator=generator, device="cuda").bfloat16()
            history = tuple(
                torch.randn((1, 32, 16, 128), generator=generator, device="cuda").bfloat16()
                for _ in range(2)
            )
            encoded = tuple(packet(encode(t)) for t in raw)
            unpacked = tuple(packet(PackedBF16(shape, True, (t,))) for t in raw)
            used = 5
            k, v = (torch.cat((t, h[..., :used, :]), -2) for t, h in zip(raw, history, strict=True))

            def eager(query=query, k=k, v=v):
                p = torch.softmax(
                    query @ k.transpose(-2, -1) * 128**-0.5, -1, dtype=torch.float32
                ).bfloat16()
                return (p @ v).transpose(1, 2).contiguous(), p

            functions = {
                "eager_materialized": eager,
                "packed": partial(attend, query, *encoded, history, used=used, scaling=128**-0.5),
                "raw_kernel": partial(
                    attend, query, *unpacked, history, used=used, scaling=128**-0.5
                ),
            }
            reference, actual, raw_output = (
                functions[n]() for n in ["eager_materialized", "packed", "raw_kernel"]
            )
            exact = all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(actual, raw_output, strict=True)
            )
            error = float((actual[0].float() - reference[0].float()).abs().max())
            row = dict(
                length=length,
                raw_encoded_kernel_bitwise=exact,
                eager_max_abs_error=error,
                raw_prompt_bytes=sum(t.untyped_storage().nbytes() for t in raw),
                packed_prompt_bytes=sum(t.storage_bytes for t in encoded),
                metadata_counted=True,
                full_KV_lease_bytes=0,
                no_projection=True,
            )
            rows.append(row)
            if not exact or error > config["numerical_max_attention_output_abs"]:
                raise RuntimeError(f"attention numerical control failed: {row}")
            for trial in range(-5, 20):
                names = list(functions)
                order.shuffle(names)
                for offset, name in enumerate(names):
                    torch.cuda.synchronize()
                    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    begin.record()
                    for _ in range(10):
                        functions[name]()
                    end.record()
                    torch.cuda.synchronize()
                    trials.append(
                        dict(
                            length=length,
                            trial=trial,
                            warmup=trial < 0,
                            order=offset,
                            method=name,
                            milliseconds=begin.elapsed_time(end) / 10,
                        )
                    )
            measurements = {}
            for name in functions:
                values = [
                    r["milliseconds"]
                    for r in trials
                    if r["length"] == length and r["method"] == name and not r["warmup"]
                ]
                measurements[name] = dict(
                    p50=float(np.median(values)), p95=float(np.percentile(values, 95))
                )
            row["timings_ms"] = measurements
            row["packed_over_eager_p50"] = (
                measurements["packed"]["p50"] / measurements["eager_materialized"]["p50"]
            )
            write_once(root / f"length-{length}.json", row)
        write_once(
            root / "GATE.json",
            dict(
                status="complete_synthetic_cuda_diagnostic",
                rows=rows,
                numerical_passed=True,
                latency_gate_passed=all(
                    r["packed_over_eager_p50"] <= 1.05 for r in rows if r["length"] >= 128
                ),
                benchmark_quality_established=False,
            ),
        )
    except Exception as exc:
        write_once(root / "error.json", dict(type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        write_once(root / "partial_rows.json", rows)
        write_once(root / "trials.json", trials)
        write_once(root / "hashes.json", {p.name: sha(p) for p in root.iterdir() if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    worker(parser.parse_args().root)
