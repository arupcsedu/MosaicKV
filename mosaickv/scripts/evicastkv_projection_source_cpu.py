#!/usr/bin/env python3
"""CPU-only, label-free numerical feasibility on two consumed source captures."""

import argparse
import json
import time
from pathlib import Path

import evicastkv_request_validation as previous
from evicastkv_campaign import environment
from evicastkv_logit_composition_audit import status_header

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/full-layer-source-consumed-20261101-v1")
MODEL = "llava-hf/llava-1.5-7b-hf"
REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
ARMS = ((3, 2), (3, 3), (4, 2), (4, 3))


def prepare(root):
    from huggingface_hub import snapshot_download

    if root.exists():
        raise ValueError("new write-once output root required")
    decision = PARENT / "analysis/decision.json"
    header = status_header(decision)
    if (
        not header["complete"]
        or header["failures"]
        or header["model_level_restoration_opportunity"]
    ):
        raise ValueError("preserve the complete negative local-restoration result")
    bindings = {str(decision): sha(decision)}
    jobs = []
    for i in range(2):
        folder = PARENT / f"workers/{i}"
        result = json.loads((folder / "result.json").read_text())
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("complete source-capture producer required")
        bindings[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("source-capture artifact changed")
            bindings[str(folder / name)] = digest
        jobs.append(
            dict(
                index=i,
                capture=str(folder / "projection-inputs.pt"),
                producer_manifest=str(folder / "manifest.json"),
            )
        )
    checkpoint = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    index = checkpoint / "model.safetensors.index.json"
    bindings[str(index)] = sha(index)
    protocol = SOURCE / "mosaickv/docs/projection_source_cpu_feasibility_20261101.md"
    bindings[str(protocol)] = sha(protocol)
    write_once(
        root / "plan.json",
        dict(
            source_sha=clean_source(SOURCE),
            bound_files=bindings,
            jobs=jobs,
            checkpoint=str(checkpoint),
            model_id=MODEL,
            model_revision=REVISION,
            arms=ARMS,
            prompt_query_window=16,
            CPU_only=True,
            measurement_type="derived_numerical_feasibility_not_task_quality",
            benchmark_annotations_answers_and_scores_not_loaded=True,
            no_gpu_or_cohort_execution_authorized=True,
            independent_quality_N=0,
        ),
    )
    print(sha(root / "plan.json"))


def run(root, expected):
    import torch
    from safetensors import safe_open

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.precision_tier import pack, unpack_cpu
    from mosaickv.evaluation.projection_source_feasibility import (
        aggregate,
        attention_output,
        errors,
        project,
    )

    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    if sha(plan_path) != expected or clean_source(SOURCE) != plan["source_sha"]:
        raise ValueError("clean immutable CPU execution plan required")
    for path, digest in plan["bound_files"].items():
        if sha(Path(path)) != digest:
            raise ValueError("bound input changed")
    proof = json.loads((root / "cpu.json").read_text())
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["test_count"] < 3008
        or proof["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("complete clean CPU suite required before real source-data analysis")
    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    torch.use_deterministic_algorithms(True)
    checkpoint = Path(plan["checkpoint"])
    weights = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    started = time.monotonic()
    result = dict(source_sha=plan["source_sha"], plan_sha256=expected, failures=[], cases=[])
    try:
        with torch.inference_mode():
            for job in plan["jobs"]:
                capture = torch.load(job["capture"], map_location="cpu", weights_only=True)
                phases = capture["phases"]
                base_records, candidates, payloads = (
                    [],
                    {arm: [] for arm in ARMS},
                    {arm: 0 for arm in ARMS},
                )
                for layer, x in sorted(capture["normalized"].items()):
                    n, width = x.shape[1:]
                    if width != 4096 or x.dtype != torch.bfloat16:
                        raise ValueError("full normalized LLaVA source required")
                    matrices = {}
                    for kind in ("q", "k", "v"):
                        name = f"language_model.model.layers.{layer}.self_attn.{kind}_proj.weight"
                        with safe_open(
                            checkpoint / weights[name], framework="pt", device="cpu"
                        ) as handle:
                            matrices[kind] = handle.get_tensor(name).float()
                    query = project(x, matrices["q"], heads=32, phases=phases)[..., -16:, :]
                    key = project(x, matrices["k"], heads=32, phases=phases)
                    value = project(x, matrices["v"], heads=32)
                    qp = torch.arange(n - 16, n)
                    positions = torch.arange(n).repeat(32, 1)
                    reference = attention_output(query, key, value, positions, qp)
                    ids = torch.tensor(capture["source_memberships"][layer], dtype=torch.long)
                    gather = ids[None, :, :, None].expand(1, 32, ids.shape[1], 128)
                    pruning = attention_output(
                        query, key.gather(2, gather), value.gather(2, gather), ids, qp
                    )
                    base_records.append(errors(reference, pruning))
                    for arm in ARMS:
                        bits, axis = arm
                        packet = pack(x[:, None], bits=bits, axis=axis)
                        payloads[arm] += backing_bytes(packet.tensors())
                        reconstructed = unpack_cpu(packet).squeeze(1)
                        k = project(reconstructed, matrices["k"], heads=32, phases=phases)
                        v = project(reconstructed, matrices["v"], heads=32)
                        out = attention_output(query, k, v, positions, qp)
                        candidates[arm].append(errors(reference, out))
                    del matrices
                baseline = aggregate(base_records)
                rows = []
                for arm in ARMS:
                    stats = aggregate(candidates[arm])
                    rows.append(
                        dict(
                            bits=arm[0],
                            axis=arm[1],
                            statistics=stats,
                            source_payload_lower_bound_bytes=payloads[arm],
                            original_own_cap_bytes=capture["original_cap_bytes"],
                            feasibility_pass=stats["relative_mse"] <= 0.8 * baseline["relative_mse"]
                            and stats["p95_per_query_head_relative_mse"]
                            <= baseline["p95_per_query_head_relative_mse"]
                            and payloads[arm] <= capture["original_cap_bytes"],
                        )
                    )
                result["cases"].append(
                    dict(index=job["index"], pruning_only_original_BF16_control=baseline, arms=rows)
                )
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    complete = len(result["cases"]) == 2 and not result["failures"]
    passing = [
        arm
        for arm in ARMS
        if complete
        and all(
            next(r for r in c["arms"] if (r["bits"], r["axis"]) == arm)["feasibility_pass"]
            for c in result["cases"]
        )
    ]
    result.update(
        complete=complete,
        passing_both_inputs=passing,
        runtime_seconds=time.monotonic() - started,
        source_sha=plan["source_sha"],
        model_id=MODEL,
        model_revision=REVISION,
        environment=environment(),
        GPU_count=0,
        backend="CPU_FP32_projection_BF16_rounding_oracle_not_native_GPU",
        seed=20261101,
        reference_artifact_hashes=plan["bound_files"],
        independent_quality_N=0,
        benchmark_annotations_answers_and_scores_not_loaded=True,
        native_GPU_parity_not_tested=True,
        real_cache_construction_not_implemented=True,
        payload_excludes_exact_overlays_maps_phases_history_leases_and_construction=True,
        superiority_established=False,
        GPU_execution_authorized=False,
        no_fresh_or_large_campaign_authorized=True,
    )
    write_once(root / "decision.json", result)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    else:
        run(args.root, args.plan_sha)
