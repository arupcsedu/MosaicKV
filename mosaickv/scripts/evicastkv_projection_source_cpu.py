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
PROTECTED_ARMS = ARMS


def prepare(root, *, protected=False):
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
    parent_plan_path = PARENT / "plan.json"
    parent_plan = json.loads(parent_plan_path.read_text())
    if protected:
        negative = Path(
            "/scratch/djy8hg/runs/evicastkv/projection-source-cpu-20261101-v1/decision.json"
        )
        status = json.loads(negative.read_text())
        if not status["complete"] or status["failures"] or status["passing_both_inputs"]:
            raise ValueError("preserve completed negative uniform-source study")
        bindings[str(negative)] = sha(negative)
        bindings[str(parent_plan_path)] = sha(parent_plan_path)
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
        if protected:
            input_path = Path(parent_plan["jobs"][i]["inputs"][0])
            features = json.loads(input_path.read_text())
            bindings[str(input_path)] = sha(input_path)
            for medium in features["media"]:
                media_path = Path(medium["path"])
                if sha(media_path) != medium["sha256"]:
                    raise ValueError("consumed media changed")
                bindings[str(media_path)] = medium["sha256"]
            jobs[-1]["input_metadata"] = str(input_path)
    checkpoint = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    index = checkpoint / "model.safetensors.index.json"
    bindings[str(index)] = sha(index)
    protocol = (
        SOURCE
        / "mosaickv/docs"
        / (
            "protected_projection_source_cpu_20261101.md"
            if protected
            else "projection_source_cpu_feasibility_20261101.md"
        )
    )
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
            arms=PROTECTED_ARMS if protected else ARMS,
            protected_source=protected,
            prompt_query_window=16,
            CPU_only=True,
            measurement_type="derived_numerical_feasibility_not_task_quality",
            benchmark_annotations_answers_and_scores_not_loaded=not protected,
            annotation_fields_never_consulted_or_used=True,
            no_gpu_or_cohort_execution_authorized=True,
            independent_quality_N=0,
        ),
    )
    print(sha(root / "plan.json"))


def run(root, expected):
    import torch
    from safetensors import safe_open

    from mosaickv.evaluation.lossless_bf16 import pack_cpu
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
    protected = plan.get("protected_source", False)
    arms = PROTECTED_ARMS if protected else ARMS
    if tuple(map(tuple, plan["arms"])) != arms:
        raise ValueError("unregistered source feasibility arms")
    weights = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    started = time.monotonic()
    result = dict(source_sha=plan["source_sha"], plan_sha256=expected, failures=[], cases=[])
    try:
        with torch.inference_mode():
            if protected:
                from mosaickv.evaluation.protected_projection_source import (
                    metadata_adapter,
                    original_text_positions,
                    protected_positions,
                )

                metadata = metadata_adapter(checkpoint)
            for job in plan["jobs"]:
                capture = torch.load(job["capture"], map_location="cpu", weights_only=True)
                phases = capture["phases"]
                base_records, candidates, payloads = (
                    [],
                    {arm: [] for arm in arms},
                    {arm: 0 for arm in arms},
                )
                extra_bytes, raw_exact_upper_bound = 0, 0
                layer_extra_bytes = []
                layer_payloads = {arm: [] for arm in arms}
                text, token_hash = (), None
                if protected:
                    bound_input = json.loads(Path(job["input_metadata"]).read_text())
                    # Metadata constructor receives NO dataset annotation fields.
                    features = {k: bound_input[k] for k in ("prompt", "media")}
                    text, token_hash = original_text_positions(
                        metadata,
                        previous.load_messages(features),
                        expected_length=phases[0].shape[1],
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
                    mask = None
                    layer_extra = 0
                    if protected:
                        mask = protected_positions(
                            capture["source_memberships"][layer], text, length=n
                        )
                        flat_ids = mask.flatten().nonzero().flatten().to(torch.int32)
                        exact = tuple(
                            pack_cpu(
                                t.reshape(1, 1, 32 * n, 128)
                                .index_select(2, flat_ids.long())
                                .contiguous()
                            )
                            for t in (key, value)
                        )
                        layer_extra = backing_bytes(
                            (flat_ids, *(t for p in exact for t in p.buffers))
                        )
                        extra_bytes += layer_extra
                        raw_exact_upper_bound += flat_ids.numel() * (128 * 2 * 2 + 4)
                    layer_extra_bytes.append(layer_extra)
                    for arm in arms:
                        bits, axis = arm
                        packet = pack(x[:, None], bits=bits, axis=axis)
                        size = backing_bytes(packet.tensors())
                        payloads[arm] += size
                        layer_payloads[arm].append(size)
                        reconstructed = unpack_cpu(packet).squeeze(1)
                        k = project(reconstructed, matrices["k"], heads=32, phases=phases)
                        v = project(reconstructed, matrices["v"], heads=32)
                        if protected:
                            k[0][mask] = key[0][mask]
                            v[0][mask] = value[0][mask]
                        out = attention_output(query, k, v, positions, qp)
                        candidates[arm].append(errors(reference, out))
                    del matrices
                baseline = aggregate(base_records)
                if protected:
                    n = phases[0].shape[1]
                    extra_bytes += backing_bytes(
                        (
                            *phases,
                            torch.arange(n, dtype=torch.int32),
                            torch.arange(n, dtype=torch.int16),
                        )
                    )
                rows = []
                for arm in arms:
                    stats = aggregate(candidates[arm])
                    rows.append(
                        dict(
                            bits=arm[0],
                            axis=arm[1],
                            statistics=stats,
                            source_payload_lower_bound_bytes=payloads[arm],
                            CPU_prefix_bytes_including_protection_maps_and_phase=payloads[arm]
                            + extra_bytes,
                            CPU_exact_rows_raw_upper_bound_bytes=raw_exact_upper_bound,
                            layer_numerical_statistics=[aggregate([r]) for r in candidates[arm]],
                            layer_source_payload_bytes=layer_payloads[arm],
                            layer_exact_component_bytes=layer_extra_bytes,
                            shared_phase_identity_bytes=extra_bytes - sum(layer_extra_bytes),
                            original_own_cap_bytes=capture["original_cap_bytes"],
                            feasibility_pass=stats["relative_mse"] <= 0.8 * baseline["relative_mse"]
                            and stats["p95_per_query_head_relative_mse"]
                            <= baseline["p95_per_query_head_relative_mse"]
                            and payloads[arm] + extra_bytes <= capture["original_cap_bytes"],
                        )
                    )
                result["cases"].append(
                    dict(
                        index=job["index"],
                        pruning_only_original_BF16_control=baseline,
                        arms=rows,
                        authoritative_text_position_count=len(text),
                        processor_token_hash=token_hash,
                        exact_protection_matches_existing_head_local_policy=protected,
                    )
                )
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    complete = len(result["cases"]) == 2 and not result["failures"]
    passing = [
        arm
        for arm in arms
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
        benchmark_annotations_answers_and_scores_not_loaded=not protected,
        annotation_fields_never_consulted_or_used=True,
        native_GPU_parity_not_tested=True,
        real_cache_construction_not_implemented=True,
        prefix_excludes_exact_overlays_maps_and_phases=not protected,
        prefix_excludes_history_leases_and_construction=True,
        superiority_established=False,
        GPU_execution_authorized=False,
        no_fresh_or_large_campaign_authorized=True,
    )
    write_once(root / "decision.json", result)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "prepare_protected", "cpu", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "prepare_protected":
        prepare(args.root, protected=True)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    else:
        run(args.root, args.plan_sha)
