#!/usr/bin/env python3
"""CPU-only, consumed-input feasibility of a fixed two-step affine refit."""

import argparse
import time
from pathlib import Path

import evicastkv_request_validation as previous
from evicastkv_campaign import environment
from evicastkv_projection_source_cpu import metadata_features

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
CAPTURES = Path("/scratch/djy8hg/runs/evicastkv/full-layer-source-consumed-20261101-v1")
PPL = Path("/scratch/djy8hg/runs/evicastkv/source-native-metrics-consumed-20261101-v1")
CHECKPOINT = Path(
    "/scratch/djy8hg/cache/mosaickv/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
)


def prepare(root):
    if root.exists():
        raise ValueError("new immutable affine-feasibility root required")
    negative = previous.read(PPL / "analysis/decision.json")
    if (
        not negative["complete"]
        or negative["failures"]
        or len(negative["teacher_metric_summary"]) != 2
    ):
        raise ValueError("complete PPL/ROUGE trade-off result required")
    bound = {str(PPL / "analysis/decision.json"): sha(PPL / "analysis/decision.json")}
    plan = previous.read(CAPTURES / "plan.json")
    bound[str(CAPTURES / "plan.json")] = sha(CAPTURES / "plan.json")
    expected_ids = [s["provenance"]["sample_id"] for s in negative["teacher_metric_summary"]]
    jobs = []
    for i in range(2):
        folder = CAPTURES / f"workers/{i}"
        record = previous.read(folder / "result.json")
        if record["status"] != "completed" or record["failures"]:
            raise ValueError("complete frozen captures required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in record["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("source capture changed")
            bound[str(folder / name)] = digest
        path = Path(plan["jobs"][i]["inputs"][0])
        features = metadata_features(previous.read(path))
        if features["sample_id"] != expected_ids[i]:
            raise ValueError("PPL trade-off and source capture identities differ")
        bound[str(path)] = sha(path)
        for media in features["media"]:
            if sha(Path(media["path"])) != media["sha256"]:
                raise ValueError("consumed media changed")
            bound[media["path"]] = media["sha256"]
        jobs.append(
            dict(index=i, capture=str(folder / "projection-inputs.pt"), input_metadata=str(path))
        )
    for path in (
        CHECKPOINT / "model.safetensors.index.json",
        SOURCE / "mosaickv/docs/affine_source_feasibility_20261101.md",
    ):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            jobs=jobs,
            checkpoint=str(CHECKPOINT),
            model_id="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            steps=2,
            source_bits=3,
            axis=2,
            minimum_attention_output_MSE_reduction=0.05,
            p95_must_not_increase=True,
            same_bytes_and_exact_rows=True,
            CPU_only=True,
            independent_quality_N=0,
            model_weight_cast="BF16_then_FP32_CPU_numerical_oracle",
            fit_uses_only_prompt_source_tensors=True,
            common_quantizer_not_new_selector=True,
            GPU_authorized=False,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )
    print(sha(root / "plan.json"))


def run(root, expected):
    import torch
    from safetensors import safe_open

    from mosaickv.evaluation.affine_refit import refit
    from mosaickv.evaluation.conditional_precision import pack_unprotected
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.precision_tier import unpack_cpu
    from mosaickv.evaluation.projection_source_feasibility import (
        aggregate,
        attention_output,
        errors,
        project,
    )
    from mosaickv.evaluation.protected_projection_source import (
        metadata_adapter,
        original_text_positions,
        protected_positions,
    )

    plan = previous.common.inputs(root, expected)
    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise ValueError("CPU-only numerical scope")
    start = time.monotonic()
    result = dict(
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        cases=[],
        failures=[],
        environment=environment(),
    )
    try:
        checkpoint = Path(plan["checkpoint"])
        metadata = metadata_adapter(checkpoint)
        weights = previous.read(checkpoint / "model.safetensors.index.json")["weight_map"]
        with torch.inference_mode():
            for job in plan["jobs"]:
                capture = torch.load(job["capture"], map_location="cpu", weights_only=True)
                phases = capture["phases"]
                n = phases[0].shape[1]
                features = metadata_features(previous.read(Path(job["input_metadata"])))
                text, token_hash = original_text_positions(
                    metadata, previous.load_messages(features), expected_length=n
                )
                controls, candidates, groups = [], [], []
                for layer, x in sorted(capture["normalized"].items()):
                    matrices = {}
                    for kind in ("q", "k", "v"):
                        name = f"language_model.model.layers.{layer}.self_attn.{kind}_proj.weight"
                        with safe_open(
                            checkpoint / weights[name], framework="pt", device="cpu"
                        ) as handle:
                            matrices[kind] = handle.get_tensor(name).bfloat16().float()
                    q = project(x, matrices["q"], heads=32, phases=phases)[..., -16:, :]
                    k = project(x, matrices["k"], heads=32, phases=phases)
                    v = project(x, matrices["v"], heads=32)
                    pos = torch.arange(n).repeat(32, 1)
                    queries = torch.arange(n - 16, n)
                    reference = attention_output(q, k, v, pos, queries)
                    exact = protected_positions(
                        capture["source_memberships"][layer], text, length=n
                    )
                    global_exact = exact.all(0)[None]
                    old = pack_unprotected(x[:, None], global_exact, bits=3, axis=2)
                    new, proof = refit(old, x[:, None], exact_rows=global_exact)
                    if backing_bytes(old.tensors()) != backing_bytes(new.tensors()):
                        raise RuntimeError("physical source bytes changed")
                    for packet, records in ((old, controls), (new, candidates)):
                        decoded = unpack_cpu(packet).squeeze(1)
                        ak = project(decoded, matrices["k"], heads=32, phases=phases)
                        av = project(decoded, matrices["v"], heads=32)
                        ak[0][exact] = k[0][exact]
                        av[0][exact] = v[0][exact]
                        records.append(errors(reference, attention_output(q, ak, av, pos, queries)))
                    groups.append(
                        dict(layer=layer, source_bytes=backing_bytes(old.tensors()), fit=proof)
                    )
                result["cases"].append(
                    dict(
                        index=job["index"],
                        task=features["task"],
                        sample_id=features["sample_id"],
                        original=aggregate(controls),
                        refitted=aggregate(candidates),
                        layers=groups,
                        same_source_bytes=True,
                        exact_rows_identical=True,
                        processor_token_hash=token_hash,
                        no_benchmark_label_access=True,
                    )
                )
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    result["wall_seconds"] = time.monotonic() - start
    write_once(root / "result.json", result)
    if result["failures"]:
        raise SystemExit(1)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    try:
        result = previous.read(root / "result.json")
    except OSError as exc:
        result = dict(failures=[str(exc)], cases=[])
    complete = len(result["cases"]) == 2 and not result["failures"]
    gates = []
    for case in result["cases"]:
        old, new = case["original"], case["refitted"]
        gates.append(
            dict(
                index=case["index"],
                attention_MSE_reduction=1 - new["relative_mse"] / old["relative_mse"],
                p95_not_worse=new["p95_per_query_head_relative_mse"]
                <= old["p95_per_query_head_relative_mse"],
                numerical_gate=new["relative_mse"] <= 0.95 * old["relative_mse"]
                and new["p95_per_query_head_relative_mse"]
                <= old["p95_per_query_head_relative_mse"],
            )
        )
    write_once(
        root / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            complete=complete,
            failures=result["failures"],
            gates=gates,
            numerical_feasibility_gate=complete and all(g["numerical_gate"] for g in gates),
            result_sha256=sha(root / "result.json") if (root / "result.json").exists() else None,
            no_model_quality_claim=True,
            GPU_authorized=False,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "run", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    else:
        globals()[args.mode](args.root, args.plan_sha)
