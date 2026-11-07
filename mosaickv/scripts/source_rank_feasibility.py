#!/usr/bin/env python3
"""Two consumed source captures; CPU-only lower-bound and actual-format study."""

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/full-layer-source-consumed-20261101-v1")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(p):
    return json.loads(Path(p).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable CPU plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    for p, expected in plan["bound_files"].items():
        if sha(Path(p)) != expected:
            raise ValueError("consumed capture or metadata changed")
    return plan


def submit(root, worker, after=None):
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--job-name=mk-source-rank-cpu",
        "--cpus-per-task=" + ("12" if worker else "2"),
        "--mem=" + ("16G" if worker else "4G"),
        "--time=" + ("00:30:00" if worker else "00:10:00"),
    ]
    if worker:
        args += ["--array=0-1%2"]
    if after:
        args += ["--dependency=afterany:" + after]
    args += [
        f"--output={root}/logs/%A_%a.out",
        f"--error={root}/logs/%A_%a.err",
        str(SOURCE / "mosaickv/slurm/source_rank_feasibility.sbatch"),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        "worker" if worker else "analyze",
    ]
    return subprocess.check_output(args, text=True).strip()


def prepare(root, *, refined_source=False):
    from evicastkv_projection_source_cpu import metadata_features

    from mosaickv.adapters.huggingface.registry import _resolve_model_source

    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable CPU study root required")
    parent = read(PARENT / "plan.json")
    protocol = "refined_source_feasibility.md" if refined_source else "source_rank_feasibility.md"
    jobs, paths = [], [PARENT / "plan.json", SOURCE / "mosaickv/docs" / protocol]
    prior = Path("/scratch/djy8hg/runs/evicastkv/source-rank-feasibility-20261107-v1")
    if refined_source:
        decision = read(prior / "analysis/decision.json")
        if not decision["complete"] or decision["source_factor_feasible_on_both"]:
            raise ValueError("complete negative pure-rank screen required")
        paths += [prior / "analysis/decision.json"]
        paths += [prior / f"workers/{i}/result.json" for i in (0, 1)]
    for i in (0, 1):
        folder = PARENT / f"workers/{i}"
        status = read(folder / "result.json")
        if status["status"] != "completed" or status["failures"]:
            raise ValueError("complete consumed capture producer required")
        paths.append(folder / "result.json")
        for name in ("projection-inputs.pt", "manifest.json"):
            p = folder / name
            if sha(p) != status["outputs"][name]:
                raise ValueError("original capture checksum mismatch")
            paths.append(p)
        producer = read(folder / "manifest.json")
        if producer["model"] != dict(
            id="llava-hf/llava-1.5-7b-hf",
            precision="bf16",
            revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
        ):
            raise ValueError("source capture uses a different pinned checkpoint")
        metadata = Path(parent["jobs"][i]["inputs"][0])
        features = metadata_features(read(metadata))
        if producer["input_hashes"].get(str(metadata)) != sha(metadata):
            raise ValueError("capture's original prompt/media metadata differs")
        paths.append(metadata)
        for m in features["media"]:
            p = Path(m["path"])
            if sha(p) != m["sha256"]:
                raise ValueError("consumed media changed")
            paths.append(p)
        jobs.append(
            dict(
                capture=str(folder / "projection-inputs.pt"),
                manifest=str(folder / "manifest.json"),
                input_metadata=str(metadata),
            )
        )
    checkpoint = Path(
        _resolve_model_source(
            "llava-hf/llava-1.5-7b-hf",
            "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            local_files_only=True,
        )
    )
    paths += [
        checkpoint / n
        for n in (
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        )
        if (checkpoint / n).exists()
    ]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu.txt").open("x") as stream:
        done = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_source_rank_feasibility.py",
                *(
                    ["mosaickv/tests/unit/test_refined_source_feasibility.py"]
                    if refined_source
                    else []
                ),
            ],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if done.returncode:
        raise RuntimeError("CPU synthetic gates failed; no submission")
    write_once(root / "cpu.json", dict(passed=True, log_sha256=sha(root / "cpu.txt")))
    paths += [root / "cpu.json", root / "cpu.txt"]
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(p): sha(p) for p in paths},
            jobs=jobs,
            ranks=[96, 128],
            source_family="refined_source" if refined_source else "pure_rank",
            families=["lut2", "spatial4_binary_residual", "spatial8_binary_residual"]
            if refined_source
            else [],
            expected_cases_per_worker=96 if refined_source else 64,
            prior_negative_root=str(prior) if refined_source else None,
            layers=list(range(32)),
            checkpoint=str(checkpoint),
            minimum_net_cpu_payload_reduction=0.10,
            source_mse_slack=1e-12,
            numerical_rule="every_nonexact_image_row_source_mse_not_greater_than_existing_three_bit"
            if refined_source
            else "actual_factor_source_mse_not_greater_than_existing_three_bit",
            literal_refinement="minimal_largest_scalar_errors_or_full_row_charged"
            if refined_source
            else None,
            lloyd_iterations=4 if refined_source else None,
            source_choice_rule="globally_minimum_cpu_codec_payload_with_parent_fallback"
            if refined_source
            else "minimum_rank_raw_payload",
            model_weights_loaded=False,
            benchmark_labels_accessed=False,
            original_six_compaction_backend_implemented=False,
            gpu_followup_authorized=False,
            fresh_or_paper_campaign_authorized=False,
            measurement_type="cpu_source_numerical_feasibility_not_gpu_cache_result",
        ),
    )
    array = submit(root, True)
    analyzer = submit(root, False, after=array)
    receipt = dict(
        array=array,
        analyzer=analyzer,
        gpu_count=0,
        cpu_count_per_worker=12,
        worker_limit_minutes=30,
        afterany=True,
    )
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def worker(root, expected):
    import evicastkv_request_validation as previous
    import torch
    from evicastkv_campaign import environment
    from evicastkv_projection_source_cpu import metadata_features

    from mosaickv.evaluation.protected_projection_source import (
        metadata_adapter,
        original_text_positions,
        protected_positions,
    )
    from mosaickv.evaluation.source_rank_feasibility import (
        choose_codec_coupled,
        cpu_serialized_storage,
        screen,
    )

    plan = checked(root, expected)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261107)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    started, cases, errors = perf_counter(), [], []
    result = {}
    try:
        job = plan["jobs"][i]
        producer = read(job["manifest"])
        write_once(
            folder / "manifest.json",
            dict(
                git_sha=plan["source_sha"],
                config_sha=digest(plan),
                source_dirty=False,
                model=producer["model"],
                dataset=producer["dataset"],
                producer_manifest_sha256=sha(Path(job["manifest"])),
                measurement_type=plan["measurement_type"],
                cuda="not_used",
                driver="not_used",
                gpu_type="not_used",
                gpu_count=0,
                torch=torch.__version__,
                environment=environment(),
                producer_environment=producer["environment"],
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                backend="CPU_numeric_only",
                attention_implementation="not_used",
                seed=20261107,
                vllm="not_used",
                sglang="not_used",
                model_weights_loaded=False,
            ),
        )
        capture = torch.load(job["capture"], weights_only=True, map_location="cpu")
        if sorted(capture["normalized"]) != list(range(32)):
            raise ValueError("all original projection-source layers required")
        metadata = metadata_adapter(plan["checkpoint"])
        features = metadata_features(read(job["input_metadata"]))
        text, token_hash = original_text_positions(
            metadata,
            previous.load_messages(features),
            expected_length=capture["phases"][0].shape[1],
        )
        baseline, adaptive, chosen, alternatives = [], [], [], []
        for layer in plan["layers"]:
            x = capture["normalized"][layer]
            exact = protected_positions(
                capture["source_memberships"][layer], text, length=x.shape[1]
            ).all(0)
            refined_source = plan.get("source_family") == "refined_source"
            if refined_source:
                from mosaickv.evaluation.refined_source_feasibility import screen_refined

                config = read(Path(plan["checkpoint"]) / "config.json")["vision_config"]
                if (config["image_size"], config["patch_size"]) != (336, 14):
                    raise ValueError("audited pinned raster vision geometry required")
                part = screen_refined(x, exact, text)
                if [r["family"] for r in part["rows"]] != plan["families"]:
                    raise ValueError("registered source families changed")
            else:
                part = screen(x, exact, text, tuple(plan["ranks"]))
            for row in part["rows"]:
                cases.append(dict(row, layer=layer))
            baseline.append(part["baseline"])
            alternatives.append([("parent", part["baseline"]), *sorted(part["candidates"].items())])
            rank = (
                min(
                    part["candidates"],
                    key=lambda k: (
                        part["candidates"][k]["codes"].numel()
                        + part["candidates"][k]["other_bytes"],
                        k,
                    ),
                )
                if part["candidates"] and refined_source
                else min(part["candidates"])
                if part["candidates"]
                else None
            )
            adaptive.append(part["baseline"] if rank is None else part["candidates"][rank])
            chosen.append(rank)
            write_once(folder / f"layer-{layer:02}.json", dict(rows=part["rows"], chosen_rank=rank))
        raw_minimum = cpu_serialized_storage(adaptive)
        if plan.get("source_family") == "refined_source":
            adaptive, selected = choose_codec_coupled(alternatives)
            chosen = [None if name == "parent" else name for name in selected]
        old, new = cpu_serialized_storage(baseline), cpu_serialized_storage(adaptive)
        if plan.get("source_family") == "refined_source":
            prior = read(Path(plan["prior_negative_root"]) / f"workers/{i}/result.json")
            if old != prior["result"]["baseline"]:
                raise ValueError("unchanged baseline source/codec accounting differs")
        reduction = 1 - new["net_cpu_payload_bytes"] / old["net_cpu_payload_bytes"]
        result = dict(
            baseline=old,
            adaptive=new,
            independent_raw_choice=raw_minimum,
            net_cpu_payload_reduction=reduction,
            selected_ranks=chosen,
            numerical_eligible_layers=sum(x is not None for x in chosen),
            original_tokenization_sha256=token_hash,
            local_screen_passed=reduction >= plan["minimum_net_cpu_payload_reduction"]
            and any(x is not None for x in chosen),
        )
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed",
            errors=errors,
            cases=cases,
            result=result,
            wall_seconds=perf_counter() - started,
            downstream_quality_established=False,
            gpu_followup_authorized=False,
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    plan = checked(root, expected)
    records = []
    for i in (0, 1):
        p = root / f"workers/{i}/result.json"
        records.append(
            read(p) if p.exists() else dict(status="missing", errors=["missing worker"], cases=[])
        )
    count = plan.get("expected_cases_per_worker", 64)
    complete = all(r["status"] == "completed" and len(r["cases"]) == count for r in records)
    feasible = complete and all(r["result"]["local_screen_passed"] for r in records)
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(
        out / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            complete=complete,
            source_factor_feasible_on_both=feasible,
            records=records,
            actual_cases=sum(len(r["cases"]) for r in records),
            expected_cases=count * 2,
            source_family=plan.get("source_family", "pure_rank"),
            model_or_gpu_quality_established=False,
            gpu_followup_authorized=False,
            preserve_existing_runtime=True,
            larger_or_fresh_campaign_authorized=False,
        ),
    )
    lines = [
        "# CPU-only source-factor feasibility",
        "",
        "CPU source payloads and reconstruction error; not measured GPU KV, "
        "answer quality or latency.",
        "",
        "| Input | Eligible layers | Net CPU payload reduction | Necessary screen |",
        "| --- | ---: | ---: | --- |",
    ]
    for i, r in enumerate(records):
        v = r.get("result", {})
        lines.append(
            f"| {i} | {v.get('numerical_eligible_layers', 'not_completed')} | "
            f"{v.get('net_cpu_payload_reduction', 'not_completed')} | "
            f"{v.get('local_screen_passed', False)} |"
        )
    lines += [
        "",
        "No model weights changed or loaded. No source/reader runtime is activated.",
        "Stop GPU expansion after a negative screen; retain all layers and failures.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "analyze"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--plan-sha")
    p.add_argument("--refined-source", action="store_true", help="CPU-only source-codec study")
    a = p.parse_args()
    return (
        prepare(a.root, refined_source=a.refined_source)
        if a.action == "prepare"
        else worker(a.root, a.plan_sha)
        if a.action == "worker"
        else analyze(a.root, a.plan_sha)
    )


if __name__ == "__main__":
    raise SystemExit(main())
